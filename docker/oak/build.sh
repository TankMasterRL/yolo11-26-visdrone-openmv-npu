#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the Luxonis ModelConverter Docker images locally from source.
#
# This wraps the upstream luxonis/modelconverter Dockerfiles so the YOLO
# export pipeline never has to pull from Docker Hub. Both targets are
# built from scratch:
#
#     luxonis/modelconverter-rvc2:local   (OAK   / Myriad-X / .blob)
#     luxonis/modelconverter-rvc4:local   (OAK4  / Qualcomm  / .dlc)
#
# These tags match the new defaults in train_and_export.py — running
# ``uv run python train_and_export.py`` with no flag overrides will use
# whatever this script produced.
#
# Prerequisites
# -------------
# 1. Docker Engine 24+.
# 2. The licence-gated SDK archives placed in ``docker/oak/extra_packages/``:
#       openvino-2022.3.0.tar.gz   (RVC2 — OpenVINO 2022.3 dev archive)
#       snpe-2.32.6.zip            (RVC4 — Qualcomm Neural Processing SDK)
#    See ``docker/oak/extra_packages/README.md`` for download URLs and
#    naming requirements.
# 3. Network access to clone the modelconverter sources from GitHub. The
#    clone is pinned to MODELCONVERTER_REF for reproducibility.
#
# Usage
# -----
#     docker/oak/build.sh              # both rvc2 and rvc4 (default)
#     docker/oak/build.sh rvc2         # rvc2 only
#     docker/oak/build.sh rvc4         # rvc4 only
#     docker/oak/build.sh --no-cache   # rebuild from scratch
#
# Environment overrides
# ---------------------
#     MODELCONVERTER_REF        Git ref (tag/branch/sha) of luxonis/modelconverter
#                               to check out. Default: pinned tag below.
#     OPENVINO_VERSION          OpenVINO version (matches the archive name).
#                               Default: 2022.3.0.
#     SNPE_VERSION              SNPE version (matches the archive name).
#                               Default: 2.32.6.
#     IMAGE_TAG                 Tag suffix applied to both images.
#                               Default: local.
# ---------------------------------------------------------------------------
set -euo pipefail

MODELCONVERTER_REF="${MODELCONVERTER_REF:-v0.5.3-beta}"
OPENVINO_VERSION="${OPENVINO_VERSION:-2022.3.0}"
SNPE_VERSION="${SNPE_VERSION:-2.32.6}"
IMAGE_TAG="${IMAGE_TAG:-local}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/build"
SRC_DIR="${BUILD_ROOT}/modelconverter"
EXTRA_DIR="${SCRIPT_DIR}/extra_packages"

DOCKER_BUILD_FLAGS=()
TARGETS=()

for arg in "$@"; do
    case "${arg}" in
        rvc2|rvc4|both)        TARGETS+=("${arg}") ;;
        --no-cache|--pull)     DOCKER_BUILD_FLAGS+=("${arg}") ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# Default: build both
if [[ ${#TARGETS[@]} -eq 0 || " ${TARGETS[*]} " == *" both "* ]]; then
    TARGETS=(rvc2 rvc4)
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH. Install Docker Engine 24+." >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found on PATH." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Fetch / refresh the pinned modelconverter source tree
# ---------------------------------------------------------------------------
mkdir -p "${BUILD_ROOT}"
if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo ">> Cloning luxonis/modelconverter @ ${MODELCONVERTER_REF}"
    git clone --depth 1 --branch "${MODELCONVERTER_REF}" \
        https://github.com/luxonis/modelconverter.git "${SRC_DIR}"
else
    echo ">> Updating luxonis/modelconverter to ${MODELCONVERTER_REF}"
    git -C "${SRC_DIR}" fetch --depth 1 origin "${MODELCONVERTER_REF}"
    git -C "${SRC_DIR}" checkout --quiet FETCH_HEAD
fi

# ---------------------------------------------------------------------------
# 2. Stage the user-supplied SDK archives into the build context
# ---------------------------------------------------------------------------
mkdir -p "${SRC_DIR}/docker/extra_packages"

stage_archive() {
    local needed="$1"
    local target="${SRC_DIR}/docker/extra_packages/${needed}"
    local source="${EXTRA_DIR}/${needed}"
    if [[ ! -f "${source}" ]]; then
        echo "ERROR: ${needed} not found in ${EXTRA_DIR}/" >&2
        echo "       See docker/oak/extra_packages/README.md for download URLs." >&2
        return 1
    fi
    cp -f "${source}" "${target}"
    echo ">> Staged ${needed} ($(du -h "${source}" | cut -f1))"
}

# ---------------------------------------------------------------------------
# 3. Build per target
# ---------------------------------------------------------------------------
build_rvc2() {
    stage_archive "openvino-${OPENVINO_VERSION}.tar.gz"
    echo ">> Building luxonis/modelconverter-rvc2:${IMAGE_TAG}"
    docker build "${DOCKER_BUILD_FLAGS[@]}" \
        --build-arg "VERSION=${OPENVINO_VERSION}" \
        -f "${SRC_DIR}/docker/rvc2/Dockerfile" \
        -t "luxonis/modelconverter-rvc2:${IMAGE_TAG}" \
        "${SRC_DIR}"
}

build_rvc4() {
    stage_archive "snpe-${SNPE_VERSION}.zip"
    echo ">> Building luxonis/modelconverter-rvc4:${IMAGE_TAG}"
    docker build "${DOCKER_BUILD_FLAGS[@]}" \
        --build-arg "VERSION=${SNPE_VERSION}" \
        -f "${SRC_DIR}/docker/rvc4/Dockerfile" \
        -t "luxonis/modelconverter-rvc4:${IMAGE_TAG}" \
        "${SRC_DIR}"
}

for target in "${TARGETS[@]}"; do
    case "${target}" in
        rvc2) build_rvc2 ;;
        rvc4) build_rvc4 ;;
    esac
done

echo
echo "Done. Locally-built images:"
for target in "${TARGETS[@]}"; do
    echo "  luxonis/modelconverter-${target}:${IMAGE_TAG}"
done
echo
echo "Use them with the export pipeline:"
echo "  uv run python train_and_export.py --skip-train --models yolo11n \\"
echo "      --skip-npu --oak-target ${TARGETS[0]}"
