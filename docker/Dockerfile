# syntax=docker/dockerfile:1.6
#
# GPU image for YOLO11/YOLO26 × VisDrone × OpenMV training, tuning,
# and export (including NPU compilation).
#
# Builds on top of the official Ultralytics image, which already ships
# Ubuntu + CUDA + PyTorch + Ultralytics. We then use `uv` — the same
# Python package manager the host-side workflow uses — to layer on the
# extras declared in this project's pyproject.toml:
#
#   - Ray Tune + TensorBoard for hyperparameter search
#   - Ultralytics' full TFLite INT8 export dependency chain
#     (onnx → onnx2tf → tensorflow → tflite INT8)
#   - Arm Vela for OpenMV AE3 (Ethos-U55) NPU compilation
#
# Dependency versions are defined exactly once, in pyproject.toml, and
# resolved into the container via `uv pip install -r pyproject.toml
# --extra all` — so the Docker image and the host `uv sync` workflow
# stay in lock-step automatically.
#
# STM32Cube.AI / STEdgeAI-Core (the N6 compiler) is proprietary and
# must be downloaded from ST — bind-mount it into /opt/stedgeai at
# runtime (see docker-compose.yml and the README "Docker" section).
#
# ---------------------------------------------------------------------------
# Reproducibility contract
# ---------------------------------------------------------------------------
# The build is pinned on three axes so that re-building the image on a
# different day or a different host yields the same installed software:
#
#   1. Base image — pinned to a specific Ultralytics tag AND digest. The
#      digest is authoritative; the tag is a human-readable label that
#      must match the digest or the pull fails.
#   2. uv binary  — pinned to a specific release tag of the official
#      ghcr.io/astral-sh/uv distroless image.
#   3. PyPI snapshot — `uv pip install --exclude-newer=<UTC date>` caps
#      dependency resolution to packages published on or before that
#      date, so `>=`-style constraints in pyproject.toml no longer drift
#      as new upstream releases land.
#
# All three are exposed as build ARGs and can be overridden, e.g. to
# refresh the snapshot:
#
#     docker build \
#         --build-arg ULTRALYTICS_TAG=8.4.42 \
#         --build-arg ULTRALYTICS_DIGEST=sha256:<new-digest> \
#         --build-arg UV_TAG=0.11.8 \
#         --build-arg UV_EXCLUDE_NEWER=2026-07-01 \
#         -f docker/Dockerfile -t yolo11-26-visdrone-openmv:gpu .
#
# When bumping defaults, update all three together so the pinned state
# stays internally consistent.
#
# References:
#   - https://docs.ultralytics.com/guides/docker-quickstart/
#   - https://docs.ultralytics.com/integrations/tflite/
#   - https://docs.ultralytics.com/integrations/ray-tune/
#   - https://docs.astral.sh/uv/guides/integration/docker/
#   - https://docs.astral.sh/uv/reference/cli/#uv-pip-install  (--exclude-newer)
#
ARG ULTRALYTICS_TAG=8.4.41
ARG ULTRALYTICS_DIGEST=sha256:8b4551009fe170cf81d72995047be132df8b91380c225f37217cdd0d4ffd4dab
ARG UV_TAG=0.11.7

# ---------------------------------------------------------------------------
# Alias the official uv distroless image as a named build stage. Global
# ARGs are guaranteed to expand in FROM instructions, whereas BuildKit's
# reference parser for `COPY --from=<external-image>:${ARG}` treats the
# ${...} literally ("invalid reference format"). Going through a stage
# alias is the portable workaround.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:${UV_TAG} AS uv

FROM ultralytics/ultralytics:${ULTRALYTICS_TAG}@${ULTRALYTICS_DIGEST}

LABEL org.opencontainers.image.source="https://github.com/TankMasterRL/yolo11-26-visdrone-openmv-npu"
LABEL org.opencontainers.image.description="YOLO11/YOLO26 × VisDrone × OpenMV NPU — GPU train/tune/export image"

# ---------------------------------------------------------------------------
# Install uv (fast Python package / project manager) from the aliased
# stage above. This is the same binary used by the host-side `uv sync`
# workflow, so containerised and local dependency resolution stay
# byte-for-byte compatible.
# ---------------------------------------------------------------------------
COPY --from=uv /uv /uvx /bin/

# ---------------------------------------------------------------------------
# Install project extras via uv using pyproject.toml as the single
# source of truth. We install --system rather than letting uv create a
# .venv so the Ultralytics base image's pre-installed, GPU-linked
# torch/CUDA build is preserved and our extras are layered on top.
#
# `--exclude-newer` caps dependency resolution at a fixed UTC date so
# `>=`-style constraints in pyproject.toml don't silently drift when
# new upstream releases are published — rebuilding tomorrow with the
# same ARGs resolves to the same versions as today.
#
# UV_LINK_MODE=copy is required because /tmp and /usr/local are on
# different mounts in some CI builders — symlinking would fail.
# UV_NO_CACHE shrinks the final layer and avoids stale metadata.
# ---------------------------------------------------------------------------
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1

# This date MUST be >= the publish date of the newest pinned floor in
# pyproject.toml (e.g. ultralytics 8.4.56 landed 2026-05-27, ai-edge-litert
# 2.1.5 on 2026-05-15). An older snapshot caps resolution below those floors
# and makes the constraint set unsatisfiable, so bump this whenever a
# dependency lower bound is raised.
ARG UV_EXCLUDE_NEWER=2026-06-06
ENV UV_EXCLUDE_NEWER=${UV_EXCLUDE_NEWER}

COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --no-cache \
        --exclude-newer "${UV_EXCLUDE_NEWER}" \
        -r /tmp/pyproject.toml --extra all \
    && rm -f /tmp/pyproject.toml

# ---------------------------------------------------------------------------
# Optional STEdgeAI (N6 Neural-ART) integration.
#
# ST does not redistribute STM32Cube.AI / STEdgeAI-Core, so it cannot
# be installed via pip or uv. Download it from
# https://www.st.com/en/development-tools/stedgeai-cubeai.html
# and bind-mount the install root to /opt/stedgeai at runtime, e.g.
#
#   docker compose run --rm \
#       -v /host/path/to/stedgeai:/opt/stedgeai:ro export
#
# The PATH below covers the common executable locations inside that
# install tree. If the mount is absent, `train_and_export.py` gracefully
# skips N6 NPU compilation and still produces the plain INT8 TFLite.
# ---------------------------------------------------------------------------
ENV PATH="/opt/stedgeai/bin:/opt/stedgeai/Utilities/linux:/opt/stedgeai/Utilities/windows:${PATH}"

# Ultralytics writes datasets to ~/datasets by default; we override
# this via YOLO_DATASETS_DIR so the download lands on a named volume
# that survives container recreation.
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_DATASETS_DIR=/datasets

WORKDIR /workspace

# Default to an interactive shell; docker-compose services override
# this with explicit commands (train / tune / export / tensorboard).
CMD ["/bin/bash"]
