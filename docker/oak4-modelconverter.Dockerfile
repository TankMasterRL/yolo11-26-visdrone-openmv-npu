# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Luxonis ModelConverter for RVC4 (OAK4) — local build recipe
# ---------------------------------------------------------------------------
#
# Why this file exists
# --------------------
# The official luxonis/modelconverter-rvc4 image is **not** published on
# Docker Hub: the conversion tool relies on Qualcomm's SNPE SDK, which is
# licence-gated and cannot be redistributed by Luxonis. End users must
# accept Qualcomm's licence, download the SDK themselves, and build the
# image locally. See:
#   https://docs.luxonis.com/software-v3/ai-inference/conversion/rvc-conversion/offline/snpe/
#   https://github.com/luxonis/modelconverter
#
# What you provide
# ----------------
# Place the SNPE archive (e.g. `qualcomm_neural_processing_sdk_v2.x.x.zip`
# or `.tar.gz`) at:
#     docker/snpe/qualcomm_neural_processing_sdk.zip
# (the `docker/snpe/` directory is gitignored). The build will unpack it
# inside the image at /opt/snpe.
#
# Build
# -----
#     docker build \
#         -f docker/oak4-modelconverter.Dockerfile \
#         -t luxonis/modelconverter-rvc4:local \
#         docker/
#
# Use
# ---
#     uv run python train_and_export.py --skip-train \
#         --models yolo11n --skip-npu --oak-target rvc4 \
#         --oak-rvc4-image luxonis/modelconverter-rvc4:local
#
# This image is invoked by ``compile_modelconverter`` in
# train_and_export.py with:
#     docker run --rm -v <out>:/work -w /work <tag> \
#         convert rvc4 --config modelconverter.yaml --output /work
# ---------------------------------------------------------------------------

ARG MODELCONVERTER_REF=main

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git unzip xz-utils \
        python3 python3-pip python3-venv \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 1: SNPE SDK ----------------------------------------------------
# The user must drop the SNPE archive into docker/snpe/ before building.
# We accept either .zip or .tar.gz to match Qualcomm's distribution forms.
WORKDIR /opt/snpe-stage
COPY snpe/ /opt/snpe-stage/
RUN set -eux; \
    archive=$(ls /opt/snpe-stage/*.zip /opt/snpe-stage/*.tar.gz 2>/dev/null | head -n1 || true); \
    if [ -z "${archive}" ]; then \
        echo "ERROR: no SNPE SDK archive found under docker/snpe/."; \
        echo "Download the Qualcomm Neural Processing SDK and place it there."; \
        exit 1; \
    fi; \
    mkdir -p /opt/snpe; \
    case "${archive}" in \
        *.zip)    unzip -q "${archive}" -d /opt/snpe ;; \
        *.tar.gz) tar -xzf "${archive}" -C /opt/snpe ;; \
    esac; \
    # Collapse any single top-level dir so /opt/snpe/{bin,lib,...} is canonical.
    inner=$(find /opt/snpe -mindepth 1 -maxdepth 1 -type d | head -n1); \
    if [ -n "${inner}" ] && [ "$(ls /opt/snpe | wc -l)" = "1" ]; then \
        mv "${inner}"/* /opt/snpe/ && rmdir "${inner}"; \
    fi

ENV SNPE_ROOT=/opt/snpe \
    PATH=/opt/snpe/bin/x86_64-linux-clang:/opt/snpe/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/snpe/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-} \
    PYTHONPATH=/opt/snpe/lib/python:${PYTHONPATH:-}

# --- Stage 2: ModelConverter ---------------------------------------------
# Pull the upstream Luxonis modelconverter sources and install with pip.
# Pin to MODELCONVERTER_REF so reproducible local builds are possible.
WORKDIR /opt/modelconverter
RUN git clone --depth 1 --branch "${MODELCONVERTER_REF}" \
        https://github.com/luxonis/modelconverter.git . \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir .

ENTRYPOINT ["modelconverter"]
CMD ["--help"]
