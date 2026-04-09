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
# References:
#   - https://docs.ultralytics.com/guides/docker-quickstart/
#   - https://docs.ultralytics.com/integrations/tflite/
#   - https://docs.ultralytics.com/integrations/ray-tune/
#   - https://docs.astral.sh/uv/guides/integration/docker/
#
FROM ultralytics/ultralytics:latest

LABEL org.opencontainers.image.source="https://github.com/TankMasterRL/yolo11-26-visdrone-openmv-npu"
LABEL org.opencontainers.image.description="YOLO11/YOLO26 × VisDrone × OpenMV NPU — GPU train/tune/export image"

# ---------------------------------------------------------------------------
# Install uv (fast Python package / project manager) from the official
# distroless image. This is the same binary used by the host-side
# `uv sync` workflow, so containerised and local dependency resolution
# stay byte-for-byte compatible.
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---------------------------------------------------------------------------
# Install project extras via uv using pyproject.toml as the single
# source of truth. We install --system rather than letting uv create a
# .venv so the Ultralytics base image's pre-installed, GPU-linked
# torch/CUDA build is preserved and our extras are layered on top.
#
# UV_LINK_MODE=copy is required because /tmp and /usr/local are on
# different mounts in some CI builders — symlinking would fail.
# UV_NO_CACHE shrinks the final layer and avoids stale metadata.
# ---------------------------------------------------------------------------
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1

COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --no-cache \
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
