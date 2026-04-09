# syntax=docker/dockerfile:1.6
#
# GPU image for YOLO11/YOLO26 × VisDrone × OpenMV training, tuning,
# and export (including NPU compilation).
#
# Builds on top of the official Ultralytics image, which already ships
# Ubuntu + CUDA + PyTorch + Ultralytics. We layer:
#   - Ray Tune + TensorBoard for hyperparameter search
#   - Ultralytics' full TFLite INT8 export dependency chain
#     (onnx → onnx2tf → tensorflow → tflite INT8)
#   - Arm Vela for OpenMV AE3 (Ethos-U55) NPU compilation
#
# STM32Cube.AI / STEdgeAI-Core (the N6 compiler) is proprietary and
# must be downloaded from ST — bind-mount it into /opt/stedgeai at
# runtime (see docker-compose.yml and the README "Docker" section).
#
# References:
#   - https://docs.ultralytics.com/guides/docker-quickstart/
#   - https://docs.ultralytics.com/integrations/tflite/
#   - https://docs.ultralytics.com/integrations/ray-tune/
#
FROM ultralytics/ultralytics:latest

LABEL org.opencontainers.image.source="https://github.com/TankMasterRL/yolo11-26-visdrone-openmv-npu"
LABEL org.opencontainers.image.description="YOLO11/YOLO26 × VisDrone × OpenMV NPU — GPU train/tune/export image"

# ---------------------------------------------------------------------------
# Python dependencies layered on top of the Ultralytics base.
#
# Split across RUN lines so Docker layer caching stays fine-grained and
# a failure in one block doesn't invalidate the others.
# ---------------------------------------------------------------------------

# 1. Hyperparameter tuning + visualisation
RUN pip install --no-cache-dir --upgrade \
    "ray[tune]>=2.10.0" \
    "tensorboard>=2.15.0"

# 2. TFLite INT8 export dependency chain.
#
# Ultralytics' exporter would lazy-install these at runtime, but baking
# them into the image avoids a first-run delay and keeps builds
# reproducible. Versions mirror what the Ultralytics CLI pins in
# `ultralytics/engine/exporter.py` — see also
# https://docs.ultralytics.com/integrations/tflite/
RUN pip install --no-cache-dir --upgrade \
    "tensorflow>=2.16.1" \
    "tf_keras<=2.19.0" \
    "onnx>=1.12.0,<2.0.0" \
    "onnx2tf>=1.26.3,<1.29.0" \
    "onnxslim>=0.1.71" \
    "onnxruntime" \
    "sng4onnx>=1.0.1" \
    "onnx_graphsurgeon>=0.3.26" \
    "ai-edge-litert>=1.2.0" \
    "protobuf>=5"

# 3. Arm Vela compiler for OpenMV AE3 (Ethos-U55 NPU)
RUN pip install --no-cache-dir --upgrade \
    "ethos-u-vela>=3.12.0"

# ---------------------------------------------------------------------------
# Optional STEdgeAI (N6 Neural-ART) integration.
#
# ST does not redistribute STM32Cube.AI / STEdgeAI-Core, so it cannot
# be installed via pip. Download it from
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

# Ultralytics writes datasets to ~/datasets by default; we override this
# via YOLO_DATASETS_DIR so the download lands on a named volume that
# survives container recreation.
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_DATASETS_DIR=/datasets

WORKDIR /workspace

# Default to an interactive shell; docker-compose services override this
# with explicit commands (train / tune / export / tensorboard).
CMD ["/bin/bash"]
