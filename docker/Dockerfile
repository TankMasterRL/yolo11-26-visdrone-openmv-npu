# syntax=docker/dockerfile:1.6
#
# GPU image for YOLO11/YOLO26 × VisDrone × OpenMV training & tuning.
#
# Builds on top of the official Ultralytics image, which already ships
# Ubuntu + CUDA + PyTorch + Ultralytics. We just layer the extras this
# project needs (Ray Tune, TensorBoard, Vela for AE3 NPU compilation).
#
# See https://docs.ultralytics.com/guides/docker-quickstart/
#
FROM ultralytics/ultralytics:latest

LABEL org.opencontainers.image.source="https://github.com/TankMasterRL/yolo11-26-visdrone-openmv-npu"
LABEL org.opencontainers.image.description="YOLO11/YOLO26 × VisDrone × OpenMV NPU — GPU training & tuning image"

# Additional Python dependencies layered on top of the Ultralytics base.
# --no-cache-dir keeps the image slim; --upgrade so we pick up wheels
# compatible with the base image's Python/torch version.
RUN pip install --no-cache-dir --upgrade \
    "ray[tune]>=2.10.0" \
    "tensorboard>=2.15.0" \
    "ethos-u-vela>=3.12.0"

# Ultralytics writes datasets to ~/datasets by default; we override this
# via the YOLO_DATASETS_DIR env var in docker-compose.yml so the download
# lands on a named volume that survives container recreation.
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_DATASETS_DIR=/datasets

WORKDIR /workspace

# Default to an interactive shell; docker-compose services override this
# with explicit commands (train / tune / tensorboard).
CMD ["/bin/bash"]
