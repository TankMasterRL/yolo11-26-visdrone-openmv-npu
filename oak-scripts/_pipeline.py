"""
_pipeline.py — DepthAI v3 peripheral-mode pipeline for OAK / OAK4
==================================================================

Builds and runs a DepthAI v3 pipeline that mirrors the OpenMV
region-counting scripts: Camera -> NN -> RegionCounter -> printout +
overlay window. Designed for highest-FPS / lowest-latency host-driven
operation.

Topology
--------
    Camera (CAM_A)
        └─ requestOutput(size=(imgsz,imgsz), BGR888p, fps=fps_cap,
                         resizeMode=LETTERBOX)            ← matches OpenMV
            ├─→ NeuralNetwork / YoloDetectionNetwork.input
            └─→ display queue (passthrough is time-aligned with detections)

Two NN node types are used depending on the model family:

  * ``yolo11`` (anchor-free head, NMS expected on consumer)
        → ``dai.node.YoloDetectionNetwork``. On-device decode + NMS gives
        the highest FPS — host CPU is free for region counting + display.

  * ``yolo26`` (NMS-free end-to-end head; model emits final boxes)
        → ``dai.node.NeuralNetwork`` + a small host-side parser. Using
        ``YoloDetectionNetwork`` here would re-run NMS on already-final
        boxes and break correctness.

Peripheral mode
---------------
The pipeline runs on the OAK device but is host-driven over USB: the
host opens the device, uploads the graph, and pulls decoded
``ImgDetections`` back through output queues. No standalone bootloader
flashing is involved, which matches what the user asked for.

References
----------
- DepthAI v3 Camera + NN nodes:
    https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/neural_network/
- v2 → v3 porting (createOutputQueue, host-driven loop):
    https://docs.luxonis.com/software-v3/depthai/tutorials/v2-vs-v3/
- Performance tuning (setFps, frame format choices):
    https://docs.luxonis.com/software-v3/depthai/tutorials/optimizing/
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from region_counter import RegionCounter


# ---------------------------------------------------------------------------
# Detection adapters
# ---------------------------------------------------------------------------

class _Detection:
    """Duck-typed detection object accepted by RegionCounter.

    Wraps either ``dai.ImgDetection`` (normalised xyxy in [0, 1]) or a
    NumPy row produced by the YOLO26 host parser.
    """

    __slots__ = ("_bbox_xywh", "_score", "_cls")

    def __init__(self, bbox_xywh, score, cls):
        self._bbox_xywh = bbox_xywh
        self._score = float(score)
        self._cls = int(cls)

    def rect(self):
        return self._bbox_xywh

    def score(self):
        return self._score

    def class_id(self):
        return self._cls


def _adapt_dai_detections(dai_dets, frame_w, frame_h):
    """Convert ``dai.ImgDetections`` (normalised xyxy) to ``_Detection``s."""
    out = []
    for d in dai_dets.detections:
        x1 = max(0, int(d.xmin * frame_w))
        y1 = max(0, int(d.ymin * frame_h))
        x2 = min(frame_w, int(d.xmax * frame_w))
        y2 = min(frame_h, int(d.ymax * frame_h))
        out.append(_Detection((x1, y1, x2 - x1, y2 - y1), d.confidence, d.label))
    return out


# ---------------------------------------------------------------------------
# YOLO26 host-side parser (NMS-free, end-to-end head)
# ---------------------------------------------------------------------------

def _parse_yolo26(nn_data, frame_w, frame_h, imgsz, min_score):
    """Parse a YOLO26 NMS-free head into ``_Detection``s.

    YOLO26 emits already-final boxes — typically one tensor of shape
    ``(num_dets, 6)`` with rows ``[x1, y1, x2, y2, score, class]``, with
    coordinates in input-pixel space (``imgsz × imgsz``). We score-filter
    and rescale to the display frame here. No NMS is run.

    The exact tensor layout depends on the export — we probe the first
    output tensor and reshape to ``(N, 6)`` if possible. If your export
    produces a different layout, override this parser via
    ``run(parser=...)``.
    """
    # NeuralNetwork's first output, as float32 numpy.
    layer_names = nn_data.getAllLayerNames()
    if not layer_names:
        return []
    raw = np.array(nn_data.getLayerFp16(layer_names[0]), dtype=np.float32)
    if raw.size == 0 or raw.size % 6 != 0:
        return []
    rows = raw.reshape(-1, 6)

    sx = frame_w / float(imgsz)
    sy = frame_h / float(imgsz)
    out = []
    for r in rows:
        score = float(r[4])
        if score < min_score:
            continue
        x1 = max(0, int(r[0] * sx))
        y1 = max(0, int(r[1] * sy))
        x2 = min(frame_w, int(r[2] * sx))
        y2 = min(frame_h, int(r[3] * sy))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(_Detection((x1, y1, x2 - x1, y2 - y1), score, int(r[5])))
    return out


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    model_path: Path,
    imgsz: int,
    family: str,
    min_score: float,
    iou_threshold: float,
    num_classes: int,
    fps_cap: int,
):
    """Construct the DepthAI v3 pipeline + return its handles.

    Returns
    -------
    (pipeline, q_det, q_frame, nn_kind)
        * ``pipeline`` — built ``dai.Pipeline`` (not yet started).
        * ``q_det`` — output queue producing detections (DAI or NNData).
        * ``q_frame`` — output queue producing the display frame
          (NN passthrough, time-aligned with detections).
        * ``nn_kind`` — "yolo" or "raw"; switches host-side parsing.
    """
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        size=(imgsz, imgsz),
        type=dai.ImgFrame.Type.BGR888p,
        fps=fps_cap,
        resizeMode=dai.ImgResizeMode.LETTERBOX,
    )

    if family == "yolo11":
        nn = pipeline.create(dai.node.YoloDetectionNetwork)
        nn.setBlobPath(str(model_path))
        nn.setNumClasses(num_classes)
        nn.setConfidenceThreshold(min_score)
        nn.setIouThreshold(iou_threshold)
        nn.setCoordinateSize(4)
        nn.setAnchors([])
        nn.setAnchorMasks({})
        nn.input.setBlocking(False)
        nn.input.setMaxSize(2)
        cam_out.link(nn.input)
        q_det = nn.out.createOutputQueue(maxSize=4, blocking=False)
        q_frame = nn.passthrough.createOutputQueue(maxSize=4, blocking=False)
        return pipeline, q_det, q_frame, "yolo"

    if family == "yolo26":
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(model_path))
        nn.input.setBlocking(False)
        nn.input.setMaxSize(2)
        cam_out.link(nn.input)
        q_det = nn.out.createOutputQueue(maxSize=4, blocking=False)
        q_frame = nn.passthrough.createOutputQueue(maxSize=4, blocking=False)
        return pipeline, q_det, q_frame, "raw"

    raise ValueError(f"Unsupported family: {family!r} (expected yolo11 or yolo26)")


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run(
    model_path,
    imgsz,
    family,
    regions,
    *,
    min_score=0.35,
    iou_threshold=0.45,
    num_classes=10,
    target_classes=None,
    fps_cap=30,
    display=True,
):
    """Run the OAK / OAK4 pipeline and stream detections through RegionCounter.

    Mirrors the OpenMV main loop: per-frame ``RegionCounter.update`` +
    a ``FPS:%.1f total=%d ...`` printout in the same format as
    ``openmv-scripts/{ae3,n6}/main_yolo*.py`` line 86.

    Press ``q`` in the display window to quit (when ``display=True``).
    """
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artefact not found: {model_path}")

    pipeline, q_det, q_frame, nn_kind = build_pipeline(
        model_path=model_path,
        imgsz=imgsz,
        family=family,
        min_score=min_score,
        iou_threshold=iou_threshold,
        num_classes=num_classes,
        fps_cap=fps_cap,
    )
    counter = RegionCounter(regions, min_score=min_score, target_classes=target_classes)

    pipeline.start()
    fps_window = deque(maxlen=30)
    last_t = time.monotonic()
    try:
        while pipeline.isRunning():
            frame_msg = q_frame.get()
            det_msg = q_det.get()
            frame = frame_msg.getCvFrame()
            h, w = frame.shape[:2]

            if nn_kind == "yolo":
                detections = _adapt_dai_detections(det_msg, w, h)
            else:
                detections = _parse_yolo26(det_msg, w, h, imgsz, min_score)

            counts = counter.update(detections, frame, draw=display)

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_window.append(1.0 / dt)
            fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

            total = counter.total(counts)
            parts = " | ".join("%s=%d" % (k, v) for k, v in counts.items())
            print("FPS:%.1f  total=%d  %s" % (fps, total, parts))

            if display:
                cv2.imshow("OAK YOLO region-counter", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if display:
            cv2.destroyAllWindows()


def cli_parser(default_model: Path, default_imgsz: int, default_family: str):
    """Shared argparse skeleton used by every ``main_yolo*.py`` entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=default_model,
                   help=f"Path to compiled model (default: {default_model.name})")
    p.add_argument("--imgsz", type=int, default=default_imgsz,
                   help=f"Model input size (default: {default_imgsz})")
    p.add_argument("--family", choices=["yolo11", "yolo26"], default=default_family,
                   help=f"YOLO family (default: {default_family})")
    p.add_argument("--min-score", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--fps", type=int, default=None,
                   help="Camera fps cap (default: 30 for nano, 15 for small)")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless — print only, skip the OpenCV window")
    return p
