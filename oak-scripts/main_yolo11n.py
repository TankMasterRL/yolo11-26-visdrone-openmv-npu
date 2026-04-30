"""
Region-Based Object Counting — Luxonis OAK / OAK4 + YOLO11n (VisDrone)
=======================================================================

DepthAI v3 peripheral-mode entry point. Same functionality as
openmv-scripts/{ae3,n6}/main_yolo11n.py but driven by the host:
on-device YoloDetectionNetwork → host RegionCounter → printout.

Place the converted model next to this script (or pass --model):
  - OAK  (RVC2):  yolo11n.blob          (or .superblob)
  - OAK4 (RVC4):  yolo11n.dlc

Run:
    uv sync --extra oak
    python oak-scripts/main_yolo11n.py
"""

from pathlib import Path

from _pipeline import cli_parser, run


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "yolo11n.blob"
IMGSZ = 256
FAMILY = "yolo11"

# Regions defined on the model-input canvas (imgsz × imgsz). The display
# frame is the camera passthrough at the same resolution.
REGIONS = [
    {"name": "Left",  "rect": (0,        0, IMGSZ // 2, IMGSZ), "color": (0,   0, 255)},
    {"name": "Right", "rect": (IMGSZ // 2, 0, IMGSZ // 2, IMGSZ), "color": (0, 255,   0)},
]
TARGET_CLASSES = None  # e.g. [0, 1] for pedestrian + people only


if __name__ == "__main__":
    args = cli_parser(DEFAULT_MODEL, IMGSZ, FAMILY).parse_args()
    run(
        model_path=args.model,
        imgsz=args.imgsz,
        family=args.family,
        regions=REGIONS,
        min_score=args.min_score,
        iou_threshold=args.iou,
        target_classes=TARGET_CLASSES,
        fps_cap=args.fps if args.fps is not None else 30,
        display=not args.no_display,
    )
