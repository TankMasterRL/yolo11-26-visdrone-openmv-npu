"""
Region-Based Object Counting — Luxonis OAK / OAK4 + YOLO26n (VisDrone)
=======================================================================

DepthAI v3 peripheral-mode entry point. YOLO26 has a built-in NMS-free
end-to-end head, so the pipeline uses dai.node.NeuralNetwork + a small
host parser instead of YoloDetectionNetwork (see _pipeline.py).
"""

from pathlib import Path

from _pipeline import cli_parser, run


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "yolo26n.blob"
IMGSZ = 256
FAMILY = "yolo26"

REGIONS = [
    {"name": "Left",  "rect": (0,        0, IMGSZ // 2, IMGSZ), "color": (0,   0, 255)},
    {"name": "Right", "rect": (IMGSZ // 2, 0, IMGSZ // 2, IMGSZ), "color": (0, 255,   0)},
]
TARGET_CLASSES = None


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
