"""
Region-Based Object Counting — OpenMV N6 + YOLO11n (VisDrone)
==============================================================

Board  : OpenMV N6 (STM32N6 800 MHz CM55, ST Neural-ART 600 GOPS NPU)
Model  : YOLO11n trained on VisDrone, INT8 quantised
Input  : 256×256 RGB565

The N6 firmware automatically routes INT8 TFLite ops to the
Neural-ART NPU — no manual compilation required.  If you did run
stedgeai, the resulting .raw binary is flashed separately; the
MicroPython script still loads the .tflite file normally.

Files required on the camera:
  - yolo11n_int8.tflite         (INT8 quantised model)
  - labels.txt                   (VisDrone class names)
  - region_counter.py            (shared counting module)

The N6 has 32 MB PSRAM so we can use VGA (640×480) for a wider
field-of-view and still run YOLO at good frame rates.
"""

import sensor
import time
import gc
import micropython
import ml
from ml.postprocessing import ultralytics
from region_counter import RegionCounter

micropython.alloc_emergency_exception_buf(100)

# ── Configuration ─────────────────────────────────────────────────────
MODEL_PATH = "yolo11n_int8.tflite"
LABELS_PATH = "labels.txt"
INPUT_SIZE = 256
MIN_SCORE = 0.35

# N6 has enough memory for VGA — define regions on a 640×480 canvas
REGIONS = [
    {"name": "Left",   "rect": (0,   0, 320, 480), "color": (255, 0, 0)},
    {"name": "Right",  "rect": (320, 0, 320, 480), "color": (0, 255, 0)},
]

TARGET_CLASSES = None

# ── Camera Setup ──────────────────────────────────────────────────────
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.VGA)        # 640×480 — N6 has plenty of RAM
sensor.skip_frames(time=2000)

# ── Load Model ────────────────────────────────────────────────────────
model = ml.Model(MODEL_PATH, postprocess=ultralytics.YOLO(), load_to_fb=True)

try:
    with open(LABELS_PATH, "r") as f:
        labels = [l.strip() for l in f if l.strip()]
except OSError:
    labels = None

counter = RegionCounter(REGIONS, min_score=MIN_SCORE, target_classes=TARGET_CLASSES)

# ── Main Loop ─────────────────────────────────────────────────────────
clock = time.clock()
frame_n = 0

while True:
    try:
        clock.tick()
        img = sensor.snapshot()
        detections = model.predict([img])
        counts = counter.update(detections, img, draw=True)
        total = counter.total(counts)
        parts = " | ".join("%s=%d" % (k, v) for k, v in counts.items())
        print("FPS:%.1f  total=%d  %s" % (clock.fps(), total, parts))
        frame_n += 1
        if frame_n % 50 == 0:
            gc.collect()
    except MemoryError:
        gc.collect()
    except Exception as e:
        print("ERR:", e)
        time.sleep_ms(200)
