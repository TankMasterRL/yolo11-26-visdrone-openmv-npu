"""
Region-Based Object Counting — OpenMV AE3 + YOLO11n (VisDrone)
================================================================

Board  : OpenMV AE3 (Alif Ensemble E3, Ethos-U55 256-MAC NPU)
Model  : YOLO11n trained on VisDrone, INT8 quantised, Vela-compiled
Input  : 256×256 RGB565

Counts objects whose centres fall inside user-defined rectangular
zones and prints per-zone tallies over the serial console each frame.

Files required on the camera (internal flash or SD):
  - yolo11n_int8_vela.tflite   (Vela-compiled model)
  - labels.txt                  (VisDrone class names, one per line)
  - region_counter.py           (shared counting module)

Adjust REGIONS below to match your scene geometry.
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
MODEL_PATH = "yolo11n_int8_vela.tflite"
LABELS_PATH = "labels.txt"
INPUT_SIZE = 256      # must match export --imgsz
MIN_SCORE = 0.35      # detection confidence threshold

# Counting regions — (x, y, w, h) in image coordinates (QVGA 320×240)
# Modify these to suit your camera placement.
REGIONS = [
    {"name": "Left",   "rect": (0,   0, 160, 240), "color": (255, 0, 0)},
    {"name": "Right",  "rect": (160, 0, 160, 240), "color": (0, 255, 0)},
]

# Which VisDrone classes to count (None = all).
# Indices: 0=pedestrian 1=people 2=bicycle 3=car 4=van
#          5=truck 6=tricycle 7=awning-tricycle 8=bus 9=motor
TARGET_CLASSES = None  # e.g. [0, 1] for people only

# ── Camera Setup ──────────────────────────────────────────────────────
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)       # 320×240
sensor.skip_frames(time=2000)

# ── Load Model ────────────────────────────────────────────────────────
# load_to_fb=True places model weights in frame-buffer stack (frees heap)
model = ml.Model(MODEL_PATH, postprocess=ultralytics.YOLO(), load_to_fb=True)

# ── Load Labels ───────────────────────────────────────────────────────
try:
    with open(LABELS_PATH, "r") as f:
        labels = [l.strip() for l in f if l.strip()]
except OSError:
    labels = None

# ── Region Counter ────────────────────────────────────────────────────
counter = RegionCounter(REGIONS, min_score=MIN_SCORE, target_classes=TARGET_CLASSES)

# ── Main Loop ─────────────────────────────────────────────────────────
clock = time.clock()
frame_n = 0

while True:
    try:
        clock.tick()
        img = sensor.snapshot()

        # Run YOLO inference — auto-resizes to INPUT_SIZE internally
        detections = model.predict([img])

        # Count objects per region and draw overlays
        counts = counter.update(detections, img, draw=True)

        # Print results
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
