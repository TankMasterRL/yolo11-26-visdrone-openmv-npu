# YOLO × VisDrone → OpenMV NPU Pipeline

End-to-end pipeline that **trains** YOLO11n / YOLO11s / YOLO26n / YOLO26s on the
[VisDrone](https://docs.ultralytics.com/datasets/detect/visdrone/) drone-imagery
dataset, **exports** INT8-quantised TFLite models, **compiles** NPU-optimised
binaries, and provides ready-to-run **OpenMV MicroPython** scripts for
region-based object counting on both the **OpenMV AE3** and **OpenMV N6**.

---

## Repository Layout

```
.
├── train_and_export.py              # Training + export + NPU compilation
├── openmv-scripts/
│   ├── region_counter.py            # Shared counting module (upload to all boards)
│   ├── ae3/                         # Scripts for OpenMV AE3
│   │   ├── main_yolo11n.py
│   │   ├── main_yolo11s.py
│   │   ├── main_yolo26n.py
│   │   └── main_yolo26s.py
│   └── n6/                          # Scripts for OpenMV N6
│   │   ├── main_yolo11n.py
│   │   ├── main_yolo11s.py
│   │   ├── main_yolo26n.py
│   │   └── main_yolo26s.py
└── README.md                        # This file
```

---

## Hardware Overview

| Feature | OpenMV AE3 | OpenMV N6 |
|---|---|---|
| **MCU** | Alif Ensemble E3 | STM32N6 |
| **CPU** | Dual CM55 (400 / 160 MHz) | CM55 @ 800 MHz |
| **NPU** | Dual Ethos-U55 (250 GOPS) | ST Neural-ART (600 GOPS) |
| **RAM** | 13.5 MB SRAM + 32 MB OctalSPI | 4.2 MB SRAM + 32 MB PSRAM |
| **Camera** | 1 MP global shutter | OV5640 5 MP |
| **Power** | < 60 mA @ 5 V | ~ 250 mA @ 5 V |
| **NPU tool** | Arm Vela compiler | ST STEdgeAI |
| **Model format** | `*_vela.tflite` | `*_int8.tflite` (auto NPU) |

---

## VisDrone Dataset

VisDrone contains drone-captured images with 10 object classes:

| Index | Class | Index | Class |
|---|---|---|---|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

Ultralytics auto-downloads and converts the dataset on first use.

---

## Model Comparison

| Model | Params | Export Size (INT8) | NMS | Best For |
|---|---|---|---|---|
| YOLO11n | ~2.6 M | ~256 KB–1 MB | Required | AE3 (memory-constrained) |
| YOLO11s | ~9.4 M | ~1–3 MB | Required | N6 (higher accuracy) |
| YOLO26n | ~2.5 M | ~256 KB–1 MB | Built-in (NMS-free) | AE3 / N6 (lowest latency) |
| YOLO26s | ~9.2 M | ~1–3 MB | Built-in (NMS-free) | N6 (best accuracy) |

> **YOLO26** is end-to-end NMS-free, meaning the model output already contains
> final detections — no post-processing NMS step is needed, reducing latency.

---

## 1. Prerequisites

### Training Host (PC / Server)

```bash
# Python 3.9+
pip install ultralytics>=8.3

# For OpenMV AE3 NPU compilation (Ethos-U55)
pip install ethos-u-vela

# For OpenMV N6 NPU compilation (optional — firmware auto-accelerates)
# Install STM32Cube.AI (X-CUBE-AI) from ST, add stedgeai to PATH
```

### OpenMV IDE

Download from [openmv.io](https://openmv.io/pages/download) for uploading
scripts and models to the cameras.

---

## 2. Training & Export

### Full pipeline (all 4 models)

```bash
python train_and_export.py --epochs 100 --imgsz 640
```

### Single model

```bash
python train_and_export.py --models yolo26n --epochs 50
```

### Export only (skip training, re-use existing weights)

```bash
python train_and_export.py --skip-train
```

### Custom export resolution

```bash
# Smaller input → faster inference, less memory, lower accuracy
python train_and_export.py --imgsz-export 192

# Larger input → slower but more accurate
python train_and_export.py --imgsz-export 320
```

### What happens

1. **Train** — Each model trains on VisDrone with auto-batch, cosine LR, and
   mosaic augmentation for 100 epochs (early-stopping at patience=20).
2. **Export** — Best weights are exported to TFLite with INT8 quantisation
   using VisDrone calibration images. Nano models export at 256×256, small
   models at 320×320.
3. **Vela compile (AE3)** — If `vela` is installed, the INT8 TFLite is
   compiled for the Ethos-U55-256 config matching the AE3 primary NPU.
4. **STEdgeAI compile (N6)** — If `stedgeai` is on PATH, the INT8 TFLite is
   compiled to a Neural-ART binary. *This step is optional* — the OpenMV N6
   firmware loads INT8 TFLite directly and routes compatible ops to the NPU.

### Output structure

```
export/
├── labels.txt                     # Class names for all models
├── tflite/                        # Raw INT8 TFLite exports
│   ├── yolo11n_int8.tflite
│   ├── yolo11s_int8.tflite
│   ├── yolo26n_int8.tflite
│   └── yolo26s_int8.tflite
├── ae3/                           # Vela-compiled for AE3
│   ├── yolo11n/
│   │   ├── yolo11n_int8_vela.tflite
│   │   └── yolo11n_int8.tflite    # fallback
│   └── ...
└── n6/                            # STEdgeAI output for N6
    ├── yolo11n/
    │   ├── st_ai_output/          # Neural-ART binary + headers
    │   └── yolo11n_int8.tflite    # direct-load fallback
    └── ...
```

---

## 3. Deploying to OpenMV Cameras

### OpenMV AE3

1. Connect the AE3 via USB and open OpenMV IDE.
2. Copy to the camera's filesystem:
   - `export/ae3/<model_name>/<model>_int8_vela.tflite`
   - `export/labels.txt`
   - `openmv-scripts/region_counter.py`
   - `openmv-scripts/ae3/main_<model>.py` → rename to `main.py`
3. Reset the board. The script runs on boot.

### OpenMV N6

1. Connect the N6 via USB and open OpenMV IDE.
2. Copy to the camera's filesystem:
   - `export/n6/<model_name>/<model>_int8.tflite`
   - `export/labels.txt`
   - `openmv-scripts/region_counter.py`
   - `openmv-scripts/n6/main_<model>.py` → rename to `main.py`
3. Reset the board.

> **Tip:** For the N6, if you ran `stedgeai` and have the Neural-ART `.raw`
> binary, flash it to external XSPI flash at address `0x70380000` using
> STM32CubeProgrammer. The firmware then loads the pre-compiled network
> directly from flash for maximum performance.

---

## 4. Customising the Counting Regions

Each main script defines a `REGIONS` list at the top:

```python
REGIONS = [
    {"name": "Left",   "rect": (0,   0, 160, 240), "color": (255, 0, 0)},
    {"name": "Right",  "rect": (160, 0, 160, 240), "color": (0, 255, 0)},
]
```

Each region is a rectangle `(x, y, width, height)` in **image coordinates**.
The AE3 scripts default to QVGA (320×240) and the N6 scripts to VGA (640×480).

### Example: Four-quadrant counting

```python
REGIONS = [
    {"name": "TL", "rect": (0,   0,   160, 120), "color": (255, 0,   0)},
    {"name": "TR", "rect": (160, 0,   160, 120), "color": (0,   255, 0)},
    {"name": "BL", "rect": (0,   120, 160, 120), "color": (0,   0,   255)},
    {"name": "BR", "rect": (160, 120, 160, 120), "color": (255, 255, 0)},
]
```

### Example: Filter to vehicles only

```python
# Only count: car (3), van (4), truck (5), bus (8)
TARGET_CLASSES = [3, 4, 5, 8]
```

---

## 5. Serial Output Format

Each frame prints:

```
FPS:12.3  total=7  Left=4 | Right=3
```

Parse this in your application code or pipe it over UART / WiFi to a host
system for aggregation and logging.

---

## 6. Expected Performance

| Model | Board | Resolution | Expected FPS | Notes |
|---|---|---|---|---|
| YOLO11n | AE3 | QVGA 320×240 | ~15–25 | Ethos-U55 accelerated |
| YOLO11s | AE3 | QVGA 320×240 | ~8–15 | Larger model, may need 256 input |
| YOLO26n | AE3 | QVGA 320×240 | ~18–30 | NMS-free saves ~2–3 ms/frame |
| YOLO26s | AE3 | QVGA 320×240 | ~10–18 | Best accuracy on AE3 |
| YOLO11n | N6 | VGA 640×480 | ~25–40 | Neural-ART 600 GOPS |
| YOLO11s | N6 | VGA 640×480 | ~15–25 | Good accuracy / speed balance |
| YOLO26n | N6 | VGA 640×480 | ~30–45 | Fastest option on N6 |
| YOLO26s | N6 | VGA 640×480 | ~18–30 | Best accuracy overall |

> FPS depends on scene complexity, number of detections, and whether the
> model was pre-compiled with the board's NPU toolchain.

---

## 7. Troubleshooting

**"MemoryError" on AE3:**
Reduce frame size to `sensor.QQVGA` (160×120) or lower the export `--imgsz`
to 192. Use `load_to_fb=True` when loading the model.

**Model loads but no detections:**
Check that `nms=False` was used during export (the `ultralytics.YOLO()`
postprocessor handles NMS in MicroPython). Verify `MIN_SCORE` isn't too high.

**Vela compilation fails:**
Ensure the model is fully INT8 quantised (including inputs/outputs).
Run `vela --supported-ops-report` to check operator compatibility.

**stedgeai not found:**
Install [STM32Cube.AI](https://www.st.com/en/embedded-software/x-cube-ai.html)
and add `<install_path>/Utilities/windows/` (or `linux/`) to your PATH.

---

## License

Training pipeline uses Ultralytics YOLO (AGPL-3.0 for open-source use).
VisDrone dataset: see the [original paper](https://arxiv.org/abs/2001.06303).
OpenMV MicroPython scripts: MIT.
