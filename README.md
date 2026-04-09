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
├── tune_hyperparameters.py          # Ray Tune hyperparameter search + TensorBoard
├── docker/
│   └── Dockerfile                   # GPU image extending ultralytics/ultralytics
├── docker-compose.yml               # Train / tune / tensorboard / shell services
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

This project uses [uv](https://docs.astral.sh/uv/) for Python environment
management. Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the project environment (creates `.venv/` and installs deps):

```bash
# Core training + export dependencies
uv sync

# Include Arm Vela for OpenMV AE3 NPU compilation
uv sync --extra ae3

# Include Ray Tune + TensorBoard for hyperparameter search
uv sync --extra tune
```

All subsequent commands should be prefixed with `uv run` to use the
managed environment, e.g. `uv run python train_and_export.py ...`.

For OpenMV N6 NPU compilation (optional — firmware auto-accelerates),
install [STM32Cube.AI (X-CUBE-AI)](https://www.st.com/en/embedded-software/x-cube-ai.html)
from ST and add `stedgeai` to your PATH.

### OpenMV IDE

Download from [openmv.io](https://openmv.io/pages/download) for uploading
scripts and models to the cameras.

### Docker (GPU) — recommended for training & tuning

The repository ships a Docker Compose setup that bundles CUDA, PyTorch,
Ultralytics, Ray Tune, TensorBoard and Arm Vela in a single GPU image —
no local Python setup required beyond Docker itself. The image extends
the official [`ultralytics/ultralytics`](https://hub.docker.com/r/ultralytics/ultralytics)
base (see the [Ultralytics Docker Quickstart](https://docs.ultralytics.com/guides/docker-quickstart/)).

**Host prerequisites:**
- NVIDIA GPU + driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker Engine 23+ with Compose v2

**Quickstart:**

```bash
# Build the GPU image (once)
docker compose build

# Full training + export pipeline (all 4 models)
docker compose run --rm train

# Override args — e.g. train only yolo11n for 20 epochs
docker compose run --rm train python train_and_export.py --models yolo11n --epochs 20

# Ray Tune hyperparameter sweep for all four models
docker compose run --rm tune

# Export-only: re-use existing best.pt files, produce INT8 TFLite
# + Vela-compiled AE3 models for all four variants
docker compose run --rm export

# Export a single model at a reduced input size
docker compose run --rm export python train_and_export.py \
    --skip-train --models yolo11n --imgsz-export 192

# TensorBoard on http://localhost:6006 (Ctrl-C to stop)
docker compose up tensorboard

# Interactive GPU shell for ad-hoc yolo/uv commands
docker compose run --rm shell
```

**What's in the image** (`docker/Dockerfile`):
- Base: `ultralytics/ultralytics:latest` (Ubuntu + CUDA + PyTorch + Ultralytics)
- [`uv`](https://docs.astral.sh/uv/) — the same fast Python package manager
  the host workflow uses — installed from the official distroless image
- All `[project.optional-dependencies].all` extras from `pyproject.toml`,
  resolved and installed into the base image's system Python via
  `uv pip install --system -r pyproject.toml --extra all`. This keeps
  the Docker image and host `uv sync --extra all` workflow in lock-step:
  - `ray[tune]` + `tensorboard` for hyperparameter search & visualisation
  - Full TFLite INT8 export chain: `tensorflow`, `tf_keras`, `onnx`,
    `onnx2tf`, `onnxslim`, `onnxruntime`, `sng4onnx`, `onnx_graphsurgeon`,
    `ai-edge-litert`, `protobuf` (versions pinned per the
    [Ultralytics TFLite integration guide](https://docs.ultralytics.com/integrations/tflite/))
  - `ethos-u-vela` for OpenMV AE3 (Ethos-U55) NPU compilation

**Enabling OpenMV N6 (Neural-ART) compilation:**

STM32Cube.AI / STEdgeAI-Core is proprietary and not redistributable, so
it is not baked into the image. Download the Linux installer from
[STEDGEAI-CUBEAI on st.com](https://www.st.com/en/development-tools/stedgeai-cubeai.html),
extract it on the host, and bind-mount it at `/opt/stedgeai` when
running the export service:

```bash
docker compose run --rm \
    -v /host/path/to/stedgeai:/opt/stedgeai:ro \
    export
```

The image already has `/opt/stedgeai/bin` and
`/opt/stedgeai/Utilities/{linux,windows}` on its `PATH`, so
`train_and_export.py` picks up `stedgeai` automatically and emits the
Neural-ART `.raw` binary under `export/n6/<model>/st_ai_output/`. If
the mount is absent, the script prints a warning and still produces
the plain INT8 TFLite that the N6 firmware can load directly.

**What's mounted** (`docker-compose.yml`):
- `.` → `/workspace` — live project source
- Named volume `datasets` → `/datasets` (`YOLO_DATASETS_DIR`) so VisDrone is downloaded only once
- Named volume `ultralytics-cache` → `/root/.config/Ultralytics`
- Named volume `pip-cache` → `/root/.cache/pip`

All GPU services run with `ipc: host`, `shm_size: 8gb`, and
`deploy.resources.reservations.devices: nvidia` so PyTorch DataLoader
workers and multi-GPU training work correctly.

---

## 2. Training & Export

### Full pipeline (all 4 models)

```bash
uv run python train_and_export.py --epochs 100 --imgsz 640
```

### Single model

```bash
uv run python train_and_export.py --models yolo26n --epochs 50
```

### Export only (skip training, re-use existing weights)

```bash
uv run python train_and_export.py --skip-train
```

### Custom export resolution

```bash
# Smaller input → faster inference, less memory, lower accuracy
uv run python train_and_export.py --imgsz-export 192

# Larger input → slower but more accurate
uv run python train_and_export.py --imgsz-export 320
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

## 2b. Hyperparameter Tuning (Ray Tune + TensorBoard)

This project includes `tune_hyperparameters.py`, a wrapper around
Ultralytics' built-in
[Ray Tune integration](https://docs.ultralytics.com/integrations/ray-tune/)
with a VisDrone-specific search space tuned for small-object drone imagery.

### Install tuning dependencies

```bash
uv sync --extra tune      # pulls in ray[tune] and tensorboard
```

### Run a tuning sweep

```bash
# Default: runs an independent sweep for ALL FOUR models
# (yolo11n, yolo11s, yolo26n, yolo26s), 10 trials × 30 epochs each
uv run python tune_hyperparameters.py

# Subset of models with more trials and 1 GPU per trial
uv run python tune_hyperparameters.py \
    --models yolo11s yolo26s \
    --iterations 30 \
    --epochs 50 \
    --gpu-per-trial 1

# Single model, using Ultralytics' full default 28-parameter search space
uv run python tune_hyperparameters.py --models yolo11n --default-space
```

Key flags (see `--help` for the full list):

| Flag | Default | Description |
|---|---|---|
| `--models` | all four | Subset of `yolo11n yolo11s yolo26n yolo26s` to tune |
| `--iterations` | `10` | Number of hyperparameter samples **per model** |
| `--epochs` | `30` | Max epochs per trial (ASHA prunes bad trials early) |
| `--grace-period` | `10` | ASHA grace period before pruning is allowed |
| `--gpu-per-trial` | auto | Fractional GPUs per trial (e.g. `0.5` → 2 trials/GPU) |
| `--default-space` | off | Swap in Ultralytics' full 28-parameter default space |

Internally the script calls `model.tune(use_ray=True, ...)` once per
selected model. Ray Tune orchestrates an **ASHA scheduler**
(`grace_period`, `reduction_factor=3`), and each model's best trial is
written to `runs/tune/visdrone_raytune_<model>_best_hyperparameters.json`.
A summary table of per-model mAP is printed at the end.

### Visualise with TensorBoard

Ultralytics writes TFEvent logs for each trial automatically. Launch
TensorBoard against the tuning run directory:

```bash
uv run tensorboard --logdir runs/tune
# or, to see Ray Tune's own metrics for all models:
uv run tensorboard --logdir ~/ray_results
```

Then open <http://localhost:6006>. Every trial of every model appears
as its own run (grouped by the `visdrone_raytune_<model>` experiment
name), so you can overlay `metrics/mAP50-95(B)`, `metrics/mAP50(B)`,
training losses, and learning-rate curves across the whole sweep.

### Re-train with the best hyperparameters

Copy the values from `*_best_hyperparameters.json` into the
`DEFAULT_TRAIN_ARGS` of `train_and_export.py` (or pass them as CLI
overrides via Ultralytics' YAML config) and run the full export
pipeline to produce your final deployable INT8 TFLite.

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
