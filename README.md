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
├── pyproject.toml                   # Single source of truth for deps + extras
├── docker/
│   └── Dockerfile                   # GPU image extending ultralytics/ultralytics
├── docker-compose.yml               # Train / tune / export / tensorboard / shell services
├── notebooks/
│   └── visdrone_pipeline.ipynb      # End-to-end Colab / Paperspace / Kaggle notebook
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
├── CLAUDE.md                        # Guidance for Claude Code / LLM agents
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

The five Compose services (`train`, `tune`, `export`, `tensorboard`,
`shell`) all share a single image tag (`yolo11-26-visdrone-openmv:gpu`).
Only the `train` service carries a `build:` block; the others reference
the same tag via `image:` to avoid the buildx-bake race that occurs
when multiple targets export to the same tag in parallel. As a result
`docker compose build` (no args) performs a single, deterministic build.

**Reproducible rebuilds.** The Dockerfile pins three things so a
rebuild on a different day or host yields the same installed software:

| Build ARG | Default | What it pins |
|---|---|---|
| `ULTRALYTICS_TAG` + `ULTRALYTICS_DIGEST` | `8.4.41` + sha256 | Base image — tag is the human-readable label, digest is authoritative |
| `UV_TAG` | `0.11.7` | The `uv` binary copied from `ghcr.io/astral-sh/uv` |
| `UV_EXCLUDE_NEWER` | `2026-06-06` | PyPI snapshot cutoff — `uv pip install --exclude-newer=<date>` caps resolution to releases on or before that date so `>=`-style constraints in `pyproject.toml` don't drift. Must be ≥ the newest pinned floor's release date, so bump it alongside any dependency-floor raise |

To refresh the snapshot, override all three together:

```bash
docker compose build \
    --build-arg ULTRALYTICS_TAG=8.4.42 \
    --build-arg ULTRALYTICS_DIGEST=sha256:<new-digest> \
    --build-arg UV_TAG=0.11.8 \
    --build-arg UV_EXCLUDE_NEWER=2026-07-01
```

### Cloud notebooks (Colab / Paperspace / Kaggle)

`notebooks/visdrone_pipeline.ipynb` is an end-to-end notebook that runs
the full **tune → train → export** pipeline in a single place. It
auto-detects the runtime (Google Colab, Paperspace Gradient, Kaggle, or
local Jupyter), clones the repo if needed, installs deps from
`pyproject.toml` via `uv pip install --system`, and writes outputs to
the appropriate persistent storage location for that environment.

The defaults (one small model, ~10 epochs, a few iterations) complete
in well under an hour on a free Colab / Paperspace GPU; scale up by
editing the **Configuration** cell once the pipeline works.

> **Python version note.** The project pins `requires-python = ">=3.14"`,
> but the cloud install path intentionally bypasses that pin and
> installs the dep list into the kernel's existing Python (3.11 / 3.12)
> via `uv pip install --system`. Several ML deps (`tensorflow`,
> `onnxruntime`, `ai-edge-litert`, ...) don't yet ship Python 3.14
> wheels, so `uv sync --python 3.14` would fail on Colab/Paperspace.

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

### GPU performance & autobatching

Both `train_and_export.py` and `tune_hyperparameters.py` expose the same
four runtime knobs for maximising GPU utilisation. They are forwarded to
the underlying Ultralytics `model.train(...)` call (and to every Ray
Tune trial in the tuning script):

| Flag | Default | Purpose |
|---|---|---|
| `--batch` | `-1` | Batch size. Accepts an int N (fixed), `-1` (Ultralytics AutoBatch sized to ~60% of free GPU memory), or a float `0 < f < 1` (AutoBatch to `f` × free GPU memory, e.g. `0.85`). |
| `--cache` | `disk` | Dataset cache mode: `ram` (fastest, if dataset + augmentations fit), `disk` (preprocessed images cached to disk), or `none` (Ultralytics' default — re-reads every epoch). |
| `--workers` | `8` | Dataloader worker processes per GPU. Raise on high-core hosts; lower on RAM-constrained Colab runtimes or when running many concurrent trials per GPU. |
| `--device` | auto | CUDA device(s): `0`, `0,1` (multi-GPU triggers DDP), or `cpu`. |

In addition, both scripts call a small `enable_gpu_fast_path()` helper
on startup that flips on the two CUDA knobs Ultralytics does **not**
toggle for you:

- `torch.backends.cudnn.benchmark = True` — picks the fastest cuDNN
  conv algorithm per input shape (~5–15% speedup on fixed-resolution
  training, which we are since `multi_scale` is disabled).
- `torch.set_float32_matmul_precision("high")` — enables TF32 matmuls
  on Ampere+ GPUs (A100, L4, T4-next, RTX 30xx/40xx). Roughly 20%
  faster than full FP32 with no measurable accuracy impact at YOLO
  scale.

Mixed precision (`amp=True`) is already on by default in Ultralytics.

#### Recommended presets

```bash
# Default safe single-GPU training (the built-in defaults)
uv run python train_and_export.py

# Aggressive single-GPU push: 85% AutoBatch + RAM cache
uv run python train_and_export.py --batch 0.85 --cache ram

# Multi-GPU DDP on a 2-GPU host
uv run python train_and_export.py --device 0,1 --batch 0.85 --cache ram

# Tuning: AutoBatch + disk cache (this alone is ~2-3× faster than the
# previous default, which silently fell back to batch=16 / cache=False)
uv run python tune_hyperparameters.py

# Tuning with 2 trials per GPU – use a *fractional* batch so the two
# co-tenant trials don't both grab 60% of memory and OOM
uv run python tune_hyperparameters.py --gpu-per-trial 0.5 --batch 0.4
```

#### AutoBatch + fractional GPU sharing: the OOM rule

When you set `--gpu-per-trial < 1.0`, Ray Tune schedules multiple trials
on the same GPU simultaneously by sharing `CUDA_VISIBLE_DEVICES`. Each
co-tenant trial still sees the **full** GPU memory, so calling AutoBatch
with `--batch -1` (which targets 60% of *free* memory) means the second
trial reaches for another 60% on top of the first one's allocation and
OOMs immediately.

The `tune_hyperparameters.py` driver detects this combination and
prints a warning at trial-launch time, but it does not silently
override the user's batch — you may be running on a host with enough
memory headroom to make it work. The safe rule of thumb is:

```
--batch  ≤  0.5  ×  --gpu-per-trial
```

so e.g. `--gpu-per-trial 0.5 --batch 0.25` (each trial caps at 25% of
the GPU). The previous default of `--batch -1` is only safe when
`--gpu-per-trial >= 1.0`.

#### Cache mode trade-offs

- **`ram`** is the fastest by a wide margin once the cache is warm —
  the dataloader becomes a no-op and the GPU is fed at PCIe speed. The
  catch is that VisDrone-train (~10 GB of decoded JPEG + augmented
  tensors) needs ~25–35 GB of host RAM to cache reliably. Use on
  workstations / dedicated boxes; avoid on Colab Free tier.
- **`disk`** preprocesses each image once into a `.npy` file in
  `~/.cache/ultralytics/...` and then mmap-loads it on subsequent
  epochs. About 1.5–2× faster than uncached on the second epoch
  onwards, with no RAM cost. This is the project default and the
  right choice for Colab.
- **`none`** is Ultralytics' own default and what `tune_hyperparameters.py`
  silently used before this branch — every epoch re-decodes JPEGs from
  scratch. Avoid unless you are debugging the input pipeline.

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

## 2b. Hyperparameter Tuning (Ray Tune + Optuna + TensorBoard)

This project includes `tune_hyperparameters.py`, a thin wrapper
around Ultralytics' built-in
[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) integration
(`model.tune(use_ray=True, ...)`) that runs one **OptunaSearch** (TPE)
sweep per model with an ASHA scheduler for early stopping. YOLO26
sweeps are **warm-started** from the official
[YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
via `points_to_evaluate`, so the very first trial is guaranteed to
reproduce the published baseline and the remaining budget refines
around it.

> **Requires Ultralytics ≥ 8.4.33.**
> [PR ultralytics/ultralytics#23946](https://github.com/ultralytics/ultralytics/pull/23946)
> added `search_alg` forwarding to `run_ray_tune`, which is what lets
> us pass a pre-instantiated `OptunaSearch(points_to_evaluate=[...])`
> straight through `model.tune(use_ray=True, search_alg=...)`. Earlier
> releases hard-coded Ray's default `BasicVariantGenerator` (random
> search), so this project used to ship a full in-house `tune.Tuner(...)`
> driver as a workaround. That workaround has been removed —
> `pyproject.toml` pins `ultralytics>=8.4.33` for this reason.

### Install tuning dependencies

```bash
uv sync --extra tune      # pulls in ray[tune], optuna and tensorboard
```

### Run a tuning sweep

```bash
# Default: runs an independent sweep for ALL FOUR models
# (yolo11n, yolo11s, yolo26n, yolo26s), 10 trials × 30 epochs each,
# OptunaSearch TPE with YOLO26 recipe warm-start.
uv run python tune_hyperparameters.py

# Subset of models with more trials and 1 GPU per trial
uv run python tune_hyperparameters.py \
    --models yolo11s yolo26s \
    --iterations 30 \
    --epochs 50 \
    --gpu-per-trial 1

# Fall back to plain random search (no optuna dependency needed)
uv run python tune_hyperparameters.py --search-algo random
```

Key flags (see `--help` for the full list):

| Flag | Default | Description |
|---|---|---|
| `--models` | all four | Subset of `yolo11n yolo11s yolo26n yolo26s` to tune |
| `--iterations` | `10` | Number of hyperparameter samples **per model** |
| `--epochs` | `30` | Max epochs per trial (ASHA prunes bad trials early) |
| `--grace-period` | `10` | ASHA grace period before pruning is allowed |
| `--gpu-per-trial` | auto | Fractional GPUs per trial (e.g. `0.5` → 2 trials/GPU) |
| `--batch` | `-1` | Per-trial AutoBatch / fixed batch — see [GPU performance & autobatching](#gpu-performance--autobatching) |
| `--cache` | `disk` | Per-trial dataset cache mode (`ram` / `disk` / `none`) |
| `--workers` | `8` | Dataloader workers per trial |
| `--device` | auto | CUDA device(s) per trial |
| `--search-algo` | `optuna` | `optuna` (TPE + YOLO26 warm-start) or `random` |

> **Speedup note.** `batch=-1` (AutoBatch) and `cache=disk` are the
> project defaults for tuning, which deliver a 2-3× wall-clock
> speedup per trial over the vanilla Ultralytics defaults
> (`batch=16`, `cache=False`). The knobs are forwarded into every
> Ray Tune trial via `model.tune(..., batch=..., cache=..., workers=...)`
> — Ultralytics' `run_ray_tune` merges them into each trial's
> `model.train(**config)` via `config.update(train_args)`. See the
> [GPU performance & autobatching](#gpu-performance--autobatching)
> section above for the rationale and the OOM-safety rules when
> sharing one GPU across multiple concurrent trials.

Internally the script dispatches into Ultralytics' built-in Ray Tune
integration per model:

```python
YOLO(weights).tune(
    use_ray=True,
    space=build_search_space(model_key),          # VisDrone + family-tuned
    search_alg=OptunaSearch(                      # pre-instantiated so we
        metric="metrics/mAP50-95(B)", mode="max", # can attach the recipe
        points_to_evaluate=<YOLO26 recipe for     # seed — Ultralytics'
            yolo26n/yolo26s, else None>,          # string resolver does
    ),                                            # not expose this.
    iterations=args.iterations,
    grace_period=args.grace_period,
    gpu_per_trial=args.gpu_per_trial,
    data=..., epochs=..., imgsz=..., batch=..., cache=..., workers=...,
)
```

Ultralytics builds the `ASHAScheduler` (time_attr=epoch,
metric=`metrics/mAP50-95(B)`, reduction_factor=3) internally and wires
per-epoch metric reporting through its own training callback, so we
do not need a custom trainable or a manual `tune.Tuner`. The returned
`ResultGrid` is unpacked into a per-model summary.

Each model's best trial is written to
`runs/tune/visdrone_raytune_<model>_best_hyperparameters.json` and a
summary table of per-model mAP is printed at the end.

### Why OptunaSearch (TPE)?

With our constraint mix — small budget (10 trials/model), expensive
GPU trials, ASHA-pruned, ~15-20 mixed continuous/integer parameters —
TPE is the most practical Ray Tune search algorithm:

- Handles `loguniform` / `uniform` / `randint` natively.
- Builds a useful surrogate after ~5 trials, so even a tiny budget
  learns from prior results instead of sampling uniformly.
- Composes cleanly with `ASHAScheduler` (unlike BOHB / BlendSearch /
  CFO, which bundle their own early-stopping and would conflict).
- Supports `points_to_evaluate`, which is what enables the YOLO26
  recipe warm-start.

The alternatives (`HyperOptSearch`, `AxSearch` / BoTorch, `HEBOSearch`,
`BayesOptSearch`) are all viable but either drop mixed-type support,
add heavy dependencies, or offer marginal benefits at n ≤ 10 trials.
Random search (`--search-algo random`) is retained as a dependency-
free fallback.

### VisDrone hyperparameter audit

The VisDrone-focused search spaces in `tune_hyperparameters.py`
`build_search_space()` are deliberately narrowed subsets of the
Ultralytics default ([`ultralytics/utils/tuner.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tuner.py)
`run_ray_tune`), audited against the Ultralytics community thread on
[training YOLO11/YOLO12 on VisDrone](https://community.ultralytics.com/t/standard-epochs-and-imgsz-for-training-yolo11-yolov12-on-visdrone-dataset/1614)
and — for YOLO26n / YOLO26s — the official
[YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/).
`build_search_space(model_key)` dispatches on the model family so each
variant gets a recipe-aware space:

- **YOLO11n / YOLO11s** → VisDrone-audited YOLO11 space.
- **YOLO26n** → DFL-heavy space (`dfl` 6.0 – 12.0, recipe anchor 9.04),
  aggressive `scale` / `shear` permitted by STAL label assignment.
- **YOLO26s** → box-heavy space (`box` 7.0 – 13.0, recipe anchor 9.83),
  de-emphasised `dfl` (0.5 – 2.0), `lr0` one order of magnitude lower
  than YOLO26n, and `degrees` / `shear` pinned near zero per the recipe.

Notable VisDrone-specific decisions in the YOLO11 space:

- **`flipud` is included** (`0.0 – 0.5`). Aerial imagery has no natural
  vertical orientation, so vertical flip roughly doubles the effective
  dataset with zero label cost.
- **`fliplr` uses the full `0.0 – 1.0` range** so the tuner can settle
  on the standard `0.5` for laterally symmetric classes (car, bus,
  person, ...). An earlier 0 – 0.5 cap excluded the best value.
- **`close_mosaic` is tuned** (`randint(5, 15)`). The mosaic-shutdown
  window is one of the strongest levers for small-object detection —
  mosaic distortion hurts the final fine-tuning epochs.
- **`cutmix` and `copy_paste` are both enabled** with conservative
  upper bounds (0.3 each). Both help VisDrone's many-small-objects
  regime without destroying tiny bounding boxes.
- **`box` gain is pushed higher** (`5.0 – 12.0` vs. default 7.5) to
  emphasise localisation accuracy on tiny boxes.
- **`shear` and `perspective` are deliberately omitted** — both
  destroy small-object bboxes via pixel interpolation loss.
- **`multi_scale` is intentionally disabled** in `DEFAULT_TRAIN_ARGS`.
  The Ultralytics multi-scale path calls
  `nn.functional.interpolate(imgs, size=ns, mode="bilinear", ...)`
  inside `DetectionTrainer.preprocess_batch`, and on the PyTorch 2.x
  builds shipped with Google Colab the upsample decomposition crashes
  inside `_compute_scale` with `ZeroDivisionError: division by zero`
  (the bug is upstream in PyTorch). The small-object benefit is
  largely recovered through the `scale`, `mosaic` and `copy_paste`
  ranges already covered by the Ray Tune search space. Re-enable
  once Colab ships a PyTorch with the decomposition fix.

### YOLO26 training recipe integration

YOLO26 ships with an official training recipe that differs sharply
from YOLO11 defaults — new MuSGD optimiser, end-to-end NMS-free head,
Small-Target-Aware Label Assignment (STAL), and distinct per-size
hyperparameters. `train_and_export.py` layers a
`MODEL_TRAIN_OVERRIDES` table on top of `DEFAULT_TRAIN_ARGS` so the
YOLO11 VisDrone baseline is preserved for YOLO11n / YOLO11s while
YOLO26n and YOLO26s get their recipe values applied verbatim:

| Hyperparameter | YOLO26n recipe | YOLO26s recipe |
|----------------|---------------:|---------------:|
| `lr0`          | 0.0054         | 0.00038        |
| `lrf`          | 0.0495         | 0.882          |
| `momentum`     | 0.947          | 0.948          |
| `weight_decay` | 0.00064        | 0.00027        |
| `box`          | 5.63           | 9.83           |
| `cls`          | 0.56           | 0.65           |
| `dfl`          | 9.04           | 0.96           |
| `mosaic`       | 0.909          | 0.992          |
| `mixup`        | 0.012          | 0.05           |
| `copy_paste`   | 0.075          | 0.304          |
| `scale`        | 0.562          | 0.9            |
| `degrees`      | 1.11           | 0.0            |
| `shear`        | 1.46           | 0.0            |
| `fliplr`       | 0.606          | 0.304          |
| `close_mosaic` | 10             | 10             |

Note the sharp split between the nano and small variants: YOLO26n
prioritises **DFL** (`dfl=9.04`) with a relatively high `lr0` and
aggressive geometry, while YOLO26s prioritises **box regression**
(`box=9.83`) with a much lower `lr0`, near-unity `lrf` (gentle LR
decay) and zeroed rotation / shear. The Ray Tune search spaces in
`_visdrone_yolo26n_space()` and `_visdrone_yolo26s_space()` bracket
these recipe anchors so the tuner can refine around — not regress
away from — the published values.

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
