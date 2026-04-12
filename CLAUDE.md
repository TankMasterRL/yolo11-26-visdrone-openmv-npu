# CLAUDE.md

Guidance for Claude Code (and other LLM agents) working in this repo.

## What this project is

End-to-end pipeline that **trains** YOLO11n / YOLO11s / YOLO26n / YOLO26s
on the VisDrone aerial-imagery dataset, **exports** them to INT8 TFLite,
**compiles** NPU-optimised binaries, and ships ready-to-run **OpenMV
MicroPython** scripts for region-based object counting on two boards:

- **OpenMV AE3** — Alif Ensemble E3 / dual Ethos-U55 NPU, compiled via
  **Arm Vela** → `*_vela.tflite`.
- **OpenMV N6** — STM32N6 / ST Neural-ART NPU, compiled via **STEdgeAI**
  → `*_int8.tflite` (firmware auto-routes ops to the NPU).

It is a **script collection**, not an installable package
(`pyproject.toml` sets `[tool.uv] package = false`). Treat
`train_and_export.py` and `tune_hyperparameters.py` as the two real
entry points; everything else supports them.

## Repo layout

```
.
├── train_and_export.py        # Train + INT8 TFLite export + Vela / STEdgeAI compile
├── tune_hyperparameters.py    # In-house Ray Tune (Optuna TPE + ASHA) sweep
├── pyproject.toml             # Single source of truth for deps + extras
├── docker/Dockerfile          # GPU image: ultralytics/ultralytics + uv layer
├── docker-compose.yml         # Services: train / tune / export / tensorboard / shell
├── notebooks/visdrone_pipeline.ipynb   # End-to-end Colab/Paperspace notebook
├── openmv-scripts/
│   ├── region_counter.py      # Shared MicroPython region-counting module
│   ├── ae3/main_yolo*.py      # Per-model AE3 entry points
│   └── n6/main_yolo*.py       # Per-model N6 entry points
└── README.md                  # Long-form user docs (architecture + recipes)
```

There is **no test suite, linter, formatter, or pre-commit hook** in
this repo. Don't fabricate one. Validation = `python -c "import ast; ..."`
syntax checks and `--help` smoke tests on the two CLIs.

## Toolchain

- **Python 3.14** (`.python-version`).
- **uv** for dependency management. Optional-dependency groups defined
  in `pyproject.toml`:
  - `ae3` → `ethos-u-vela` (AE3 NPU compiler)
  - `tune` → `ray[tune]`, `optuna`, `tensorboard`
  - `export` → full TFLite INT8 chain (`tensorflow`, `tf_keras`, `onnx`,
    `onnx2tf`, `onnxslim`, `onnxruntime`, `ai-edge-litert`, `protobuf`, ...)
  - `all` → everything pip-installable
- **STEdgeAI** is proprietary (ST), download separately and bind-mount
  at `/opt/stedgeai` for the Docker `export` service.

## Common commands

```bash
# Install deps (host)
uv sync                       # base only (ultralytics)
uv sync --extra all           # everything pip-installable

# Train + export + NPU compile (all four models)
uv run python train_and_export.py --epochs 100 --imgsz 640

# Single model, custom export size
uv run python train_and_export.py --models yolo26n --imgsz-export 192

# Hyperparameter sweep (Optuna TPE + ASHA, all four models)
uv run python tune_hyperparameters.py

# Tuning with aggressive GPU usage (85% AutoBatch + RAM cache)
uv run python tune_hyperparameters.py --batch 0.85 --cache ram

# Two trials per GPU – fractional batch is REQUIRED to avoid OOM
uv run python tune_hyperparameters.py --gpu-per-trial 0.5 --batch 0.4

# TensorBoard for all training runs and tuning trials
uv run tensorboard --logdir runs

# Docker (GPU host with NVIDIA Container Toolkit)
docker compose build
docker compose run --rm train
docker compose run --rm tune
docker compose run --rm export       # skip-train: re-uses runs/visdrone/*/best.pt
docker compose up tensorboard        # http://localhost:6006
```

The Docker image uses the **same `pyproject.toml`** the host uses (via
`uv pip install --system -r pyproject.toml --extra all`), so host and
container dependency resolution stay byte-for-byte aligned. If you add
a dependency, add it to `pyproject.toml` only — never to the Dockerfile
directly.

## Architecture notes

### `train_and_export.py`
- `DEFAULT_TRAIN_ARGS` — VisDrone-audited YOLO11 baseline. **Do not
  re-enable `multi_scale=True`** — see "Gotchas" below.
- `MODEL_TRAIN_OVERRIDES` — official YOLO26 training recipe values
  applied verbatim for `yolo26n` / `yolo26s` (MuSGD-tuned `lr0` / `lrf`,
  recipe-specific `box`/`cls`/`dfl` gains, geometric augmentation per
  size). Layered on top of `DEFAULT_TRAIN_ARGS` inside `train_model`.
- GPU runtime knobs (`--batch` / `--cache` / `--workers` / `--device`)
  are CLI-overridable. `parse_batch` handles Ultralytics' three-mode
  syntax: int (fixed), `-1` (AutoBatch ~60% mem), `0 < f < 1` (AutoBatch
  to f% mem). `enable_gpu_fast_path()` flips on cuDNN benchmark + TF32.
- Per-model export sizes default to 256 (nano) / 320 (small) to fit
  the AE3/N6 NPU memory budgets — tune with `--imgsz-export`.

### `tune_hyperparameters.py`
- Drives Ultralytics' built-in `model.tune(use_ray=True, ...)`
  integration. Ultralytics ≥ **8.4.33** (PR
  ultralytics/ultralytics#23946) added `search_alg` forwarding to
  `run_ray_tune`, which is what lets us plug in `OptunaSearch`; the
  repo used to ship a full in-house `tune.Tuner(...)` driver as a
  workaround for that missing forwarding — it has been removed.
  **Do not reintroduce it.** `pyproject.toml` pins `ultralytics>=8.4.33`
  for this reason.
- Ultralytics internally builds the `ASHAScheduler` (time_attr=epoch,
  metric=`metrics/mAP50-95(B)`, reduction_factor=3), constructs the
  Ray `RunConfig`, and wires per-epoch metric reporting via its own
  training callback. We only assemble the search space and search
  algorithm on top.
- **Search spaces** (`build_search_space`) dispatch by model family:
  YOLO11n/s share one space; `yolo26n` and `yolo26s` get distinct
  spaces bracketed around the recipe anchors.
- **Warm-start** (`_recipe_seed_for`) injects the YOLO26 recipe values
  from `train_and_export.MODEL_TRAIN_OVERRIDES` as Optuna's
  `points_to_evaluate`, so trial #0 of every YOLO26 sweep is the
  published baseline and the rest of the budget refines around it.
  This is why YOLO26 always gets a **pre-instantiated** `OptunaSearch`
  rather than the string `"optuna"` — Ultralytics' string resolver
  does not expose `points_to_evaluate`.
- `batch` / `cache` / `workers` / `amp=True` flow through
  `model.tune(..., **train_kwargs)` into each trial's
  `model.train(**config)` via Ultralytics' `config.update(train_args)`.
  Without those knobs trials silently fall back to the vanilla
  `batch=16` / `cache=False` defaults and burn ~50% of throughput.

### `openmv-scripts/`
- `region_counter.py` is a small MicroPython module — pure stdlib +
  `micropython`/`image`. Don't import host-side libs into it.
- The `main_yolo*.py` scripts are board-specific entry points that
  `import region_counter` and run the corresponding `*_int8.tflite` /
  `*_vela.tflite` model from on-board flash or SD. Keep them small —
  the OpenMV firmware has tight RAM budgets.

## Branching & commit conventions

- Develop on a `claude/<topic>-<suffix>` branch (e.g.
  `claude/gpu-utilization-speedup`).
- Branches are **fast-forward merged** into `main`
  (`git checkout main && git merge <branch>`). This keeps the history
  linear and bisectable. If the branch is behind `main`, rebase it
  first (`git rebase main`) rather than creating a merge commit.
- **Pull requests are squash-merged.** When a PR lands via GitHub, all
  commits on the branch are squashed into a single commit on `main`.
  Write the PR title as if it were the final commit message (imperative,
  concise) and use the PR body for detail.
- Commit messages: imperative, single-line subject, optional body
  explaining *why*. No `Co-Authored-By: Claude` trailers, no Anthropic
  attribution lines — keep them clean.
- **Never** push directly to `main` from a working branch — always
  merge via the local `git checkout main && git merge <branch>` flow
  or open a pull request.

## Gotchas (read these before debugging)

1. **`multi_scale=True` crashes on Colab.** Ultralytics'
   `DetectionTrainer.preprocess_batch` calls
   `nn.functional.interpolate(... mode="bilinear")`, and on the
   PyTorch 2.x builds shipped with Google Colab, AMP autocast routes
   that through `torch._decomp.decompositions._compute_scale`, which
   crashes with `ZeroDivisionError: division by zero` because
   `out_size` comes through as 0. The fix is `multi_scale=False` in
   `DEFAULT_TRAIN_ARGS`. The community small-object benefit is
   recovered via the `scale` / `mosaic` / `copy_paste` ranges in the
   tuning search space. Re-enable only when Colab ships a PyTorch
   with the upstream decomposition fix.

2. **Ray 2.x version drift lives in Ultralytics now.** The Colab Ray
   build moves between v1 and v2 of the Train API roughly every
   release (RunConfig `verbose` bugs, `ray.tune.report` vs
   `ray.train.report` deprecations, `get_context` namespace shifts).
   Our code used to carry a stack of `try/except` shims for those,
   but since we now dispatch through `model.tune(use_ray=True, ...)`
   those shims live inside Ultralytics' `run_ray_tune` — if Colab
   ships a Ray version that breaks tuning, the fix is to bump
   `ultralytics>=<next-version>` in `pyproject.toml`, not to
   reintroduce a local driver.

3. **AutoBatch + fractional GPU sharing OOMs.** `--batch -1` targets
   60% of free GPU memory. With `--gpu-per-trial 0.5`, two co-tenant
   trials each see the full GPU and both reach for 60% → OOM. The
   safe rule is `--batch ≤ 0.5 × --gpu-per-trial`. The tuner already
   warns at trial-launch time but does not silently override.

4. **Python 3.14.** The `.python-version` is intentional. Some older
   wheels (notably parts of the TFLite export chain) may not yet ship
   3.14 builds — if you hit a wheel-not-found error in CI, the fix is
   to wait for the upstream wheel, not to downgrade Python.

5. **`runs/`, `export/`, `*.pt`, `*.tflite`, `*.onnx` are gitignored.**
   Don't commit training artefacts or weights.

## When in doubt

- The README is the long-form user-facing documentation; **this file
  is for agents**. Don't duplicate the README into CLAUDE.md — link
  to its sections by anchor instead (e.g. "see README §GPU performance
  & autobatching").
- If a Colab error trace lands in your lap with no obvious local
  cause, suspect **Ray version drift inside Ultralytics**
  (bump `ultralytics>=...` in `pyproject.toml`) or the **PyTorch
  upsample decomposition** before suspecting our code.
- Prefer small, focused branches. The git history shows the cadence:
  one bug = one branch = one commit on `main`.
