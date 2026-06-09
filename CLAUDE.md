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
  explaining *why*. The human submitter is responsible for reviewing all AI-generated code, compliance, and taking responsibility for the contribution. Only humans can use "Signed-off", and the AI tools must be reported with the "Assisted-by" tag. e.g: Assisted-by: Claude:claude-3-opus coccinelle sparse
  Don't include any references to session links from agentic coding tools
  in commit messages **or PR descriptions** — e.g. https://claude.ai/code/
- **Never** push directly to `main` from a working branch — always
  merge via the local `git checkout main && git merge <branch>` flow
  or open a pull request.

## PR workflow

These are the actions to perform when shepherding a PR end-to-end.
Follow them in order; skip steps only when the user explicitly
declines them.

1. **Branch.** Start from an up-to-date `main`
   (`git checkout main && git pull origin main`), then create the
   `claude/<topic>-<suffix>` branch the task specifies.
2. **Implement + commit.** Keep changes focused. Before committing,
   run the repo's validation: `python -c "import ast;
   ast.parse(open('<changed_file>').read())"` syntax-checks on any
   edited Python files, plus `uv run python train_and_export.py --help`
   and `uv run python tune_hyperparameters.py --help` smoke tests when
   either CLI's surface changed. There is no test suite, linter, or
   formatter — don't fabricate one. Use the commit-message conventions
   above, including the `Assisted-by:` trailer.
3. **Push.** `git push -u origin <branch>`. Retry with exponential
   backoff on network failures only.
4. **Open the PR only when asked.** Use a short imperative title and
   a body with a `## Summary` section and a `## Test plan` checklist.
   The checklist should enumerate what *must* be true for the change
   to ship (green CI, manual verification steps, cache/artefact paths,
   etc.). **The agentic harness may auto-append a
   `_Generated by [Claude Code](...)_` footer to the PR body —
   always strip it immediately after creation via `update_pull_request`,
   as it violates the no-session-links rule above.**
5. **Subscribe to PR activity automatically** via
   `subscribe_pr_activity` immediately after the PR is opened — don't
   wait for the user to ask. Investigate CI failures and review
   comments; make small fixes directly, ask the user when ambiguous.
   The PR-merged webhook unsubscribes on its own.
6. **Keep the test plan up to date continuously.** Update the PR body
   via `update_pull_request` after *every* event that changes the
   status of a checklist item — don't batch updates to the end.
   - **CI check passes:** tick the box and annotate inline, e.g.
     `- [x] Docker image builds — green on \`abc1234\` in 2m44s`
   - **CI check fails:** leave the box unticked and append the failure
     summary inline, e.g.
     `- [ ] Docker image builds — FAILED on \`abc1234\`: <one-line reason>`,
     then investigate and push a fix.
   - **New commit pushed to the branch:** review every previously-ticked
     item; if the commit touches code that the item covers, un-tick it
     and annotate with `(re-opened by \`<sha>\`)` so the reviewer knows
     it needs re-verification on the new head.
   - **Manual check completed:** tick the box and note how it was
     verified, e.g.
     `- [x] \`--help\` smoke test — verified locally`
   - **Item not exercised in this PR:** mark it
     `- [ ] <item> — N/A: <short reason>` rather than deleting it, so
     the scope of the PR remains visible to reviewers.
7. **Merge only on explicit request.** PRs are squash-merged via
   `merge_pull_request` with `merge_method: squash`. Write the squash
   commit title as the final imperative subject; put the body detail
   in the squash commit message. The `Assisted-by:` trailer must
   appear in the squash message.
8. **Clean up.** Delete any local worktree branches that are no
   longer needed after the squash-merge lands.

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

6. **TFLite export must run in a GPU-free subprocess — TensorFlow
   crashes on too-new GPUs.** The `onnx2tf` step of `export_tflite_int8`
   imports TensorFlow, which grabs the first visible CUDA device. If the
   GPU's compute capability is newer than the kernels bundled with the
   installed TF build (e.g. an RTX 5090 / Blackwell `sm_120` under
   TF 2.19), TF JIT-compiles from PTX and the kernel launch dies with
   `CUDA_ERROR_INVALID_HANDLE` on `[Op:Cast]`. Setting
   `CUDA_VISIBLE_DEVICES=-1` **in-process does not help**: the CUDA driver
   only reads it at the first `cuInit()`, and training has already
   initialised CUDA via PyTorch by then, so the change is ignored by
   every library in the process (TF included). The fix is
   `run_export_isolated`, which runs the export in a fresh subprocess with
   `CUDA_VISIBLE_DEVICES=-1` set *before* the interpreter starts — that
   child's `cuInit()` honours it, hiding the GPU from both PyTorch and TF
   so the (CPU-appropriate) conversion runs on the CPU while the parent
   training loop keeps the GPU. Do **not** "fix" this by bumping
   TensorFlow, downgrading the GPU, or moving the export back in-process —
   the export never needed the GPU.

7. **`model.export(int8=True)` returns a *dynamic-range* model, not a
   full-integer one — Vela rejects it.** Ultralytics builds its
   `*_int8.tflite` by **renaming onnx2tf's `*_dynamic_range_quant.tflite`**
   (see `ultralytics/utils/export/tensorflow.py`): INT8 *weights* but
   **FLOAT32 feature maps and FLOAT32 I/O** (Ultralytics' own code comments
   it `# fp32 in/out`). Hand that to Arm Vela and every operator fails the
   supported-operator check with `Reason: Operation has tensor with
   unsupported DataType Float32` → `CPU operators = 100%`, `NPU operators =
   0`, `0 MACs`. The file size is a trap: it's ~INT8-weight-sized (e.g.
   2.7 MB for YOLO11n), so it *looks* quantised. The genuinely deployable
   models are onnx2tf's siblings in the same `*_saved_model/` dir:
   `*_full_integer_quant.tflite` (INT8 weights **+ activations + int8 I/O**;
   correct for Ethos-U55 / Neural-ART) and `*_integer_quant.tflite` (INT8
   core, FP32 I/O). `export_tflite_int8` calls `_select_npu_tflite` to pick
   `*_full_integer_quant.tflite` (then `*_integer_quant.tflite`) instead of
   the returned `*_int8.tflite`. Do **not** "simplify" it back to copying
   `model.export(...)`'s return value — that silently reintroduces the
   all-CPU model. The OpenMV `ml`/`tf` runtime reads quant params from the
   model, so INT8 I/O is transparent on-device (and is what the NPU path
   wants); no `openmv-scripts/` change is needed.

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
