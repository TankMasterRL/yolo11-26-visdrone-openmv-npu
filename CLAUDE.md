# CLAUDE.md

Guidance for Claude Code (and other LLM agents) working in this repo.

## What this project is

End-to-end pipeline that **trains** YOLO11n / YOLO11s / YOLO26n / YOLO26s
on the VisDrone aerial-imagery dataset, **exports** them to INT8 TFLite
(plus FP32 ONNX as a shared intermediate), **compiles** NPU-optimised
binaries, and ships ready-to-run scripts for region-based object
counting on four hardware targets:

- **OpenMV AE3** — Alif Ensemble E3 / dual Ethos-U55 NPU, compiled via
  **Arm Vela** → `*_vela.tflite`. MicroPython, on-device firmware.
- **OpenMV N6** — STM32N6 / ST Neural-ART NPU, compiled via **STEdgeAI**
  → `*_int8.tflite` (firmware auto-routes ops to the NPU). MicroPython,
  on-device firmware.
- **Luxonis OAK** (RVC2 / Myriad-X) — `.blob` / `.superblob` via the
  upstream `luxonis/modelconverter-rvc2` Docker image (built locally).
  Host Python + DepthAI v3 peripheral mode.
- **Luxonis OAK4** (RVC4 / Qualcomm) — `.dlc` via
  `luxonis/modelconverter-rvc4` (built locally; SNPE SDK is
  licence-gated). Host Python + DepthAI v3 peripheral mode.

It is a **script collection**, not an installable package
(`pyproject.toml` sets `[tool.uv] package = false`). Treat
`train_and_export.py` and `tune_hyperparameters.py` as the two real
entry points; everything else supports them.

## Repo layout

```
.
├── train_and_export.py        # Train + INT8 TFLite + ONNX + Vela / STEdgeAI / ModelConverter
├── tune_hyperparameters.py    # In-house Ray Tune (Optuna TPE + ASHA) sweep
├── pyproject.toml             # Single source of truth for deps + extras
├── docker/Dockerfile          # GPU image: ultralytics/ultralytics + uv layer
├── docker/oak/
│   ├── build.sh               # Builds luxonis/modelconverter-{rvc2,rvc4}:local from upstream
│   └── extra_packages/        # Drop-in for licence-gated OpenVINO + SNPE archives (gitignored)
├── docker-compose.yml         # Services: train / tune / export / oak-export / tensorboard / shell
├── notebooks/visdrone_pipeline.ipynb   # End-to-end Colab/Paperspace notebook
├── openmv-scripts/            # On-device MicroPython (AE3 + N6)
│   ├── region_counter.py      # Shared MicroPython region-counting module
│   ├── ae3/main_yolo*.py      # Per-model AE3 entry points
│   └── n6/main_yolo*.py       # Per-model N6 entry points
├── oak-scripts/               # Host Python (DepthAI v3 peripheral mode for OAK + OAK4)
│   ├── region_counter.py      # NumPy/OpenCV port of the MicroPython module (host-only)
│   ├── _pipeline.py           # Shared DepthAI v3 pipeline builder + run loop
│   └── main_yolo*.py          # Per-model thin entry points
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
  - `oak` → `depthai>=3`, `opencv-python`, `numpy` (host runtime for the
    DepthAI v3 peripheral-mode scripts under `oak-scripts/`)
  - `all` → everything pip-installable
- **STEdgeAI** is proprietary (ST), download separately and bind-mount
  at `/opt/stedgeai` for the Docker `export` service.
- **OAK / OAK4 conversion toolchain** is intentionally NOT pip-installable.
  It runs as a sibling Docker container (`luxonis/modelconverter-rvc2:local`
  / `-rvc4:local`) built from upstream sources by `docker/oak/build.sh`
  using user-supplied OpenVINO + Qualcomm SNPE archives. See
  `docker/oak/extra_packages/README.md`.

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

# OAK / OAK4: build the converter images locally from upstream sources
# (drop OpenVINO + SNPE archives into docker/oak/extra_packages/ first)
docker/oak/build.sh                  # both rvc2 and rvc4
docker/oak/build.sh rvc4 --no-cache  # rebuild RVC4 only

# Compile every model for both OAK targets (skips train + Vela/STEdgeAI)
uv run python train_and_export.py --skip-train --skip-npu --oak-target both

# Run the host-side DepthAI v3 region-counter (peripheral mode)
uv sync --extra oak
python oak-scripts/main_yolo11n.py
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
- **Compile-step dispatch.** Each per-model loop iteration produces
  TFLite INT8, then optionally compiles for AE3 (Vela), N6 (STEdgeAI),
  and OAK / OAK4. The OAK path goes via `export_onnx()` (FP32 ONNX with
  `nms=False`, opset 12, static shapes — Ultralytics writes it once and
  both RVC2 and RVC4 reuse the same artefact) and `compile_modelconverter()`
  (Docker-driven; auto-generates a per-target YAML or honours
  `--oak-config`). Skip flags: `--skip-npu` (Vela + STEdgeAI),
  `--skip-oak` (ONNX + ModelConverter). Targets: `--oak-target {rvc2,rvc4,both}`.

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

### `oak-scripts/`
- **Host Python**, not MicroPython — separate sibling directory to
  `openmv-scripts/` because the runtime is structurally different
  (DepthAI v3 + OpenCV + NumPy, not on-device firmware). Don't try to
  share `region_counter.py` between the two — `oak-scripts/region_counter.py`
  is a deliberate NumPy/OpenCV port with the same public API.
- `_pipeline.py` builds the pipeline once and is shared by every
  `main_yolo*.py`. Topology: `Camera.requestOutput(BGR888p, LETTERBOX)`
  → NN node → host queues. The `LETTERBOX` resize happens at the ISP
  (no extra `ImageManip` hop), matching OpenMV's auto-letterbox.
- **YOLO family dispatch.** `family="yolo11"` uses `dai.node.YoloDetectionNetwork`
  for on-device decode + NMS (highest FPS). `family="yolo26"` uses
  `dai.node.NeuralNetwork` plus a small NumPy parser, because YOLO26's
  end-to-end head emits already-final boxes and re-running NMS would
  corrupt them.
- The `main_yolo*.py` scripts are 30-line entry points that import
  `cli_parser` + `run` from `_pipeline`. Keep them as thin shims —
  add new model families by extending `_pipeline.build_pipeline`.
- The build artefacts (`*.blob`, `*.superblob`, `*.dlc`) are gitignored
  alongside `*.tflite` / `*.onnx`. Don't commit them.

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
  Don't include any references to session links from agentic coding tools, e.g. https://claude.ai/code/
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
   etc.).
5. **Subscribe to PR activity automatically** via
   `subscribe_pr_activity` immediately after the PR is opened — don't
   wait for the user to ask. Investigate CI failures and review
   comments; make small fixes directly, ask the user when ambiguous.
   The PR-merged webhook unsubscribes on its own.
6. **Keep the test plan up to date.** As CI runs land and manual
   checks complete, tick the corresponding boxes by editing the PR
   body via `update_pull_request`. Annotate ticked items with the
   commit SHA and the CI duration (e.g. "green on `abc1234` in 2m44s").
   Mark items that aren't exercised in this PR as unticked with a
   short reason, rather than deleting them.
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

5. **`runs/`, `export/`, `*.pt`, `*.tflite`, `*.onnx`, `*.blob`,
   `*.superblob`, `*.dlc`, `docker/oak/build/`, and
   `docker/oak/extra_packages/*.{tar.gz,tgz,zip}` are gitignored.**
   Don't commit training artefacts, weights, or licence-gated SDK
   archives.

6. **OpenVINO archive name vs contents.** `docker/oak/build.sh` defaults
   `OPENVINO_VERSION=2022.3.0` and expects `openvino-2022.3.0.tar.gz` in
   `docker/oak/extra_packages/`, but the README points users at the
   **2022.3.2 patch release** download. This is intentional: the upstream
   `luxonis/modelconverter` RVC2 Dockerfile has three hard-coded
   conditionals that only fire on `VERSION=2022.3.0` (archive
   strip-components count, `/opt/intel/tools/*.whl` cleanup, and the
   `convert_impl.py` patch selection). Anything else drops into the
   2021.4.0 legacy branch and fails. 2022.3.2 is a drop-in patch with
   the same archive layout and `/opt/intel/...` paths, so staging it
   under the `2022.3.0` filename + arg gives users the newer binaries
   while keeping the build steps that work. **Do not rename it to
   `openvino-2022.3.2.tar.gz`** unless you also patch the upstream
   Dockerfile's conditionals.

7. **OAK / OAK4 conversion is `linux/amd64` only.** Both upstream
   Dockerfiles `COPY` files from `/usr/lib/x86_64-linux-gnu/` in their
   second stage, and the OpenVINO + SNPE SDKs ship x86_64 binaries
   only. `docker/oak/build.sh` and the `docker run` in
   `compile_modelconverter` both pin `--platform linux/amd64`; on ARM
   hosts (Apple Silicon, ARM Linux) the build is correct but slow
   (QEMU emulation). If you ever need to disable the platform pin,
   override `BUILD_PLATFORM` in the script — but expect the build to
   fail on non-x86_64 hosts without it.

8. **YOLO11 vs YOLO26 in DepthAI v3.** YOLO11 ships an anchor-free
   head that expects an external NMS — drive it with
   `dai.node.YoloDetectionNetwork` for on-device decode + NMS. YOLO26
   ships a built-in NMS-free end-to-end head; the model emits already
   final boxes, so re-running NMS via `YoloDetectionNetwork` would
   corrupt them. Use `dai.node.NeuralNetwork` + the small NumPy parser
   in `oak-scripts/_pipeline.py` (`_parse_yolo26`). The dispatch lives
   in `build_pipeline(family=...)`.

9. **OAK (RVC2) + small @ 320 may exceed Myriad-X SHAVE memory.**
   `compile_modelconverter` warns when `target=="rvc2"` and the model
   is `*s` and `imgsz > 288`, then continues. If the converter exits
   non-zero with an OOM/SHAVE allocation error, retry with
   `--imgsz-export 256`. RVC4 has no such limit.

## When in doubt

- The README is the long-form user-facing documentation; **this file
  is for agents**. Don't duplicate the README into CLAUDE.md — link
  to its sections by anchor instead (e.g. "see README §GPU performance
  & autobatching").
- If a Colab error trace lands in your lap with no obvious local
  cause, suspect **Ray version drift inside Ultralytics**
  (bump `ultralytics>=...` in `pyproject.toml`) or the **PyTorch
  upsample decomposition** before suspecting our code.
- If the OAK / OAK4 `docker build` fails with "no such file or
  directory" pointing at `/usr/lib/x86_64-linux-gnu/...`, you're
  building on a non-x86_64 host without QEMU set up. See gotcha #7.
  If it fails on the OpenVINO `rm -r` or `patch` step, you renamed
  the archive away from `openvino-2022.3.0.tar.gz` — see gotcha #6.
- For the upstream `luxonis/modelconverter` Dockerfiles: prefer
  cloning their repo as a build context (what `docker/oak/build.sh`
  does) over forking our own Dockerfiles. The upstream owns the
  RVC2/RVC4 build recipes; we just call them. If a Luxonis update
  breaks our defaults, bump `MODELCONVERTER_REF` in `build.sh`.
- Prefer small, focused branches. The git history shows the cadence:
  one bug = one branch = one commit on `main`.
