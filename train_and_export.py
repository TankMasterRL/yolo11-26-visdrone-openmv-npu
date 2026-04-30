#!/usr/bin/env python3
"""
YOLO Object Detection Training Pipeline for OpenMV AE3 / N6 + Luxonis OAK / OAK4
=================================================================================

Trains YOLO11n, YOLO11s, YOLO26n, YOLO26s on VisDrone dataset, exports
INT8 TFLite + ONNX, then compiles NPU-optimised binaries for:

  - OpenMV AE3   (Alif Ensemble E3 / Ethos-U55)  → Vela-compiled TFLite
  - OpenMV N6    (STM32N6 / Neural-ART NPU)       → stedgeai network binary
  - Luxonis OAK  (RVC2 / Myriad-X)               → .blob via ModelConverter (Docker)
  - Luxonis OAK4 (RVC4 / Qualcomm)               → .dlc  via ModelConverter (Docker)

Usage:
    python train_and_export.py                       # full pipeline
    python train_and_export.py --skip-train          # export only (re-uses best.pt)
    python train_and_export.py --models yolo11n      # single model
    python train_and_export.py --imgsz 256           # custom input size
    python train_and_export.py --skip-oak            # skip OAK / OAK4 compilation
    python train_and_export.py --oak-target rvc4     # OAK4 only

Requirements:
    pip install ultralytics ethos-u-vela             # vela for AE3
    # stedgeai CLI from ST must be on PATH for N6    # or skip N6 compilation
    # docker (Engine 24+) with luxonis/modelconverter-rvc2:local /
    #   -rvc4:local images for OAK / OAK4 — build both from scratch with
    #   docker/oak/build.sh (requires user-supplied OpenVINO + SNPE archives,
    #   see docker/oak/extra_packages/README.md).

VisDrone classes (10):
    pedestrian, people, bicycle, car, van, truck,
    tricycle, awning-tricycle, bus, motor
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISDRONE_YAML = "VisDrone.yaml"  # Ultralytics auto-downloads VisDrone

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# Models to train: (friendly name, Ultralytics model id)
ALL_MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
}

# Board-specific export targets
BOARDS = {
    "ae3": {
        "description": "OpenMV AE3 – Alif Ensemble E3, Ethos-U55 NPU",
        "npu_tool": "vela",
        # Ethos-U55 256 MAC config matching the AE3 primary NPU
        "vela_args": [
            "--accelerator-config", "ethos-u55-256",
            "--system-config", "Ethos_U55_High_End_Embedded",
            "--memory-mode", "Shared_Sram",
            "--optimise", "Performance",
        ],
    },
    "n6": {
        "description": "OpenMV N6 – STM32N6, ST Neural-ART 600 GOPS NPU",
        "npu_tool": "stedgeai",
        "stedgeai_args": [
            "--target", "stm32n6",
            "--input-data-type", "uint8",
            "--inputs-ch-position", "chlast",
        ],
    },
}

# Default training hyper-parameters (tuned for small-object drone imagery).
# See the "Hyperparameter audit" section of the README for the rationale
# behind each non-default choice and the cross-references to the Ultralytics
# docs / community thread that motivates them.
#
# GPU-utilisation knobs (``batch`` / ``cache`` / ``workers``) are listed
# here as the safe defaults but are also overridable from the CLI – see
# the ``--batch`` / ``--cache`` / ``--workers`` / ``--device`` flags in
# ``parse_args`` and the "GPU performance & autobatching" section of the
# README for the trade-offs. ``batch=-1`` runs Ultralytics' AutoBatch
# helper which sizes the batch to ~60% of free GPU memory; passing a
# float in (0, 1) on the CLI (e.g. ``--batch 0.85``) targets that
# fraction instead, which is the most reliable single-knob speedup on a
# dedicated GPU.
DEFAULT_TRAIN_ARGS = dict(
    data=VISDRONE_YAML,
    epochs=100,
    patience=20,
    batch=-1,            # auto-batch (~60% GPU memory; CLI-overridable)
    imgsz=640,           # VisDrone benefits from larger input
    optimizer="auto",
    cos_lr=True,
    amp=True,            # mixed precision – major GPU speedup, default on
    close_mosaic=10,
    # NOTE: ``multi_scale=True`` is intentionally **disabled**.
    # Ultralytics' multi-scale path calls
    # ``nn.functional.interpolate(imgs, size=ns, mode="bilinear", ...)``
    # inside ``DetectionTrainer.preprocess_batch``. On the PyTorch 2.x
    # builds shipped with Google Colab (and other environments where
    # AMP autocast routes the call through ``torch._decomp``), the
    # upsample decomposition crashes inside ``_compute_scale`` with
    # ``ZeroDivisionError: division by zero`` because ``out_size``
    # comes through as 0. The bug is upstream in PyTorch and not
    # something we can patch from the training script.
    #
    # The community-recommended small-object benefit is largely
    # recovered via the ``scale`` / ``mosaic`` / ``copy_paste`` ranges
    # already tuned by the Ray Tune search space, so the loss is
    # mostly cosmetic. Re-enable once Colab ships a PyTorch with the
    # decomposition fix.
    multi_scale=False,
    cache="disk",
    workers=8,
    verbose=True,
    exist_ok=True,
)

# TFLite export parameters (INT8 quantisation with calibration data)
DEFAULT_EXPORT_ARGS = dict(
    format="tflite",
    int8=True,
    data=VISDRONE_YAML,  # calibration data
    nms=False,           # no NMS baked in – post-process on device
)

# ---------------------------------------------------------------------------
# Per-model training overrides
# ---------------------------------------------------------------------------
#
# YOLO26n / YOLO26s ship with a distinct, well-published training recipe
# that differs sharply from YOLO11 defaults — different optimiser (MuSGD),
# much lower lr0 for S, higher DFL gain for N, aggressive scale, etc. See
# https://docs.ultralytics.com/guides/yolo26-training-recipe/.
#
# The table below mirrors that recipe verbatim for the hyperparameters
# the recipe specifies. Anything not listed falls back to
# ``DEFAULT_TRAIN_ARGS`` above (which is the VisDrone-audited baseline
# used for YOLO11 and as a generic starting point). Epochs / patience /
# imgsz / batch are intentionally left to the CLI so users can still
# drive runtime from --epochs / --imgsz on the command line.
MODEL_TRAIN_OVERRIDES: dict[str, dict] = {
    "yolo26n": dict(
        # Optimiser schedule — YOLO26n uses a relatively high lr0 with
        # steep decay, per the recipe.
        lr0=0.0054,
        lrf=0.0495,
        momentum=0.947,
        weight_decay=0.00064,
        warmup_epochs=0.98,
        # Loss gains — YOLO26n prioritises DFL.
        box=5.63,
        cls=0.56,
        dfl=9.04,
        # Augmentation — recipe values.
        mosaic=0.909,
        mixup=0.012,
        copy_paste=0.075,
        scale=0.562,
        degrees=1.11,
        shear=1.46,
        fliplr=0.606,
        close_mosaic=10,
    ),
    "yolo26s": dict(
        # Optimiser schedule — YOLO26s uses a much lower initial LR
        # with gentle decay, per the recipe.
        lr0=0.00038,
        lrf=0.882,
        momentum=0.948,
        weight_decay=0.00027,
        warmup_epochs=0.99,
        # Loss gains — S/M/L/X shift emphasis from DFL to box regression.
        box=9.83,
        cls=0.65,
        dfl=0.96,
        # Augmentation — recipe values. Note the aggressive scale
        # (0.9) is tolerable on YOLO26 because its Small-Target-Aware
        # Label Assignment (STAL) head is robust to tiny boxes.
        mosaic=0.992,
        mixup=0.05,
        copy_paste=0.304,
        scale=0.9,
        degrees=0.0,
        shear=0.0,
        fliplr=0.304,
        close_mosaic=10,
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"\n{'='*72}\n  {msg}\n{'='*72}\n")


def which(tool: str) -> str | None:
    """Return path to tool or None."""
    return shutil.which(tool)


def parse_batch(value):
    """
    Parse a ``--batch`` argument supporting Ultralytics' three modes:

      * ``-1``               → AutoBatch to ~60% of free GPU memory
                               (Ultralytics' default safe target).
      * ``0 < f < 1``        → AutoBatch to ``f`` * free GPU memory,
                               e.g. ``0.85`` for 85% (more aggressive).
      * positive integer N   → fixed batch size of N images.

    Floats outside (0, 1) are coerced to int so e.g. ``--batch 32`` is
    accepted as a fixed batch even when typed as ``32.0``.
    """
    if value is None:
        return -1
    f = float(value)
    if f == -1.0:
        return -1
    if 0.0 < f < 1.0:
        return f
    return int(f)


def enable_gpu_fast_path() -> None:
    """
    Enable cuDNN benchmark + TF32 matmul for maximum CUDA throughput.

    Ultralytics already turns on AMP autocast (``amp=True``) and
    pinned-memory dataloaders by default. The two extra knobs we toggle
    here are the ones it does **not** flip:

      * ``torch.backends.cudnn.benchmark = True`` — picks the fastest
        cuDNN convolution algorithm per input shape. Worth ~5-15% on
        fixed-resolution training (which we are, since ``multi_scale``
        is disabled).
      * ``torch.set_float32_matmul_precision("high")`` — enables TF32
        matmuls on Ampere+ GPUs (A100, L4, T4-next, RTX 30xx/40xx),
        ~20% faster than full FP32 with no measurable accuracy impact
        at YOLO scale.

    Both are no-ops on CPU and on GPUs that don't support them, so it
    is safe to call unconditionally at the top of ``main``.
    """
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(
            f"  GPU fast path: {name}  "
            f"({free / 1e9:.1f} / {total / 1e9:.1f} GB free) "
            f"– cuDNN benchmark on, TF32 matmul on"
        )
    except Exception:
        pass


def run_cmd(cmd: list[str], cwd: str | None = None) -> int:
    """Run a subprocess, stream stdout/stderr, return exit code."""
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


# ---------------------------------------------------------------------------
# Step 1: Training
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    pretrained: str,
    imgsz: int,
    epochs: int,
    project_dir: Path,
    *,
    batch=-1,
    cache: str | bool = "disk",
    workers: int = 8,
    device: str | None = None,
) -> Path:
    """Train one YOLO model on VisDrone. Returns path to best.pt."""
    from ultralytics import YOLO

    log(f"TRAINING {model_name} on VisDrone ({epochs} epochs, imgsz={imgsz})")

    model = YOLO(pretrained)

    # Start from the VisDrone-audited baseline, then layer any per-model
    # overrides from the YOLO26 training recipe on top (see the
    # MODEL_TRAIN_OVERRIDES table above for the rationale).
    train_args = {**DEFAULT_TRAIN_ARGS}
    overrides = MODEL_TRAIN_OVERRIDES.get(model_name)
    if overrides:
        print(f"  Applying {model_name} training recipe overrides "
              f"({len(overrides)} hyperparameters)")
        train_args.update(overrides)

    train_args["imgsz"] = imgsz
    train_args["epochs"] = epochs
    train_args["project"] = str(project_dir)
    train_args["name"] = model_name

    # GPU-utilisation knobs (CLI-overridable). See ``parse_batch`` and
    # ``enable_gpu_fast_path`` for the rationale.
    train_args["batch"] = batch
    train_args["cache"] = cache
    train_args["workers"] = workers
    if device is not None:
        train_args["device"] = device

    print(
        f"  Runtime: batch={batch}  cache={cache}  workers={workers}  "
        f"device={device or 'auto'}"
    )

    model.train(**train_args)

    best_pt = project_dir / model_name / "weights" / "best.pt"
    assert best_pt.exists(), f"Training failed – {best_pt} not found"
    print(f"  ✓ Best weights: {best_pt}")
    return best_pt


# ---------------------------------------------------------------------------
# Step 2: TFLite INT8 Export
# ---------------------------------------------------------------------------

def export_tflite_int8(
    best_pt: Path,
    model_name: str,
    imgsz_export: int,
    output_dir: Path,
) -> Path:
    """Export trained model to INT8 TFLite. Returns path to .tflite file."""
    from ultralytics import YOLO

    log(f"EXPORTING {model_name} → TFLite INT8 (imgsz={imgsz_export})")

    model = YOLO(str(best_pt))
    export_args = {**DEFAULT_EXPORT_ARGS}
    export_args["imgsz"] = imgsz_export

    result_path = model.export(**export_args)

    # Ultralytics places the tflite alongside best.pt or in a _saved_model dir
    result_path = Path(result_path)

    # Copy to our organised output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{model_name}_int8.tflite"
    shutil.copy2(result_path, dst)
    print(f"  ✓ TFLite INT8: {dst}  ({dst.stat().st_size / 1024:.0f} KB)")
    return dst


# ---------------------------------------------------------------------------
# Step 3a: Vela Compilation (AE3 – Ethos-U55)
# ---------------------------------------------------------------------------

def compile_vela(tflite_path: Path, model_name: str, output_dir: Path) -> Path | None:
    """Compile INT8 TFLite with Arm Vela for AE3 Ethos-U55 NPU."""
    log(f"COMPILING {model_name} with Vela for OpenMV AE3 (Ethos-U55)")

    if not which("vela"):
        print("  ⚠  'vela' not found on PATH – install with: pip install ethos-u-vela")
        print("     Skipping AE3 NPU compilation.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vela",
        str(tflite_path),
        "--output-dir", str(output_dir),
        *BOARDS["ae3"]["vela_args"],
    ]
    rc = run_cmd(cmd)
    if rc != 0:
        print(f"  ✗ Vela compilation failed (exit {rc})")
        return None

    # Vela output name: <stem>_vela.tflite
    vela_out = output_dir / f"{tflite_path.stem}_vela.tflite"
    if vela_out.exists():
        print(f"  ✓ Vela model: {vela_out}  ({vela_out.stat().st_size / 1024:.0f} KB)")
        return vela_out

    # Try alternative naming
    for f in output_dir.glob("*.tflite"):
        print(f"  ✓ Vela model: {f}  ({f.stat().st_size / 1024:.0f} KB)")
        return f

    print("  ✗ No Vela output found")
    return None


# ---------------------------------------------------------------------------
# Step 3b: STEdgeAI Compilation (N6 – ST Neural-ART)
# ---------------------------------------------------------------------------

def compile_stedgeai(
    tflite_path: Path, model_name: str, output_dir: Path
) -> Path | None:
    """Compile INT8 TFLite with STEdgeAI for OpenMV N6 Neural-ART NPU."""
    log(f"COMPILING {model_name} with STEdgeAI for OpenMV N6 (Neural-ART)")

    stedgeai_bin = which("stedgeai") or which("stedgeai.exe")
    if not stedgeai_bin:
        print("  ⚠  'stedgeai' not found on PATH.")
        print("     Install STM32Cube.AI / X-CUBE-AI and add Utilities/ to PATH.")
        print("     Skipping N6 NPU compilation – the INT8 TFLite can still be")
        print("     loaded directly by OpenMV N6 firmware (with CPU fallback).")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal neural-art JSON config for the OpenMV N6 profile
    neuralart_cfg = {
        "Profiles": {
            "default": {
                "options": (
                    "--native-float --mvei --cache-maintenance "
                    "--Ocache-opt --enable-virtual-mem-pools --Os --Oauto"
                ),
            }
        }
    }
    cfg_path = output_dir / "neuralart_openmv_n6.json"
    cfg_path.write_text(json.dumps(neuralart_cfg, indent=2))

    cmd = [
        stedgeai_bin,
        "generate",
        "--model", str(tflite_path),
        "--st-neural-art", f"default@{cfg_path}",
        *BOARDS["n6"]["stedgeai_args"],
    ]
    rc = run_cmd(cmd, cwd=str(output_dir))
    if rc != 0:
        print(f"  ✗ STEdgeAI compilation failed (exit {rc})")
        return None

    # stedgeai outputs to st_ai_output/ with .raw network binary + .h headers
    st_out = output_dir / "st_ai_output"
    if st_out.exists():
        raw_files = list(st_out.glob("*.raw"))
        if raw_files:
            print(f"  ✓ Neural-ART binary: {raw_files[0]}")
            return raw_files[0]

    print("  ℹ  No .raw binary found – the INT8 TFLite will still work on")
    print("     OpenMV N6 (firmware handles NPU acceleration automatically).")
    return None


# ---------------------------------------------------------------------------
# Step 3c: ONNX export (shared intermediate for Luxonis ModelConverter)
# ---------------------------------------------------------------------------

def export_onnx(
    best_pt: Path,
    model_name: str,
    imgsz_export: int,
    output_dir: Path,
) -> Path:
    """Export trained model to FP32 ONNX with static shapes.

    ONNX is the input format consumed by ``luxonis/modelconverter`` for
    both RVC2 (.blob via OpenVINO) and RVC4 (.dlc via SNPE) targets. INT8
    quantisation happens inside ModelConverter using the calibration
    images supplied to ``compile_modelconverter`` — keeping ONNX in FP32
    avoids double-quantisation and lets each backend pick its own
    quantiser.

    Returns path to ``<output_dir>/<model>.onnx``.
    """
    from ultralytics import YOLO

    log(f"EXPORTING {model_name} → ONNX FP32 (imgsz={imgsz_export})")

    model = YOLO(str(best_pt))
    result_path = model.export(
        format="onnx",
        imgsz=imgsz_export,
        opset=12,
        simplify=True,
        nms=False,
        dynamic=False,
        int8=False,
    )
    result_path = Path(result_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{model_name}.onnx"
    shutil.copy2(result_path, dst)
    print(f"  ✓ ONNX FP32: {dst}  ({dst.stat().st_size / 1024:.0f} KB)")
    return dst


# ---------------------------------------------------------------------------
# Step 3d: Luxonis ModelConverter (OAK RVC2 / OAK4 RVC4)
# ---------------------------------------------------------------------------

def _resolve_oak_calib_dir(user_dir: str | None) -> Path | None:
    """Pick a calibration image directory for OAK INT8 conversion.

    Priority:
      1. User-supplied ``--oak-calib-dir`` (must exist).
      2. Ultralytics-default VisDrone val split:
         ``~/datasets/VisDrone/VisDrone2019-DET-val/images``.

    Returns ``None`` if neither is usable; the caller should warn and
    skip rather than fail the whole pipeline.
    """
    if user_dir:
        p = Path(user_dir).expanduser().resolve()
        if not p.is_dir():
            print(f"  ⚠  --oak-calib-dir {p} does not exist or is not a directory.")
            return None
        return p

    default = (
        Path.home()
        / "datasets"
        / "VisDrone"
        / "VisDrone2019-DET-val"
        / "images"
    )
    if default.is_dir():
        return default

    print(
        f"  ⚠  No --oak-calib-dir given and the Ultralytics default "
        f"({default}) is missing.\n"
        "     Train once or download VisDrone first, then re-run with "
        "--oak-calib-dir <path>."
    )
    return None


def _write_modelconverter_yaml(
    onnx_path: Path,
    imgsz: int,
    target: str,
    calib_dir: Path,
    output_dir: Path,
) -> Path:
    """Write a minimal modelconverter YAML for one (model, target) pair.

    See https://docs.luxonis.com/software-v3/ai-inference/conversion/rvc-conversion/offline/modelconverter/
    for the schema. We deliberately keep this minimal — anything more
    elaborate (per-output dequant, custom op fusion, multi-input models)
    should be supplied via ``--oak-config``.
    """
    cfg_path = output_dir / "modelconverter.yaml"
    superblob = "true" if target == "rvc2" else "false"
    yaml = (
        "input_model: {onnx_rel}\n"
        "inputs:\n"
        "  - name: images\n"
        "    shape: [1, 3, {imgsz}, {imgsz}]\n"
        "    mean: [0.0, 0.0, 0.0]\n"
        "    scale: [255.0, 255.0, 255.0]\n"
        "    encoding:\n"
        "      from: RGB\n"
        "      to: BGR\n"
        "calibration:\n"
        "  path: {calib_rel}\n"
        "  max_images: 200\n"
        "targets:\n"
        "  {target}:\n"
        "    precision: int8\n"
        "    superblob: {superblob}\n"
    ).format(
        onnx_rel=onnx_path.name,
        imgsz=imgsz,
        calib_rel=str(calib_dir),
        target=target,
        superblob=superblob,
    )
    cfg_path.write_text(yaml)
    return cfg_path


def compile_modelconverter(
    onnx_path: Path,
    model_name: str,
    imgsz: int,
    target: str,
    output_dir: Path,
    image_tag: str,
    calib_dir: Path | None,
    user_config: Path | None = None,
) -> Path | None:
    """Run ``luxonis/modelconverter`` Docker image to compile ONNX → blob/dlc.

    Parameters
    ----------
    onnx_path : Path
        FP32 ONNX produced by ``export_onnx``.
    target : {"rvc2", "rvc4"}
        ``rvc2`` → OAK / Myriad-X (.blob/.superblob).
        ``rvc4`` → OAK4 / Qualcomm (.dlc).
    image_tag : str
        Docker image to invoke. Default ``luxonis/modelconverter-rvc2:latest`` /
        ``luxonis/modelconverter-rvc4:latest``. The RVC4 image must be built
        locally — see ``docker/oak4-modelconverter.Dockerfile``.
    user_config : Path | None
        If provided, used verbatim instead of the auto-generated YAML.

    Returns the path to the produced artefact, or ``None`` on failure.
    """
    pretty_target = "OAK / RVC2" if target == "rvc2" else "OAK4 / RVC4"
    log(f"COMPILING {model_name} with ModelConverter for {pretty_target}")

    if not which("docker"):
        print("  ⚠  'docker' not found on PATH – install Docker Engine 24+.")
        print(f"     Skipping {pretty_target} compilation.")
        return None

    if calib_dir is None and user_config is None:
        print(
            f"  ⚠  No calibration directory available for {pretty_target}; "
            "skipping. Pass --oak-calib-dir or --oak-config."
        )
        return None

    # RVC2 + small @ 320 is borderline on Myriad-X SHAVE memory. Warn but
    # continue — the converter will surface the real error if it OOMs.
    if target == "rvc2" and model_name.endswith("s") and imgsz > 288:
        print(
            f"  ⚠  RVC2 + {model_name} @ {imgsz}px may exceed Myriad-X "
            "SHAVE budget. If conversion fails, retry with "
            "--imgsz-export 256."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage the ONNX next to the YAML so paths in the config are
    # relative — keeps the Docker mount layout simple.
    staged_onnx = output_dir / onnx_path.name
    if staged_onnx.resolve() != onnx_path.resolve():
        shutil.copy2(onnx_path, staged_onnx)

    if user_config is not None:
        cfg_path = output_dir / "modelconverter.yaml"
        shutil.copy2(user_config, cfg_path)
    else:
        cfg_path = _write_modelconverter_yaml(
            staged_onnx, imgsz, target, calib_dir, output_dir
        )

    # Mount the work dir AND the calibration dir (which may live outside
    # the project tree, e.g. ~/datasets/...).
    work_abs = output_dir.resolve()
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_abs}:/work",
        "-w", "/work",
    ]
    if calib_dir is not None:
        docker_cmd += ["-v", f"{calib_dir.resolve()}:{calib_dir}:ro"]
    docker_cmd += [
        image_tag,
        "convert", target,
        "--config", "modelconverter.yaml",
        "--output", "/work",
    ]

    rc = run_cmd(docker_cmd)
    if rc != 0:
        print(f"  ✗ ModelConverter ({pretty_target}) failed (exit {rc})")
        return None

    suffix = ".blob" if target == "rvc2" else ".dlc"
    candidates = sorted(output_dir.rglob(f"*{suffix}"))
    if not candidates:
        # ModelConverter sometimes writes .superblob for RVC2; accept either.
        if target == "rvc2":
            candidates = sorted(output_dir.rglob("*.superblob"))
    if not candidates:
        print(f"  ✗ No {suffix} artefact found under {output_dir}")
        return None

    artefact = candidates[0]
    print(f"  ✓ {pretty_target} model: {artefact}  "
          f"({artefact.stat().st_size / 1024:.0f} KB)")
    return artefact


# ---------------------------------------------------------------------------
# Step 4: Generate labels file
# ---------------------------------------------------------------------------

def write_labels(output_dir: Path) -> Path:
    """Write VisDrone labels.txt for OpenMV."""
    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(VISDRONE_CLASSES) + "\n")
    print(f"  ✓ Labels: {labels_path}")
    return labels_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO models on VisDrone and export for OpenMV AE3/N6"
    )
    p.add_argument(
        "--models", nargs="+", default=list(ALL_MODELS.keys()),
        choices=list(ALL_MODELS.keys()),
        help="Which models to train/export (default: all four)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Training image size")
    p.add_argument(
        "--imgsz-export", type=int, default=None,
        help="Export image size (default: 256 for nano, 320 for small)"
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    # ---- GPU performance knobs --------------------------------------
    p.add_argument(
        "--batch", type=parse_batch, default=-1,
        help="Batch size: int N (fixed), -1 (AutoBatch ~60%% GPU mem, "
             "default), or 0<f<1 (AutoBatch f%% GPU mem, e.g. 0.85)",
    )
    p.add_argument(
        "--cache", choices=["ram", "disk", "none"], default="disk",
        help="Dataset cache mode (default: disk; 'ram' is fastest if "
             "the dataset + augmentations fit in host RAM)",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Dataloader worker processes per GPU (default: 8; raise on "
             "high-core hosts, lower on RAM-constrained Colab runtimes)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="CUDA device(s) for training, e.g. '0', '0,1', or 'cpu' "
             "(default: auto-detect; multi-GPU triggers DDP)",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training, only export from existing best.pt"
    )
    p.add_argument(
        "--skip-npu", action="store_true",
        help="Skip NPU compilation (Vela / STEdgeAI)"
    )
    # ---- Luxonis OAK / OAK4 -----------------------------------------
    p.add_argument(
        "--skip-oak", action="store_true",
        help="Skip OAK / OAK4 compilation (ONNX export + ModelConverter)"
    )
    p.add_argument(
        "--oak-target", choices=["rvc2", "rvc4", "both"], default="both",
        help="Which Luxonis target(s) to compile for (default: both)"
    )
    p.add_argument(
        "--oak-rvc2-image", type=str,
        default="luxonis/modelconverter-rvc2:local",
        help="Docker image tag for the RVC2 (OAK) converter "
             "(default: luxonis/modelconverter-rvc2:local — built from "
             "scratch by docker/oak/build.sh)"
    )
    p.add_argument(
        "--oak-rvc4-image", type=str,
        default="luxonis/modelconverter-rvc4:local",
        help="Docker image tag for the RVC4 (OAK4) converter "
             "(default: luxonis/modelconverter-rvc4:local — built from "
             "scratch by docker/oak/build.sh)"
    )
    p.add_argument(
        "--oak-calib-dir", type=str, default=None,
        help="Calibration image dir for INT8 quantisation (default: "
             "~/datasets/VisDrone/VisDrone2019-DET-val/images)"
    )
    p.add_argument(
        "--oak-config", type=str, default=None,
        help="Optional path to a user-supplied modelconverter YAML, "
             "applied verbatim (overrides auto-generated config)."
    )
    p.add_argument(
        "--project", type=str, default="runs/visdrone",
        help="Project directory for training outputs"
    )
    p.add_argument(
        "--output", type=str, default="export",
        help="Root output directory for exported models"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project)
    output_root = Path(args.output)

    # Determine per-model export size
    def get_export_imgsz(model_name: str) -> int:
        if args.imgsz_export is not None:
            return args.imgsz_export
        # Nano models → 256, Small models → 320 (fits NPU memory budgets)
        return 256 if model_name.endswith("n") else 320

    log("YOLO × VisDrone → OpenMV NPU Pipeline")
    print(f"  Models : {', '.join(args.models)}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Train sz: {args.imgsz}")
    print(f"  Project: {project_dir}")
    print(f"  Output : {output_root}")

    # Flip on cuDNN benchmark + TF32 before any CUDA work happens.
    enable_gpu_fast_path()

    # Translate "none" sentinel into Ultralytics' False (cache disabled).
    cache_arg: str | bool = False if args.cache == "none" else args.cache

    # Write labels once
    output_root.mkdir(parents=True, exist_ok=True)
    write_labels(output_root)

    for model_name in args.models:
        pretrained = ALL_MODELS[model_name]
        export_imgsz = get_export_imgsz(model_name)

        # --- Train --------------------------------------------------------
        best_pt = project_dir / model_name / "weights" / "best.pt"
        if not args.skip_train:
            best_pt = train_model(
                model_name, pretrained, args.imgsz, args.epochs, project_dir,
                batch=args.batch,
                cache=cache_arg,
                workers=args.workers,
                device=args.device,
            )
        else:
            if not best_pt.exists():
                print(f"  ✗ {best_pt} not found – cannot skip training")
                continue
            print(f"  ℹ  Re-using existing weights: {best_pt}")

        # --- Export TFLite INT8 -------------------------------------------
        tflite_dir = output_root / "tflite"
        tflite_path = export_tflite_int8(
            best_pt, model_name, export_imgsz, tflite_dir
        )

        if not args.skip_npu:
            # --- Compile for AE3 (Vela) ----------------------------------
            ae3_dir = output_root / "ae3" / model_name
            compile_vela(tflite_path, model_name, ae3_dir)

            # Also copy the raw INT8 TFLite for AE3 (fallback / CPU mode)
            shutil.copy2(tflite_path, ae3_dir / tflite_path.name)

            # --- Compile for N6 (STEdgeAI) -------------------------------
            n6_dir = output_root / "n6" / model_name
            compile_stedgeai(tflite_path, model_name, n6_dir)

            # Also copy the raw INT8 TFLite for N6 (firmware loads it directly)
            n6_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tflite_path, n6_dir / tflite_path.name)

        # --- ONNX + Luxonis ModelConverter (OAK / OAK4) ------------------
        if not args.skip_oak:
            onnx_dir = output_root / "onnx"
            onnx_path = export_onnx(best_pt, model_name, export_imgsz, onnx_dir)
            calib_dir = _resolve_oak_calib_dir(args.oak_calib_dir)
            user_cfg = Path(args.oak_config).expanduser() if args.oak_config else None

            if args.oak_target in ("rvc2", "both"):
                compile_modelconverter(
                    onnx_path, model_name, export_imgsz, "rvc2",
                    output_root / "oak" / model_name,
                    args.oak_rvc2_image, calib_dir, user_cfg,
                )
            if args.oak_target in ("rvc4", "both"):
                compile_modelconverter(
                    onnx_path, model_name, export_imgsz, "rvc4",
                    output_root / "oak4" / model_name,
                    args.oak_rvc4_image, calib_dir, user_cfg,
                )

    # --- Summary ----------------------------------------------------------
    log("PIPELINE COMPLETE")
    print("Output directory structure:")
    for dirpath, dirnames, filenames in os.walk(output_root):
        depth = dirpath.replace(str(output_root), "").count(os.sep)
        indent = "  " * (depth + 1)
        print(f"{indent}{os.path.basename(dirpath)}/")
        sub_indent = "  " * (depth + 2)
        for f in sorted(filenames):
            fpath = Path(dirpath) / f
            size_kb = fpath.stat().st_size / 1024
            print(f"{sub_indent}{f}  ({size_kb:.0f} KB)")

    print(
        "\n  Next steps:\n"
        "  OpenMV (AE3 / N6):\n"
        "    1. Copy the model .tflite (or _vela.tflite for AE3) + labels.txt\n"
        "       to your OpenMV Cam's internal flash or SD card.\n"
        "    2. Upload the matching MicroPython script from openmv-scripts/.\n"
        "    3. Run from OpenMV IDE or on boot.\n"
        "  Luxonis OAK / OAK4 (DepthAI v3 peripheral mode):\n"
        "    1. Place the .blob (OAK) or .dlc (OAK4) next to the matching\n"
        "       oak-scripts/main_yolo*.py, or pass --model on the command line.\n"
        "    2. uv sync --extra oak  (installs depthai + opencv-python).\n"
        "    3. python oak-scripts/main_yolo<model>.py\n"
    )


if __name__ == "__main__":
    main()
