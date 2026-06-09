#!/usr/bin/env python3
"""
YOLO Object Detection Training Pipeline for OpenMV AE3 & N6
=============================================================

Trains YOLO11n, YOLO11s, YOLO26n, YOLO26s on VisDrone dataset, exports
INT8 TFLite models, then compiles NPU-optimised images for:

  - OpenMV AE3  (Alif Ensemble E3 / Ethos-U55)  → Vela-compiled TFLite
  - OpenMV N6   (STM32N6 / Neural-ART NPU)       → stedgeai network binary

Usage:
    python train_and_export.py                       # full pipeline
    python train_and_export.py --skip-train          # export only (re-uses best.pt)
    python train_and_export.py --models yolo11n      # single model
    python train_and_export.py --imgsz 256           # custom input size

Requirements:
    pip install ultralytics ethos-u-vela             # vela for AE3
    # stedgeai CLI from ST must be on PATH for N6    # or skip N6 compilation

VisDrone classes (10):
    pedestrian, people, bicycle, car, van, truck,
    tricycle, awning-tricycle, bus, motor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from pipeline_logging import setup_logging, shutdown_logging, stream_subprocess

logger = logging.getLogger("train_and_export")

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
    logger.info("")
    logger.info("=" * 72)
    logger.info("  %s", msg)
    logger.info("=" * 72)


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
        logger.info(
            "  GPU fast path: %s  (%.1f / %.1f GB free) "
            "– cuDNN benchmark on, TF32 matmul on",
            name, free / 1e9, total / 1e9,
        )
    except Exception:
        pass


def run_cmd(cmd: list[str], cwd: str | None = None, *, tag: str = "cmd") -> int:
    """Run a subprocess, relaying its output into the unified log.

    Output is captured and re-emitted line-by-line through the module
    logger (tagged with ``tag``) so it is timestamped and sequentially
    integrated with the main process's log — see ``pipeline_logging``.
    """
    return stream_subprocess(cmd, logger=logger, tag=tag, cwd=cwd)


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
        logger.info(
            "  Applying %s training recipe overrides (%d hyperparameters)",
            model_name, len(overrides),
        )
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

    logger.info(
        "  Runtime: batch=%s  cache=%s  workers=%s  device=%s",
        batch, cache, workers, device or "auto",
    )

    t0 = time.perf_counter()
    model.train(**train_args)
    logger.info("  ✓ Training completed in %.1fs", time.perf_counter() - t0)

    best_pt = project_dir / model_name / "weights" / "best.pt"
    assert best_pt.exists(), f"Training failed – {best_pt} not found"
    logger.info("  ✓ Best weights: %s", best_pt)
    return best_pt


# ---------------------------------------------------------------------------
# Step 2: TFLite INT8 Export
# ---------------------------------------------------------------------------

def _select_npu_tflite(returned: Path) -> tuple[Path, str]:
    """Pick the genuinely-integer TFLite for NPU compilation.

    ``model.export(format="tflite", int8=True)`` returns ``*_int8.tflite``,
    which Ultralytics creates by **renaming onnx2tf's
    ``*_dynamic_range_quant.tflite``** (see
    ``ultralytics/utils/export/tensorflow.py``). That is a *weight-only*
    ("hybrid") model: INT8 weights but **FLOAT32 feature maps and FLOAT32
    input/output**. It is unusable on the Ethos-U55 — Arm Vela rejects every
    operator with ``unsupported DataType Float32`` and the whole graph falls
    back to the CPU (``NPU operators = 0``). onnx2tf also emits, in the same
    ``*_saved_model`` directory, the integer models we can actually deploy:

      * ``*_full_integer_quant.tflite`` — INT8 weights, activations *and*
        int8 input/output. The correct artefact for the Ethos-U55 (Vela)
        and ST Neural-ART (STEdgeAI) NPUs, and what the OpenMV firmware
        expects to route to the accelerator.
      * ``*_integer_quant.tflite`` — INT8 internals but FLOAT32 I/O. Vela
        still accelerates the convolution core; only the leading QUANTIZE /
        trailing DEQUANTIZE run on the CPU. Used only if the fully-integer
        variant is missing.

    Returns the chosen path plus a short human-readable label. Falls back to
    the returned dynamic-range file (the previous, broken behaviour) only if
    neither integer variant exists, so the pipeline never loses a file.
    """
    saved_model = returned.parent
    for pattern, label in (
        ("*_full_integer_quant.tflite", "full-integer (INT8 I/O)"),
        ("*_integer_quant.tflite", "integer (INT8 core, FP32 I/O)"),
    ):
        # The first pattern is a suffix of the second, but we only reach the
        # ``_integer_quant`` branch when no ``_full_integer_quant`` file
        # exists, so the glob there can only match the plain integer model.
        matches = sorted(saved_model.glob(pattern))
        if matches:
            return matches[0], label
    return returned, "dynamic-range (FP32 — NOT NPU-deployable)"


def export_tflite_int8(
    best_pt: Path,
    model_name: str,
    imgsz_export: int,
    output_dir: Path,
) -> Path:
    """Export trained model to INT8 TFLite. Returns path to .tflite file.

    Imports and initialises TensorFlow (via Ultralytics' onnx2tf chain),
    so it MUST run in a process where the GPU is hidden. Drive it through
    ``run_export_isolated`` rather than calling it directly — see that
    wrapper for why an in-process ``CUDA_VISIBLE_DEVICES`` tweak is too
    late once training has initialised CUDA.
    """
    from ultralytics import YOLO

    log(f"EXPORTING {model_name} → TFLite INT8 (imgsz={imgsz_export})")

    model = YOLO(str(best_pt))
    export_args = {**DEFAULT_EXPORT_ARGS}
    export_args["imgsz"] = imgsz_export

    t0 = time.perf_counter()
    result_path = model.export(**export_args)
    logger.info("  Conversion finished in %.1fs", time.perf_counter() - t0)

    # Ultralytics places the tflite alongside best.pt or in a _saved_model dir
    result_path = Path(result_path)

    # ``model.export`` returns the *dynamic-range* model (INT8 weights but
    # FLOAT32 feature maps/I/O) renamed to ``*_int8.tflite``. The Ethos-U55
    # cannot run it — Vela flags every op as ``unsupported DataType Float32``
    # and puts 0 ops on the NPU. Swap in the fully-integer sibling onnx2tf
    # wrote to the same dir; see ``_select_npu_tflite`` for the ranking.
    npu_path, variant = _select_npu_tflite(result_path)
    if npu_path != result_path:
        logger.info("  Selected %s model for NPU: %s", variant, npu_path.name)
    else:
        logger.warning(
            "  ⚠  No fully-integer TFLite alongside %s — using the %s model. "
            "Vela will reject every op as Float32 and run nothing on the "
            "Ethos-U55 NPU.", result_path.name, variant,
        )

    # Copy to our organised output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{model_name}_int8.tflite"
    shutil.copy2(npu_path, dst)
    logger.info("  ✓ TFLite INT8: %s  (%.0f KB)", dst, dst.stat().st_size / 1024)
    return dst


def run_export_isolated(
    best_pt: Path,
    model_name: str,
    imgsz_export: int,
    output_dir: Path,
) -> Path:
    """Run ``export_tflite_int8`` in a subprocess with the GPU hidden.

    Why a subprocess instead of just setting ``CUDA_VISIBLE_DEVICES=-1``
    in-process: TensorFlow grabs the first CUDA device when its context
    initialises, and the ``onnx2tf`` step of the export then runs ops
    (e.g. ``tf.cast``) on it. On a GPU newer than the kernels bundled with
    the installed TF build — an RTX 5090 (Blackwell, ``sm_120``) under
    TensorFlow 2.19 — those kernels are absent, so TF JIT-compiles from
    PTX and the launch dies with ``CUDA_ERROR_INVALID_HANDLE`` on
    ``[Op:Cast]``.

    ``CUDA_VISIBLE_DEVICES`` is only consulted by the CUDA driver at the
    first ``cuInit()`` of a process. By export time the pipeline has
    already trained on the GPU, so PyTorch initialised the driver and the
    variable is frozen — flipping it afterwards is ignored by every
    library in the process, TensorFlow included (an earlier in-process
    attempt failed for exactly this reason). The only reliable fix is a
    fresh process: we launch the export with ``CUDA_VISIBLE_DEVICES=-1``
    set *before* the interpreter starts, so its ``cuInit()`` honours the
    setting and neither PyTorch nor TensorFlow can see the GPU. The
    conversion (graph transform + INT8 calibration on a ≤320 px model) is
    CPU-appropriate, so running it GPU-free is both correct and fast.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # The child re-imports this module and calls export_tflite_int8 with
    # absolute paths, so it is independent of its working directory. We
    # keep the child's cwd inherited from the parent so the relative
    # ``data=VisDrone.yaml`` calibration path resolves exactly as it does
    # in the in-process path.
    child_src = (
        "import sys; from pathlib import Path; "
        "import pipeline_logging; "
        "pipeline_logging.setup_child_logging('train_and_export'); "
        "import train_and_export as t; "
        "t.export_tflite_int8(Path(sys.argv[1]), sys.argv[2], "
        "int(sys.argv[3]), Path(sys.argv[4]))"
    )
    cmd = [
        sys.executable, "-c", child_src,
        str(best_pt.resolve()), model_name, str(imgsz_export),
        str(output_dir.resolve()),
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # honoured at the child's cuInit()
    # Ensure this script is importable in the child regardless of cwd.
    script_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = (
        script_dir + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else script_dir
    )

    logger.info(
        "  Exporting in an isolated CPU process "
        "(CUDA_VISIBLE_DEVICES=-1, GPU hidden from TensorFlow)"
    )
    rc = stream_subprocess(cmd, logger=logger, tag=f"export:{model_name}", env=env)

    dst = output_dir / f"{model_name}_int8.tflite"
    if rc != 0 or not dst.exists():
        raise RuntimeError(
            f"TFLite export for {model_name} failed in the isolated "
            f"export process (exit code {rc})."
        )
    return dst


# ---------------------------------------------------------------------------
# Step 3a: Vela Compilation (AE3 – Ethos-U55)
# ---------------------------------------------------------------------------

def compile_vela(tflite_path: Path, model_name: str, output_dir: Path) -> Path | None:
    """Compile INT8 TFLite with Arm Vela for AE3 Ethos-U55 NPU."""
    log(f"COMPILING {model_name} with Vela for OpenMV AE3 (Ethos-U55)")

    if not which("vela"):
        logger.warning("  ⚠  'vela' not found on PATH – install with: pip install ethos-u-vela")
        logger.warning("     Skipping AE3 NPU compilation.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vela",
        str(tflite_path),
        "--output-dir", str(output_dir),
        *BOARDS["ae3"]["vela_args"],
    ]
    rc = run_cmd(cmd, tag="vela")
    if rc != 0:
        logger.error("  ✗ Vela compilation failed (exit %d)", rc)
        return None

    # Vela output name: <stem>_vela.tflite
    vela_out = output_dir / f"{tflite_path.stem}_vela.tflite"
    if vela_out.exists():
        logger.info("  ✓ Vela model: %s  (%.0f KB)", vela_out, vela_out.stat().st_size / 1024)
        return vela_out

    # Try alternative naming
    for f in output_dir.glob("*.tflite"):
        logger.info("  ✓ Vela model: %s  (%.0f KB)", f, f.stat().st_size / 1024)
        return f

    logger.error("  ✗ No Vela output found")
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
        logger.warning("  ⚠  'stedgeai' not found on PATH.")
        logger.warning("     Install STM32Cube.AI / X-CUBE-AI and add Utilities/ to PATH.")
        logger.warning("     Skipping N6 NPU compilation – the INT8 TFLite can still be")
        logger.warning("     loaded directly by OpenMV N6 firmware (with CPU fallback).")
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
    rc = run_cmd(cmd, cwd=str(output_dir), tag="stedgeai")
    if rc != 0:
        logger.error("  ✗ STEdgeAI compilation failed (exit %d)", rc)
        return None

    # stedgeai outputs to st_ai_output/ with .raw network binary + .h headers
    st_out = output_dir / "st_ai_output"
    if st_out.exists():
        raw_files = list(st_out.glob("*.raw"))
        if raw_files:
            logger.info("  ✓ Neural-ART binary: %s", raw_files[0])
            return raw_files[0]

    logger.info("  ℹ  No .raw binary found – the INT8 TFLite will still work on")
    logger.info("     OpenMV N6 (firmware handles NPU acceleration automatically).")
    return None


# ---------------------------------------------------------------------------
# Step 4: Generate labels file
# ---------------------------------------------------------------------------

def write_labels(output_dir: Path) -> Path:
    """Write VisDrone labels.txt for OpenMV."""
    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(VISDRONE_CLASSES) + "\n")
    logger.info("  ✓ Labels: %s", labels_path)
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
    p.add_argument(
        "--project", type=str, default="runs/visdrone",
        help="Project directory for training outputs"
    )
    p.add_argument(
        "--output", type=str, default="export",
        help="Root output directory for exported models"
    )
    p.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for the timestamped run log file (default: logs/). "
             "Subprocess output (export / Vela / STEdgeAI) is captured into "
             "the same file, in order."
    )
    return p.parse_args()


def _run_pipeline(args: argparse.Namespace) -> None:
    project_dir = Path(args.project)
    output_root = Path(args.output)

    # Determine per-model export size
    def get_export_imgsz(model_name: str) -> int:
        if args.imgsz_export is not None:
            return args.imgsz_export
        # Nano models → 256, Small models → 320 (fits NPU memory budgets)
        return 256 if model_name.endswith("n") else 320

    log("YOLO × VisDrone → OpenMV NPU Pipeline")
    logger.info("  Models : %s", ", ".join(args.models))
    logger.info("  Epochs : %s", args.epochs)
    logger.info("  Train sz: %s", args.imgsz)
    logger.info("  Project: %s", project_dir)
    logger.info("  Output : %s", output_root)

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
                logger.error("  ✗ %s not found – cannot skip training", best_pt)
                continue
            logger.info("  ℹ  Re-using existing weights: %s", best_pt)

        # --- Export TFLite INT8 -------------------------------------------
        tflite_dir = output_root / "tflite"
        tflite_path = run_export_isolated(
            best_pt, model_name, export_imgsz, tflite_dir
        )

        if args.skip_npu:
            continue

        # --- Compile for AE3 (Vela) --------------------------------------
        ae3_dir = output_root / "ae3" / model_name
        compile_vela(tflite_path, model_name, ae3_dir)

        # Also copy the raw INT8 TFLite for AE3 (fallback / CPU mode)
        shutil.copy2(tflite_path, ae3_dir / tflite_path.name)

        # --- Compile for N6 (STEdgeAI) -----------------------------------
        n6_dir = output_root / "n6" / model_name
        compile_stedgeai(tflite_path, model_name, n6_dir)

        # Also copy the raw INT8 TFLite for N6 (firmware loads it directly)
        n6_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tflite_path, n6_dir / tflite_path.name)

    # --- Summary ----------------------------------------------------------
    log("PIPELINE COMPLETE")
    logger.info("Output directory structure:")
    for dirpath, dirnames, filenames in os.walk(output_root):
        depth = dirpath.replace(str(output_root), "").count(os.sep)
        indent = "  " * (depth + 1)
        logger.info("%s%s/", indent, os.path.basename(dirpath))
        sub_indent = "  " * (depth + 2)
        for f in sorted(filenames):
            fpath = Path(dirpath) / f
            size_kb = fpath.stat().st_size / 1024
            logger.info("%s%s  (%.0f KB)", sub_indent, f, size_kb)

    logger.info(
        "\n  Next steps:\n"
        "  1. Copy the model .tflite (or _vela.tflite for AE3) + labels.txt\n"
        "     to your OpenMV Cam's internal flash or SD card.\n"
        "  2. Upload the matching MicroPython script from openmv-scripts/.\n"
        "  3. Run from OpenMV IDE or on boot.\n"
    )


def main() -> None:
    args = parse_args()
    _, log_path = setup_logging("train_and_export", log_dir=args.log_dir)
    pipeline_t0 = time.perf_counter()
    try:
        _run_pipeline(args)
    except Exception:
        logger.exception("✗ Pipeline aborted by an unhandled exception")
        raise
    finally:
        logger.info("Total pipeline wall time: %.1fs",
                    time.perf_counter() - pipeline_t0)
        logger.info("Full log written to %s", log_path)
        shutdown_logging()


if __name__ == "__main__":
    main()
