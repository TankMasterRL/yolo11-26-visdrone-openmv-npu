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
DEFAULT_TRAIN_ARGS = dict(
    data=VISDRONE_YAML,
    epochs=100,
    patience=20,
    batch=-1,            # auto-batch
    imgsz=640,           # VisDrone benefits from larger input
    optimizer="auto",
    cos_lr=True,
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
                model_name, pretrained, args.imgsz, args.epochs, project_dir
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
        "  1. Copy the model .tflite (or _vela.tflite for AE3) + labels.txt\n"
        "     to your OpenMV Cam's internal flash or SD card.\n"
        "  2. Upload the matching MicroPython script from openmv-scripts/.\n"
        "  3. Run from OpenMV IDE or on boot.\n"
    )


if __name__ == "__main__":
    main()
