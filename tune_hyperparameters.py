#!/usr/bin/env python3
"""
Hyperparameter Tuning for YOLO on VisDrone with Ray Tune + TensorBoard
========================================================================

Wraps Ultralytics' built-in ``model.tune(use_ray=True)`` integration
(see https://docs.ultralytics.com/integrations/ray-tune/) with a
VisDrone-focused search space tuned for small-object drone imagery.

Ray Tune uses the ASHA scheduler by default – trials that underperform
are stopped early, so ``--iterations`` can be set relatively high.

TensorBoard integration:
    Ultralytics writes per-trial TFEvent logs into each trial's
    training directory (``runs/detect/tune*/train/``). Ray Tune mirrors
    these under its storage path. Launch TensorBoard on either:

        tensorboard --logdir runs/detect
        tensorboard --logdir <ray_storage_path>/<exp_name>

    The script prints the exact commands at the end of the run.

Usage:
    # Default: yolo11n, 10 trials, 30 epochs each, grace period 10
    uv run python tune_hyperparameters.py

    # Small model, more trials, single GPU per trial
    uv run python tune_hyperparameters.py \\
        --model yolo11s --iterations 30 --gpu-per-trial 1

    # Re-use a custom dataset
    uv run python tune_hyperparameters.py --data VisDrone.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_search_space():
    """
    VisDrone-focused hyperparameter search space.

    The default Ultralytics space covers 28 parameters; we narrow it to
    the ones that matter most for small-object drone detection and
    tighten some ranges based on common VisDrone training practice.
    """
    from ray import tune

    return {
        # --- Optimiser -------------------------------------------------
        "lr0":          tune.loguniform(1e-5, 1e-2),
        "lrf":          tune.uniform(0.01, 0.2),
        "momentum":     tune.uniform(0.85, 0.98),
        "weight_decay": tune.uniform(0.0, 5e-4),
        "warmup_epochs":   tune.uniform(0.0, 5.0),
        "warmup_momentum": tune.uniform(0.5, 0.95),

        # --- Loss weights (small-object detection is sensitive here) --
        "box": tune.uniform(4.0, 10.0),
        "cls": tune.uniform(0.3, 1.5),
        "dfl": tune.uniform(1.0, 3.0),

        # --- Augmentation ---------------------------------------------
        # Small objects are easily destroyed by aggressive scale/shear
        "hsv_h":    tune.uniform(0.0, 0.03),
        "hsv_s":    tune.uniform(0.3, 0.9),
        "hsv_v":    tune.uniform(0.2, 0.7),
        "degrees":  tune.uniform(0.0, 10.0),
        "translate": tune.uniform(0.0, 0.2),
        "scale":    tune.uniform(0.2, 0.6),   # narrower than default
        "fliplr":   tune.uniform(0.0, 0.5),
        "mosaic":   tune.uniform(0.8, 1.0),   # keep mosaic high
        "mixup":    tune.uniform(0.0, 0.2),
        "copy_paste": tune.uniform(0.0, 0.3),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ray Tune hyperparameter search for YOLO on VisDrone"
    )
    p.add_argument(
        "--model", default="yolo11n.pt",
        help="Pretrained YOLO weights to start from (default: yolo11n.pt)",
    )
    p.add_argument(
        "--data", default="VisDrone.yaml",
        help="Ultralytics dataset YAML (default: VisDrone.yaml)",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Max epochs per trial (ASHA will stop early – default: 30)",
    )
    p.add_argument(
        "--iterations", type=int, default=10,
        help="Number of Ray Tune trials to sample (default: 10)",
    )
    p.add_argument(
        "--grace-period", type=int, default=10,
        help="ASHA grace period in epochs before a trial can be pruned "
             "(default: 10)",
    )
    p.add_argument(
        "--gpu-per-trial", type=float, default=None,
        help="GPUs per trial (fractional allowed, e.g. 0.5 for 2 trials/GPU). "
             "Default: None (auto-detect)",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size (default: 640)",
    )
    p.add_argument(
        "--output", default="runs/tune",
        help="Directory for tuning results (default: runs/tune)",
    )
    p.add_argument(
        "--name", default="visdrone_raytune",
        help="Experiment name used by Ray Tune (default: visdrone_raytune)",
    )
    p.add_argument(
        "--default-space", action="store_true",
        help="Use Ultralytics' full default 28-parameter search space "
             "instead of the VisDrone-tuned one",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Dependency checks – Ray Tune + TensorBoard are optional extras
    # ------------------------------------------------------------------
    try:
        import ray  # noqa: F401
        from ray import tune  # noqa: F401
    except ImportError:
        print(
            "ERROR: Ray Tune is not installed.\n"
            "Install the tuning extras with:\n"
            "    uv sync --extra tune\n"
            "or directly with pip:\n"
            "    pip install 'ray[tune]' tensorboard",
            file=sys.stderr,
        )
        sys.exit(1)

    from ultralytics import YOLO

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Ray Tune × Ultralytics YOLO × VisDrone")
    print("=" * 72)
    print(f"  Model        : {args.model}")
    print(f"  Data         : {args.data}")
    print(f"  Iterations   : {args.iterations}")
    print(f"  Epochs/trial : {args.epochs} (grace={args.grace_period})")
    print(f"  Image size   : {args.imgsz}")
    print(f"  GPU/trial    : {args.gpu_per_trial if args.gpu_per_trial else 'auto'}")
    print(f"  Output       : {output_dir}")
    print(f"  Search space : {'default (28 params)' if args.default_space else 'VisDrone-tuned'}")
    print("=" * 72, "\n")

    # ------------------------------------------------------------------
    # Build the YOLO model and kick off Ray Tune search.
    # Ultralytics' built-in integration handles:
    #   - ASHAScheduler (grace_period, reduction_factor=3)
    #   - Per-trial TensorBoard logging via its default TB callback
    #   - Result aggregation into a ray.tune.ResultGrid
    # ------------------------------------------------------------------
    model = YOLO(args.model)

    tune_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        iterations=args.iterations,
        grace_period=args.grace_period,
        use_ray=True,
        project=str(output_dir),
        name=args.name,
    )
    if args.gpu_per_trial is not None:
        tune_kwargs["gpu_per_trial"] = args.gpu_per_trial
    if not args.default_space:
        tune_kwargs["space"] = build_search_space()

    result_grid = model.tune(**tune_kwargs)

    # ------------------------------------------------------------------
    # Report best trial and persist hyperparameters
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" TUNING COMPLETE")
    print("=" * 72)

    try:
        best_result = result_grid.get_best_result(
            metric="metrics/mAP50-95(B)", mode="max"
        )
        best_cfg = best_result.config
        best_metrics = best_result.metrics or {}

        print("\n  Best trial config:")
        for k, v in sorted(best_cfg.items()):
            print(f"    {k:20s} = {v}")

        map5095 = best_metrics.get("metrics/mAP50-95(B)")
        map50 = best_metrics.get("metrics/mAP50(B)")
        if map5095 is not None:
            print(f"\n  mAP50-95 : {map5095:.4f}")
        if map50 is not None:
            print(f"  mAP50    : {map50:.4f}")

        best_cfg_path = output_dir / f"{args.name}_best_hyperparameters.json"
        best_cfg_path.write_text(json.dumps(best_cfg, indent=2, default=str))
        print(f"\n  Best hyperparameters saved to: {best_cfg_path}")
    except Exception as e:
        print(f"  Could not extract best trial automatically: {e}")
        print("  Inspect the ResultGrid manually via the saved Ray experiment.")

    # ------------------------------------------------------------------
    # TensorBoard instructions
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(" TensorBoard")
    print("-" * 72)
    print("  Ultralytics writes TFEvent logs for every trial. Launch with:")
    print(f"    uv run tensorboard --logdir {output_dir}")
    print("  Or point TensorBoard at Ray Tune's storage path (default ~/ray_results):")
    print(f"    uv run tensorboard --logdir ~/ray_results/{args.name}")
    print(
        "\n  Next step – re-train with the best hyperparameters by passing\n"
        f"  {args.name}_best_hyperparameters.json values into train_and_export.py\n"
    )


if __name__ == "__main__":
    main()
