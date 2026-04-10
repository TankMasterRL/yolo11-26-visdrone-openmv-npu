#!/usr/bin/env python3
"""
Hyperparameter Tuning for YOLO on VisDrone with Ray Tune + TensorBoard
========================================================================

Wraps Ultralytics' built-in ``model.tune(use_ray=True)`` integration
(see https://docs.ultralytics.com/integrations/ray-tune/) with a
VisDrone-focused search space tuned for small-object drone imagery.

By default, runs an independent tuning sweep for **all four models**
(YOLO11n, YOLO11s, YOLO26n, YOLO26s) — each gets its own Ray Tune
experiment directory and best-hyperparameters JSON.

Ray Tune uses the ASHA scheduler by default – trials that underperform
are stopped early, so ``--iterations`` can be set relatively high.

TensorBoard integration:
    Ultralytics writes per-trial TFEvent logs into each trial's
    training directory (``runs/tune/<name>/.../``). Launch TensorBoard
    against the root output directory to compare every trial of every
    model in one UI:

        tensorboard --logdir runs/tune

    Or point at Ray Tune's own storage (default ``~/ray_results``):

        tensorboard --logdir ~/ray_results

    The script prints the exact commands at the end of the run.

Usage:
    # Default: tune all 4 models, 10 trials each, 30 epochs, grace 10
    uv run python tune_hyperparameters.py

    # Subset of models, more trials, 1 GPU per trial
    uv run python tune_hyperparameters.py \\
        --models yolo11s yolo26s --iterations 30 --gpu-per-trial 1

    # Use a different dataset
    uv run python tune_hyperparameters.py --data VisDrone.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Models to tune: (friendly name → Ultralytics weights id).
# Matches ALL_MODELS in train_and_export.py so downstream workflows
# share a consistent set of model keys.
ALL_MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
}


def build_search_space():
    """
    VisDrone-focused hyperparameter search space.

    Derived from Ultralytics' default ``run_ray_tune`` space in
    ``ultralytics/utils/tuner.py`` and audited against the community
    recommendations on the Ultralytics forum for VisDrone / small-object
    drone imagery. Notable VisDrone-specific decisions:

      * ``flipud`` is included (Ultralytics default keeps it) — aerial
        imagery has no natural "up" and enabling vertical flip roughly
        doubles the effective dataset with zero label cost.
      * ``fliplr`` covers the full 0-1 range so the tuner can converge
        on the standard 0.5 — our previous 0-0.5 cap excluded the best
        value for laterally symmetric classes (car, bus, person, ...).
      * ``close_mosaic`` is tuned — the mosaic-shutdown window is one
        of the strongest knobs for small-object detection because
        mosaic distortion hurts the final fine-tuning epochs.
      * ``cutmix`` and ``copy_paste`` are both enabled with conservative
        upper bounds — both help VisDrone's many-small-objects regime
        without destroying the tiny bounding boxes.
      * ``shear`` and ``perspective`` are **intentionally omitted** —
        they destroy small-object bboxes via pixel interpolation loss.
      * ``box`` loss gain range is pushed up (5-12) to let the tuner
        emphasise localisation accuracy, which matters most for the
        tiny boxes that dominate VisDrone.

    Note: ``dfl`` only applies to YOLO11 models. YOLO26 is NMS-free and
    removes the Distribution Focal Loss head entirely, so the ``dfl``
    gain has no effect on YOLO26n / YOLO26s trials.
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
        # `box` gain is pushed higher than the Ultralytics default (7.5)
        # to emphasise localisation of the tiny VisDrone bboxes.
        # `cls` upper bound widened to cover VisDrone's class imbalance
        # (pedestrian/car dominate, awning-tricycle is rare).
        # `dfl` applies to YOLO11 only — YOLO26 is DFL-free (see docstring).
        "box": tune.uniform(5.0, 12.0),
        "cls": tune.uniform(0.3, 2.0),
        "dfl": tune.uniform(1.0, 3.0),

        # --- Augmentation ---------------------------------------------
        # Small objects are easily destroyed by aggressive scale/shear,
        # so `shear` and `perspective` are deliberately omitted.
        "hsv_h":    tune.uniform(0.0, 0.03),
        "hsv_s":    tune.uniform(0.3, 0.9),
        "hsv_v":    tune.uniform(0.2, 0.7),
        "degrees":  tune.uniform(0.0, 10.0),
        "translate": tune.uniform(0.0, 0.2),
        "scale":    tune.uniform(0.2, 0.6),   # narrower than default
        # Aerial imagery has no natural vertical orientation — include
        # vertical flip (the Ultralytics default tuner also does).
        "flipud":   tune.uniform(0.0, 0.5),
        # Full 0-1 range so the tuner can converge on the standard 0.5
        # for laterally symmetric classes (car, bus, person, ...).
        "fliplr":   tune.uniform(0.0, 1.0),
        "mosaic":   tune.uniform(0.8, 1.0),   # keep mosaic high
        "mixup":    tune.uniform(0.0, 0.2),
        "cutmix":   tune.uniform(0.0, 0.3),
        "copy_paste": tune.uniform(0.0, 0.3),
        # Epochs-before-end to shut mosaic off. Mosaic distortion hurts
        # final fine-tuning, especially for tiny objects, so exposing
        # this as a tuned param is one of the stronger small-object
        # levers available.
        "close_mosaic": tune.randint(5, 15),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ray Tune hyperparameter search for YOLO on VisDrone"
    )
    p.add_argument(
        "--models", nargs="+", default=list(ALL_MODELS.keys()),
        choices=list(ALL_MODELS.keys()),
        help="Which models to tune (default: all four)",
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
        help="Number of Ray Tune trials to sample per model (default: 10)",
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
        help="Root directory for tuning results (default: runs/tune)",
    )
    p.add_argument(
        "--name-prefix", default="visdrone_raytune",
        help="Experiment name prefix; each model gets '<prefix>_<model>' "
             "(default: visdrone_raytune)",
    )
    p.add_argument(
        "--default-space", action="store_true",
        help="Use Ultralytics' full default 28-parameter search space "
             "instead of the VisDrone-tuned one",
    )
    return p.parse_args()


def tune_one_model(
    model_key: str,
    weights: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict | None:
    """
    Run a Ray Tune sweep for a single model.

    Returns a summary dict (model, name, best metrics + cfg path) on
    success, or None if the sweep failed.
    """
    from ultralytics import YOLO

    exp_name = f"{args.name_prefix}_{model_key}"

    print("\n" + "=" * 72)
    print(f" [{model_key}]  Ray Tune sweep → {exp_name}")
    print("=" * 72)
    print(f"  Weights      : {weights}")
    print(f"  Iterations   : {args.iterations}")
    print(f"  Epochs/trial : {args.epochs} (grace={args.grace_period})")
    print(f"  Image size   : {args.imgsz}")
    print(f"  GPU/trial    : {args.gpu_per_trial if args.gpu_per_trial else 'auto'}")

    model = YOLO(weights)

    tune_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        iterations=args.iterations,
        grace_period=args.grace_period,
        use_ray=True,
        project=str(output_dir),
        name=exp_name,
    )
    if args.gpu_per_trial is not None:
        tune_kwargs["gpu_per_trial"] = args.gpu_per_trial
    if not args.default_space:
        tune_kwargs["space"] = build_search_space()

    try:
        result_grid = model.tune(**tune_kwargs)
    except Exception as e:
        print(f"  ✗ Tuning failed for {model_key}: {e}")
        traceback.print_exc()
        return None

    # --- Extract & persist best trial -----------------------------------
    summary = {"model": model_key, "name": exp_name, "weights": weights}
    try:
        best_result = result_grid.get_best_result(
            metric="metrics/mAP50-95(B)", mode="max"
        )
        best_cfg = best_result.config
        best_metrics = best_result.metrics or {}

        print(f"\n  Best trial config for {model_key}:")
        for k, v in sorted(best_cfg.items()):
            print(f"    {k:20s} = {v}")

        map5095 = best_metrics.get("metrics/mAP50-95(B)")
        map50 = best_metrics.get("metrics/mAP50(B)")
        if map5095 is not None:
            print(f"\n  mAP50-95 : {map5095:.4f}")
        if map50 is not None:
            print(f"  mAP50    : {map50:.4f}")

        best_cfg_path = output_dir / f"{exp_name}_best_hyperparameters.json"
        best_cfg_path.write_text(json.dumps(best_cfg, indent=2, default=str))
        print(f"\n  Best hyperparameters saved to: {best_cfg_path}")

        summary["best_cfg_path"] = str(best_cfg_path)
        summary["mAP50-95"] = map5095
        summary["mAP50"] = map50
    except Exception as e:
        print(f"  Could not extract best trial for {model_key}: {e}")
        print("  Inspect the ResultGrid manually via the saved Ray experiment.")

    return summary


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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Ray Tune × Ultralytics YOLO × VisDrone")
    print("=" * 72)
    print(f"  Models       : {', '.join(args.models)}")
    print(f"  Data         : {args.data}")
    print(f"  Iterations   : {args.iterations}  (per model)")
    print(f"  Epochs/trial : {args.epochs} (grace={args.grace_period})")
    print(f"  Image size   : {args.imgsz}")
    print(f"  GPU/trial    : {args.gpu_per_trial if args.gpu_per_trial else 'auto'}")
    print(f"  Output       : {output_dir}")
    print(f"  Search space : {'default (28 params)' if args.default_space else 'VisDrone-tuned'}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Run one Ray Tune sweep per model.
    # Each sweep is independent – a failure in one does not abort the
    # rest; results are collected into a summary printed at the end.
    # ------------------------------------------------------------------
    summaries: list[dict] = []
    for model_key in args.models:
        weights = ALL_MODELS[model_key]
        summary = tune_one_model(model_key, weights, args, output_dir)
        if summary is not None:
            summaries.append(summary)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" TUNING COMPLETE")
    print("=" * 72)
    if summaries:
        print(f"\n  {'Model':<10} {'mAP50-95':>10} {'mAP50':>10}  Best hyperparameters")
        print("  " + "-" * 68)
        for s in summaries:
            map5095 = s.get("mAP50-95")
            map50 = s.get("mAP50")
            map5095_str = f"{map5095:.4f}" if map5095 is not None else "   -  "
            map50_str = f"{map50:.4f}" if map50 is not None else "   -  "
            cfg_path = s.get("best_cfg_path", "(unavailable)")
            print(f"  {s['model']:<10} {map5095_str:>10} {map50_str:>10}  {cfg_path}")
    else:
        print("\n  No successful tuning runs – see errors above.")

    # ------------------------------------------------------------------
    # TensorBoard instructions
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(" TensorBoard")
    print("-" * 72)
    print("  Ultralytics writes TFEvent logs for every trial of every model.")
    print("  Compare all sweeps in one UI with:")
    print(f"    uv run tensorboard --logdir {output_dir}")
    print("  Or view Ray Tune's own metrics (default ~/ray_results):")
    print("    uv run tensorboard --logdir ~/ray_results")
    print(
        "\n  Next step – re-train each model with its best hyperparameters\n"
        "  by passing the *_best_hyperparameters.json values into\n"
        "  train_and_export.py.\n"
    )


if __name__ == "__main__":
    main()
