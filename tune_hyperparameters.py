#!/usr/bin/env python3
"""
Hyperparameter Tuning for YOLO on VisDrone with Ray Tune + Optuna
========================================================================

In-house Ray Tune driver that runs one **OptunaSearch** (TPE) sweep
per model, with an ASHA scheduler for early stopping. The YOLO26
sweeps are **warm-started** by seeding Optuna's
``points_to_evaluate`` with the official YOLO26 training recipe
(see https://docs.ultralytics.com/guides/yolo26-training-recipe/),
so trial #0 is guaranteed to be the published baseline and the
remaining budget refines around it.

Why we do NOT use ``model.tune(use_ray=True)``:
    Ultralytics' built-in Ray Tune integration
    (``ultralytics/utils/tuner.py::run_ray_tune``) hard-codes Ray's
    default ``BasicVariantGenerator`` (random search) and does not
    forward a ``search_alg`` argument. To use Optuna + recipe
    warm-starting we therefore build the ``tune.Tuner`` ourselves and
    call ``YOLO(weights).train(**config)`` from our own trainable.

Model-family-aware search spaces:
    * YOLO11n / YOLO11s → VisDrone-audited YOLO11 space.
    * YOLO26n           → space bracketed around the official YOLO26n
                          training recipe (DFL-heavy, aggressive scale).
    * YOLO26s           → space bracketed around the official YOLO26s
                          training recipe (box-heavy, gentle LR decay,
                          no rotation / shear).

    YOLO26 variants get distinct spaces because the nano and small
    recipes differ sharply — see ``_visdrone_yolo26n_space`` and
    ``_visdrone_yolo26s_space`` below.

Ray Tune uses the ASHA scheduler – trials that underperform are
stopped early, so ``--iterations`` can be set relatively high.

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
    # Default: tune all 4 models, 10 trials each, 30 epochs, grace 10,
    # OptunaSearch TPE + YOLO26 recipe warm-start
    uv run python tune_hyperparameters.py

    # Subset of models, more trials, 1 GPU per trial
    uv run python tune_hyperparameters.py \\
        --models yolo11s yolo26s --iterations 30 --gpu-per-trial 1

    # Fall back to plain random search (no Optuna dependency)
    uv run python tune_hyperparameters.py --search-algo random

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

# Metric both the ASHA scheduler and OptunaSearch optimise.
# Must match the key Ultralytics reports in ``trainer.metrics`` after
# each validation pass.
_METRIC_NAME = "metrics/mAP50-95(B)"
_METRIC_MODE = "max"


def _visdrone_yolo11_space():
    """
    VisDrone-focused hyperparameter search space for YOLO11.

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


def _visdrone_yolo26n_space():
    """
    VisDrone × YOLO26n search space.

    Bracketed around the official YOLO26 training recipe values for the
    nano variant (see https://docs.ultralytics.com/guides/yolo26-training-recipe/).
    YOLO26 uses the new MuSGD optimiser and Small-Target-Aware Label
    Assignment (STAL), which tolerates more aggressive geometric
    augmentation than YOLO11 — so this space allows ``shear`` (recipe
    uses 1.46) and a higher ``scale``. The nano recipe prioritises
    ``dfl`` heavily (recipe value: 9.04), so the DFL range is pushed
    way above YOLO11's.

    Recipe anchors for reference:
        lr0=0.0054, lrf=0.0495, momentum=0.947, weight_decay=0.00064,
        warmup_epochs=0.98, box=5.63, cls=0.56, dfl=9.04,
        mosaic=0.909, mixup=0.012, copy_paste=0.075,
        scale=0.562, degrees=1.11, shear=1.46, fliplr=0.606.
    """
    from ray import tune

    return {
        # --- Optimiser (MuSGD) ----------------------------------------
        "lr0":          tune.loguniform(1e-3, 1e-2),   # ~0.0054
        "lrf":          tune.uniform(0.02, 0.10),      # ~0.0495
        "momentum":     tune.uniform(0.92, 0.97),      # ~0.947
        "weight_decay": tune.uniform(1e-4, 1e-3),      # ~0.00064
        "warmup_epochs":   tune.uniform(0.0, 3.0),     # ~0.98
        "warmup_momentum": tune.uniform(0.5, 0.95),

        # --- Loss weights ---------------------------------------------
        # YOLO26n prioritises DFL: recipe value 9.04.
        "box": tune.uniform(4.0, 8.0),                 # ~5.63
        "cls": tune.uniform(0.3, 1.0),                 # ~0.56
        "dfl": tune.uniform(6.0, 12.0),                # ~9.04

        # --- Augmentation ---------------------------------------------
        # STAL tolerates aggressive geometry: shear is allowed.
        "hsv_h":    tune.uniform(0.0, 0.03),
        "hsv_s":    tune.uniform(0.3, 0.9),
        "hsv_v":    tune.uniform(0.2, 0.7),
        "degrees":  tune.uniform(0.0, 3.0),            # ~1.11
        "translate": tune.uniform(0.0, 0.2),
        "scale":    tune.uniform(0.4, 0.7),            # ~0.562
        "shear":    tune.uniform(0.0, 3.0),            # ~1.46
        # Aerial imagery — keep flipud on.
        "flipud":   tune.uniform(0.0, 0.5),
        "fliplr":   tune.uniform(0.4, 0.8),            # ~0.606
        "mosaic":   tune.uniform(0.85, 1.0),           # ~0.909
        "mixup":    tune.uniform(0.0, 0.05),           # ~0.012
        "copy_paste": tune.uniform(0.0, 0.2),          # ~0.075
        "close_mosaic": tune.randint(5, 15),           # recipe: 10
    }


def _visdrone_yolo26s_space():
    """
    VisDrone × YOLO26s search space.

    Bracketed around the official YOLO26 training recipe values for the
    small variant (see https://docs.ultralytics.com/guides/yolo26-training-recipe/).
    The S/M/L/X recipes differ sharply from N — much lower ``lr0``,
    gentler LR decay (high ``lrf``), higher ``box`` loss gain, and
    de-emphasised ``dfl``. The recipe also zeroes out rotation and
    shear for S, so those ranges are correspondingly narrow.

    Recipe anchors for reference:
        lr0=0.00038, lrf=0.882, momentum=0.948, weight_decay=0.00027,
        warmup_epochs=0.99, box=9.83, cls=0.65, dfl=0.96,
        mosaic=0.992, mixup=0.05, copy_paste=0.304,
        scale=0.9, degrees=0.0, shear=0.0, fliplr=0.304.
    """
    from ray import tune

    return {
        # --- Optimiser (MuSGD) ----------------------------------------
        "lr0":          tune.loguniform(1e-4, 1e-3),   # ~0.00038
        "lrf":          tune.uniform(0.5, 1.0),        # ~0.882
        "momentum":     tune.uniform(0.92, 0.97),      # ~0.948
        "weight_decay": tune.uniform(1e-4, 1e-3),      # ~0.00027
        "warmup_epochs":   tune.uniform(0.0, 3.0),     # ~0.99
        "warmup_momentum": tune.uniform(0.5, 0.95),

        # --- Loss weights ---------------------------------------------
        # S/M/L/X de-emphasise DFL (recipe: 0.96) and push box (9.83).
        "box": tune.uniform(7.0, 13.0),                # ~9.83
        "cls": tune.uniform(0.3, 1.2),                 # ~0.65
        "dfl": tune.uniform(0.5, 2.0),                 # ~0.96

        # --- Augmentation ---------------------------------------------
        # Recipe zeros out rotation and shear for S — keep them near 0.
        "hsv_h":    tune.uniform(0.0, 0.03),
        "hsv_s":    tune.uniform(0.3, 0.9),
        "hsv_v":    tune.uniform(0.2, 0.7),
        "degrees":  tune.uniform(0.0, 1.0),            # ~0.0
        "translate": tune.uniform(0.0, 0.2),
        "scale":    tune.uniform(0.6, 0.9),            # ~0.9
        "shear":    tune.uniform(0.0, 0.5),            # ~0.0
        # Aerial imagery — keep flipud on.
        "flipud":   tune.uniform(0.0, 0.5),
        "fliplr":   tune.uniform(0.2, 0.5),            # ~0.304
        "mosaic":   tune.uniform(0.9, 1.0),            # ~0.992
        "mixup":    tune.uniform(0.0, 0.1),            # ~0.05
        "copy_paste": tune.uniform(0.1, 0.4),          # ~0.304
        "close_mosaic": tune.randint(5, 15),           # recipe: 10
    }


def build_search_space(model_key: str):
    """
    Return a Ray Tune search space tailored to ``model_key``.

    Three families are supported:

      * ``yolo11n`` / ``yolo11s`` → VisDrone-audited YOLO11 space.
      * ``yolo26n``               → space bracketed around the YOLO26n
                                    training recipe (DFL-heavy).
      * ``yolo26s``               → space bracketed around the YOLO26s
                                    training recipe (box-heavy,
                                    gentle LR decay).

    YOLO26 variants get distinct spaces because the official training
    recipe (https://docs.ultralytics.com/guides/yolo26-training-recipe/)
    differs sharply between the nano and small variants in ``lr0``,
    ``lrf``, ``box`` / ``dfl`` gain, and geometric augmentation. YOLO26
    also uses Small-Target-Aware Label Assignment (STAL) which tolerates
    more aggressive ``scale`` / ``shear`` than YOLO11.
    """
    if model_key == "yolo26n":
        return _visdrone_yolo26n_space()
    if model_key == "yolo26s":
        return _visdrone_yolo26s_space()
    # yolo11n, yolo11s, and any unknown key fall through to the
    # VisDrone-audited YOLO11 baseline.
    return _visdrone_yolo11_space()


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
        "--search-algo",
        choices=["optuna", "random"],
        default="optuna",
        help="Ray Tune search algorithm. 'optuna' uses OptunaSearch "
             "(TPE) with YOLO26 recipe warm-starting via "
             "points_to_evaluate; 'random' uses Ray's default "
             "BasicVariantGenerator. Default: optuna",
    )
    return p.parse_args()


def _recipe_seed_for(model_key: str, space: dict) -> list[dict] | None:
    """
    Return the YOLO26 training recipe as an ``OptunaSearch``
    ``points_to_evaluate`` seed, filtered to the keys that actually
    exist in ``space``. Returns ``None`` for YOLO11 (no recipe warm
    start) or if the recipe import fails.

    This is what makes the YOLO26 sweeps *refine* the published recipe
    rather than explore the space from scratch — trial #0 is always
    the exact recipe configuration.
    """
    # Import lazily + defensively: train_and_export lives in the repo
    # root next to this script, but Ray Tune workers may have a
    # different cwd, so we also ensure our directory is on sys.path.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from train_and_export import MODEL_TRAIN_OVERRIDES
    except Exception:
        return None

    recipe = MODEL_TRAIN_OVERRIDES.get(model_key)
    if not recipe:
        return None
    seed = {k: v for k, v in recipe.items() if k in space}
    return [seed] if seed else None


def _yolo_tune_trainable(
    config: dict,
    *,
    weights: str,
    base_kwargs: dict,
) -> None:
    """
    Ray Tune trainable: trains a YOLO model with a sampled
    ``config`` and reports per-epoch validation metrics back to Ray
    Tune so the ASHA scheduler can prune under-performing trials.

    Must be defined at module level so Ray can pickle it when
    dispatching to remote actors. Extra kwargs (``weights``,
    ``base_kwargs``) are injected via ``tune.with_parameters`` in
    ``tune_one_model``.
    """
    from ultralytics import YOLO
    from ray import train as ray_train

    # Namespace each trial's Ultralytics run directory by the Ray
    # Tune trial ID so concurrent trials don't stomp on each other's
    # project dir (important for fractional --gpu-per-trial).
    try:
        trial_id = ray_train.get_context().get_trial_id()
    except Exception:
        trial_id = None

    train_kwargs = dict(base_kwargs)
    train_kwargs.update(config)
    if trial_id:
        base_name = train_kwargs.get("name", "trial")
        train_kwargs["name"] = f"{base_name}_{trial_id}"

    model = YOLO(weights)

    def _on_fit_epoch_end(trainer):
        raw = getattr(trainer, "metrics", None) or {}
        metrics: dict[str, float] = {}
        for k, v in raw.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                continue
        metrics["epoch"] = int(getattr(trainer, "epoch", 0))
        if metrics:
            ray_train.report(metrics)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.train(**train_kwargs)


def _make_run_config(name: str, output_dir: Path):
    """
    Build a Ray ``RunConfig`` that works across Ray 2.x versions
    (the class moved from ``ray.tune`` to ``ray.train`` in 2.7+).
    ``storage_path`` must be absolute for Ray.
    """
    storage = str(output_dir.resolve())
    try:
        from ray.train import RunConfig  # type: ignore
    except ImportError:
        from ray.tune import RunConfig  # type: ignore
    return RunConfig(name=name, storage_path=storage)


def _resolve_gpus_per_trial(gpu_per_trial_arg: float | None) -> float:
    """Auto-detect GPU availability if ``--gpu-per-trial`` is not set."""
    if gpu_per_trial_arg is not None:
        return float(gpu_per_trial_arg)
    try:
        import torch
        return 1.0 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def tune_one_model(
    model_key: str,
    weights: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict | None:
    """
    Run an in-house Ray Tune sweep (OptunaSearch TPE + ASHA) for a
    single model, warm-starting YOLO26 variants from the official
    training recipe.

    Returns a summary dict (model, name, best metrics + cfg path) on
    success, or None if the sweep failed.
    """
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    exp_name = f"{args.name_prefix}_{model_key}"
    space = build_search_space(model_key)

    # ---- Search algorithm ---------------------------------------------
    search_alg = None
    search_desc = "Random (BasicVariantGenerator)"
    if args.search_algo == "optuna":
        try:
            from ray.tune.search.optuna import OptunaSearch
        except ImportError:
            print(
                "ERROR: optuna is not installed; install it via\n"
                "    uv sync --extra tune\n"
                "or re-run with --search-algo random to use plain random search.",
                file=sys.stderr,
            )
            return None
        points_to_evaluate = _recipe_seed_for(model_key, space)
        search_alg = OptunaSearch(
            metric=_METRIC_NAME,
            mode=_METRIC_MODE,
            points_to_evaluate=points_to_evaluate,
        )
        if points_to_evaluate:
            search_desc = (
                f"OptunaSearch (TPE, seeded with YOLO26 recipe, "
                f"{len(points_to_evaluate[0])} anchors)"
            )
        else:
            search_desc = "OptunaSearch (TPE, cold start)"

    # ---- Scheduler -----------------------------------------------------
    # ASHA owns early stopping. max_t / grace_period are expressed in
    # training_iteration units, which equal epochs thanks to our
    # per-epoch `ray_train.report` in _yolo_tune_trainable.
    scheduler = ASHAScheduler(
        metric=_METRIC_NAME,
        mode=_METRIC_MODE,
        max_t=args.epochs,
        grace_period=args.grace_period,
        reduction_factor=3,
    )

    # ---- Resources -----------------------------------------------------
    gpus_per_trial = _resolve_gpus_per_trial(args.gpu_per_trial)

    # ---- Base Ultralytics train kwargs shared by all trials -----------
    base_train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(output_dir),
        name=f"{exp_name}_trial",
        exist_ok=True,
        verbose=False,
    )

    print("\n" + "=" * 72)
    print(f" [{model_key}]  Ray Tune sweep → {exp_name}")
    print("=" * 72)
    print(f"  Weights      : {weights}")
    print(f"  Iterations   : {args.iterations}")
    print(f"  Epochs/trial : {args.epochs} (grace={args.grace_period})")
    print(f"  Image size   : {args.imgsz}")
    print(f"  GPU/trial    : {gpus_per_trial}")
    print(f"  Search algo  : {search_desc}")

    trainable = tune.with_parameters(
        _yolo_tune_trainable,
        weights=weights,
        base_kwargs=base_train_kwargs,
    )
    trainable = tune.with_resources(
        trainable, {"cpu": 1, "gpu": gpus_per_trial}
    )

    tuner = tune.Tuner(
        trainable,
        param_space=space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.iterations,
        ),
        run_config=_make_run_config(exp_name, output_dir),
    )

    try:
        result_grid = tuner.fit()
    except Exception as e:
        print(f"  ✗ Tuning failed for {model_key}: {e}")
        traceback.print_exc()
        return None

    # ---- Extract & persist best trial ---------------------------------
    summary = {"model": model_key, "name": exp_name, "weights": weights}
    try:
        best_result = result_grid.get_best_result(
            metric=_METRIC_NAME, mode=_METRIC_MODE
        )
        best_cfg = best_result.config
        best_metrics = best_result.metrics or {}

        print(f"\n  Best trial config for {model_key}:")
        for k, v in sorted(best_cfg.items()):
            print(f"    {k:20s} = {v}")

        map5095 = best_metrics.get(_METRIC_NAME)
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
            "    pip install 'ray[tune]' 'optuna>=3.4' tensorboard",
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
    print(f"  Search algo  : {args.search_algo}")
    print(f"  Search space : VisDrone + model-family tuned")
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
