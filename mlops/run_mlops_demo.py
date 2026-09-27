"""End-to-end demo: train Phase 7's real XGBoost model, log it to
MLflow (Phase 8), register it, attempt promotion through the real gate,
and show the retrieval function reconstructing everything about it.

Run with: python -m mlops.run_mlops_demo
"""
from __future__ import annotations

import json

from ml.data.fetch_price_history import load_cached
from ml.dataset import build_dataset
from ml.train import train_and_evaluate
from ml.walk_forward import run_walk_forward_evaluation
from mlops.config import MODEL_NAME
from mlops.registry import get_model_info, promote_if_beats_baseline, register_model
from mlops.tracking import log_training_run
from mlops.versioning import compute_dataset_version, compute_feature_version


def main() -> None:
    prices = load_cached()
    dataset = build_dataset(prices)
    result = train_and_evaluate(dataset)

    print("=== VERSIONING ===")
    dv = compute_dataset_version(prices)
    fv = compute_feature_version()
    print(f"Dataset version: {dv.as_dict()}")
    print(f"Feature version hash: {fv}")

    print("\n=== WALK-FORWARD CROSS-VALIDATION (supplementary diagnostic) ===")
    wf_folds = run_walk_forward_evaluation(dataset, n_folds=5)
    for f in wf_folds:
        print(
            f"fold {f.fold}: train_end={f.train_end} test=[{f.test_start}..{f.test_end}] "
            f"n_train={f.train_size} n_test={f.test_size} "
            f"label_rate_train={f.label_rate_train:.3f} label_rate_test={f.label_rate_test:.3f} "
            f"baseline_f1={f.baseline_f1:.3f} model_f1={f.model_f1:.3f} "
            f"delta_f1={f.delta_f1:+.3f} beats_baseline={f.model_beats_baseline}"
        )

    print("\n=== LOGGING TRAINING RUN TO MLFLOW (incl. walk-forward folds) ===")
    run_id = log_training_run(prices, result, walk_forward_folds=wf_folds)
    print(f"Logged run_id: {run_id}")
    print(f"Baseline metrics logged: {result['baseline_metrics']}")
    print(f"Model metrics logged: {result['model_metrics']}")
    print(f"Deltas logged: {result['comparison']['deltas']}")

    print("\n=== REGISTERING MODEL ===")
    version = register_model(run_id, model_name=MODEL_NAME)
    print(f"Registered '{MODEL_NAME}' version {version}")

    print("\n=== ATTEMPTING PROMOTION ===")
    promotion = promote_if_beats_baseline(MODEL_NAME, version)
    print(f"Promoted: {promotion.promoted}")
    print(f"Reason: {promotion.reason}")

    print("\n=== RETRIEVAL (what Phase 10's Streamlit MLOps page will call) ===")
    info = get_model_info(MODEL_NAME, alias_or_version=version)
    printable = {k: v for k, v in info.items() if k != "model"}
    print(json.dumps(printable, indent=2, default=str))
    print(f"\nLoaded model object: {type(info['model'])}")

    print("\n=== SUMMARY ===")
    if promotion.promoted:
        print(f"Model version {version} is now the '{MODEL_NAME}' production alias.")
    else:
        print(
            f"Model version {version} was NOT promoted -- consistent with Phase 7's "
            f"own honest finding that this XGBoost model does not beat the persistence "
            f"baseline on the held-out test set. The promotion gate applied to this "
            f"project's real model correctly reaches the same conclusion Phase 7 "
            f"already reported: no production model is set from this run, and any "
            f"previously-promoted production alias (there is none yet) would remain "
            f"untouched."
        )


if __name__ == "__main__":
    main()
