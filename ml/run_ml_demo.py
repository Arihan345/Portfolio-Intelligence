"""End-to-end demo: fetch extended price history -> build dataset ->
train XGBoost -> evaluate against the persistence baseline on the real
chronological test set.

Run with: python -m ml.run_ml_demo
"""
from __future__ import annotations

import json

from ml.data.fetch_price_history import load_cached
from ml.dataset import build_dataset
from ml.train import train_and_evaluate


def main() -> None:
    prices = load_cached()
    print(f"Loaded {len(prices)} rows of price history for {prices['ticker'].nunique()} tickers")
    print(prices.groupby("ticker")["date"].agg(["min", "max", "count"]).to_string())

    dataset = build_dataset(prices)
    print(f"\nBuilt dataset: {len(dataset)} rows (after dropping warm-up/tail NaNs)")
    print("Label balance:")
    print(dataset["label"].value_counts(normalize=True).to_string())

    result = train_and_evaluate(dataset)

    print(f"\nChronological split sizes: train={result['train_size']}, "
          f"val={result['val_size']}, test={result['test_size']}")
    print(f"Train date range: {result['train_dates_range'][0]} -> {result['train_dates_range'][1]}")
    print(f"Test date range:  {result['test_dates_range'][0]} -> {result['test_dates_range'][1]}")
    print(f"scale_pos_weight used: {result['scale_pos_weight']:.3f}")
    print(f"\nPositive-label rate by split (a big gap here signals a real volatility-regime "
          f"shift across the chronological boundary, not a bug):")
    for name in ("train", "val", "test"):
        print(f"  {name}: {result['label_rate_by_split'][name]:.2%}")

    print("\n--- Baseline (persistence of current vol regime) ---")
    print(json.dumps(result["baseline_metrics"], indent=2))

    print("\n--- XGBoost ---")
    print(json.dumps(result["model_metrics"], indent=2))

    print("\n--- Comparison (model - baseline) ---")
    print(json.dumps(result["comparison"], indent=2))

    verdict = result["comparison"]["model_beats_baseline_on_f1_and_pr_auc"]
    print(f"\nVERDICT: XGBoost {'DOES' if verdict else 'DOES NOT'} meaningfully beat "
          f"the baseline on F1 and PR-AUC.")

    print("\n--- Feature importances (XGBoost, gain-based) ---")
    print(result["feature_importances"].to_string())

    print("\nReminder (per this project's dev rules): predictive accuracy on this "
          "classification task does not by itself imply investment usefulness -- "
          "it says nothing about transaction costs, whether the signal is tradeable "
          "at the frequency implied, or whether the historical regime relationships "
          "will hold going forward.")


if __name__ == "__main__":
    main()
