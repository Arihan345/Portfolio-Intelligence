"""MLflow store configuration.

The tracking store and artifact root are deliberately placed OUTSIDE
this repo's Desktop-synced directory, at ~/.mlflow-store/. This project
has repeatedly hit multi-minute stalls this session from macOS
iCloud Drive fighting to sync large trees of small files written
inside Desktop (the .venv rebuild, dbt's target/ directory, and plain
source-file reads all hit this). An MLflow run writes many small
artifact files per run (model binaries, metric/param files) -- exactly
the pattern that triggered those stalls -- so the store lives outside
iCloud's reach from the start rather than repeating that mistake a
fourth time.

A SQLite backend store (not the default plain file store) is used
because the Model Registry (Phase 8's registration/promotion/alias
features) requires a database-backed backend store; MLflow's file
store does not support it.
"""
from __future__ import annotations

from pathlib import Path

STORE_ROOT = Path.home() / ".mlflow-store" / "portfolio-analysis"
TRACKING_URI = f"sqlite:///{STORE_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = str(STORE_ROOT / "artifacts")

EXPERIMENT_NAME = "portfolio-vol-regime-classifier"
MODEL_NAME = "vol_regime_xgboost"


def ensure_store_dirs() -> None:
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    Path(ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
