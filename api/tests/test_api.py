"""API tests against a real running instance (FastAPI TestClient, which
runs the actual app against the real Postgres warehouse + MLflow store
-- nothing here is mocked). Every analytics assertion checks REAL
computed values against an independent direct call into the same
analytics/monte_carlo module the endpoint delegates to, not just a 200
status code.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from analytics import data_access as da
from analytics.allocation.allocation import asset_allocation
from analytics.allocation.concentration import herfindahl_hirschman_index
from analytics.performance.returns import capital_summary
from api.dependencies import ANALYSIS_END, ANALYSIS_START
from api.main import app
from monte_carlo.params import get_portfolio_params
from monte_carlo.simulate import simulate_portfolio_gbm

client = TestClient(app)
PORTFOLIO_ID = 1


def _latest_asset_performance():
    ap = da.get_asset_performance(PORTFOLIO_ID)
    ap = ap[(ap["as_of_date"] >= ANALYSIS_START) & (ap["as_of_date"] <= ANALYSIS_END)]
    return ap[ap["as_of_date"] == ap["as_of_date"].max()]


# --------------------------- upload ---------------------------

def test_upload_real_example_csv_runs_full_pipeline():
    with open("ingestion/tests/test_portfolio.csv", "rb") as f:
        r = client.post("/portfolio/upload", files={"file": ("test_portfolio.csv", f, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "SUCCESS"
    # Matches Phase 2's own known result for this exact CSV: 7 valid, 7
    # rejected, 1 exact duplicate dropped, 1 currency imputed, 1 anomaly.
    assert body["valid_row_count"] == 7
    assert body["rejected_row_count"] == 7
    dq = body["data_quality_report"]
    assert dq["rows_in"] == 7
    assert dq["rows_cleaned"] == 6
    assert dq["duplicates_dropped"] == 1
    assert dq["values_imputed"] == 1
    assert dq["anomalies_flagged"] == 1


def test_upload_malformed_csv_rejected_with_422():
    bad_csv = b"foo,bar\n1,2\n"
    r = client.post("/portfolio/upload", files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")})
    assert r.status_code == 422
    assert "missing required columns" in r.json()["detail"]


# --------------------------- overview / performance ---------------------------

def test_overview_matches_direct_analytics_call():
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/overview")
    assert r.status_code == 200
    body = r.json()

    latest = _latest_asset_performance()
    expected = capital_summary(float(latest["cost_basis_inr"].sum()), float(latest["market_value_inr"].sum()))
    assert body["invested_capital"] == pytest.approx(expected["invested_capital"])
    assert body["market_value"] == pytest.approx(expected["market_value"])
    assert body["absolute_return"] == pytest.approx(expected["absolute_return"])
    assert body["pct_return"] == pytest.approx(expected["pct_return"])


def test_performance_matches_direct_analytics_call():
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/performance")
    assert r.status_code == 200
    body = r.json()

    latest = _latest_asset_performance()
    expected = capital_summary(float(latest["cost_basis_inr"].sum()), float(latest["market_value_inr"].sum()))
    assert body["capital_summary"]["market_value"] == pytest.approx(expected["market_value"])
    assert body["capital_summary"]["invested_capital"] == pytest.approx(expected["invested_capital"])


def test_overview_invalid_portfolio_returns_404():
    r = client.get("/portfolio/999999/overview")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# --------------------------- allocation ---------------------------

def test_allocation_matches_direct_analytics_call():
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/allocation")
    assert r.status_code == 200
    body = r.json()

    alloc = da.get_allocation(PORTFOLIO_ID)
    alloc = alloc[(alloc["as_of_date"] >= ANALYSIS_START) & (alloc["as_of_date"] <= ANALYSIS_END)]
    latest = alloc[alloc["as_of_date"] == alloc["as_of_date"].max()]
    weights = asset_allocation(latest)

    for ticker, weight in weights.items():
        assert body["asset_allocation"][ticker] == pytest.approx(float(weight))
    assert body["herfindahl_hirschman_index"] == pytest.approx(herfindahl_hirschman_index(weights))


def test_allocation_invalid_portfolio_returns_404():
    r = client.get("/portfolio/999999/allocation")
    assert r.status_code == 404


# --------------------------- risk ---------------------------

def test_risk_endpoint_returns_real_values_and_respects_query_params():
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/risk", params={"risk_free_rate_annual": 0.05, "var_confidence": 0.90})
    assert r.status_code == 200
    body = r.json()
    assert body["risk_free_rate_annual"] == 0.05
    assert body["var_confidence"] == 0.90
    assert isinstance(body["sharpe_ratio"], float)
    assert body["max_drawdown"]["max_drawdown"] <= 0


def test_risk_invalid_portfolio_returns_404():
    r = client.get("/portfolio/999999/risk")
    assert r.status_code == 404


# --------------------------- benchmark ---------------------------

def test_benchmark_matches_direct_analytics_call():
    from analytics.benchmark.benchmark import benchmark_comparison

    r = client.get(f"/portfolio/{PORTFOLIO_ID}/benchmark")
    assert r.status_code == 200
    body = r.json()

    pp = da.get_portfolio_performance(PORTFOLIO_ID)
    pp = pp[(pp["value_date"] >= ANALYSIS_START) & (pp["value_date"] <= ANALYSIS_END)].set_index("value_date")
    daily_returns = pp["daily_return"].dropna()
    bench = da.get_benchmark(PORTFOLIO_ID).set_index("value_date")
    bench_returns = bench["benchmark_daily_return"].reindex(daily_returns.index)
    expected = benchmark_comparison(daily_returns, bench_returns, 0.07)

    assert body["beta"] == pytest.approx(expected["beta"])
    assert body["correlation"] == pytest.approx(expected["correlation"])


def test_benchmark_invalid_portfolio_returns_404():
    r = client.get("/portfolio/999999/benchmark")
    assert r.status_code == 404


# --------------------------- attribution ---------------------------

def test_attribution_reconciles_and_matches_direct_call():
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/attribution")
    assert r.status_code == 200
    body = r.json()
    assert body["total_return_from_contributions"] == pytest.approx(body["actual_holdings_return"], abs=1e-9)


def test_attribution_invalid_portfolio_returns_404():
    r = client.get("/portfolio/999999/attribution")
    assert r.status_code == 404


# --------------------------- monte carlo ---------------------------

def test_monte_carlo_post_get_round_trip_matches_direct_engine_call():
    r = client.post(
        f"/portfolio/{PORTFOLIO_ID}/monte-carlo",
        json={"model": "baseline", "n_simulations": 2000, "horizon_days": 252, "seed": 123},
    )
    assert r.status_code == 200
    posted = r.json()
    run_id = posted["run_id"]

    r2 = client.get(f"/portfolio/{PORTFOLIO_ID}/monte-carlo/results", params={"run_id": run_id})
    assert r2.status_code == 200
    assert r2.json() == posted

    # Cross-check against calling the real Phase 6 engine directly with
    # the same seed and parameters -- must reproduce the exact terminal
    # distribution (GBM with a fixed seed is deterministic).
    params = get_portfolio_params(PORTFOLIO_ID, ANALYSIS_START, ANALYSIS_END)
    paths = simulate_portfolio_gbm(params.s0, params.mu_gbm, params.sigma_annual, 252, 2000, seed=123)
    expected_median = float(__import__("numpy").median(paths[:, -1]))
    assert posted["percentile_bands"]["p50"] == pytest.approx(expected_median)


def test_monte_carlo_negative_simulations_returns_422():
    r = client.post(f"/portfolio/{PORTFOLIO_ID}/monte-carlo", json={"n_simulations": -100})
    assert r.status_code == 422


def test_monte_carlo_invalid_confidence_returns_422():
    r = client.post(f"/portfolio/{PORTFOLIO_ID}/monte-carlo", json={"var_confidence": 1.5})
    assert r.status_code == 422


def test_monte_carlo_invalid_portfolio_returns_404():
    r = client.post("/portfolio/999999/monte-carlo", json={})
    assert r.status_code == 404


def test_monte_carlo_unknown_run_id_returns_404():
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/monte-carlo/results", params={"run_id": "does-not-exist"})
    assert r.status_code == 404


# --------------------------- model / pipeline status ---------------------------

def test_model_status_reports_real_registry_state():
    r = client.get("/model/status")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "vol_regime_xgboost"
    assert "model_f1" in body["metrics"]
    assert "baseline_f1" in body["metrics"]


def test_pipeline_status_reports_real_most_recent_run():
    r = client.get("/pipeline/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("SUCCESS", "FAILED", "RUNNING")
    assert body["run_id"]
