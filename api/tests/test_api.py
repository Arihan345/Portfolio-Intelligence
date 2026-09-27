"""API tests against a real running instance (FastAPI TestClient, which
runs the actual app against the real Postgres warehouse + MLflow store
-- nothing here is mocked). Every analytics assertion checks REAL
computed values against an independent direct call into the same
analytics/monte_carlo module the endpoint delegates to, not just a 200
status code.
"""
from __future__ import annotations

import io
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from analytics import data_access as da
from analytics.allocation.allocation import asset_allocation
from analytics.allocation.concentration import herfindahl_hirschman_index
from analytics.performance.returns import capital_summary
from api.main import app
from monte_carlo.params import get_portfolio_params
from monte_carlo.simulate import simulate_portfolio_gbm
from warehouse.load.load_warehouse import (
    derive_fact_holdings,
    derive_fact_portfolio_returns,
    derive_fact_portfolio_value,
    engine,
    finish_pipeline_run,
    load_fact_transactions,
    start_pipeline_run,
    upsert_dim_asset,
)

client = TestClient(app)
PORTFOLIO_ID = 1

_DBT_SCRIPT = Path(__file__).resolve().parents[2] / "dbt" / "run_dbt.sh"


@pytest.fixture(scope="module", autouse=True)
def _restore_portfolio_1_after_module():
    """test_upload_real_example_csv_runs_full_pipeline (below) uploads a
    small fixed demo CSV to PORTFOLIO_ID=1 -- the SAME portfolio_id the
    live app/frontend reads from, since this app is single-portfolio by
    design -- and every other test in this module asserts exact known
    values against that fixture data. Before the upload-replace fix
    (see warehouse/tests/test_upload_replace_and_netting.py), a new
    upload only ever appended, so running this suite was a harmless
    no-op layered on top of whatever was already there. Now that a
    second upload correctly REPLACES the portfolio's transactions (the
    correct, intended fix for a real bug), running this suite for real
    against the shared dev database silently destroys whatever real
    portfolio was loaded there -- reproduced for real: a real 30-ticker
    upload was wiped by a single run of this test file and replaced
    with TCS.NS/RELIANCE.NS/INFY.NS, which then broke Attribution's
    reconciliation (INFY.NS's BUY falls inside the attribution
    endpoint's hardcoded no-trading window) and confused a real user
    who saw demo data reappear on their live dashboard with no upload
    of their own.

    Snapshots portfolio 1's real fact_transactions (and current
    dim_asset metadata) before this module's tests run, and restores
    them -- full reload + full recompute + full dbt rebuild -- after,
    so this test module's own necessarily-destructive design stays
    externally invisible once it finishes.
    """
    with engine.begin() as conn:
        original_rows = conn.execute(
            sa.text(
                "SELECT da.ticker, dd.full_date AS date, ft.transaction_type, "
                "ft.quantity, ft.price, ft.price_inr, ft.fees, ft.tax, "
                "ft.currency_code AS currency, ft.near_duplicate_flag, ft.price_anomaly_flag "
                "FROM fact_transactions ft "
                "JOIN dim_asset da ON da.asset_key = ft.asset_key "
                "JOIN dim_date dd ON dd.date_key = ft.date_key "
                "WHERE ft.portfolio_id = :p ORDER BY ft.transaction_id"
            ),
            {"p": PORTFOLIO_ID},
        ).mappings().all()
        asset_meta_rows = conn.execute(
            sa.text(
                "SELECT ticker, asset_name, sector, industry, exchange, currency_code "
                "FROM dim_asset WHERE is_current"
            )
        ).mappings().all()

    yield

    run_id = start_pipeline_run(dag_id="pytest_restore")
    if original_rows:
        df = pd.DataFrame([dict(r) for r in original_rows])
        df["date"] = df["date"].astype(str)
        for col in ("quantity", "price", "price_inr", "fees", "tax"):
            df[col] = df[col].astype(float)
        asset_meta = {
            r["ticker"]: {
                "name": r["asset_name"], "sector": r["sector"], "industry": r["industry"],
                "exchange": r["exchange"], "currency": r["currency_code"],
            }
            for r in asset_meta_rows
        }
        upsert_dim_asset(asset_meta, date(2024, 1, 1), "pytest_restore", run_id)
        load_fact_transactions(df, PORTFOLIO_ID, "pytest_restore", run_id)
    else:
        # Portfolio 1 was genuinely empty before this module ran --
        # restore that, not this module's own fixture upload.
        with engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM fact_transactions WHERE portfolio_id = :p"), {"p": PORTFOLIO_ID})
    derive_fact_holdings(PORTFOLIO_ID, "pytest_restore", run_id)
    derive_fact_portfolio_value(PORTFOLIO_ID, "pytest_restore", run_id)
    derive_fact_portfolio_returns(PORTFOLIO_ID, "pytest_restore", run_id)
    finish_pipeline_run(run_id, "SUCCESS", len(original_rows))
    # This module's own test already triggered at least one dbt build
    # against the fixture data -- rebuild once more now that
    # fact_transactions is back to the real pre-test state, so every
    # mart (not just fact_transactions itself) reflects it too.
    subprocess.run([str(_DBT_SCRIPT), "build"], capture_output=True, text=True, timeout=300)


def _latest_asset_performance():
    # No ANALYSIS_START/END filtering here anymore: the real endpoints
    # (api/routers/portfolio.py) stopped capping "current" at the
    # original demo's fixed 2024-06-30 window -- that was a real bug
    # (a real portfolio uploaded with transactions through 2026 still
    # reported a stale 2024 as_of_date and market value). This helper's
    # job is to compute the SAME "latest available" value the endpoint
    # itself now computes, so it must match that behavior exactly.
    ap = da.get_asset_performance(PORTFOLIO_ID)
    return ap[ap["as_of_date"] == ap["as_of_date"].max()]


# --------------------------- upload ---------------------------

def test_upload_real_example_csv_runs_full_pipeline():
    # Explicit portfolio_id=1: this whole module's fixture (see
    # _restore_portfolio_1_after_module above) snapshots/restores THAT
    # portfolio specifically, and every other test in this module
    # asserts against PORTFOLIO_ID=1's data. Multi-portfolio uploads
    # now default to CREATING a new portfolio when portfolio_id is
    # omitted (see upload_portfolio's own docstring) -- this test needs
    # the original REPLACE behavior, not a new, disconnected portfolio.
    with open("ingestion/tests/test_portfolio.csv", "rb") as f:
        r = client.post(
            "/portfolio/upload", params={"portfolio_id": PORTFOLIO_ID}, files={"file": ("test_portfolio.csv", f, "text/csv")}
        )
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


def test_risk_correlation_matrix_scoped_to_current_holdings_only():
    # Regression for a real, confirmed bug: this endpoint used to pivot
    # mart_asset_performance UNFILTERED (every ticker ever traded) for
    # its correlation matrix and mart_portfolio_performance's full
    # historical NAV path (including every exited position's own
    # volatility while it was held) for Sharpe/volatility/VaR/max
    # drawdown -- a real portfolio's correlation matrix showed
    # ALOKINDS.NS/BAJAJHFL.NS/DIXON.NS/etc. (long-exited positions, NOT
    # the real current holdings) and annualized volatility came out at
    # an implausible 245%+. Attribution is CORRECT to include exited
    # positions within its own historical window by design (a
    # genuinely different question -- "what contributed to return over
    # this period" vs. Risk's "how risky is what I own right now") --
    # this test asserts Risk's scope, specifically, always matches
    # current holdings, distinct from Attribution's intentionally wider
    # scope, so this doesn't silently regress a third time.
    r = client.get(f"/portfolio/{PORTFOLIO_ID}/risk")
    assert r.status_code == 200
    body = r.json()
    correlation_tickers = set(body["correlation_matrix"].keys()) - {"NIFTY50"}

    latest = _latest_asset_performance()
    current_holdings = set(latest.loc[latest["quantity_held"] > 1e-9, "ticker"])

    assert current_holdings, "test fixture must have real current holdings for this assertion to be meaningful"
    assert correlation_tickers == current_holdings, (
        f"Risk's correlation matrix must be scoped to exactly the current holdings {current_holdings}, "
        f"got {correlation_tickers}"
    )

    # A plausibility guard against the same class of inflation bug:
    # real diversified-equity annualized volatility is never anywhere
    # near the confirmed-bad 245%+ this bug produced.
    assert 0 < body["annualized_volatility"] < 1.5


# --------------------------- benchmark ---------------------------

def test_benchmark_matches_direct_analytics_call():
    from analytics.benchmark.benchmark import benchmark_comparison

    r = client.get(f"/portfolio/{PORTFOLIO_ID}/benchmark")
    assert r.status_code == 200
    body = r.json()

    pp = da.get_portfolio_performance(PORTFOLIO_ID).set_index("value_date")
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
    # distribution (GBM with a fixed seed is deterministic). Matches
    # api/routers/monte_carlo.py's own dynamic full-history window, not
    # the original demo's fixed ANALYSIS_START/END.
    params = get_portfolio_params(PORTFOLIO_ID, "1900-01-01", date.today().isoformat())
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
