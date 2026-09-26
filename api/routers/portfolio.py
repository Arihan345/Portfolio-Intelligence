"""Portfolio endpoints: a thin layer over the existing ingestion,
warehouse-load, and analytics modules. No business logic is
reimplemented here -- every computation is a direct call into Phase
2/3/4/5's real functions.
"""
from __future__ import annotations

import subprocess
import tempfile
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from analytics import data_access as da
from analytics.allocation.allocation import (
    asset_allocation,
    cash_allocation,
    currency_allocation,
    sector_allocation,
)
from analytics.allocation.concentration import herfindahl_hirschman_index, top_n_pct
from analytics.attribution.attribution import asset_contribution, sector_contribution
from analytics.benchmark.benchmark import benchmark_comparison
from analytics.performance.returns import cagr, capital_summary, periodic_returns, rolling_returns, time_weighted_return, xirr
from analytics.risk.risk import (
    annualized_volatility,
    beta_alpha,
    conditional_var,
    correlation_matrix,
    downside_volatility,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)
from ingestion.cleaning import clean_and_standardize
from ingestion.portfolio_validation import validate_csv
from ingestion.providers.yfinance_provider import YFinanceProvider
from warehouse.load.load_warehouse import (
    derive_fact_holdings,
    derive_fact_portfolio_returns,
    derive_fact_portfolio_value,
    finish_pipeline_run,
    load_fact_daily_prices,
    load_fact_transactions,
    start_pipeline_run,
    upsert_dim_asset,
)

from api.dependencies import ANALYSIS_END, ANALYSIS_START, get_portfolio_or_404
from api.schemas.allocation import AllocationResponse
from api.schemas.attribution import AttributionResponse
from api.schemas.benchmark import BenchmarkResponse
from api.schemas.performance import CapitalSummary, PerformanceResponse
from api.schemas.portfolio import (
    DataQualityReportResponse,
    HoldingWeight,
    PortfolioOverviewResponse,
    RejectedRow,
    UploadResponse,
)
from api.schemas.risk import BetaAlpha, MaxDrawdown, RiskResponse

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# The only tickers this project has seeded metadata for (dim_asset,
# analytics marts) -- see warehouse/load/run_load.py's own asset_meta.
# Rows for any other ticker are validated/cleaned like everything else,
# but excluded from the warehouse load stage, mirroring that script's
# existing, documented behavior exactly rather than inventing new scope.
KNOWN_TICKERS = ["TCS.NS", "RELIANCE.NS"]
ASSET_META = {
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "Technology",
               "industry": "IT Services", "exchange": "NSE", "currency": "INR"},
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy",
                     "industry": "Conglomerate", "exchange": "NSE", "currency": "INR"},
}
DEFAULT_PORTFOLIO_ID = 1


@router.post("/upload", response_model=UploadResponse)
async def upload_portfolio(file: UploadFile) -> UploadResponse:
    """Runs the full existing pipeline: Phase 2 validate -> clean -> Phase
    3 warehouse load -> Phase 4 dbt rebuild. Returns the same
    DataQualityReport shape Phase 2 already defines."""
    contents = await file.read()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        try:
            vresult = validate_csv(tmp_path, today=date.today())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"malformed CSV: {exc}") from exc

        cleaned, dq_report = clean_and_standardize(vresult.valid_rows)
        cleaned = cleaned[cleaned["ticker"].isin(KNOWN_TICKERS)].reset_index(drop=True)

        run_id = start_pipeline_run(dag_id="api_upload")
        total_rows = 0
        try:
            asset_keys = upsert_dim_asset(ASSET_META, date(2024, 1, 1), "api_upload", run_id)

            n = load_fact_transactions(cleaned, DEFAULT_PORTFOLIO_ID, "api_upload", run_id)
            total_rows += n

            if not cleaned.empty:
                tickers_present = sorted(cleaned["ticker"].unique().tolist())
                start_d = pd.to_datetime(cleaned["date"]).min().date()
                end_d = date.today()
                provider = YFinanceProvider()
                ohlcv = provider.fetch_ohlcv(tickers_present, start_d, end_d)
                n = load_fact_daily_prices(ohlcv.data, "api_upload", run_id)
                total_rows += n

            derive_fact_holdings(DEFAULT_PORTFOLIO_ID, "api_upload", run_id)
            derive_fact_portfolio_value(DEFAULT_PORTFOLIO_ID, "api_upload", run_id)
            derive_fact_portfolio_returns(DEFAULT_PORTFOLIO_ID, "api_upload", run_id)

            dbt_script = Path(__file__).resolve().parents[2] / "dbt" / "run_dbt.sh"
            dbt_result = subprocess.run(
                [str(dbt_script), "build"], capture_output=True, text=True, timeout=300
            )
            if dbt_result.returncode != 0:
                raise RuntimeError(f"dbt build failed: {dbt_result.stdout}\n{dbt_result.stderr}")

            finish_pipeline_run(run_id, "SUCCESS", total_rows)
            status = "SUCCESS"
        except Exception as exc:
            finish_pipeline_run(run_id, "FAILED", total_rows)
            raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc

        return UploadResponse(
            portfolio_id=DEFAULT_PORTFOLIO_ID,
            pipeline_run_id=run_id,
            status=status,
            valid_row_count=len(vresult.valid_rows),
            rejected_row_count=len(vresult.rejected_rows),
            rejected_rows=[
                RejectedRow(row_index=int(idx), reason=row["rejection_reason"])
                for idx, row in vresult.rejected_rows.iterrows()
            ],
            data_quality_report=DataQualityReportResponse(**dq_report.as_dict()),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/{portfolio_id}/overview", response_model=PortfolioOverviewResponse)
def get_overview(portfolio_id: int, _: None = Depends(get_portfolio_or_404)) -> PortfolioOverviewResponse:
    ap = da.get_asset_performance(portfolio_id)
    ap = ap[(ap["as_of_date"] >= ANALYSIS_START) & (ap["as_of_date"] <= ANALYSIS_END)]
    latest = ap[ap["as_of_date"] == ap["as_of_date"].max()]
    if latest.empty:
        raise HTTPException(status_code=404, detail=f"no holdings data for portfolio {portfolio_id}")

    invested_capital = float(latest["cost_basis_inr"].sum())
    market_value = float(latest["market_value_inr"].sum())
    summary = capital_summary(invested_capital, market_value)

    alloc = da.get_allocation(portfolio_id)
    alloc_latest = alloc[alloc["as_of_date"] == latest["as_of_date"].iloc[0]]
    weights = asset_allocation(alloc_latest)

    return PortfolioOverviewResponse(
        portfolio_id=portfolio_id,
        as_of_date=latest["as_of_date"].iloc[0].date(),
        invested_capital=summary["invested_capital"],
        market_value=summary["market_value"],
        absolute_return=summary["absolute_return"],
        pct_return=summary["pct_return"],
        holdings=[HoldingWeight(ticker=t, weight=float(w)) for t, w in weights.items()],
    )


@router.get("/{portfolio_id}/performance", response_model=PerformanceResponse)
def get_performance(portfolio_id: int, _: None = Depends(get_portfolio_or_404)) -> PerformanceResponse:
    pp = da.get_portfolio_performance(portfolio_id)
    pp = pp[(pp["value_date"] >= ANALYSIS_START) & (pp["value_date"] <= ANALYSIS_END)].set_index("value_date")
    if pp.empty:
        raise HTTPException(status_code=404, detail=f"no performance data for portfolio {portfolio_id}")

    ap = da.get_asset_performance(portfolio_id)
    ap = ap[(ap["as_of_date"] >= ANALYSIS_START) & (ap["as_of_date"] <= ANALYSIS_END)]
    latest = ap[ap["as_of_date"] == ap["as_of_date"].max()]
    summary = capital_summary(float(latest["cost_basis_inr"].sum()), float(latest["market_value_inr"].sum()))

    start_nav, end_nav = float(pp["total_nav_inr"].iloc[0]), float(pp["total_nav_inr"].iloc[-1])
    cagr_value = cagr(start_nav, end_nav, pp.index[0].date(), pp.index[-1].date())
    monthly = periodic_returns(pp["daily_return"], freq="ME")
    rolling_30d = rolling_returns(pp["daily_return"], window=30)
    twr = time_weighted_return(pp["total_nav_inr"])

    tx = da.get_transactions(portfolio_id)
    buy_sell_div = tx[tx["transaction_type"].isin(["BUY", "SELL", "DIVIDEND"])]
    cash_flows = []
    for _, r in buy_sell_div.iterrows():
        if r["transaction_type"] == "BUY":
            amt = -(r["price_inr"] * r["quantity"] + r["fees"])
        elif r["transaction_type"] == "SELL":
            amt = r["price_inr"] * r["quantity"] - r["fees"] - r["tax"]
        else:
            amt = r["price_inr"]
        cash_flows.append((r["txn_date"].date(), float(amt)))
    cash_flows.append((pp.index[-1].date(), summary["market_value"]))
    xirr_value = xirr(cash_flows)

    return PerformanceResponse(
        portfolio_id=portfolio_id,
        capital_summary=CapitalSummary(**summary),
        cagr=cagr_value,
        monthly_returns={str(k.date()): float(v) for k, v in monthly.items()},
        latest_30d_rolling_return=float(rolling_30d.iloc[-1]),
        time_weighted_return=twr,
        xirr=xirr_value,
    )


@router.get("/{portfolio_id}/allocation", response_model=AllocationResponse)
def get_allocation_endpoint(portfolio_id: int, _: None = Depends(get_portfolio_or_404)) -> AllocationResponse:
    alloc = da.get_allocation(portfolio_id)
    alloc = alloc[(alloc["as_of_date"] >= ANALYSIS_START) & (alloc["as_of_date"] <= ANALYSIS_END)]
    if alloc.empty:
        raise HTTPException(status_code=404, detail=f"no allocation data for portfolio {portfolio_id}")
    latest = alloc[alloc["as_of_date"] == alloc["as_of_date"].max()]

    weights = asset_allocation(latest)

    pp = da.get_portfolio_performance(portfolio_id)
    pp = pp[(pp["value_date"] >= ANALYSIS_START) & (pp["value_date"] <= ANALYSIS_END)]
    last_pp = pp.iloc[-1]

    return AllocationResponse(
        portfolio_id=portfolio_id,
        asset_allocation={t: float(w) for t, w in weights.items()},
        sector_allocation={s: float(w) for s, w in sector_allocation(latest).items()},
        currency_allocation={c: float(w) for c, w in currency_allocation(latest).items()},
        cash_allocation=cash_allocation(
            float(last_pp["holdings_market_value_inr"]), float(last_pp["cash_balance_inr"])
        ),
        largest_holding_pct=float(weights.max()),
        top_2_pct=top_n_pct(weights, 2),
        herfindahl_hirschman_index=herfindahl_hirschman_index(weights),
    )


@router.get("/{portfolio_id}/risk", response_model=RiskResponse)
def get_risk(
    portfolio_id: int,
    risk_free_rate_annual: float = 0.07,
    var_confidence: float = 0.95,
    _: None = Depends(get_portfolio_or_404),
) -> RiskResponse:
    pp = da.get_portfolio_performance(portfolio_id)
    pp = pp[(pp["value_date"] >= ANALYSIS_START) & (pp["value_date"] <= ANALYSIS_END)].set_index("value_date")
    if pp.empty:
        raise HTTPException(status_code=404, detail=f"no risk data for portfolio {portfolio_id}")
    daily_returns = pp["daily_return"].dropna()

    bench = da.get_benchmark(portfolio_id).set_index("value_date")
    bench_returns = bench["benchmark_daily_return"].reindex(daily_returns.index)
    ba = beta_alpha(daily_returns, bench_returns, risk_free_rate_annual)
    mdd = max_drawdown(pp["total_nav_inr"])

    ap = da.get_asset_performance(portfolio_id)
    ap = ap[(ap["as_of_date"] >= ANALYSIS_START) & (ap["as_of_date"] <= ANALYSIS_END)]
    wide = ap.pivot(index="as_of_date", columns="ticker", values="market_value_inr").pct_change().dropna(how="all")
    wide["NIFTY50"] = bench_returns.reindex(wide.index)
    corr = correlation_matrix(wide)

    return RiskResponse(
        portfolio_id=portfolio_id,
        risk_free_rate_annual=risk_free_rate_annual,
        var_confidence=var_confidence,
        annualized_volatility=annualized_volatility(daily_returns),
        downside_volatility=downside_volatility(daily_returns),
        sharpe_ratio=sharpe_ratio(daily_returns, risk_free_rate_annual),
        sortino_ratio=sortino_ratio(daily_returns, risk_free_rate_annual),
        beta_alpha=BetaAlpha(**ba),
        max_drawdown=MaxDrawdown(
            max_drawdown=mdd["max_drawdown"],
            peak_date=mdd["peak_date"].date(),
            trough_date=mdd["trough_date"].date(),
            recovery_date=mdd["recovery_date"].date() if mdd["recovery_date"] is not None else None,
            drawdown_duration_days=mdd["drawdown_duration_days"],
        ),
        historical_var=historical_var(daily_returns, var_confidence),
        parametric_var=parametric_var(daily_returns, var_confidence),
        conditional_var=conditional_var(daily_returns, var_confidence),
        correlation_matrix={
            row: {col: float(val) for col, val in corr.loc[row].items()} for row in corr.index
        },
    )


@router.get("/{portfolio_id}/benchmark", response_model=BenchmarkResponse)
def get_benchmark_endpoint(
    portfolio_id: int, risk_free_rate_annual: float = 0.07, _: None = Depends(get_portfolio_or_404)
) -> BenchmarkResponse:
    pp = da.get_portfolio_performance(portfolio_id)
    pp = pp[(pp["value_date"] >= ANALYSIS_START) & (pp["value_date"] <= ANALYSIS_END)].set_index("value_date")
    if pp.empty:
        raise HTTPException(status_code=404, detail=f"no benchmark data for portfolio {portfolio_id}")
    daily_returns = pp["daily_return"].dropna()

    bench = da.get_benchmark(portfolio_id).set_index("value_date")
    bench_returns = bench["benchmark_daily_return"].reindex(daily_returns.index)

    comparison = benchmark_comparison(daily_returns, bench_returns, risk_free_rate_annual)
    return BenchmarkResponse(portfolio_id=portfolio_id, benchmark_ticker="^NSEI", **comparison)


@router.get("/{portfolio_id}/attribution", response_model=AttributionResponse)
def get_attribution(portfolio_id: int, _: None = Depends(get_portfolio_or_404)) -> AttributionResponse:
    # Same stable-composition window used in analytics/run_analytics_demo.py:
    # the only period with no BUY/SELL activity, required for the
    # weight*return reconciliation identity to hold (see that script's
    # ATTRIBUTION section for the full reasoning).
    window_start, window_end = "2024-02-01", "2024-05-31"

    alloc = da.get_allocation(portfolio_id)
    start_alloc = alloc[alloc["as_of_date"] == window_start]
    if start_alloc.empty:
        raise HTTPException(status_code=404, detail=f"no attribution data for portfolio {portfolio_id}")
    start_weights = asset_allocation(start_alloc)

    ap = da.get_asset_performance(portfolio_id)
    end_prices = ap[ap["as_of_date"] == window_end].set_index("ticker")["last_close_inr"]
    start_prices = ap[ap["as_of_date"] == window_start].set_index("ticker")["last_close_inr"]
    asset_period_returns = (end_prices / start_prices - 1).dropna()

    contrib = asset_contribution(start_weights, asset_period_returns)
    sector_map = ap.drop_duplicates("ticker").set_index("ticker")["sector"]
    sector_contrib = sector_contribution(start_weights, asset_period_returns, sector_map)

    pp = da.get_portfolio_performance(portfolio_id).set_index("value_date")
    holdings_start = float(pp.loc[window_start, "holdings_market_value_inr"])
    holdings_end = float(pp.loc[window_end, "holdings_market_value_inr"])
    actual_return = holdings_end / holdings_start - 1

    return AttributionResponse(
        portfolio_id=portfolio_id,
        window_start=date.fromisoformat(window_start),
        window_end=date.fromisoformat(window_end),
        asset_contribution={t: float(v) for t, v in contrib.items()},
        sector_contribution={s: float(v) for s, v in sector_contrib.items()},
        total_return_from_contributions=float(contrib.sum()),
        actual_holdings_return=actual_return,
    )
