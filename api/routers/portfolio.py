"""Portfolio endpoints: a thin layer over the existing ingestion,
warehouse-load, and analytics modules. No business logic is
reimplemented here -- every computation is a direct call into Phase
2/3/4/5's real functions.
"""
from __future__ import annotations

import subprocess
import tempfile
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from analytics import data_access as da
from analytics.data_access import engine
from analytics.allocation.allocation import (
    asset_allocation,
    cash_allocation,
    currency_allocation,
    sector_allocation,
)
from analytics.allocation.concentration import herfindahl_hirschman_index, top_n_pct
from analytics.attribution.attribution import asset_contribution, sector_contribution
from analytics.benchmark.benchmark import benchmark_comparison
from analytics.performance.returns import (
    annualize_return,
    cagr,
    capital_summary,
    daily_returns_from_value_and_flows,
    implied_market_value_cash_flows_by_day,
    periodic_returns,
    realized_pnl_summary,
    rolling_returns,
    time_weighted_return,
    xirr,
)
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
from forecasting.arima.series import build_extended_portfolio_series
from ingestion.cleaning import clean_and_standardize
from ingestion.portfolio_validation import TICKER_PATTERN, validate_csv
from ingestion.providers.yfinance_provider import YFinanceProvider
from warehouse.load.load_benchmark import BENCHMARK_TICKER, upsert_benchmark_asset
from warehouse.load.load_warehouse import (
    create_portfolio,
    dbt_build_lock,
    delete_portfolio,
    derive_fact_holdings,
    derive_fact_portfolio_returns,
    derive_fact_portfolio_value,
    ensure_dim_date_coverage,
    finish_pipeline_run,
    load_fact_daily_prices,
    load_fact_transactions,
    start_pipeline_run,
    upsert_dim_asset,
)

from api.dependencies import get_portfolio_or_404
from api.schemas.allocation import AllocationResponse
from api.schemas.attribution import AttributionResponse
from api.schemas.benchmark import BenchmarkResponse
from api.schemas.performance import CapitalSummary, PerformanceResponse
from api.schemas.portfolio import (
    DataQualityReportResponse,
    DeletePortfolioResponse,
    HoldingWeight,
    PortfolioListResponse,
    PortfolioOverviewResponse,
    PortfolioSummary,
    RejectedRow,
    UploadResponse,
)
from api.schemas.risk import BetaAlpha, MaxDrawdown, RiskResponse

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# GET /portfolios (plural, no {id}) doesn't fit under router's own
# "/portfolio" prefix -- APIRouter always prepends its prefix to every
# route registered on it, with no per-route override -- so it lives on
# this second, unprefixed router instead. Both are registered in
# api/main.py.
portfolios_router = APIRouter(tags=["portfolio"])

# Ticker suffix -> exchange, per the same .NS/.BO convention
# ingestion/portfolio_validation.py's TICKER_PATTERN already requires.
_EXCHANGE_BY_SUFFIX = {"NS": "NSE", "BO": "BSE"}

# A genuinely actively-traded NSE/BSE ticker has a trading day's row for
# nearly every business day in the probe window (confirmed real:
# 600+ rows over a ~2.5 year window). A phantom/misresolved symbol
# returns exactly 1 stray row regardless of window length. 10 is a
# deliberately generous floor -- comfortably above the phantom signal,
# comfortably below what any real multi-month listing would have.
_MIN_REAL_ROWS = 10


def _resolve_real_exchange_suffix(
    tickers: list[str], provider: YFinanceProvider, probe_start: date, probe_end: date
) -> tuple[dict[str, str], list[str]]:
    """Ticker SHAPE validation (portfolio_validation.TICKER_PATTERN)
    only checks that a ticker looks like "BODY.NS" or "BODY.BO" -- it
    can't catch a syntactically-plausible ticker that isn't a real,
    tradeable listing at all. Real, confirmed case: a converted CSV
    recorded "MON100.BO", "MONQ50.BO", "MAFANG.BO" for three NSE-only
    Motilal Oswal/Mirae Asset ETFs with NO real BSE listing -- each
    returns exactly ONE stray row of data from yfinance for the whole
    probe window (a fallback/misresolved-symbol artifact) versus 600+
    rows for a genuinely actively-traded ticker over the same window,
    and a generic/unrelated company name from BSE's own symbol lookup
    (e.g. "CHF/BAM" for "MON100.BO"), while the real ".NS" ticker for
    the same company has real, dense trading data. Silently proceeding
    with the phantom ".BO" ticker produces a position with essentially
    fabricated-empty price history instead of the real one.

    Tickers are probed ONE AT A TIME, not batched together: batching a
    phantom ticker's OHLCV request alongside a genuinely real one was
    observed to make yfinance return a false "this has real data" read
    for the phantom too (a real, reproduced quirk, not a hypothetical
    -- rows appeared for "MON100.BO" only when queried in the same
    batch as "TCS.NS", not when queried alone or with only the other
    phantom tickers). One request per ticker is slower but reliable.

    For each candidate ticker with sparse OHLCV (fewer than
    _MIN_REAL_ROWS rows) over the upload's own transaction date range,
    tries the OTHER exchange suffix for the same company; if THAT has
    real (dense) data, treats it as the correct ticker. probe_start/
    probe_end should cover the actual dates this upload's transactions
    need real prices for, not an arbitrary recent window.

    Returns (rename_map, notes) -- rename_map only contains corrections
    actually needed (old -> new). A ticker where NEITHER suffix has
    real data is left alone (a genuinely delisted/wrong ticker is a
    different, separate problem from this specific exchange-suffix
    mixup, and will still surface on its own downstream).
    """
    shaped = [t for t in tickers if TICKER_PATTERN.match(t)]
    if not shaped:
        return {}, []

    def _row_count(ticker: str) -> int:
        result = provider.fetch_ohlcv([ticker], probe_start, probe_end)
        return len(result.data)

    row_counts = {t: _row_count(t) for t in shaped}
    rename_map: dict[str, str] = {}
    notes: list[str] = []
    for ticker in shaped:
        if row_counts[ticker] >= _MIN_REAL_ROWS:
            continue
        body, suffix = ticker.rsplit(".", 1)
        swapped = f"{body}.{'NS' if suffix == 'BO' else 'BO'}"
        swapped_rows = row_counts.get(swapped)
        if swapped_rows is None:
            swapped_rows = _row_count(swapped)
        if swapped_rows >= _MIN_REAL_ROWS:
            rename_map[ticker] = swapped
            notes.append(
                f"'{ticker}' has essentially no real trading data on that exchange "
                f"({row_counts[ticker]} row(s) vs. {swapped_rows} for '{swapped}' -- likely no real "
                f"listing there) -- corrected to '{swapped}'"
            )
    return rename_map, notes


def _build_asset_meta(tickers: list[str]) -> dict[str, dict]:
    """Real dim_asset metadata for whatever tickers actually appear in
    this upload -- NOT a hardcoded allowlist. An earlier version of
    this endpoint copied warehouse/load/run_load.py's demo scoping
    (KNOWN_TICKERS = ["TCS.NS", "RELIANCE.NS"]) and used it to silently
    filter the cleaned rows before they ever reached the warehouse: any
    real upload with different tickers validated and cleaned
    successfully, then had every single row dropped by that filter,
    while the pipeline still reported SUCCESS with total_rows=0. Sector/
    industry are fetched for real via the provider abstraction
    (fetch_sector_industry already existed for exactly this and was
    unused here); a ticker yfinance has no info for still gets a row --
    sector/industry are nullable in dim_asset -- rather than being
    dropped again at this later stage.
    """
    provider = YFinanceProvider()
    sector_result = provider.fetch_sector_industry(tickers)
    meta: dict[str, dict] = {}
    for ticker in tickers:
        info = sector_result.data.get(ticker)
        suffix = ticker.rsplit(".", 1)[-1] if "." in ticker else ""
        meta[ticker] = {
            "name": ticker.split(".")[0],
            "sector": info.sector if info else None,
            "industry": info.industry if info else None,
            "exchange": _EXCHANGE_BY_SUFFIX.get(suffix),
            "currency": "INR",
        }
    return meta


@router.post("/upload", response_model=UploadResponse)
def upload_portfolio(
    file: UploadFile,
    portfolio_id: int | None = Query(
        None, description="Replace this EXISTING portfolio's data instead of creating a new one."
    ),
    name: str | None = Query(None, description="Display name for a newly-created portfolio."),
) -> UploadResponse:
    """Runs the full existing pipeline: Phase 2 validate -> clean -> Phase
    3 warehouse load -> Phase 4 dbt rebuild. Returns the same
    DataQualityReport shape Phase 2 already defines.

    Multi-portfolio behavior: omitting `portfolio_id` CREATES a new
    saved portfolio (this is the default -- uploading no longer always
    overwrites portfolio_id=1, a real, deliberate change from this
    project's original single-portfolio design). Passing an existing
    `portfolio_id` instead REPLACES that portfolio's data in place
    (the original behavior, still needed for "re-upload a corrected
    CSV for the same portfolio" without spawning a duplicate). Passing
    a `portfolio_id` that doesn't exist yet is a 404, not a silent
    create -- an explicit id is a claim about an existing portfolio,
    never a request to pick that id for a new one.

    Declared as a plain `def`, not `async def`: this function's real
    work (pandas, psycopg, a blocking dbt subprocess) has no `await`
    points, so an `async def` version of it runs entirely on FastAPI's
    single event loop and blocks it for the whole ~6-9s pipeline
    duration -- during which even unrelated requests (e.g. GET
    /portfolio/1/overview) would hang, and concurrent uploads would be
    serialized as an accidental side effect rather than by any real
    concurrency guarantee. A plain `def` runs in Starlette's threadpool
    instead, giving genuine concurrency for other requests and
    exercising the actual DB-level idempotency/locking this endpoint
    depends on (see _lock_portfolio_recompute in
    warehouse/load/load_warehouse.py) rather than masking it.
    """
    if portfolio_id is not None:
        get_portfolio_or_404(portfolio_id)

    contents = file.file.read()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Real stock splits/bonus issues must be applied BEFORE
        # validation's oversold check (a pre-split BUY otherwise reads
        # as impossibly smaller than a post-split SELL of the same real
        # position -- confirmed for real: PCJEWELLER.NS's 1:10 split).
        # This needs the tickers up front, so a lightweight raw read
        # happens here, before the real validate_csv call below -- never
        # raises on a garbage/unreadable file; that's validate_csv's job.
        ticker_correction_notes: list[str] = []
        try:
            raw_peek = pd.read_csv(tmp_path, dtype=str, keep_default_na=False)
            candidate_tickers = sorted(
                {str(t).strip().upper() for t in raw_peek.get("ticker", []) if str(t).strip()}
            )
        except Exception:
            candidate_tickers = []
            raw_peek = None

        if candidate_tickers and raw_peek is not None:
            # Best-effort earliest date across the raw file (structural
            # validation hasn't run yet, so this tolerates unparseable
            # rows rather than failing on them) -- the resolver needs to
            # probe the ACTUAL transaction date range, not an arbitrary
            # recent window (see its own docstring for why).
            parsed_dates = pd.to_datetime(raw_peek.get("date", []), errors="coerce").dropna()
            probe_start = parsed_dates.min().date() if not parsed_dates.empty else date(2020, 1, 1)
            rename_map, ticker_correction_notes = _resolve_real_exchange_suffix(
                candidate_tickers, YFinanceProvider(), probe_start, date.today()
            )
            if rename_map:
                raw_peek["ticker"] = raw_peek["ticker"].apply(
                    lambda t: rename_map.get(str(t).strip().upper(), t)
                )
                raw_peek.to_csv(tmp_path, index=False)
                candidate_tickers = sorted(
                    {str(t).strip().upper() for t in raw_peek["ticker"] if str(t).strip()}
                )

        splits: dict[str, list] = {}
        if candidate_tickers:
            actions = YFinanceProvider().fetch_corporate_actions(
                candidate_tickers, start=date(2000, 1, 1), end=date.today()
            )
            splits = actions.data

        try:
            vresult = validate_csv(tmp_path, today=date.today(), splits=splits)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"malformed CSV: {exc}") from exc

        cleaned, dq_report = clean_and_standardize(
            vresult.valid_rows, split_adjustment_notes=vresult.split_adjustment_notes
        )
        if ticker_correction_notes:
            dq_report.notes = [f"[ticker-corrected] {n}" for n in ticker_correction_notes] + dq_report.notes

        # Portfolio creation happens HERE, after validation/cleaning
        # succeeded -- not up front -- so a malformed or unreadable CSV
        # (rejected above with a 422) never leaves an orphaned, empty
        # new portfolio behind. See this function's own docstring for
        # the create-vs-replace semantics `portfolio_id` selects.
        if portfolio_id is not None:
            target_portfolio_id = portfolio_id
            created_new_portfolio = False
        else:
            default_name = name or file.filename or f"Portfolio ({date.today().isoformat()})"
            target_portfolio_id = create_portfolio(default_name)
            created_new_portfolio = True

        run_id = start_pipeline_run(dag_id="api_upload")
        total_rows = 0
        try:
            tickers_in_upload = sorted(cleaned["ticker"].unique().tolist()) if not cleaned.empty else []
            asset_meta = _build_asset_meta(tickers_in_upload) if tickers_in_upload else {}
            start_d = pd.to_datetime(cleaned["date"]).min().date() if not cleaned.empty else None
            end_d = date.today()

            # Both dimension tables this upload's facts will reference
            # must be extended to cover the upload's own real date/
            # ticker range BEFORE any fact table is loaded -- dim_date
            # and dim_asset were each originally a fixed, one-time-
            # built range from the project's early demo setup (dim_date:
            # warehouse/sql/04_seed_reference.sql's 2023-2026 seed;
            # dim_asset: whichever tickers existed at the time) with no
            # mechanism to extend for a later upload's real data. Both
            # extensions are idempotent (ON CONFLICT DO NOTHING /
            # backdate-if-later-needed) and committed in their own
            # transaction each, so if a later step in this same upload
            # fails, the only effect is that these dimension tables
            # already cover a wider range than any fact table
            # references yet -- harmless, and reused safely on retry.
            if start_d is not None:
                ensure_dim_date_coverage(start_d, end_d)
            # Every ticker's dim_asset effective_from is backdated to
            # this SAME upload-wide start_d (not each ticker's own
            # individual earliest transaction date, and not a fixed
            # constant). This was tried per-ticker first and broke a
            # different way: fact_daily_prices' OHLCV fetch below also
            # starts at this same upload-wide start_d for EVERY ticker
            # (a ticker's real market price history exists before you
            # personally first bought it), so a per-ticker dim_asset
            # date narrower than start_d left load_fact_daily_prices
            # unable to resolve an asset_key for that ticker's own
            # price rows between start_d and its first transaction --
            # confirmed for real: HDFCBANK.NS's dim_asset row correctly
            # covered its own 2022-06-01 first trade, but the upload's
            # OHLCV fetch (start_d = 2022-01-03, TCS.NS's earlier first
            # trade) pulled HDFCBANK.NS price rows back to 2022-01-03
            # too, and resolve_asset_key_for_date failed on those. One
            # shared start_d for both dim_date and every ticker's
            # dim_asset row removes the mismatch instead of chasing it
            # ticker by ticker; backdating a ticker's dim_asset further
            # than its own first trade is harmless (see
            # upsert_dim_asset's docstring).
            asset_keys = upsert_dim_asset(asset_meta, start_d, "api_upload", run_id) if start_d is not None else {}

            n = load_fact_transactions(cleaned, target_portfolio_id, "api_upload", run_id)
            total_rows += n

            if not cleaned.empty:
                tickers_present = tickers_in_upload
                provider = YFinanceProvider()
                ohlcv = provider.fetch_ohlcv(tickers_present, start_d, end_d)
                n = load_fact_daily_prices(ohlcv.data, "api_upload", run_id)
                total_rows += n

                # The NIFTY 50 benchmark index's own OHLCV was only ever
                # loaded once, by a standalone demo script
                # (warehouse/load/load_benchmark.py), hardcoded to the
                # original Jan-Jun 2024 demo window -- never refreshed by
                # a real upload. mart_asset_performance/mart_portfolio_
                # performance correctly extend to today (their own OHLCV
                # is fetched above), but mart_benchmark and Risk's
                # correlation-to-NIFTY50 stayed capped at 2024-06-28
                # regardless, since the benchmark series itself had
                # nothing past that date to join against. Re-fetching it
                # here (upsert_benchmark_asset/load_fact_daily_prices are
                # both idempotent -- ON CONFLICT DO UPDATE / no-op if the
                # asset row already exists) keeps it current with every
                # real upload, the same way the portfolio's own tickers
                # already are.
                upsert_benchmark_asset(start_d, run_id)
                benchmark_ohlcv = provider.fetch_ohlcv([BENCHMARK_TICKER], start_d, end_d)
                n = load_fact_daily_prices(benchmark_ohlcv.data, "api_upload", run_id)
                total_rows += n

            derive_fact_holdings(target_portfolio_id, "api_upload", run_id)
            derive_fact_portfolio_value(target_portfolio_id, "api_upload", run_id)
            derive_fact_portfolio_returns(target_portfolio_id, "api_upload", run_id)

            dbt_script = Path(__file__).resolve().parents[2] / "dbt" / "run_dbt.sh"
            # dbt's own table-swap materialization isn't safe for
            # concurrent `dbt build` runs against the same warehouse
            # (reproduced: two concurrent uploads both hit "relation
            # ...__dbt_backup already exists"). Serialize the whole
            # warehouse-wide build step, not just this portfolio's rows.
            with dbt_build_lock():
                dbt_result = subprocess.run(
                    [str(dbt_script), "build"], capture_output=True, text=True, timeout=300
                )
            if dbt_result.returncode != 0:
                raise RuntimeError(f"dbt build failed: {dbt_result.stdout}\n{dbt_result.stderr}")

            finish_pipeline_run(run_id, "SUCCESS", total_rows)
            status = "SUCCESS"
        except Exception as exc:
            # error_message is now captured on the pipeline_runs row
            # itself (Phase 10 investigation found FAILED runs with no
            # record of why -- a confirmed gap, not just an inconvenience,
            # since it made distinguishing "known dbt/env hiccup" from
            # "new concurrency bug" impossible without terminal scrollback).
            finish_pipeline_run(run_id, "FAILED", total_rows, error_message=f"{type(exc).__name__}: {exc}")
            if created_new_portfolio:
                # A freshly-created portfolio whose very first upload
                # failed shouldn't linger as a broken, empty entry in
                # GET /portfolios' picker -- there's no prior data on
                # it worth protecting (unlike replacing an EXISTING
                # portfolio, where load_fact_transactions deliberately
                # never wipes on a bad upload).
                delete_portfolio(target_portfolio_id)
            raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc

        return UploadResponse(
            portfolio_id=target_portfolio_id,
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


@portfolios_router.get("/portfolios", response_model=PortfolioListResponse)
def list_portfolios() -> PortfolioListResponse:
    """Every saved portfolio, newest first, with enough summary data
    for a picker/switcher UI to render without a follow-up request per
    portfolio. market_value/holdings_count are null for a portfolio
    with no analytics yet (e.g. it exists but every uploaded row was
    rejected, or dbt hasn't built its marts yet).
    """
    with engine.begin() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT
                    p.portfolio_id,
                    p.portfolio_name AS name,
                    p.created_at,
                    (
                        SELECT SUM(m.market_value_inr)
                        FROM public_marts.mart_asset_performance m
                        WHERE m.portfolio_id = p.portfolio_id
                          AND m.as_of_date = (
                              SELECT MAX(as_of_date) FROM public_marts.mart_asset_performance
                              WHERE portfolio_id = p.portfolio_id
                          )
                    ) AS market_value,
                    (
                        SELECT COUNT(*)
                        FROM public_marts.mart_asset_performance m
                        WHERE m.portfolio_id = p.portfolio_id
                          AND m.as_of_date = (
                              SELECT MAX(as_of_date) FROM public_marts.mart_asset_performance
                              WHERE portfolio_id = p.portfolio_id
                          )
                          AND m.quantity_held > 1e-9
                    ) AS holdings_count
                FROM dim_portfolio p
                ORDER BY p.created_at DESC
                """
            )
        ).mappings().all()

    return PortfolioListResponse(
        portfolios=[
            PortfolioSummary(
                portfolio_id=r["portfolio_id"],
                name=r["name"],
                created_at=r["created_at"],
                market_value=float(r["market_value"]) if r["market_value"] is not None else None,
                holdings_count=int(r["holdings_count"]) if r["holdings_count"] is not None else None,
            )
            for r in rows
        ]
    )


@router.delete("/{portfolio_id}", response_model=DeletePortfolioResponse)
def delete_portfolio_endpoint(portfolio_id: int, _: None = Depends(get_portfolio_or_404)) -> DeletePortfolioResponse:
    """Permanently deletes a saved portfolio and every fact row scoped
    to it. Destructive and irreversible -- the frontend is responsible
    for requiring the user to confirm before calling this; this
    endpoint itself performs the delete unconditionally once called,
    the same way DELETE endpoints conventionally do.
    """
    deleted = delete_portfolio(portfolio_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"portfolio {portfolio_id} not found")

    # The mart_* tables are plain physical tables (dbt TABLE
    # materialization, not views) -- deleting fact_transactions etc.
    # above doesn't remove this portfolio's now-orphaned rows from
    # them. Rebuilding keeps GET /portfolios (which reads directly from
    # mart_asset_performance) from showing a deleted portfolio's stale
    # summary stats, and matches the same rebuild-after-fact-change
    # pattern the upload endpoint already uses.
    dbt_script = Path(__file__).resolve().parents[2] / "dbt" / "run_dbt.sh"
    with dbt_build_lock():
        subprocess.run([str(dbt_script), "build"], capture_output=True, text=True, timeout=300)

    return DeletePortfolioResponse(portfolio_id=portfolio_id, deleted=True)


@router.get("/{portfolio_id}/overview", response_model=PortfolioOverviewResponse)
def get_overview(portfolio_id: int, _: None = Depends(get_portfolio_or_404)) -> PortfolioOverviewResponse:
    ap = da.get_asset_performance(portfolio_id)
    latest = ap[ap["as_of_date"] == ap["as_of_date"].max()]
    if latest.empty:
        raise HTTPException(status_code=404, detail=f"no holdings data for portfolio {portfolio_id}")

    invested_capital = float(latest["cost_basis_inr"].sum())
    market_value = float(latest["market_value_inr"].sum())
    realized = realized_pnl_summary(da.get_transactions(portfolio_id))
    summary = capital_summary(invested_capital, market_value, realized)

    alloc = da.get_allocation(portfolio_id)
    alloc_latest = alloc[alloc["as_of_date"] == latest["as_of_date"].iloc[0]]
    # "Current holdings" means positions actually held today -- a fully
    # exited position (quantity_held netted to 0, confirmed real cases:
    # PCJEWELLER.NS, ALOKINDS.NS, SWANCORP.NS, and others in this same
    # real portfolio) has market_value_inr == 0 and contributed exactly
    # 0 weight, but still showed up as its own 0.00% row in the list --
    # every ticker ever traded, not what's actually owned. Filtered out
    # here (not in asset_allocation itself, which stays a general-
    # purpose weighted breakdown); invested_capital/market_value above
    # deliberately still sum ALL rows, unfiltered -- those are portfolio
    # totals, not the holdings list, and changing them isn't this fix.
    alloc_latest_held = alloc_latest[alloc_latest["market_value_inr"] > 1e-9]
    weights = asset_allocation(alloc_latest_held)

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
    pp = da.get_portfolio_performance(portfolio_id).set_index("value_date")
    if pp.empty:
        raise HTTPException(status_code=404, detail=f"no performance data for portfolio {portfolio_id}")

    ap = da.get_asset_performance(portfolio_id)
    latest = ap[ap["as_of_date"] == ap["as_of_date"].max()]
    tx = da.get_transactions(portfolio_id)
    realized = realized_pnl_summary(tx)
    summary = capital_summary(float(latest["cost_basis_inr"].sum()), float(latest["market_value_inr"].sum()), realized)

    # CAGR/TWR are computed from a CORRECTED daily holdings-value series,
    # not mart_portfolio_performance.total_nav_inr/daily_return directly.
    # Two real, confirmed problems with that mart series for this
    # purpose: (1) total_nav_inr's cash_balance_inr leg only ever grows
    # (every SELL's proceeds sit there forever, even when later
    # reinvested -- see monte_carlo/params.py's docstring for the same
    # root cause already fixed there), inflating the ending value; (2) a
    # position bought before it has any real market price yet (e.g. an
    # IPO-day purchase recorded the day before listing) prices at INR 0
    # for that gap -- a real, already-held position with real cost
    # basis, not actually worthless -- so the day real price data
    # begins reads as a fake explosive "return" (confirmed for real: a
    # portfolio showed +443% on the single day a pre-listing holding's
    # price data started, which chain-linked into a 5.2495 CAGR and an
    # 86.8 cumulative TWR for the whole multi-year history). Filling
    # that gap at cost (a real, already-committed position without an
    # observable market price is conventionally valued at cost, never
    # at zero) removes the artifact at its source rather than papering
    # over the resulting number.
    ap_gap_filled = ap.copy()
    no_price_yet = (ap_gap_filled["quantity_held"] > 0) & (ap_gap_filled["market_value_inr"].abs() < 1e-9)
    ap_gap_filled.loc[no_price_yet, "market_value_inr"] = ap_gap_filled.loc[no_price_yet, "cost_basis_inr"]
    holdings_value = ap_gap_filled.groupby("as_of_date")["market_value_inr"].sum().sort_index()

    # Cash flows for THIS holdings-value series must be valued on the
    # SAME real-market-price basis holdings_value itself uses -- not
    # the transaction's own recorded price (net_external_cash_flows_by_
    # day, still used below for XIRR, where the real recorded price IS
    # what should count). Confirmed real, non-hypothetical case: a
    # ticker string that happens to have its own real, unrelated
    # yfinance price data priced a SELL's cash-out at that real close
    # while the transaction itself recorded a wildly different price,
    # producing a fake ~-92% single-day return purely from the
    # mismatch. See implied_market_value_cash_flows_by_day's own
    # docstring for the full mechanism and hand-traced numbers.
    cash_flows_by_day = implied_market_value_cash_flows_by_day(ap_gap_filled)

    daily_ret_adj = daily_returns_from_value_and_flows(holdings_value, cash_flows_by_day)
    twr = time_weighted_return(holdings_value, cash_flows_by_day)
    history_days = (holdings_value.index[-1] - holdings_value.index[0]).days
    cagr_value = annualize_return(twr, history_days)
    monthly = periodic_returns(daily_ret_adj, freq="ME")
    rolling_30d = rolling_returns(daily_ret_adj, window=30)

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
    if alloc.empty:
        raise HTTPException(status_code=404, detail=f"no allocation data for portfolio {portfolio_id}")
    latest = alloc[alloc["as_of_date"] == alloc["as_of_date"].max()]
    # Exclude fully-exited positions (quantity_held netted to 0, so
    # market_value_inr == 0) from the current-holdings breakdown -- see
    # the matching comment in get_overview for the real case this fixes.
    latest = latest[latest["market_value_inr"] > 1e-9]

    weights = asset_allocation(latest)

    pp = da.get_portfolio_performance(portfolio_id)
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
    start_date: date | None = Query(None, description="Restrict the risk window to this start (inclusive)."),
    end_date: date | None = Query(None, description="Restrict the risk window to this end (inclusive)."),
    _: None = Depends(get_portfolio_or_404),
) -> RiskResponse:
    # Every metric on this page must reflect only CURRENTLY-HELD
    # positions -- Risk answers "how risky is what I own right now,"
    # a genuinely different question from Attribution's "what
    # contributed to return over a historical window" (which correctly
    # includes since-exited positions by design). Confirmed real bug:
    # this endpoint used to build its return series from
    # mart_portfolio_performance.daily_return/total_nav_inr (the
    # portfolio's ACTUAL historical NAV path, correctly including every
    # exited position's own volatility while it was held) and pivot
    # mart_asset_performance UNFILTERED for the correlation matrix
    # (every ticker ever traded, not just today's holdings) -- a real
    # portfolio's correlation matrix showed ALOKINDS.NS/BAJAJHFL.NS/
    # DIXON.NS/etc. (mostly long-exited positions) instead of the 9
    # real current holdings, and annualized volatility came out at an
    # implausible 245%+, inflated by highly volatile exited penny
    # stocks no longer owned. build_extended_portfolio_series already
    # builds exactly the right series for this (Monte Carlo's baseline
    # params and Performance's CAGR/TWR fix both already use it): each
    # CURRENTLY-HELD ticker's own real price history, weighted at
    # today's actual allocation -- unifying Risk onto it here removes
    # the same class of bug from a third place instead of patching this
    # one spot in isolation.
    try:
        series = build_extended_portfolio_series(portfolio_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"risk metrics aren't available for this portfolio right now: {exc}"
        ) from exc

    # Optional user-selected window (e.g. "Last 3 months" or a custom
    # range) -- restricts every metric to that slice of the SAME
    # currently-held-tickers series above, never widening back out to
    # exited positions just because a narrower date range was picked.
    # Applied here, before computing any metric, so every downstream
    # calculation (Sharpe, volatility, VaR, correlation, ...) sees
    # exactly the same windowed data consistently.
    if start_date is not None or end_date is not None:
        lo = pd.Timestamp(start_date) if start_date is not None else series.index[0]
        hi = pd.Timestamp(end_date) if end_date is not None else series.index[-1]
        if lo >= hi:
            raise HTTPException(status_code=422, detail="start_date must be before end_date")
        series = series.loc[(series.index >= lo) & (series.index <= hi)]
        if len(series) < 3:
            raise HTTPException(
                status_code=422,
                detail=f"the range {lo.date()} to {hi.date()} has too little real price data to compute risk metrics",
            )

    daily_returns = series.pct_change().dropna()

    bench = da.get_benchmark(portfolio_id).set_index("value_date")
    bench_returns = bench["benchmark_daily_return"].reindex(daily_returns.index)
    ba = beta_alpha(daily_returns, bench_returns, risk_free_rate_annual)
    mdd = max_drawdown(series)

    ap = da.get_asset_performance(portfolio_id)
    latest_date = ap["as_of_date"].max()
    held_tickers = ap.loc[(ap["as_of_date"] == latest_date) & (ap["quantity_held"] > 1e-9), "ticker"]
    ap_held = ap[ap["ticker"].isin(held_tickers)]
    wide = ap_held.pivot(index="as_of_date", columns="ticker", values="market_value_inr").pct_change().dropna(how="all")
    wide = wide.reindex(wide.index.intersection(series.index))
    wide["NIFTY50"] = bench_returns.reindex(wide.index)
    corr = correlation_matrix(wide)

    return RiskResponse(
        portfolio_id=portfolio_id,
        window_start=series.index[0].date(),
        window_end=series.index[-1].date(),
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
    pp = da.get_portfolio_performance(portfolio_id).set_index("value_date")
    if pp.empty:
        raise HTTPException(status_code=404, detail=f"no benchmark data for portfolio {portfolio_id}")
    daily_returns = pp["daily_return"].dropna()

    bench = da.get_benchmark(portfolio_id).set_index("value_date")
    bench_returns = bench["benchmark_daily_return"].reindex(daily_returns.index)

    # Real total return on invested capital (same calculation
    # get_overview uses) -- overrides the naive compounded-daily-return
    # cumulative figure below, which a real portfolio built up through
    # many separate purchases over time (not one lump-sum investment)
    # distorts into a fabricated headline number (confirmed real case:
    # ~9,724% vs. the portfolio's real 10.67% total return). See
    # benchmark_comparison's own docstring for the full mechanism.
    ap = da.get_asset_performance(portfolio_id)
    latest_ap = ap[ap["as_of_date"] == ap["as_of_date"].max()]
    real_total_return = capital_summary(
        float(latest_ap["cost_basis_inr"].sum()),
        float(latest_ap["market_value_inr"].sum()),
        realized_pnl_summary(da.get_transactions(portfolio_id)),
    )["pct_return"]

    comparison = benchmark_comparison(
        daily_returns, bench_returns, risk_free_rate_annual,
        portfolio_cumulative_return_override=real_total_return,
    )
    return BenchmarkResponse(portfolio_id=portfolio_id, benchmark_ticker="^NSEI", **comparison)


def _find_stable_composition_window(portfolio_id: int) -> tuple[str, str] | None:
    """Finds the widest real period with NO BUY/SELL activity for this
    portfolio. The weight*return attribution identity below only holds
    exactly when nothing was bought/sold mid-window -- otherwise the
    portfolio's actual composition changed partway through, and
    "start weights x period return" no longer equals what actually
    happened (a real, non-negligible gap, not floating-point noise).

    A hardcoded window ("2024-02-01" to "2024-05-31", picked for the
    original 2-ticker demo's own specific trading history) is real bug
    for any other upload: confirmed directly against a real 51-
    transaction portfolio whose earliest trade (2024-04-18) postdates
    that hardcoded window's own START date entirely, so no allocation
    snapshot exists there at all (a 404, not even a reconciliation
    mismatch) -- and even for a portfolio where both dates DID exist,
    trading activity happening to fall inside that fixed window (also
    confirmed real: an INFY.NS BUY landing inside it) breaks the
    reconciliation identity outright (a real ~21 percentage-point gap
    was observed this way, not a bug in the contribution math itself).

    Returns (window_start, window_end) as ISO date strings -- the
    widest gap between two consecutive real trade dates for this
    portfolio -- or None if there are fewer than 2 trading days on
    record (nothing to find a gap between).
    """
    tx = da.get_transactions(portfolio_id)
    trade_dates = sorted(tx[tx["transaction_type"].isin(["BUY", "SELL"])]["txn_date"].dt.date.unique())
    if len(trade_dates) < 2:
        return None

    best_gap_days = -1
    best_start, best_end = None, None
    for prev_date, next_date in zip(trade_dates, trade_dates[1:]):
        gap_days = (next_date - prev_date).days
        if gap_days > best_gap_days:
            best_gap_days = gap_days
            best_start, best_end = prev_date, next_date - timedelta(days=1)

    if best_start is None or best_start >= best_end:
        return None
    return best_start.isoformat(), best_end.isoformat()


@router.get("/{portfolio_id}/attribution", response_model=AttributionResponse)
def get_attribution(
    portfolio_id: int,
    start_date: date | None = Query(None, description="Custom window start (inclusive). Requires end_date too."),
    end_date: date | None = Query(None, description="Custom window end (inclusive). Requires start_date too."),
    _: None = Depends(get_portfolio_or_404),
) -> AttributionResponse:
    if start_date is not None or end_date is not None:
        if start_date is None or end_date is None:
            raise HTTPException(status_code=422, detail="start_date and end_date must both be given for a custom window")
        if start_date >= end_date:
            raise HTTPException(status_code=422, detail="start_date must be before end_date")

        # Restriction approach (not the weight-change extension): the
        # weight*return identity below only holds exactly when nothing
        # was bought/sold mid-window, so a user-picked range containing
        # a trade is rejected outright with the exact offending date(s)
        # -- clearer and more honest than silently producing a number
        # that fails its own reconciliation check, and simpler than
        # correctly handling a composition that changes partway through
        # (which _find_stable_composition_window's own docstring
        # already treats as a real, non-negligible gap, not a detail to
        # paper over).
        tx = da.get_transactions(portfolio_id)
        trade_dates = sorted(tx[tx["transaction_type"].isin(["BUY", "SELL"])]["txn_date"].dt.date.unique())
        offending = [d for d in trade_dates if start_date < d < end_date]
        if offending:
            shown = ", ".join(d.isoformat() for d in offending[:5])
            more = f", and {len(offending) - 5} more" if len(offending) > 5 else ""
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the range {start_date} to {end_date} includes {len(offending)} buy/sell "
                    f"date(s) ({shown}{more}) -- pick a range with no trading activity strictly "
                    "between the two dates, so every holding's weight stayed constant throughout "
                    "(the boundary dates themselves are fine)."
                ),
            )
        window_start, window_end = start_date.isoformat(), end_date.isoformat()
    else:
        window = _find_stable_composition_window(portfolio_id)
        if window is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no stable no-trading-activity window available for portfolio {portfolio_id} "
                    "(attribution needs at least 2 trading days with a gap between them to compute "
                    "a real weight x return reconciliation)"
                ),
            )
        window_start, window_end = window

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
