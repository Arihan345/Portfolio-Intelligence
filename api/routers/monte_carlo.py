"""Monte Carlo endpoints: thin wrapper over the real Phase 6 engine.
Every simulation call goes through monte_carlo.simulate directly; this
router only translates the HTTP request into that function's real
parameters and its numpy output into the response schema.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from monte_carlo import DISCLAIMER
from monte_carlo.outputs import (
    drawdown_statistics,
    percentile_bands,
    probability_of_exceeding,
    probability_of_loss,
    simulated_var_es,
)
from monte_carlo.params import get_asset_level_params, get_portfolio_params
from monte_carlo.simulate import simulate_portfolio_gbm, simulate_portfolio_multi_asset

from api.dependencies import ANALYSIS_END, ANALYSIS_START, get_portfolio_or_404
from api.schemas.monte_carlo import (
    DrawdownStatistics,
    MonteCarloRequest,
    MonteCarloResponse,
    PercentileBands,
    SimulatedVaRES,
)
from api.store import MONTE_CARLO_RESULTS

router = APIRouter(prefix="/portfolio", tags=["monte-carlo"])


@router.post("/{portfolio_id}/monte-carlo", response_model=MonteCarloResponse)
def run_monte_carlo(
    portfolio_id: int, request: MonteCarloRequest, _: None = Depends(get_portfolio_or_404)
) -> MonteCarloResponse:
    if request.model == "baseline":
        params = get_portfolio_params(portfolio_id, ANALYSIS_START, ANALYSIS_END)
        s0 = params.s0
        mu = request.mu_annual_override if request.mu_annual_override is not None else params.mu_gbm
        sigma = request.sigma_annual_override if request.sigma_annual_override is not None else params.sigma_annual
        paths = simulate_portfolio_gbm(
            s0, mu, sigma, request.horizon_days, request.n_simulations, seed=request.seed
        )
    else:
        params = get_asset_level_params(portfolio_id, ANALYSIS_START, ANALYSIS_END)
        s0 = params.s0_total
        sigma = params.sigma_annual.copy()
        if request.sigma_annual_override is not None:
            sigma = sigma * 0 + request.sigma_annual_override  # apply uniformly across assets
        mu = params.mu_gbm
        if request.mu_annual_override is not None:
            mu = mu * 0 + request.mu_annual_override
        _, paths = simulate_portfolio_multi_asset(
            params.weights, s0, mu, sigma, params.correlation,
            request.horizon_days, request.n_simulations, seed=request.seed,
        )

    terminal = paths[:, -1]
    run_id = str(uuid.uuid4())

    result = MonteCarloResponse(
        run_id=run_id,
        portfolio_id=portfolio_id,
        model=request.model,
        n_simulations=request.n_simulations,
        horizon_days=request.horizon_days,
        initial_value=s0,
        disclaimer=DISCLAIMER,
        created_at=datetime.now(timezone.utc),
        percentile_bands=PercentileBands(**percentile_bands(terminal)),
        probability_of_loss=probability_of_loss(terminal, s0),
        probability_of_exceeding_target=probability_of_exceeding(terminal, request.target_value),
        target_value=request.target_value,
        simulated_var_es=SimulatedVaRES(**simulated_var_es(terminal, s0, request.var_confidence)),
        drawdown_statistics=DrawdownStatistics(**drawdown_statistics(paths)),
    )

    MONTE_CARLO_RESULTS[run_id] = result.model_dump()
    return result


@router.get("/{portfolio_id}/monte-carlo/results", response_model=MonteCarloResponse)
def get_monte_carlo_results(portfolio_id: int, run_id: str) -> MonteCarloResponse:
    stored = MONTE_CARLO_RESULTS.get(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"monte-carlo run {run_id} not found")
    if stored["portfolio_id"] != portfolio_id:
        raise HTTPException(
            status_code=404, detail=f"monte-carlo run {run_id} does not belong to portfolio {portfolio_id}"
        )
    return MonteCarloResponse(**stored)
