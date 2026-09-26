import numpy as np
import pytest

from monte_carlo.simulate import simulate_portfolio_gbm, simulate_portfolio_multi_asset


def test_zero_volatility_all_paths_converge_baseline():
    """With sigma=0, every simulation must be IDENTICAL and equal to
    the deterministic compounded value s0 * exp(mu*T) -- this isolates
    whether the simulation's drift/compounding logic is correct before
    trusting it with actual randomness."""
    s0, mu, sigma = 100.0, 0.10, 0.0
    horizon_days, dt = 252, 1 / 252
    paths = simulate_portfolio_gbm(s0, mu, sigma, horizon_days, n_sims=500, seed=42)

    terminal_values = paths[:, -1]
    assert np.allclose(terminal_values, terminal_values[0]), "paths diverged despite zero volatility"

    expected_terminal = s0 * np.exp(mu * horizon_days * dt)
    assert terminal_values[0] == pytest.approx(expected_terminal)

    # Every intermediate day must also match across simulations.
    assert np.allclose(paths, paths[0, :])


def test_zero_volatility_all_paths_converge_multi_asset():
    weights = np.array([0.6, 0.4])
    s0_total = 1000.0
    mu = np.array([0.08, 0.12])
    sigma = np.array([0.0, 0.0])
    correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
    horizon_days, dt = 252, 1 / 252

    asset_paths, portfolio_path = simulate_portfolio_multi_asset(
        weights, s0_total, mu, sigma, correlation, horizon_days, n_sims=300, seed=7
    )

    assert np.allclose(portfolio_path, portfolio_path[0, :])

    expected_asset_terminal = weights * s0_total * np.exp(mu * horizon_days * dt)
    assert np.allclose(asset_paths[0, -1, :], expected_asset_terminal)

    expected_portfolio_terminal = expected_asset_terminal.sum()
    assert portfolio_path[0, -1] == pytest.approx(expected_portfolio_terminal)


def test_higher_volatility_produces_wider_terminal_distribution():
    """Not a hand-calculable exact value, but a directional sanity
    check any correct GBM implementation must satisfy: more volatility
    -> wider spread of outcomes, same seed/drift held fixed."""
    s0, mu = 100.0, 0.08
    low = simulate_portfolio_gbm(s0, mu, 0.10, 252, n_sims=5000, seed=1)
    high = simulate_portfolio_gbm(s0, mu, 0.40, 252, n_sims=5000, seed=1)
    assert high[:, -1].std() > low[:, -1].std()


def test_positive_correlation_increases_portfolio_variance_vs_independent():
    """Directional check for the correlation-matters claim documented
    in simulate.py: simulating two equally-weighted, equal-vol assets
    with a strong POSITIVE correlation must produce a higher-variance
    portfolio terminal value than simulating them as independent
    (correlation = identity), holding everything else fixed."""
    weights = np.array([0.5, 0.5])
    s0_total = 1000.0
    mu = np.array([0.08, 0.08])
    sigma = np.array([0.20, 0.20])
    horizon_days = 252

    independent_corr = np.eye(2)
    _, portfolio_independent = simulate_portfolio_multi_asset(
        weights, s0_total, mu, sigma, independent_corr, horizon_days, n_sims=20000, seed=99
    )

    correlated_corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    _, portfolio_correlated = simulate_portfolio_multi_asset(
        weights, s0_total, mu, sigma, correlated_corr, horizon_days, n_sims=20000, seed=99
    )

    assert portfolio_correlated[:, -1].var() > portfolio_independent[:, -1].var()
