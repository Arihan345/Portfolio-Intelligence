"""Phase 6: Monte Carlo simulation engine.

DISCLAIMER (carried into every output this module produces): this
module generates a DISTRIBUTION of possible future portfolio outcomes
under stated statistical assumptions (returns are log-normally
distributed with the given drift/volatility/correlation parameters,
typically estimated from a short historical window). It is SCENARIO
ANALYSIS, not a prediction or a guarantee. Real markets do not follow
geometric Brownian motion exactly (fat tails, volatility clustering,
regime changes are all real and unmodeled here), and historical
parameters estimated from a few months of data are themselves highly
uncertain estimates of the future. Nothing this module produces should
be read as "the portfolio will be worth X" -- only as "under these
assumptions, outcomes in this range were plausible."

ARCHITECTURE (mirrors analytics/'s pure-function / data-access split):
  - simulate.py: pure numpy simulation math (GBM path generation). No
    database access, no calls into analytics/ -- takes plain
    mu/sigma/correlation/weights arrays so it is testable with
    hand-picked parameters (e.g. the zero-volatility convergence test).
  - params.py: the impure layer that fetches historical
    return/volatility/correlation estimates FROM Phase 5's analytics
    engine (analytics.data_access + analytics.risk), so this module
    never recomputes those statistics independently and cannot drift
    out of sync with Phase 5's numbers.
  - outputs.py: pure functions that summarize a simulation's output
    (percentile bands, probability of loss/target, simulated VaR/ES,
    drawdown statistics) into the reportable metrics below.
"""

DISCLAIMER = (
    "Monte Carlo scenario analysis, not a prediction. Outcomes are "
    "generated under stated statistical assumptions (log-normal "
    "returns; historical drift/volatility/correlation estimated from "
    "a limited window) and do not guarantee, forecast, or bound any "
    "actual future portfolio value."
)
