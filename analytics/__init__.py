"""Phase 5 analytics engine.

BOUNDARY: what lives in dbt (Phase 4) vs. here, and why
--------------------------------------------------------
dbt owns deterministic, per-row derivations that every consumer needs
pre-aggregated and that must be queryable directly via SQL by Power BI
and Streamlit without invoking Python: daily holdings, daily NAV,
simple day-over-day return, cumulative compounded return, allocation
weights. These have no configurable parameters and the same answer for
every caller, so materializing them once in the warehouse is strictly
better than recomputing them in every Python process.

This package owns analytics that are at least one of:
  - parameterized per caller (risk-free rate for Sharpe/Sortino,
    confidence level for VaR/CVaR, which benchmark to compare against)
    -- baking a parameter into a materialized SQL table would force
    picking one value for everyone;
  - numerically iterative (XIRR requires root-finding on an NPV
    equation; there is no closed-form SQL expression for it);
  - naturally a vector/matrix operation (correlation matrices, rolling
    windows with statistical functions) that pandas/numpy/scipy do
    directly and SQL would need to fake with repeated window functions;
  - needed on-demand by the interactive Streamlit layer with
    user-adjustable inputs (e.g. a user picking their own VaR
    confidence level in the UI), which is a Python-process concern, not
    a warehouse-materialization concern.

Every function here is pure: it takes plain pandas/numpy inputs and
returns a plain result, with no database access inside the computation
itself. Database access is isolated to analytics/data_access.py, which
pulls DataFrames from the marts. This keeps every formula testable with
hand-built inputs, independent of the warehouse being up.
"""
