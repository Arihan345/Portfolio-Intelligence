"""Regression test for a real, reproduced concurrency bug (Phase 10
monitoring investigation): concurrent derive_fact_* calls for the same
portfolio_id used to race their DELETE-then-INSERT full recompute and
raise a real psycopg.errors.UniqueViolation on fact_holdings_pkey.

Fixed by _lock_portfolio_recompute (a Postgres advisory transaction
lock scoped to portfolio_id) in warehouse/load/load_warehouse.py. This
test runs against the REAL database (integration-style, like
mlops/tests/test_promotion_gate.py) rather than mocking the DB, because
the bug is inherently about real transaction interleaving that a mock
cannot reproduce.

Requires the example portfolio (portfolio_id=1) to already have
transactions loaded (Phase 3/4's own load or a prior /portfolio/upload)
so derive_fact_holdings has real rows to recompute against.
"""
from __future__ import annotations

import concurrent.futures

from warehouse.load.load_warehouse import (
    derive_fact_holdings,
    derive_fact_portfolio_returns,
    derive_fact_portfolio_value,
)

PORTFOLIO_ID = 1
N_CONCURRENT = 4


def _run_full_derive(i: int):
    try:
        a = derive_fact_holdings(PORTFOLIO_ID, "concurrency_test", f"concurrency-test-{i}")
        b = derive_fact_portfolio_value(PORTFOLIO_ID, "concurrency_test", f"concurrency-test-{i}")
        c = derive_fact_portfolio_returns(PORTFOLIO_ID, "concurrency_test", f"concurrency-test-{i}")
        return ("ok", a, b, c)
    except Exception as exc:  # noqa: BLE001 - we want to see ANY failure, not just UniqueViolation
        return ("error", f"{type(exc).__name__}: {exc}")


def test_concurrent_derive_fact_calls_do_not_race():
    """N_CONCURRENT threads call the full derive_fact_* chain for the
    SAME portfolio_id at the same time. Before the advisory-lock fix,
    this reliably raised a UniqueViolation on fact_holdings_pkey (or
    fact_portfolio_value_pkey / fact_portfolio_returns_pkey) for at
    least one of them. After the fix, the lock serializes them and
    every call must succeed with an identical row count (since they're
    all recomputing from the same underlying fact_transactions data)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        futures = [ex.submit(_run_full_derive, i) for i in range(N_CONCURRENT)]
        results = [f.result() for f in futures]

    errors = [r for r in results if r[0] == "error"]
    assert not errors, f"concurrent derive_fact_* calls raced and failed: {errors}"

    row_counts = [r[1:] for r in results]
    assert len(set(row_counts)) == 1, (
        f"concurrent calls produced DIFFERENT row counts, suggesting a partial/interleaved "
        f"recompute rather than a clean serialized one: {row_counts}"
    )
