"""In-memory store for Monte Carlo run results, keyed by run_id, so
GET /portfolio/{id}/monte-carlo/results can retrieve a prior run
without recomputing it. Process-local (fine for this phase's scope --
a real deployment would persist this in Postgres alongside
pipeline_runs, following the same pattern)."""
from __future__ import annotations

MONTE_CARLO_RESULTS: dict[str, dict] = {}
