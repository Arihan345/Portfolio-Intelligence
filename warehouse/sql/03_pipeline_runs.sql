-- =====================================================================
-- PIPELINE MONITORING (Postgres-only monitoring, per Phase 1 decision:
-- no Prometheus/Grafana by default; this table + Airflow's own UI is
-- the monitoring surface Streamlit's Monitoring page queries).
-- =====================================================================

-- pipeline_runs
-- Grain: one row per Airflow DAG run (or, in this local dev harness,
--     one row per manual pipeline invocation).
-- PK: run_id.
-- FKs: none (referenced BY fact tables' pipeline_run_id, not the
--     reverse, to keep fact loads decoupled from this table's schema).
-- Update frequency: one row inserted at DAG start, updated at
--     completion/failure.
CREATE TABLE pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    dag_id          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('RUNNING','SUCCESS','FAILED')),
    rows_processed  INT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    duration_seconds NUMERIC(10,2)
);
CREATE INDEX idx_pipeline_runs_dag ON pipeline_runs(dag_id);
CREATE INDEX idx_pipeline_runs_started ON pipeline_runs(started_at);
