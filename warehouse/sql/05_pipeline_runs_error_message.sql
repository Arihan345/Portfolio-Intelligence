-- Confirmed gap found during the Phase 10 monitoring investigation:
-- pipeline_runs recorded FAILED status but nothing about WHY. Every
-- failure had to be reconstructed from memory/terminal scrollback
-- instead of being queryable. This column closes that gap.
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
