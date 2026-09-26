#!/usr/bin/env bash
# Loads .env from the repo root (never committed - see .gitignore) and
# runs dbt against the existing Phase 3 warehouse. dbt itself has no
# built-in .env support, so credentials are exported here and read by
# ~/.dbt/profiles.yml via env_var(), never hard-coded in the project.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

set -a
source "$REPO_ROOT/.env"
set +a

cd "$SCRIPT_DIR"
dbt "$@"
