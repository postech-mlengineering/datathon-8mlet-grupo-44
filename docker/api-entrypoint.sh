#!/bin/sh
set -eu

cd /app/feature_repo
feast apply
feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"

cd /app
exec uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
