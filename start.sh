#!/bin/bash
# Startup script — used by cloud platforms that run shell scripts directly
set -e

mkdir -p data/chroma_db data/legal_corpus

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 2
