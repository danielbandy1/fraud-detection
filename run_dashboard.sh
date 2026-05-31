#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/.venv/bin/activate"
streamlit run app.py --server.port 8503 --server.address 0.0.0.0
