#!/usr/bin/env bash
# AUTHORED BY ORION. Local-only; no submission, publishing or server processes.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-portfolio.lock
.venv/bin/python portfolio_check.py
.venv/bin/python -m pytest -q tests
