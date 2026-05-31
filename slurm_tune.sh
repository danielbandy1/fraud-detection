#!/bin/bash
#SBATCH --job-name=fraud-tune
#SBATCH --account=coa_ich248_uksr
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/tune_%A_%a.out
#SBATCH --error=logs/tune_%A_%a.err

set -uo pipefail

mkdir -p logs models

PYTHON=/mnt/gpfs3_amd/share/apps/amd/Miniforge3-24.9.0-0/bin/python3

# Create venv once; all array tasks share it (task 0 builds, others wait)
if [ ! -f ".venv/bin/activate" ]; then
    "$PYTHON" -m venv .venv
    source .venv/bin/activate
    pip install -q lightgbm scikit-learn pandas numpy optuna pyarrow
else
    source .venv/bin/activate
fi

echo "Array task ${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

python tune.py \
    --n-trials 20 \
    --storage "sqlite:///models/optuna.db" \
    --study-name "fraud-lgbm" \
    --sample-frac 0.3

echo "Task ${SLURM_ARRAY_TASK_ID} done"
