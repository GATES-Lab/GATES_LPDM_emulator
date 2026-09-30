#!/bin/bash
#SBATCH --job-name=compute_loro_metrics
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=output_logs/%j_compute_loro_metrics.out

set -euo pipefail

REPO=/home/b5bn/jeffc.b5bn/GATES_LPDM_emulator
SIF=/projects/b5bn/data/env/gates_env_v2.sif
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "=== Metrics job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="

apptainer exec \
    --bind "${BINDS}" \
    --env PYTHONPATH="${PYTHONPATH}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
    "${SIF}" \
    python scripts/compute_prediction_metrics.py \
    --predictions-root model_test_predictions \
    --year 2014

apptainer exec \
    --bind "${BINDS}" \
    --env PYTHONPATH="${PYTHONPATH}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
    "${SIF}" \
    python scripts/collate_prediction_metrics.py \
    --predictions-root model_test_predictions

echo "=== Metrics job finished at $(date) ==="