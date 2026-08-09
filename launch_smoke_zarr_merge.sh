#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mem=32GB
#SBATCH --job-name=smoke_zarr_merge
#SBATCH --output=smoke_zarr_merge_%j.out
#SBATCH --time=00:30:00
#SBATCH --account=SEMT030444

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_MODE=disabled

echo "python: $(which python)"
echo "branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
python -u tests/smoke_test_zarr_merge.py
echo "SMOKE_EXIT_CODE=$?"
