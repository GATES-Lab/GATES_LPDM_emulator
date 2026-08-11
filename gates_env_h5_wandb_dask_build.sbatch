#!/bin/bash
#SBATCH --job-name=gates_env_h5_wandb_dask_build
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=01:00:00
#SBATCH --output=output_logs/gates_env_h5_wandb_dask_build_slurm_%j.log

# Builds a NEW image gates_env_h5_wandb_dask.sif. All earlier .sif files are
# different files and are never touched.
set -euo pipefail
cd /lus/lfs1aip2/projects/public/b5bn/Nawid/GATES_LPDM_emulator

echo "=== build started $(date) on $(hostname) ==="
apptainer build --fakeroot gates_env_h5_wandb_dask.sif gates_env_h5_wandb_dask.def
echo "=== build finished $(date) ==="
