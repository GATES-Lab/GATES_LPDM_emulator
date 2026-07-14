#!/bin/bash
#SBATCH --job-name=gates_env_v2_build
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=04:00:00
#SBATCH --output=output_logs/gates_env_v2_build_slurm_%j.log

# Clean from-scratch build of the corrected canonical gates_env.def to a NEW
# image /projects/b5bn/data/env/gates_env_v2.sif. Re-validates the whole recipe
# (base pull + cartopy / torch-scatter recompile) with the dask/distributed fix
# baked in. All existing .sif files are left untouched.
set -euo pipefail
cd /lus/lfs1aip2/projects/public/b5bn/Nawid/GATES_LPDM_emulator

echo "=== build started $(date) on $(hostname) ==="
apptainer build --fakeroot /projects/b5bn/data/env/gates_env_v2.sif gates_env.def
echo "=== build finished $(date) ==="
