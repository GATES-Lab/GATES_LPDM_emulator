#!/bin/bash
#SBATCH --job-name=gates_env_v3_build
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=output_logs/gates_env_v3_build_slurm_%j.log

# Overlay build (pattern B): gates_env_v2.sif + editable gates install, to a
# NEW image gates_env_v3.sif. gates_env_v2.sif is left untouched as rollback.
# The packaging files are copied in via %files, so no build-time bind is needed.
set -euo pipefail
cd /lus/lfs1aip2/projects/public/b5bn/Nawid/GATES_LPDM_emulator

echo "=== build started $(date) on $(hostname) ==="
apptainer build --fakeroot \
    /projects/b5bn/data/env/gates_env_v3.sif gates_env_v3.def
echo "=== build finished $(date) ==="
