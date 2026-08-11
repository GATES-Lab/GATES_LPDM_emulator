#!/bin/bash
#SBATCH --job-name=dual_zarr
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_logs/dual_zarr_%j.out
# NB: the normal/workq QoS caps wall time at 24h; longer runs must checkpoint
# and resubmit. A 300G request can sit PENDING on a busy day — for short smoke
# runs resubmit with e.g.:  sbatch --mem=60G --time=00:30:00 launch_dual_zarr_isambard.sh

# Isambard-AI (Grace-Hopper / aarch64) launcher for the dual-head GATES model
# reading Zarr meteorology (met_archive_zarr). Runs inside the apptainer
# container; no conda/module environment is used.
# Submit with:  sbatch launch_dual_zarr_isambard.sh
# Override the parameter file with:
#   PARAM_FILE=my_params.json sbatch launch_dual_zarr_isambard.sh

REPO=/projects/b5bn/public/Nawid/GATES_LPDM_emulator
# v3 = v2 + editable install of the gates package (see gates_env_v3.def /
# gates_env_v3_build.sbatch). Fall back to gates_env_v2.sif if v3 is absent;
# PYTHONPATH below keeps imports working either way.
SIF="${SIF:-/projects/b5bn/data/env/gates_env_v3.sif}"
# Parameter file name; resolved under config.yml's parameter_files_dir by
# train_dual_model.py (i.e. ${REPO}/parameter_files/<name>). The default file
# points met loading at /projects/b5bn/data/met_archive_zarr via its
# "data_dirs" key — config.yml's met_datadir still targets the NetCDF archive
# used by other branches, so do not repoint it globally.
PARAM_FILE="${PARAM_FILE:-parameter_dual_zarr_isambard.json}"

cd "${REPO}"

echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="
nvidia-smi -L || true

# --- Weights & Biases -------------------------------------------------------
# No W&B credentials are configured on Isambard, so default to offline logging
# (an unattended batch job would otherwise hang on wandb.login()). To log
# online, run `wandb login` (or export WANDB_API_KEY=...) once on the login
# node, then submit with:  WANDB_MODE=online sbatch launch_dual_zarr_isambard.sh
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${REPO}/wandb"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# --- Container bind paths ---------------------------------------------------
# /projects is a host symlink to /lus/lfs1aip2/projects. Bind the real Lustre
# tree and re-expose it at /projects so config.yml's /projects/... paths (data,
# model_runs, parameter_files) resolve inside the container.
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

echo "=== Training (container: ${SIF}, params: ${PARAM_FILE}) ==="
echo "WANDB_MODE=${WANDB_MODE}"
apptainer exec --nv \
  --bind "${BINDS}" \
  --env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1,PYTHONPATH="${REPO}" \
  "${SIF}" \
  python -u train_dual_model.py "${PARAM_FILE}"
EXIT_CODE=$?
echo "TRAIN_EXIT_CODE=${EXIT_CODE}"

echo "=== Job finished at $(date) ==="
exit ${EXIT_CODE}
