#!/bin/bash
#SBATCH --job-name=dual_full
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_logs/dual_full_%j.out
# NB: the normal/workq QoS caps wall time at 24h; longer runs must checkpoint
# and resubmit (or use the restricted QoS).

# Isambard-AI (Grace-Hopper / aarch64) launcher for the dual-head GATES model.
# Runs inside the apptainer container built from gates_env.def; no conda/module
# environment is used. Submit with:  sbatch launch_dual_full.sh

REPO=/projects/b5bn/public/Nawid/GATES_LPDM_emulator
SIF="${REPO}/gates_env_h5_wandb_dask.sif"
# Parameter file name; resolved under config.yml's parameter_files_dir by
# train_dual_model.py (i.e. ${REPO}/parameter_files/<name>).
PARAM_FILE=parameter_template_dual.json

cd "${REPO}"

echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="
nvidia-smi -L || true

# --- Weights & Biases -------------------------------------------------------
# No W&B credentials are configured on Isambard, so default to offline logging
# (an unattended batch job would otherwise hang on wandb.login()). To log
# online, run `wandb login` (or export WANDB_API_KEY=...) once on the login
# node, then submit with:  WANDB_MODE=online sbatch launch_dual_full.sh
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${REPO}/wandb"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# --- Container bind paths ---------------------------------------------------
# /projects is a host symlink to /lus/lfs1aip2/projects. Bind the real Lustre
# tree and re-expose it at /projects so config.yml's /projects/... paths (data,
# model_runs, parameter_files) resolve inside the container.
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

echo "=== Training (container: ${SIF}) ==="
echo "WANDB_MODE=${WANDB_MODE}"
apptainer exec --nv \
  --bind "${BINDS}" \
  --env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
  "${SIF}" \
  python -u train_dual_model.py "${PARAM_FILE}"
EXIT_CODE=$?
echo "TRAIN_EXIT_CODE=${EXIT_CODE}"

echo "=== Job finished at $(date) ==="
exit ${EXIT_CODE}
