#!/bin/bash
#SBATCH --job-name=dual_refit
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_logs/dual_refit_%j.out

# Isambard-AI launcher for the bg-head freeze/refit experiment group
# (train_dual_refit_model.py driven through run_dual_experiments_shared_data.py).
# The three arms — frozen_only / refit_trunk_frozen / refit_joint, all 150 epochs
# at bg_loss_weight 0.1 — run sequentially in this one job, sharing a single
# data load. Sizing: the 250-epoch dual_zarr run took ~3.4 h including the load
# (~40 s/epoch), so 3 x 150 epochs + one load is ~6 h; 12 h gives slack.
#
# Submit with:            sbatch launch_dual_refit_isambard.sh
# Run a single arm with:  EXP_INDEX=1 sbatch launch_dual_refit_isambard.sh
# List the arms with:
#   python run_dual_experiments_shared_data.py parameter_dual_refit_isambard.json \
#       experiments_dual_bg_refit.json --list

REPO=/projects/b5bn/public/Nawid/GATES_LPDM_emulator
SIF="${SIF:-/projects/b5bn/data/env/gates_env_v3.sif}"
PARAM_FILE="${PARAM_FILE:-parameter_dual_refit_isambard.json}"
EXPERIMENTS_FILE="${EXPERIMENTS_FILE:-experiments_dual_bg_refit.json}"

cd "${REPO}"

echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="
nvidia-smi -L || true

# --- Weights & Biases -------------------------------------------------------
# Default to ONLINE logging when W&B credentials exist (~/.netrc, written once by
# `wandb login`) and this node can reach api.wandb.ai; otherwise fall back to
# offline (an unattended job would hang on wandb.login() without credentials)
# and FLAG it loudly in this log so unsynced runs are never a surprise.
# Force a mode with e.g.:  WANDB_MODE=offline sbatch launch_dual_refit_isambard.sh
if [[ -z "${WANDB_MODE}" ]]; then
  if grep -qs "api.wandb.ai" "${HOME}/.netrc" \
      && curl -s -o /dev/null --max-time 15 https://api.wandb.ai/; then
    WANDB_MODE=online
  else
    WANDB_MODE=offline
    echo "**********************************************************************"
    echo "*** FLAG: W&B is OFFLINE for this job (missing ~/.netrc credentials"
    echo "*** or no route to api.wandb.ai from $(hostname))."
    echo "*** Runs are recorded locally only; upload them later with:"
    echo "***   wandb sync ${REPO}/wandb/wandb/offline-run-*"
    echo "**********************************************************************"
  fi
fi
export WANDB_MODE
export WANDB_DIR="${REPO}/wandb"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# /projects is a host symlink to /lus/lfs1aip2/projects (see launch_dual_zarr_isambard.sh).
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

# Optionally run just one experiment arm (EXP_INDEX=0/1/2).
INDEX_ARGS=()
if [[ -n "${EXP_INDEX}" ]]; then
  INDEX_ARGS=(--index "${EXP_INDEX}")
fi

echo "=== Refit experiments (container: ${SIF}, params: ${PARAM_FILE}, experiments: ${EXPERIMENTS_FILE}, index: ${EXP_INDEX:-all}) ==="
echo "WANDB_MODE=${WANDB_MODE}"
apptainer exec --nv \
  --bind "${BINDS}" \
  --env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1,PYTHONPATH="${REPO}" \
  "${SIF}" \
  python -u run_dual_experiments_shared_data.py "${PARAM_FILE}" "${EXPERIMENTS_FILE}" \
    --trainer train_dual_refit_model "${INDEX_ARGS[@]}"
EXIT_CODE=$?
echo "TRAIN_EXIT_CODE=${EXIT_CODE}"

if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "FLAG: W&B was OFFLINE for job ${SLURM_JOB_ID} — runs were NOT uploaded to wandb.ai."
  echo "FLAG: upload them with:  wandb sync ${REPO}/wandb/wandb/offline-run-*"
fi

echo "=== Job finished at $(date) ==="
exit ${EXIT_CODE}
