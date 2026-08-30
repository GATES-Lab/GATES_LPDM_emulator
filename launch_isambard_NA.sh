#!/bin/bash
#SBATCH --job-name=is_NA_match_oracle_round_time
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=output_logs/%j_is_NA_match_oracle_round_time.out

# Runs inside the apptainer container; no conda/module environment is used. Submit with:
#   sbatch launch_isambard_NA.sh

REPO=/home/b5bn/jeffc.b5bn/GATES_LPDM_emulator
SIF=/projects/b5bn/data/env/gates_env_v2.sif
# Parameter file name; resolved under config.yml's parameter_files_dir by the runner
# (i.e. ${REPO}/parameter_files/<name>). Must contain a "__sweep__" section.
# Overridable at submit time:  PARAM_FILE=<name>.json sbatch launch_dual_sweep_shared_data.sh
PARAM_FILE="${PARAM_FILE:-NEW_parameter_NA.json}"

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="
nvidia-smi -L || true

# --- Weights & Biases -------------------------------------------------------
# No W&B credentials are configured on Isambard, so default to offline logging
# (an unattended batch job would otherwise hang on wandb.login()). To log
# online, run `wandb login` (or export WANDB_API_KEY=...) once on the login
# node, then submit with:  WANDB_MODE=online sbatch launch_dual_sweep_shared_data.sh
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO}/wandb"
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES=""


export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# --- Container bind paths ---------------------------------------------------
# /projects is a host symlink to /lus/lfs1aip2/projects. Bind the real Lustre
# tree and re-expose it at /projects so config.yml's /projects/... paths (data,
# model_runs, parameter_files) resolve inside the container.
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

echo "=== Sweep (container: ${SIF}, param file: ${PARAM_FILE}) ==="
echo "WANDB_MODE=${WANDB_MODE}"
apptainer exec --nv \
  --bind "${BINDS}" \
  --env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONPATH="${PYTHONPATH}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
  "${SIF}" \
  python scripts/train_GATES_model.py ${PARAM_FILE}

EXIT_CODE=$?
echo "SWEEP_EXIT_CODE=${EXIT_CODE}"

echo "=== Job finished at $(date) ==="
exit ${EXIT_CODE}