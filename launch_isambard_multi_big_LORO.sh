#!/bin/bash
#SBATCH --job-name=is_MRM_big_LORO
#SBATCH --partition=workq
#SBATCH --account=brics.u6za
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=449G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_logs/%j_is_MRM_big_LORO_SA_auto.out

# Runs inside the apptainer container; no conda/module environment is used. Submit with:
#   HOLDOUT=INDIA sbatch launch_isambard_multi_big_LORO.sh
# HOLDOUT is one of SAHARA, INDIA, CHINA, SOUTHAMERICA (required). Override the job name with
#   sbatch --job-name=is_MRM_big_LORO_INDIA ...

REPO=/home/u6za/jeffc.u6za/GATES_LPDM_emulator
SIF=/projects/u6za/data/env/gates_env_v2.sif
# Parameter file name; resolved under config.yml's parameter_files_dir by the runner
# (i.e. ${REPO}/parameter_files/<name>). Must contain a "__sweep__" section.
# Overridable at submit time:  PARAM_FILE=<name>.json sbatch launch_dual_sweep_shared_data.sh
PARAM_FILE="${PARAM_FILE:-NEW_parameter_multiregion.json}"

# Base model name written into the run (the "_LORO_<HOLDOUT>" suffix is added by the script).
# Overridable:  MODEL_NAME=<name> HOLDOUT=INDIA sbatch launch_isambard_multi_big_LORO.sh
MODEL_NAME="${MODEL_NAME:-is_MRM_big}"
HOLDOUT="${HOLDOUT:?Set HOLDOUT to the region to leave out, e.g. HOLDOUT=INDIA}"

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
# Run name shared by the W&B run and (via scontrol) the Slurm job: <MODEL_NAME>_LORO_<HOLDOUT>.
# This matches the model_name the training script saves (which then adds a timestamp).
RUN_NAME="${MODEL_NAME}_LORO_${HOLDOUT}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || echo "Could not rename Slurm job (non-fatal)"
export WANDB_NAME="${SLURM_JOB_ID}_${RUN_NAME}"
export WANDB_NOTES="Big multiregion LORO, held out ${HOLDOUT}, with batch size 20 and 6 workers and 1e-4 learning rate"


export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# --- Container bind paths ---------------------------------------------------
# /projects is a host symlink to /lus/lfs1aip2/projects. Bind the real Lustre
# tree and re-expose it at /projects so config.yml's /projects/... paths (data,
# model_runs, parameter_files) resolve inside the container.
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

echo "=== Sweep (container: ${SIF}, param file: ${PARAM_FILE}, holdout: ${HOLDOUT}) ==="
echo "WANDB_MODE=${WANDB_MODE}"
apptainer exec --nv \
  --bind "${BINDS}" \
  --env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONPATH="${PYTHONPATH}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
  "${SIF}" \
  python scripts/train_GATES_model_multiregion.py "${PARAM_FILE}" --model_name "${MODEL_NAME}" --holdout "${HOLDOUT}"

EXIT_CODE=$?
echo "SWEEP_EXIT_CODE=${EXIT_CODE}"

echo "=== Job finished at $(date) ==="
exit ${EXIT_CODE}