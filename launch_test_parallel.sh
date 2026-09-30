#!/bin/bash
#SBATCH --job-name=test_loro_models_BRAZIL
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=225G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --array=0-3%4
#SBATCH --output=output_logs/%A_%a_test_loro_models_BRAZIL.out

set -euo pipefail

# REGIONS: CHINA, INDIA, BRAZIL, SAHARA
REGION_TO_TEST_ON="BRAZIL"
REPO=/home/b5bn/jeffc.b5bn/GATES_LPDM_emulator
SIF=/projects/b5bn/data/env/gates_env_v2.sif
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

REFERENCE_MODELS=(
    "is_MRM_small_LORO_Brazil_20260831_221430"
    "is_MRM_small_LORO_INDIA_20260831_221207"
    "is_MRM_small_LORO_NA_20260831_221523"
    "is_MRM_small_LORO_CHINA_20260831_221108"
)
MODEL_LABELS=("Brazil" "India" "Sahara" "China")
SAVE_PATHS=(
    "model_test_predictions/brazil_loro_model"
    "model_test_predictions/india_loro_model"
    "model_test_predictions/sahara_loro_model"
    "model_test_predictions/china_loro_model"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
REFERENCE_MODEL=${REFERENCE_MODELS[$TASK_ID]}
MODEL_LABEL=${MODEL_LABELS[$TASK_ID]}
SAVE_PATH=${SAVE_PATHS[$TASK_ID]}

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO}/wandb"
export WANDB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES=""
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "=== Array task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started at $(date) on $(hostname) ==="
echo "Region: ${REGION_TO_TEST_ON}"
echo "Reference model: ${REFERENCE_MODEL}"
echo "WANDB_MODE=${WANDB_MODE}"
nvidia-smi -L || true

apptainer exec --nv \
    --bind "${BINDS}" \
    --env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONPATH="${PYTHONPATH}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
    "${SIF}" \
    python scripts/predict_GATES_model.py \
    --test_year 2014 \
    --reference_model "${REFERENCE_MODEL}" \
    --save_path "${SAVE_PATH}" \
    --region "${REGION_TO_TEST_ON}"

echo "=== Array task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
