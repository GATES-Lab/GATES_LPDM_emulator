#!/bin/bash
#SBATCH --job-name=test_all_loro_models_CHINA
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=output_logs/%j_test_all_loro_models_CHINA.out

set -euo pipefail

#REGIONS: CHINA, INDIA, BRAZIL, SAHARA

REGION_TO_TEST_ON="CHINA"
REPO=/home/b5bn/jeffc.b5bn/GATES_LPDM_emulator
SIF=/projects/b5bn/data/env/gates_env_v2.sif
BINDS="/lus,/lus/lfs1aip2/projects:/projects"

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO}/wandb"
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES=""
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="
echo "Region: ${REGION_TO_TEST_ON}"
echo "WANDB_MODE=${WANDB_MODE}"
nvidia-smi -L || true

run_prediction() {
	local description=$1
	local reference_model=$2
	local save_path=$3

	echo "Testing with LORO ${description} model..."
	apptainer exec --nv \
		--bind "${BINDS}" \
		--env WANDB_MODE="${WANDB_MODE}",WANDB_DIR="${WANDB_DIR}",PYTHONPATH="${PYTHONPATH}",PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1 \
		"${SIF}" \
		python scripts/predict_GATES_model.py \
		--test_year 2014 \
		--reference_model "${reference_model}" \
		--save_path "${save_path}" \
		--region "${REGION_TO_TEST_ON}"
}

# --- Predictions ------------------------------------------------------------
run_prediction \
	"Brazil" \
	"is_MRM_small_LORO_Brazil_20260831_221430" \
	"model_test_predictions/brazil_loro_model"

run_prediction \
	"India" \
	"is_MRM_small_LORO_INDIA_20260831_221207" \
	"model_test_predictions/india_loro_model"

run_prediction \
	"Sahara" \
	"is_MRM_small_LORO_NA_20260831_221523" \
	"model_test_predictions/sahara_loro_model"

run_prediction \
	"China" \
	"is_MRM_small_LORO_CHINA_20260831_221108" \
	"model_test_predictions/china_loro_model"

echo "=== Job finished at $(date) ==="