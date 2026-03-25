#!/bin/bash
#SBATCH --job-name=SA_2014_test
#SBATCH --output=logs/%j_test.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=240G
#SBATCH --time=1:00:00

echo "Running on host: $(hostname)"
echo "Node info:"
nvidia-smi --list-gpus
echo "=== Job started at $(date) ==="

# Load packages
module load gcc python openmpi py-pip

# Paths to scripts, etc
SCRIPT=./general_train.py
JSON_CONFIG=${JSON_CONFIG:-parameter_file_paper.json}
OUTPUT_DIR=${OUTPUT_DIR:-training_output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}}
SEED=${SEED:-34}
FILE_PATH=./parameter_files/

# --- W&B setup ---------------------------------------------------------------
# If WANDB_API_KEY is not provided, default to offline to avoid crashes.
if [[ -z "${WANDB_API_KEY}" ]]; then
  export WANDB_MODE=offline
  echo "[wandb] WANDB_API_KEY not set. Using offline mode."
fi

# Construct a descriptive W&B run name: jobName_jobID
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES=""

# Ensure pip user installs are on PATH (inside container this points to ~/.local/bin)
export PATH="${HOME}/.local/bin:${PATH}"


# --- Training ---------------------------------------------------------------
echo "training..."
python "$SCRIPT" "$JSON_CONFIG" --file_path "$FILE_PATH" --output_dir "$OUTPUT_DIR" --seed "$SEED"

echo "=== Job finished at $(date) ==="