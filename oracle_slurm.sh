#!/bin/bash
#SBATCH --job-name=test_1
#SBATCH --output=logs/%j_test_1.out
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
FILE_PATH=/mnt/shared/home/kr21883/GATES_LPDM_emulator/parameter_files/

# --- W&B setup ---------------------------------------------------------------

# If WANDB_API_KEY is not provided, default to offline to avoid crashes.
if [[ -z "${WANDB_API_KEY}" ]]; then
  export WANDB_MODE=offline
  echo "[wandb] WANDB_API_KEY not set. Using offline mode."
fi

# Construct a descriptive W&B run name: jobName_jobID
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES="Baseline performance, first full training loop"

# Ensure pip user installs are on PATH (inside container this points to ~/.local/bin)
export PATH="${HOME}/.local/bin:${PATH}"

echo "[wandb] Checking/installing wandb in the container user site-packages..."
# Try to import wandb; if it fails, install it with pip --user so it persists.

#singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" \
#  python -c "import wandb; print('wandb already present:', wandb.__version__)" \
#  || singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" \
#       bash -lc 'python -m pip install --user --upgrade pip && python -m pip install --user "wandb>=0.17"'


# --- Training ---------------------------------------------------------------
echo "training..."
python "$SCRIPT" "$JSON_CONFIG" --file_path "$FILE_PATH" --output_dir "$OUTPUT_DIR" --seed "$SEED"

# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python wandb_test_job.py

# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python forecast_check.py
# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python forecast_period_drop.py
# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python float32_converter.py

echo "=== Job finished at $(date) ==="