#!/bin/bash
#SBATCH --job-name=SAHARA_445GB_201415_test_2016
#SBATCH --output=logs/%j_SAHARA_445GB_201415_test_2016.out
#SBATCH --gpus=1
#SBATCH --mem=445G
# SBATCH --mem-per-gpu=1G
#SBATCH --time=12:00:00

echo "Running on host: $(hostname)"
echo "Node info:"
nvidia-smi --list-gpus
echo "=== Job started at $(date) ==="

# Load singularity module
module load brics/singularity-multi-node/1.0-r1

# Paths to singularity image and scripts, etc
SIF=./pytorch_24.01-py3.sif
SCRIPT=./general_train.py
JSON_CONFIG=${JSON_CONFIG:-parameter_file_paper.json}
OUTPUT_DIR=${OUTPUT_DIR:-training_output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}}
SEED=${SEED:-34}
FILE_PATH=/home/b5s/jeffc.b5s/graphnet_LPDM_emulator/parameter_files/

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
singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" \
  python -c "import wandb; print('wandb already present:', wandb.__version__)" \
  || singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" \
       bash -lc 'python -m pip install --user --upgrade pip && python -m pip install --user "wandb>=0.17"'

# (Optional) sanity print
singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" \
  python -c "import sys, site; print('[wandb] sys.executable:', sys.executable); print('[wandb] user site:', site.getusersitepackages()); import wandb; print('[wandb] version:', wandb.__version__)"

# --- Training ---------------------------------------------------------------
echo "training..."
singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF"  python "$SCRIPT" "$JSON_CONFIG" --file_path "$FILE_PATH" --output_dir "$OUTPUT_DIR" --seed "$SEED"

# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python wandb_test_job.py

# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python forecast_check.py
# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python forecast_period_drop.py
# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python float32_converter.py

echo "=== Job finished at $(date) ==="