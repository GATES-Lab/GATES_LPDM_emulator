#!/bin/bash
#SBATCH --job-name=SAHARA_2014_1_5_100GB_paperjson
#SBATCH --output=logs/%j_SAHARA_2014_1_5_100GB_paperjson.out
#SBATCH --gpus=1
#SBATCH --mem=150G
# SBATCH --mem-per-gpu=1G
#SBATCH --time=01:00:00

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

# Run the script inside the container with GPU support
echo "training..."
singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF"  python "$SCRIPT" "$JSON_CONFIG" --file_path "$FILE_PATH" --output_dir "$OUTPUT_DIR" --seed "$SEED"

# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python check_all_nc.py

# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python forecast_check.py
# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python forecast_period_drop.py
# singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python float32_converter.py

echo "=== Job finished at $(date) ==="