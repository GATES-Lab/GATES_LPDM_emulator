#!/bin/bash
#SBATCH --job-name=isambardai_training_india
#SBATCH --output=isambardai_training_india.out
#SBATCH --gpus=1

#SBATCH --time=00:05:00

echo "Running on host: $(hostname)"
echo "Node info:"
nvidia-smi --list-gpus
echo "=== Job started at $(date) ==="

# Load singularity module
module load brics/singularity-multi-node/1.0-r1

# Paths to singularity image and scripts, etc
SIF=./pytorch_24.01-py3.sif
SCRIPT=./general_train.py
JSON_CONFIG=parameter_template_train_small.json
FILE_PATH=/home/b5s/jeffc.b5s/graphnet_LPDM_emulator/parameter_files/

# Run the script inside the container with GPU support
echo "training..."
singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv "$SIF" python "$SCRIPT" "$JSON_CONFIG" --file_path "$FILE_PATH"
#singularity exec --bind=/projects/b5s:/projects/b5s:rw --nv ./pytorch_24.01-py3.sif python general_train.py parameter_template_train_small.json --file_path /home/b5s/jeffc.b5s/graphnet_LPDM_emulator/parameter_files/

echo "=== Job finished at $(date) ==="