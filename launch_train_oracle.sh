#!/bin/bash
#SBATCH --job-name=INDIA_v3features_whole_data_test_preprocess_met_data
#SBATCH --output=logs/%j_INDIA_v3features_whole_data_test_preprocess_met_data.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=400G
#SBATCH --time=24:00:00

echo "Running on host: $(hostname)"
echo "Node info:"

# Restrict to only the GPU(s) allocated by SLURM
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
echo "cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "gpus: $SLURM_GPUS"
echo "memory: $SLURM_MEM_PER_NODE"
echo "job name: $SLURM_JOB_NAME"
echo "ntasks: $SLURM_NTASKS"

echo "starting from gates_env, activating in file also"
echo "Active env: ${CONDA_DEFAULT_ENV:-none}"
echo "Python path: $(which python)"

echo "=== Job started at $(date) ==="

# Load packages
module load gcc python openmpi py-pip

source ~/my_gates_env/bin/activate
export PYTHONPATH="${HOME}/my_gates_env/lib/python3.12/site-packages:${PYTHONPATH}"

export PYTHONNOUSERSITE=1

# --- W&B setup ---------------------------------------------------------------
# Construct a descriptive W&B run name: jobName_jobID
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES="5 CPU, shuffle"

# Ensure pip user installs are on PATH (inside container this points to ~/.local/bin)
export PATH="${HOME}/.local/bin:${PATH}"


# --- Training ---------------------------------------------------------------
echo "training..."
python train_GATES_model.py NEW_parameter_template_gpu_new3.json

echo "=== Job finished at $(date) ==="