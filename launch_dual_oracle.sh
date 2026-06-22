#!/bin/bash
#SBATCH --job-name=dual_test
#SBATCH --output=logs/%j_dual_test.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

source "${SLURM_SUBMIT_DIR}/my_gates_env/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/my_gates_env/lib/python3.12/site-packages:${PYTHONPATH}"

export PYTHONNOUSERSITE=1

# --- W&B setup ---------------------------------------------------------------
# Construct a descriptive W&B run name: jobName_jobID
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES="dual-head (footprint + background) small test"

# Ensure pip user installs are on PATH (inside container this points to ~/.local/bin)
export PATH="${HOME}/.local/bin:${PATH}"


# --- Training ---------------------------------------------------------------
echo "training..."
python -u train_dual_model.py parameter_template_dual.json

echo "=== Job finished at $(date) ==="
