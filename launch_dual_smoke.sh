#!/bin/bash
#SBATCH --job-name=dual_smoke
#SBATCH --output=logs/%j_dual_smoke.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

echo "Running on host: $(hostname)"
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
echo "=== Job started at $(date) ==="

module load gcc python openmpi py-pip

source "${SLURM_SUBMIT_DIR}/my_gates_env/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/my_gates_env/lib/python3.12/site-packages:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="${HOME}/.local/bin:${PATH}"

echo "running dual model smoke test..."
python -u tests/dual_model_smoke_test.py

echo "=== Job finished at $(date) ==="
