#!/bin/bash
#SBATCH --job-name=dual_exp
#SBATCH --output=logs/%A_%a_dual_exp.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3

# One SLURM array task per experiment. The --array range must match the number of
# experiments in the experiments file (0..N-1). Each task runs exactly one experiment,
# selected via SLURM_ARRAY_TASK_ID inside run_dual_experiments.py.
#
# To run all experiments sequentially in a single job instead, drop the #SBATCH --array
# line above and replace the python call with:
#     python -u run_dual_experiments.py parameter_template_dual.json experiments_dual_example.json

echo "Running on host: $(hostname)"
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
echo "array task id: ${SLURM_ARRAY_TASK_ID}"
echo "=== Job started at $(date) ==="

module load gcc python openmpi py-pip

source "${SLURM_SUBMIT_DIR}/my_gates_env/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/my_gates_env/lib/python3.12/site-packages:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="${HOME}/.local/bin:${PATH}"

# --- W&B setup ---------------------------------------------------------------
export WANDB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}"

echo "running dual experiment ${SLURM_ARRAY_TASK_ID}..."
python -u run_dual_experiments.py parameter_template_dual.json experiments_dual_example.json --index "${SLURM_ARRAY_TASK_ID}"

echo "=== Job finished at $(date) ==="
