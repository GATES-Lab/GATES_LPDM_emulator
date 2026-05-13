#!/bin/bash
#SBATCH --job-name=brazil_freq_sweep
#SBATCH --output=logs/%A_%a_brazil_freq_sweep.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=230G
#SBATCH --time=36:00:00
#SBATCH --array=0-3

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch --array=0-7 launch_wandb_sweep_agent_brazil_array.sh <entity/project/sweep_id> [count_per_agent]"
  exit 1
fi

SWEEP_PATH="$1"
COUNT_PER_AGENT="${2:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Running on host: $(hostname)"
echo "Sweep path: ${SWEEP_PATH}"
echo "Array task id: ${TASK_ID}"
echo "Runs per agent: ${COUNT_PER_AGENT}"

echo "=== Job started at $(date) ==="

module load gcc python openmpi py-pip

source ~/my_gates_env/bin/activate
export PYTHONPATH="${HOME}/my_gates_env/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PATH="${HOME}/.local/bin:${PATH}"

# Do not force a fixed WANDB_NAME for sweep agents.
# A fixed name makes multiple distinct runs appear identical in W&B charts.
unset WANDB_NAME

wandb agent --count "${COUNT_PER_AGENT}" "${SWEEP_PATH}"

echo "=== Job finished at $(date) ==="
