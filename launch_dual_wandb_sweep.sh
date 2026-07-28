#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=480GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dual_wandb_sweep
#SBATCH --output=output_logs/dual_wandb_sweep_%j.out
#SBATCH --time=96:00:00
#SBATCH --account=chem007981
#SBATCH --qos=normal
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

# W&B sweep agent for the dual-head model.
#
# Usage:
#   1. Create the sweep (lightweight, no GPU needed):
#        python run_dual_sweep_wandb.py parameter_template_dual.json --create sweep_dual_wandb.yaml
#   2. Launch one or more agents (submit several times for parallel trials):
#        sbatch launch_dual_wandb_sweep.sh <entity/project/sweep_id> [count]

SWEEP_ID=$1
COUNT=${2:-10}

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch launch_dual_wandb_sweep.sh <sweep_id> [count]"
    exit 1
fi

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_NOTES="W&B sweep agent, shared-data, ${COUNT} trials max"

echo "python: $(which python)"
echo "sweep id: ${SWEEP_ID}, max trials: ${COUNT}"

# Loads the data ONCE, then pulls trials from the sweep until COUNT trials are
# done or the job hits its time limit.
python -u run_dual_sweep_wandb.py parameter_template_dual.json --sweep_id "$SWEEP_ID" --count "$COUNT"
echo "AGENT_EXIT_CODE=$?"

echo "=== Job finished at $(date) ==="
