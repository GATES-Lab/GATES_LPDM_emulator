#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=480GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dual_sweep
#SBATCH --output=output_logs/dual_sweep_%j.out
#SBATCH --time=96:00:00
#SBATCH --account=chem007981
#SBATCH --qos=normal
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# --- W&B setup ---------------------------------------------------------------
# The run name per experiment is built inside train_and_save_model from the
# SLURM job info + each experiment's name, so we only set notes here.
export WANDB_NOTES="5 CPU, shuffle, shared-data sweep"

echo "python: $(which python)"

# --- Training ----------------------------------------------------------------
# Loads the data ONCE from the base file, then runs every experiment in the
# sweep file reusing that data. All experiments must share the same data-loading
# config (train_load_data / test_load_data / variables / background_setup).
echo "training..."
python -u run_dual_experiments_shared_data.py parameter_template_dual.json experiments_dual_bg_sweep.json
echo "SWEEP_EXIT_CODE=$?"

echo "=== Job finished at $(date) ==="
