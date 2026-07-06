#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=480GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dual_exp
#SBATCH --output=output_logs/dual_exp_%j.out
#SBATCH --time=96:00:00
#SBATCH --account=chem007981
#SBATCH --qos=normal
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

# All experiments run sequentially in this single job (no --index, so
# run_dual_experiments.py iterates over every experiment in the file in order).
#
# To run them in parallel instead (one SLURM array task per experiment), add:
#     #SBATCH --array=0-3
# and pass --index "${SLURM_ARRAY_TASK_ID}" to the python call below.

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "python: $(which python)"

# --- Experiments ------------------------------------------------------------
echo "running all dual experiments sequentially..."
python -u run_dual_experiments.py parameter_template_dual.json experiments_dual_bg_sweep.json
echo "TRAIN_EXIT_CODE=$?"

echo "=== Job finished at $(date) ==="
