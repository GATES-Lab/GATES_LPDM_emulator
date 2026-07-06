#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=480GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dual_full
#SBATCH --output=output_logs/dual_full_%j.out
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


echo "python: $(which python)"


# --- Training ---------------------------------------------------------------
echo "training..."
python -u train_dual_model.py parameter_template_dual_small.json
echo "TRAIN_EXIT_CODE=$?"

echo "=== Job finished at $(date) ==="
