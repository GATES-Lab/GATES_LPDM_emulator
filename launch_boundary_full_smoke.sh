#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=bnd_smoke
#SBATCH --output=boundary_full_smoke_%j.out
#SBATCH --time=03:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "python: $(which python)"
echo "===================== SMOKE TRAINING (one month, size 50) ====================="
python -u train_boundary_model.py parameter_template_boundary_full_smoke.json
echo "TRAIN_EXIT_CODE=$?"
