#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=320GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=bnd_full
#SBATCH --output=boundary_full_%j.out
#SBATCH --time=48:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "python: $(which python)"
echo "===================== FULL TRAINING ====================="
python -u train_boundary_model.py parameter_template_boundary_full.json
echo "TRAIN_EXIT_CODE=$?"
