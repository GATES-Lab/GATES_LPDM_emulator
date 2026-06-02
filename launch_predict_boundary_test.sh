#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pred_bound
#SBATCH --output=predict_boundary_%j.out
#SBATCH --time=02:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "python: $(which python)"
echo "===================== INFERENCE ====================="
python -u predict_boundary_model.py \
    --test_year 2016 \
    --month 02 \
    --reference_model boundary_conds_test_20260601_162703 \
    --checkpoint best
echo "PREDICT_EXIT_CODE=$?"
