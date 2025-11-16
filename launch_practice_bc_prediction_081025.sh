#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=400GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=modloss
#SBATCH --output=modloss
#SBATCH --time=48:00:00
#SBATCH --account=semt030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

# Possible project codes I could use chem007981, semt030444
# the two gpus above seem to have a different version of cuda! avoid
cd user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

echo "activate env"
# activate your own environment here
module load cuda/12.4.1
module load cudnn/8.9.7.29-12

source ~/initConda.sh
#conda activate /user/work/yl18410/miniconda3/envs/new_graphnet_v2
conda activate /user/work/yl18410/graphnet_bp_220324_backup

python Run/run_bc_prediction_practice.py
#python model_predictions.py
echo "boxcox trained"