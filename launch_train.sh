#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=modloss
#SBTACH --output=modloss
#SBATCH --time=24:00:00
#SBATCH --account=chem007981
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

# the two gpus above seem to have a different version of cuda! avoid

echo "activate env"
# activate your own environment here
source activate /user/work/ef17148/oldstuff/ef17148/.conda/envs/graphnet

echo "lang/python/anaconda/3.8.8-2021.05-torch"
module load lang/python/anaconda/3.8.8-2021.05-torch


echo "train"
python general_train.py parameter_template.txt --file_path /user/work/ef17148/GCN/graphnet/repo_test/graphnet_LPDM_emulator/
