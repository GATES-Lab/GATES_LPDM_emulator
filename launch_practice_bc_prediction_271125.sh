#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=400GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=bcpred
#SBATCH --output=bcpred_%j.out
#SBATCH --error=bcpred_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=semt030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "Starting job on $(hostname)"

# CD to directory
cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator || { echo "CD FAILED"; exit 1; }

echo "Loading modules..."
module load cuda/12.4.1
module load cudnn/8.9.7.29-12

echo "Sourcing conda..."
source ~/initConda.sh || { echo "Conda init failed"; exit 1; }

echo "Activating environment..."
conda activate /user/work/yl18410/graphnet_bp_220324_backup || { echo "conda activate failed"; exit 1; }

echo "Environment ready. Running script..."
python Run/run_bc_prediction_practice.py

echo "Job finished."