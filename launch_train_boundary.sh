#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --cpus-per-task=8
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035


echo $SLURM_CPUS_PER_TASK

echo "activate env"
source ~/initConda.sh
conda activate gates_env

echo "loading modules"
module load cuda/12.4.1
module load cudnn/8.9.7.29-12

echo "train"
python train_boundary_model.py parameter_train_small_boundary.json