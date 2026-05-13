#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBTACH --output=train
#SBATCH --time=24:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035
#### this is specific to the University of Bristol's BluePebble
#### make sure you modify to remove/add any relevant modules

## chem007981  SEMT030444 

# the two gpus above seem to have a different version of cuda! avoid

echo "activate env"
# activate your own environment here
source ~/initConda.sh
conda activate /user/work/yl18410/graphnet_bp_220324_backup


echo "loading modules"
module load cuda/12.4.1
module load cudnn/8.9.7.29-12
#module add languages/python/3.12.9.tensorflow-2.16.1


echo "train"
python general_train_boundary.py parameter_template_train_small.json --file_path parameter_files/