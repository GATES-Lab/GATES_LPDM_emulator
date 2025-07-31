#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=emulate
#SBTACH --output=emulate
#SBATCH --time=24:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035
#### this is specific to the University of Bristol's BluePebble
#### make sure you modify to remove/add any relevant modules

## chem007981  SEMT030444 

# the two gpus above seem to have a different version of cuda! avoid

echo "activate env"
# activate your own environment here
conda init
conda activate new_graphnet


echo "loading modules"
module load cuda/12.4.1
module load cudnn/8.9.7.29-12
module add languages/python/3.12.9.tensorflow-2.16.1


echo "emulate"

python general_make_prediction.py training_settings_satellite_Brazil_200x200_from_50x50.json --file_path /graphnet_LPDM_emulator/trained_models/satellite_Brazil_200x200_from_50x50/ 
