#!/bin/bash

REGION_TO_TEST_ON="CHINA"


#SBATCH --job-name=test_on_all_models_for_region_${REGION_TO_TEST_ON}
#SBATCH --output=logs/%j_test_on_all_models_for_region_${REGION_TO_TEST_ON}.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=110G
#SBATCH --time=12:00:00

echo "Region: ${REGION_TO_TEST_ON}"
echo "job name: $SLURM_JOB_NAME"

echo "starting from gates_env, activating in file also"
echo "Active env: ${CONDA_DEFAULT_ENV:-none}"
echo "Python path: $(which python)"

echo "=== Job started at $(date) ==="

# Load packages
module load gcc python openmpi py-pip

source ~/my_gates_env/bin/activate
export PYTHONPATH="${HOME}/my_gates_env/lib/python3.12/site-packages:${PYTHONPATH}"

export PYTHONNOUSERSITE=1

# Ensure pip user installs are on PATH (inside container this points to ~/.local/bin)
export PATH="${HOME}/.local/bin:${PATH}"


# --- Training ---------------------------------------------------------------
#echo "testing with Brazil model..."
#python predict_GATES_model.py --test_year 2014 --reference_model BRAZIL_proper_20260428_230038 --save_path model_test_predictions/brazil_model --region ${REGION_TO_TEST_ON}

echo "testing with India model..."
python predict_GATES_model.py --test_year 2014 --reference_model INDIA_proper_train_freq_2_test_freq_25_20260428_162444 --save_path model_test_predictions/india_model --region ${REGION_TO_TEST_ON}

echo "testing with Sahara model..."
python predict_GATES_model.py --test_year 2014 --reference_model SAHARA_proper_20260428_111328 --save_path model_test_predictions/sahara_model --region ${REGION_TO_TEST_ON}

echo "testing with China model..."
python predict_GATES_model.py --test_year 2014 --reference_model CHINA_proper_20260428_093201 --save_path model_test_predictions/china_model --region ${REGION_TO_TEST_ON}


echo "=== Job finished at $(date) ==="