#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=practice010626
#SBATCH --output=practice_model_010626_%j.out
#SBATCH --time=02:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "activate env"
source ~/initConda.sh
# gates_env has the full GATES stack incl. h5netcdf (the backup env is stale).
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

# Unbuffered output + fault handler so a crash prints the Python stack
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "===================== TRAINING ====================="
python -u train_boundary_model.py parameter_practice_model_010626.json
train_status=$?
echo "TRAIN_EXIT_CODE=$train_status"

if [ $train_status -ne 0 ]; then
    echo "Training failed; skipping inference."
    exit $train_status
fi

echo "===================== INFERENCE ====================="
# Reference the model by its base name; predict resolves the timestamped dir.
python -u predict_boundary_model.py \
    --test_year 2016 \
    --month 06 \
    --reference_model practice_model_010626 \
    --checkpoint best
predict_status=$?
echo "PREDICT_EXIT_CODE=$predict_status"
exit $predict_status
