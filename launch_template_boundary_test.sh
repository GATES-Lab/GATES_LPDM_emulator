#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=tmpl_bound
#SBATCH --output=template_boundary_%j.out
#SBATCH --time=02:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

echo "activate env"
source ~/initConda.sh
# Use gates_env: it has the full GATES stack incl. h5netcdf (the launch_train_boundary.sh
# env graphnet_bp_220324_backup is stale and missing it).
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

# Unbuffered output + fault handler so any crash prints the Python stack.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "python: $(which python)"
echo "===================== TRAINING (parameter_template_boundary.json) ====================="
python -u train_boundary_model.py parameter_template_boundary.json
echo "TRAIN_EXIT_CODE=$?"
