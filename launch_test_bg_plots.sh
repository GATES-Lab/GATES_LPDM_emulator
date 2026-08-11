#!/bin/bash
#SBATCH --partition=test
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --job-name=test_bg_plots
#SBATCH --output=test_bg_plots_%j.out
#SBATCH --time=00:20:00
#SBATCH --account=SEMT030444

echo "activate env"
source ~/initConda.sh
conda activate /user/work/yl18410/miniconda3/envs/gates_env

cd /user/work/yl18410/new_graphnet/graphnet_LPDM_emulator

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "python: $(which python)"
python -m pytest tests/test_bg_timeseries_plots.py -v -Wignore::DeprecationWarning
echo "TEST_EXIT_CODE=$?"
