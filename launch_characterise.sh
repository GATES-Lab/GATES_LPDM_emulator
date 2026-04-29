#!/bin/bash
#SBATCH --mem=150GB
#SBATCH --job-name=characterise
#SBATCH --time=01:00:00
#SBATCH --account=SEMT030444
#### this is specific to the University of Bristol's BluePebble
#### make sure you modify to remove/add any relevant modules

## chem007981  SEMT030444 

# Usage:
#   sbatch launch_characterise.sh --date 201602 --region SAHARA
#   sbatch launch_characterise.sh --date "201[4-6]*" --region SOUTHAMERICA --countries BRAZIL,PERU
#   sbatch launch_characterise.sh --list-regions
#
# Default if no arguments: --date 201602 --region SAHARA

echo "activate env"
# Source conda setup (non-interactive-safe approach)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gates_env

echo "characterise run with args: $@"
cd "$SLURM_SUBMIT_DIR"

# Ensure absolute imports like graphnet_LPDM_emulator.* resolve
REPO_PARENT="$(dirname "$SLURM_SUBMIT_DIR")"
export PYTHONPATH="$REPO_PARENT:${PYTHONPATH}"

# If no arguments provided, use defaults
if [ $# -eq 0 ]; then
    echo "No arguments provided. Using defaults: --date 201602 --region SAHARA"
    python vis/characterisation_run.py --date 201602 --region SAHARA
else
    python vis/characterisation_run.py "$@"
fi
