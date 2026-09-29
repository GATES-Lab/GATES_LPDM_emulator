#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=gates_experiment
#SBATCH --time=18:00:00
#SBATCH --account=SEMT030444
#SBATCH --export=NONE
#SBATCH --exclude=bp1-gpu030,bp1-gpu035
#SBATCH --output=launch/logs/%x_%j.out
#
# NOTE: --account is the SLURM project code the job is charged to. Replace
# SEMT030444 with your own project code.
#
# Parametrised launcher for GATES experiments.
#
# Usage (submit from the repo root, and create launch/logs/ first):
#   sbatch --job-name=<JOB_NAME> launch/launch_train_file.sh <JOB_NAME> <PARAM_JSON>
#
#   <JOB_NAME>   : name used for wandb + logging (pass the same value to --job-name
#                  so the SLURM job name and the %x log file match).
#   <PARAM_JSON> : parameter file path relative to parameter_files_dir in config.yml,
#                  e.g. NEW_parameter_template.json

JOB_NAME="$1"
PARAM_JSON="$2"
if [ -z "$JOB_NAME" ] || [ -z "$PARAM_JSON" ]; then
    echo "Usage: sbatch --job-name=<JOB_NAME> launch/launch_train_file.sh <JOB_NAME> <PARAM_JSON>"
    echo "  e.g. sbatch --job-name=exp01_benchmark launch/launch_train_file.sh exp01_benchmark experiments/01_benchmark.json"
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-.}" || { echo "Could not cd to submit dir $SLURM_SUBMIT_DIR"; exit 1; }

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda activate gates_env

echo "job name:       ${SLURM_JOB_NAME:-$JOB_NAME}"
echo "param file:     $PARAM_JSON"
echo "cpus-per-task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "memory:         ${SLURM_MEM_PER_NODE:-?}"
echo "Active env:     ${CONDA_DEFAULT_ENV:-none}"
echo "Python path:    $(which python)"
echo "Working dir:    $(pwd)"

export WANDB_NAME="${SLURM_JOB_ID}_${JOB_NAME}"
export WANDB_NOTES="experiment: $JOB_NAME ($PARAM_JSON). Launch settings - cpus-per-task: $SLURM_CPUS_PER_TASK, memory: $SLURM_MEM_PER_NODE"

echo "starting: $JOB_NAME"
python scripts/train_GATES_model.py "$PARAM_JSON"
echo "done: $JOB_NAME"
