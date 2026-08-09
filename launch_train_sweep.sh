#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --account=SEMT030444
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

# Job name and output path are set by launch_sweep.py via sbatch --job-name and --output.
# SWEEP_PARAM_FILE and SWEEP_JOB_NAME are injected via --export=NONE,SWEEP_PARAM_FILE=...,SWEEP_JOB_NAME=...

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda init
conda activate new_gates_env

echo "cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "gpus: $SLURM_GPUS"
echo "memory: $SLURM_MEM_PER_NODE"
echo "job name: $SLURM_JOB_NAME"
echo "ntasks: $SLURM_NTASKS"
echo "sweep param file: $SWEEP_PARAM_FILE"

echo "Active env: ${CONDA_DEFAULT_ENV:-none}"
echo "Python path: $(which python)"

if [ -z "$SWEEP_PARAM_FILE" ]; then
    echo "ERROR: SWEEP_PARAM_FILE is not set. Submit this script via launch_sweep.py."
    exit 1
fi

if [ ! -f "$SWEEP_PARAM_FILE" ]; then
    echo "ERROR: Parameter file not found: $SWEEP_PARAM_FILE"
    exit 1
fi

PARAM_DIR=$(dirname "$SWEEP_PARAM_FILE")
PARAM_FILE=$(basename "$SWEEP_PARAM_FILE")

export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES="Sweep job - cpus-per-task: $SLURM_CPUS_PER_TASK, memory: $SLURM_MEM_PER_NODE, params: $SWEEP_PARAM_FILE"

echo "train: $PARAM_FILE from $PARAM_DIR"
python train_GATES_model.py "$PARAM_FILE" --file_path "$PARAM_DIR"
echo "done: $SLURM_JOB_NAME"
