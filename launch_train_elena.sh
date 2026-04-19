#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=75GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=fullyear_test_train
#SBATCH --time=10:00:00
#SBATCH --account=SEMT030444
#SBATCH --export=NONE

## chem007981 SEMT030444



export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda init
conda activate gates_env

echo "cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "gpus: $SLURM_GPUS"
echo "memory: $SLURM_MEM_PER_NODE"
echo "job name: $SLURM_JOB_NAME"
echo "ntasks: $SLURM_NTASKS"

echo "starting from gates_env, activating in file also"
echo "Active env: ${CONDA_DEFAULT_ENV:-none}"
echo "Python path: $(which python)"

echo "train"
python train_GATES_model.py NEW_parameter_template_gpu_new.json
echo "done test job"







