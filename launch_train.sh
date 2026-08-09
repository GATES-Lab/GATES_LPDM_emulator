#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=gatesimports2_SAHARA_dynamic_wind__latlon_dist_deg2rad
#SBATCH --time=18:00:00
#SBATCH --account=SEMT030444
#SBATCH --export=NONE
#SBATCH --exclude=bp1-gpu030,bp1-gpu035

## chem007981 SEMT030444



export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda init
conda activate new_gates_env

echo "cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "gpus: $SLURM_GPUS"
echo "memory: $SLURM_MEM_PER_NODE"
echo "job name: $SLURM_JOB_NAME"
echo "ntasks: $SLURM_NTASKS"

echo "starting from new_gates_env, activating in file also"
echo "Active env: ${CONDA_DEFAULT_ENV:-none}"
echo "Python path: $(which python)"

echo "train"
export WANDB_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export WANDB_NOTES="Launch settings - cpus-per-task: $SLURM_CPUS_PER_TASK, memory: $SLURM_MEM_PER_NODE"

python train_GATES_model.py NEW_parameter_template_gpu_new_edge_exps_4.json
echo "done test job"







