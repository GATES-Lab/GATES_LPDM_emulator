#!/bin/bash
#SBATCH --job-name=data_cache
#SBATCH --partition=workq
#SBATCH --account=brics.b5bn
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=output_logs/data_cache_%j.out

# Isambard-AI launcher: build the on-disk month cache of a dual-model parameter file
# (prepare_dual_data_cache.py), so training jobs with "data_cache": {"enabled": true} start
# from cached months. See How_Tos/HOW_TO_LARGE_DATASETS.md.
#
# One job, all months one after another:
#     PARAM_FILE=my_params.json sbatch launch_prepare_data_cache.sh
# Four jobs at once, each building every 4th month (they can run on different nodes):
#     PARAM_FILE=my_params.json sbatch --array=0-3 launch_prepare_data_cache.sh
# Rebuild months that are already cached:  REBUILD=1 ...     Only one split:  SPLIT=train|test ...
#
# Memory: the loader holds the full-domain met of every footprint of a month while it cuts the
# windows (~65 MiB per footprint: ~160 GiB for a month of 2,300 footprints at freq 1), hence the
# whole node's memory (the nodes report 418 GiB: 400G is the largest request that is accepted, 2026-10-05). No GPU is needed.

REPO=/projects/b5bn/public/Nawid/GATES_LPDM_emulator
SIF="${SIF:-/projects/b5bn/data/env/gates_env_v3.sif}"
PARAM_FILE="${PARAM_FILE:?set PARAM_FILE=<parameter file name>}"
SPLIT="${SPLIT:-both}"

cd "${REPO}"
echo "=== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) ==="

ARGS=(--split "${SPLIT}")
if [[ -n "${PARAM_PATH}" ]]; then
  ARGS+=(--file_path "${PARAM_PATH}")
fi
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
  ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
fi
if [[ -n "${REBUILD}" ]]; then
  ARGS+=(--rebuild)
fi

echo "=== Month cache for ${PARAM_FILE} (container: ${SIF}, args: ${ARGS[*]}) ==="
apptainer exec \
  --bind "/lus,/lus/lfs1aip2/projects:/projects" \
  --env PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1,PYTHONPATH="${REPO}" \
  "${SIF}" \
  python -u prepare_dual_data_cache.py "${PARAM_FILE}" "${ARGS[@]}"
EXIT_CODE=$?
echo "CACHE_EXIT_CODE=${EXIT_CODE}"
echo "=== Job finished at $(date) ==="
exit ${EXIT_CODE}
