#!/bin/bash
# Launch JupyterLab inside the GATES Apptainer container on an Isambard-AI
# compute node (interactive reservation, 1 GH200 Superchip: 72 cores, ~115 GB RAM, 1 GPU).
#
# Usage, from a login node:
#
#   bash launch/isambard/jupyter_interactive.sh            # 2 hours
#   bash launch/isambard/jupyter_interactive.sh 04:00:00   # custom walltime (max 8h on interactive)
#
# It prints a URL; in VS Code use Select Kernel -> Existing Jupyter Server and paste it.
# Ctrl-C (twice) stops the server and ends the job, so you stop being billed.
#
# Overrides: SIF=/path/to/other.sif REPO=/path/to/repo bash launch/isambard/jupyter_interactive.sh

set -euo pipefail

TIME=${1:-02:00:00}
SIF=${SIF:-/projects/b5bn/data/env/gates_env_v2.sif}
REPO=${REPO:-$HOME/GATES_LPDM_emulator}

# On the login node: request the allocation, then re-run this script on the compute node.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Requesting 1 GPU on the interactive reservation for $TIME ..."
    exec srun --reservation=interactive --gpus=1 --time="$TIME" --pty \
        env SIF="$SIF" REPO="$REPO" bash "$(readlink -f "$0")" "$TIME"
fi

# On the compute node. Nodes are shared between users, so pick a random port.
NODE=$(hostname)
PORT=$(shuf -i 20000-29999 -n 1)
# Token is passed via the environment (not the command line) so other users can't see it in `ps`.
export APPTAINERENV_JUPYTER_TOKEN=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
URL="http://$NODE:$PORT/lab?token=$APPTAINERENV_JUPYTER_TOKEN"

cat <<EOF

================================================================================
 Job $SLURM_JOB_ID on $NODE, walltime $TIME
 Container: $SIF
 gates from: $REPO

 Paste this into VS Code (Select Kernel -> Existing Jupyter Server):

   $URL

 Ctrl-C twice to stop the server and end the job.
================================================================================

EOF

exec apptainer exec --nv \
    --bind /lus,/lus/lfs1aip2/projects:/projects \
    --env PYTHONPATH="$REPO",PYTHONNOUSERSITE=1 \
    "$SIF" \
    jupyter lab --no-browser \
        --ip="$NODE" --port="$PORT" --ServerApp.port_retries=0 \
        --ServerApp.custom_display_url="http://$NODE:$PORT" \
        --notebook-dir="$REPO"
