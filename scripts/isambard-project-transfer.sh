#!/bin/bash

#SBATCH --job-name=project-transfer
#SBATCH --output=project-transfer.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

rsync --archive --verbose --partial --human-readable \
    /projects/b5bn/data/ \
    u6za.aip2.isambard:/projects/u6za/data/