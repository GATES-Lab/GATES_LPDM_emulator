# Launch a Training Job

Once you have a [config file](config.md) and a parameter file set up (see the [Parameter File reference](../user_guide/parameter_file.md)), start a training run with:

```bash
python train_GATES_model.py parameter_file.json
```

If your parameter file is not in the `parameter_files_dir` set in `config.yml`, pass its location explicitly:

```bash
python train_GATES_model.py parameter_file.json --file_path /path/to/folder/where/file/is
```

## Launching on a SLURM Cluster

`launch_train.sh` shows a working example of a SLURM batch script that activates the environment and launches a run:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00

conda activate <your_env_name>

python train_GATES_model.py parameter_file.json
```

Adjust the `#SBATCH` resource directives (`--mem`, `--cpus-per-task`, `--account`, etc.) for your cluster.

```{note}
A fuller walkthrough of run outputs, monitoring, and what to expect during training is still to be written.
```
