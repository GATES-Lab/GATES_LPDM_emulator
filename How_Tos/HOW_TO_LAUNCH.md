# How to Launch Jobs

Once you have a [config file](HOW_TO_CONFIG.md) and a parameter file set up (see the [Parameter File reference](HOW_TO_PARAMETER_FILE.md)), start a training run with:

```bash
python scripts/train_GATES_model.py parameter_file.json
```

If your parameter file is not in the `parameter_files_dir` set in `config.yml`, pass its location explicitly:

```bash
python scripts/train_GATES_model.py parameter_file.json --file_path /path/to/folder/where/file/is
```

## Launching on a SLURM Cluster

There are three SLURM batch scripts in `launch/`:

| Script | Use it for |
|--------|-----------|
| `launch/launch_train.sh` | A single run. The parameter file is written into the script. |
| `launch/launch_train_file.sh` | A single run. The job name and parameter file are passed on the command line, so one script serves every experiment. |
| `launch/launch_train_sweep.sh` | One job of a sweep. It is submitted by `scripts/train_GATES_sweep.py`, not by hand (see [Launching a Sweep](#launching-a-sweep)). |

All three follow the same pattern. They request a GPU, activate the environment inside the job, and run `scripts/train_GATES_model.py`:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=my_run
#SBATCH --time=18:00:00
#SBATCH --account=<your_project_code>
#SBATCH --export=NONE
#SBATCH --output=launch/logs/%x_%j.out

# Activate the environment inside the job (--export=NONE gives a clean shell).
# PYTHONNOUSERSITE=1 stops packages in ~/.local from shadowing the ones in the env.
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda activate gates_env

python scripts/train_GATES_model.py parameter_file.json
```

The scripts assume the environment is called `gates_env`, which is the name in `env_gates_pytorch.yml`. If you created yours under a different name, change the `conda activate` line.

### Before you submit

- **Submit from the repo root.** The `--output` path and `scripts/...` are relative to the directory you run `sbatch` from.
- **Create the log directory first:** `mkdir -p launch/logs`. SLURM will not create it. Logs are written as `<job-name>_<job-id>.out` (`%x` = job name, `%j` = job ID).
- **Set your project code.** `--account` is the SLURM project code the job is charged to. The scripts in `launch/` have the GATES project code, so replace it with your own.

### A single run: `launch_train.sh`

Edit the parameter file name at the bottom of the script (and the `--job-name`), then:

```bash
sbatch launch/launch_train.sh
```

### A single run from the command line: `launch_train_file.sh`

Pass the job name and the parameter file as arguments. The parameter file path is relative to `parameter_files_dir` in `config.yml`:

```bash
sbatch --job-name=<JOB_NAME> launch/launch_train_file.sh <JOB_NAME> <PARAM_JSON>

# e.g.
sbatch --job-name=exp01_benchmark launch/launch_train_file.sh exp01_benchmark experiments/01_benchmark.json
```

Pass the same name to `--job-name` and as the first argument. The SLURM job name sets the log file name (`%x`). The argument sets the W&B run name (`<job-id>_<JOB_NAME>`) and is recorded in `WANDB_NOTES` together with the parameter file.

### Resource directives

Adjust the `#SBATCH` resource directives for your cluster:

| Directive | What it controls |
|-----------|------------------|
| `--partition` / `--gres=gpu:1` | The GPU partition and one GPU per job. |
| `--mem` | Total node memory. Also sizes the Dask cluster (see below). |
| `--cpus-per-task` | CPU cores for the job. Drives the number of Dask workers. |
| `--time` | Wall-clock limit. Checkpoints are saved every `epochs.model_save` epochs, but there is no resume option yet: a job that hits the limit has to be restarted from scratch. |
| `--account` | The SLURM project code the job is charged to. Change it to your own. |
| `--export=NONE` | Starts from a clean environment (hence the explicit `conda activate`). |
| `--exclude` | Nodes to avoid. The scripts exclude two BluePebble nodes that caused problems. |

The scripts export a few W&B variables (`WANDB_NAME`, `WANDB_NOTES`) derived from the SLURM job so runs are easy to trace back to their launch settings — see the [W&B guide](HOW_TO_WandB.md).

### Dask cluster and per-worker memory

Data loading (`load_GATES_data_v2`) runs on a local Dask cluster that `train_and_save_model` spins up automatically via `make_cluster()` (in `gates/training/training.py`). The cluster is **sized from the SLURM allocation** using two environment variables SLURM sets for the job:

- `SLURM_CPUS_PER_TASK` — from `--cpus-per-task`
- `SLURM_MEM_PER_NODE` — from `--mem` (in MB)

The sizing logic is:

```python
n_cpus = SLURM_CPUS_PER_TASK
mem_gb = SLURM_MEM_PER_NODE / 1024          # MB → GB

if n_cpus < 4:
    # too few cores — skip Dask, use the synchronous scheduler
    ...

n_workers      = max(1, n_cpus - 2)         # reserve 2 CPUs for python processes
mem_per_worker = 0.8 * mem_gb / n_workers   # 80% of node RAM, split evenly
```

So each worker runs a single thread (`threads_per_worker=1`) with a hard `memory_limit` of `mem_per_worker`. The key points when tuning your job:

- **Two CPUs are held back** for the main process / scheduler, so `n_workers = cpus_per_task − 2`. With `--cpus-per-task=5` you get **3 workers**.
- **Only 80% of `--mem` is handed to Dask**; the remaining 20% is left as headroom for the driver process, PyTorch, and CUDA. That 80% is divided **evenly** across workers, so per-worker memory is `0.8 × mem / n_workers`.
- With `--mem=180GB` and `--cpus-per-task=5`: `0.8 × 180 / 3 ≈ 48GB` per worker.
- If a worker exceeds its `memory_limit`, Dask starts spilling to disk (`local_directory="/tmp"`) and, past a higher threshold, kills and restarts the worker. If you hit worker restarts or spilling during loading, **raise `--mem` or lower `--cpus-per-task`** (fewer workers → more memory each).
- Requesting **fewer than 4 CPUs** disables the cluster entirely and loading falls back to Dask's synchronous scheduler — fine for small runs, slow for large ones.

A dashboard link is printed at startup for monitoring worker memory and task progress.

#### Parameter-file settings that affect memory

Besides `--mem` and `--cpus-per-task`, two settings in the parameter file control peak memory (see the [Parameter File reference](HOW_TO_PARAMETER_FILE.md)):

| Setting | Effect on memory |
|---------|------------------|
| [`variables.load_into_memory`](HOW_TO_PARAMETER_FILE.md#variables) | `true` loads the full-domain met for every needed timestamp before cropping: fast, but the highest peak memory. `false` keeps the met lazy while cropping. |
| [`variables.chunk_size`](HOW_TO_PARAMETER_FILE.md#variables) | With `load_into_memory: false`, the met is read and cropped in blocks of this many samples, so peak memory scales with `chunk_size` rather than the dataset. Lower it if workers spill or restart during loading. |

For large runs, a good starting point is `load_into_memory: false` with `chunk_size` around 64 (as in [`NEW_parameter_template.json`](../parameter_files/NEW_parameter_template.json)). If loading is memory-bound, lower `chunk_size` before raising `--mem`.

## Launching a Sweep

To submit many training jobs from a single parameter file, use `scripts/train_GATES_sweep.py`. Add a `__sweep__` section to the parameter JSON; everything else in the file is shared across all jobs. The sweep launcher writes one parameter file per job into `<param_dir>/sweep_configs/` and submits a SLURM job for each, using `launch/launch_train_sweep.sh` as the batch script.

A ready-to-use example is [`parameter_files/NEW_parameter_template_sweep.json`](../parameter_files/NEW_parameter_template_sweep.json): the standard template plus a two-job list (a larger dynamic encoder, and a different `PixelWeightedMSELoss` weighting).

Keys in the sweep section use **dot-notation** to address nested parameters at any depth (e.g. `model_parameters.num_blocks`). There are two ways to define the jobs:

### Cartesian product

Map each parameter to a list of values. One job is launched for **every combination** of values:

```json
"__sweep__": {
    "learning_rate": [1e-4, 5e-5, 1e-5],
    "model_parameters.num_blocks": [2, 4, 6]
}
```

→ 3 × 3 = **9 jobs**.

### Explicit list of configs

Provide a **list of dicts**, where each dict is one job. Use this when you want specific parameter combinations rather than the full grid:

```json
"__sweep__": [
    {"learning_rate": 1e-4, "model_parameters.num_blocks": 2},
    {"learning_rate": 5e-5, "model_parameters.num_blocks": 6}
]
```

→ **2 jobs** (only the combinations listed).

### Running the sweep

```bash
# Preview the jobs without submitting them
python scripts/train_GATES_sweep.py my_config.json --dry-run

# Submit the jobs
python scripts/train_GATES_sweep.py my_config.json
```

The sweep launcher can be run from any directory. It always submits the jobs from the repo root, so `scripts/train_GATES_model.py` resolves inside each job.

**Options:**

| Option | Description |
|--------|-------------|
| `--dry-run` | Print the jobs and their `sbatch` commands without submitting. The per-job configs are still written. |
| `--output-dir` | Where to write the per-job configs (default: `<param_dir>/sweep_configs/`). |

**Job names** come from the `model_name` in the sweep file (the file name is used only if `model_name` is missing). For `"model_name": "my_model"`, job `NNN` gets:

| What | Name |
|------|------|
| `model_name` in its config | `my_model_sweep_NNN` |
| Training output folder | `my_model_sweep_NNN_<YYYYmmdd_HHMMSS>/` |
| SLURM job name | `sweep_my_model_NNN` |
| SLURM log | `launch/logs/sweep_my_model_NNN_<job-id>.out` |
| W&B run name | `<job-id>_sweep_my_model_NNN` |
| Generated config | `sweep_configs/<param_file_stem>_sweep_NNN_<values>.json` |

Each generated config also records `sweep_id` and `sweep_combination`, so a run can be traced back to its sweep entry.

## Multi-region training

To train a single model on footprints from several regions at once, use
`scripts/train_GATES_model_multiregion.py`. It takes the same kind of parameter file and CLI as
the standard trainer:

```bash
python scripts/train_GATES_model_multiregion.py parameter_file.json
```

On SLURM, swap the `python scripts/train_GATES_model.py ...` line in your batch script for this
command. The same `--mem` / `--cpus-per-task` sizing applies.

> **Note:** Multi-region training does **not** work with sweeps yet. `scripts/train_GATES_sweep.py`
> only targets the standard single-region trainer.

See [HOW_TO_MULTIREGION.md](HOW_TO_MULTIREGION.md) for the parameter file format and outputs.

## Predicting

Once a model is trained, run inference with `scripts/predict_GATES_model.py`. All data and model
configuration is read from the reference model's saved training settings
(`training_outputs/training_settings_<model_name>.json`), so **no separate parameter file is
needed** — you only point at the trained model and the year to predict:

```bash
python scripts/predict_GATES_model.py --test_year 2019 --reference_model my_model_20240115_143022
```

`--reference_model` accepts either the full timestamped directory name (e.g.
`my_model_20240115_143022`) for an exact match, or a base name (e.g. `my_model`) to use the most
recently created matching run.

On SLURM, swap the `python scripts/train_GATES_model.py ...` line in your batch script for the
predict command.

See [HOW_TO_PREDICT.md](HOW_TO_PREDICT.md) for all arguments and outputs.
