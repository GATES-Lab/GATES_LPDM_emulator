# How to Use Weights & Biases (W&B)

Weights & Biases (W&B) is an experiment tracking platform used in this project to log training metrics, save model checkpoints, and version datasets and artifacts. This guide covers everything you need to get started.

---

## 1. Create an Account

1. Go to [https://wandb.ai/site](https://wandb.ai/site) and click **Sign Up**.
2. You can register with a Google account, GitHub account, or email.
3. Once registered, you will land on your personal dashboard.

> **Academic use:** W&B is free for individuals and academic researchers. If you are at a university, you may also be eligible for a free **Teams** plan — see [https://wandb.ai/site/research](https://wandb.ai/site/research).

---

## 2. Install the W&B Python Package

```bash
pip install wandb
```

---

## 3. Log In from the Command Line

After installing, authenticate your machine by running:

```bash
wandb login
```

This will prompt you to paste an **API key**, which you can find at:
[https://wandb.ai/authorize](https://wandb.ai/authorize)

On an HPC cluster you may need to run this once in an interactive session before submitting jobs.

---

## 4. How W&B is Used in This Project

The training code integrates W&B in three main ways:

### Experiment Initialisation
W&B logging is opt-in: set `"use_wandb": true` in the parameter file and provide a
`wandb` section with `project`, `entity`, and (optionally) `tags` — see
[HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md). If `use_wandb` is true but
`project` or `entity` is missing, the run prints a warning and continues with W&B
disabled.

At the start of `train_and_save_model()`, a run is created from those values:
```python
wandb.init(
    entity=wandb_entity,   # from parameters["wandb"]["entity"]
    project=wandb_project,  # from parameters["wandb"]["project"]
    config=parameters,      # the full parameter file is logged as run config
    tags=wandb_tags,        # from parameters["wandb"]["tags"]
)
```
This creates a new run inside the project you named on your W&B account, and
automatically logs all values in `parameters` as the run configuration.

### Metric Logging
Inside the training loop in `run_full_training()`, metrics are logged each epoch:
```python
wandb.log({
    "epoch": epoch + 1,
    "MSE/train": avg_train_loss,
    "MSE/test": avg_test_loss,
    "train/loss": avg_train_loss,
    "test/loss": avg_test_loss,
    "LossFn/train": avg_train_transformed_loss,
    # plus the evaluation metrics dicts, flattened with prefixes:
    #   "metrics_transformed-<name>"        (transformed-space eval metrics)
    #   "metrics_original-<name>"           (original-space eval metrics)
    #   "metrics_fluxes_static/<mode>/<name>"  (static-flux metrics per mode)
    #   "metrics_fluxes/<name>"             (flux metrics, when a flux var is present)
})
```
These appear as live charts on your run page as training progresses. The
`metrics_fluxes*` families are the flux-evaluation metrics — see
[HOW_TO_evaluation.md](HOW_TO_evaluation.md).

### Artifact Versioning
Model checkpoints, predictions, grids, and transform parameters are all saved as W&B **artifacts** — versioned files you can download or compare across runs. For example:
```python
artifact = wandb.Artifact(name="myModel-checkpoint", type="model")
artifact.add_file("path/to/checkpoint.pt")
wandb.log_artifact(artifact)
```

---

## 5. Viewing Your Results

Once a run has started, go to [https://wandb.ai](https://wandb.ai) and navigate to your project (the name you set in the `wandb` section of the parameter file) to see:

- **Charts** — live training and validation loss curves, NMAE, IoU, flux metrics, and more.
- **Config** — the full `parameters` dict logged at the start of the run.
- **Artifacts** — versioned checkpoints, NetCDF prediction files, grids, and transform parameters.
- **System metrics** — GPU utilisation, memory, CPU usage (logged automatically).

---

## 6. Comparing Runs

To compare multiple training runs side by side:

1. Go to your project page.
2. Select two or more runs using the checkboxes on the left.
3. Click **Compare** to overlay their metric charts.

This is useful for comparing the effect of different learning rates, model architectures, or data configurations.

---

## 7. Running on an HPC Cluster

If you are submitting jobs via SLURM or similar, W&B will still work as long as the node has internet access. A few tips:

- Run `wandb login` once in an interactive session before submitting batch jobs — the API key will be cached and reused automatically.
- If the compute nodes are **offline**, you can run in offline mode and sync later:
  ```bash
  # Before your job
  export WANDB_MODE=offline

  # After your job completes, sync from the login node
  wandb sync wandb/run-<run-id>
  ```
- To disable W&B entirely without changing the code:
  ```bash
  export WANDB_MODE=disabled
  ```

---

## 8. Hyperparameter Sweeps (dual-head model)

The dual-head model has a W&B sweep setup for hyperparameter optimisation:

- **Sweep space:** [`parameter_files/sweep_dual_wandb.yaml`](../parameter_files/sweep_dual_wandb.yaml) — parameter names are dotted paths into the base parameter JSON (e.g. `loss_functions.bg_loss_weight`), deep-merged onto it per trial.
- **Driver:** [`run_dual_sweep_wandb.py`](../run_dual_sweep_wandb.py) — creates the sweep or runs an agent. Each agent loads the data **once** and reuses it for all its trials, so data-loading keys (`train_load_data`, `test_load_data`, `variables`, `background_setup`, `data_dirs`, `load_into_memory`) cannot be swept (this is checked and rejected).
- **Launcher:** [`launch_dual_wandb_sweep.sh`](../launch_dual_wandb_sweep.sh) — SLURM job that runs one agent.

### Workflow

```bash
# 1. Create the sweep (lightweight API call, fine on the login node):
python run_dual_sweep_wandb.py parameter_template_dual.json --create sweep_dual_wandb.yaml
# -> prints e.g. "Created sweep: my-entity/my-project/abc123"

# 2. Launch agents on GPU nodes. Each pulls trials until <count> are done or the
#    job times out. Submit the script several times to run trials in parallel:
sbatch launch_dual_wandb_sweep.sh my-entity/my-project/abc123 10
sbatch launch_dual_wandb_sweep.sh my-entity/my-project/abc123 10
```

Progress and the best trial appear on the sweep page under **Sweeps** in the project.

### The optimisation objective

The sweep minimises **`best/objective`** = `best/test_loss_fp` + `objective_bg_weight` × `best/test_loss_bg`, logged every epoch by `run_full_training` in `train_dual_model.py`. Notes:

- The `best/*` metrics are **best-so-far** (monotone), so the objective is robust to when a run stops and to the bg head's late-training overfitting.
- The two per-head bests may occur at **different epochs** — the objective assumes you select each head from its own best epoch, not a single checkpoint.
- `objective_bg_weight` (default `1.0`) can be set under `"sweep": {"objective_bg_weight": ...}` in the base parameter file. Keep it **fixed for the whole sweep**, otherwise trials are not comparable. It is deliberately independent of the *training* weight `loss_functions.bg_loss_weight`, which the sweep can vary.
- Trials that crash (e.g. OOM on a large config) are marked failed and the agent moves on to the next trial.

### The data-loading summary run

When the data is loaded **once and shared** across several runs (`run_dual_sweep_wandb.py` agents and `run_dual_experiments_shared_data.py`), the loading summary belongs to no single training run. It is therefore logged to its **own W&B run**, created right after the shared load finishes and closed *before* any training run starts:

- **Name / type:** `<slurm-job-id>_<job-name>_dataload`, with `job_type="data_loading"` and an extra `data_loading` tag — filter by either to find (or hide) these runs. It joins the same `wandb.group` as the training runs it fed.
- **Contents:** per-month loading times as the table `data_loading/month_load_mins`, plus summary scalars (`train/test/total_load_mins`, `n_train_samples`, `n_test_samples`). The run config records the data-loading keys (`train_load_data`, `test_load_data`, `variables`, `background_setup`, `load_into_memory`) so you can see exactly what was loaded.
- One such run is created **per process/agent** (each agent loads its own copy of the data), so two sweep agents produce two data-loading runs.
- Ordinary single runs (`train_dual_model.py` directly) are unchanged — no extra run is created.

Unit-tested by [`tests/test_dataload_summary_run.py`](../tests/test_dataload_summary_run.py) (stubbed loader, offline W&B — no data or GPU needed).

---

## 9. Useful Links

| Resource | Link |
|---|---|
| W&B Homepage | https://wandb.ai/site |
| Sign Up | https://wandb.ai/login |
| API Key | https://wandb.ai/authorize |
| Python Docs | https://docs.wandb.ai |
| Quickstart Guide | https://docs.wandb.ai/quickstart |
| Logging Metrics | https://docs.wandb.ai/guides/track/log |
| Artifacts Guide | https://docs.wandb.ai/guides/artifacts |
| HPC / Offline Mode | https://docs.wandb.ai/guides/track/environment-variables |
| Academic Plan | https://wandb.ai/site/research |