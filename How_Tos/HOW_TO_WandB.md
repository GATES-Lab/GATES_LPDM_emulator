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
At the start of `train_and_save_model()`, a run is created with:
```python
wandb.init(
    project="BoundaryCondition-Prediction",
    config=parameters
)
```
This creates a new run inside the `BoundaryCondition-Prediction` (different names can be used) project on your W&B account, and automatically logs all values in `parameters` as the run configuration.

### Metric Logging
Inside the training loop in `run_full_training()`, metrics are logged each epoch:
```python
wandb.log({
    "train/loss": avg_train_loss,
    "test/loss": avg_test_loss,
    "test/NMAE": nmae_val,
    ...
}, step=epoch)
```
These appear as live charts on your run page as training progresses.

### Artifact Versioning
Model checkpoints, predictions, grids, and transform parameters are all saved as W&B **artifacts** — versioned files you can download or compare across runs. For example:
```python
artifact = wandb.Artifact(name="myModel-checkpoint", type="model")
artifact.add_file("path/to/checkpoint.pt")
wandb.log_artifact(artifact)
```

### Data-Loading Summary (dual model)
For dual-model runs, `load_dual_data()` builds a summary of the raw data it actually loaded —
sample counts, time ranges, array dims/sizes, load times, the resolved `background_setup`,
and (for in-memory data) NaN counts and min/max/mean/std per component. Because the data can
be loaded **once** and shared across several experiments (`run_dual_experiments_shared_data.py`),
this summary is computed at load time and then recorded separately by **every** run that
consumes the bundle:

- in the run's **Config** tab under `data_summary` (filter/group runs by it in the UI), and
- as `data_summary_<model_name>.json` in the run's `training_outputs/` directory
  (also logged as an artifact when W&B is enabled).

Value statistics read every array element, so by default they are only computed when the data
is already in memory (`load_into_memory: true`). Override this with the optional top-level
parameter-file key `"data_summary_stats"`: set it to `true` to force statistics even for lazy
(dask-backed) data, or to `false` to record metadata only (dims, sizes, time ranges).

---

## 5. Viewing Your Results

Once a run has started, go to [https://wandb.ai](https://wandb.ai) and navigate to your project (`BoundaryCondition-Prediction`) to see:

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

## 8. Useful Links

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