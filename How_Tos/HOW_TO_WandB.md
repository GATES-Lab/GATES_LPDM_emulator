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