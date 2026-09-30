# How to Use Weights & Biases (W&B)

Weights & Biases (W&B) is an experiment tracking platform used in this project to log training metrics, save model checkpoints, and version datasets and artifacts. This guide covers everything you need to get started, and lists exactly what the training scripts log.

---

## 1. Create an Account

1. Go to [https://wandb.ai/site](https://wandb.ai/site) and click **Sign Up**.
2. You can register with a Google account, GitHub account, or email.
3. Once registered, you will land on your personal dashboard.

> **Academic use:** W&B is free for individuals and academic researchers. If you are at a university, you may also be eligible for a free **Teams** plan — see [https://wandb.ai/site/research](https://wandb.ai/site/research).

---

## 2. Install the W&B Python Package

`wandb` is included in the project environment file (`env_gates_pytorch.yml`, pinned to `wandb=0.25.1`), so if you created the environment from that file there is nothing more to install. Every training script imports `wandb`, even when `use_wandb` is `false`, so it must be installed either way.

To add it to an existing environment:

```bash
conda install -c conda-forge wandb=0.25.1
# or
pip install wandb==0.25.1
```

---

## 3. Log In from the Command Line

After installing, authenticate your machine by running:

```bash
wandb login
```

This will prompt you to paste an **API key**, which you can find at:
[https://wandb.ai/authorize](https://wandb.ai/authorize)

The key is stored in `~/.netrc` and reused by later runs. When `use_wandb` is `true`, the training scripts call `wandb.login()` at start-up, which picks up this stored key. On an HPC cluster, run `wandb login` once in an interactive session before submitting jobs.

---

## 4. How W&B is Used in This Project

W&B is used by both training scripts: `scripts/train_GATES_model.py` (single region) and `scripts/train_GATES_model_multiregion.py`. Prediction (`scripts/predict_GATES_model.py`) does not use W&B.

### 4.1 Turning it on

W&B logging is opt-in: set `"use_wandb": true` in the parameter file and provide a
`wandb` section with `project`, `entity`, and (optionally) `tags`:

```json
"use_wandb": true,
"wandb": {
    "project": "gates-review-fixes",
    "entity": "gates-lab",
    "tags": ["SAHARA", "gpu"]
}
```

See [HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md). If `use_wandb` is true but
`project` or `entity` is missing, the run prints a warning, sets `use_wandb` to `false`
and continues without W&B.

### 4.2 Run initialisation

At the start of training, a run is created from those values:

```python
wandb.init(
    entity=wandb_entity,    # from parameters["wandb"]["entity"]
    project=wandb_project,  # from parameters["wandb"]["project"]
    config=parameters,      # the full parameter file is logged as run config
    tags=wandb_tags,        # from parameters["wandb"]["tags"]
    save_code=True,         # single-region trainer only
)
```

- **Config:** the whole parameter dict is logged as the run config. The values that training writes back later (`num_features`, `plotted_dates`, and so on) are not in the config, because it is logged before they exist. `num_features` is added to the run summary instead (see 4.5).
- **`save_code=True`:** the single-region trainer saves the launching script with the run, so you can see the exact code on the run's **Code** tab. The multiregion trainer does not set it.
- **Run name and notes:** the code does not set a run name. The SLURM launch scripts set the `WANDB_NAME` and `WANDB_NOTES` environment variables, which `wandb.init` reads:

  | Launch script | `WANDB_NAME` | `WANDB_NOTES` |
  |---|---|---|
  | `launch/launch_train.sh` | `<SLURM_JOB_ID>_<job name>` | CPUs and memory of the job |
  | `launch/launch_train_file.sh` | `<SLURM_JOB_ID>_<JOB_NAME argument>` | the job name, parameter file, CPUs and memory |
  | `launch/launch_train_sweep.sh` | `<SLURM_JOB_ID>_<job name>` | CPUs, memory and the generated parameter file |

  Without these (for example when running the script directly), W&B picks a random run name.
- **Model watching:** after the model is built, `wandb.watch(model, log="all", log_freq=100)` logs histograms of the parameters and gradients every 100 batches. They appear under the **gradients/** and **parameters/** panels.
- **Local files:** W&B writes its local run files to `wandb/` in the directory you launched from (the repo root for the launch scripts). This folder is in `.gitignore`.

### 4.3 Data-loading metrics (`loading/*`)

While the data is loading, `load_GATES_data_v2` logs one point per year that it loads:

| Key | Meaning |
|---|---|
| `loading/loaded_year` | Counter of years loaded so far (1, 2, 3, …). This is the x-axis of the other `loading/*` charts. |
| `loading/train_time` | Minutes taken to load this year. Despite the name, this is logged for test years too. |
| `loading/total_time` | Cumulative loading time in minutes. |
| `loading/samples_loaded` | Number of footprints loaded for this year (0 if the year failed). |
| `loading/total_samples` | Cumulative number of footprints loaded. |

The counter and totals are shared across all the loads in a run: the training years first, then the test years (and, for multiregion runs, every region in turn). For example, two training years and one test year give `loaded_year` 1, 2 and 3. The charts therefore show one continuous series rather than restarting at 1 for the test set.

### 4.4 Per-epoch metrics (single region)

After each epoch, `run_full_training()` logs:

```python
wandb.log({
    "epoch": epoch + 1,
    "MSE/train": avg_train_loss,
    "MSE/test": avg_test_loss,
    "train/loss": avg_train_loss,
    "test/loss": avg_test_loss,
    "LossFn/train": avg_train_transformed_loss,
    # plus the evaluation metrics, flattened with prefixes:
    #   "metrics_transformed-<name>"           transformed-space metrics
    #   "metrics_original-<name>"              original-space metrics
    #   "metrics_fluxes_static/<mode>/<name>"  static-flux metrics per mode
    #   "metrics_fluxes/<name>"                flux metrics, when a flux is loaded
})
```

`MSE/*` and `train/loss` / `test/loss` hold the same values: the training loss (`criterion`) and the test loss (`criterion_test`). Every key is plotted against `epoch`.

The evaluation metrics come from `calculate_losses` (see [HOW_TO_evaluation.md](HOW_TO_evaluation.md)):

| Prefix | Metric names |
|---|---|
| `metrics_original-` | `iou` (at a threshold of 1e-5), `mae`, `mse`, `bias`, `nmae`, `corrcoef`, `corrcoef_log`, `iou_q25`, `iou_q50`, `iou_q90`, `iou_q99` |
| `metrics_transformed-` | `iou` (at a threshold of 0), `mae`, `mse`, `bias`, `nmae`, `corrcoef`, `iou_q25`, `iou_q50`, `iou_q90`, `iou_q99` |
| `metrics_fluxes_static/<mode>/` | `corrcoef`, `mae`, `mean_bias`, `r2_score`. `<mode>` is `uniform`, `checkerboard`, `checkerboard_10`, `checkerboard_25`, and `checkerboard_50` when the domain is larger than 100 × 100. |
| `metrics_fluxes/` | the same four metrics, computed with the flux loaded through the `flux` block. Only logged when a flux is loaded. |

The `iou_q*` metrics are IoUs at the 25th, 50th, 90th and 99th percentiles of the true footprint.

### 4.5 Run summary

These values are added to the run **Overview → Summary**, not plotted:

- `num_training_samples`, `num_testing_samples`
- `num_features`: the number of input features. Predict needs this value; it is also saved in `training_settings_<name>.json`.
- Multiregion only: `num_testing_samples_<region>` for each test region.

### 4.6 Training plots (`training_plots`)

Every `epochs.visualize` epochs, the 4×4 comparison figure saved to `training_imgs/<name>_<epoch>.png` is also logged as an image under the key **`training_plots`**. It shows four random test dates, fixed for the whole run (their dates are saved as `plotted_dates` in the training settings). Use the slider on the media panel to step through the epochs.

### 4.7 Artifacts

Checkpoints, scalers, grids, settings and sample predictions are saved as W&B **artifacts**: versioned files you can download or compare across runs. They are logged with `save_wandb_artifact` (`gates/training/training_helperfuns.py`), which names each artifact `<model_name>-<name>`, with underscores in `<name>` turned into hyphens:

```python
artifact = wandb.Artifact(name=f"{model_name}-{name.replace('_', '-')}", type=file_type, description=description)
artifact.add_file(path)
wandb.log_artifact(artifact)
```

Here `<ts_name>` is the timestamped run folder name (`<model_name>_<YYYYmmdd_HHMMSS>`) and `<base_name>` is the `model_name` from the parameter file.

| Artifact name | Type | File | When |
|---|---|---|---|
| `<ts_name>-scalers` | `pickle` | `training_outputs/scalers_<ts_name>.pickle` | once, after the scalers are fitted |
| `<ts_name>-training-settings` | `json` | `training_outputs/training_settings_<ts_name>.json` | once, before training |
| `<ts_name>-grid` | `pickle` | `training_outputs/grid_<ts_name>.pickle` | once, before training |
| `<ts_name>-checkpoint-epoch-<k>` | `model` | `<ts_name>_best.pt` | each time the test loss improves (EarlyStopping). See the note below. |
| `<base_name>-checkpoint-epoch-<epoch>` | `model` | `<base_name>_<epoch>.pt` | every `epochs.model_save` epochs |
| `<ts_name>-predictions` | `dataset` | `sample_predictions_test.nc` | once, at the end of training (single region) |
| `<ts_name>-predictions-<idx>-<region>` | `dataset` | `sample_predictions_test_<idx>-<region>.nc` | once per test region, at the end of training (multiregion) |

Notes:

- **Periodic checkpoints use the base name**, without the timestamp, so runs with the same `model_name` add new versions to the same artifact (`v0`, `v1`, …). The other artifacts use the timestamped name and are unique to the run.
- **Best-model artifact name:** the `<k>` in the EarlyStopping artifact is not the epoch. It is the EarlyStopping counter, the number of epochs without improvement before this one. The file is always `<ts_name>_best.pt`, a bare `state_dict` (not the dict saved in the epoch checkpoints). To find the latest best model, look at the most recent version across these artifacts, or use the local `_best.pt` file.

---

## 5. Multiregion Runs

`scripts/train_GATES_model_multiregion.py` logs the same training losses, `loading/*` metrics, summary values and artifacts, but uses a different key scheme for the evaluation metrics. `<name>` below is either `aggregate` (all test regions concatenated) or a region name such as `0-SAHARA`:

| Key | Meaning |
|---|---|
| `MSE/train`, `train/loss`, `LossFn/train` | training loss, as for a single region |
| `MSE/test`, `test/loss` | mean of the per-region test losses |
| `test_loss/<region>` | test loss for each region |
| `metrics_original/<name>/<metric>` | original-space metrics |
| `metrics_transformed/<name>/<metric>` | transformed-space metrics |
| `metrics_fluxes_static/<name>/<mode>/<metric>` | static-flux metrics |
| `fps_epoch_<epoch>` | the training plot for that epoch (see below) |

The metric names are the same as in 4.4. Differences from single-region runs:

- **Separators:** the metric keys use `/` (`metrics_original/aggregate/mse`) rather than `-` (`metrics_original-mse`), so the two kinds of run don't share charts in the same W&B project.
- **No `metrics_fluxes/*`:** flux metrics against the loaded flux are not logged. Only the static-flux metrics are.
- **Training plots:** the plot is drawn from the first test region. It is saved to disk every `epochs.visualize` epochs, but only uploaded to W&B every 3 × `epochs.visualize` epochs, and each upload gets its own key (`fps_epoch_0`, `fps_epoch_30`, …) rather than the single `training_plots` key.
- **No `save_code`:** the code is not saved with the run.

See [HOW_TO_MULTIREGION.md](HOW_TO_MULTIREGION.md) for how to set up a multiregion run.

---

## 6. Viewing Your Results

Once a run has started, go to [https://wandb.ai](https://wandb.ai) and navigate to your project (the name you set in the `wandb` section of the parameter file) to see:

- **Charts:** live training and test loss curves, the `metrics_*` families (NMAE, IoU, flux metrics, …) and the `loading/*` metrics.
- **Media:** the `training_plots` images (or `fps_epoch_*` for multiregion runs).
- **Overview:** the full `parameters` dict as the run config, the summary values from 4.5, and the notes set by the launch script.
- **Code:** the launching script (single-region runs).
- **Artifacts:** versioned checkpoints, NetCDF prediction files, grids, scalers and settings.
- **System metrics:** GPU utilisation, memory, CPU usage (logged automatically).

---

## 7. Comparing Runs

To compare multiple training runs side by side:

1. Go to your project page.
2. Select two or more runs using the checkboxes on the left.
3. Click **Compare** to overlay their metric charts.

This is useful for comparing the effect of different learning rates, model architectures, or data configurations. Tags from the parameter file (`wandb.tags`) can be used to filter and group runs. Runs launched by `scripts/train_GATES_sweep.py` are separate runs, one per sweep combination; see [HOW_TO_LAUNCH.md](HOW_TO_LAUNCH.md).

---

## 8. Running on an HPC Cluster

If you are submitting jobs via SLURM or similar, W&B will still work as long as the node has internet access. A few tips:

- Run `wandb login` once in an interactive session before submitting batch jobs. The API key will be cached in `~/.netrc` and reused automatically.
- If the compute nodes are **offline**, you can run in offline mode and sync later:
  ```bash
  # Before your job
  export WANDB_MODE=offline

  # After your job completes, sync from the login node
  wandb sync wandb/offline-run-<timestamp>-<run-id>
  ```
- To disable W&B without editing the parameter file:
  ```bash
  export WANDB_MODE=disabled
  ```
  Setting `"use_wandb": false` in the parameter file also disables it, and skips the `wandb.login()` call entirely.

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
