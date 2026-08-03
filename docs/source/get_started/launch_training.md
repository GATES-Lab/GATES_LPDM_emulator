# Launch a Training Job

Once you have a [config file](config.md) and a parameter file set up (see the [Parameter File reference](../user_guide/parameter_file.md)), start a training run with:

```bash
python scripts/train_GATES_model.py parameter_file.json
```

If your parameter file is not in the `parameter_files_dir` set in `config.yml`, pass its location explicitly:

```bash
python scripts/train_GATES_model.py parameter_file.json --file_path /path/to/folder/where/file/is
```

## Launching on a SLURM Cluster

`launch_train.sh` shows a working example of a SLURM batch script that activates the environment and launches a run:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00

conda activate <your_env_name>

python scripts/train_GATES_model.py parameter_file.json
```

Adjust the `#SBATCH` resource directives (`--mem`, `--cpus-per-task`, `--account`, etc.) for your cluster.

```{note}
A fuller walkthrough of run outputs, monitoring, and what to expect during training is still to be written.
```

## Inside `scripts/train_GATES_model.py`

The training script is a thin driver over the `gates` package: all heavy lifting is delegated to `gates.training.training` (imported as `gates_training`). It defines four functions, layered from the CLI entry point down to the innermost batch loop.

```
__main__
  parse args → get_config() → load_parameter_file() → resolve model_save_dir
    │
    └─> train_and_save_model(parameters, model_save_dir)   [orchestrator: one-time setup]
           │  builds model, dataloaders, and context objects via gates_training.*
           └─> run_full_training(...)                       [epoch loop]
                  ├─> train_one_epoch(...)       per epoch   [one training pass]
                  └─> validate_and_predict(...)  per epoch   [predictions + validation loss]
```

**`train_one_epoch(model, loader, model_ctx, epoch, paths_ctx=None)`**
Runs a single training pass over the loader. Per batch: moves `(features, fp)` to the device, forward pass, computes the training loss (`criterion`) and backpropagates, then separately computes a display loss (`criterion_test`) under `no_grad` for monitoring. Returns `(mean_display_loss, mean_train_loss)`.

**`validate_and_predict(model, model_ctx, loader)`**
Decorated with `@torch.no_grad()`. Runs the model over the validation/test loader in `eval()` mode, accumulates `criterion_test`, and collects every batch's outputs. Returns `(mean_test_loss, all_preds)`.

**`run_full_training(model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader, test_fp_dataset, losses, epoch_so_far=0)`**
The epoch loop. Each epoch: trains, validates/predicts, inverse-transforms predictions back to original space via the fp scaler, writes them into `test_fp_dataset`, computes metrics (`gates_training.calculate_losses`), logs to W&B, and handles early stopping, periodic plotting, and checkpointing. Exports final predictions to NetCDF at the end.

**`train_and_save_model(parameters, model_save_dir)`**
Top-level orchestrator. Handles all one-time setup — timestamped model name, `PathContext` and output dirs, seeding, optional W&B init, train/test data loading (`load_GATES_data_v2` on a dask cluster), dataloader and scaler setup, grid creation, optional dynamic edges, `TrainingContext`, and model construction (`setup_GATES_model`) — before delegating to `run_full_training`.

State is passed around through three context dataclasses to keep argument lists manageable: `PathContext` (all output paths), `TrainingContext` (scalers, grid, device, image dates, dynamic edges), and `ModelContext` (optimizer, loss functions, early stopping, W&B flag). Note the two distinct losses used throughout: `criterion` is trained on, while `criterion_test` is only monitored and reported.


## To come: Predicting

### Predicting at a different size from training.

## To come: Multi-region training