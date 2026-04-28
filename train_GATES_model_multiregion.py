import sys


import matplotlib.pyplot as plt

import numpy as np
import torch

import random
import copy
import wandb


sys.path.insert(0, "/user/work/ef17148/GCN/graphnet/")
sys.path.insert(1, "/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/")
from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.loss_functions import *


import time
from datetime import datetime

import argparse

import random
from pathlib import Path

import xarray as xr

import gates.training.training as gates_training
import gates
import gates.data.datasets as gates_datasets
from gates.data.load_data import get_grid
from gates.training.training_dataclasses import PathContext, TrainingContext

from gates.training.training_helperfuns import load_parameter_file, save_object, write_to_file, save_training_plots, export_results_to_netcdf, save_wandb_artifact, set_reproducibility


def train_one_epoch(model, loader, model_ctx, epoch, paths_ctx=None):
    """
    Runs a single training epoch, iterating over all batches, computing the loss, and updating model weights.
    A separate display loss (criterion_test) is tracked for monitoring without affecting gradients.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()
    mean_loss = 0.0
    for i, batch in enumerate(loader):
        features_batch, fp_batch = batch[0].to(model_ctx.device), batch[1].to(model_ctx.device)

        if len(fp_batch.shape) == 3:
            true_values = fp_batch[:,:,0].unsqueeze(-1)
        else:
            true_values = fp_batch.unsqueeze(-1)

        model_ctx.optimizer.zero_grad()

        outputs = model(features_batch)
        loss = model_ctx.criterion(outputs, true_values, fp_batch)
        mean_loss += loss.item()
        loss.backward()
        model_ctx.optimizer.step()

        with torch.no_grad():
            display_loss = model_ctx.criterion_test(outputs, true_values, fp_batch)
            running_loss += display_loss.item()

        if i % 25 == 0:
            print(f"[{epoch}, {i:5d}] Training Loss: {mean_loss/(i+1):.3f} Loss: {running_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s")
            write_to_file(f"[{epoch}, {i:5d}] Training Loss: {mean_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s", paths_ctx.updates_path)

    return running_loss / len(loader), mean_loss / len(loader)

#from .train_GATES_model import validate_and_predict, train_one_epoch

@torch.no_grad()
def validate_and_predict(model, model_ctx, loader):
    """
    Evaluates the model on a test set, collecting predictions and computing the mean loss.
    """
    model.eval()
    test_error = 0.0
    preds_list = []

    print("validating and predicting")

    for i, batch in enumerate(loader):
        features_batch, fp_batch = batch[0].to(model_ctx.device), batch[1].to(model_ctx.device)

        if len(fp_batch.shape) == 3:
            true_values = fp_batch[:,:,0].unsqueeze(-1)
        else:
            true_values = fp_batch.unsqueeze(-1)

        outputs = model(features_batch)
        test_error += model_ctx.criterion_test(outputs, true_values, fp_batch).item()
        preds_list.append(outputs)

    all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
    return test_error / len(loader), all_preds


def build_region_configs(regions_dict, shared_load_parameters):
    """
    Convert the parameter-file regions dict to the list format expected by
    load_multiregion_data, merging shared_load_parameters into each region's
    train_load_data (region-specific keys override shared ones).

    Keys in regions_dict are integer strings optionally followed by a label
    ("0", "1", or "0-SAHARA", "1-BRAZIL"). The integer prefix determines sort
    order. A unique 'name' of the form "{index}-{region}" is constructed from
    the integer prefix and the resolved 'region' field, so that two entries for
    the same physical region (e.g. "0-SAHARA" and "1-SAHARA") remain distinct.

    Args:
        regions_dict (dict): Keyed by strings like "0" or "0-SAHARA". Each value
            has 'train_load_data' and 'test_load_data' sub-dicts.
        shared_load_parameters (dict): Parameters common to all regions (e.g. 'size',
            'met_args'). Merged as base values; per-region keys take precedence.

    Returns:
        list of dict: Each entry has 'name', 'train_load_data', and 'test_load_data'.
    """
    region_configs = []
    for key in sorted(regions_dict.keys(), key=lambda k: int(k.split("-")[0])):
        region_entry = regions_dict[key]
        train_load = copy.deepcopy(shared_load_parameters)
        train_load.update(region_entry["train_load_data"])
        test_load = copy.deepcopy(region_entry.get("test_load_data", {}))
        index = key.split("-")[0]
        region_label = train_load.get("region", key)
        region_configs.append({
            "name": f"{index}-{region_label}",
            "train_load_data": train_load,
            "test_load_data": test_load,
        })
    return region_configs


def load_multiregion_data(region_configs, input_variables, datapath_args, verbose=True, load_into_memory=True, load_monthly=False):
    """
    Loads each region's train and test data sequentially, then returns concatenated train data
    and a per-region list of test data.

    Each region is loaded and optionally computed to memory before the next region begins,
    avoiding large cross-region Dask task graphs. After loading, train inputs and footprints
    are concatenated along the time dimension with integer-reindexed coordinates to prevent
    collisions between regions that share overlapping calendar dates.

    Args:
        region_configs (list of dict): Each entry must have 'train_load_data' and 'test_load_data'
            keys, matching the structure expected by load_GATES_data.
        input_variables (dict): Variable extraction settings forwarded to load_GATES_data.
        datapath_args (dict): Path overrides forwarded to load_GATES_data.
        verbose (bool): Print progress messages.
        load_into_memory (bool): If True, call .compute() on each region's inputs before moving
            to the next. Defaults to True for multiregion use to avoid cross-region Dask graphs.

    Returns:
        all_train_inputs (xr.DataArray): Concatenated train inputs with integer fp_time coord.
        all_train_fps: Concatenated train footprints with integer time coord.
        test_regions (list of dict): One entry per region with keys 'name', 'inputs', 'fp_xr', 'data'.
    """
    train_inputs_list = []
    train_fps_list = []
    test_regions = []

    for region_config in region_configs:
        train_params = copy.deepcopy(region_config["train_load_data"])
        test_params = copy.deepcopy(region_config["train_load_data"])
        test_params.update(region_config["test_load_data"])
        region_name = region_config.get("name", train_params.get("region", f"region_{len(test_regions)}"))

        if verbose:
            print(f"\n--- Loading train data for region: {region_name} ---")
        if not load_monthly:
            data_r, train_inputs_r = gates_training.load_GATES_data(
                train_params, input_variables=input_variables,
                datapath_args=datapath_args, verbose=verbose)
            train_fp_r = data_r.fp_xr
        else:
            data_r, train_inputs_r = gates_training.load_GATES_data_v2(
                train_params, input_variables=input_variables,
                datapath_args=datapath_args, verbose=verbose, load_into_memory=load_into_memory)
            train_fp_r = data_r

        if verbose:
            print(f"--- Loading test data for region: {region_name} ---")
            print("------------------------------------------")
        if not load_monthly:
            test_data_r, test_inputs_r = gates_training.load_GATES_data(
                test_params, input_variables=input_variables,
                datapath_args=datapath_args, verbose=verbose)
            test_fp_r = test_data_r.fp_xr
        else:
            test_data_r, test_inputs_r = gates_training.load_GATES_data_v2(
                test_params, input_variables=input_variables,
                datapath_args=datapath_args, verbose=verbose, load_into_memory=load_into_memory)
            test_fp_r = test_data_r

        if verbose:
            # print number of samples in each dataset
            print("------------------------------------------")
            print(f"Loaded train inputs for region '{region_name}' with {train_inputs_r.sizes['fp_time']} samples")
            print(f"Loaded test inputs for region '{region_name}' with {test_inputs_r.sizes['fp_time']} samples")
        if load_into_memory:
            if verbose:
                print(f"  computing train inputs into memory ({train_inputs_r.nbytes / 1e9:.2f} GB)...")
            train_inputs_r = train_inputs_r.compute()
            if verbose:
                print(f"  computing test inputs into memory ({test_inputs_r.nbytes / 1e9:.2f} GB)...")
            test_inputs_r = test_inputs_r.compute()

        train_inputs_list.append(train_inputs_r)
        train_fps_list.append(train_fp_r)
        test_regions.append({
            "name": region_name,
            "inputs": test_inputs_r,
            "fp_xr": test_fp_r,
        })

    # Validate consistent spatial size across all regions
    lat_sizes = [inp.sizes.get("lat") for inp in train_inputs_list]
    if len(set(lat_sizes)) > 1:
        raise ValueError(
            f"All regions must have the same spatial size ('size' in train_load_data), "
            f"but found lat sizes: {dict(zip([r['name'] for r in test_regions], lat_sizes))}.")

    # Validate consistent variable_name coordinates across regions
    ref_vars = list(train_inputs_list[0].variable_name.values)
    for i, inp in enumerate(train_inputs_list[1:], start=1):
        if list(inp.variable_name.values) != ref_vars:
            raise ValueError(
                f"variable_name coords differ between region 0 and region {i} "
                f"({test_regions[i]['name']}). Ensure all regions use identical 'variables' settings.")

    # Concatenate train data; drop duplicate timestamps arising from overlapping regions
    all_train_inputs = xr.concat(train_inputs_list, dim="fp_time").drop_duplicates(dim="fp_time")
    all_train_fps = xr.concat(train_fps_list, dim="time").drop_duplicates(dim="time")

    if verbose:
        total_before = sum(inp.sizes['fp_time'] for inp in train_inputs_list)
        n_dropped = total_before - all_train_inputs.sizes['fp_time']
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} duplicate fp_time entries from overlapping regions.")
        print(f"\nConcatenated train data: {all_train_inputs.sizes['fp_time']} total samples "
              f"across {len(region_configs)} region(s)")
        for r in test_regions:
            print(f"  Test '{r['name']}': {r['inputs'].sizes['fp_time']} samples")

    return all_train_inputs, all_train_fps, test_regions


def transform_test_region(test_inputs, test_fps, scalers, add_nan_mask, batch_size):
    """
    Transform a single test region's inputs and footprints using already-fitted scalers
    (no re-fitting), and return a DataLoader and the footprint xarray Dataset.

    The footprint Dataset mirrors the structure produced by FootprintDataset.transform():
    it contains 'fp_transformed', 'fp_original', and optionally 'fp_nan_mask' variables,
    and is used directly by calculate_losses and save_training_plots.

    Args:
        test_inputs (xr.DataArray): Stacked input DataArray (fp_time, lat, lon, variable_name).
        test_fps: Footprint DataArray or Dataset with a 'time' dimension.
        scalers (dict): Must contain 'inputs_scaler' and 'fp_scaler' (already fitted).
        add_nan_mask (bool): Whether to add fp_nan_mask and fill NaNs with zero.
        batch_size (int): Batch size for the test DataLoader.

    Returns:
        test_loader: PyTorch DataLoader (num_workers=0, no shuffle).
        test_fp_ds (xr.Dataset): Footprint Dataset ready for metric calculation and plotting.
    """
    # Transform inputs using fitted scaler
    test_scaled_inputs = scalers["inputs_scaler"].transform(test_inputs.astype("float32"))

    # Extract DataArray from Dataset if needed
    if isinstance(test_fps, xr.Dataset):
        test_fps_da = test_fps["fp"]
    else:
        test_fps_da = test_fps

    # Apply fitted fp scaler without re-fitting (replicates FootprintDataset.transform)
    transformed = scalers["fp_scaler"].transform(test_fps_da)
    test_fp_ds = xr.Dataset({
        "fp_transformed": transformed,
        "fp_original": test_fps_da,
    })
    if add_nan_mask:
        test_fp_ds = gates_datasets.add_fp_nan_mask(
            test_fp_ds, fill_nans=True, fp_var_name="fp_original")
    test_fp_ds = test_fp_ds.chunk({"time": 1})
    test_fp_ds = test_fp_ds.assign_coords(idx=("time", list(range(len(test_fp_ds.time)))))

    test_scaled_inputs, test_fp_ds = gates_datasets.trim_to_batch_size(
        test_scaled_inputs, test_fp_ds, batch_size)

    # num_workers=0 avoids forkserver deadlock — same as single-region test loader
    test_loader, _ = gates_datasets.make_dataloader(
        test_scaled_inputs, test_fp_ds, batch_size,
        randomize=False,
        dataloader_params={"num_workers": 0, "persistent_workers": False, "prefetch_factor": None},
        flatten=True,
    )
    return test_loader, test_fp_ds


def run_full_training_multiregion(model, model_ctx, training_ctx, paths_ctx,
                                   train_loader, test_region_entries, losses, epoch_so_far=0):
    """
    Full training loop for multiregion training.

    Each epoch: trains on the combined loader, then validates each region separately.
    Per-region and aggregate metrics are computed and logged. Early stopping is based
    on the aggregate test loss. Periodic plots use the first test region.
    One NetCDF file per region is exported at the end of training.

    Args:
        model: GraphSatelliteForecaster instance.
        model_ctx (ModelContext): Optimizer, loss functions, early stopping, epoch settings.
        training_ctx (TrainingContext): Device, scalers, grid, image plot settings.
        paths_ctx (PathContext): File paths for saving checkpoints and logs.
        train_loader: DataLoader over all concatenated, shuffled train regions.
        test_region_entries (list of dict): Each entry has 'name', 'loader', 'fp_ds'.
        losses (dict): Accumulated metric lists (from initialise_losses()), tracks aggregate.
        epoch_so_far (int): Epoch offset for resuming training.
    """
    write_to_file("starting multiregion training loop", paths_ctx.updates_path)

    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} ---")

        avg_train_loss, avg_train_transformed_loss = train_one_epoch(
            model, train_loader, model_ctx, epoch, paths_ctx=paths_ctx)
        print(f"Finished training epoch {epoch} with average training loss: {avg_train_loss:.4f}")

        # --- Per-region validation ---
        region_log = {}
        all_test_fp_datasets = []

        for entry in test_region_entries:
            region_name = entry["name"]
            print(f"Validating region: {region_name}")

            avg_test_loss_r, test_out_r = validate_and_predict(model, model_ctx, entry["loader"])
            outputs_orig_r = training_ctx.scalers["fp_scaler"].inverse_transform(test_out_r)

            fp_ds_r = entry["fp_ds"]
            fp_ds_r["fp_transformed_pred"] = (
                ("time", "lat", "lon"),
                test_out_r.reshape(*fp_ds_r.fp_original.shape))
            fp_ds_r["fp_pred"] = (
                ("time", "lat", "lon"),
                outputs_orig_r.reshape(*fp_ds_r.fp_original.shape))

            # Per-region metrics — use a fresh losses dict so each region's values don't accumulate
            _, computed_metrics_r = gates_training.calculate_losses(
                gates_training.initialise_losses(), fp_ds_r)

            region_log[region_name] = {
                "test_loss": avg_test_loss_r,
                "eval": computed_metrics_r["eval_metrics"],
                "trans_eval": computed_metrics_r["transformed_eval_metrics"],
                "static_mf_eval": computed_metrics_r["static_mf_eval_metrics"],
            }
            all_test_fp_datasets.append(fp_ds_r)

        # --- Aggregate metrics ---
        fp_ds_all = xr.concat(all_test_fp_datasets, dim="time")
        avg_test_loss_agg = float(np.mean([v["test_loss"] for v in region_log.values()]))

        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss_agg)

        losses, computed_metrics = gates_training.calculate_losses(
            losses, fp_ds_all)

        # --- W&B logging ---
        if model_ctx.use_wandb:
            logging_dict = {
                "epoch": epoch + 1,
                "MSE/train": avg_train_loss,
                "MSE/test": avg_test_loss_agg,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss_agg,
                "LossFn/train": avg_train_transformed_loss,
            }
            logging_dict.update({
                f"metrics_transformed/aggregate/{k}": v
                for k, v in computed_metrics["transformed_eval_metrics"].items()})
            logging_dict.update({
                f"metrics_original/aggregate/{k}": v
                for k, v in computed_metrics["eval_metrics"].items()})
            logging_dict.update({
                f"flux/{flux_mode}/{metric_name}": metric_value
                for flux_mode, metrics in computed_metrics["static_mf_eval_metrics"].items()
                for metric_name, metric_value in metrics.items()})
            for region_name, rd in region_log.items():
                logging_dict.update({
                    f"metrics_transformed/{region_name}/{k}": v
                    for k, v in rd["trans_eval"].items()})
                logging_dict.update({
                    f"metrics_original/{region_name}/{k}": v
                    for k, v in rd["eval"].items()})
                logging_dict[f"test_loss/{region_name}"] = rd["test_loss"]

            wandb.log(logging_dict, step=epoch)

        model_ctx.early_stopping(avg_test_loss_agg, model)
        if model_ctx.early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        log_text = (f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, "
                    f"Aggregate Test Loss: {avg_test_loss_agg:.4f}")
        write_to_file(log_text, paths_ctx.updates_path)
        for region_name, rd in region_log.items():
            write_to_file(
                f"  {region_name}: test_loss={rd['test_loss']:.4f}, "
                f"metrics_original={rd['eval']}", paths_ctx.updates_path)

        # Periodic plotting — from the first test region
        if epoch % model_ctx.epochs_visualise == 0:
            img_save_path = save_training_plots(
                epoch, test_region_entries[0]["fp_ds"],
                training_ctx, paths_ctx.model_path, paths_ctx.model_name)
        if epoch % (3 * model_ctx.epochs_visualise) == 0:
            if model_ctx.use_wandb:
                wandb.log({f"fps_epoch_{epoch}": wandb.Image(img_save_path)}, step=epoch)

        if epoch % model_ctx.epochs_save == 0:
            checkpoint_path = paths_ctx.model_path / f"{model_ctx.model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_ctx.optimizer.state_dict(),
                'loss': losses,
                'learning_rate': model_ctx.lr,
            }, checkpoint_path)
            if model_ctx.use_wandb:
                save_wandb_artifact(
                    model_ctx.model_name,
                    f"checkpoint_epoch_{epoch}",
                    "model",
                    f"Model checkpoint at epoch {epoch} with agg loss {avg_test_loss_agg:.4f}",
                    checkpoint_path)

        write_to_file("Finished one loop!", paths_ctx.updates_path)

    # Export one NetCDF per region
    for entry in test_region_entries:
        nc_path = paths_ctx.model_path / f"sample_predictions_test_{entry['name']}.nc"
        entry["fp_ds"].attrs.update({
            "creation_date": str(datetime.now()),
            "model_name": model_ctx.model_name,
            "region": entry["name"],
        })
        entry["fp_ds"].to_netcdf(nc_path)
        print(f"NetCDF saved for region '{entry['name']}': {nc_path}")
        if model_ctx.use_wandb:
            save_wandb_artifact(
                model_ctx.model_name,
                f"predictions_{entry['name']}",
                "dataset",
                f"Test predictions for region {entry['name']}, model {model_ctx.model_name}",
                nc_path)

    print("Finished Training.")


def train_and_save_model_multiregion(parameters, model_save_dir):
    """
    Top-level entry point for multiregion training. Handles all setup, data loading,
    scaler fitting on concatenated train data, per-region test loader construction,
    model instantiation, and the training loop.

    The parameter file must contain a 'regions' dict keyed by integer strings ("0", "1", ...),
    where each entry has 'train_load_data' and 'test_load_data' sub-dicts with region-specific
    keys (year, freq, region name). Parameters shared across all regions (size, met_args) live
    in 'shared_load_parameters' and are merged as base values before per-region keys are applied.
    All other keys (variables, dataloader, model_parameters, etc.) are shared across regions.

    Args:
        parameters (dict): Full parameter dict loaded from JSON.
        model_save_dir (str or Path): Directory under which model outputs are saved.
    """
    use_wandb = parameters.get("use_wandb", False)

    cfg = gates.config.get_config()
    verbose = parameters.get("verbose", True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    print(f"Initialising multiregion model run for model_name: {model_name}")

    paths_ctx = PathContext(
        model_save_dir=model_save_dir,
        model_name=model_name,
        model_path=model_path,
    )
    paths_ctx.make_dirs()

    seed = parameters.get("seed", 34)
    set_reproducibility(seed)
    if verbose:
        print(f"Set random seed to {seed} for reproducibility.")

    if use_wandb:
        wandb_project = parameters.get("wandb", {}).get("project", None)
        wandb_entity = parameters.get("wandb", {}).get("entity", None)
        wandb_tags = parameters.get("wandb", {}).get("tags", [])
        if wandb_project is None or wandb_entity is None:
            print("Warning: 'use_wandb' is True but no 'wandb.project' or 'wandb.entity' specified. W&B disabled.")
            use_wandb = False
            parameters["use_wandb"] = False
        if use_wandb:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=parameters,
                tags=wandb_tags,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(
        f"using device {device}, starting at " + datetime.now().strftime("%d/%m/%y %H:%M:%S"),
        paths_ctx.updates_path)

    shared_load_params = parameters.get("shared_load_parameters", {})
    region_configs = build_region_configs(parameters["regions"], shared_load_params)
    input_variables = parameters["variables"]
    datapath_args = paths_ctx.resolve_datapath_args(parameters)
    load_into_memory = parameters.get("load_into_memory", True)

    client, cluster = gates_training.make_cluster()
    write_to_file(f"loading data for {len(region_configs)} region(s)", paths_ctx.updates_path)

    load_monthly = parameters.get("load_data_monthly", False)

    all_train_inputs, all_train_fps, test_regions = load_multiregion_data(
        region_configs, input_variables, datapath_args,
        verbose=verbose, load_into_memory=load_into_memory, load_monthly=load_monthly)

    write_to_file("successfully loaded all region data", paths_ctx.updates_path)

    if cluster is not None:
        cluster.close()
        client.close()

    # Fit scalers on concatenated train data
    write_to_file("fitting scalers on concatenated train data", paths_ctx.updates_path)
    print("Fitting scalers on concatenated train data...")
    input_dataset = gates_training.setup_input_dataset(parameters, all_train_inputs)
    fp_dataset = gates_training.setup_fp_dataset(parameters, all_train_fps)

    train_scaled_inputs = input_dataset.transform(all_train_inputs)
    train_scaled_fps = fp_dataset.transform(all_train_fps)

    dataloader_info = parameters.get("dataloader", {})
    batch_size = dataloader_info.get("batch_size", 5)
    test_batch_size = dataloader_info.get("test_batch_size", 5)
    dataloader_params = dataloader_info.get("dataloader_params", {})
    if "prefetch_factor" in dataloader_params and dataloader_params["prefetch_factor"] == 0:
        dataloader_params["prefetch_factor"] = None

    train_scaled_inputs, train_scaled_fps = gates_datasets.trim_to_batch_size(
        train_scaled_inputs, train_scaled_fps, batch_size)

    train_loader, fp_labels = gates_datasets.make_dataloader(
        train_scaled_inputs, train_scaled_fps, batch_size,
        randomize=True, dataloader_params=dataloader_params, flatten=True)

    scalers = {"inputs_scaler": input_dataset.scaler, "fp_scaler": fp_dataset.scaler}

    # Build per-region test loaders using the already-fitted scalers
    add_nan_mask = dataloader_info.get("nans_to_zeros", True)
    test_region_entries = []
    for region in test_regions:
        if verbose:
            print(f"Building test loader for region: {region['name']}")
        test_loader_r, test_fp_ds_r = transform_test_region(
            region["inputs"], region["fp_xr"], scalers, add_nan_mask, test_batch_size)
        test_region_entries.append({
            "name": region["name"],
            "loader": test_loader_r,
            "fp_ds": test_fp_ds_r,
        })

    # Save scalers
    save_object(scalers, "scalers", paths_ctx.training_outputs_path, model_name,
                description=f"Input and output scalers for model {model_name}",
                use_wandb=use_wandb)

    # Image plots are drawn from the first test region
    first_region = test_regions[0]
    image_plots = random.sample(list(range(len(first_region["inputs"]))), k=min(4, len(first_region["inputs"])))
    image_dates = np.datetime_as_string(
        first_region["fp_xr"].time.values[sorted(image_plots)])
    parameters["plotted_dates"] = image_dates.tolist()
    parameters["plotted_region"] = first_region["name"]
    parameters["n_regions"] = len(region_configs)
    parameters["region_names"] = [r["name"] for r in test_regions]

    save_object(parameters, "training_settings", paths_ctx.training_outputs_path, model_name,
                file_type="json",
                description=f"Training settings and hyperparameters for model {model_name}",
                use_wandb=use_wandb)

    # Grid from first test region (all regions share the same spatial structure)
    grid, _ = get_grid(first_region["fp_xr"], parameters.get("grid_reference_fp"))
    save_object(grid, "grid", paths_ctx.training_outputs_path, model_name,
                description="Grid object used during training", use_wandb=use_wandb)

    training_ctx = TrainingContext(
        parameters, device, use_wandb, image_dates, image_plots, grid, fp_labels, scalers,
        all_train_inputs.variable_name.size, first_region["fp_xr"].lat.size)

    write_to_file("setting up model", paths_ctx.updates_path)
    model, model_ctx = gates_training.setup_GATES_model(parameters, training_ctx, paths_ctx)

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    losses = gates_training.initialise_losses()

    print("Successfully set up model. Starting multiregion training loop.")

    run_full_training_multiregion(
        model, model_ctx, training_ctx, paths_ctx,
        train_loader, test_region_entries, losses, epoch_so_far=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiregion GATES training. Example usage: "
                    "python train_GATES_model_multiregion.py parameters.json")
    parser.add_argument("file_name", help="Parameter file name")
    parser.add_argument("--file_path",
                        help="Parameter file path. By default, the path in config.yml is used.",
                        default=None)

    args = parser.parse_args()
    cfg = gates.config.get_config()

    file_path = args.file_path if args.file_path is not None else cfg.parameter_files_dir
    parameter_path = Path(file_path) / args.file_name

    parameters = load_parameter_file(parameter_path)

    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    if parameters.get("use_wandb", False):
        wandb.login()

    if parameters.get("model_save_dir", None) is None:
        model_saving_dir = cfg.save_models_dir
    else:
        model_saving_dir = Path(parameters["model_save_dir"])

    train_and_save_model_multiregion(parameters, model_save_dir=model_saving_dir)