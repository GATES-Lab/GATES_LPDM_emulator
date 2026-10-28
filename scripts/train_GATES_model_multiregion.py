import sys


import matplotlib.pyplot as plt

import numpy as np
import torch

import random
import copy
import wandb


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


def build_region_configs(regions_dict, shared_load_parameters, global_flux_args=None):
    """
    Convert the parameter-file regions dict to the list format expected by
    load_multiregion_data, merging shared_load_parameters into each region's
    train_load_data (region-specific keys override shared ones), and resolving
    each region's flux settings.

    Keys in regions_dict are integer strings optionally followed by a label
    ("0", "1", or "0-SAHARA", "1-BRAZIL"). The integer prefix determines sort
    order. A unique 'name' of the form "{index}-{region}" is constructed from
    the integer prefix and the resolved 'region' field, so that two entries for
    the same physical region (e.g. "0-SAHARA" and "1-SAHARA") remain distinct.

    Flux: global_flux_args (the top-level "flux" block) is the default for every
    region. A region can override it with its own "flux" block, placed next to
    'train_load_data' (not inside it, since every train_load_data key is passed to
    LoadSquareSatelliteData). The merge is one level deep: region keys replace
    global keys, so a region that sets 'convert_units_args' replaces the whole
    sub-dict. A region with "get_flux": false loads no flux.

    Args:
        regions_dict (dict): Keyed by strings like "0" or "0-SAHARA". Each value
            has 'train_load_data' and 'test_load_data' sub-dicts, and optionally a
            'flux' dict.
        shared_load_parameters (dict): Parameters common to all regions (e.g. 'size',
            'met_args'). Merged as base values; per-region keys take precedence.
        global_flux_args (dict, optional): The top-level "flux" block, used as the
            default flux settings for every region. Defaults to None.

    Returns:
        list of dict: Each entry has 'name', 'train_load_data', 'test_load_data',
        and 'flux' (the region's resolved flux settings, or None if neither the
        global block nor the region sets any).

    Raises:
        ValueError: If a region's 'flux' is not a dict, or if 'flux' is placed
            inside 'train_load_data' or 'test_load_data'.
    """
    region_configs = []
    for key in sorted(regions_dict.keys(), key=lambda k: int(k.split("-")[0])):
        region_entry = regions_dict[key]
        for load_key in ("train_load_data", "test_load_data"):
            if "flux" in region_entry.get(load_key, {}):
                raise ValueError(f"Region '{key}': 'flux' is inside '{load_key}'. Put the region's "
                                 f"'flux' block next to 'train_load_data', not inside it.")
        region_flux_override = region_entry.get("flux", {})
        if not isinstance(region_flux_override, dict):
            raise ValueError(f"Region '{key}': 'flux' should be a dict, got {type(region_flux_override).__name__}.")

        train_load = copy.deepcopy(shared_load_parameters)
        train_load.update(region_entry["train_load_data"])
        test_load = copy.deepcopy(region_entry.get("test_load_data", {}))
        # deepcopy so regions don't share nested dicts (e.g. convert_units_args)
        region_flux = {**copy.deepcopy(global_flux_args or {}), **copy.deepcopy(region_flux_override)}
        index = key.split("-")[0]
        region_label = train_load.get("region", key)
        region_configs.append({
            "name": f"{index}-{region_label}",
            "train_load_data": train_load,
            "test_load_data": test_load,
            "flux": region_flux or None,
        })
    return region_configs


def load_multiregion_data(region_configs, input_variables, datapath_args, train_flux=False, verbose=True, load_into_memory=True, use_wandb=False):
    """
    Loads each region's train and test data sequentially, then returns concatenated train data
    and a per-region list of test data.

    Each region is loaded and optionally computed to memory before the next region begins,
    avoiding large cross-region Dask task graphs. After loading, train inputs and footprints
    are concatenated along the time dimension. Duplicate timestamps that arise when regions
    share overlapping calendar dates are dropped (first occurrence kept), applying the same
    positional mask to inputs and footprints so they stay aligned. Reassigning integer time
    coordinates to keep every sample instead is left as a future improvement (see the
    commented-out lines).

    Flux is loaded per region with that region's resolved settings (region_config['flux'],
    see build_region_configs). Flux metrics only use the test set, so test data always
    loads flux (if the region's 'get_flux' is on), while train data loads it only if
    train_flux is True (i.e. the loss is flux-weighted).

    Args:
        region_configs (list of dict): Each entry must have 'train_load_data' and 'test_load_data'
            keys, matching the structure expected by load_GATES_data_v2, and optionally
            'flux' (the region's flux settings; None for no flux).
        input_variables (dict): Variable extraction settings forwarded to each data loader.
        datapath_args (dict): Path overrides forwarded to each data loader.
        train_flux (bool): If True, also load flux for the train data. Set it to
            gates_training.loss_uses_flux(parameters). Defaults to False.
        verbose (bool): Print per-region progress messages. Defaults to True.
        load_into_memory (bool): If True, call .compute() on each region's inputs before moving
            to the next region. Defaults to True to avoid cross-region Dask task graphs.
        use_wandb (bool): Passed to load_GATES_data_v2 to enable per-year W&B loading metrics
            with a continuous counter across all regions. Defaults to False.

    Returns:
        all_train_inputs (xr.DataArray): Concatenated train inputs (fp_time, lat, lon, variable_name).
        all_train_fps (xr.Dataset): Concatenated train footprints (time, lat, lon).
        test_regions (list of dict): One entry per region with keys 'name', 'inputs', 'fp_xr'.
    """
    train_inputs_list = []
    train_fps_list = []
    test_regions = []
    # Shared loading state so W&B loading metrics stay continuous across every
    # region's train and test load rather than restarting at 1 each time.
    wandb_loading_state = gates_training.initialise_wandb_loading() if use_wandb else None

    for region_config in region_configs:
        train_params = copy.deepcopy(region_config["train_load_data"])
        test_params = copy.deepcopy(region_config["train_load_data"])
        test_params.update(region_config["test_load_data"])
        region_name = region_config.get("name", train_params.get("region", f"region_{len(test_regions)}"))
        region_flux = region_config.get("flux")
        if train_flux and not (region_flux and region_flux.get("get_flux", True)):
            # fail before loading: the train concat and the loss both need flux in every region
            raise ValueError(f"The loss is flux-weighted (weight_label='flux'), but flux is off for "
                             f"region '{region_name}'. Enable \"get_flux\" for every region.")

        if verbose:
            print(f"\n--- Loading train data for region: {region_name} ---")
        data_r, train_inputs_r = gates_training.load_GATES_data_v2(
            train_params, input_variables=input_variables,
            datapath_args=datapath_args, flux_args=region_flux if train_flux else None, verbose=verbose,
            load_into_memory=load_into_memory,
            use_wandb=use_wandb, wandb_state=wandb_loading_state)
        train_fp_r = data_r

        if verbose:
            print(f"--- Loading test data for region: {region_name} ---")
            print("------------------------------------------")
        test_data_r, test_inputs_r = gates_training.load_GATES_data_v2(
            test_params, input_variables=input_variables,
            datapath_args=datapath_args, flux_args=region_flux, verbose=verbose,
            load_into_memory=load_into_memory,
            use_wandb=use_wandb, wandb_state=wandb_loading_state)
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

    # Concatenate train data across regions.
    all_train_inputs = xr.concat(train_inputs_list, dim="fp_time")
    all_train_fps = xr.concat(train_fps_list, dim="time")

    # Drop duplicate timestamps that arise when regions share overlapping calendar
    # dates. Keep the first occurrence and apply the same positional mask to inputs
    # (fp_time) and footprints (time) so the two stay aligned.
    time_values = all_train_fps["time"].values
    _, first_occurrence = np.unique(time_values, return_index=True)
    keep_positions = np.sort(first_occurrence)
    n_dropped = len(time_values) - len(keep_positions)
    if n_dropped > 0:
        all_train_inputs = all_train_inputs.isel(fp_time=keep_positions)
        all_train_fps = all_train_fps.isel(time=keep_positions)
        if verbose:
            print(f"Dropped {n_dropped} duplicate timestamp(s) from concatenated train data")

    # A future improvement is to instead reassign integer coords to keep every sample:
    #all_train_inputs = all_train_inputs.assign_coords(
    #    fp_time=np.arange(len(all_train_inputs.fp_time)))
    #all_train_fps = all_train_fps.assign_coords(
    #    time=np.arange(len(all_train_fps.time)))

    if verbose:
        print(f"\nConcatenated train data: {all_train_inputs.sizes['fp_time']} total samples "
              f"across {len(region_configs)} region(s)")
        for r in test_regions:
            print(f"  Test '{r['name']}': {r['inputs'].sizes['fp_time']} samples")

    return all_train_inputs, all_train_fps, test_regions


def summarise_region_flux(test_regions, resolved_region_flux):
    """
    One summary line per region saying whether its test set has any non-NaN flux,
    i.e. whether the region will get flux metrics.

    Args:
        test_regions (list of dict): From load_multiregion_data; each has 'name' and 'fp_xr'.
        resolved_region_flux (dict): Region name -> resolved flux settings (or None),
            from build_region_configs.

    Returns:
        list of str: e.g. "  0-SAHARA: test flux yes (flux_time=2016)".
    """
    lines = []
    for region in test_regions:
        name, test_fps = region["name"], region["fp_xr"]
        flux_cfg = resolved_region_flux.get(name) or {}
        if not flux_cfg or not flux_cfg.get("get_flux", True):
            lines.append(f"  {name}: test flux off")
            continue
        has_flux = (isinstance(test_fps, xr.Dataset) and "flux" in test_fps
                    and bool(test_fps["flux"].notnull().any()))
        lines.append(f"  {name}: test flux {'yes' if has_flux else 'NO (all NaN)'} "
                     f"(flux_time={flux_cfg.get('flux_time')})")
    return lines


def transform_test_region(test_inputs, test_fps, scalers, add_nan_mask, batch_size, fp_labels=None):
    """
    Transform a single test region's inputs and footprints using already-fitted scalers
    (no re-fitting), and return a DataLoader and the footprint xarray Dataset.

    The footprint Dataset mirrors the structure produced by FootprintDataset.transform():
    it contains 'fp_transformed', 'fp_original', and optionally 'fp_nan_mask' variables,
    and is used directly by calculate_losses and save_training_plots.

    If test_fps has a 'flux' variable, it is added to the returned Dataset (trimmed to
    the same timestamps) so calculate_losses can compute flux metrics. It is only fed
    to the DataLoader if 'flux' is in fp_labels (a flux-weighted loss); otherwise it
    is attached after the loader is built, so the loader yields the same labels as
    training.

    Args:
        test_inputs (xr.DataArray): Stacked input DataArray (fp_time, lat, lon, variable_name).
        test_fps: Footprint DataArray or Dataset with a 'time' dimension.
        scalers (dict): Must contain 'inputs_scaler' and 'fp_scaler' (already fitted).
        add_nan_mask (bool): Whether to add fp_nan_mask and fill NaNs with zero.
        batch_size (int): Batch size for the test DataLoader.
        fp_labels (list, optional): Labels of the training DataLoader. If given, the
            test DataLoader labels must match them. Defaults to None (no check, and
            flux is never fed to the loader).

    Returns:
        test_loader: PyTorch DataLoader (num_workers=0, no shuffle).
        test_fp_ds (xr.Dataset): Footprint Dataset ready for metric calculation and plotting.

    Raises:
        ValueError: If 'flux' is in fp_labels but test_fps has no 'flux', or if the
            test DataLoader labels differ from fp_labels.
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

    test_flux = test_fps["flux"] if isinstance(test_fps, xr.Dataset) and "flux" in test_fps else None
    flux_in_loader = fp_labels is not None and "flux" in fp_labels
    if flux_in_loader:
        if test_flux is None:
            raise ValueError("The training labels include 'flux' (flux-weighted loss), but this test "
                             "region has no 'flux' variable.")
        # added after the NaN mask, as FootprintDataset.transform does for training
        test_fp_ds["flux"] = test_flux

    test_fp_ds = test_fp_ds.chunk({"time": 1})
    test_fp_ds = test_fp_ds.assign_coords(idx=("time", list(range(len(test_fp_ds.time)))))

    test_scaled_inputs, test_fp_ds = gates_datasets.trim_to_batch_size(
        test_scaled_inputs, test_fp_ds, batch_size)

    # num_workers=0 avoids forkserver deadlock — same as single-region test loader
    test_loader, test_labels = gates_datasets.make_dataloader(
        test_scaled_inputs, test_fp_ds, batch_size,
        randomize=False,
        dataloader_params={"num_workers": 0, "persistent_workers": False, "prefetch_factor": None},
        flatten=True,
    )
    if fp_labels is not None and list(test_labels) != list(fp_labels):
        raise ValueError(f"Test loader labels {list(test_labels)} differ from training loader labels "
                         f"{list(fp_labels)}.")

    if test_flux is not None and not flux_in_loader:
        # For metrics only: the loader has already stacked its variables, so adding
        # flux now doesn't change what it yields. Trim to the same timestamps (trimming is positional).
        test_flux = test_flux.isel(time=slice(0, test_fp_ds.sizes["time"]))
        if not np.array_equal(test_flux.time.values, test_fp_ds.time.values):
            raise ValueError("Test flux and footprint timestamps don't match.")
        test_fp_ds["flux"] = test_flux.chunk({"time": 1})
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
                # None if the region has no valid flux
                "flux_eval": computed_metrics_r.get("flux_eval_metrics"),
            }
            all_test_fp_datasets.append(fp_ds_r)

        # --- Aggregate metrics ---
        # flux metrics are per region only: drop flux so the aggregate never computes them
        # (this also lets regions with and without flux be concatenated)
        fp_ds_all = xr.concat(
            [ds.drop_vars("flux", errors="ignore") for ds in all_test_fp_datasets], dim="time")
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
            for flux_mode, metrics in computed_metrics["static_mf_eval_metrics"].items():
                logging_dict.update({
                    f"metrics_fluxes_static/aggregate/{flux_mode}/{k}": v
                    for k, v in metrics.items()})
            # logging_dict.update({
            #     f"flux/{flux_mode}/{metric_name}": metric_value
            #     for flux_mode, metrics in computed_metrics["static_mf_eval_metrics"].items()
            #     for metric_name, metric_value in metrics.items()})
            for region_name, rd in region_log.items():
                # log per-region metrics with region name in the key, e.g. "metrics_transformed/0-SAHARA/MSE"
                logging_dict.update({
                    f"metrics_transformed/{region_name}/{k}": v
                    for k, v in rd["trans_eval"].items()})
                logging_dict.update({
                    f"metrics_original/{region_name}/{k}": v
                    for k, v in rd["eval"].items()})
                for flux_mode, metrics in rd["static_mf_eval"].items():
                    logging_dict.update({
                        f"metrics_fluxes_static/{region_name}/{flux_mode}/{k}": v
                        for k, v in metrics.items()})
                # metrics against the region's loaded flux, e.g. "metrics_fluxes/1-BRAZIL/mae"
                if rd["flux_eval"] is not None:
                    logging_dict.update({
                        f"metrics_fluxes/{region_name}/{k}": v
                        for k, v in rd["flux_eval"].items()})
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
            flux_text = f", metrics_fluxes={rd['flux_eval']}" if rd["flux_eval"] is not None else ""
            write_to_file(
                f"  {region_name}: test_loss={rd['test_loss']:.4f}, "
                f"metrics_original={rd['eval']}{flux_text}", paths_ctx.updates_path)

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
    parameters["seed"] = seed
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
    region_configs = build_region_configs(parameters["regions"], shared_load_params,
                                          global_flux_args=parameters.get("flux"))
    # record each region's effective flux settings (global block + region override)
    parameters["resolved_region_flux"] = {cfg["name"]: cfg["flux"] for cfg in region_configs}
    if use_wandb:
        wandb.config.update({"resolved_region_flux": parameters["resolved_region_flux"]})
    input_variables = parameters["variables"]
    datapath_args = paths_ctx.resolve_datapath_args(parameters)
    # flux is loaded for test data only, unless the loss needs it for training
    train_flux = gates_training.loss_uses_flux(parameters)
    # `load_before_training` (formerly the top-level `load_into_memory`) — materialise
    # each region's inputs into RAM before dataloader setup. Defaults to True; the old
    # `load_into_memory` key is still honoured for back-compat.
    load_before_training = parameters.get("load_before_training", parameters.get("load_into_memory", True))
    parameters["load_before_training"] = load_before_training

    client, cluster = gates_training.make_cluster()
    write_to_file(f"loading data for {len(region_configs)} region(s)", paths_ctx.updates_path)

    all_train_inputs, all_train_fps, test_regions = load_multiregion_data(
        region_configs, input_variables, datapath_args, train_flux=train_flux,
        verbose=verbose, load_into_memory=load_before_training, use_wandb=use_wandb)
    gates_training.check_train_flux_for_loss(parameters, all_train_fps)

    num_train_samples = len(all_train_fps.time)
    num_test_samples_per_region = {r["name"]: r["inputs"].sizes["fp_time"] for r in test_regions}
    num_test_samples_total = sum(num_test_samples_per_region.values())

    write_to_file(
        f"Successfully loaded all region data. "
        f"Train samples: {num_train_samples}. "
        f"Test samples per region: {num_test_samples_per_region}. "
        f"Total test samples: {num_test_samples_total}",
        paths_ctx.updates_path)
    print(f"Train samples: {num_train_samples}, total test samples: {num_test_samples_total}")

    # which regions will get flux metrics (missing flux files only warn while loading)
    flux_summary = "Flux per region:\n" + "\n".join(
        summarise_region_flux(test_regions, parameters["resolved_region_flux"]))
    write_to_file(flux_summary, paths_ctx.updates_path)
    print(flux_summary)

    if use_wandb:
        wandb.summary.update({
            "num_training_samples": num_train_samples,
            "num_testing_samples": num_test_samples_total,
            **{f"num_testing_samples_{name}": n for name, n in num_test_samples_per_region.items()},
        })

    if cluster is not None:
        print("closing client before cluster!")
        client.close()
        cluster.close()

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

    # dataloader.seed overrides the run-wide seed for the shuffle, as in setup_GATES_dataloaders
    train_loader, fp_labels = gates_datasets.make_dataloader(
        train_scaled_inputs, train_scaled_fps, batch_size,
        randomize=True, random_seed=dataloader_info.get("seed", seed),
        dataloader_params=dataloader_params, flatten=True)

    scalers = {"inputs_scaler": input_dataset.scaler, "fp_scaler": fp_dataset.scaler,
               "input_names": list(train_scaled_inputs.variable_name.values)}

    num_features = train_scaled_inputs.sizes["variable_name"]
    parameters["num_features"] = num_features
    if use_wandb:
        wandb.summary.update({"num_features": num_features})

    # Build per-region test loaders using the already-fitted scalers
    add_nan_mask = dataloader_info.get("nans_to_zeros", True)
    test_region_entries = []
    for region in test_regions:
        if verbose:
            print(f"Building test loader for region: {region['name']}")
        test_loader_r, test_fp_ds_r = transform_test_region(
            region["inputs"], region["fp_xr"], scalers, add_nan_mask, test_batch_size,
            fp_labels=fp_labels)
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
    image_plots = random.sample(list(range(len(first_region["inputs"]))), k=4)
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
    grid, _ = get_grid(first_region["fp_xr"], parameters.get("grid_reference_fp"),
                       fix_transpose=parameters.get("review_fixes", {}).get("fix_grid_transpose", False))
    save_object(grid, "grid", paths_ctx.training_outputs_path, model_name,
                description="Grid object used during training", use_wandb=use_wandb)

    print("dynamic edges parameters:", parameters.get("dynamic_edges", None))
    if parameters.get("dynamic_edges", None) is not None:
        if isinstance(parameters["dynamic_edges"], dict):
            dynamic_edges_params = gates_training.setup_dynamic_edges(input_names=scalers["input_names"], **parameters["dynamic_edges"])
        elif parameters.get("dynamic_edges") is True:
            dynamic_edges_params = gates_training.setup_dynamic_edges(input_names=scalers["input_names"])
        else:
            dynamic_edges_params = {}
    else:
        dynamic_edges_params = {}

    training_ctx = TrainingContext(
        parameters, device, use_wandb, image_dates, image_plots, grid, fp_labels, scalers,
        all_train_inputs.variable_name.size, first_region["fp_xr"].lat.size, dynamic_edges_params)

    write_to_file("setting up model", paths_ctx.updates_path)
    model, model_ctx = gates_training.setup_GATES_model(parameters, training_ctx, paths_ctx)

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    losses = gates_training.initialise_losses()

    if use_wandb:
        wandb.define_metric("epoch")
        wandb.define_metric("MSE/*", step_metric="epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("test_loss/*", step_metric="epoch")
        wandb.define_metric("LossFn/*", step_metric="epoch")
        wandb.define_metric("metrics_*", step_metric="epoch")
        wandb.define_metric("fps_epoch_*", step_metric="epoch")

        # Flux metrics need explicit definitions due to deep nesting ({region}/{flux_mode}/{metric})
        region_names = parameters.get("region_names", [])
        if "metrics_fluxes_static" in losses:
            for flux_mode, flux_metrics in losses["metrics_fluxes_static"].items():
                for metric_name in flux_metrics:
                    wandb.define_metric(f"metrics_fluxes_static/aggregate/{flux_mode}/{metric_name}", step_metric="epoch")
                    for region_name in region_names:
                        wandb.define_metric(f"metrics_fluxes_static/{region_name}/{flux_mode}/{metric_name}", step_metric="epoch")
        # Real-flux metrics: per region only, no aggregate ({region}/{metric})
        for metric_name in losses["metrics_fluxes"]:
            for region_name in region_names:
                wandb.define_metric(f"metrics_fluxes/{region_name}/{metric_name}", step_metric="epoch")

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