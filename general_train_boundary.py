import sys
import copy
import os
import pickle
import random
import re
import time
import json
import argparse
from datetime import datetime

import einops
import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import BoundaryDataset
from model.data.load_data import *
from model.forecast import GraphSatelliteForecasterClassifier, GraphSatelliteForecasterConvClassifier
from model.loss_functions import *

#from general_train_nawid import write_to_file, load_file, set_reproducibility, log_object_as_artifact, EarlyStopping
from general_train_nawid import  load_file,  EarlyStopping

import sys


import matplotlib.pyplot as plt

import numpy as np
import torch
import xarray as xr

import random
import copy
import wandb


sys.path.insert(0, "/user/work/yl18410/new_graphnet")
sys.path.insert(1, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")
from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
#from model.data.dataloader_graphnet import *
#from model.data.load_data import *
from model.loss_functions import *



import time
from datetime import datetime

import argparse

import random
from pathlib import Path

import gates.training.training as gates_training
import gates
from gates.data.load_data import get_grid
from gates.training.training_dataclasses import PathContext, TrainingContext, BoundaryTrainingContext

from gates.training.training_helperfuns import load_parameter_file, save_object, write_to_file, save_training_plots, export_results_to_netcdf, save_wandb_artifact, set_reproducibility

# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------

def normalize_boundary_data(outputs, outputs_norm_vals=None):
    """
    Normalise an array or xarray DataArray using provided or computed mean and std.

    Args:
        outputs (np.ndarray or xr.DataArray): Array to normalise.
        outputs_norm_vals (tuple, optional): (mean, std) to reuse from training.
            If None, mean and std are computed from outputs.

    Returns:
        tuple:
            - np.ndarray or xr.DataArray: Normalised array, same type as input.
            - tuple: (mean, std) used for normalisation.
    """
    if outputs_norm_vals is None:
        if isinstance(outputs, xr.DataArray):
            outputs_mean = float(outputs.mean())
            outputs_std = float(outputs.std())
        else:
            outputs_mean = np.mean(outputs)
            outputs_std = np.std(outputs)
        outputs_norm_vals = (outputs_mean, outputs_std)

    outputs_mean, outputs_std = outputs_norm_vals
    normalized_outputs = (outputs - outputs_mean) / outputs_std

    return normalized_outputs, outputs_norm_vals


def denormalize(array, mean, std):
    """
    Reverse normalisation.

    Args:
        array (np.ndarray): Normalised array.
        mean (float or np.ndarray): Mean used during normalisation.
        std (float or np.ndarray): Std used during normalisation.

    Returns:
        np.ndarray: Denormalised array.
    """
    return (array * std) + mean


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def parse_years(year_input):
    """
    Parse a year or list of years into a flat list of integer years.

    Args:
        year_input (str, int, or list): Accepted formats:
            - int: 2014
            - str: '2014' or bracketed range '201[4-5]'
            - list of int or str: [2014, 2015] or ['2014', '2015']

    Returns:
        list of int: List of years.

    Raises:
        ValueError: If the format is not recognised.
    """
    # Handle list or tuple input by parsing each element and flattening
    # Must happen before str() conversion to avoid str(['2015']) = "['2015']"
    if isinstance(year_input, (list, tuple)):
        years = []
        for y in year_input:
            years.extend(parse_years(y))
        return years

    # Handle plain integer
    if isinstance(year_input, int):
        year_str = str(year_input)
        if re.fullmatch(r'20\d{2}', year_str):
            return [year_input]
        raise ValueError(f"Invalid year format: {year_input}")

    # Handle string
    year_str = str(year_input)

    # Bracketed range like '201[4-5]'
    match = re.fullmatch(r'201\[(\d)-(\d)\]', year_str)
    if match:
        start, end = map(int, match.groups())
        return [2010 + i for i in range(start, end + 1)]

    # Exact four-digit year like '2014'
    if re.fullmatch(r'20\d{2}', year_str):
        return [int(year_str)]

    raise ValueError(f"Invalid year format: {year_input}")


# ---------------------------------------------------------------
# CAMS boundary condition computation
# ---------------------------------------------------------------
def baseline_mol_correction_xr(desired_data, months, years, output_format, height_indices=None):
    """
    Compute CAMS-based boundary condition outputs and auxiliary features,
    keeping everything as xarray operations.

    Args:
        desired_data: Data object with fp_data_full containing particle locations and times.
        months (list of str): Months to process, e.g. ['01', '02'].
        years (list of int): Years to process.
        output_format (str): One of 'sum' or 'corrected'.
        height_indices (list of int, optional): Height levels to extract. Defaults to [4].

    Returns:
        tuple:
            - xr.DataArray: Baseline list, shape (time, 4), dim 'direction'.
            - xr.DataArray: Outputs, shape (time, 1) or (time,), dim 'time'.
            - xr.DataArray: Auxiliary CAMS values, shape (time, n_aux), dim 'aux'.
            - xr.DataArray: Correction values, shape (time,).
    """
    if height_indices is None:
        height_indices = [4]

    fp_full = desired_data.fp_data_full
    times = fp_full.particle_locations_n.time

    # Parse time coordinates into year and month arrays
    times_pd = pd.DatetimeIndex(times.values)
    time_years  = xr.DataArray(times_pd.year,  dims=["time"], coords={"time": times})
    time_months = xr.DataArray(times_pd.month, dims=["time"], coords={"time": times})

    df = pd.read_csv(
        '/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv'
    )

    n_times = len(times)
    n_aux = 4 * len(height_indices)

    # Initialise output DataArrays
    baseline_list = xr.DataArray(
        np.zeros((n_times, 4)),
        dims=["time", "direction"],
        coords={"time": times, "direction": ["north", "south", "east", "west"]}
    )
    north_list          = xr.DataArray(np.zeros(n_times), dims=["time"], coords={"time": times})
    south_list          = xr.DataArray(np.zeros(n_times), dims=["time"], coords={"time": times})
    east_list           = xr.DataArray(np.zeros(n_times), dims=["time"], coords={"time": times})
    west_list           = xr.DataArray(np.zeros(n_times), dims=["time"], coords={"time": times})
    total_list          = xr.DataArray(np.zeros(n_times), dims=["time"], coords={"time": times})
    collated_corrections = xr.DataArray(np.zeros(n_times), dims=["time"], coords={"time": times})

    aux_names = [f"aux_{i}" for i in range(n_aux)]
    auxiliary_cams = xr.DataArray(
        np.zeros((n_times, n_aux)),
        dims=["time", "aux"],
        coords={"time": times, "aux": aux_names}
    )

    for year in years:
        print(f"Processing year {year}")
        for month in months:
            # Boolean mask for this year/month combination
            mask = (time_years == int(year)) & (time_months == int(month))
            if not mask.any():
                continue

            cams_path = (
                f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc"
                if int(year) > 2017
                else f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc"
            )
            cams = xr.open_dataset(cams_path)

            # Baseline CSV lookup
            filtered_df = df[(df["Year"] == int(year)) & (df["Month"] == int(month))]
            baseline_vals = filtered_df.iloc[:, [3, 4, 5, 6]].values / 1000
            baseline_list.loc[{"time": mask}] = baseline_vals

            # Select particle locations for this month
            # north/south have dims (height, lon, time)
            # east/west have dims (height, lat, time)
            pl_n = fp_full.particle_locations_n.sel(time=mask)
            pl_s = fp_full.particle_locations_s.sel(time=mask)
            pl_e = fp_full.particle_locations_e.sel(time=mask)
            pl_w = fp_full.particle_locations_w.sel(time=mask)

            # CAMS vmr arrays have different grid sizes from particle locations
            # so interpolate CAMS onto the particle location grid before multiplying
            cams_vmr_n = cams.vmr_n.interp(lon=pl_n.lon, method="linear")  # (height, lon)
            cams_vmr_s = cams.vmr_s.interp(lon=pl_s.lon, method="linear")  # (height, lon)
            cams_vmr_e = cams.vmr_e.interp(lat=pl_e.lat, method="linear")  # (height, lat)
            cams_vmr_w = cams.vmr_w.interp(lat=pl_w.lat, method="linear")  # (height, lat)

            # Weighted sum over height and boundary dimension, result is (time,)
            north_mol = (cams_vmr_n * pl_n).sum(dim=["height", "lon"])
            south_mol = (cams_vmr_s * pl_s).sum(dim=["height", "lon"])
            east_mol  = (cams_vmr_e * pl_e).sum(dim=["height", "lat"])
            west_mol  = (cams_vmr_w * pl_w).sum(dim=["height", "lat"])

            north_list.loc[{"time": mask}] = north_mol.values
            south_list.loc[{"time": mask}] = south_mol.values
            east_list.loc[{"time": mask}]  = east_mol.values
            west_list.loc[{"time": mask}]  = west_mol.values

            # Auxiliary CAMS values at specific height levels and boundary midpoints
            mid_n_index = cams.vmr_n.shape[1] // 2
            mid_e_index = cams.vmr_e.shape[1] // 2

            all_vals = []
            for h in height_indices:
                all_vals.extend([
                    float(cams.vmr_n.values[h, mid_n_index]),
                    float(cams.vmr_s.values[h, mid_n_index]),
                    float(cams.vmr_e.values[h, mid_e_index]),
                    float(cams.vmr_w.values[h, mid_e_index]),
                ])
            # Same auxiliary value repeated for all timesteps in this month
            auxiliary_cams.loc[{"time": mask}] = np.array(all_vals)

            # Correction term — mean of southern boundary at level 1
            correction = float(cams.vmr_s[1].mean())
            total_mol = north_mol + south_mol + east_mol + west_mol
            total_list.loc[{"time": mask}] = (total_mol - correction).values
            collated_corrections.loc[{"time": mask}] = correction

            cams.close()

    # Stack direction outputs into a single DataArray (time, direction)
    outputs_stacked = xr.concat(
        [north_list, south_list, east_list, west_list],
        dim=pd.Index(["north", "south", "east", "west"], name="direction")
    ).transpose("time", "direction")

    if output_format == "sum":
        outputs = outputs_stacked.sum(dim="direction").expand_dims("direction", axis=-1)
    elif output_format == "corrected":
        outputs = total_list.expand_dims("direction", axis=-1)
    else:
        outputs = outputs_stacked

    return baseline_list, outputs, auxiliary_cams, collated_corrections


def baseline_mol_correction(desired_data, months, years, output_format, height_indices=None):
    """
    Compute CAMS-based boundary condition outputs and auxiliary features.

    Args:
        desired_data: Data object with fp_data_full containing particle locations and times.
        months (list of str): Months to process, e.g. ['01', '02'].
        years (list of int): Years to process.
        output_format (str): One of 'sum' or 'corrected'.
        height_indices (list of int, optional): Height levels to extract. Defaults to [4].

    Returns:
        tuple:
            - np.ndarray: Baseline list, shape (N, 4).
            - np.ndarray: Outputs, shape (N, 1).
            - np.ndarray: Auxiliary CAMS values, shape (N, 4 * len(height_indices)).
            - np.ndarray: Correction values, shape (N,).
    """
    if height_indices is None:
        height_indices = [4]

    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]

    df = pd.read_csv(
        '/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv'
    )

    baseline_list = np.zeros((total_data_points, 4))
    north_list = np.zeros(total_data_points)
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)
    total_list = np.zeros(total_data_points)
    auxiliary_cams = np.zeros((total_data_points, 4 * len(height_indices)))
    collated_corrections = np.zeros(total_data_points)

    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)
    specific_years = np.array([np.datetime64(date, 'Y').astype(int) + 1970 for date in datetime_array])
    specific_months = np.array([np.datetime64(date, 'M').astype(int) % 12 + 1 for date in datetime_array])

    for year in years:
        print(f"Processing year {year}")
        for month in months:
            cams_path = (
                f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc"
                if year > 2017
                else f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc"
            )
            cams = xr.open_dataset(cams_path)

            indices = np.where(
                (specific_years == year) & (specific_months == int(month))
            )[0]

            if len(indices) == 0:
                continue

            filtered_data = df[(df["Year"] == year) & (df["Month"] == int(month))]
            baseline_list[indices] = filtered_data.iloc[:, [3, 4, 5, 6]].values / 1000

            north_mol = np.sum(
                cams.vmr_n * desired_data.fp_data_full.particle_locations_n[:, :, indices], axis=(0, 1)
            )
            south_mol = np.sum(
                cams.vmr_s * desired_data.fp_data_full.particle_locations_s[:, :, indices], axis=(0, 1)
            )
            east_mol = np.sum(
                cams.vmr_e * desired_data.fp_data_full.particle_locations_e[:, :, indices], axis=(0, 1)
            )
            west_mol = np.sum(
                cams.vmr_w * desired_data.fp_data_full.particle_locations_w[:, :, indices], axis=(0, 1)
            )

            north_list[indices] = north_mol
            south_list[indices] = south_mol
            east_list[indices] = east_mol
            west_list[indices] = west_mol

            mid_n_index = cams.vmr_n.shape[1] // 2
            mid_e_index = cams.vmr_e.shape[1] // 2

            all_vals = []
            for h in height_indices:
                all_vals.extend([
                    cams.vmr_n.values[h, mid_n_index],
                    cams.vmr_s.values[h, mid_n_index],
                    cams.vmr_e.values[h, mid_e_index],
                    cams.vmr_w.values[h, mid_e_index],
                ])
            auxiliary_cams[indices, :] = np.array(all_vals)

            correction = np.mean(cams.vmr_s[1].values)
            total_list[indices] = (north_mol + south_mol + east_mol + west_mol).values - correction
            collated_corrections[indices] = correction

    outputs = np.stack((north_list, south_list, east_list, west_list), axis=1)

    if output_format == "sum":
        outputs = np.sum(outputs, axis=1, keepdims=True)
    elif output_format == "corrected":
        outputs = np.array(total_list).reshape(-1, 1)

    return baseline_list, outputs, auxiliary_cams, collated_corrections


# ---------------------------------------------------------------
# Data loading and normalisation
# ---------------------------------------------------------------

def load_and_normalise_boundary_data(data, months, years, output_format, height_indices,
                                     norm_vals=None):
    """
    Run baseline_mol_correction and normalise all outputs.

    Args:
        data: LoadSquareSatelliteData object.
        months (list of str): Months to process.
        years (list of int): Years to process.
        output_format (str): 'sum' or 'corrected'.
        height_indices (list of int): Height indices to extract.
        norm_vals (dict, optional): Dict with keys 'outputs', 'baselines', 'auxiliary',
            each a (mean, std) tuple. If None, normalisation values are computed from data.

    Returns:
        tuple:
            - np.ndarray: Normalised outputs.
            - np.ndarray: Normalised baseline list.
            - np.ndarray: Normalised auxiliary CAMS.
            - np.ndarray: Raw correction values.
            - dict: Normalisation values used, with keys 'outputs', 'baselines', 'auxiliary'.
    """
    baseline_list, outputs, auxiliary_cams, corrections = baseline_mol_correction_xr(
        data, months, years, output_format=output_format, height_indices=height_indices
    )
    '''
    baseline_list, outputs, auxiliary_cams, corrections = baseline_mol_correction(
        data, months, years, output_format=output_format, height_indices=height_indices
    )
    '''
    if norm_vals is None:
        outputs, outputs_norm = normalize_boundary_data(outputs)
        baseline_list, baselines_norm = normalize_boundary_data(baseline_list)
        auxiliary_cams, auxiliary_norm = normalize_boundary_data(auxiliary_cams)
    else:
        outputs, outputs_norm = normalize_boundary_data(outputs, norm_vals['outputs'])
        baseline_list, baselines_norm = normalize_boundary_data(baseline_list, norm_vals['baselines'])
        auxiliary_cams, auxiliary_norm = normalize_boundary_data(auxiliary_cams, norm_vals['auxiliary'])

    norm_vals_out = {
        'outputs': outputs_norm,
        'baselines': baselines_norm,
        'auxiliary': auxiliary_norm,
    }

    return outputs, baseline_list, auxiliary_cams, corrections, norm_vals_out




# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, criterion_test, device, epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    n_batches = 0

    for i, batch in enumerate(loader):
        ins, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(ins)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            display_loss = criterion_test(outputs, labels)
            running_loss += display_loss.item()

        n_batches += 1

        if i % 10 == 0:
            print(f"[{epoch}, {i:5d}] Loss: {running_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate_and_predict(model, loader, criterion_test, device):
    model.eval()
    test_error = 0.0
    preds_list = []
    n_batches = 0

    for batch in loader:
        ins, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(ins)

        test_error += criterion_test(outputs, labels).item()
        pred_np = outputs.cpu().numpy().reshape(outputs.shape[0], -1)
        preds_list.append(pred_np)
        n_batches += 1

    return test_error / max(n_batches, 1), np.vstack(preds_list)


def run_full_training(model, train_loader, test_loader, model_ctx, training_ctx, paths_ctx, losses):
    """
    Executes the full training loop with early stopping and checkpointing,
    using context dataclasses for configuration.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation data loader.
        model_ctx (ModelContext): Model configuration including optimizer, criterion,
            early stopping, and epoch settings.
        training_ctx (BoundaryTrainingContext): Training context including device,
            use_wandb flag, and other training parameters.
        paths_ctx (PathContext): Path context including model_path, model_name,
            and updates_path.
        losses (dict): Accumulator dict with 'train' and 'test' lists.

    Returns:
        None
    """
    updates_path = paths_ctx.model_path / f"{paths_ctx.model_name}_updates.txt"
    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx
        print(f"\n--- Start Epoch: {epoch} ---")
        avg_train_loss = train_one_epoch(
            model, train_loader, model_ctx.optimizer,
            model_ctx.criterion, model_ctx.criterion_test,
            model_ctx.device, epoch
        )
        avg_test_loss, _ = validate_and_predict(
            model, test_loader, model_ctx.criterion_test, model_ctx.device
        )

        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)

        if training_ctx.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss,
            }, step=epoch)

        model_ctx.early_stopping(avg_test_loss, model)

        if model_ctx.early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        write_to_file(
            f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}",
            updates_path
        )

        if epoch % model_ctx.epochs_save == 0:
            checkpoint_path = paths_ctx.model_path / f"{paths_ctx.model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_ctx.optimizer.state_dict(),
                'loss': losses,
                'learning_rate': model_ctx.lr,
            }, checkpoint_path)

            if training_ctx.use_wandb:
                checkpoint_artifact = wandb.Artifact(
                    name=f"{paths_ctx.model_name}-checkpoint",
                    type="model",
                    description="Model checkpoint saved during training"
                )
                checkpoint_artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(checkpoint_artifact)

    print("Finished Training.")


# ---------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------

def train_and_save_model(parameters, model_save_dir):
    """
    Top-level entry point for a full training run.

    Args:
        parameters (dict): Full training configuration. Expected top-level keys:
            'model_name', 'env', 'learning_rate', 'epochs', 'variables',
            'dataloader_parameters', 'model_parameters', 'loss_functions',
            'train_load_data', 'test_load_data', 'output_format', 'normalization',
            'use_baselines', 'num_classes', 'network_decoder', 'auxiliary', 'use_wandb'.
        path (str): Base directory for saving all outputs.

    Returns:
        None
    """
    use_wandb = parameters.get('use_wandb', True)
    #name_output_format = parameters['output_format']
    name_output_format = 'corrected'
    cfg = gates.config.get_config()
    verbose = parameters.get("verbose", True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    print(f"Initialising model run for model_name: {model_name}")

    # the paths are all stored in this dataclass (e.g. paths_ctx.model_path)
    paths_ctx = PathContext(
        model_save_dir=model_save_dir,
        model_name=model_name,
        model_path=model_path # model_save_dir / model_name
    )

    paths_ctx.make_dirs()

    seed = parameters.get("seed", 34)
    set_reproducibility(seed)
    if use_wandb:
        wandb_project = parameters.get("wandb", {}).get("project", None)
        wandb_entity = parameters.get("wandb", {}).get("entity", None)
        wandb_tags = parameters.get("wandb", {}).get("tags", [])
        if use_wandb and wandb_project is None or wandb_entity is None:
            print("Warning: 'use_wandb' is True but no 'wandb.project' or 'wandb.entity' specified in parameters. W&B will not be initialised.")
            use_wandb = False
            parameters["use_wandb"] = False

        if use_wandb:
            wandb.init(
                project=wandb_project, 
                config=parameters,
                tags=wandb_tags
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"), paths_ctx.updates_path)
    write_to_file("loading data", paths_ctx.updates_path)

    train_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params.update(parameters["test_load_data"])
    input_variables = parameters["variables"]

    # the data is extracted from the config file, unless it is superced from parameters. resolve_datapath_args returns the correct path in a dictionary passed to the data objects 
    datapath_args = paths_ctx.resolve_datapath_args(parameters)

    print("Loading met and fp data for model", model_name)
    write_to_file("Load training and testing met and fp data", paths_ctx.updates_path)

    client, cluster = gates_training.make_cluster()

    load_monthly = parameters.get("load_data_monthly", False)
    
    if not load_monthly:
        data, train_inputs = gates_training.load_GATES_data(train_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose)
        train_fp_data = data.fp_xr

    if load_monthly:
        data, train_inputs = gates_training.load_GATES_data_v2(train_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose, load_into_memory=parameters.get("load_into_memory", False))
        train_fp_data = data.fp_xr
    
    write_to_file(f"Successfully load training met and fp data with {len(train_fp_data.time)} time samples", paths_ctx.updates_path)
    print("Successfully load training met and fp data with", len(train_fp_data.time), "time samples")

    if not load_monthly:
        test_data, test_inputs = gates_training.load_GATES_data(test_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose)  # if load_into_memory is True, this will load the test data into memory immediately; if False, it will remain as dask arrays until needed
        test_fp_data = test_data.fp_xr
    if load_monthly:
        test_data, test_inputs = gates_training.load_GATES_data_v2(test_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose, load_into_memory=parameters.get("load_into_memory", False))  # if load_into_memory is True, this will load the test data into memory immediately; if False, it will remain as dask arrays until needed
        test_fp_data = test_data.fp_xr

    write_to_file(f"Successfully load test met and fp data with {len(test_fp_data.time)} time samples", paths_ctx.updates_path)
    print("Successfully load test met and fp data with", len(test_fp_data.time), "time samples")

    if use_wandb:
        # save the number of testing and training samples to wandb config for reference
        wandb.summary.update({
            "num_training_samples": len(train_fp_data.time),
            "num_testing_samples": len(test_fp_data.time),
        })

    ## add the number of features to the parameter file, and to wandb
    num_features = train_inputs.shape[-1]
    parameters["num_features"] = num_features
    if use_wandb:
        wandb.summary.update({"num_features": num_features})


    '''
    os.makedirs(f"{path}{model_name}", exist_ok=True)
    os.makedirs(f"{path}{model_name}/training_imgs", exist_ok=True)
    training_outputs_path = f"{path}{model_name}/training_outputs"
    os.makedirs(training_outputs_path, exist_ok=True)
    open(f"{path}{model_name}/{model_name}_updates.txt", "x").close()
    
    # Load environment paths from config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    #env = parameters['env']
    env_paths = config["data_paths"]
    base_data_path = env_paths["base_data_path"]
    fp_datadir = os.path.join(base_data_path, env_paths["fp_datadir"].lstrip("/"))
    met_datadir = os.path.join(base_data_path, env_paths["met_datadir"].lstrip("/"))
    topog_datadir = os.path.join(base_data_path, env_paths["topog_datadir"].lstrip("/"))
    landcover_datadir = os.path.join(base_data_path, env_paths["landcover_datadir"].lstrip("/"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    write_to_file(
        f"using device {device}, starting at {datetime.now().strftime('%d/%m/%y %H:%M:%S')}",
        log_file
    )

    # Build data loading args
    train_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])

    shared_data_args = dict(
        load_everything=True,
        base_data_path=base_data_path,
        fp_datadir=fp_datadir,
        met_datadir=met_datadir,
        topog_args={"topog_path": topog_datadir, "landcover_path": landcover_datadir}
    )

    # Load data
    write_to_file("Load training met and fp data", log_file)
    data = LoadSquareSatelliteData(**train_load_data, **shared_data_args)
    write_to_file("Successfully loaded training data. Now loading test data.", log_file)
    test_data = LoadSquareSatelliteData(**test_load_data, **shared_data_args)
    write_to_file("Successfully loaded test data.", log_file)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"), paths_ctx.updates_path)
    write_to_file("loading data", paths_ctx.updates_path)

    train_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params.update(parameters["test_load_data"])

    input_variables = parameters["variables"]

    # the data is extracted from the config file, unless it is superced from parameters. resolve_datapath_args returns the correct path in a dictionary passed to the data objects 
    datapath_args = paths_ctx.resolve_datapath_args(parameters)

    print("Loading met and fp data for model", model_name)
    write_to_file("Load training and testing met and fp data", paths_ctx.updates_path)

    client, cluster = gates_training.make_cluster()

    load_monthly = parameters.get("load_data_monthly", False)
    
    if not load_monthly:
        data, train_inputs = gates_training.load_GATES_data(train_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose)
        train_fp_data = data.fp_xr

    if load_monthly:
        data, train_inputs = gates_training.load_GATES_data_v2(train_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose, load_into_memory=parameters.get("load_into_memory", False))
        train_fp_data = data.fp_xr

    write_to_file(f"Successfully load training met and fp data with {len(train_fp_data.time)} time samples", paths_ctx.updates_path)
    print("Successfully load training met and fp data with", len(train_fp_data.time), "time samples")

    if not load_monthly:
        test_data, test_inputs = gates_training.load_GATES_data(test_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose)  # if load_into_memory is True, this will load the test data into memory immediately; if False, it will remain as dask arrays until needed
        test_fp_data = test_data.fp_xr
    if load_monthly:
        test_data, test_inputs = gates_training.load_GATES_data_v2(test_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose, load_into_memory=parameters.get("load_into_memory", False))  # if load_into_memory is True, this will load the test data into memory immediately; if False, it will remain as dask arrays until needed
        test_fp_data = test_data.fp_xr

    if use_wandb:
        # save the number of testing and training samples to wandb config for reference
        wandb.summary.update({
            "num_training_samples": len(train_fp_data.time),
            "num_testing_samples": len(test_fp_data.time),
        })

    ## add the number of features to the parameter file, and to wandb
    num_features = train_inputs.shape[-1]
    parameters["num_features"] = num_features
    if use_wandb:
        wandb.summary.update({"num_features": num_features})
    write_to_file("scaling data and setting up dataloaders", paths_ctx.updates_path)
    if cluster is not None:
        cluster.close()
        client.close()

    # Compute boundary condition outputs and normalise
    # Note: 'auxiliary' key used here — check parameter file uses this spelling
    height_indices = [4, 5, 6, 7] if parameters.get('auxiliary') == 'multiple' else [4]
    all_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    train_year = parse_years(train_load_data_params['years'])
    test_year = parse_years(test_load_data_params['years'])
    
    outputs, baseline_list, auxiliary_cams, _, norm_vals = load_and_normalise_boundary_data(
        data, all_months, train_year, name_output_format, height_indices
    )
    test_outputs, test_baseline_list, test_auxiliary_cams, _, _ = load_and_normalise_boundary_data(
        test_data, all_months, test_year, name_output_format, height_indices,
        norm_vals=norm_vals
    )

    # Save auxiliary normalisation values into checkpoint-compatible format
    outputs_mean_values, outputs_std_values = norm_vals['outputs']
    baseline_mean_values, baseline_std_values = norm_vals['baselines']
    auxiliary_mean_values, auxiliary_std_values = norm_vals['auxiliary']


    use_baselines = True
    aux_dim = auxiliary_cams.sizes["aux"] if use_baselines and auxiliary_cams is not None else 0

    # Build grid and datasets
    '''
    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    use_baselines = parameters['use_baselines']
    aux_dim = auxiliary_cams.shape[1] if use_baselines else 0
    '''
    
    '''
    train_loader, test_loader, scalers = gates_training.setup_boundary_dataloaders(
        parameters,
        train_inputs=train_inputs,
        train_outputs=outputs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        train_times=train_times,
        test_times=test_times,
        train_auxiliary_cams=auxiliary_cams if use_baselines else None,
        test_auxiliary_cams=test_auxiliary_cams if use_baselines else None,
    output_names=["sum"]  # adjust to match your output_format
)
    '''
    print("Computing inputs into memory before dataloader setup...")
    train_inputs = train_inputs.compute()
    test_inputs = test_inputs.compute()
    train_loader, test_loader,boundary_labels, scalers = gates_training.setup_boundary_dataloaders(parameters, train_inputs, outputs, test_inputs, test_outputs,
                                auxiliary_cams, test_auxiliary_cams)

    
    save_object(scalers, "scalers",
    paths_ctx.training_outputs_path, model_name, description=f"Input and output scaler objects used in model {model_name}", use_wandb=use_wandb)

    # images will get plotted and saved for a random selection of 4 dates from the test set - these indeces are saved to parameters for reference 
    image_plots = random.sample(list(range(len(test_inputs))), k=4)
    image_dates = np.datetime_as_string(test_fp_data.time.values[sorted(image_plots)])

    parameters["plotted_dates"] = image_dates.tolist()

    save_object(parameters, "training_settings", paths_ctx.training_outputs_path, model_name,  file_type="json", description=f"Training settings and hyperparameters for model {model_name}", use_wandb=use_wandb)

    # create the grid object to make the mesh with and save
    grid, _ = get_grid(train_fp_data, parameters.get("grid_reference_fp"))
    save_object(grid, "grid", paths_ctx.training_outputs_path, model_name,
                           description="Grid object used during training", use_wandb=use_wandb)
    
    

    training_ctx = BoundaryTrainingContext(parameters, device, use_wandb, image_dates, image_plots, grid, boundary_labels, scalers, train_inputs.variable_name.size, len(train_fp_data.lat.values),aux_dim) # get size from train params


    print("Successfully set up dataloaders!!! using the contexts!!!")

    write_to_file("setting up model", paths_ctx.updates_path)

    
    model, model_ctx = gates_training.setup_boundary_model(parameters, training_ctx, paths_ctx)


    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
    
    losses = gates_training.initialise_boundary_losses()

    if torch.cuda.is_available():
        model.cuda()
    run_full_training(
    model,
    train_loader,
    test_loader,
    model_ctx=model_ctx,
    training_ctx=training_ctx,
    paths_ctx=paths_ctx,
    losses=losses
)

    if use_wandb:
        wandb.finish()

    '''
    # Log artifacts
    log_object_as_artifact(grid, "grid", training_outputs_path, model_name,
                           description="Grid used during training", use_wandb=use_wandb)
    log_object_as_artifact(train_dataset.transform_parameters, "transform_parameters",
                           training_outputs_path, model_name,
                           description="Transform parameters used in training", use_wandb=use_wandb)
    log_object_as_artifact(parameters, "training_settings", training_outputs_path, model_name,
                           file_type="json", description="Training settings", use_wandb=use_wandb)

    # Save auxiliary normalisation values so inference can reuse them without reloading train data
    auxiliary_norm_vals = {
        "auxiliary_mean": auxiliary_mean_values,
        "auxiliary_std": auxiliary_std_values
    }
    log_object_as_artifact(auxiliary_norm_vals, "auxiliary_norm_vals", training_outputs_path,
                           model_name, file_type="json",
                           description="Auxiliary normalisation values", use_wandb=use_wandb)

    # Build model
    feature_dim = np.shape(train_inputs)[-1]
    num_lat, num_lon = len(data.met.lat.values), len(data.met.lon.values)

    model = build_model(parameters, grid, feature_dim, aux_dim, num_lat, num_lon)
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    optimizer = optim.AdamW(model.parameters(), lr=parameters["learning_rate"])

    normalization_vals = {"outputs_mean": outputs_mean_values, "outputs_std": outputs_std_values}
    baseline_normalization_vals = {"baselines_mean": baseline_mean_values, "baselines_std": baseline_std_values}
    losses = {"train": [], "test": []}

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
    '''



# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":  
    os.environ["WANDB_API_KEY"] = "11d787a211e05ca01c50131c5724e375cd5d3364"  # <<-- REPLACE THIS
    wandb.login()
    parser = argparse.ArgumentParser(description="Load parameters. Example usage: python train_GATES_model.py parameters.json")
    parser.add_argument("file_name", help="Parameter file name")
    parser.add_argument("--file_path", help="Parameter file path. By default, the path in config.yml will be used.", default=None)

    args = parser.parse_args()
    file_name = args.file_name
    file_path = args.file_path

    cfg = gates.config.get_config()

    if file_path is None:
        file_path = cfg.parameter_files_dir
    
    parameter_path = Path(file_path) / file_name
    
    #### 1 Set up
    parameters = load_parameter_file(parameter_path)
    
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    if parameters.get("use_wandb", False):
        wandb.login()
    
    if parameters.get("model_save_dir", None) is None: 
        model_saving_dir=cfg.save_models_dir
    else:
        model_saving_dir = Path(parameters["model_save_dir"])

    #TODO write a function that checks minimum parameters exist
    

    # Train the model with the loaded parameters
    train_and_save_model(parameters, model_save_dir=model_saving_dir)