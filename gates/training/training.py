from pyexpat import model

from model.forecast import GraphSatelliteForecaster, GraphSatelliteForecasterConvClassifier,GraphSatelliteForecasterClassifier


import torch.optim as optim
import time
from datetime import datetime
import json
import argparse
from pathlib import Path

import random
import yaml

import sys

import matplotlib.pyplot as plt

import numpy as np
import torch
import os
import pickle
import random
import xarray as xr
import pandas as pd

from pathlib import Path

import wandb


from gates import LoadSquareSatelliteData, get_square_satellite_inputs
import gates.data.datasets as gates_datasets
from gates.data.datasets import get_square_satellite_inputs_v2
import gates.evaluation.metrics as gates_metrics
import gates.evaluation.loss_functions as gates_losses

from .training_helperfuns import save_wandb_artifact, EarlyStopping

from .training_dataclasses import ModelContext
import dask.array as da

def load_GATES_data(data_parameters, input_variables, datapath_args = {}, verbose=True):
    """
    Loads training and test datasets for the GATES model using the LoadSquareSatelliteData class.

    Args:
        data_parameters (dict): A dictionary containing parameters for loading the data, expected to have 'train_load_data' and 'test_load_data' keys.
        datapath_args (dict): A dictionary of additional arguments required for data loading, such as file paths.
        verbose (bool): If True, prints verbose output during data loading.

    Returns:
        tuple: A tuple containing the training dataset and test dataset objects.
    """
    ## if the met args dict is in both data_parameters and datapath_args, merge
    if "met_args" in data_parameters and "met_args" in datapath_args:
        merged_met_args = {**data_parameters["met_args"], **datapath_args["met_args"]}
        data_parameters["met_args"] = merged_met_args
        datapath_args.pop("met_args")

    parallel_loading = data_parameters.get("parallel_loading", False)

    data = LoadSquareSatelliteData(**data_parameters, **datapath_args, verbose=verbose, parallel_loading=parallel_loading)

    inputs, data = get_square_satellite_inputs_v2(data, **input_variables, verbose=verbose)

    return data, inputs

def _normalise_month(m):
    """Convert int or str month to zero-padded two-character string, e.g. 1 -> '01'."""
    return f"{int(m):02d}"


def _resolve_years_months(data_parameters):
    """
    Extract year/years and month/months from data_parameters.

    Accepts:
        year  : int | str           — single year
        years : list[int | str]     — multiple years
        month : int | str | None    — single month, or None meaning all 12
        months: list[int | str]     — explicit list of months

    Returns (years_list, months_list) with months as zero-padded strings.
    """
    if "years" in data_parameters:
        years = data_parameters["years"]
        years = list(years) if isinstance(years, (list, tuple)) else [years]
    elif "year" in data_parameters:
        year = data_parameters["year"]
        years = list(year) if isinstance(year, (list, tuple)) else [year]
    else:
        raise ValueError("data_parameters must contain 'year' or 'years'")

    if "months" in data_parameters:
        months = [_normalise_month(m) for m in data_parameters["months"]]
    elif data_parameters.get("month") is not None:
        m = data_parameters["month"]
        months = [_normalise_month(mm) for mm in m] if isinstance(m, (list, tuple)) else [_normalise_month(m)]
    else:
        months = [f"{m:02d}" for m in range(1, 13)]

    return years, months


def load_GATES_data_v2(data_parameters, input_variables, datapath_args={}, verbose=True, load_into_memory=True):
    """
    Loads footprints and inputs for each year-month pair specified in data_parameters,
    returning them as concatenated xarrays rather than a LoadSquareSatelliteData object.

    data_parameters may contain:
        year  : int | str           — single year
        years : list[int | str]     — multiple years (alternative to year)
        month : int | str | None    — single month; omit or None for all 12
        months: list[int | str]     — explicit list of months (alternative to month)
        load_into_memory : bool     — if True, materialise each month into memory before
                                      concatenating (avoids large dask graphs at the cost
                                      of sequential I/O); default False

    All other keys are forwarded to LoadSquareSatelliteData.

    Returns:
        fp_xr  : xr.Dataset   — concatenated footprints (time, lat, lon)
        inputs : xr.DataArray — concatenated met inputs  (time, lat, lon, variable_name)
    """
    if "met_args" in data_parameters and "met_args" in datapath_args:
        merged_met_args = {**data_parameters["met_args"], **datapath_args["met_args"]}
        data_parameters["met_args"] = merged_met_args
        datapath_args.pop("met_args")

    #load_into_memory = data_parameters.get("load_into_memory", False)
    years, months = _resolve_years_months(data_parameters)

    base_params = {
        k: v for k, v in data_parameters.items()
        if k not in ("year", "years", "month", "months", "load_into_memory")
    }

    all_inputs = []
    all_fp_xr = []
    loading_times = {}

    for year in years:
        for month in months:
            month_start = time.perf_counter()
            month_key = f"{year}-{month}"
            if verbose:
                print(f"Loading year={year}, month={month}")
            month_params = {**base_params, "year": year, "month": month}
            try:
                data = LoadSquareSatelliteData(**month_params, **datapath_args, verbose=verbose)
            except Exception as e:
                print(f"Error loading data for {year}-{month}: {e}")
                elapsed_mins = (time.perf_counter() - month_start) / 60
                loading_times[month_key] = f"{elapsed_mins:.2f}mins"
                print(f"{month_key} : {loading_times[month_key]}")
                continue
            inputs, data = get_square_satellite_inputs_v2(data, **input_variables, verbose=verbose)
            if load_into_memory:
                print(f"Loading data into memory for {year}-{month} before concatenation...")
                inputs = inputs.load()
                data.fp_xr = data.fp_xr.load()
            all_inputs.append(inputs)
            all_fp_xr.append(data.fp_xr)

            elapsed_mins = (time.perf_counter() - month_start) / 60
            loading_times[month_key] = f"{elapsed_mins:.2f}mins"
            print(f"{month_key} : {loading_times[month_key]}")

    print("")
    print("")
    print("----- Loading times for each month -----")
    print("\n".join(f"{k} : {v}" for k, v in loading_times.items()))
    print("")


    fp_xr = xr.concat(all_fp_xr, dim="time").sortby("time")
    inputs = xr.concat(all_inputs, dim="fp_time").sortby("fp_time")
    return data, inputs
    #return fp_xr, inputs

def load_GATES_data_v2_boundary(data_parameters, input_variables, datapath_args={}, verbose=True, load_into_memory=True):
    """
    Loads footprints and inputs for each year-month pair specified in data_parameters,
    returning them as concatenated xarrays rather than a LoadSquareSatelliteData object.

    data_parameters may contain:
        year  : int | str           — single year
        years : list[int | str]     — multiple years (alternative to year)
        month : int | str | None    — single month; omit or None for all 12
        months: list[int | str]     — explicit list of months (alternative to month)
        load_into_memory : bool     — if True, materialise each month into memory before
                                      concatenating (avoids large dask graphs at the cost
                                      of sequential I/O); default False

    All other keys are forwarded to LoadSquareSatelliteData.

    Returns:
        data  : LoadSquareSatelliteData-like object with concatenated fp_data_full and fp_xr
        inputs : xr.DataArray — concatenated met inputs (fp_time, lat, lon, variable_name)
    """
    if "met_args" in data_parameters and "met_args" in datapath_args:
        merged_met_args = {**data_parameters["met_args"], **datapath_args["met_args"]}
        data_parameters["met_args"] = merged_met_args
        datapath_args.pop("met_args")

    years, months = _resolve_years_months(data_parameters)

    base_params = {
        k: v for k, v in data_parameters.items()
        if k not in ("year", "years", "month", "months", "load_into_memory")
    }

    all_inputs = []
    all_fp_xr = []
    all_fp_data_full = []
    loading_times = {}
    last_data = None

    for year in years:
        for month in months:
            month_start = time.perf_counter()
            month_key = f"{year}-{month}"
            if verbose:
                print(f"Loading year={year}, month={month}")
            month_params = {**base_params, "year": year, "month": month}
            try:
                data = LoadSquareSatelliteData(**month_params, **datapath_args, verbose=verbose)
            except Exception as e:
                print(f"Error loading data for {year}-{month}: {e}")
                elapsed_mins = (time.perf_counter() - month_start) / 60
                loading_times[month_key] = f"{elapsed_mins:.2f}mins"
                print(f"{month_key} : {loading_times[month_key]}")
                continue

            inputs, data = get_square_satellite_inputs_v2(data, **input_variables, verbose=verbose)

            if load_into_memory:
                print(f"Loading data into memory for {year}-{month} before concatenation...")
                inputs = inputs.load()
                data.fp_xr = data.fp_xr.load()
                # Also load particle locations into memory if they exist
                if hasattr(data, 'fp_data_full') and data.fp_data_full is not None:
                    for var in ["particle_locations_n", "particle_locations_s",
                                "particle_locations_e", "particle_locations_w"]:
                        if var in data.fp_data_full:
                            data.fp_data_full[var].load()

            all_inputs.append(inputs)
            all_fp_xr.append(data.fp_xr)

            # Collect fp_data_full for particle location concatenation
            if hasattr(data, 'fp_data_full') and data.fp_data_full is not None:
                all_fp_data_full.append(data.fp_data_full)

            last_data = data

            elapsed_mins = (time.perf_counter() - month_start) / 60
            loading_times[month_key] = f"{elapsed_mins:.2f}mins"
            print(f"{month_key} : {loading_times[month_key]}")

    print("")
    print("")
    print("----- Loading times for each month -----")
    print("\n".join(f"{k} : {v}" for k, v in loading_times.items()))
    print("")

    fp_xr = xr.concat(all_fp_xr, dim="time").sortby("time")
    inputs = xr.concat(all_inputs, dim="fp_time").sortby("fp_time")

    # Concatenate fp_data_full across all months so particle locations
    # cover the full time range, not just the last month
    if len(all_fp_data_full) > 1:
        print("Concatenating fp_data_full across all months...")
        fp_data_full_concat = xr.concat(all_fp_data_full, dim="time").sortby("time")
        last_data.fp_data_full = fp_data_full_concat
        print(f"  Concatenated fp_data_full: {dict(fp_data_full_concat.sizes)}")
    elif len(all_fp_data_full) == 1:
        last_data.fp_data_full = all_fp_data_full[0]

    last_data.fp_xr = fp_xr

    return last_data, inputs

def _get_scaler(scaler_name, scaler_module=None):
    """
    Dynamically retrieves a scaler class from a specified module based on its name.

    Args:
        scaler_name (str): The name of the scaler class to retrieve (e.g., "StandardScaler").
        scaler_module (module): The module from which to retrieve the scaler class. If None, defaults to gates_datasets 
    """
    if scaler_module is None:
        scaler_module = gates_datasets

    if hasattr(scaler_module, scaler_name):
        return getattr(scaler_module, scaler_name)
    else:
        raise ValueError(f"Scaler '{scaler_name}' not found in module '{scaler_module.__name__}'")
    

def setup_input_dataset(parameters, train_inputs):
    train_inputs = train_inputs.astype('float32')

    input_scaler_params = parameters.get("input_scaler", {})
    if input_scaler_params:
        if "scaler" in input_scaler_params:
            inputs_scaler = _get_scaler(input_scaler_params["scaler"], gates_datasets)
            # remove scaler from input_scaler_params
            input_scaler_params.pop("scaler")
        else:
            inputs_scaler = None

    input_dataset = gates_datasets.InputsDataset(train_inputs, inputs_scaler, **input_scaler_params)
    input_dataset.fit()

    return input_dataset

def setup_fp_dataset(parameters, train_fps):
    fp_scaler_params = parameters.get("fp_scaler", {})
    if fp_scaler_params:
        if "scaler" in fp_scaler_params:
            fp_scaler = _get_scaler(fp_scaler_params["scaler"], gates_datasets)
            # remove scaler from fp_scaler_params
            fp_scaler_params.pop("scaler")
        else:
            fp_scaler = None

    ## TODO add add_nan_mask option to parameter file, maybe with an alias nan_to_zero

    fp_dataset = gates_datasets.FootprintDataset(train_fps, scaler=fp_scaler, **fp_scaler_params, add_nan_mask=parameters["dataloader"].get("nans_to_zeros", True))
    fp_dataset.fit()

    return fp_dataset

def setup_GATES_dataloaders(parameters, train_inputs, train_fps, test_inputs, test_fps):


    input_dataset = setup_input_dataset(parameters, train_inputs)

    train_scaled_inputs = input_dataset.transform(train_inputs)
    test_scaled_inputs = input_dataset.transform(test_inputs)

    fp_dataset = setup_fp_dataset(parameters, train_fps)
    train_scaled_fp = fp_dataset.transform(train_fps)
    test_scaled_fp = fp_dataset.transform(test_fps)

    dataloader_info = parameters.get("dataloader", {})
    batch_size = dataloader_info.get("batch_size", 5)
    test_batch_size = dataloader_info.get("test_batch_size", 5)
    dataloader_params = dataloader_info.get("dataloader_params", {})
    if "prefetch_factor" in dataloader_params and dataloader_params["prefetch_factor"] == 0:
        dataloader_params["prefetch_factor"] = None 

    train_scaled_inputs, train_scaled_fp = gates_datasets.trim_to_batch_size(train_scaled_inputs, train_scaled_fp, batch_size)
    test_scaled_inputs, test_scaled_fp = gates_datasets.trim_to_batch_size(test_scaled_inputs, test_scaled_fp, test_batch_size)

    train_loader, fp_labels = gates_datasets.make_dataloader(train_scaled_inputs, train_scaled_fp, batch_size, randomize=True, dataloader_params=dataloader_params, flatten=True)   

    #print("WAAAAAAAAAAAARNING")
    #print("Loading inputs and footprints for test set into memory!!!")
    #test_scaled_inputs.load()
    #test_scaled_fp.load()

    # num_workers=0 avoids a deadlock: the train_loader's persistent forkserver workers
    # are still alive when validation starts, and xarray's internal threading locks
    # cannot safely cross the forkserver process boundary on the first batch fetch.
    
    test_dataloader_params = {"num_workers": 0, "persistent_workers": False, "prefetch_factor": None}
    print("SPECIAL TEST PARAMS", test_dataloader_params)
    # test_dataloader_params = {**dataloader_params, "persistent_workers": False}
    test_loader, fp_labels_test = gates_datasets.make_dataloader(test_scaled_inputs, test_scaled_fp, test_batch_size, randomize=False, dataloader_params=test_dataloader_params, flatten=True)

    if fp_labels != fp_labels_test:
        raise ValueError("The labels for the training and test datasets do not match - something went wrong. Please check the data loading and scaling steps to ensure consistency between train and test sets.")  

    scalers = {"inputs_scaler": input_dataset.scaler, "fp_scaler": fp_dataset.scaler}

    return train_loader, test_loader, fp_labels, test_scaled_fp, scalers

def concat_auxiliary_to_inputs(scaled_inputs, auxiliary_cams):
    """
    Concatenate auxiliary CAMS features onto scaled inputs along the variable_name dimension,
    broadcasting the auxiliary values across all lat/lon grid cells.
    Keeps everything lazy — no compute() is called, data is only loaded when batches are fetched.

    Args:
        scaled_inputs (xr.DataArray): Shape (fp_time, lat, lon, variable_name).
        auxiliary_cams (xr.DataArray): Shape (time, aux) with aux coordinate names.

    Returns:
        xr.DataArray: Concatenated inputs of shape (fp_time, lat, lon, variable_name)
            with aux features appended along variable_name. Remains lazy/dask-backed.
    """
    
    print('USING THE CONTATENTATION TO AUXILLARY')
    aux_names = list(auxiliary_cams.aux.values)
    n_aux = len(aux_names)
    n_time = scaled_inputs.sizes["fp_time"]
    lat_size = scaled_inputs.sizes["lat"]
    lon_size = scaled_inputs.sizes["lon"]

    # Rename 'time' to 'fp_time' to match scaled_inputs
    aux_fp_time = auxiliary_cams.rename({"time": "fp_time"})

    # Get the underlying data as dask array — stays lazy
    if hasattr(aux_fp_time.data, 'compute'):
        aux_data = aux_fp_time.data  # already dask
    else:
        # Convert numpy to dask so everything stays lazy
        aux_data = da.from_array(aux_fp_time.values, chunks=(n_time, n_aux))

    # Broadcast (fp_time, n_aux) -> (fp_time, lat, lon, n_aux) lazily using dask
    # reshape to (fp_time, 1, 1, n_aux) then broadcast
    aux_data_4d = aux_data[:, None, None, :]  # (fp_time, 1, 1, n_aux)
    aux_data_broadcast = da.broadcast_to(
        aux_data_4d,
        (n_time, lat_size, lon_size, n_aux)
    )

    # Build MultiIndex compatible with the variable_name MultiIndex on scaled_inputs
    aux_multiindex = pd.MultiIndex.from_arrays(
        [aux_names,
         [0] * n_aux,
         [0] * n_aux],
        names=["variable", "levels", "time_delta"]
    )

    # Build the auxiliary DataArray with dask backing — no compute triggered
    aux_expanded = xr.DataArray(
        aux_data_broadcast,
        dims=["fp_time", "lat", "lon", "variable_name"],
        coords={
            "fp_time": scaled_inputs.fp_time,
            "lat":     scaled_inputs.lat,
            "lon":     scaled_inputs.lon,
            "variable_name": aux_multiindex,
        }
    )

    # Concatenate along variable_name — stays lazy since both arrays are dask-backed
    return xr.concat([scaled_inputs, aux_expanded], dim="variable_name")

def setup_boundary_dataloaders(parameters, train_inputs, train_outputs, test_inputs, test_outputs,
                                train_auxiliary_cams=None, test_auxiliary_cams=None):

    # Nawid - removing just for debugging purposes, since the input transform step does usually take a while   
    # Scale inputs using training statistics
    input_dataset = setup_input_dataset(parameters, train_inputs)
    train_scaled_inputs = input_dataset.transform(train_inputs)
    test_scaled_inputs = input_dataset.transform(test_inputs)
    

    # Append pre-normalised auxiliary CAMS features lazily —
    # no compute() called here, data loads batch by batch via preload_batch=True
    use_baselines = parameters.get("use_baselines", True)

    if use_baselines and train_auxiliary_cams is not None and test_auxiliary_cams is not None:
        print('USING BASELINES AND CONCENATING ')
        print("Concatenating inputs and auxiliary cams (lazy)")
        train_scaled_inputs = concat_auxiliary_to_inputs(train_scaled_inputs, train_auxiliary_cams)
        test_scaled_inputs = concat_auxiliary_to_inputs(test_scaled_inputs, test_auxiliary_cams)
    print('IS THE CONTENATION DONE')
    # Outputs are small so compute them into memory
    train_outputs = train_outputs.compute() if hasattr(train_outputs, 'compute') else train_outputs
    test_outputs = test_outputs.compute() if hasattr(test_outputs, 'compute') else test_outputs

    dataloader_info = parameters.get("dataloader", {})
    batch_size = dataloader_info.get("batch_size", 5)
    test_batch_size = dataloader_info.get("test_batch_size", 5)
    dataloader_params = dataloader_info.get("dataloader_params", {})

    if "prefetch_factor" in dataloader_params and dataloader_params["prefetch_factor"] == 0:
        dataloader_params["prefetch_factor"] = None

    # Trim to batch size
    train_scaled_inputs, train_outputs = gates_datasets.trim_to_batch_size(
        train_scaled_inputs, train_outputs, batch_size
    )
    test_scaled_inputs, test_outputs = gates_datasets.trim_to_batch_size(
        test_scaled_inputs, test_outputs, test_batch_size
    )

    train_loader, boundary_labels_train = gates_datasets.make_boundary_dataloader(
        train_scaled_inputs, train_outputs,
        batch_size=batch_size,
        randomize=True,
        dataloader_params=dataloader_params,
        flatten=True
    )

    '''
    test_dataloader_params = {
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None
    }
    print("SPECIAL TEST PARAMS", test_dataloader_params)
    '''

    test_loader, boundary_labels_test = gates_datasets.make_boundary_dataloader(
        test_scaled_inputs, test_outputs,
        batch_size=test_batch_size,
        randomize=False,
        dataloader_params=dataloader_params,
        flatten=True
    )

    if boundary_labels_train != boundary_labels_test:
        raise ValueError(
            "The labels for the training and test boundary datasets do not match. "
            "Check that train and test outputs have the same structure and number of columns."
        )

    scalers = {"inputs_scaler": input_dataset.scaler}

    return train_loader, test_loader, boundary_labels_train, scalers

def initialise_losses():
    """
    Return a dictionary to store losses and metrics during training and evaluation. 
    """
    metrics_dict = {"nmae": [], "mse": [], "bias": [], "mae": [], "iou": []}
    flux_metrics_dict = {"corrcoef": [], "mae": [], "mean_bias": [], "r2_score": []}
    losses = {
        "train": [],
        "test": [],
        "test_criterion": {"train": [], "test": []},
        "metrics_transformed": metrics_dict.copy(),
        "metrics_original": metrics_dict.copy(),
        "metrics_fluxes_static": {
            "uniform": flux_metrics_dict.copy(),
            "checkerboard": flux_metrics_dict.copy(),
            "checkerboard_10": flux_metrics_dict.copy(),
        }
    }
    return losses

def initialise_boundary_losses():
    """
    Return a dictionary to store losses and metrics during training and evaluation. 
    """
    losses = {
        "train": [],
        "test": [],

    }
    return losses

def calculate_losses(losses, test_outputs_xr):
    """
    Calculate evaluation metrics for the test outputs and store them in the losses dictionary.
    Args:
    - losses (dict): A dictionary to store losses and metrics during training and evaluation. Generate it with initialise_losses().
    - test_outputs_xr (xarray.Dataset): An xarray Dataset containing the original and predicted footprints for the test set, as well as any relevant masks (e.g., fp_nan_mask) if needed for ignoring certain values in the metric calculations.
    """

    losses = losses.copy()  # make a copy of the losses dict to avoid modifying the original

    if "fp_nan_mask" in test_outputs_xr:
        fp_mask = test_outputs_xr.fp_nan_mask
    else:
        fp_mask = None

    computed_metrics = {}

    eval_metrics = gates_metrics.compute_footprint_metrics(
        test_outputs_xr.fp_original, test_outputs_xr.fp_pred, metrics=["iou", "mae", "mse","bias", "nmae"], nonzero=False, ignore_mask=fp_mask, threshold=1e-5)

    transformed_eval_metrics = gates_metrics.compute_footprint_metrics(
        test_outputs_xr.fp_transformed, test_outputs_xr.fp_transformed_pred, metrics=["iou", "mae", "mse","bias", "nmae"], ignore_mask=fp_mask, threshold=0, nonzero=False)
    
    static_mf_eval_metrics = gates_metrics.compute_static_mf_metrics(test_outputs_xr.fp_original, test_outputs_xr.fp_pred)

    for metric_name, metric_value in transformed_eval_metrics.items():
        if metric_name in losses["metrics_transformed"]:
            losses["metrics_transformed"][metric_name].append(metric_value)   

    for metric_name, metric_value in eval_metrics.items():
        if metric_name in losses["metrics_original"]:
            losses["metrics_original"][metric_name].append(metric_value) 

    for flux_type, metrics in static_mf_eval_metrics.items():
        if flux_type in losses["metrics_fluxes_static"]:
            for metric_name, metric_value in metrics.items():
                if metric_name in losses["metrics_fluxes_static"][flux_type]:
                    losses["metrics_fluxes_static"][flux_type][metric_name].append(metric_value)  

    computed_metrics["eval_metrics"] = eval_metrics
    computed_metrics["transformed_eval_metrics"] = transformed_eval_metrics
    computed_metrics["static_mf_eval_metrics"] = static_mf_eval_metrics

    return losses, computed_metrics
"""
def evaluate_outputs(test_outpts, true_fp, fp_mask):
    eval_metrics = gates_metrics.compute_footprint_metrics(
    true_fp, test_outpts,
    spatial_shape=(H, W),
    ignore_mask=nan_mask,
    threshold=1e-4,   # forwarded to iou
    nonzero=False,
)"""

def setup_GATES_model(parameters, training_ctx, paths_ctx):
    lr = parameters["learning_rate"]

    model = GraphSatelliteForecaster(training_ctx.grid, whole_world=False, feature_dim=training_ctx.n_variables, **parameters["model_parameters"])

    loss_fn = eval(parameters["loss_functions"]["criterion"])

    if parameters["dataloader"].get("nans_to_zeros", True):
        # if nans are being converted to zeros in the dataloader, we need to pass the fp_nan_mask to the loss function so it can ignore those values in the loss calculation
        nan_mask_label = "fp_nan_mask"
    else:
        nan_mask_label = None

    criterion = loss_fn(fp_labels=training_ctx.fp_labels, nan_mask_label=nan_mask_label, **parameters["loss_functions"].get("criterion_params", {}))

    loss_fn_test = eval(parameters["loss_functions"]["criterion_test"])
    criterion_test = loss_fn_test(fp_labels=training_ctx.fp_labels, nan_mask_label=nan_mask_label, **parameters["loss_functions"].get("criterion_test_params", {}))

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=parameters["epochs"]["patience"], verbose=parameters["verbose"], path=paths_ctx.model_path/f"{paths_ctx.model_name}_best.pt", use_wandb=parameters["use_wandb"], model_name=paths_ctx.model_name)

    if torch.cuda.is_available():
        model.cuda()
    
    model_ctx = ModelContext(
        model_name=parameters["model_name"],
        use_wandb=parameters["use_wandb"],
        device=training_ctx.device,
        optimizer=optimizer,
        criterion=criterion,
        criterion_test=criterion_test,
        lr=lr,
        early_stopping=early_stopping,
        epochs_num=parameters["epochs"]["training"],
        epochs_visualise=parameters["epochs"]["visualize"],
        epochs_save=parameters["epochs"]["model_save"],
        epochs_patience=parameters["epochs"]["patience"]
    )



    return model, model_ctx


def setup_boundary_model(parameters, training_ctx, paths_ctx):
    lr = parameters["learning_rate"]
    

    if parameters["network_decoder"] == "conv":
        print("Using conv network")
        model = GraphSatelliteForecasterConvClassifier(
            training_ctx.grid,
            whole_world=False,
            feature_dim=training_ctx.n_variables,
            aux_dim=training_ctx.aux_dim,
            num_classes=parameters['num_classes'],
            input_height=training_ctx.size,
            input_width=training_ctx.size,
            **parameters["model_parameters"],
        )
    else:
        print("Using normal network")
        model = GraphSatelliteForecasterClassifier(
            training_ctx.grid,
            whole_world=False,
            feature_dim=training_ctx.n_variables,
            aux_dim=training_ctx.aux_dim,
            num_classes=parameters['num_classes'],
            **parameters["model_parameters"],
        )

    criterion_params = parameters["loss_functions"].get("criterion_params", {})
    criterion_test_params = parameters["loss_functions"].get("criterion_test_params", {})

    loss_fn = eval(parameters["loss_functions"]["criterion"])
    loss_fn_test = eval(parameters["loss_functions"]["criterion_test"])

    
    # Simple loss like MSELoss with no spatial weighting
    criterion = loss_fn(**criterion_params)
    criterion_test = loss_fn_test(**criterion_test_params)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(
        patience=parameters["epochs"]["patience"],
        verbose=parameters.get("verbose", True),
        path=paths_ctx.model_path / f"{paths_ctx.model_name}_best.pt",
        use_wandb=parameters["use_wandb"],
        model_name=paths_ctx.model_name
    )

    if torch.cuda.is_available():
        model.cuda()

    model_ctx = ModelContext(
        model_name=parameters["model_name"],
        use_wandb=parameters["use_wandb"],
        device=training_ctx.device,
        optimizer=optimizer,
        criterion=criterion,
        criterion_test=criterion_test,
        lr=lr,
        early_stopping=early_stopping,
        epochs_num=parameters["epochs"]["training"],
        epochs_visualise=parameters["epochs"].get("visualize", 5),
        epochs_save=parameters["epochs"]["model_save"],
        epochs_patience=parameters["epochs"]["patience"]
    )

    return model, model_ctx

    
    

import os
from dask.distributed import Client, LocalCluster

def make_cluster():
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    mem_gb = int(os.environ.get("SLURM_MEM_PER_NODE", 8000)) / 1024  # MB → GB

    if n_cpus < 4:
        print("Fewer than 4 CPUs — skipping Dask cluster, using synchronous scheduler")
        return None, None

    
    n_workers = max(1, n_cpus - 2)
    mem_per_worker = f"{0.8 * mem_gb / n_workers:.1f}GB"

    print(f"{n_cpus} CPUs detected — setting up Dask cluster with {n_workers} workers")
    print(f"Memory: {mem_gb:.0f}GB total, allocating 80% → {mem_per_worker} per worker")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=mem_per_worker,
        local_directory="/tmp",
    )
    client = Client(cluster)
    print(f"Dask cluster: {n_workers} workers | Dashboard: {client.dashboard_link}")
    return client, cluster
    


