#from pyexpat import model

#from model.forecast import GraphSatelliteForecaster


import torch.optim as optim
import time
# from datetime import datetime
# import json
# import argparse
# from pathlib import Path

# import random
# import yaml

# import sys
# import pickle
# import random

import numpy as np
import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


import xarray as xr

from pathlib import Path

import wandb

from gates.model.forecast import GraphSatelliteForecaster


from gates import LoadSquareSatelliteData, get_square_satellite_inputs
import gates.data.datasets as gates_datasets
from gates.data.datasets import get_square_satellite_inputs_v2
import gates.evaluation.metrics as gates_metrics
import gates.evaluation.loss_functions as gates_losses

from .training_helperfuns import EarlyStopping #save_wandb_artifact, 

from .training_dataclasses import ModelContext

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

def setup_dynamic_edges(dynamic_wind=True, dynamic_latlon=False, wind_tuples=None, latlon_tuples=None, input_names=None, dynamic_earthdistance=False):
    """
    Prepare the input dictionary to pass to the GraphSatelliteForecaster relating to the mesh edges attributes. 
    Args:
    - dynamic_wind (bool): Whether to include dynamic wind-based edges in the graph. If True, the function will look for wind feature tuples in input_names based on wind_tuples.
    - dynamic_latlon (bool): Whether to include dynamic lat/lon-based edges in the graph. If True, the function will look for lat/lon feature tuples in input_names based on latlon_tuples.
    - wind_tuples (list of tuples): Optional list of tuples specifying the names and positions of wind features in input_names. Each tuple should be (feature_name, feature_dim, time_lag). If None, defaults to [("x_wind", 3, 0), ("y_wind", 3, 0)].
    - latlon_tuples (list of tuples): Optional list of tuples specifying the names and positions of lat/lon features in input_names. Each tuple should be (feature_name, feature_dim, time_lag). If None, defaults to [("lat_coords", 0, 0), ("lon_coords", 0, 0)].
    - input_names (list of str): List of feature names corresponding to the input variables, used to identify the indices of wind and lat/lon features based on the provided tuples.
    - dynamic_earthdistance (bool): Whether to compute dynamic earth distance edges based on lat/lon coordinates. Requires dynamic_latlon to be True (if dynamic_latlon is False, dynamic_earthdistance will be set to False and a warning will be printed)

    Returns:
    - dynamic_edge_params (dict): A dictionary containing the parameters to be passed to the GraphSatelliteForecaster for configuring dynamic edges. Example:
        {
            "wind_mesh_edges": True,
            "wind_indices": [3, 4],
            "latlon_mesh_edges": True,
            "latlon_indices": [0, 1],
            "dynamic_earthdistance": True
        }
    """
    
    dynamic_edge_params = {}

    if dynamic_wind:
        if wind_tuples is None:
            wind_tuples = [("x_wind", 3, 0), ("y_wind", 3, 0)]
        elif type(wind_tuples[0]) == list:
            wind_tuples = [tuple(t) for t in wind_tuples]
        wind_indices = [i for i, name in enumerate(input_names) if name in wind_tuples]
        dynamic_edge_params["wind_mesh_edges"] = True
        dynamic_edge_params["wind_indices"] = wind_indices
    else:
        dynamic_edge_params["wind_mesh_edges"] = False

    if dynamic_earthdistance and not dynamic_latlon:
        print("Warning: dynamic_earthdistance is True but dynamic_latlon is False - earth distance edges will not be computed because lat/lon coordinates are required for this. Setting dynamic_earthdistance to False.")
        dynamic_earthdistance = False
        dynamic_edge_params["dynamic_earthdistance"] = False

    if dynamic_latlon:
        if latlon_tuples is None:
            latlon_tuples = [("lat_coords", 0, 0), ("lon_coords", 0, 0)]
        elif type(latlon_tuples[0]) == list:
            latlon_tuples = [tuple(t) for t in latlon_tuples]
        latlon_indices = [i for i, name in enumerate(input_names) if name in latlon_tuples]
        if len(latlon_indices) != 2:
            raise ValueError(
                f"Expected exactly 2 latlon feature indices, found {len(latlon_indices)} "
                f"for tuples {latlon_tuples}. Check that lat/lon are included in input_names."
            )
        dynamic_edge_params["latlon_mesh_edges"] = True
        dynamic_edge_params["latlon_indices"] = latlon_indices
    else:
        dynamic_edge_params["latlon_mesh_edges"] = False

    if dynamic_earthdistance:
        dynamic_edge_params["dynamic_earthdistance"] = True

    return dynamic_edge_params


def load_GATES_data_v2(data_parameters, input_variables, datapath_args={}, verbose=True, load_into_memory=False, use_wandb=False):
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

    month_counter = 1
    total_time=0
    total_samples = 0

    if use_wandb:
        wandb.define_metric("loading/loaded_month")
        wandb.define_metric("loading/*", step_metric="loading/loaded_month")
    
    for year in years:
        for month in months:
            month_start = time.perf_counter()
            month_key = f"{year}-{month}"
            if verbose:
                print(f"Loading year={year}, month={month}")
            month_params = {**base_params, "year": year, "month": month}
            try:
                data = LoadSquareSatelliteData(**month_params, **datapath_args, verbose=verbose)

                inputs, data = get_square_satellite_inputs_v2(data, **input_variables, verbose=verbose)
                if load_into_memory:
                    print(f"Loading data into memory for {year}-{month} before concatenation...")
                    inputs = inputs.load()
                    inputs.close()
                    print("and footprints...")
                    data.fp_xr = data.fp_xr.load()
                    data.fp_xr.close()
                all_inputs.append(inputs)
                all_fp_xr.append(data.fp_xr)

                # Close the source file handles so they don't accumulate across months.
                if hasattr(data, "met_file") and data.met_file is not None:
                    data.met_file.close()
                if hasattr(data, "fp_data_full") and data.fp_data_full is not None:
                    data.fp_data_full.close()
                
                loaded_samples = len(data.fp_xr.fp.time)
                

            except Exception as e:
                print(f"Error loading data for {year}-{month}: {e}")

                # elapsed_mins = (time.perf_counter() - month_start) / 60
                # loading_times[month_key] = f"{elapsed_mins:.2f}mins"
                # print(f"{month_key} : {loading_times[month_key]}")

                loaded_samples = 0

            elapsed_mins = (time.perf_counter() - month_start) / 60
            loading_times[month_key] = f"{elapsed_mins:.2f}mins"

            print(f"{month_key} : {loading_times[month_key]}")

            total_samples += loaded_samples
            # sum of all loading times
            total_time += elapsed_mins
            if use_wandb:
                wandb.log({
                    "loading/loaded_month": month_counter,
                    "loading/train_time": elapsed_mins,
                    "loading/total_time": total_time,
                    "loading/samples_loaded": loaded_samples,
                    "loading/total_samples": total_samples,
                })
            month_counter += 1 

    print("")
    print("")
    print("----- Loading times for each month -----")
    print("\n".join(f"{k} : {v}" for k, v in loading_times.items()))
    print("")

    
    fp_xr = xr.concat(all_fp_xr, dim="time").sortby("time")
    inputs = xr.concat(all_inputs, dim="fp_time").sortby("fp_time")

    return fp_xr, inputs


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

    input_dataset = gates_datasets.InputsDataset(train_inputs, inputs_scaler, **input_scaler_params, verbose=parameters.get("verbose", False))
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

    print(parameters["input_scaler"])
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

    scalers = {"inputs_scaler": input_dataset.scaler, "input_names": list(train_scaled_inputs.variable_name.values), "fp_scaler": fp_dataset.scaler}

    return train_loader, test_loader, fp_labels, test_scaled_fp, scalers


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

    model = GraphSatelliteForecaster(training_ctx.grid, whole_world=False, feature_dim=training_ctx.n_variables, **parameters["model_parameters"], **training_ctx.dynamic_edges_params)

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
    


