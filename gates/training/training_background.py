from .training import _resolve_years_months, setup_input_dataset
import time
from gates import LoadSquareSatelliteData
from gates.data.datasets import get_square_satellite_inputs_v2
import xarray as xr
import pandas as pd
import numpy as np
from gates.data.load_background_data import load_cams_data, calculate_bg
import gates.data.datasets as gates_datasets

import torch
from .training_helperfuns import EarlyStopping
import torch.optim as optim
from .training_dataclasses import ModelContext
from model.forecast import GraphSatelliteForecasterConvClassifier,GraphSatelliteForecasterClassifier


def calculate_detrending_factor(bc_file, boundary="south", height_index=1):
    """
    Calculate a detrending factor for the boundary condition correction by taking the mean value of the specified boundary direction (e.g., "south") at the specified height index across all time points in the bc_file. This factor can be used to detrend the boundary condition correction by dividing the correction values by this factor, which helps to remove any systematic bias in the boundary condition data.
    """
    var_name = f"vmr_{boundary[0]}"
    mean_dim = "lat" if boundary in ["east", "west"] else "lon"
    detrending_factor = bc_file[var_name].isel(height=height_index).mean(dim=mean_dim)

    return detrending_factor

def get_auxiliary_bc_data(bc_file, height_indeces=[4], verbose=True):
    """
    Extract boundary condition information for each side of the domain (north, south, east, west) from the CAMS reanalysis data by selecting the variable corresponding to each direction (e.g., "vmr_n" for north), and taking the value at the specified height indeces along the height dimension, and at the midpoint along the latitude or longitude dimension as appropriate.

    Returns:
        xarray Dataset with coords "time" and "height_index" and variables "north", "south", "east", and "west", each containing the boundary condition at the midpoint for that direction at the specified height indeces.
    """
    bc_auxiliary_dict = {}
    for height_index in height_indeces:
        bc_auxiliary_dict[height_index] = {}
        var_names_dict = {"n":"north", "s":"south", "e":"east", "w":"west"}
        for direction in ["n", "s", "e", "w"]:
            var_name = f"vmr_{direction}"
            mid_index = bc_file[var_name].shape[-1] // 2
            if direction in ["n", "s"]:
                bc_auxiliary_dict[height_index][var_names_dict[direction]] = bc_file[var_name].isel(height=height_index, lon=mid_index)
            else:
                bc_auxiliary_dict[height_index][var_names_dict[direction]] = bc_file[var_name].isel(height=height_index, lat=mid_index)
    
            ## join the dict into an xarray dataset
    
        bc_auxiliary_here = xr.Dataset(bc_auxiliary_dict[height_index])
        # add height as a coordinate to the dataset
        bc_auxiliary_here = bc_auxiliary_here.assign_coords(height_index=height_index)
        bc_auxiliary_dict[height_index] = bc_auxiliary_here

    bc_auxiliary = xr.concat([bc_auxiliary_dict[height_index] for height_index in height_indeces], dim="height_index")

    bc_auxiliary.attrs["heights"] = [bc_file.height.values[height_index] for height_index in height_indeces]

    if verbose:
        print(f"Extracted boundary condition information at the midpoint of the lat and lon, for heights: {bc_auxiliary.attrs['heights']}")
    return bc_auxiliary 



def load_GATES_data_with_bg(data_parameters, input_variables, datapath_args={}, detrend=True, verbose=True, load_into_memory=True, use_aux_bc=True, aux_indeces=[4]):
    """
    Loads footprints, inputs  and background data for each year-month pair specified in data_parameters,
    returning them as concatenated xarrays rather than a LoadSquareSatelliteData object.

    data_parameters may contain:
        years : list[int | str]     — multiple years (alternative to year)
        months: list[int | str]     — explicit list of months
        load_into_memory : bool     — if True, materialise each month into memory before
                                      concatenating (avoids large dask graphs at the cost
                                      of sequential I/O); default False
        detrend : bool             — if True, apply a simple detrending to the background data by calculating a detrending factor based on the value of the boundary conditions at a specified height and subtracting the background correction by this factor; default True
        use_aux_bc : bool          — if True, extract and return auxiliary boundary condition data; default True
        aux_indeces : list[int]    — list of height indices for auxiliary boundary condition data; default [4]

    Returns:
        fp_xr  : xr.Dataset   — concatenated footprints (time, lat, lon)
        inputs : xr.DataArray — concatenated met inputs  (time, lat, lon, variable_name)
        bgs    : xr.DataArray — concatenated background corrections (time, lat, lon)
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
    all_bgs = []
    all_aux_data = []
    loading_times = {}

    for year in years:
        for month in months:
            month_start = time.perf_counter()
            month_key = f"{year}-{month}"
            if verbose:
                print(f"Loading year={year}, month={month}")
            month_params = {**base_params, "year": year, "month": month}
            try:
                data = LoadSquareSatelliteData(**month_params, **datapath_args, verbose=verbose, load_bcs=True)
            except Exception as e:
                print(f"Error loading data for {year}-{month}: {e}")
                elapsed_mins = (time.perf_counter() - month_start) / 60
                loading_times[month_key] = f"{elapsed_mins:.2f}mins"
                print(f"{month_key} : {loading_times[month_key]}")
                continue

            # set up the inputs
            inputs, data = get_square_satellite_inputs_v2(data, **input_variables, verbose=verbose)


            ### load the boundary condition data for this month
            monthly_boundary = load_cams_data(data.domain, species="ch4", year=year, month=month)
            monthly_boundary.load()

            # calculate the convolution between the boundary conditions and the boundary footprints to get the background contribution to the concentrations for this month
            background = calculate_bg(data.fp_data_full, monthly_boundary)

            if detrend:
                # extract the monthly boundary condition information at the midpoint of the south domain to use as a detrending factor, and reindex to the time coordinate of the background data using forward fill
                det_factor = calculate_detrending_factor(monthly_boundary)
                det_factor_reindexed = det_factor.reindex(time=background.time, method="ffill", tolerance=pd.Timedelta("32D") )

                background = background - det_factor_reindexed

            if use_aux_bc:
                print("Getting auxiliary cams data for month")
                aux_data = get_auxiliary_bc_data(monthly_boundary, height_indeces=aux_indeces, verbose=verbose)
            else:
                aux_data = None
                

            if load_into_memory:
                print(f"Loading data into memory for {year}-{month} before concatenation...")
                inputs = inputs.load()
                data.fp_xr = data.fp_xr.load()
                background = background.load()
                aux_data = aux_data.load()

            all_inputs.append(inputs)
            all_fp_xr.append(data.fp_xr)
            all_bgs.append(background)
            all_aux_data.append(aux_data)



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
    bgs = xr.concat(all_bgs, dim="time").sortby("time")
    if all_aux_data[0] is not None:
        aux_data = xr.concat(all_aux_data, dim="time").sortby("time")
    else:
        aux_data = None
    return fp_xr, inputs, bgs, aux_data

# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------

def normalize_data(outputs, norm_vals=None):
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
    if norm_vals is None:
        if isinstance(outputs, xr.DataArray):
            outputs_mean = float(outputs.mean())
            outputs_std = float(outputs.std())
        else:
            outputs_mean = np.mean(outputs)
            outputs_std = np.std(outputs)
        norm_vals = (outputs_mean, outputs_std)

    outputs_mean, outputs_std = norm_vals
    normalized_outputs = (outputs - outputs_mean) / outputs_std

    return normalized_outputs, norm_vals


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

def format_aux_data(auxiliary_cams, time_coord):
    ## stack and relable the auxiliary_cams data to prepare for concatenation
    auxiliary_cams = auxiliary_cams.reindex(time=time_coord, method="ffill", tolerance=pd.Timedelta("32D") )

    aux_labels = [
    f"cams_aux_{h_idx}"
    for var in auxiliary_cams.data_vars
    for h_idx in auxiliary_cams.height_index.values
    ]

    auxiliary_cams = auxiliary_cams.to_array(dim="variable")

    auxiliary_cams = auxiliary_cams.stack(aux=("variable", "height_index"))
    auxiliary_cams = auxiliary_cams.drop_vars(['aux', 'variable', 'height_index', "height", "lat","lon"])
    auxiliary_cams["aux"] = ("aux", aux_labels)

    return auxiliary_cams


def concat_auxiliary_to_inputs(scaled_inputs, auxiliary_cams):
    """
    Concatenate auxiliary CAMS features onto scaled inputs along the variable_name dimension,
    broadcasting the auxiliary values across all lat/lon grid cells.

    Args:
        scaled_inputs (xr.DataArray): Shape (fp_time, lat, lon, variable_name).
        auxiliary_cams (xr.DataArray): Shape (time, aux) with aux coordinate names.

    Returns:
        xr.DataArray: Concatenated inputs of shape (fp_time, lat, lon, variable_name)
            with aux features appended along variable_name.
    """

    

    aux_names = list(auxiliary_cams.aux.values)
    lat_size = scaled_inputs.sizes["lat"]
    lon_size = scaled_inputs.sizes["lon"]
    n_aux = len(aux_names)

    # Rename 'time' to 'fp_time' to match scaled_inputs
    aux_fp_time = auxiliary_cams.rename({"time": "fp_time"})

    # Expand (fp_time, aux) -> (fp_time, lat, lon, aux) by broadcasting
    # Use expand_dims then broadcast_to via xarray
    aux_expanded = aux_fp_time.expand_dims(
        dim={"lat": scaled_inputs.lat, "lon": scaled_inputs.lon},
    )
    # Reorder to match scaled_inputs dim order
    aux_expanded = aux_expanded.transpose("fp_time", "lat", "lon", "aux")

    # Build a MultiIndex compatible with the variable_name MultiIndex on scaled_inputs
    aux_multiindex = pd.MultiIndex.from_arrays(
        [aux_names,
         [0] * n_aux,   # levels — no pressure level for auxiliary
         [0] * n_aux],  # time_delta — no time shift for auxiliary
        names=["variable", "levels", "time_delta"]
    )
    
    # Rename aux dim to variable_name and assign the MultiIndex coordinate
    aux_expanded = aux_expanded.rename({"aux": "variable_name"})
    aux_expanded = aux_expanded.assign_coords(
       xr.Coordinates.from_pandas_multiindex(aux_multiindex, "variable_name")
    )

    return xr.concat([scaled_inputs, aux_expanded], dim="variable_name")

def normalize_boundary_data(bgs, aux_data=None, norm_vals=None):
    """
    Normalize the boundary condition background data and auxiliary CAMS data (if provided) using the mean and std from the training data, or compute them if norm_vals is None. Returns the normalized boundary data, normalized auxiliary data, and the normalization values used.
    """
    if norm_vals is None:
        norm_bgs, bg_norm = normalize_data(bgs)
        if aux_data is not None:
            norm_aux_data, aux_norm = normalize_data(aux_data)
        else:
            norm_aux_data, aux_norm = None, None
    
    else:
        norm_bgs, bg_norm = normalize_data(bgs, norm_vals=norm_vals["outputs"])
        if aux_data is not None:
            norm_aux_data, aux_norm = normalize_data(aux_data, norm_vals=norm_vals["auxiliary"])
        else:
            norm_aux_data, aux_norm = None, None
        
    norm_vals_out = {"outputs": bg_norm}
    if aux_data is not None:
        norm_vals_out["auxiliary"] = aux_norm

    return norm_bgs, norm_aux_data, norm_vals_out


def setup_boundary_dataloaders(parameters, train_inputs, train_outputs, test_inputs, test_outputs,
                                train_auxiliary_cams=None, test_auxiliary_cams=None):
    # Scale inputs using training statistics
    
    input_dataset = setup_input_dataset(parameters, train_inputs)
    train_scaled_inputs = input_dataset.transform(train_inputs)
    test_scaled_inputs = input_dataset.transform(test_inputs)

    '''
    # Force compute after scaling — scaler may return dask-backed xarray
    print("Computing scaled inputs into memory...")
    train_scaled_inputs = train_scaled_inputs.compute() if hasattr(train_scaled_inputs, 'compute') else train_scaled_inputs
    test_scaled_inputs = test_scaled_inputs.compute() if hasattr(test_scaled_inputs, 'compute') else test_scaled_inputs
    '''
    # Append pre-normalised auxiliary CAMS features to inputs if use_baselines is True
    ## TO CHANGE PARAMS
    use_auxiliary_bc = parameters.get("use_auxiliary_bc", True)

    if use_auxiliary_bc:
        train_auxiliary_cams.load()
        test_auxiliary_cams.load()
        
        print("Concatenating inputs and auxiliary cams")
        train_scaled_inputs = concat_auxiliary_to_inputs(train_scaled_inputs, train_auxiliary_cams)
        test_scaled_inputs = concat_auxiliary_to_inputs(test_scaled_inputs, test_auxiliary_cams)


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

    train_loader = gates_datasets.make_boundary_dataloader(
        train_scaled_inputs, train_outputs,
        batch_size=batch_size,
        randomize=True,
        dataloader_params=dataloader_params,
        flatten=True
    )

    test_dataloader_params = {
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None
    }
    print("SPECIAL TEST PARAMS", test_dataloader_params)

    test_loader = gates_datasets.make_boundary_dataloader(
        test_scaled_inputs, test_outputs,
        batch_size=test_batch_size,
        randomize=False,
        dataloader_params=test_dataloader_params,
        flatten=True
    )

    # if boundary_labels_train != boundary_labels_test:
    #     raise ValueError(
    #         "The labels for the training and test boundary datasets do not match. "
    #         "Check that train and test outputs have the same structure and number of columns."
    #     )

    scalers = {"inputs_scaler": input_dataset.scaler}

    return train_loader, test_loader, scalers


def setup_boundary_model(parameters, training_ctx, paths_ctx):
    lr = parameters["learning_rate"]
    
    decoder = parameters["model_parameters"].get("decoder", "conv")
    parameters["model_parameters"].pop("decoder", None)

    num_classes = parameters["model_parameters"].get("num_classes", 1)
    parameters["model_parameters"].pop("num_classes", None)
    
    if decoder == "conv":
        print("Using conv network")
        model = GraphSatelliteForecasterConvClassifier(
            training_ctx.grid,
            whole_world=False,
            feature_dim=training_ctx.n_variables,
            aux_dim=training_ctx.aux_dim, # this will be soon be deprecated
            num_classes=num_classes,
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
            num_classes=num_classes,
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


def initialise_boundary_losses():
    """
    Return a dictionary to store losses and metrics during training and evaluation. 
    """
    losses = {
        "train": [],
        "test": [],

    }
    return losses