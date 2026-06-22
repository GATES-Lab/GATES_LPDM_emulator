"""
author: Elena Fillola @elenafillo
"""

import numpy as np
import xarray as xr
import datetime
import pandas as pd
import joblib
from pathlib import Path
import warnings

import torch
import xbatcher as xb
import xbatcher.loaders.torch

from .load_data_helper_funs import *

from .load_data import cut_satellite_met, _get_release_idxs, _pad_domain

def _stack_and_label_variables(ds, var_names, var_type, met_variables_dict=None, verbose=False):
    """
    Stack requested variables into a single variable_name dimension with tuple labels.
    Returns (stacked_dataarray_or_None, warnings).
    """
    if len(var_names) == 0:
        return None
    if verbose:
        print(f"Setting up {var_type}: {var_names}")


    if var_type == "met_with_levels":

        filtered_vars = {}
        filtered_vars = []
        for var in var_names:
            if var not in ds.data_vars:
                warnings.warn(f"requested variable {var} is not available and will be skipped")
                continue

            if "levels" not in ds[var].coords:
                warnings.warn(f"variable {var} has no 'levels' coordinate and will be skipped")
                continue

            #requested_levels = list(met_variables_dict.get(var, [])) if met_variables_dict is not None else []
            #available_levels = list(ds[var].levels.values)
            #valid_levels = [lev for lev in requested_levels if lev in available_levels]
            #dropped_levels = [lev for lev in requested_levels if lev not in available_levels]

            #if dropped_levels:
            #    warnings.warn(f"requested levels {dropped_levels} for variable {var} are not available and will be skipped")

            #if len(valid_levels) == 0:
            #    warnings.warn(f"variable {var} has no valid levels left after filtering and will be skipped")
            #    continue

            #filtered_vars[var] = ds[var].sel(levels=valid_levels)
            filtered_vars.append(var)
        

        if len(filtered_vars) == 0:
            return None
        
        data = ds[filtered_vars].transpose("fp_time", "lat", "lon", "levels", "time_delta")

        #data = xr.Dataset(filtered_vars).transpose("fp_time", "lat", "lon", "levels", "time_delta")

        #data = ds.transpose("fp_time", "lat", "lon", "levels", "time_delta")

    elif var_type == "surface_met":
        valid_vars = [v for v in var_names if v in ds.data_vars]
        missing_vars = [v for v in var_names if v not in ds.data_vars]
        for mv in missing_vars:
            warnings.warn(f"requested surface variable {mv} is not available and will be skipped")

        if len(valid_vars) == 0:
            return None

        data = ds[valid_vars].assign_coords(levels=0).expand_dims("levels")
        data = data.transpose("fp_time", "lat", "lon", "levels", "time_delta")

    elif var_type == "static":
        valid_vars = [v for v in var_names if v in ds.data_vars]
        missing_vars = [v for v in var_names if v not in ds.data_vars]
        for mv in missing_vars:
            warnings.warn(f"requested static variable {mv} is not available and will be skipped")

        if len(valid_vars) == 0:
            return None

        data = ds[valid_vars].assign_coords(levels=0, time_delta=0).expand_dims("levels").expand_dims("time_delta")
        data = data.transpose("fp_time", "lat", "lon", "levels", "time_delta")

    else:
        raise ValueError(f"unknown var_type: {var_type}")
    
    stacked = data.to_stacked_array(
        new_dim="variable_name",
        sample_dims=["fp_time", "lat", "lon"],
        name=f"stacked_{var_type}",
    )

    stacked = stacked.reorder_levels(dim_order={'variable_name': ["variable", 'levels', 'time_delta']})
    variable_labels = list(stacked.variable_name.values)

    if len(variable_labels) == 0:
        return None

    # Always rebuild from named coords to guarantee (variable, levels, time_delta) order,
    # regardless of the internal stacking order used by to_stacked_array.
    if all(coord in stacked.coords for coord in ["variable", "levels", "time_delta"]):
        variable_labels = list(
            zip(
                stacked["variable"].values,
                stacked["levels"].values,
                stacked["time_delta"].values,
            )
        )
    elif not isinstance(variable_labels[0], tuple):
        raise ValueError(
            "Could not construct variable_name tuples from stacked data. "
            "Please check requested variables and levels."
        )

    mindex = pd.MultiIndex.from_tuples(variable_labels, names=["variable", "levels", "time_delta"])
    stacked = stacked.drop_vars({"time_delta", "levels", "variable"}, errors="ignore")
    stacked = stacked.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, "variable_name"))

    return stacked


def get_square_satellite_inputs(data, met_variables, time_deltas=None, static_variables=None, verbose=True, add_timedelta_zero=True):
    """
    Refactored version of get_square_satellite_inputs with unified stacking logic and explicit edge-case handling.

    Inputs:
    - data: LoadSquareSatelliteData object 
    - met_variables: dict, of shape {'variable_name':levels_to_extract, 'surface_variable':[], ...}. For each atmospheric variable with levels, pass the levels to extract as a list. For each surface variable, pass an empty list
    - time_deltas: list, of t-H hours to extract the met variables. t=0 (ie the time of the satellite measurement) is extracted automatically if add_timedelta_zero is True. time_deltas=[6,12] extracts the data at t=0, t-6h and t-12h.
    - static_variables: list of static variables to add, eg topog, landcover, lat_coords. You can see or increase the list of valid names get_static_variables_functions()
    - verbose: bool, if true prints out the steps of the function
    - add_timedelta_zero: bool, if true adds t=0 to the time_deltas if it is not already present, to ensure that the meteorology at the time of the measurement is included. If false, only the time deltas passed in time_deltas will be extracted
    

    Returns:
    - xarray DataArray with dims (fp_time, lat, lon, variable_name)
    - input data object, which may be updated if indices were removed during meteorology filtering

    Notes:
    - ``met_variables`` must map variable names to lists of levels. Use an empty list for surface variables.
    - ``time_deltas`` is normalized to a sorted, unique list and ``0`` is added automatically when ``add_timedelta_zero`` is True.
    - variables that are requested as surface variables but actually have levels in the meteorology are warned about and skipped.
    """
    assert hasattr(data, "dataset_format"), "It doesn't seem this is a SatelliteData object"
    assert data.met_processed is True, "Make sure that you have loaded and cut the meteorology in the SatelliteData object"

    if type(met_variables) is not dict:
        raise ValueError("met_variables should be a dict of shape {'variable_name':levels_to_extract, 'surface_variable':[], ...}. For each atmospheric variable with levels, pass the levels to extract as a list. For each surface variable, pass an empty list")

    if time_deltas is None:
        time_deltas = []
    else:
        time_deltas = list(time_deltas)

    if static_variables is None:
        static_variables = []
    else:
        static_variables = list(static_variables)

    if verbose:
        print("------------------------")
        print("---EXTRACTING MET DATA---")

    if add_timedelta_zero and 0 not in time_deltas:
        time_deltas.append(0)
    time_deltas = sorted(set(time_deltas))
    if verbose: print(f"Time deltas: {time_deltas}")

    met_variables_needed = list(met_variables.keys())
    surface_variables_needed = [var_name for var_name, levs in met_variables.items() if levs == []]
    levels_variables_needed = [var_name for var_name, levs in met_variables.items() if len(levs) > 0]
    min_levels_needed = sorted({lev for levels in met_variables.values() for lev in levels})

    all_met_files = {}
    met_nan_idxs = {}

    if len(time_deltas) > 0:
        if verbose: print(f"extracting met at t-H for H in: {time_deltas}")
        for delta in time_deltas:
            if delta == 0:
                met = data.met
                met = select_met_levels(met, levels=min_levels_needed.copy())
                met = select_met_variables(met, variables=met_variables_needed.copy())
                met = met.swap_dims({"time": "fp_time"})
                met = met.drop_vars("time", errors="ignore")

                for var in [v for v in met_variables_needed if v in met.data_vars]:
                    met[var] = met[var].expand_dims("time_delta").assign_coords({"time_delta": ("time_delta", [delta])})

                missing_in_zero = [v for v in met_variables_needed if v not in met.data_vars]
                for mv in missing_in_zero:
                    warnings.warn(f"Warning: requested variable {mv} is not available in met data at t=0 and will be skipped")
            else:
                if data.dataset_format == "square":
                    met, nan_idxs = cut_satellite_met(
                        data.met_file,
                        data.fp_data_full,
                        metsize=data.metsize,
                        time_delta=delta,
                        relevant_levels=min_levels_needed,
                        relevant_variables=met_variables_needed,
                        pad_mode=data.fill_outofdomain_with,
                        load=False,
                        add_wind_direction=True,
                        return_nan_idxs=True,
                    )
                    met_nan_idxs[delta] = nan_idxs

                met = met.swap_dims({"time": "fp_time"})
                met = met.drop_vars("time", errors="ignore")
                # if met_timestamps is a coordinate or variable, drop it
                if "met_timestamps" in met.data_vars:
                    met = met.drop_vars("met_timestamps", errors="ignore")


            all_met_files[delta] = met
            del met

    if time_deltas != list(all_met_files.keys()):
        warnings.warn(f"the time_deltas passed {time_deltas} are not the same as the time_deltas of the extracted met {list(all_met_files.keys())}. Check that cut_satellite_met is working correctly for the passed time_deltas")

    if len(all_met_files.keys()) == 1:
        only_delta = list(all_met_files.keys())[0]
        full_met = all_met_files[only_delta].transpose("fp_time", "lat", "lon", ..., "time_delta")
    else:
        full_met = xr.concat(list(all_met_files.values()), dim="time_delta", data_vars=met_variables_needed)
        full_met = full_met.transpose("fp_time", "lat", "lon", ..., "time_delta")
    
    all_nan_idxs = np.unique(np.concatenate([np.atleast_1d(v) for v in met_nan_idxs.values()])) if len(met_nan_idxs) > 0 else np.array([], dtype=int)

    for v in full_met.data_vars:
        full_met[v] = full_met[v].astype("float32", copy=False)

    wrongly_surface_vars = [
        var_name
        for var_name in surface_variables_needed
        if var_name in full_met.data_vars and "levels" in full_met[var_name].coords
    ]
    if len(wrongly_surface_vars) > 0:
        warnings.warn(
            "The following variables were passed as surface variables but have levels in meteorology and "
            f"will be skipped: {wrongly_surface_vars}. Pass explicit levels, e.g. {{'x_wind': [3]}}.",
            UserWarning,
        )
        surface_variables_needed = [v for v in surface_variables_needed if v not in wrongly_surface_vars]

    if len(all_nan_idxs) > 0:
        full_met = full_met.drop_sel(fp_time=all_nan_idxs)
        warnings.warn(f"removing {len(all_nan_idxs)} indeces due to problems with interpolating meteorology for the passed time_deltas")
        data.remove_indeces(all_nan_idxs)

    input_arrays = []

    ### SETTING UP VARIABLES WITH LEVELS
    if len(levels_variables_needed) > 0:
        stacked_levels_met = _stack_and_label_variables(
            full_met,
            levels_variables_needed,
            "met_with_levels",
            met_variables_dict=met_variables,
            verbose=verbose,
        )

        if stacked_levels_met is not None:
            input_arrays.append(stacked_levels_met)

    ### SETTING UP VARIABLES WITH NO LEVELS
    if len(surface_variables_needed) > 0:
        stacked_surface_met = _stack_and_label_variables(
            full_met,
            surface_variables_needed,
            "surface_met",
            verbose=verbose,
        )

        if stacked_surface_met is not None:
            input_arrays.append(stacked_surface_met)

    ### SETTING UP NON-MET VARIABLES
    if len(static_variables) > 0:
        #if verbose:
        #   print(f"Setting up static variables: {static_variables}")

        static_variables_functions = get_static_variables_functions()

        if data.dataset_format == "square":
            (static_ds,) = xr.broadcast(full_met[["lat_coords", "lon_coords"]])
        elif data.dataset_format == "domain":
            (static_ds,) = xr.broadcast(
                full_met.assign({"lat_coords": (("lat"), full_met.lat.values), "lon_coords": (("lon"), full_met.lon.values)})
                [["lat_coords", "lon_coords", "fp_time"]]
            )
            static_ds = static_ds[["lat_coords", "lon_coords"]]

            if "time" not in data.topog.coords:
                data.topog = data.topog.broadcast_like(static_ds, exclude=["lat", "lon", "landcover_level"])
                data.topog = data.topog.assign_coords({"lat": static_ds.lat.values, "lon": static_ds.lon.values}).rename({"fp_time": "time"})
        else:
            raise ValueError(f"Unsupported dataset_format: {data.dataset_format}")

        for var in static_variables:
            if var in ["topog", "landcover", "landcover_disaggregated"]:
                assert hasattr(data, "topog"), "Load topog on the data object before trying to extract this as an input!"
                static_ds = static_variables_functions[var](data.topog, static_ds)
            elif "domain" in var:
                if var in static_variables_functions:
                    static_ds = static_variables_functions[var](data.fp_data_full, static_ds)
                else:
                    warnings.warn(f"variable {var} was not found in the list of known functions and will be skipped")
            elif var in list(static_variables_functions.keys()) and var not in ["lat_coords", "lon_coords"]:
                static_ds = static_variables_functions[var](static_ds)
            elif var not in ["lat_coords", "lon_coords"]:
                warnings.warn(f"variable {var} was not found in the list of known functions and will be skipped")

        if "lat_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lat_coords"], errors="ignore")
        if "lon_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lon_coords"], errors="ignore")

        stacked_static_inputs = _stack_and_label_variables(
            static_ds,
            list(static_ds.data_vars),
            "static",
            verbose=verbose,
        )

        if stacked_static_inputs is not None:
            input_arrays.append(stacked_static_inputs)

    ### FINAL CONCATENATION
    if len(input_arrays) == 0:
        raise ValueError(
            "All requested variables/levels were unavailable after filtering. "
            "Please check met_variables/static_variables against dataset contents."
        )

    concatenated_inputs = xr.concat(input_arrays, dim="variable_name")
    mindex = pd.MultiIndex.from_tuples(concatenated_inputs.variable_name.values, names=["variable", "levels", "time_delta"])
    concatenated_inputs = concatenated_inputs.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, "variable_name"))

    concatenated_inputs.attrs = {
        "source": concatenated_inputs.attrs["source"] if "source" in concatenated_inputs.attrs else "unknown",
        "time_deltas": time_deltas,
        "generated on": str(datetime.datetime.now()),
    }
    concatenated_inputs = concatenated_inputs.astype("float32", copy=False)

    if concatenated_inputs.sizes.get("variable_name", 0) == 0:
        raise ValueError(
            "All requested variables/levels were unavailable after filtering. "
            "Please check met_variables/static_variables against dataset contents."
        )

    return concatenated_inputs, data


####
####
####
####

### Transforming and scaling data - inputs
class XarrayScaler:
    """
    A simple scaler for xarray DataArrays that applies the standard scaling transformation. 
    """
    def __init__(self, compute=True, kwargs={}, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.scaler_type = "standard"
        self.scaler_name = "XarrayScaler"
        self.compute = compute

    def fit(self, da: xr.DataArray):
        """Compute stats over all dims except the ones you want to preserve (e.g. variable)."""
        if self.mean is None:
            self.mean = da.mean()
            self.std = da.std()
            if self.compute:
                print("Computing mean and std for scaler...")
                self.mean = self.mean.compute().astype("float32").values
                self.std = self.std.compute().astype("float32").values
        self.params = {"mean": self.mean, "std": self.std}
        return self
    
    def transform(self, da: xr.DataArray) -> xr.DataArray:
        #da = da.astype("float32", copy=False)
        return (da - self.mean) / (self.std + 1e-8)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        return self.fit(da).transform(da)
    
class XarrayMinMaxScaler:
    """
    A simple scaler for xarray DataArrays that applies minmax transformation. 
    Pass manual_min and manual_max to the constructor if you want to specify the min and max values to use for scaling, otherwise they will be computed from the data while fitting. Pass feature_range to specify the range to scale the data to, default is (0, 1).
    """
    def __init__(self, feature_range=(0, 1), manual_min =None, manual_max=None, compute=True):
        self.min = manual_min
        self.max = manual_max
        self.feature_range = feature_range
        self.scaler_type = "minmax"
        self.scaler_name = "XarrayMinMaxScaler"
        self.compute = compute

    def fit(self, da: xr.DataArray):
        #if self.min is None or self.max is None:
            #da = da.astype("float32", copy=False)
        
        if self.min is None:
            self.min = da.min()
            if self.compute:
                print("Computing min for scaler...")
                self.min = self.min.compute().astype("float32").values
        if self.max is None:
            self.max = da.max()
            if self.compute:
                self.max = self.max.compute().astype("float32").values

        self.params = {"min": self.min, "max": self.max, "feature_range": self.feature_range}
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        #da = da.astype("float32", copy=False)
        scale = self.feature_range[1] - self.feature_range[0]
        return self.feature_range[0] + scale * (da - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        return self.fit(da).transform(da)


class InputsDataset:
    """
    Wrapper for an input ``xarray.DataArray`` that applies a scaler to the inputs.

    The scaler is fitted on the stored inputs when `fit` is called and can then be applied to compatible inputs via `transform`. When ``fit_on_subsample`` is
    between 0 and 1, only a random subset of ``fp_time`` indices is used for fitting and the selected subset is exposed on ``subsampled_inputs``.

    Valid scalers:
     - DefaultInputsScaler, applies a standard scaler to each variable across all timesteps per level for meteorological variables, and a minmax scaler to land cover and topog variables
    - HandcraftedInputsScaler (only a placeholder for now), which applies different scalers to different variables based on some predefined logic
    """
    def __init__(self, inputs: xr.DataArray, scaler=None, fit_on_subsample=1, scaler_params={}, verbose=False, compute=True):
        self.inputs = inputs
        self.scaler = scaler
        self.verbose = verbose
        self.compute = compute

        self.fit_on_subsample = fit_on_subsample

        if scaler is None:
            self.scaler = DefaultInputsScaler(**scaler_params, verbose=self.verbose, compute=self.compute)
        else:
            self.scaler = scaler(**scaler_params)

    def fit(self):
        if self.fit_on_subsample<0 or type(self.fit_on_subsample) not in [int, float] or self.fit_on_subsample>1: 
            raise ValueError(f"fit_on_subsample should be a float between 0 and 1, but got {self.fit_on_subsample}. Please provide a valid value for fit_on_subsample.")
        elif self.fit_on_subsample<1:
            times = pd.DatetimeIndex(self.inputs.fp_time.values)
            n_samples = int(len(times)*self.fit_on_subsample)
            if self.verbose: print(f"fit_on_subsample is {self.fit_on_subsample}, so only using {n_samples} samples to fit the scaler. samples chosen randomly")

            selected_times = np.sort(np.random.choice(times, n_samples, replace=False))

            self.subsampled_inputs = self.inputs.sel(fp_time=selected_times) 

            self.scaler.fit(self.subsampled_inputs)

        elif self.fit_on_subsample>=1:
            self.scaler.fit(self.inputs)
        
    def transform(self, inputs):
        transformed = self.scaler.transform(inputs)
        if transformed.dtype != "float32":
            transformed = transformed.astype("float32", copy=False)
        return transformed

    def fit_transform(self):
        self.fit()
        return self.transform(self.inputs)


class HandcraftedInputsScaler:
    def __init__(self):
        raise NotImplementedError("This is a placeholder for a scaler that applies different scalers to different variables based on some predefined logic, e.g. centering windspeeds around 0 and applying a minmax scaler to topog with specified heights ")

class DefaultInputsScaler:
    """
    Default input scaler for the stacked input ``xarray.DataArray``.

    Meteorological variables are standardized per variable and level (ie across all time deltas).
    Variables listed in ``minmax_variables`` are scaled with min-max scaling. The fitted scalers are stored by full variable tuple, and `transform` returns a new DataArray
    named ``stacked_transformed_inputs`` with the same dims and ``variable_name`` labels as the inputs.
    """
    def __init__(self, minmax_variables=["land_cover", "topog", "x_coords", "y_coords", "lat_coords", "lon_coords", "xy_distance_centre", "sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords"], verbose=True, compute=True):
        self.minmax_variables = minmax_variables
        self.scalers = {}
        self.scaler_name = "DefaultInputsScaler"
        self.verbose = verbose
        self.compute = compute

    def fit(self, inputs: xr.DataArray):

        variable_names = inputs.variable_name.values
        varnames = [var[0] for var in inputs.variable_name.values]
        varnames = np.unique(varnames)
        self.fitted_variable_names = list(variable_names)

        for varname in np.unique(varnames):
            var_data = inputs.sel(variable=varname)
            if varname in self.minmax_variables:
                if self.verbose: print(f"fitting minmax scaler for var {varname}")
                scaler = XarrayMinMaxScaler(compute=self.compute)
                scaler = scaler.fit(var_data)

                # save the scaler
                for vc in variable_names:
                    if vc[0] == varname and len(vc)==3:
                        self.scalers[vc] = scaler
                
            else:
                if self.verbose: print(f"fitting standardise scaler for var {varname}")
                levels = np.unique(var_data.levels.values)

                for level in levels:
                    if self.verbose: print(f"      at level {level}")
                    scaler = XarrayScaler(compute=self.compute)
                    level_data = var_data.sel(levels=level)
                    scaler = scaler.fit(level_data)

                    # save the scaler for this variable and level for each variable tuple in the multiindex that matches this variable and level
                    for vc in variable_names:
                        if vc[0] == varname and vc[1] == level and len(vc)==3:
                            self.scalers[vc] = scaler

        
    def transform(self, inputs: xr.DataArray) -> xr.DataArray:
        variable_names = inputs.variable_name.values
        for varname in variable_names:
            if varname not in self.fitted_variable_names:
                raise ValueError(f"Variable name {varname} in inputs is not in the variable names that were fitted on: {self.fitted_variable_names}. Please fit the scaler on data that contains all the variable names in the inputs.")

        transformed = inputs.copy()
        
        transformed_variables = []

        for varname in np.unique(variable_names):
            #print(f"Transforming {varname}")
            var_data = inputs.sel(variable_name=varname)
            
            scaler = self.scalers.get(varname, None)
            if scaler is None:
                print(f"No scaler found for variable {varname}, skipping transformation.")
                transformed_variables.append(var_data)
                continue
            #print(f"Using scaler {scaler.scaler_type} for variable {varname}")
            transformed_data = scaler.transform(var_data)
            transformed_data = transformed_data.rename(varname)

            transformed_variables.append(transformed_data)

                 
        transformed = xr.concat(transformed_variables, dim="variable_name")
        mindex = pd.MultiIndex.from_tuples(transformed.variable_name.values, names=["variable", "levels", "time_delta"])
        mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "variable_name")
        transformed = transformed.assign_coords(mindex_coords)

        transformed = transformed.rename("stacked_transformed_inputs")
        transformed.attrs = {"transformer": "DefaultInputsScaler", "minmax_variables": self.minmax_variables, "generated_on": str(datetime.datetime.now())}
        transformed = transformed.transpose("fp_time", "lat", "lon", "variable_name")
        return transformed

    def fit_transform(self, inputs: xr.DataArray) -> xr.DataArray:
        self.fit(inputs)
        return self.transform(inputs)

class HandcraftedInputsScaler:
    """
    IN DEVELOPMENT - NOT READY FOR USE
    Input scaler using pre-determined statistics rather than computing them from data.

    ``stats`` is a path to a JSON file or a dict. Each variable entry has a ``"type"`` key
    (``"standard"`` or ``"minmax"``) plus per-level statistics::

        {
            "x_wind":   {"type": "standard", "3": {"mean": 5.21, "std": 3.14}},
            "topog":    {"type": "minmax",   "0": {"min": -50.0, "max": 3200.0}}
        }

    Level keys are strings (JSON requirement) and are cast to int internally.
    ``fit()`` populates ``self.scalers`` without touching the data — all stats come from the dict.
    """
    def __init__(self, stats):
        if isinstance(stats, str):
            import json
            with open(stats) as f:
                stats = json.load(f)
        self.stats = stats
        self.compute = False # no computation needed since stats are provided, but keeping the attribute for compatibility with the XarrayScaler interface
        self.scalers = {}
        self.scaler_name = "HandcraftedInputsScaler"

    def fit(self, inputs: xr.DataArray):
        variable_names = inputs.variable_name.values
        self.fitted_variable_names = list(variable_names)

        for varname, var_stats in self.stats.items():
            scaler_type = var_stats.get("type", "standard")
            level_entries = {k: v for k, v in var_stats.items() if k != "type"}

            if scaler_type == "minmax":
                s = next(iter(level_entries.values()))
                scaler = XarrayMinMaxScaler(manual_min=s["min"], manual_max=s["max"], compute=self.compute)
                for vc in variable_names:
                    if vc[0] == varname and len(vc) == 3:
                        self.scalers[vc] = scaler
            else:
                for level_str, s in level_entries.items():
                    level = int(level_str)
                    scaler = XarrayScaler(compute=self.compute, mean=s["mean"], std=s["std"])
                    for vc in variable_names:
                        if vc[0] == varname and vc[1] == level and len(vc) == 3:
                            self.scalers[vc] = scaler

        return self

    def transform(self, inputs: xr.DataArray) -> xr.DataArray:
        variable_names = inputs.variable_name.values
        for varname in variable_names:
            if varname not in self.fitted_variable_names:
                raise ValueError(f"Variable name {varname} in inputs is not in the variable names that were fitted on: {self.fitted_variable_names}.")

        transformed_variables = []
        for varname in np.unique(variable_names):
            var_data = inputs.sel(variable_name=varname)
            scaler = self.scalers.get(varname, None)
            if scaler is None:
                print(f"No scaler found for variable {varname}, skipping transformation.")
                transformed_variables.append(var_data)
                continue
            transformed_data = scaler.transform(var_data)
            transformed_data = transformed_data.rename(varname)
            transformed_variables.append(transformed_data)

        transformed = xr.concat(transformed_variables, dim="variable_name")
        mindex = pd.MultiIndex.from_tuples(transformed.variable_name.values, names=["variable", "levels", "time_delta"])
        mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "variable_name")
        transformed = transformed.assign_coords(mindex_coords)
        transformed = transformed.rename("stacked_transformed_inputs")
        transformed.attrs = {"transformer": "HandcraftedInputsScaler", "generated_on": str(datetime.datetime.now())}
        transformed = transformed.transpose("fp_time", "lat", "lon", "variable_name")
        return transformed

    def fit_transform(self, inputs: xr.DataArray) -> xr.DataArray:
        self.fit(inputs)
        return self.transform(inputs)

####
####
####
####

### Transforming and scaling data - outputs - footprints

class LogAndShiftFpScaler:
    """
    Footprint scaler.
    Takes log of fp data where non-zero, and offsets by minimum order-of-magnitude value so its above zero. Does not need fitting.
    Note: translates across domain sizes
    """
    def __init__(self, minimum_oom=5, non_negative=True):
        self.minimum_oom = minimum_oom
        self.non_negative = non_negative
        self.scaler_name = "LogAndShiftFpScaler"

    def fit(self, fp):
        self.params = {"minimum_oom": self.minimum_oom, "non_negative": self.non_negative}
        pass 
    
    def transform(self, fp):
        #fp = fp.astype("float32", copy=False)
        transformed_fp = np.log10(fp.where(fp > 0)) + self.minimum_oom  # take the log and add the minimum_oom), leave the zeros as is
        if self.non_negative:
            transformed_fp = transformed_fp.where(transformed_fp > 0, 0) # make all values that are zero or below zero (which can happen if the original fp was between 0 and 10**(-minimum_oom)) zero, to avoid having negative values in the transformed fp
        
        return transformed_fp
    
    def inverse_transform(self, transformed_fp):

        original_fp = 10**(transformed_fp - self.minimum_oom)
        
        #original_fp = transformed_fp.where(transformed_fp <= self.minimum_oom, original_fp)
        # make all negative values zero
        if isinstance(original_fp, xr.DataArray):
            original_fp = original_fp.where(original_fp >= 0, 0)
            original_fp = original_fp.where(original_fp > 10**-int(self.minimum_oom), 0)
        elif isinstance(original_fp, np.ndarray):
            original_fp = np.where(original_fp >= 0, original_fp, 0)
            original_fp = np.where(original_fp > 10**-int(self.minimum_oom), original_fp, 0)
        return original_fp

    def get_params(self):
        if not hasattr(self, "params"):
            raise ValueError("Scaler has not been fitted yet, so parameters are not available. Please fit the scaler before trying to get parameters.")
        return self.params
    
class LogAndShiftMeanFpScaler:
    """
    Footprint scaler.
    Takes the mean of the log of fp data where non-zero, and offsets the log of the data by the mean, so that the mean of the logged data is at zero. 
    Note: translates across domain sizes
    """
    def __init__(self, minimum_oom=5):
        self.minimum_oom = minimum_oom
        self.scaler_name = "LogAndShiftMeanFpScaler"
        
    def fit(self, fp):
        self.logged_mean = np.mean(np.log10(fp.values.flatten()[fp.values.flatten()>0])).astype("float32")
        self.params = {"logged_mean": self.logged_mean, "minimum_oom": self.minimum_oom}
    
    def transform(self, fp):
        #fp = fp.astype("float32", copy=False)
        transformed_fp = np.log10(fp+10**(-self.minimum_oom))  # take the log and shift by the logged mean
        transformed_fp = transformed_fp + abs(self.logged_mean)

        return transformed_fp
    
    def inverse_transform(self, transformed_fp):
        original_fp = 10**(transformed_fp - abs(self.logged_mean)) - 10**(-self.minimum_oom)
        if isinstance(original_fp, xr.DataArray):
            original_fp = original_fp.where(original_fp >= 0, 0)
        elif isinstance(original_fp, np.ndarray):
            original_fp = np.where(original_fp >= 0, original_fp, 0)
        return original_fp
    
    def get_params(self):
        if not hasattr(self, "params"):
            raise ValueError("Scaler has not been fitted yet, so parameters are not available. Please fit the scaler before trying to get parameters.")
        return self.params


def add_fp_nan_mask(ds, fill_nans=False, fp_var_name="fp", nan_mask_name="fp_nan_mask"):
    if nan_mask_name not in ds:
        ds[nan_mask_name] = np.isnan(ds[fp_var_name]).astype("int32")
    if fill_nans:
        ds = ds.fillna(0)
    return ds

class FootprintDataset:
    """
    Wrapper around footprint data that applies a footprint scaler.

    The dataset accepts either an ``xarray.DataArray`` or an ``xarray.Dataset`` containing
    a variable named ``fp``. ``fit`` fits the underlying scaler, ``transform`` returns a
    Dataset containing ``fp_transformed`` and ``fp_original``, and ``inverse_transform``
    accepts either the Dataset returned by ``transform`` or the transformed DataArray.
    """

    def __init__(self, fp, scaler=None, scaler_params={}, add_nan_mask=False, verbose=False):
        
        fp = self._check_fp_format(fp)
        
        self.fp = fp.copy()
        self.add_nan_mask = add_nan_mask
        self.verbose = verbose

        if scaler is None:
            self.scaler = LogAndShiftMeanFpScaler(**scaler_params)
        else:
            self.scaler = scaler(**scaler_params)

        #print("done")
        #self.transformed_fp = self.transform(self.fp_dataset.fp)

        self.filename = "fp_scaler.joblib"

    def _check_fp_format(self, fp):
        if isinstance(fp, xr.Dataset):
            if "fp" not in fp.data_vars:
                raise ValueError("fp must be an xarray DataSet with a variable named 'fp', or an xarray DataArray")
            else:
                fp = fp["fp"]
        elif not isinstance(fp, xr.DataArray):
            raise ValueError("fp must be an xarray DataSet with a variable named 'fp', or an xarray DataArray")
        return fp
    
    def fit(self):
        self.scaler.fit(self.fp)

    def transform(self, fp):
        fp = self._check_fp_format(fp)
        
        ## check this
        transformed_fp = self.scaler.transform(fp)

        if self.verbose: 
            if hasattr(self.scaler, "scaler_name"):
                print(f"Transformed footprints using {self.scaler.scaler_name} scaler.")
            if hasattr(self.scaler, "params"):
                print(f"Using scaler parameters: {self.scaler.params}")

        transformed_fp = transformed_fp.rename("transformed_fp")
        transformed_fp.attrs = {"transformer": self.scaler.__class__.__name__, "generated_on": str(datetime.datetime.now())}
        ds = xr.Dataset({
            "fp_transformed": transformed_fp,
            "fp_original": fp,
        })

        if self.add_nan_mask:
            ds = add_fp_nan_mask(ds, fill_nans=True, fp_var_name="fp_original", nan_mask_name="fp_nan_mask")
            if self.verbose: print("Added fp_nan_mask to the dataset, which indicates where the original fp had NaN values. The transformed_fp has been filled with zeros where the original fp had NaN values.")


        # add coord linked to time index with idx
        ds = ds.chunk({"time": 1})
        ds = ds.assign_coords(idx=("time", list(range(len(ds.time)))))
        
        self.transformed_fp = transformed_fp

        return ds

    def fit_transform(self):
        self.fit()
        return self.transform(self.fp)

    def inverse_transform(self, transformed_fp):
        if isinstance(transformed_fp, xr.Dataset) and "fp_transformed" in transformed_fp.data_vars:
            transformed_fp = transformed_fp["fp_transformed"]
        elif not isinstance(transformed_fp, xr.DataArray) and not isinstance(transformed_fp, np.ndarray):
            raise ValueError("transformed fp must be an xarray DataArray or numpy array!")        
        return self.scaler.inverse_transform(transformed_fp)

    def save_scaler(self, path):
        print("warning this is untested")
        path = Path(path) / self.filename
        #path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "scaler": self.scaler,
            "scaler_type": self.scaler.scaler_name,
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        joblib.dump(bundle, path)

    def load_scaler(self, path):
        print("warning this is untested")
        bundle = joblib.load(Path(path) / self.filename)
        #print(f"Loaded {bundle['scaler_type']} saved at {bundle['saved_at']}")
        self.scaler = bundle["scaler"]


def make_inputs_batcher(inputs, batch_size=10, flatten=False):
    """
    Build an xbatcher BatchGenerator for the input meteorological fields.

    Inputs:
    - inputs: xarray DataArray of size (fp_time, lat, lon, variable_name)
    - batch_size: int, batch size
    - flatten: bool, whether to stack lat/lon into a single flat_lat_lon dimension

    Returns:
    - X_bgen: xbatcher BatchGenerator
    """
    if not isinstance(inputs, xr.DataArray):
        raise ValueError("inputs must be an xarray DataArray with dims (fp_time, lat, lon, variable_name)")
    if not all(dim in inputs.dims for dim in ["fp_time", "lat", "lon", "variable_name"]):
        raise ValueError("inputs must have dimensions (fp_time, lat, lon, variable_name). Use the get_square_satellite_inputs function to extract inputs in the correct format.")
    if inputs.dtype != "float32":
        inputs = inputs.astype("float32", copy=False)

    inputs = inputs.chunk(fp_time=batch_size)
    inputs = inputs.transpose("fp_time", "lat", "lon", "variable_name")

    if flatten:
        inputs = inputs.stack(flat_lat_lon=["lat", "lon"])
        inputs = inputs.transpose("fp_time", "flat_lat_lon", "variable_name")
        input_dims = {"flat_lat_lon": len(inputs.flat_lat_lon), "variable_name": len(inputs.variable_name)}
    else:
        input_dims = {"lat": len(inputs.lat), "lon": len(inputs.lon), "variable_name": len(inputs.variable_name)}

    X_bgen = xb.BatchGenerator(
        inputs,
        input_dims=input_dims,
        batch_dims={'fp_time': batch_size},
        preload_batch=True,
    )
    return X_bgen

def make_boundary_batcher(outputs, batch_size=5):
    """
    Build an xbatcher BatchGenerator for boundary condition outputs.

    Unlike footprint outputs which have spatial dimensions (time, lat, lon),
    boundary condition outputs are 1D per sample with shape (time, num_classes).

    Inputs:
    - outputs: xarray DataArray of size (time, num_classes)
    - batch_size: int, batch size

    Returns:
    - y_bgen: xbatcher BatchGenerator
    - output_labels: list of output label names, one per class
    """
    if not isinstance(outputs, xr.DataArray):
        raise ValueError(
            f"outputs must be an xarray DataArray of shape (time, num_classes), "
            f"got {type(outputs).__name__}"
        )

    if "time" not in outputs.dims:
        raise ValueError("outputs must have a 'time' dimension")

    if "lat" in outputs.dims or "lon" in outputs.dims:
        raise ValueError(
            "outputs should not have spatial dimensions (lat/lon). "
            "For footprint outputs use make_fps_batcher instead."
        )

    # Find the output dimension (everything that is not time)
    output_dims = [d for d in outputs.dims if d != "time"]
    if len(output_dims) != 1:
        raise ValueError(
            f"outputs should have exactly 2 dimensions (time, num_classes), "
            f"got dims: {outputs.dims}"
        )
    output_dim = output_dims[0]

    # Extract labels from the coordinate values if they are strings,
    # otherwise generate default names
    # coord_vals = outputs[output_dim].values
    # if coord_vals.dtype == object or np.issubdtype(coord_vals.dtype, np.str_):
    #     output_labels = list(coord_vals)
    # else:
    #     output_labels = [f"output_{i}" for i in range(outputs.sizes[output_dim])]

    if outputs.dtype != "float32":
        outputs = outputs.astype("float32", copy=False)

    outputs = outputs.chunk(time=batch_size)
    outputs = outputs.transpose("time", output_dim)

    y_bgen = xb.BatchGenerator(
        outputs,
        input_dims={output_dim: outputs.sizes[output_dim]},
        batch_dims={"time": batch_size},
        preload_batch=True,
    )

    return y_bgen #, output_labels

def make_fps_batcher(fps, batch_size=10, flatten=False):
    """
    Build an xbatcher BatchGenerator for the footprint labels.

    Inputs:
    - fps: xarray DataArray of size (time, lat, lon), or xarray Dataset containing
      one or more footprint variables of size (time, lat, lon)
    - batch_size: int, batch size
    - flatten: bool, whether to stack lat/lon into a single flat_lat_lon dimension

    Returns:
    - y_bgen: xbatcher BatchGenerator
    - fps_labels: list of footprint variable name(s)
    """
    # stack all variables in the fps along a new variable dimension, so that the dataloader returns all variables in the fps dataset. If fps is already an xarray DataArray, this will just add a variable dimension of size 1.
    if isinstance(fps, xr.Dataset):
        fps_labels = list(fps.data_vars)
        for var in fps:
            if fps[var].dtype != "float32" and fps[var].dtype != "int32":
                fps[var] = fps[var].astype("float32", copy=False)

        fps = fps.to_stacked_array(new_dim="variable_name", sample_dims=["time", "lat", "lon"], name="stacked_fps")
        fps = fps.transpose("time", "lat", "lon", "variable_name")
        fps = fps.chunk(time=batch_size, variable_name=-1)
    
        if flatten:
            fps = fps.stack(flat_lat_lon=["lat", "lon"])
            fps = fps.transpose("time", "flat_lat_lon", "variable_name")
            input_dims = {"flat_lat_lon": len(fps.flat_lat_lon), "variable_name": len(fps.variable_name)}
        else:
            input_dims = {"lat": len(fps.lat), "lon": len(fps.lon), "variable_name": len(fps.variable_name)}


    elif isinstance(fps, xr.DataArray):
        fps_labels = [fps.name if fps.name is not None else "fp"]
        fps = fps.chunk(time=batch_size)
        if fps.dtype != "float32":
            fps = fps.astype("float32", copy=False)

        if flatten:
            fps = fps.stack(flat_lat_lon=["lat", "lon"])
            input_dims = {"flat_lat_lon": len(fps.flat_lat_lon)}
        else:
            input_dims = {"lat": len(fps.lat), "lon": len(fps.lon)}
    else:
            raise ValueError(
                f"Unsupported fps type: expected xr.Dataset or xr.DataArray, got {type(fps).__name__}"
            )    
    y_bgen = xb.BatchGenerator(
        fps,
        input_dims=input_dims,
        batch_dims={'time': batch_size},
        preload_batch=False,
    )
    return y_bgen, fps_labels




def trim_to_batch_size(inputs, fps, batch_size):
    """
    Trim the last N timepoints from inputs and fps so that the number of
    timepoints is divisible by batch_size.

    Inputs:
    - inputs: xarray DataArray with a 'fp_time' dimension
    - fps: xarray DataArray or Dataset with a 'time' dimension
    - batch_size: int

    Returns:
    - inputs, fps trimmed along their respective time dimensions
    """
    n = inputs.sizes["fp_time"]
    remainder = n % batch_size
    if remainder != 0:
        inputs = inputs.isel(fp_time=slice(None, n - remainder))
        fps = fps.isel(time=slice(None, n - remainder))
    return inputs, fps


def make_dataloader(inputs, fps, batch_size=10, randomize=False, random_seed=42, dataloader_params=None, flatten=False):
    """
    Build a PyTorch dataloader from input and footprint datasets using xbatcher.

    The inputs must have dimensions (fp_time, lat, lon, variable_name). The footprints
    must have matching ``time`` length and either be a DataArray with dims ``(time, lat, lon)``
    or a Dataset containing one or more footprint variables. When a Dataset is passed, the
    footprints are stacked along a new ``variable_name`` dimension and the returned labels are
    the stacked variable tuples; when a DataArray is passed, the returned labels contain the single footprint variable name.

    If ``randomize`` is True, the time dimension is permuted before batching. ``random_seed``
    controls the permutation. ``dataloader_params`` is forwarded to ``torch.utils.data.DataLoader``.

    Inputs:
    - inputs: xarray DataArray of size (fp_time, lat, lon, variable_name) from get_square_satellite_inputs or similar function
    - fps: xarray datarray, or xarray Dataset with variable "fp_transformed" of size (time, lat, lon)
    - batch_size: int, batch size for the dataloader
    - randomize: bool, whether to shuffle the dataset along the time dimension. Recommended for training and not for testing
    - random_seed: int, seed for reproducibility of the shuffling when randomize is True
    - dataloader_params: dict, additional parameters to pass to the PyTorch DataLoader
    - flatten: bool, whether to flatten the input tensors in the lat-lon dimension before passing them to the model

    Returns:
        - dataloader: PyTorch DataLoader that yields batches of ``(inputs, fps)``.
            Inputs has shape (batch_size, lat, lon, variable_name) or (batch_size, flat_lat_lon, variable_name) if flatten is True.
            If passing a dataarray as fps, Fps has shape (batch_size, lat, lon) or (batch_size, flat_lat_lon) if flatten is True. If passing a dataset as fps, fps has shape (batch_size, lat, lon, variable_name) or (batch_size, flat_lat_lon, variable_name) if flatten is True.
        - fps_labels: footprint variable labels, as a list

    """
    # ensure that both have the right dimensions
    if inputs.sizes["fp_time"] != fps.sizes["time"]:
        raise ValueError("Incompatible dimensions between inputs and fps")
    if isinstance(fps, xr.Dataset):
        print(f"you passed a fps dataset with multiple variables: {list(fps.data_vars)}. All variables will be returned in the dataloader along a new dimension. ")
        ## but make sure this is what you want! If you only want to return the transformed fps variable, pass fps['fp_transformed'] instead of the whole dataset when calling this function.
    if not isinstance(fps, (xr.DataArray, xr.Dataset)) or not isinstance(inputs, xr.DataArray):
        print(f"inputs type: {type(inputs)}, fps type: {type(fps)}")
        raise ValueError("fps must be an xarray DataArray or Dataset, and inputs must be an xarray DataArray")

    if randomize:
        # set the random seed for reproducibility
        np.random.seed(random_seed)
        # shuffle the data by permuting the fp_time dimension
        permuted_time = np.random.permutation(inputs.fp_time)
        inputs = inputs.sel(fp_time=permuted_time)
        fps = fps.sel(time=permuted_time)

    if len(inputs.lat) != len(fps.lat) or len(inputs.lon) != len(fps.lon):
         raise ValueError(
             "Latitude and longitude dimensions of inputs and fps must match: "
             f"inputs(lat={len(inputs.lat)}, lon={len(inputs.lon)}), "
             f"fps(lat={len(fps.lat)}, lon={len(fps.lon)})"
         )


    X_bgen = make_inputs_batcher(inputs, batch_size=batch_size, flatten=flatten)
    y_bgen, fps_labels = make_fps_batcher(fps, batch_size=batch_size, flatten=flatten)

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)
    if dataloader_params is None:
        dataloader_params = {
            "prefetch_factor": 3,  # Prefetch up to 3 batches in advance to reduce data loading latency
            "num_workers": 4,  # Use 4 parallel worker processes to load data concurrently
            "persistent_workers": True,  # Keep workers alive between epochs for faster subsequent epochs
            "multiprocessing_context": 'forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
        }
    #print("dataloader params = ", dataloader_params)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        **dataloader_params
    )

    
    return dataloader, fps_labels

def make_boundary_dataloader(inputs, outputs, batch_size=10, randomize=False,
                              random_seed=42, dataloader_params=None, flatten=False):
    """
    Build a PyTorch DataLoader for boundary condition prediction using xbatcher.

    Inputs must have dimensions (fp_time, lat, lon, variable_name).
    Outputs must be an xarray DataArray of shape (time, num_classes).

    Args:
        inputs (xr.DataArray): Shape (fp_time, lat, lon, variable_name).
        outputs (xr.DataArray): Shape (time, num_classes).
        batch_size (int): Batch size.
        randomize (bool): Whether to shuffle along the time dimension.
        random_seed (int): Seed for reproducibility when randomize=True.
        dataloader_params (dict, optional): Additional kwargs for DataLoader.
        flatten (bool): Whether to flatten lat/lon in the inputs.

    Returns:
        torch.utils.data.DataLoader: yields (inputs_batch, outputs_batch) tensors.
    """
    if inputs.sizes["fp_time"] != outputs.sizes["time"]:
        raise ValueError(
            f"inputs and outputs have mismatched time dimensions: "
            f"inputs fp_time={inputs.sizes['fp_time']}, outputs time={outputs.sizes['time']}"
        )

    if randomize:
        np.random.seed(random_seed)
        permuted_time = np.random.permutation(inputs.fp_time.values)
        inputs = inputs.sel(fp_time=permuted_time)
        outputs = outputs.sel(time=permuted_time)

    X_bgen = make_inputs_batcher(inputs, batch_size=batch_size, flatten=flatten)

    y_bgen = make_boundary_batcher(outputs, batch_size=batch_size)

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)

    if dataloader_params is None:
        dataloader_params = {
            "prefetch_factor": 3,
            "num_workers": 4,
            "persistent_workers": True,
            "multiprocessing_context": "forkserver",
        }

    # prefetch_factor is only valid when num_workers > 0 — remove it otherwise
    # to avoid a ValueError from PyTorch
    dataloader_params = dataloader_params.copy()
    if dataloader_params.get("num_workers", 0) == 0:
        dataloader_params.pop("prefetch_factor", None)
        dataloader_params.pop("persistent_workers", None)
        dataloader_params.pop("multiprocessing_context", None)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        **dataloader_params
    )

    return dataloader


class _DualMapDataset(torch.utils.data.Dataset):
    """xbatcher-backed dataset yielding ``(inputs, fps, boundary)`` triplets.

    Wraps three ``xbatcher.BatchGenerator`` objects that share the same batching
    along the time dimension, so index ``i`` returns the i-th batch of the inputs,
    the footprint labels and the boundary/background labels respectively. Tensor
    conversion uses xbatcher's own ``to_tensor`` transform so the output matches
    ``make_dataloader`` / ``make_boundary_dataloader`` exactly.
    """

    def __init__(self, X_bgen, fp_bgen, bg_bgen):
        from xbatcher.loaders.torch import to_tensor
        self._to_tensor = to_tensor
        self.X_bgen = X_bgen
        self.fp_bgen = fp_bgen
        self.bg_bgen = bg_bgen

    def __len__(self):
        return len(self.X_bgen)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_batch = self.X_bgen[idx]
        fp_batch = self.fp_bgen[idx]
        bg_batch = self.bg_bgen[idx]
        try:
            import dask
            x_batch, fp_batch, bg_batch = dask.compute(x_batch, fp_batch, bg_batch)
        except ImportError:
            pass
        return (
            self._to_tensor(x_batch),
            self._to_tensor(fp_batch),
            self._to_tensor(bg_batch),
        )


def make_dual_dataloader(inputs, fps, outputs, batch_size=10, randomize=False,
                          random_seed=42, dataloader_params=None, flatten=False):
    """
    Build a PyTorch DataLoader yielding ``(inputs, fps, boundary)`` for the dual-head model.

    Combines the targets of ``make_dataloader`` (per-node footprints) and
    ``make_boundary_dataloader`` (whole-grid background) so a single batch carries
    everything both decoder heads need. Inputs, footprints and boundary outputs must
    all be aligned along the time dimension (use ``load_GATES_data_with_bg`` which
    returns them aligned).

    Args:
        inputs (xr.DataArray): Shape (fp_time, lat, lon, variable_name).
        fps (xr.DataArray or xr.Dataset): Footprint labels, (time, lat, lon[, variable_name]).
        outputs (xr.DataArray): Background targets, shape (time, num_classes).
        batch_size (int): Batch size.
        randomize (bool): Shuffle along the time dimension (recommended for training only).
        random_seed (int): Seed for the shuffle when randomize=True.
        dataloader_params (dict, optional): Extra kwargs for the DataLoader.
        flatten (bool): Flatten lat/lon in the inputs/footprints.

    Returns:
        dataloader (torch.utils.data.DataLoader): yields ``(inputs, fps, boundary)`` tensors.
        fps_labels (list): footprint variable label(s), matching make_dataloader.
    """
    if inputs.sizes["fp_time"] != fps.sizes["time"]:
        raise ValueError("Incompatible time dimensions between inputs and fps")
    if inputs.sizes["fp_time"] != outputs.sizes["time"]:
        raise ValueError("Incompatible time dimensions between inputs and boundary outputs")

    if randomize:
        np.random.seed(random_seed)
        permuted_time = np.random.permutation(inputs.fp_time.values)
        inputs = inputs.sel(fp_time=permuted_time)
        fps = fps.sel(time=permuted_time)
        outputs = outputs.sel(time=permuted_time)

    X_bgen = make_inputs_batcher(inputs, batch_size=batch_size, flatten=flatten)
    fp_bgen, fps_labels = make_fps_batcher(fps, batch_size=batch_size, flatten=flatten)
    bg_bgen = make_boundary_batcher(outputs, batch_size=batch_size)

    dataset = _DualMapDataset(X_bgen, fp_bgen, bg_bgen)

    if dataloader_params is None:
        dataloader_params = {
            "prefetch_factor": 3,
            "num_workers": 4,
            "persistent_workers": True,
            "multiprocessing_context": "forkserver",
        }

    dataloader_params = dataloader_params.copy()
    if dataloader_params.get("num_workers", 0) == 0:
        dataloader_params.pop("prefetch_factor", None)
        dataloader_params.pop("persistent_workers", None)
        dataloader_params.pop("multiprocessing_context", None)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        **dataloader_params
    )

    return dataloader, fps_labels



####
####  v2: single-interpolation-pass functions
####

def _cut_satellite_met_multi_delta(
    met_source,
    fp,
    metsize,
    time_deltas,
    pad_mode="nans",
    add_wind_direction=True,
    closest_tolerance="4h",
    verbose=False,
    load_into_memory=False,
    interp_to=None,
):
    """
    Like cut_satellite_met but handles all time_deltas in one call.

    Computes the union of required met timestamps across all time_deltas,
    loads that minimal set from dask once, computes the spatial crop structure
    once, then builds per-delta cropped datasets entirely from in-memory arrays.

    Parameters
    ----------
    met_source : xr.Dataset
        Full-domain met, already filtered to desired levels and variables.
    fp : xr.Dataset
        Footprint dataset with ``release_lat``, ``release_lon``, ``time``.
    metsize : int
        Size of the square to crop (must be even).
    time_deltas : list[int]
        Hours to shift backwards (0 = no shift).
    pad_mode : str
        "nans" or "edge" — passed to ``_pad_domain``.
    add_wind_direction : bool
        Compute ``wind_angle`` and ``wind_speed`` from ``x_wind``/``y_wind``.
    closest_tolerance : str
        Max allowed distance for nearest-timestamp lookup, e.g. "4h".
    verbose : bool
    interp_to : str or None
        If None (default), each fp timestamp is matched to the nearest met
        timestamp (existing behaviour).  If a pandas offset string such as
        ``"1h"`` or ``"15min"``, each target time is rounded to that resolution
        and the met is linearly interpolated to that rounded time from the two
        bracketing met timestamps.  fp times whose rounded target falls outside
        the met record are treated as NaN and dropped, same as the nearest
        case.

    Returns
    -------
    results : dict[int, xr.Dataset]
        {delta: cropped dataset} with dims ``(fp_time, lat, lon[, levels])``.
        No ``time_delta`` dimension — caller handles the concat.
        Contains ``lat_coords`` and ``lon_coords`` (same values across deltas).
    nan_idxs_per_delta : dict[int, np.ndarray]
        fp_time values that could not be interpolated for each delta.
    """
    if metsize % 2 != 0:
        raise ValueError("metsize must be even")
    half = metsize // 2

    fp_times = fp.time.values
    tol = pd.Timedelta(closest_tolerance)
    met_time_index = met_source.indexes["time"]

    # --- Phase 1: timestamp lookup for every delta ---
    nearest_info = {}
    all_unique_times = set()

    for delta in time_deltas:
        target_times = pd.DatetimeIndex(
            fp_times if delta == 0
            else pd.DatetimeIndex(fp_times) - pd.Timedelta(f"{delta}h")
        )

        if interp_to is None:
            # choose the nearest met timestamp for each target time, within the specified tolerance
            nearest = met_time_index.get_indexer(target_times, method="nearest", tolerance=tol)
            nan_mask = nearest == -1
            nearest_safe = np.where(~nan_mask, nearest, 0)
            nearest_timestamps = met_time_index.values[nearest_safe]
            nan_idxs = (target_times[nan_mask] + pd.Timedelta(f"{delta}h")).to_numpy()

            nearest_info[delta] = {
                "lookup_times": nearest_timestamps,
                "nan_mask": nan_mask,
                "nan_idxs": nan_idxs,
            }
            all_unique_times.update(nearest_timestamps[~nan_mask])

        else:
            # If required, calculate the target times rounded to the specified resolution to interpolate to later, and find the nearest met timestamps that bracket those targets on either side (floor and ceil).
            interp_targets = target_times.round(interp_to)
            floor_idxs = met_time_index.get_indexer(interp_targets, method="ffill")
            ceil_idxs = met_time_index.get_indexer(interp_targets, method="bfill")
            nan_mask = (floor_idxs == -1) | (ceil_idxs == -1)
            nan_idxs = (target_times[nan_mask] + pd.Timedelta(f"{delta}h")).to_numpy()

            interp_targets_arr = interp_targets.to_numpy()
            valid_targets = interp_targets_arr[~nan_mask]
            safe_targets = np.where(
                ~nan_mask,
                interp_targets_arr,
                valid_targets[0] if len(valid_targets) > 0 else interp_targets_arr,
            )

            nearest_info[delta] = {
                "lookup_times": safe_targets,
                "nan_mask": nan_mask,
                "nan_idxs": nan_idxs,
            }
            all_unique_times.update(met_time_index.values[floor_idxs[~nan_mask]])
            all_unique_times.update(met_time_index.values[ceil_idxs[~nan_mask]])

    # --- Phase 2: select the union of required timestamps ---
    all_unique_times_sorted = sorted(all_unique_times)
    size_chunks = len(all_unique_times_sorted) // 10
    size_chunks = max(size_chunks, 1)
    size_chunks = min(100, size_chunks)

    if verbose:
        print(f"Selecting {len(all_unique_times_sorted)} unique met timestamps ")
        print(f"First few unique timestamps: {all_unique_times_sorted[:3]} ...")

    chunk_kw = {"time": size_chunks, "lat": -1, "lon": -1}
    if "levels" in met_source.dims:
        chunk_kw["levels"] = -1

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        met_loaded = met_source.sel(time=list(all_unique_times_sorted))
    if load_into_memory:
        print("Loading selected met data into memory...")
        met_loaded = met_loaded.compute()

    # --- Phase 3: spatial structure — computed once, shared across all deltas ---
    release_idxs = _get_release_idxs(fp, domain_lats=met_source.lat.values, domain_lons=met_source.lon.values)
    met_loaded, _, _, _ = _pad_domain(
        met_loaded, fp, release_idxs, half, pad_mode, verbose=verbose
    )

    lat_ds = fp.lat_coords
    lon_ds = fp.lon_coords

    # --- Phase 3c (interp_to only): linearly interpolate to rounded target times ---
    if interp_to is not None:
        all_interp_targets = sorted({
            t for info in nearest_info.values()
            for t, bad in zip(info["lookup_times"], info["nan_mask"])
            if not bad
        })
        if verbose:
            print(f"Interpolating met to {len(all_interp_targets)} unique '{interp_to}' targets...")

        met_for_crop = met_loaded.interp(
            time=np.array(all_interp_targets, dtype="datetime64[ns]"),
            method="linear",
        )
        if load_into_memory:
            met_for_crop = met_for_crop.compute()
    else:
        met_for_crop = met_loaded

    # Lookup: timestamp → integer position in met_for_crop
    time_to_pos = {t: i for i, t in enumerate(met_for_crop.time.values)}

    # --- Phase 4: per-delta crop from fully in-memory data ---
    results = {}
    nan_idxs_per_delta = {}

    for delta, info in nearest_info.items():
        nan_idxs_per_delta[delta] = info["nan_idxs"]

        pos = np.array([
            time_to_pos.get(t, 0)
            for t in info["lookup_times"]
        ])

        met_delta = met_for_crop.isel(time=pos)
        met_delta = met_delta.assign_coords(time=fp_times)

        # Spatial crop using 2D lat/lon DataArrays from the footprint
        cropped = met_delta.sel(lat=lat_ds, lon=lon_ds, method="nearest")

        # After sel with 2D DataArray indexers, lat and lon become 2D variables
        # of shape (time, lat) and (time, lon), which conflicts with assign_coords
        # in older xarray versions. Save the real values, drop the 2D vars,
        # then reassign integer coordinates and store real coords separately.
        real_lats = cropped.lat.values
        real_lons = cropped.lon.values

        cropped = cropped.drop_vars(["lat", "lon"], errors="ignore")
        cropped = cropped.assign_coords(
            lat=np.arange(metsize),
            lon=np.arange(metsize)
        )
        cropped["lat_coords"] = xr.DataArray(
            real_lats, dims=["time", "lat"],
            coords={"time": cropped.time}
        )
        cropped["lon_coords"] = xr.DataArray(
            real_lons, dims=["time", "lon"],
            coords={"time": cropped.time}
        )
        cropped = cropped.set_coords(["lat_coords", "lon_coords"])

        if add_wind_direction:
            try:
                if "x_wind" in cropped.data_vars and "y_wind" in cropped.data_vars:
                    cropped["wind_angle"] = np.arctan2(-cropped.x_wind, -cropped.y_wind)
                    cropped["wind_speed"] = np.sqrt(cropped.x_wind**2 + cropped.y_wind**2)
            except Exception as e:
                warnings.warn(f"Could not compute wind variables for delta={delta}: {e}")

        # Swap time → fp_time to match convention in rest of pipeline
        cropped = cropped.assign({"fp_time": ("time", fp_times)})
        cropped = cropped.swap_dims({"time": "fp_time"}).drop_vars("time", errors="ignore")

        results[delta] = cropped

    return results, nan_idxs_per_delta


def get_square_satellite_inputs_v2(
    data,
    met_variables,
    met_levels,
    time_deltas=None,
    static_variables=None,
    verbose=True,
    add_timedelta_zero=True,
    add_wind_direction=False,
    load_into_memory=False,
    interp_to=None,
):
    """
    Optimised version of ``get_square_satellite_inputs``.

    Differences from v1:
    - ``met_variables`` is a plain list of variable names (not a dict).
    - ``met_levels`` is a single list of levels shared by all atmospheric variables.
      Surface variables (those without a "levels" dimension in the met dataset)
      are detected automatically.
    - Met levels and variables are filtered once; all time_deltas share a single
      dask ``compute()`` call, eliminating repeated graph construction and I/O.
    - Spatial structure (release indices, domain padding, crop index arrays) is
      computed once and reused across all time_deltas.

    Parameters
    ----------
    data : LoadSquareSatelliteData
        Must have ``met_processed=True`` and ``dataset_format='square'``.
    met_variables : list[str]
        Variable names to extract, e.g. ``["x_wind", "y_wind", "atmosphere_boundary_layer_thickness"]``.
        Include ``"wind_angle"`` / ``"wind_speed"`` to derive them from x/y wind.
    met_levels : list[int]
        Pressure/model levels to extract for all atmospheric variables.
    time_deltas : list[int] or None
        Hours to shift backwards, e.g. ``[6, 12]``. 0 is added automatically
        unless ``add_timedelta_zero=False``.
    static_variables : list[str] or None
        Static fields to append, e.g. ``["topog", "lat_coords", "lon_coords"]``.
    verbose : bool
    add_timedelta_zero : bool
    interp_to : str or None
        If None (default), each fp timestamp is matched to the nearest met
        timestamp.  If a pandas offset string such as ``"1h"`` or ``"15min"``,
        each target time is rounded to that resolution and the met is linearly
        interpolated between the two bracketing met timestamps.

    Returns
    -------
    concatenated_inputs : xr.DataArray
        Shape ``(fp_time, lat, lon, variable_name)`` where ``variable_name`` is a
        MultiIndex of ``(variable, level, time_delta)`` tuples.
    data : LoadSquareSatelliteData
        Updated object — fp_time indices with failed met interpolation are removed.
    """
    assert hasattr(data, "dataset_format"), "Not a SatelliteData object"
    #assert data.met_processed is True, "Load and cut the meteorology first"
    assert data.dataset_format == "square", (
        "get_square_satellite_inputs_v2 only supports dataset_format='square'"
    )

    if not isinstance(met_variables, list):
        raise ValueError("met_variables should be a list of variable names")
    if not isinstance(met_levels, list):
        raise ValueError("met_levels should be a list of level integers")

    if time_deltas is None:
        time_deltas = []
    else:
        time_deltas = list(time_deltas)

    if static_variables is None:
        static_variables = []
    else:
        static_variables = list(static_variables)

    if add_timedelta_zero and 0 not in time_deltas:
        time_deltas.append(0)
    time_deltas = sorted(set(time_deltas))

    if verbose:
        print("------------------------")
        print("---EXTRACTING MET DATA (v2)---")
        print(f"Time deltas: {time_deltas}")
        print(f"Met variables: {met_variables}")
        print(f"Met levels: {met_levels}")

    # --- Filter source met once ---
    met_source = select_met_levels(data.met_file, levels=met_levels.copy())
    met_source = select_met_variables(met_source, variables=met_variables.copy())
    first_var = list(met_source.data_vars)[0]
    if met_source[first_var].dtype != "float32":
        if verbose:
            print(f"Casting met to float32 from {met_source[first_var].dtype}")
        met_source = met_source.astype("float32")

    pad_mode = "edge"

    # --- Single multi-delta crop (one dask compute call) ---
    all_met_files, nan_idxs_per_delta = _cut_satellite_met_multi_delta(
        met_source,
        #data.fp_data_full,
        data.fp_xr,
        metsize=data.size,
        time_deltas=time_deltas,
        pad_mode=pad_mode,
        add_wind_direction=add_wind_direction,
        verbose=verbose,
        load_into_memory=load_into_memory,
        interp_to=interp_to,
    )



    # Collect all nan indices across deltas
    all_nan_idxs = (
        np.unique(np.concatenate([np.atleast_1d(v) for v in nan_idxs_per_delta.values()]))
        if any(len(v) > 0 for v in nan_idxs_per_delta.values())
        else np.array([], dtype="datetime64[ns]")
    )

    # Extract lat/lon coords before adding time_delta dim (same across all deltas)
    first_delta = time_deltas[0]
    lat_coords_da = all_met_files[first_delta]["lat_coords"]
    lon_coords_da = all_met_files[first_delta]["lon_coords"]

    # Build full_met: concat per-delta datasets along a new time_delta dimension
    met_datasets_for_concat = []
    for delta in time_deltas:
        ds = all_met_files[delta].drop_vars(["lat_coords", "lon_coords"], errors="ignore")
        ds = ds.expand_dims(dim={"time_delta": [delta]})
        met_datasets_for_concat.append(ds)

    if len(met_datasets_for_concat) == 1:
        full_met = met_datasets_for_concat[0]
    else:
        full_met = xr.concat(met_datasets_for_concat, dim="time_delta")
    # Re-attach geographic coords (no time_delta dim — used for static_ds below)
    full_met["lat_coords"] = lat_coords_da
    full_met["lon_coords"] = lon_coords_da

    for v in full_met.data_vars:
        full_met[v] = full_met[v].astype("float32", copy=False)

    full_met = full_met.transpose("fp_time", "lat", "lon", ..., "time_delta")

    # Detect which variables have levels vs surface (after wind vars have been added)
    _skip = {"lat_coords", "lon_coords"}
    levels_variables_needed = [
        v for v in full_met.data_vars
        if v not in _skip and "levels" in full_met[v].dims
    ]
    surface_variables_needed = [
        v for v in met_variables
        if v in full_met.data_vars and v not in _skip
        and "levels" not in full_met[v].dims
    ]   
    # also pick up wind_speed/wind_angle if they're surface (no levels)

    for v in ("wind_speed", "wind_angle"):
        if (v in full_met.data_vars and v not in levels_variables_needed
                and v not in surface_variables_needed):
            surface_variables_needed.append(v)

    if len(all_nan_idxs) > 0:
        full_met = full_met.drop_sel(fp_time=all_nan_idxs)
        warnings.warn(
            f"removing {len(all_nan_idxs)} indices due to problems with "
            "interpolating meteorology for the passed time_deltas"
        )
        data.remove_indeces(all_nan_idxs)

    input_arrays = []

    if len(levels_variables_needed) > 0:
        stacked_levels_met = _stack_and_label_variables(
            full_met,
            levels_variables_needed,
            "met_with_levels",
            met_variables_dict={v: met_levels for v in levels_variables_needed},
            verbose=verbose,
        )
        if stacked_levels_met is not None:
            input_arrays.append(stacked_levels_met)

    if len(surface_variables_needed) > 0:
        stacked_surface_met = _stack_and_label_variables(
            full_met, surface_variables_needed, "surface_met", verbose=verbose,
        )
        if stacked_surface_met is not None:
            input_arrays.append(stacked_surface_met)

    if len(static_variables) > 0:
        static_variables_functions = get_static_variables_functions()
        (static_ds,) = xr.broadcast(full_met[["lat_coords", "lon_coords"]])

        for var in static_variables:
            if var in ["topog", "landcover", "landcover_disaggregated"]:
                assert hasattr(data, "topog"), (
                    "Load topog on the data object before extracting this variable!"
                )
                static_ds = static_variables_functions[var](data.topog, static_ds)
            elif "domain" in var:
                if var in static_variables_functions:
                    static_ds = static_variables_functions[var](data.fp_data_full, static_ds)
                else:
                    warnings.warn(f"variable {var} was not found and will be skipped")
            elif var in static_variables_functions and var not in ["lat_coords", "lon_coords"]:
                static_ds = static_variables_functions[var](static_ds)
            elif var not in ["lat_coords", "lon_coords"]:
                warnings.warn(f"variable {var} was not found and will be skipped")

        if "lat_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lat_coords"], errors="ignore")
        if "lon_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lon_coords"], errors="ignore")

        stacked_static = _stack_and_label_variables(
            static_ds, list(static_ds.data_vars), "static", verbose=verbose,
        )
        print("loading stadcked static into memory")
        stacked_static = stacked_static.compute()
        if stacked_static is not None:
            input_arrays.append(stacked_static)

    if len(input_arrays) == 0:
        raise ValueError(
            "All requested variables/levels were unavailable after filtering. "
            "Please check met_variables/met_levels/static_variables against dataset contents."
        )

    concatenated_inputs = xr.concat(input_arrays, dim="variable_name")
    mindex = pd.MultiIndex.from_tuples(
        concatenated_inputs.variable_name.values,
        names=["variable", "levels", "time_delta"],
    )
    concatenated_inputs = concatenated_inputs.assign_coords(
        xr.Coordinates.from_pandas_multiindex(mindex, "variable_name")
    )

    concatenated_inputs.attrs = {
        "source": concatenated_inputs.attrs.get("source", "unknown"),
        "time_deltas": time_deltas,
        "generated on": str(datetime.datetime.now()),
    }
    concatenated_inputs = concatenated_inputs.astype("float32", copy=False)

    if concatenated_inputs.sizes.get("variable_name", 0) == 0:
        raise ValueError(
            "All requested variables/levels were unavailable after filtering. "
            "Please check met_variables/met_levels/static_variables against dataset contents."
        )

    return concatenated_inputs, data
