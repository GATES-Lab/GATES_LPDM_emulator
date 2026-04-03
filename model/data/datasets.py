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

from .load_data import cut_satellite_met

def _stack_and_label_variables(ds, var_names, var_type, met_variables_dict=None, verbose=False):
    """
    Stack requested variables into a single variable_name dimension with tuple labels.
    Returns (stacked_dataarray_or_None, warnings).
    """
    if not var_names:
        return None, []

    if verbose:
        print(f"Setting up {var_type}: {var_names}")


    if var_type == "met_with_levels":
        filtered_vars = {}
        for var in var_names:
            if var not in ds.data_vars:
                warnings.warn(f"requested variable {var} is not available and will be skipped")
                continue

            if "levels" not in ds[var].coords:
                warnings.warn(f"variable {var} has no 'levels' coordinate and will be skipped")
                continue

            requested_levels = list(met_variables_dict.get(var, [])) if met_variables_dict is not None else []
            available_levels = list(ds[var].levels.values)
            valid_levels = [lev for lev in requested_levels if lev in available_levels]
            dropped_levels = [lev for lev in requested_levels if lev not in available_levels]

            if dropped_levels:
                warnings.warn(f"requested levels {dropped_levels} for variable {var} are not available and will be skipped")

            if len(valid_levels) == 0:
                warnings.warn(f"variable {var} has no valid levels left after filtering and will be skipped")
                continue

            filtered_vars[var] = ds[var].sel(levels=valid_levels)

        if len(filtered_vars) == 0:
            return None

        data = xr.Dataset(filtered_vars).transpose("fp_time", "lat", "lon", "levels", "time_delta")

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

    variable_labels = list(stacked.variable_name.values)
    if len(variable_labels) == 0:
        return None

    # Rebuild tuple labels from split coords if variable_name is not tuple-like.
    if not isinstance(variable_labels[0], tuple):
        if all(coord in stacked.coords for coord in ["variable", "levels", "time_delta"]):
            variable_labels = list(
                zip(
                    stacked["variable"].values,
                    stacked["levels"].values,
                    stacked["time_delta"].values,
                )
            )
        else:
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
    print(f"Time deltas: {time_deltas}")

    met_variables_needed = list(met_variables.keys())
    surface_variables_needed = [var_name for var_name, levs in met_variables.items() if levs == []]
    levels_variables_needed = [var_name for var_name, levs in met_variables.items() if len(levs) > 0]
    min_levels_needed = sorted({lev for levels in met_variables.values() for lev in levels})

    all_met_files = {}
    met_nan_idxs = {}

    if len(time_deltas) > 0:
        print(f"extracting met at t-H for H in: {time_deltas}")
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
    def __init__(self, kwargs={}):
        self.mean = None
        self.std = None
        self.scaler_type = "standard"

    def fit(self, da: xr.DataArray):
        """Compute stats over all dims except the ones you want to preserve (e.g. variable)."""
        self.mean = da.mean()
        self.std = da.std()
        return self
    
    def transform(self, da: xr.DataArray) -> xr.DataArray:
        return (da - self.mean) / (self.std + 1e-8)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        return self.fit(da).transform(da)
    
class XarrayMinMaxScaler:
    """
    A simple scaler for xarray DataArrays that applies minmax transformation. 
    Pass manual_min and manual_max to the constructor if you want to specify the min and max values to use for scaling, otherwise they will be computed from the data while fitting. Pass feature_range to specify the range to scale the data to, default is (0, 1).
    """
    def __init__(self, feature_range=(0, 1), manual_min =None, manual_max=None):
        self.min = manual_min
        self.max = manual_max
        self.feature_range = feature_range
        self.scaler_type = "minmax"

    def fit(self, da: xr.DataArray):
        if self.min is None:
            self.min = da.min()
        if self.max is None:
            self.max = da.max()
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
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
    def __init__(self, inputs: xr.DataArray, scaler=None, scaler_params={"fit_on_subsample": 1}, verbose=False):
        self.inputs = inputs
        self.scaler = scaler
        self.verbose = verbose

        self.fit_on_subsample = scaler_params.pop("fit_on_subsample", 1)

        if scaler is None:
            self.scaler = DefaultInputsScaler(**scaler_params, verbose=self.verbose)
        else:
            self.scaler = scaler(**scaler_params)
        
        def fit(self):
            self.scaler.fit(self.inputs)

    def fit(self):
        if self.fit_on_subsample<0: 
            raise ValueError(f"fit_on_subsample should be between 0 and 1, but got {self.fit_on_subsample}. Please provide a valid value for fit_on_subsample.")
        elif self.fit_on_subsample<1:
            times = pd.DatetimeIndex(self.inputs.fp_time.values)
            n_samples = int(len(times)*self.fit_on_subsample)
            if self.verbose: print(f"fit_on_subsample is {self.fit_on_subsample}, so only using {n_samples} samples to fit the scaler")

            selected_times = np.sort(np.random.choice(times, n_samples, replace=False))

            self.subsampled_inputs = self.inputs.sel(fp_time=selected_times) 

            self.scaler.fit(self.subsampled_inputs)

        elif self.fit_on_subsample>=1:
            self.scaler.fit(self.inputs)
        
    def transform(self, inputs=None):
        return self.scaler.transform(inputs)


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
    def __init__(self, minmax_variables=["land_cover", "topog", "x_coords", "y_coords", "lat_coords", "lon_coords"], verbose=False):
        self.minmax_variables = minmax_variables
        self.scalers = {}
        self.scaler_name = "DefaultInputsScaler"
        self.verbose = verbose

    def fit(self, inputs: xr.DataArray):

        variable_names = inputs.variable_name.values
        varnames = [var[0] for var in inputs.variable_name.values]
        varnames = np.unique(varnames)
        self.fitted_variable_names = list(variable_names)

        for varname in np.unique(varnames):
            var_data = inputs.sel(variable=varname)
            if varname in self.minmax_variables:
                if self.verbose: print(f"fitting minmax scaler for var {varname}")
                scaler = XarrayMinMaxScaler()
                scaler = scaler.fit(var_data)

                # save the scaler
                for vc in variable_names:
                    if vc[0] == varname and len(vc)==3:
                        self.scalers[vc] = scaler
                
            else:
                if self.verbose: print(f"fitting standardise scaler for var {varname}")
                levels = np.unique(var_data.levels.values)

                for level in levels:
                    scaler = XarrayScaler()
                    level_data = var_data.sel(levels=level)
                    scaler = scaler.fit(level_data)

                    # save the scaler for this variable and level for each variable tuple in the multiindex that matches this variable and level
                    for vc in variable_names:
                        if vc[0] == varname and len(vc)==3:
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
        pass 
    
    def transform(self, fp):
        transformed_fp = np.log10(fp.where(fp > 0)) + self.minimum_oom  # take the log and add the minimum_oom), leave the zeros as is
        if self.non_negative:
            transformed_fp = transformed_fp.where(transformed_fp > 0, 0) # make all values that are zero or below zero (which can happen if the original fp was between 0 and 10**(-minimum_oom)) zero, to avoid having negative values in the transformed fp
        return transformed_fp
    
    def inverse_transform(self, transformed_fp):

        original_fp = 10**(transformed_fp - self.minimum_oom)
        
        #original_fp = transformed_fp.where(transformed_fp <= self.minimum_oom, original_fp)
        # make all negative values zero
        original_fp = original_fp.where(original_fp >= 0, 0)
        original_fp = original_fp.where(original_fp > 10**-int(self.minimum_oom), 0)
        return original_fp
    
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
        self.logged_mean = np.mean(np.log10(fp.values.flatten()[fp.values.flatten()>0]))
        self.parameters = {"logged_mean": self.logged_mean, "minimum_oom": self.minimum_oom}
    
    def transform(self, fp):
        transformed_fp = np.log10(fp+10**(-self.minimum_oom))  # take the log and shift by the logged mean
        transformed_fp = transformed_fp + abs(self.logged_mean)

        return transformed_fp
    
    def inverse_transform(self, transformed_fp):
        original_fp = 10**(transformed_fp - abs(self.logged_mean)) - 10**(-self.minimum_oom)
        original_fp = original_fp.where(original_fp >= 0, 0)
        return original_fp
    
class FootprintDataset:
    """
    Wrapper around footprint data that applies a footprint scaler.

    The dataset accepts either an ``xarray.DataArray`` or an ``xarray.Dataset`` containing
    a variable named ``fp``. ``fit`` fits the underlying scaler, ``transform`` returns a
    Dataset containing ``fp_transformed`` and ``fp_original``, and ``inverse_transform``
    accepts either the Dataset returned by ``transform`` or the transformed DataArray.
    """

    def __init__(self, fp, scaler=None, scaler_params={}, add_nan_mask=False):
        
        fp = self._check_fp_format(fp)
        
        self.fp = fp.copy()
        self.add_nan_mask = add_nan_mask

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
        transformed_fp = transformed_fp.rename("transformed_fp")
        transformed_fp.attrs = {"transformer": self.scaler.__class__.__name__, "generated_on": str(datetime.datetime.now())}
        ds = xr.Dataset({
            "fp_transformed": transformed_fp,
            "fp_original": fp,
        })

        if self.add_nan_mask:
            ds["fp_nan_mask"] = xr.where(fp.isnull(), 1, 0)

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
        elif not isinstance(transformed_fp, xr.DataArray):
            raise ValueError("transformed fp must be an xarray DataArray!")        
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


def make_dataloader(inputs, fps, batch_size=10, randomize=False, random_seed=42, dataloader_params=None):
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

    Returns:
        - dataloader: PyTorch DataLoader that yields batches of ``(inputs, fps)``.
        - fps_labels: footprint variable labels, as a list
    
    """
    # ensure that both have the right dimensions
    if inputs.sizes["fp_time"] != fps.sizes["time"]:
        raise ValueError("Incompatible dimensions between inputs and fps")
    if isinstance(fps, xr.Dataset):
        print(f"you passed a fps dataset with multiple variables: {list(fps.data_vars)}. All variables will be returned in the dataloader along a new dimension. ")
        ## but make sure this is what you want! If you only want to return the transformed fps variable, pass fps['fp_transformed'] instead of the whole dataset when calling this function.

    if randomize:
        print("randomizing dataset!")
        # set the random seed for reproducibility
        np.random.seed(random_seed)
        # shuffle the data by permuting the fp_time dimension
        permuted_time = np.random.permutation(inputs.fp_time)
        inputs = inputs.sel(fp_time=permuted_time)
        fps = fps.sel(time=permuted_time)


    inputs = inputs.chunk(fp_time=batch_size)
    inputs = inputs.transpose("fp_time", "lat", "lon", "variable_name")
    
    X_bgen = xb.BatchGenerator(
        inputs,
        input_dims={"lat":len(inputs.lat), "lon":len(inputs.lon), "variable_name": len(inputs.variable_name)},
        batch_dims={'fp_time': batch_size},
        preload_batch=False,
    )

    # stack all variables in the fps along a new variable dimension, so that the dataloader returns all variables in the fps dataset. If fps is already an xarray DataArray, this will just add a variable dimension of size 1.
    if isinstance(fps, xr.Dataset):
        fps = fps.to_stacked_array(new_dim="variable_name", sample_dims=["time", "lat", "lon"], name="stacked_fps")
        fps = fps.transpose("time", "lat", "lon", "variable_name")
        fps_labels = list(fps.variable_name.values)
        fps = fps.chunk(time=batch_size, variable_name=-1)

        y_bgen = xb.BatchGenerator(
            fps,
            input_dims={"lat":len(fps.lat), "lon":len(fps.lon), "variable_name": len(fps.variable_name)},
            batch_dims={'time': batch_size},
            preload_batch=False,
        )

    else:
        fps_labels = [fps.name if fps.name is not None else "fp"]
        fps = fps.chunk(time=batch_size)


        y_bgen = xb.BatchGenerator(
            fps,
            input_dims={"lat":len(fps.lat), "lon":len(fps.lon)},
            batch_dims={'time': batch_size},
            preload_batch=False,
        )

    dataset = xbatcher.loaders.torch.MapDataset(X_bgen, y_bgen)
    if dataloader_params is None:
        dataloader_params = {
            "prefetch_factor": 3,  # Prefetch up to 3 batches in advance to reduce data loading latency
            "num_workers": 4,  # Use 4 parallel worker processes to load data concurrently
            "persistent_workers": True,  # Keep workers alive between epochs for faster subsequent epochs
            #"multiprocessing_context": 'forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
        }
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        **dataloader_params
    )

    
    return dataloader, fps_labels