"""
author: Elena Fillola @elenafillo
"""

import numpy as np
import xarray as xr
import datetime
import pandas as pd
import joblib
from pathlib import Path

import torch
import xbatcher as xb
import xbatcher.loaders.torch

from .load_data_helper_funs import *

from .load_data import cut_satellite_met

def get_square_satellite_inputs(data, met_variables, time_deltas=[], static_variables=[], verbose=True, return_variable_names=False, return_asarray=False):
    """
    LATEST VERSION - get inputs from LoadSquareSatelliteData object and format as a DataArray of size (time, lat, lon, variables)

    ToDo: add option for it to work with LoadDomainSatelliteData! 

    Inputs:
    - data: LoadSquareSatelliteData object
    - met_variables: dict, of shape {'variable_name':levels_to_extract, 'surface_variable':[], ...}. For each atmospheric variable with levels, pass the levels to extract as a list. For each surface variable, pass an empty list
    - time_deltas: list, of t-H hours to extract the met variables. t=0 (ie the time of the satellite measurement) is extracted automatically. time_deltas=[6,12] extracts the data at t=0, t-6h and t-12h.
    - static_variables: list of static variables to add, eg topog, landcover, lat_coords. You can see or increase the list of valid names get_static_variables_functions()
    - return_variable_names: bool, if true also returns a list of dicts with the variable names
    - return_asarray: bool, if true returns as an np array of shape (time, lat, lon, variables) and the variable_names

    returns:
    - if return_asarray=False and return_variable_names=False: xarray DataArray of size (fp_time, lat, lon, variable_name) where fp_time is the time of the reference footprint, and lat and lon are the artificial coordinates centered around the release point. The variable_name is a multiIndex, shown as a tuple of form (variable_name, level, time_delta) e.g. ('x_wind', 15, 6) for the x_wind at model height 15 and at t-6h. Surface variables and static variables have level = 0, and static variables always have time_delta = 0. 

    - if return_asarray=True returns a np array of shape (time, lat, lon, variables) and a list of variable names in the same order as the variable dimension of the array. The variable names are shown as tuples as described above. We are trying to move away from this and towards only using xarrays.


    """
    assert hasattr(data, "dataset_format"), "It doesn't seem this is a SatelliteData object"
    #assert data.dataset_format == "square", "At the moment this only works for LoadSquareSatelliteData objects"
    assert data.met_processed == True, "Make sure that you have loaded and cut the meteorology in the SatelliteData object"
    
    if type(met_variables) is not dict:
            raise ValueError("met_variables should be a dict of shape {'variable_name':levels_to_extract, 'surface_variable':[], ...}. For each atmospheric variable with levels, pass the levels to extract as a list. For each surface variable, pass an empty list")


    if verbose: 
        print("------------------------")
        print("---EXTRACTING MET DATA---")

    if not (0 in time_deltas):
        time_deltas.append(0) # append 0 to get present met too

    time_deltas=list(sorted(set(time_deltas)))

    all_met_files={}

    # subset before saving to the dict

    min_levels_needed = list(set([levels[i] for levels in list(met_variables.values()) for i in range(len(levels))]))
                             
    met_variables_needed = list(met_variables.keys())

    surface_variables_needed = [var_name for var_name in met_variables if met_variables[var_name]==[]]

    levels_variables_needed = [var_name for var_name in met_variables if len(met_variables[var_name])>0]

    # check that the passed variables and levels are available in data.met (cut data object for 0)

    if len(time_deltas)>0:
        print(f"extracting met at t-H for H in: {time_deltas}")
        met_nan_idxs = {}
        # filename here 
        for delta in time_deltas:
            if delta==0:
                met = data.met
                met = select_met_levels(met, levels=min_levels_needed)
                met = select_met_variables(met, variables=met_variables_needed)
                # swap the dimensions so that each dataset, no matter the time delta, is aligned along the fp_time (the time of the measurement)
                # so that met.time = met.fp_time - met.time_delta in hours
                met = met.swap_dims({"time":"fp_time"})
                #met = met.reset_coords(["time"])
                met = met.drop_vars("time")
                #met = met.rename({"fp_time":"time"})

            else:
                # to make this extendable to LoadDomainSatelliteData, add an option here that processes it met for the fix domain instead of this function, which does square cropping (to be written)
                if data.dataset_format == "square":
                    met, nan_idxs = cut_satellite_met(data.met_file, data.fp_data_full, metsize=data.metsize, time_delta=delta, relevant_levels = min_levels_needed, relevant_variables = met_variables_needed, pad_mode=data.fill_outofdomain_with, load=False, add_wind_direction=True, return_nan_idxs=True)
                    met_nan_idxs[delta] = nan_idxs
                #if data.dataset_format == "domain":
                #    met = process_domain_met(data.met_file, data.fp_data_full,time_delta=delta, relevant_levels = min_levels_needed, relevant_variables = met_variables_needed, add_wind_direction=True)

                met = met.swap_dims({"time":"fp_time"})
                #met = met.reset_coords(["time"])
                met = met.drop_vars("time")
                
            # does having .copy load it into memory? check
            all_met_files[delta] = met

            del met 

    
    # concatenate all met datasets, which should have the same coordinates except the time_delta dimension
    full_met = xr.concat(list(all_met_files.values()), dim="time_delta", data_vars =met_variables_needed).transpose("fp_time", "lat", "lon", ..., "time_delta")
    for v in full_met.data_vars:
        full_met[v].astype("float32", copy=False)

    # if the time_delta is large, cut met might have interpolated to t-time_delta outside of the known met. check and if so remove indeces
    all_nan_idxs = np.unique(np.concatenate(list(met_nan_idxs.values())))
    if len(all_nan_idxs)>0:
        full_met = full_met.drop_isel(fp_time=all_nan_idxs)
        print(f"removing {len(all_nan_idxs)} indeces due to problems with interpolating meteorology for the passed time_deltas")
        data.remove_indeces(all_nan_idxs)

    """
    if data.met.time.values[0] - pd.Timedelta(f"{max(time_deltas)}h") < data.met_file.time.values[0]:
        badly_interpolated = data.met.time.values - pd.Timedelta(f"{max(time_deltas)}h") < data.met_file.time.values[0]
        full_met = full_met.drop_sel(fp_time=full_met.fp_time.values[badly_interpolated])

        # this updates any indeces that couldnt be interpolated
        # i think it updates data without needing to return it as a new object
        data.remove_indeces(np.where(badly_interpolated)[0])

    """

    input_arrays = []
    varnames_dict = []

    
    ### SETTING UP VARIABLES WITH LEVELS
    # stack along the variable dimension, so that the new variable has shape (variable name, level, time_delta)
    if len(levels_variables_needed)>0:
        if verbose: print(f"Setting up variables with levels: {levels_variables_needed}")
        stacked_levels_met = full_met[levels_variables_needed].to_stacked_array(new_dim="variable_name", sample_dims=["fp_time", "lat", "lon"], name="stacked_levels_met")

        # make sure we keep only the levels passed in met_variables
        indexes = []
        for v in list(set(levels_variables_needed)):
            for delta in time_deltas:
                for lev in met_variables[v]:
                    indexes.append((v, lev, delta))
        
        stacked_levels_met = stacked_levels_met.sel(variable_name=indexes)

        varnames_dict = varnames_dict + [{"var":tup[0], "level":tup[1], "time_delta":tup[2], "type":"met"} for tup in stacked_levels_met.variable_name.values]
        

        stacked_levels_met = stacked_levels_met.drop_vars({'time_delta', 'variable_name', 'levels','variable'}).assign_coords({"variable_name":stacked_levels_met.variable_name.values})

        if return_asarray:
            stacked_levels_met = stacked_levels_met.chunk({"fp_time":100, "variable_name":1})
            stacked_levels_met.load()

        input_arrays.append(stacked_levels_met)
        
    else:
        stacked_levels_met = None
    
    ### SETTING UP VARIABLES WITH NO LEVELS
    if len(surface_variables_needed)>0:
        if verbose: print(f"Setting up surface variables: {surface_variables_needed}")
        
        # add empty variable levels so it aligns with the met dataset
        stacked_surface_met = full_met[surface_variables_needed].assign_coords(levels=0).expand_dims("levels").transpose("fp_time", "lat", "lon", "levels", "time_delta").to_stacked_array(new_dim="variable_name", sample_dims=["fp_time", "lat", "lon"], name="stacked_surface_met")
        varnames_dict = varnames_dict + [{"var":tup[0], "time_delta":tup[2], "type":"surface_met"} for tup in stacked_surface_met.variable_name.values]
        stacked_surface_met = stacked_surface_met.drop_vars({'time_delta', 'variable_name', 'variable'}).assign_coords({"variable_name":stacked_surface_met.variable_name.values})

        if return_asarray:
            stacked_surface_met.load()

        input_arrays.append(stacked_surface_met)

    else:
        stacked_surface_met = None

    ### SETTING UP NON-MET VARIABLES
    if len(static_variables)>0:
        if verbose: print(f"Setting up static variables: {static_variables}")       

        # dict of arguments and the functions that they return
        static_variables_functions = get_static_variables_functions()

        # broadcast lat_coords and lon_coords from shape (time, lat) and (time, lon) to shared shape(time, lat, lon). we will use these as a starting array to add all the static variables, and will remove them at the end if they were not passed in "static_variables"
        if data.dataset_format=="square":
            (static_ds, ) = xr.broadcast(full_met[["lat_coords", "lon_coords"]])
        if data.dataset_format == "domain":
            (static_ds, ) = xr.broadcast(full_met.assign({"lat_coords":(("lat"), full_met.lat.values), "lon_coords":(("lon"), full_met.lon.values)})[["lat_coords", "lon_coords", "fp_time"]])

            static_ds = static_ds[["lat_coords", "lon_coords"]]

            if "time" not in data.topog.coords:
                data.topog = data.topog.broadcast_like(static_ds, exclude=["lat", "lon", "landcover_level"])
                data.topog = data.topog.assign_coords({"lat":static_ds.lat.values, "lon":static_ds.lon.values}).rename({"fp_time":"time"})
        



        for var in static_variables:
            # add each input as a variable in the static_ds
            if var in ["topog", "landcover", "landcover_disaggregated"]:
                assert hasattr(data, "topog"), "Load topog on the data object before trying to extract this as an input!"
                static_ds = static_variables_functions[var](data.topog, static_ds)
            elif "domain" in var:
                static_ds = static_variables_functions[var](data.fp_data_full, static_ds)
            elif var in list(static_variables_functions.keys()) and var not in ["lat_coords", "lon_coords"]:
                static_ds = static_variables_functions[var](static_ds)
            elif var != "lat_coords" and var != "lon_coords":
                print(f"variable {var} was not found in the list of known functions!")

        
        if "lat_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lat_coords"])
        if "lon_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lon_coords"])

        # add empty variable levels and time_delta so it aligns with the met dataset
        stacked_static_inputs = static_ds.assign_coords(levels=0, time_delta=0).expand_dims("levels").expand_dims("time_delta").transpose("fp_time", "lat", "lon", "levels", "time_delta").to_stacked_array(new_dim="variable_name", sample_dims=["fp_time", "lat", "lon"], name="stacked_static_inputs")  

        
        varnames_dict = varnames_dict + [{"var":tup[0], "type":"static"} for tup in stacked_static_inputs.variable_name.values]
        stacked_static_inputs = stacked_static_inputs.drop_vars({'variable_name', 'variable'}).assign_coords({"variable_name":stacked_static_inputs.variable_name.values})

        if return_asarray:
            stacked_static_inputs.load()

        input_arrays.append(stacked_static_inputs)

    else:
        stacked_static_inputs=None

    # make the variable name a multiindex of variable name, level and time delta, to keep track of what the variables are after concatenation
    concatenated_inputs = xr.concat(input_arrays, dim="variable_name")
    mindex = pd.MultiIndex.from_tuples(concatenated_inputs.variable_name.values, names=["variable", "levels", "time_delta"])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "variable_name")
    concatenated_inputs = concatenated_inputs.assign_coords(mindex_coords)

    concatenated_inputs.attrs = {"source": concatenated_inputs.attrs["source"] if "source" in concatenated_inputs.attrs else "unknown","time_deltas":time_deltas, "generated on": str(datetime.datetime.now())}
    concatenated_inputs = concatenated_inputs.astype("float32", copy=False)
    #latlons, idx_latlons = get_grid(data, latlon_fp)

    if not return_asarray:
        if return_variable_names:
            return concatenated_inputs, varnames_dict
        else:
            return concatenated_inputs
    
    ## ideally we move away from this! it requires loading on the spot and takes a long time
    if return_asarray:
        concatenated_inputs = concatenated_inputs.chunk({"fp_time":100, "variable_name":1})

        concatenated_inputs = concatenated_inputs.load()

        print("We are moving away from this!")

        if return_variable_names:

            return np.reshape(concatenated_inputs.values, (concatenated_inputs.fp_time.size, concatenated_inputs.lat.size*concatenated_inputs.lon.size, concatenated_inputs.variable_name.size)), varnames_dict
        else:
            return np.reshape(concatenated_inputs.values, (concatenated_inputs.fp_time.size, concatenated_inputs.lat.size*concatenated_inputs.lon.size, concatenated_inputs.variable_name.size))


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
    Wrapper for the inputs dataset that applies a scaler to the inputs. The scaler is fitted on the inputs when the fit method is called, and can be applied to the inputs or any other dataset with the same structure using the transform method. The scaler can be fitted on a subsample of the data by specifying the fit_on_subsample parameter in the constructor, which is a float between 0 and 1 that indicates the portion of the data to use for fitting, chosen randomly - which can make fitting significantly faster, rather than loading all into memory. If fit_on_subsample is 1, all the data will be used for fitting.

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
    Default inputs scaler. It scales meteorological inputs per variable and level, and applies a minmax scaler to auxiliary variables (e.g. topography). The scalers are stored in a dictionary with keys corresponding to the variable names in the multiindex of the variable_name coordinate of the inputs xarray, which are tuples of (variable, level, time_delta) for meteorological variables and (variable, "", "") for static variables. The fit method fits the scalers to the inputs, and the transform method applies the scalers to the inputs and returns a transformed xarray with the same structure as the inputs but with transformed values.
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
    def __init__(self, fp, scaler=None, scaler_params={}):
        
        fp = self._check_fp_format(fp)
        
        self.fp = fp.copy()

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
    Build a PyTorch dataloader from the inputs and fps datasets, using xbatcher to handle batching and parallel loading. 
    The inputs and fps should be aligned along the time dimension (fp_time for inputs and time for fps), and should have the same length along this dimension. The inputs should have dimensions (fp_time, lat, lon, variable_name) and the fps should have dimensions (time, lat, lon). If randomize is True, the data will be shuffled by permuting the time dimension before creating the dataloader. The random_seed parameter controls the seed for reproducibility of the shuffling. The dataloader_params can be used to pass additional parameters to the PyTorch DataLoader, such as num_workers for parallel loading.

    Inputs:
    - inputs: xarray DataArray of size (fp_time, lat, lon, variable_name) from get_square_satellite_inputs or similar function
    - fps: xarray datarray, or xarray Dataset with variable "fp_transformed" of size (time, lat, lon) 
    - batch_size: int, batch size for the dataloader
    - randomize: bool, whether to shuffle the dataset along the time dimension. Recommended for training and not for testing
    - random_seed: int, seed for reproducibility of the shuffling when randomize is True
    - dataloader_params: dict, additional parameters to pass to the PyTorch DataLoader

    Returns:
    - dataloader: PyTorch DataLoader that yields batches of (inputs, fps), where inputs is a batch of the input data and fps is a batch of the corresponding footprints, withs shape (batch_size, lat, lon, variable_name) and (batch_size, lat, lon) respectively. The inputs and fps in each batch are aligned along the time dimension.
    
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
        fps_labels = fps.variable_name.values
        fps = fps.chunk(time=batch_size, variable_name=-1)

        y_bgen = xb.BatchGenerator(
            fps,
            input_dims={"lat":len(fps.lat), "lon":len(fps.lon), "variable_name": len(fps.variable_name)},
            batch_dims={'time': batch_size},
            preload_batch=False,
        )

    else:
        fps_labels = fps.name if fps.name is not None else "fp"
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
            "multiprocessing_context": 'forkserver',  # Use "forkserver" to spawn subprocesses, ensuring stability in multiprocessing
        }
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # Using batches defined by the dataset itself (via xbatcher)
        **dataloader_params
    )

    
    return dataloader, fps_labels