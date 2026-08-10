"""
author: Elena Fillola @elenafillo
"""

import numpy as np
import xarray as xr
import dask
import datetime
import pandas as pd
import joblib
from pathlib import Path
import warnings

import torch
import xbatcher as xb
import xbatcher.loaders.torch

from .load_data_helper_funs import *

from .load_data import _get_release_idxs, _pad_domain

def _stack_and_label_variables(ds, var_names, var_type,  met_variables_dict=None, verbose=False):
    """Stack requested variables into a single ``variable_name`` dimension with tuple labels.

    Args:
        ds (xr.Dataset): dataset containing the requested variables.
        var_names (list[str]): List of variable names to stack. If a variable is not present in the dataset, it will be skipped with a warning.
        var_type (str): One of "met_with_levels", "surface_met", or "static".
        verbose (bool, optional): If True, prints setup information. Defaults to False.

    Returns:
        xr.DataArray or None: Stacked data array with a MultiIndex ``variable_name``
        dimension of (variable, levels, time_delta) tuples, or None if no requested
        variables were available.

    Raises:
        ValueError: If ``var_type`` is not one of the supported types, or if
            variable_name tuples cannot be constructed from the stacked data.
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

            requested_levels = list(met_variables_dict.get(var, [])) if met_variables_dict is not None else []
            available_levels = list(ds[var].levels.values)
            valid_levels = [lev for lev in requested_levels if lev in available_levels]
            dropped_levels = [lev for lev in requested_levels if lev not in available_levels]

            if dropped_levels:
               warnings.warn(f"requested levels {dropped_levels} for variable {var} are not available and will be skipped")

            if len(valid_levels) == 0:
               warnings.warn(f"variable {var} has no valid levels left after filtering and will be skipped")
               continue

            filtered_vars.append(var)
        

        if len(filtered_vars) == 0:
            return None
        
        data = ds[filtered_vars].transpose("fp_time", "lat", "lon", "levels", "time_delta")


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
    
    # to_stacked_array reshapes all variables into a single variable_name axis.
    # NOTE: do not wrap this in split_large_chunks=True — in dask 2024.5.0 that can
    # drive the reshape through a broken auto_chunks path (IndexError: tuple index
    # out of range). When the input is chunked along fp_time (see the chunk_size
    # handling in _cut_satellite_met_multi_delta) this reshape stays bounded; if not,
    # it may emit a harmless "large chunk" warning.
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



### Transforming and scaling data - inputs
class XarrayScaler:
    """A simple scaler for xarray DataArrays that applies the standard scaling transformation.

    Attributes:
        mean (float or None): Fitted mean, or the manually-provided mean.
        std (float or None): Fitted standard deviation, or the manually-provided value.
        scaler_type (str): Identifier for this scaler type, "standard".
        scaler_name (str): Name of the scaler class, "XarrayScaler".
        compute (bool): Whether to eagerly compute dask-backed mean/std into numpy arrays.
    """
    def __init__(self, compute=True, kwargs={}, mean=None, std=None):
        """Initialize the scaler.

        Args:
            compute (bool, optional): If True, computes the fitted mean/std immediately
                and stores them as numpy arrays. Defaults to True.
            kwargs (dict, optional): <FILL IN>. Defaults to {}.
            mean (float, optional): Manually-provided mean to use instead of fitting. Defaults to None.
            std (float, optional): Manually-provided standard deviation to use instead of fitting. Defaults to None.
        """
        self.mean = mean
        self.std = std
        self.scaler_type = "standard"
        self.scaler_name = "XarrayScaler"
        self.compute = compute

    def fit(self, da: xr.DataArray):
        """Compute stats over all dims except the ones you want to preserve (e.g. variable).

        Args:
            da (xr.DataArray): Data to compute the mean and standard deviation from.

        Returns:
            XarrayScaler: self, with ``mean``, ``std``, and ``params`` populated.
        """
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
        """Apply standard scaling to the data.

        Args:
            da (xr.DataArray): Data to transform.

        Returns:
            xr.DataArray: Standardised data, ``(da - mean) / std``.
        """
        #da = da.astype("float32", copy=False)
        return (da - self.mean) / (self.std + 1e-8)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        """Fit the scaler on ``da`` and then transform it.

        Args:
            da (xr.DataArray): Data to fit and transform.

        Returns:
            xr.DataArray: Standardised data.
        """
        return self.fit(da).transform(da)
    

class GhostScaler:
    """A placeholder scaler that does not apply any transformation, but has the same interface as the other scalers.

    Useful for testing and ablation when you want to use the same code but without
    scaling the inputs.

    Attributes:
        scaler_type (str): Identifier for this scaler type, "ghost".
        scaler_name (str): Name of the scaler class, "GhostScaler".
    """
    def __init__(self):
        """Initialize the scaler."""
        self.scaler_type = "ghost"
        self.scaler_name = "GhostScaler"

    def fit(self, da: xr.DataArray):
        """No-op fit, kept for interface compatibility with the other scalers.

        Args:
            da (xr.DataArray): Unused.

        Returns:
            GhostScaler: self.
        """
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        """Return the data unchanged.

        Args:
            da (xr.DataArray): Data to "transform".

        Returns:
            xr.DataArray: The same data, unchanged.
        """
        return da
    


class XarrayMinMaxScaler:
    """A simple scaler for xarray DataArrays that applies minmax transformation.

    Pass ``manual_min`` and ``manual_max`` to the constructor if you want to specify
    the min and max values to use for scaling, otherwise they will be computed from
    the data while fitting. Pass ``feature_range`` to specify the range to scale the
    data to, default is (0, 1).

    Attributes:
        min (float or None): Fitted or manually-provided minimum value.
        max (float or None): Fitted or manually-provided maximum value.
        feature_range (tuple): Target range for the scaled data.
        scaler_type (str): Identifier for this scaler type, "minmax".
        scaler_name (str): Name of the scaler class, "XarrayMinMaxScaler".
        compute (bool): Whether to eagerly compute dask-backed min/max into numpy arrays.
    """
    def __init__(self, feature_range=(0, 1), manual_min =None, manual_max=None, compute=True):
        """Initialize the scaler.

        Args:
            feature_range (tuple, optional): Target (min, max) range to scale the data to.
                Defaults to (0, 1).
            manual_min (float, optional): Manually-provided minimum to use instead of fitting.
                Defaults to None.
            manual_max (float, optional): Manually-provided maximum to use instead of fitting.
                Defaults to None.
            compute (bool, optional): If True, computes the fitted min/max immediately and
                stores them as numpy arrays. Defaults to True.
        """
        self.min = manual_min
        self.max = manual_max
        self.feature_range = feature_range
        self.scaler_type = "minmax"
        self.scaler_name = "XarrayMinMaxScaler"
        self.compute = compute

    def fit(self, da: xr.DataArray):
        """Compute the min and max of ``da``, unless manual values were already provided.

        Args:
            da (xr.DataArray): Data to compute the min and max from.

        Returns:
            XarrayMinMaxScaler: self, with ``min``, ``max``, and ``params`` populated.
        """
        #if self.min is None or self.max is None:
            #da = da.astype("float32", copy=False)
        if self.min is None:
            self.min = da.min()
            if self.compute:
                print("Computing min for scaler...")
                self.min = self.min.compute().astype("float32").values
                print("min:", self.min)
                if abs(self.min) > 1e25 or self.min<-50000:
                    self.min = np.min(da.values).astype("float32")
                    print("recalculated min from values:", self.min)
        if self.max is None:
            self.max = da.max()
            if self.compute:
                print("Computing max for scaler...")
                self.max = self.max.compute().astype("float32").values
                #self.max = np.max(da.values).astype("float32")
                print("max:", self.max)
                if abs(self.max) > 1e25:
                    self.max = np.max(da.values).astype("float32")
                    print("recalculated max from values:", self.max)


        self.params = {"min": self.min, "max": self.max, "feature_range": self.feature_range}
        return self

    def transform(self, da: xr.DataArray) -> xr.DataArray:
        """Apply minmax scaling to the data.

        Args:
            da (xr.DataArray): Data to transform.

        Returns:
            xr.DataArray: Data scaled to ``feature_range``.
        """
        #da = da.astype("float32", copy=False)
        scale = self.feature_range[1] - self.feature_range[0]
        return self.feature_range[0] + scale * (da - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, da: xr.DataArray) -> xr.DataArray:
        """Fit the scaler on ``da`` and then transform it.

        Args:
            da (xr.DataArray): Data to fit and transform.

        Returns:
            xr.DataArray: Data scaled to ``feature_range``.
        """
        return self.fit(da).transform(da)


class InputsDataset:
    """Wrapper for an input ``xarray.DataArray`` that applies a scaler to the inputs.

    The scaler is fitted on the stored inputs when ``fit`` is called and can then be
    applied to compatible inputs via ``transform``. When ``fit_on_subsample`` is
    between 0 and 1, only a random subset of ``fp_time`` indices is used for fitting
    and the selected subset is exposed on ``subsampled_inputs``.

    Valid scalers:
        - DefaultInputsScaler: applies a standard scaler to each variable across all
          timesteps per level for meteorological variables, and a minmax scaler to
          land cover and topog variables.
        - HandcraftedInputsScaler (only a placeholder for now): applies different
          scalers to different variables based on some predefined logic.

    Attributes:
        inputs (xr.DataArray): The stored input data.
        scaler: The scaler instance used to fit/transform the inputs.
        verbose (bool): Whether to print information during fitting.
        compute (bool): Whether the scaler computes parameters eagerly.
        fit_on_subsample (float): Fraction of samples used for fitting.
        subsampled_inputs (xr.DataArray): Subset of ``inputs`` used for fitting, set
            only when ``fit_on_subsample`` < 1 after calling ``fit``.
    """
    def __init__(self, inputs: xr.DataArray, scaler=None, fit_on_subsample=1, scaler_params={}, verbose=False, compute=True):
        """Initialize the dataset wrapper.

        Args:
            inputs (xr.DataArray): DataArray with dims (fp_time, lat, lon, variable_name)
                containing the input variables to be scaled.
            scaler (type or str, optional): Scaler class, or string name of a scaler
                class, to use. If None, defaults to ``DefaultInputsScaler``. See class
                docstring for valid scalers. Defaults to None.
            fit_on_subsample (float, optional): Float between 0 and 1 that determines how
                many samples to use for fitting the scaler. If a float between 0 and 1,
                it is the fraction of samples to use. If 1, all samples are used.
                Defaults to 1.
            scaler_params (dict, optional): Parameters to pass to the scaler when
                initializing it. For example, for ``DefaultInputsScaler``, you can pass
                ``{"minmax_variables": ["topog", "land_cover"]}`` to specify which
                variables to apply minmax scaling to instead of standard scaling.
                Defaults to {}.
            verbose (bool, optional): If True, prints out information about the fitting
                process. Defaults to False.
            compute (bool, optional): If True, computes the scaler parameters immediately
                and stores them as numpy arrays. If False, stores them as dask arrays and
                computes them on demand during transformation. Defaults to True, which is
                recommended for most use cases to avoid issues with dask arrays during
                transformation.
        """
        self.inputs = inputs
        self.scaler = scaler
        self.verbose = verbose
        self.compute = compute

        self.fit_on_subsample = fit_on_subsample

        if scaler is None:
            self.scaler = DefaultInputsScaler(**scaler_params, verbose=self.verbose, compute=self.compute)
        else:
            scaler = eval(scaler) if isinstance(scaler, str) else scaler
            self.scaler = scaler(**scaler_params)

    def fit(self):
        """Fit the underlying scaler on ``self.inputs`` (or a random subsample of it).

        Raises:
            ValueError: If ``fit_on_subsample`` is not a float/int between 0 and 1.
        """
        if self.fit_on_subsample<=0 or type(self.fit_on_subsample) not in [int, float] or self.fit_on_subsample>1:
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
        """Transform ``inputs`` using the fitted scaler and cast to float32.

        Args:
            inputs (xr.DataArray): Inputs to transform, in the same format as ``self.inputs``.

        Returns:
            xr.DataArray: Transformed inputs, cast to float32.
        """
        transformed = self.scaler.transform(inputs)
        if transformed.dtype != "float32":
            transformed = transformed.astype("float32", copy=False)
        return transformed

    def fit_transform(self):
        """Fit the scaler on ``self.inputs`` and then transform ``self.inputs``.

        Returns:
            xr.DataArray: Transformed inputs, cast to float32.
        """
        self.fit()
        return self.transform(self.inputs)


class DefaultInputsScaler:
    """Default input scaler for the stacked input ``xarray.DataArray``.

    Meteorological variables are standardized per variable and level (ie across all
    time deltas). Variables listed in ``minmax_variables`` are scaled with min-max
    scaling. The fitted scalers are stored by full variable tuple, and ``transform``
    returns a new DataArray named ``stacked_transformed_inputs`` with the same dims
    and ``variable_name`` labels as the inputs. Variables listed in
    ``ignore_variables`` are not transformed, but are still included in the output
    and have a ``GhostScaler`` assigned to them in ``self.scalers`` for consistency.
    If any variable is listed in both ``minmax_variables`` and ``ignore_variables``,
    it will be ignored and a warning will be printed.

    Attributes:
        ignore_variables (list[str]): Variables excluded from transformation.
        minmax_variables (list[str]): Variables scaled with minmax scaling.
        scalers (dict): Maps each (variable, levels, time_delta) tuple to its fitted scaler.
        scaler_name (str): Name of the scaler class, "DefaultInputsScaler".
        verbose (bool): Whether to print information during fitting.
        compute (bool): Whether per-variable scalers compute parameters eagerly.
    """
    def __init__(self, minmax_variables=["land_cover", "topog", "x_coords", "y_coords", "lat_coords", "lon_coords", "xy_distance_centre", "earth_distance_centre", "sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords"], ignore_variables=[], verbose=True, compute=True):
        """Initialize the scaler.

        Args:
            minmax_variables (list[str], optional): Variable names to scale with
                ``XarrayMinMaxScaler`` instead of standard scaling. Defaults to a
                predefined list of static/coordinate variables.
            ignore_variables (list[str], optional): Variable names to leave untransformed
                (assigned a ``GhostScaler``). Defaults to [].
            verbose (bool, optional): If True, prints out information about the fitting
                process. Defaults to True.
            compute (bool, optional): If True, computes the scaler parameters immediately
                and stores them as numpy arrays. Defaults to True.
        """

        self.ignore_variables = ignore_variables

        # remove any variables from minmax that are in ignore_variables, and warn about it
        if len(set(minmax_variables).intersection(set(ignore_variables))) > 0:
            ignored_minmax_vars = set(minmax_variables).intersection(set(ignore_variables))
            minmax_variables = [v for v in minmax_variables if v not in ignored_minmax_vars]

        self.minmax_variables = minmax_variables
        
        self.scalers = {}
        self.scaler_name = "DefaultInputsScaler"
        self.verbose = verbose
        self.compute = compute

    def fit(self, inputs: xr.DataArray):
        """Fit a scaler for each variable (and level, where applicable) in ``inputs``.

        Args:
            inputs (xr.DataArray): Stacked input data with a ``variable_name`` MultiIndex
                dimension of (variable, levels, time_delta) tuples.
        """
        variable_names = inputs.variable_name.values
        varnames = []
        for var in variable_names:
            if var[0] not in varnames:
                varnames.append(var[0])
        #varnames = np.unique(varnames)
        self.full_variable_names = list(variable_names)
        self.fitted_variable_names = varnames
        #print(varnames)

        for varname in varnames: #np.unique(varnames):
            var_data = inputs.sel(variable=varname)
            if varname in self.ignore_variables:
                if self.verbose: print(f"Not transforming variable {varname} because it is in ignore_variables")
                scaler = GhostScaler()
                for vc in variable_names:
                    if vc[0] == varname and len(vc)==3:
                        self.scalers[vc] = scaler
                

            elif varname in self.minmax_variables:
                if self.verbose: print(f"fitting minmax scaler for var {varname}")
                scaler = XarrayMinMaxScaler(compute=self.compute)
                scaler = scaler.fit(var_data)

                # save the scaler
                for vc in variable_names:
                    if vc[0] == varname and len(vc)==3:
                        self.scalers[vc] = scaler

                        #print(f"saved minmax scaler for variable {vc}")
                
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
                        
                            #print(f"saved standardise scaler for variable {vc}")
            


        
    def transform(self, inputs: xr.DataArray) -> xr.DataArray:
        """Apply the fitted per-variable scalers to ``inputs``.

        Args:
            inputs (xr.DataArray): Stacked input data with the same ``variable_name``
                labels as the data the scaler was fitted on.

        Returns:
            xr.DataArray: Transformed data, named ``stacked_transformed_inputs``, with
            the same dims and ``variable_name`` labels as ``inputs``.

        Raises:
            ValueError: If ``inputs`` contains a variable name that was not fitted on.
        """
        variable_names = inputs.variable_name.values

        for var in variable_names:
            if var not in self.full_variable_names:
                raise ValueError(
                    f"Variable name {var} in inputs is not in the variable names that were fitted on: {self.full_variable_names}. "
                    "Please make sure that the inputs you are trying to transform have the same variable names as the inputs you fitted the scaler on."
                )
        transformed = inputs.copy()

        transformed_variables = []

        for varname in variable_names:
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
        """Fit the scaler on ``inputs`` and then transform ``inputs``.

        Args:
            inputs (xr.DataArray): Stacked input data to fit and transform.

        Returns:
            xr.DataArray: Transformed data, named ``stacked_transformed_inputs``.
        """
        self.fit(inputs)
        return self.transform(inputs)

class HandcraftedInputsScaler:
    """IN DEVELOPMENT - NOT READY FOR USE.

    Input scaler using pre-determined statistics rather than computing them from data.

    ``stats`` is a path to a JSON file or a dict. Each variable entry has a ``"type"`` key
    (``"standard"`` or ``"minmax"``) plus per-level statistics::

        {
            "x_wind":   {"type": "standard", "3": {"mean": 5.21, "std": 3.14}},
            "topog":    {"type": "minmax",   "0": {"min": -50.0, "max": 3200.0}}
        }

    Level keys are strings (JSON requirement) and are cast to int internally.
    ``fit()`` populates ``self.scalers`` without touching the data — all stats come from the dict.

    Attributes:
        stats (dict): Per-variable statistics used to build scalers.
        compute (bool): Always False; kept for interface compatibility with ``XarrayScaler``.
        scalers (dict): Maps each (variable, levels, time_delta) tuple to its scaler.
        scaler_name (str): Name of the scaler class, "HandcraftedInputsScaler".
        fitted_variable_names (list): Variable names seen during ``fit``.
    """
    def __init__(self, stats):
        """Initialize the scaler.

        Args:
            stats (str or dict): Path to a JSON file, or a dict, of per-variable
                statistics as described in the class docstring.
        """
        if isinstance(stats, str):
            import json
            with open(stats) as f:
                stats = json.load(f)
        self.stats = stats
        self.compute = False # no computation needed since stats are provided, but keeping the attribute for compatibility with the XarrayScaler interface
        self.scalers = {}
        self.scaler_name = "HandcraftedInputsScaler"

    def fit(self, inputs: xr.DataArray):
        """Build per-variable scalers from ``self.stats`` (without touching ``inputs`` data).

        Args:
            inputs (xr.DataArray): Stacked input data with a ``variable_name`` MultiIndex dimension; used only to enumerate the variable tuples to assign scalers to.

        Returns:
            HandcraftedInputsScaler: self, with ``scalers`` and ``fitted_variable_names`` populated.
        """
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
        """Apply the fitted per-variable scalers to ``inputs``.

        Args:
            inputs (xr.DataArray): Stacked input data with the same ``variable_name``  labels as the data the scaler was fitted on.

        Returns:
            xr.DataArray: Transformed data, named ``stacked_transformed_inputs``, with
            the same dims and ``variable_name`` labels as ``inputs``.

        Raises:
            ValueError: If ``inputs`` contains a variable name that was not fitted on.
        """
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
        """Fit the scaler on ``inputs`` and then transform ``inputs``.

        Args:
            inputs (xr.DataArray): Stacked input data to fit and transform.

        Returns:
            xr.DataArray: Transformed data, named ``stacked_transformed_inputs``.
        """
        self.fit(inputs)
        return self.transform(inputs)

####
####
####
####

### Transforming and scaling data - outputs - footprints

class LogAndShiftFpScaler:
    """Footprint scaler.

    Takes log of fp data where non-zero, and offsets by minimum order-of-magnitude
    value so its above zero. Does not need fitting.

    Note: translates across domain sizes.

    Attributes:
        minimum_oom (int): Order-of-magnitude offset applied to the log-transformed data.
        non_negative (bool): Whether to clip the transformed data at zero.
        scaler_name (str): Name of the scaler class, "LogAndShiftFpScaler".
    """
    def __init__(self, minimum_oom=5, non_negative=True):
        """Initialize the scaler.

        Args:
            minimum_oom (int, optional): Order-of-magnitude offset added after taking
                the log. Defaults to 5.
            non_negative (bool, optional): If True, clips transformed values below zero
                to zero. Defaults to True.
        """
        self.minimum_oom = minimum_oom
        self.non_negative = non_negative
        self.scaler_name = "LogAndShiftFpScaler"

    def fit(self, fp):
        """Populate ``self.params``; this scaler does not need fitting from data.

        Args:
            fp (xr.DataArray): Unused.
        """
        self.params = {"minimum_oom": self.minimum_oom, "non_negative": self.non_negative}
        pass

    def transform(self, fp):
        """Log-transform and shift the footprint data.

        Args:
            fp (xr.DataArray): Footprint data to transform.

        Returns:
            xr.DataArray: Log-transformed and shifted data, with zeros left as zero.
        """
        #fp = fp.astype("float32", copy=False)
        transformed_fp = np.log10(fp.where(fp > 0)) + self.minimum_oom  # take the log and add the minimum_oom), leave the zeros as is
        if self.non_negative:
            transformed_fp = transformed_fp.where(transformed_fp > 0, 0) # make all values that are zero or below zero (which can happen if the original fp was between 0 and 10**(-minimum_oom)) zero, to avoid having negative values in the transformed fp

        return transformed_fp

    def inverse_transform(self, transformed_fp):
        """Undo the log-and-shift transformation.

        Args:
            transformed_fp (xr.DataArray or np.ndarray): Transformed footprint data.

        Returns:
            xr.DataArray or np.ndarray: Footprint data in the original scale, with
            negative and near-zero values clipped to zero.
        """
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
        """Return the scaler's parameters.

        Returns:
            dict: The scaler parameters set by ``fit``.

        Raises:
            ValueError: If the scaler has not been fitted yet.
        """
        if not hasattr(self, "params"):
            raise ValueError("Scaler has not been fitted yet, so parameters are not available. Please fit the scaler before trying to get parameters.")
        return self.params

class LogAndShiftMeanFpScaler:
    """Footprint scaler.

    Takes the mean of the log of fp data where non-zero, and offsets the log of the
    data by the mean, so that the mean of the logged data is at zero.

    Note: translates across domain sizes.

    Attributes:
        minimum_oom (int): Order-of-magnitude offset applied before taking the log.
        scaler_name (str): Name of the scaler class, "LogAndShiftMeanFpScaler".
        logged_mean (float): Mean of the log-transformed non-zero data, set by ``fit``.
    """
    def __init__(self, minimum_oom=5):
        """Initialize the scaler.

        Args:
            minimum_oom (int, optional): Order-of-magnitude offset added inside the log
                before shifting by the mean. Defaults to 5.
        """
        self.minimum_oom = minimum_oom
        self.scaler_name = "LogAndShiftMeanFpScaler"

    def fit(self, fp):
        """Compute the mean of the log of the non-zero values of ``fp``.

        Args:
            fp (xr.DataArray): Footprint data to compute the logged mean from.
        """
        self.logged_mean = np.mean(np.log10(fp.values.flatten()[fp.values.flatten()>0])).astype("float32")
        self.params = {"logged_mean": self.logged_mean, "minimum_oom": self.minimum_oom}

    def transform(self, fp):
        """Log-transform and shift the footprint data by ``logged_mean``.

        Args:
            fp (xr.DataArray): Footprint data to transform.

        Returns:
            xr.DataArray: Log-transformed and mean-shifted data.
        """
        #fp = fp.astype("float32", copy=False)
        transformed_fp = np.log10(fp+10**(-self.minimum_oom))  # take the log and shift by the logged mean
        transformed_fp = transformed_fp + abs(self.logged_mean)

        return transformed_fp

    def inverse_transform(self, transformed_fp):
        """Undo the log-and-shift-by-mean transformation.

        Args:
            transformed_fp (xr.DataArray or np.ndarray): Transformed footprint data.

        Returns:
            xr.DataArray or np.ndarray: Footprint data in the original scale, with
            negative values clipped to zero.
        """
        original_fp = 10**(transformed_fp - abs(self.logged_mean)) - 10**(-self.minimum_oom)
        if isinstance(original_fp, xr.DataArray):
            original_fp = original_fp.where(original_fp >= 0, 0)
        elif isinstance(original_fp, np.ndarray):
            original_fp = np.where(original_fp >= 0, original_fp, 0)
        return original_fp

    def get_params(self):
        """Return the scaler's parameters.

        Returns:
            dict: The scaler parameters set by ``fit``.

        Raises:
            ValueError: If the scaler has not been fitted yet.
        """
        if not hasattr(self, "params"):
            raise ValueError("Scaler has not been fitted yet, so parameters are not available. Please fit the scaler before trying to get parameters.")
        return self.params


def add_fp_nan_mask(ds, fill_nans=False, fp_var_name="fp", nan_mask_name="fp_nan_mask"):
    """Add a mask of NaN locations in a footprint variable to a dataset.

    Args:
        ds (xr.Dataset): Dataset containing the footprint variable.
        fill_nans (bool, optional): If True, fills NaN values in ``ds`` with 0.
            Defaults to False.
        fp_var_name (str, optional): Name of the footprint variable to check for NaNs.
            Defaults to "fp".
        nan_mask_name (str, optional): Name to give the added mask variable.
            Defaults to "fp_nan_mask".

    Returns:
        xr.Dataset: ``ds`` with the ``nan_mask_name`` variable added (int32, 1 where
        ``fp_var_name`` was NaN), and NaNs filled with 0 if ``fill_nans`` is True.
    """
    if nan_mask_name not in ds:
        ds[nan_mask_name] = np.isnan(ds[fp_var_name]).astype("int32")
    if fill_nans:
        ds = ds.fillna(0)
    return ds

class FootprintDataset:
    """Wrapper around footprint data that applies a footprint scaler.

    The dataset accepts either an ``xarray.DataArray`` or an ``xarray.Dataset``
    containing a variable named ``fp``. ``fit`` fits the underlying scaler,
    ``transform`` returns a Dataset containing ``fp_transformed`` and
    ``fp_original``, and ``inverse_transform`` accepts either the Dataset returned
    by ``transform`` or the transformed DataArray.

    Valid scalers:
        - LogAndShiftMeanFpScaler (default): takes the mean of the log of fp data
          where non-zero, and offsets the log of the data by the mean, so that the
          mean of the logged data is at zero.
        - LogAndShiftFpScaler: takes log of fp data where non-zero, and offsets by
          minimum order-of-magnitude value so its strictly positive.

    Attributes:
        fp (xr.DataArray): The stored footprint data.
        add_nan_mask (bool): Whether ``transform`` adds a NaN mask to its output.
        verbose (bool): Whether to print information during transformation.
        scaler: The scaler instance used to fit/transform the footprints.
        filename (str): Default filename used by ``save_scaler``/``load_scaler``.
        transformed_fp (xr.DataArray): Result of the last call to ``transform`` (only after ``fit`` has been called).
        
    
    Usage:
        >>> fp_dataset = FootprintDataset(fp_dataarray, scaler=LogAndShiftMeanFpScaler)
        >>> fp_dataset.fit()
        >>> transformed_ds = fp_dataset.transform(fp_dataarray)
        >>> original_fp = fp_dataset.inverse_transform(transformed_ds)


        
    """

    def __init__(self, fp, scaler=None, scaler_params={}, add_nan_mask=False, verbose=False, keep_vars=True):
        """Initialize the dataset wrapper.

        Args:
            fp (xr.DataArray or xr.Dataset): Footprint data. If a Dataset, must
                contain a variable named "fp".
            scaler (type, optional): Scaler class to use. If None, defaults to
                ``LogAndShiftMeanFpScaler``. See class docstring for valid scalers.
                Defaults to None.
            scaler_params (dict, optional): Parameters to pass to the scaler when
                initializing it. Defaults to {}.
            add_nan_mask (bool, optional): If True, ``transform`` adds a NaN mask
                variable to its output. Defaults to False.
            verbose (bool, optional): If True, prints information during transformation.
                Defaults to False.
            keep_vars (bool, optional): If True and ``fp`` is an ``xarray.Dataset``, any
                other variables in the Dataset that have the same dimensions as ``fp``
                (``time``, ``lat``, ``lon``) are preserved in the transformed Dataset. Defaults to True.
        """
        self.keep_vars = keep_vars
        fp = self._check_fp_format(fp)

        self.fp = fp.copy()
        self.add_nan_mask = add_nan_mask
        self.verbose = verbose

        if scaler is None:
            self.scaler = LogAndShiftMeanFpScaler(**scaler_params)
        else:
            self.scaler = scaler(**scaler_params)


        self.filename = "fp_scaler.joblib"

    def _check_fp_format(self, fp):
        """Validate and normalize ``fp`` to an ``xr.DataArray``.

        Args:
            fp (xr.DataArray or xr.Dataset): Footprint data. If a Dataset, must
                contain a variable named "fp".

        Returns:
            xr.DataArray: The footprint data as a DataArray.

        Raises:
            ValueError: If ``fp`` is neither a DataArray nor a Dataset containing
                a "fp" variable.
        """
        if isinstance(fp, xr.Dataset):
            if "fp" not in fp.data_vars:
                raise ValueError("fp must be an xarray DataSet with a variable named 'fp', or an xarray DataArray")
            else:
                if self.keep_vars:
                    # save all other vars that are not fp but that also have dims (time, lat, lon) to the transformed dataset
                    self.other_vars = {var: fp[var] for var in fp.data_vars if var != "fp" and set(fp[var].dims) == set(["time", "lat", "lon"])}

                fp = fp["fp"]
        elif not isinstance(fp, xr.DataArray):
            raise ValueError("fp must be an xarray DataSet with a variable named 'fp', or an xarray DataArray")
        return fp

    def fit(self):
        """Fit the underlying scaler on ``self.fp``."""
        self.scaler.fit(self.fp)

    def transform(self, fp):
        """Transform footprint data and package it into a Dataset.

        Args:
            fp (xr.DataArray or xr.Dataset): Footprint data to transform, in the
                same format accepted by the constructor.

        Returns:
            xr.Dataset: Dataset with ``fp_transformed`` and ``fp_original`` variables
            (plus ``fp_nan_mask`` if ``self.add_nan_mask`` is True), chunked by
            ``time`` and with an ``idx`` coordinate along ``time``.
        """
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

        if self.keep_vars and hasattr(self, "other_vars"):
            for var, data in self.other_vars.items():
                ds[var] = data


        # add coord linked to time index with idx
        ds = ds.chunk({"time": 1})
        ds = ds.assign_coords(idx=("time", list(range(len(ds.time)))))
        
        self.transformed_fp = transformed_fp

        return ds

    def fit_transform(self):
        """Fit the scaler on ``self.fp`` and then transform ``self.fp``.

        Returns:
            xr.Dataset: Dataset with ``fp_transformed`` and ``fp_original`` variables,
            as returned by ``transform``.
        """
        self.fit()
        return self.transform(self.fp)

    def inverse_transform(self, transformed_fp):
        """Undo the scaler's transformation.

        Args:
            transformed_fp (xr.Dataset, xr.DataArray, or np.ndarray): Either the
                Dataset returned by ``transform`` (from which ``fp_transformed`` is
                extracted), or the transformed data directly.

        Returns:
            xr.DataArray or np.ndarray: Footprint data in the original scale.

        Raises:
            ValueError: If ``transformed_fp`` is not a Dataset with a
                ``fp_transformed`` variable, a DataArray, or a numpy array.
        """
        if isinstance(transformed_fp, xr.Dataset) and "fp_transformed" in transformed_fp.data_vars:
            transformed_fp = transformed_fp["fp_transformed"]
        elif not isinstance(transformed_fp, xr.DataArray) and not isinstance(transformed_fp, np.ndarray):
            raise ValueError("transformed fp must be an xarray DataArray or numpy array!")
        return self.scaler.inverse_transform(transformed_fp)

    def save_scaler(self, path):
        """Save the fitted scaler to disk with joblib.

        Args:
            path (str or Path): Directory to save the scaler bundle into; the file
                is written to ``path / self.filename``.
        """
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
        """Load a previously-saved scaler from disk and assign it to ``self.scaler``.

        Args:
            path (str or Path): Directory containing the saved scaler bundle; the
                file is read from ``path / self.filename``.
        """
        print("warning this is untested")
        bundle = joblib.load(Path(path) / self.filename)
        #print(f"Loaded {bundle['scaler_type']} saved at {bundle['saved_at']}")
        self.scaler = bundle["scaler"]


def make_inputs_batcher(inputs, batch_size=10, flatten=False):
    """Build an xbatcher BatchGenerator for the input meteorological fields.

    Args:
        inputs (xr.DataArray): DataArray of size (fp_time, lat, lon, variable_name).
        batch_size (int, optional): Batch size. Defaults to 10.
        flatten (bool, optional): Whether to stack lat/lon into a single flat_lat_lon
            dimension. Defaults to False.

    Returns:
        xbatcher.BatchGenerator: Batch generator over ``inputs``.

    Raises:
        ValueError: If ``inputs`` is not an ``xr.DataArray`` with dims
            (fp_time, lat, lon, variable_name).
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
        preload_batch=False,
    )
    return X_bgen


def make_fps_batcher(fps, batch_size=10, flatten=False):
    """Build an xbatcher BatchGenerator for the footprint labels.

    Args:
        fps (xr.DataArray or xr.Dataset): DataArray of size (time, lat, lon), or
            Dataset containing one or more footprint variables of size (time, lat, lon).
        batch_size (int, optional): Batch size. Defaults to 10.
        flatten (bool, optional): Whether to stack lat/lon into a single flat_lat_lon
            dimension. Defaults to False.

    Returns:
        tuple:
            - y_bgen (xbatcher.BatchGenerator): Batch generator over ``fps``.
            - fps_labels (list): Footprint variable name(s).

    Raises:
        ValueError: If ``fps`` is neither an ``xr.Dataset`` nor an ``xr.DataArray``.
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
    """Trim the last N timepoints from inputs and fps so that the number of
    timepoints is divisible by batch_size.

    Args:
        inputs (xr.DataArray): DataArray with a 'fp_time' dimension.
        fps (xr.DataArray or xr.Dataset): DataArray or Dataset with a 'time' dimension.
        batch_size (int): Batch size to trim the time dimensions down to a multiple of.

    Returns:
        tuple: ``(inputs, fps)`` trimmed along their respective time dimensions.
    """
    n = inputs.sizes["fp_time"]
    remainder = n % batch_size
    if remainder != 0:
        inputs = inputs.isel(fp_time=slice(None, n - remainder))
        fps = fps.isel(time=slice(None, n - remainder))
    return inputs, fps


def make_dataloader(inputs, fps, batch_size=10, randomize=False, random_seed=42, dataloader_params=None, flatten=False):
    """Build a PyTorch dataloader from input and footprint datasets using xbatcher.

    The inputs must have dimensions (fp_time, lat, lon, variable_name). The footprints
    must have matching ``time`` length and either be a DataArray with dims ``(time, lat, lon)``
    or a Dataset containing one or more footprint variables. When a Dataset is passed, the
    footprints are stacked along a new ``variable_name`` dimension and the returned labels are
    the stacked variable tuples; when a DataArray is passed, the returned labels contain the single footprint variable name.

    If ``randomize`` is True, the time dimension is permuted before batching. ``random_seed``
    controls the permutation. ``dataloader_params`` is forwarded to ``torch.utils.data.DataLoader``.

    Args:
        inputs (xr.DataArray): DataArray of size (fp_time, lat, lon, variable_name)
            from ``get_square_satellite_inputs`` or a similar function.
        fps (xr.DataArray or xr.Dataset): DataArray, or Dataset with variable
            "fp_transformed", of size (time, lat, lon).
        batch_size (int, optional): Batch size for the dataloader. Defaults to 10.
        randomize (bool, optional): Whether to shuffle the dataset along the time
            dimension. Recommended for training and not for testing. Defaults to False.
        random_seed (int, optional): Seed for reproducibility of the shuffling when
            ``randomize`` is True. Defaults to 42.
        dataloader_params (dict, optional): Additional parameters to pass to the
            PyTorch DataLoader. Defaults to None, which uses a predefined set of
            parameters (prefetch_factor, num_workers, persistent_workers,
            multiprocessing_context).
        flatten (bool, optional): Whether to flatten the input tensors in the
            lat-lon dimension before passing them to the model. Defaults to False.

    Returns:
        tuple:
            - dataloader (torch.utils.data.DataLoader): DataLoader that yields batches
              of ``(inputs, fps)``. Inputs has shape (batch_size, lat, lon, variable_name)
              or (batch_size, flat_lat_lon, variable_name) if flatten is True. If passing
              a DataArray as fps, fps has shape (batch_size, lat, lon) or
              (batch_size, flat_lat_lon) if flatten is True. If passing a Dataset as fps,
              fps has shape (batch_size, lat, lon, variable_name) or
              (batch_size, flat_lat_lon, variable_name) if flatten is True.
            - fps_labels (list): Footprint variable labels.

    Raises:
        ValueError: If ``inputs`` and ``fps`` have mismatched ``fp_time``/``time``
            lengths, mismatched lat/lon dimensions, or unsupported types.
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
    chunk_size=None,
):
    """Like cut_satellite_met but handles all time_deltas in one call.

    Computes the union of required met timestamps across all time_deltas,
    loads that minimal set from dask once, computes the spatial crop structure
    once, then builds per-delta cropped datasets entirely from in-memory arrays.

    Args:
        met_source (xr.Dataset): Full-domain met, already filtered to desired levels
            and variables.
        fp (xr.Dataset): Footprint dataset with ``release_lat``, ``release_lon``, ``time``.
        metsize (int): Size of the square to crop (must be even).
        time_deltas (list[int]): Hours to shift backwards (0 = no shift).
        pad_mode (str, optional): "nans" or "edge" — passed to ``_pad_domain``.
            Defaults to "nans".
        add_wind_direction (bool, optional): Compute ``wind_angle`` and ``wind_speed``
            from ``x_wind``/``y_wind``. Defaults to True.
        closest_tolerance (str, optional): Max allowed distance for nearest-timestamp
            lookup, e.g. "4h". Defaults to "4h".
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        load_into_memory (bool, optional): Whether to load the met data into memory before cropping. Good for performance when the dataset is small enough to fit in memory, or when the chunks are not fit for processing. Defaults to False.
        interp_to (str or None, optional): If None (default), each fp timestamp is
            matched to the nearest met timestamp (existing behaviour). If a pandas
            offset string such as ``"1h"`` or ``"15min"``, each target time is rounded
            to that resolution and the met is linearly interpolated to that rounded
            time from the two bracketing met timestamps. fp times whose rounded target
            falls outside the met record are treated as NaN and dropped, same as the
            nearest case. Defaults to None.
        chunk_size (int or None, optional): If set (and ``load_into_memory`` is
            False), the crop is done in blocks of this many samples: each block of met
            is read into memory first (a plain parallel zarr read), then cropped in
            numpy. This keeps the met source lazy (only a block's timestamps are read
            at a time), bounds peak memory to one block's read, and avoids building the
            dask vindex graph that otherwise causes large-chunk/large-graph warnings
            and heavy GC. Larger values mean fewer, bigger reads. Ignored when
            ``load_into_memory`` is True (the full met is already numpy, so the
            per-block load is skipped). When None and ``load_into_memory`` is False the
            crop is left lazy as a single chunk (improving that path is future work).
            Defaults to None.

    Returns:
        tuple:
            - results (dict[int, xr.Dataset]): {delta: cropped dataset} with dims
              ``(fp_time, lat, lon[, levels])``. No ``time_delta`` dimension — caller
              handles the concat. Contains ``lat_coords`` and ``lon_coords`` (same
              values across deltas).
            - nan_idxs_per_delta (dict[int, np.ndarray]): fp_time values that could
              not be interpolated for each delta.

    Raises:
        ValueError: If ``metsize`` is not even.
    """
    if metsize % 2 != 0:
        raise ValueError("metsize must be even")
    half = metsize // 2

    fp_times = fp.time.values
    tol = pd.Timedelta(closest_tolerance)
    met_time_index = met_source.indexes["time"]

    # --- Phase 2: timestamp lookup for every delta ---
    nearest_info = {}
    all_unique_times = set()

    for delta in time_deltas:
        target_times = pd.DatetimeIndex(
            fp_times if delta == 0
            else pd.DatetimeIndex(fp_times) - pd.Timedelta(f"{delta}h")
        )

        if interp_to is None:
            # Existing behaviour: snap each target to the nearest met timestamp.
            nearest = met_time_index.get_indexer(target_times, method="nearest", tolerance=tol)
            nan_mask = nearest == -1
            nearest_safe = np.where(~nan_mask, nearest, 0)
            nearest_timestamps = met_time_index.values[nearest_safe]
            # nan_idxs reported in fp_time space (undo the delta shift)
            nan_idxs = (target_times[nan_mask] + pd.Timedelta(f"{delta}h")).to_numpy()

            nearest_info[delta] = {
                "lookup_times": nearest_timestamps,
                "nan_mask": nan_mask,
                "nan_idxs": nan_idxs,
            }
            all_unique_times.update(nearest_timestamps[~nan_mask])

        else:
            # interp_to mode: round target to the requested resolution, then
            # find the floor/ceiling met timestamps that bracket each rounded time.
            interp_targets = target_times.round(interp_to)
            floor_idxs = met_time_index.get_indexer(interp_targets, method="ffill")
            ceil_idxs = met_time_index.get_indexer(interp_targets, method="bfill")
            nan_mask = (floor_idxs == -1) | (ceil_idxs == -1)
            nan_idxs = (target_times[nan_mask] + pd.Timedelta(f"{delta}h")).to_numpy()

            # For nan positions use the first valid interp target as a safe fallback
            # (those rows are dropped later via all_nan_idxs in the caller).
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
            # Collect the bracket timestamps that will need to be loaded
            all_unique_times.update(met_time_index.values[floor_idxs[~nan_mask]])
            all_unique_times.update(met_time_index.values[ceil_idxs[~nan_mask]])

    # --- Phase 2b: select the union of required timestamps ---
    # The met is a yearly Zarr store natively chunked {time:1, lat:-1, lon:-1,
    # levels:3}, which is ideal for this scattered per-timestamp selection — no
    # rechunking needed. Trade-off: computing any batch loads all unique
    # timestamps at once.
    all_unique_times_sorted = sorted(all_unique_times)

    if verbose:
        print(f"Selecting {len(all_unique_times_sorted)} unique met timestamps ")
        # print the first three
        print(f"First few unique timestamps: {all_unique_times_sorted[:3]} ...")

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        met_loaded = met_source.sel(time=list(all_unique_times_sorted))
    if load_into_memory:
        print("Loading selected met data into memory...")
        met_loaded = met_loaded.compute()

    # --- Phase 3: spatial structure — computed once, shared across all deltas ---
    ## the bug was here - but am now skipping over release_idxs, and just using the lat_coords and lon_coords generated when cropping the footprints
    release_idxs = _get_release_idxs(fp, domain_lats=met_source.lat.values, domain_lons=met_source.lon.values)
    met_loaded, _, _, _ = _pad_domain(
        met_loaded, fp, release_idxs, half, pad_mode, verbose=verbose
    )


    lat_ds = fp.lat_coords
    lon_ds = fp.lon_coords

    # --- Phase 2c (interp_to only): linearly interpolate the padded bracket data
    # to the rounded target times.  Padding is spatial-only so order doesn't matter.
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

        # Spatial crop. sel(lat=lat_ds, lon=lon_ds) uses 2-D (time, lat)/(time, lon)
        # indexers. On a dask array that becomes a vindex reshape whose graph embeds
        # the per-sample index arrays — the source of the "large chunk"/"large graph"
        # warnings (and, with split_large_chunks, a dask 2024.5.0 crash).
        n = len(pos)
        if bool(chunk_size) and not load_into_memory:
            # Met is lazy and chunk_size is set: read the met in chunk_size-sample
            # blocks and crop each block in numpy. Reading a block first is a plain,
            # parallel zarr read (no vindex); cropping the numpy block then builds no
            # dask graph at all. This bounds peak memory to one block's read and avoids
            # the large-graph warnings entirely.
            blocks = []
            for start in range(0, n, chunk_size):
                sl = slice(start, min(start + chunk_size, n))
                md = met_for_crop.isel(time=pos[sl]).assign_coords(time=fp_times[sl]).compute()
                blocks.append(
                    md.sel(lat=lat_ds.isel(time=sl),
                           lon=lon_ds.isel(time=sl), method="nearest")
                )
            cropped = xr.concat(blocks, dim="time") if len(blocks) > 1 else blocks[0]
        else:
            # Met is already in memory (load_into_memory) or intentionally left lazy
            # (no chunk_size): a single crop, no per-block loading needed. When lazy
            # this still builds the vindex graph and may warn — improving that path is
            # future work.
            md = met_for_crop.isel(time=pos).assign_coords(time=fp_times)
            cropped = md.sel(lat=lat_ds, lon=lon_ds, method="nearest")
        # store the lat and lon values in cropped as coordinates before reassigning the lat and lon coordinates to be the index values (0 to metsize-1)
        cropped = cropped.assign_coords(lat_coords=(("time", "lat"), cropped.lat.values), lon_coords=(("time", "lon"), cropped.lon.values))

        cropped = cropped.assign_coords(lat=np.arange(metsize), lon=np.arange(metsize))


        if add_wind_direction:
            try:
                if "x_wind" in cropped.data_vars and "y_wind" in cropped.data_vars:
                    cropped["wind_angle"] = np.arctan2(-cropped.x_wind, -cropped.y_wind)
                    cropped["wind_speed"] = np.sqrt(cropped.x_wind**2 + cropped.y_wind**2)
            except Exception as e:
                warnings.warn(f"Could not compute wind variables for delta={delta}: {e}")

        # Swap time → fp_time (matching convention in rest of pipeline)
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
    chunk_size=None,
):
    """Optimised version of ``get_square_satellite_inputs``.

    Differences from v1:
    - ``met_variables`` is a plain list of variable names (not a dict).
    - ``met_levels`` is a single list of levels shared by all atmospheric variables.
      Surface variables (those without a "levels" dimension in the met dataset)
      are detected automatically.
    - Met levels and variables are filtered once; all time_deltas share a single
      dask ``compute()`` call, eliminating repeated graph construction and I/O.
    - Spatial structure (release indices, domain padding, crop index arrays) is
      computed once and reused across all time_deltas.

    Args:
        data (LoadSquareSatelliteData): Must have ``met_processed=True`` and
            ``dataset_format='square'``.
        met_variables (list[str]): Variable names to extract, e.g.
            ``["x_wind", "y_wind", "atmosphere_boundary_layer_thickness"]``. Include
            ``"wind_angle"`` / ``"wind_speed"`` to derive them from x/y wind.
        met_levels (list[int]): Pressure/model levels to extract for all atmospheric
            variables.
        time_deltas (list[int] or None, optional): Hours to shift backwards, e.g.
            ``[6, 12]``. 0 is added automatically unless ``add_timedelta_zero=False``.
            Defaults to None.
        static_variables (list[str] or None, optional): Static fields to append, e.g.
            ``["topog", "lat_coords", "lon_coords"]``. Defaults to None.
        verbose (bool, optional):  Whether to print verbose output. Defaults to True.
        add_timedelta_zero (bool, optional): Whether to add a time delta of 0. Defaults to True.
        add_wind_direction (bool, optional): Whether to compute wind direction and speed. Defaults to False.
        load_into_memory (bool, optional): Whether to load the met data into memory before cropping. Good for performance when the dataset is small enough to fit in memory, or when the chunks are not fit for processing. Defaults to False.
        interp_to (str or None, optional): If None (default), each fp timestamp is
            matched to the nearest met timestamp. If a pandas offset string such as
            ``"1h"`` or ``"15min"``, each target time is rounded to that resolution
            and the met is linearly interpolated between the two bracketing met
            timestamps. Defaults to None.
        chunk_size (int or None, optional): Only used when ``load_into_memory`` is
            False. If set, the spatial crop is done in blocks of this many samples and
            each block is computed into memory as it is produced (the met source stays
            lazy and is read per block). This bounds peak memory and avoids the large
            task graph that otherwise causes a "large chunk" warning and heavy GC.
            Larger values mean fewer, bigger reads; it is independent of the training
            ``batch_size`` (the dataloader rechunks separately). When None, the crop is
            left lazy as a single chunk (future work). Defaults to None.

    Returns:
        tuple:
            - concatenated_inputs (xr.DataArray): Shape
              ``(fp_time, lat, lon, variable_name)`` where ``variable_name`` is a
              MultiIndex of ``(variable, level, time_delta)`` tuples.
            - data (LoadSquareSatelliteData): Updated object — fp_time indices with
              failed met interpolation are removed.

    Raises:
        ValueError: If ``met_variables`` is not a list, ``met_levels`` is not a list,
            or no requested variables/levels are available after filtering.
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
        chunk_size=chunk_size,
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
            elif "earth" in var:
                if var in static_variables_functions:
                    static_ds = static_variables_functions[var](data.fp_xr, static_ds)
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
    
    len_inputs_before = len(concatenated_inputs.fp_time)

    if len(concatenated_inputs.fp_time) < len_inputs_before:
        warnings.warn(
            f"Dropped {len_inputs_before - len(concatenated_inputs.fp_time)} duplicate fp_time entries after concatenating inputs. "
            "The duplicates have been dropped, but you may want to investigate the underlying met timestamp issues for those fp_time entries."
        )
        len_fp_before = len(data.fp_xr.time)
        data.fp_xr = data.fp_xr.sel(time=concatenated_inputs.fp_time.values)
        print(f"Filtered fp_xr to keep only {len(data.fp_xr.time)} unique time steps (from the original number of samples {len_fp_before})")

    # Report the chunking of the final inputs so we can tell whether the lazy crop
    # produced chunk_size-sized fp_time blocks or collapsed into one giant chunk.
    if verbose:
        chunks = getattr(concatenated_inputs.data, "chunks", None)
        if chunks is None:
            print(f"Inputs are numpy-backed (not chunked); shape={concatenated_inputs.shape}")
        else:
            fp_axis = concatenated_inputs.get_axis_num("fp_time")
            print(f"Inputs chunking: dims={concatenated_inputs.dims}, chunks={chunks}")
            print(f"  -> fp_time chunk blocks: {chunks[fp_axis]}")




    return concatenated_inputs, data