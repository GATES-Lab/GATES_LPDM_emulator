"""
Data loading functions, loading footprints, meteorology and topography data.
Can be fed into the GATES model as a dataset, or used for other models

author: Elena Fillola @elenafillo
"""

import gates
import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os
import copy
import warnings
from pathlib import Path
import re

import cartopy.crs as ccrs
import cartopy

from .load_data_helper_funs import haversine, select_met_levels,select_met_variables,get_static_variables_functions,convert_flux_units

from gates.config import get_config


def _rename_latlon(ds):
    """Rename latitude→lat and longitude→lon if those names are present as dims."""
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds


def _wrap_longitudes(ds):
    """Convert longitudes from 0–360 to –180 to 180 if any values exceed 180.
    Detects the coordinate name automatically (lon or longitude).
    Sorts by the longitude coordinate after wrapping so the grid stays monotonic.
    """
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords), None)
    if lon_name is None or float(ds[lon_name].max()) <= 180:
        return ds
    ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
    return ds.sortby(lon_name)


def detect_fp_format(fp_datadir):
    """
    Return the footprint format ("zarr" or "nc") implied by a path or glob pattern.
    Anything that does not end in .zarr is treated as NetCDF, so patterns using
    other NetCDF suffixes keep working.
    """
    return "zarr" if str(fp_datadir).endswith(".zarr") else "nc"


def load_fps(fp_datadir, verbose=False, chunk=True, parallel_loading=False, drop_variables_except=None, bad_files_list=None):
    """
    Load footprints from datadir, in either NetCDF or Zarr format. The format is
    taken from the suffix of fp_datadir: a pattern ending in .zarr is loaded as
    yearly Zarr stores, anything else as monthly NetCDF files.

    Args:
        - fp_datadir (str or Path): string or path pointing to the directory with footprints to load,
        including special characters (eg "/path/to/footprints/*.nc", or "/path/to/footprints/*2020*.nc",
        or "/path/to/footprints/*2020*.zarr")
        Note that fp_datadir is passed directly to glob, so it needs to specify filetype (i.e. finish with .nc or .zarr)
        - verbose: if True, prints out the steps throughout the data loading process
        - chunk: if True, loads the data with dask chunking, which can help with memory issues but can cause some problems with certain files. NetCDF only.
        - parallel_loading: if True, uses dask to load the files in parallel, which can speed up loading but can cause some problems with certain files. Parallel loading can only be used if chunk=True. NetCDF only.
        - drop_variables_except: if specified, only keeps the listed variables in the dataset.
        - bad_files_list: if specified, a list of files that are known to be problematic and should be skipped. If None, fetches list from config. Pass an empty list to not skip any files. NetCDF only.
    Returns:
        - fp_data_full: xarray dataset with the footprints specified in the fp_datadir, opened correctly
    """
    fp_format = detect_fp_format(fp_datadir)

    if fp_format == "zarr":
        if not chunk or parallel_loading or bad_files_list:
            warnings.warn(
                "chunk, parallel_loading and bad_files_list only apply to NetCDF footprints "
                "and are ignored when loading Zarr stores: the stores are written with their "
                "own chunks, and the bad-file workarounds are applied at conversion time."
            )
        fp_data_full = _load_fps_zarr(fp_datadir, verbose=verbose)
    else:
        fp_data_full = _load_fps_netcdf(
            fp_datadir, verbose=verbose, chunk=chunk,
            parallel_loading=parallel_loading, bad_files_list=bad_files_list,
        )

    if drop_variables_except is not None:
        vars_to_drop = [var for var in fp_data_full.data_vars if var not in drop_variables_except]
        fp_data_full = fp_data_full.drop_vars(vars_to_drop)

    return fp_data_full


def _load_fps_zarr(fp_datadir, verbose=False):
    """
    Load footprints from one or more yearly Zarr stores, concatenating along time.
    ``fp_datadir`` is a glob pattern (e.g. *BRAZIL*SOUTHAMERICA_2016*.zarr) resolved
    to a list of stores; usually a single year, but a year pattern such as "201[4-5]"
    resolves to several stores opened together.

    The Zarr stores are written by convert_fps_to_zarr(), which already renames
    latitude/longitude -> lat/lon, sorts and de-duplicates timestamps, and applies
    the bad-file workarounds. So none of the NetCDF-era preprocessing is needed here.
    """
    fp_datadir = Path(fp_datadir)

    if not os.path.exists(fp_datadir.parents[0]):
        raise ValueError(f"Specified directory does not exist:\n {fp_datadir}")

    fp_stores = sorted(glob.glob(str(fp_datadir)))
    if len(fp_stores) == 0:
        raise ValueError(
            f"No matching Zarr stores found in the specified directory:\n {fp_datadir} \n"
            "Check that the path is correct and that there are stores matching the pattern."
        )

    if verbose: print(f"Loading footprints from {len(fp_stores)} zarr store(s): {fp_stores}")

    return xr.open_mfdataset(
        fp_stores,
        engine="zarr",
        concat_dim="time",
        combine="nested",
        consolidated=True,
    )


def _load_fps_netcdf(fp_datadir, verbose=False, chunk=True, parallel_loading=False, bad_files_list=None):
    """
    Load footprints from monthly NetCDF files, using workaround if problematic files are encountered. Will throw an error if ANY of the specified files is problematic and NOT on the bad_files list
    note that the list of problematic files is currently updated manually!

    See load_fps() for the arguments.

    Potential Improvements:
        - Add capability to ignore any files that couldn't be opened, and return only the successful files
    """
    fp_datadir = Path(fp_datadir)
    print("WITH CHUNKING -")
    try:
        if chunk:
            #time_chunk = 25
            #chunk_args = {"chunks" : {"time": time_chunk}, "parallel": True}
            chunk_args = {"chunks":{"lat":-1, "lon":-1, "time":"auto"}, "parallel": parallel_loading}
        else:
            chunk_args = {}
        # check that path exists
        if not os.path.exists(fp_datadir.parents[0]):
            raise ValueError(f"Specified directory does not exist:\n {fp_datadir}")
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # check that there are any files to open
            if len(glob.glob(str(fp_datadir)))==0:
                raise ValueError(f"No matching files found in the specified directory:\n {fp_datadir} \nCheck that the path is correct and that there are files matching the pattern.")
            # attempt to load dataset of multiple files thfe standard way
            fp_data_full = xr.open_mfdataset(sorted(glob.glob(str(fp_datadir))), combine='by_coords', **chunk_args)

    except Exception as e:
        # some files have small errors in format that prevent xr from concatenating and opening together. This is a workaround to open those separately. This list only contains known files and could be more! can add manually whenever you encounter one 
        # the bad_files contains full paths, first the full path is checked 
        if verbose: print("there was an error opening the dataset. checking if any of the files are in the bad files list")
        fp_files = sorted(glob.glob(str(fp_datadir)))
        path = os.path.split(str(fp_datadir))[0] + "/"
        filenames = [os.path.split(x)[1] for x in fp_files]

        if bad_files_list is None:
            cfg = get_config()
            bad_files_list = cfg.bad_fp_files.copy()
        elif len(bad_files_list) == 0:
            if verbose: print("you have passed an empty list for bad_files_list, so no files will be skipped. If you are encountering errors opening files, check that the problematic files are in the bad files list")
        if not isinstance(bad_files_list, list):
            raise ValueError("bad_files_list should be a list of filenames (not full paths), or None to load from config. Pass an empty list to not skip any files.")

        # remove any files from the list that were in the bad files list

        
        without_bad_files = list(set(filenames) - set(bad_files_list))
        without_bad_files = [path+f for f in without_bad_files]
        bad_files_list = [path+f for f in bad_files_list]

        if len(without_bad_files) == len(fp_files):
            print("There was a problem opening files!! compared against the list of known bad files and couldnt find a match")
            print("check if there has been a problem, or maybe a new bad file needs to be added to the list!")

        if len(without_bad_files) < len(fp_files):
            if verbose: print("at least one of the files was in the bad files list, opening with workaround")
            # error arises because fp file for Brazil Nov 2015 has non-monotonic timestamps, use workaround
            # error arises because fp file for Sahara Nov 2014 has non-monotonic timestamps, use workaround
            # also some Sahara 2015 files are missing mean_age_particles_ variable - drop and concat
            # Add clauses here to catch other known exceptions
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                # load non-problematic arrays all together
                most = xr.open_mfdataset(sorted(without_bad_files), **chunk_args)
                bad_arrays = []
                for badfile in bad_files_list:
                    if badfile in fp_files:
                        print("loading bad file with workaround:", badfile)
                        # load each bad file separately
                        f_bad = xr.open_mfdataset(badfile, **chunk_args)
                        #f_bad = xr.open_mfdataset(badfile)
                        if "NORTHAFRICA_2015" in badfile:
                            try:
                                f_bad = f_bad.drop(["mean_age_particles_n", "mean_age_particles_e", "mean_age_particles_w", "mean_age_particles_s"])        
                            except Exception as e:
                                print("something went wrong trying to load the bad North Africa 2015 files")
                                print(e)
                        bad_arrays.append(f_bad)

                # concatenate all the good files with the bad ones along the time dimension
                fp_data_full = xr.concat([most]+bad_arrays, dim="time")
        else:
            print("there was a problem", e)

    fp_data_full = _rename_latlon(fp_data_full)
    fp_data_full = fp_data_full.sortby('time')

    return fp_data_full


def remove_duplicates(ds, dim="longitude"):
    """
    Remove duplicate values along a specified dimension in an xarray Dataset.
    
    Parameters:
    - ds: xarray Dataset
    - dim: Dimension along which to check for duplicates (default is "longitude")
    
    Returns:
    - xarray Dataset with duplicates removed
    """
    if isinstance(dim, (list, tuple)):
        dims = list(dim)
    else:
        dims = [dim]

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        for d in dims:
            if d in ds.dims:
                ds = ds.drop_duplicates(d)
    return ds


def preprocess_met_data(ds, duplicate_dim="longitude"):
    """
    Remove duplicate values along a specified dimension in an xarray Dataset.
    
    Parameters:
    - ds: xarray Dataset
    - dim: Dimension along which to check for duplicates (default is "longitude")
    
    Returns:
    - xarray Dataset with duplicates removed
    """
    ds = _rename_latlon(ds)
    if duplicate_dim in ds.dims:
        ds = remove_duplicates(ds, dim=duplicate_dim)
    elif duplicate_dim == "longitude" and "lon" in ds.dims:
        ds = remove_duplicates(ds, dim="lon")
    return ds


class LoadBaseSatelliteData:
    """
    Parent class for loading satellite data.
    Loads footprint, meteorology, and topography/landcover data for a given domain and time period.

    main inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - month: str in format "01" for January etc, None if loading a whole year
        - region: region identifyer, as a string. Default is Brazil. Current set-up has regions "BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA"
        (note - Brazil is a subset of South America!)
        - domain: Domain related to the region, used for file search (due to existing filenaming conventions). Set-up regions ("BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA") have a default domain, all others need domain passed
        NOTE: Update dict of region-domain using config rather than being hard-coded
    -  load_everything: bool, if True, loads all data (footprints, meteorology and topography) at once when initializing the class. If False, only loads footprints, and meteorology and topography can be loaded later with the load_meteorology() and load_topog() functions. Default is False 

    Paths:
        The paths to the files are by default loaded from the config file, where the file structure is expected to be path/to/footprints/domain/region_*domain*_yearmonth.nc for the NetCDF footprints, path/to/footprints_zarr/domain/region_*domain*_year.zarr for the Zarr footprints, and path/to/meteorology/domain/domain_Met_year.zarr for the meteorology.
        The config paths are superceded by passing the paths as arguments, as strings or path objects:
        - fp_datadir for the NetCDF footprints, which should point to the folder containing the files for each month/year with format example_name_yearmonth.nc (eg brazil_201601.nc), so that the date can be automatically added.
        - fp_zarr_datadir for the Zarr footprints, which should point to the folder containing the yearly stores with format example_name_year.zarr (eg brazil_2016.zarr), so that the year can be automatically added.
        - fp_format: "auto" (default), "zarr" or "nc". "auto" looks for Zarr stores first and falls back to NetCDF, so both formats can coexist and Zarr wins for a year available in both. Passing only one of fp_datadir/fp_zarr_datadir also selects that format. "zarr" or "nc" force a format.
        - met_args = {"met_datadir": "path/to/meteorology/domain/domain_Met_"}, which should point to the folder containing the files for each month/year with format example_name_yearmonth.nc (eg brazil_201601.nc), so that the date can be automatically added.
        - topog_args = {"topog_path": "path/to/topography/file.nc", "landcover_path": "path/to/landcover/file.nc"}, which should point to the specific files for topography and landcover. These files will be interpolated to the same resolution and domain as the footprints, so they can be from a different source and with a different original resolution.

    other inputs
        - freq: int, frequency of the data to load. 
        freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset for testing
        - sampling_mode: str, default "regular".
        if "regular", subsamples footprints regularly (e.g. one in every two, sequentially with freq=2). if random, subsamples N/freq footprints randomly (where N is the total number of footprints)
        - freq_offset: int
        if using sampling_mode="regular", offsets the start of the regular sampling, e.g. freq=2 and freq_offset=0 will sample even footprints, and freq_offset=1 will sample uneven footprints
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. 
        Applied after sampling with freq (or pass freq=1 to load all footprints)
        - verbose: if True, prints out the steps throughout the data loading process
        - lazy_load: if True, does not load the met data into memory (lazy array), False, loads the met data into memory.
        - cfg: a gates.config.Config object containing the config values. If None, the config will be loaded from the config file. If passing as an argument, this supercedes loading from the config file, and allows you to pass a custom config object with custom paths and settings.
    
    met_args:
        see load_meteorology()
    load_meteorology: Loads meteorology files and optionally subsets levels/variables.
    topog_args:
        see load_topog()
        load_topog: Loads topography and landcover and interpolates both to footprint grid.

    Attributes (if using load_everything=True, otherwise only fp_data_full is initialised):
        - data.fp_data_full: xarray dataset with the footprints specified in the fp_datadir, opened correctly, and subsampled according to the freq and sampling_mode parameters. Dimensions are time, lat and lon.
        - data.met_file: xarray dataset with the meteorology data, opened correctly and with selected levels and variables if specified. Dimensions are time, levels, lat and lon.
        - data.topog_file: xarray dataset with the topography data, opened correctly and interpolated to the same grid as the footprints. Dimensions are lat and lon.
        - data.landcover_file: xarray dataset with the landcover data, opened correctly and interpolated to the same grid as the footprints. Dimensions are lat and lon.


    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, freq=1, freq_offset=0, verbose = False, sampling_mode="regular", fp_datadir = None, fp_zarr_datadir = None, fp_format="auto", load_everything=False, met_args={}, topog_args={}, cfg=None, parallel_loading=False, load_fps_in_mem=True):

        self.dataset_format = "base"
        self._common_init(
            year, region=region, month=month, domain=domain,
            freq=freq, freq_offset=freq_offset, sampling_mode=sampling_mode,
            fp_datadir=fp_datadir, fp_zarr_datadir=fp_zarr_datadir, fp_format=fp_format,
            met_args=met_args, topog_args=topog_args, cfg=cfg,
            verbose=verbose, parallel_loading=parallel_loading,
        )

        #### load footprint (fp) data
        if verbose: print("---- LOADING FOOTPRINTS")
        self._load_footprints(self.fp_datadir, load_fps_in_mem=load_fps_in_mem)

        self.met_processed = False

        
        self.padding=None

        if load_everything:
            self.met_file = self.load_meteorology(**self.met_args)
            self.topog_file, self.landcover_file = self.load_topog(**self.topog_args)

        
        if verbose: print("---- All done!")

    def _common_init(self, year, region="BRAZIL", month=None, domain=None,
                     freq=1, freq_offset=0, sampling_mode="regular",
                     fp_datadir=None, fp_zarr_datadir=None, fp_format="auto",
                     met_args={}, topog_args={}, cfg=None,
                     verbose=False, parallel_loading=False,
                     data_type="satellite"):
        """
        Shared setup for all footprint loaders. Resolves the config object,
        region/domain, year/month/date, footprint/met/topog paths and the
        subsampling parameters.

        No data is loaded here: each loader (base, square, receptor) runs its own
        footprint/met/topog loading and processing sequence after calling this,
        because the order and steps differ between them. The footprints are put onto
        the canonical ``sample_id`` layout later, in _prepare_samples.
        """
        self.data_type = data_type
        self.verbose = verbose
        self.parallel_loading = parallel_loading

        # config is only needed when a path or the domain has to be looked up
        needs_cfg = (
            domain is None
            or (fp_datadir is None and fp_zarr_datadir is None)
            or "met_datadir" not in met_args
            or "topog_path" not in topog_args
        )
        if cfg is None:
            if needs_cfg:
                try:
                    self.cfg = get_config()
                except Exception as e:
                    raise FileNotFoundError(f"{e}\nTo avoid needing the config file, pass all necessary paths as arguments when initializing the class. Check the function inputs for details.")
            else:
                self.cfg = None
        elif isinstance(cfg, gates.config.Config):
            self.cfg = cfg
        else:
            raise ValueError("cfg should be either None or a gates.config.Config instance. If None, the config will be loaded from the config file. Check your input!")

        #### check domains
        self.region = region
        if domain is None:
            self.domain = self._get_domain(region, self.cfg)
        else:
            self.domain = domain

        # year/month/date: accept an int, a 4-digit year, a YYYYMM string, or a
        # multi-year glob such as "201[4-5]" (kept as a string for the file search)
        self.date = year
        if len(str(year)) > 4 and "[" not in str(year):
            self.year = int(str(year)[:4])
        elif "[" in str(year):
            self.year = str(year)
        elif year == "*":
            self.year = "*"
        else:
            self.year = int(year)

        if month is not None:
            if not isinstance(month, str):
                month = str(month).zfill(2)
            self.month = month
            self.date = str(self.year) + month

        # prepare the paths to load the data, using the config values as default and superceded by any arguments passed to the function
        self._resolve_paths(self.cfg, fp_datadir=fp_datadir, fp_zarr_datadir=fp_zarr_datadir, fp_format=fp_format, met_args=met_args, topog_args=topog_args)

        self.subsample_parameters = {"freq": freq, "sampling_mode": sampling_mode, "freq_offset": freq_offset}

    def _resolve_paths(self, cfg, fp_datadir=None, fp_zarr_datadir=None, fp_format="auto", met_args={}, topog_args={}):
        """
        Prepare the necessary loading paths. If paths are not passed as arguments, they will be constructed from the config file values. For the footprint and the meteorology paths, the config data is expected to point at a folder, which contains a folder for each domain, which in turn contains the files for each month/year with format example_name_yearmonth.nc (eg brazil_201601.nc). The function will construct the path to point directly to the files, including the date.

        If paths are passed as arguments, they will be used directly (but the date will still be added automatically, so the files should have format example_name_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/example_name_")

        The topography and landcover paths should point to a specific file, either through the config file or through the arguments.
        """
        # Footprints come in two formats, which can coexist: monthly NetCDF files
        # (*REGION*DOMAIN_YYYYMM.nc) under fp_datadir, and yearly Zarr stores
        # (*REGION*DOMAIN_YYYY.zarr) under fp_zarr_datadir. The Zarr pattern is keyed
        # on the year (not self.date, which includes the month) so a single month load
        # still points at the whole-year store; the month slice happens in
        # _load_footprints.
        self.fp_datadir_nc = self._build_fp_pattern(
            fp_datadir, getattr(cfg, "fp_datadir", None), self.date, ".nc")
        self.fp_datadir_zarr = self._build_fp_pattern(
            fp_zarr_datadir, getattr(cfg, "fp_zarr_datadir", None), self.year, ".zarr")

        # an explicit path for only one format is a direct instruction to use it
        if fp_format == "auto":
            if fp_datadir is not None and fp_zarr_datadir is None:
                fp_format = "nc"
            elif fp_zarr_datadir is not None and fp_datadir is None:
                fp_format = "zarr"

        self.fp_format, self.fp_datadir = self._resolve_fp_format(fp_format)

        # Meteorology is stored as one Zarr store per year (DOMAIN_Met_YYYY.zarr),
        # keyed on the year (not self.date, which includes the month) so a single
        # month load still points at the whole-year store; the month slice happens
        # in _get_meteorology_file. A glob pattern is kept so a year pattern like
        # "201[4-5]" resolves to multiple stores opened together.
        if met_args.get("met_datadir", None) is None:
            self.met_datadir = Path(cfg.met_datadir) / self.domain / (self.domain + "_Met_" + str(self.year) + "*.zarr")
        else:
            self.met_datadir = Path(str(met_args["met_datadir"])+ f"*{str(self.year)}*.zarr")

        self.met_args = met_args.copy()
        self.met_args["met_datadir"] = self.met_datadir

        
        self.topog_args = topog_args.copy()
        if topog_args.get("topog_path", None) is None:
            self.topog_args["topog_path"] = Path(cfg.topog_datadir)
        if topog_args.get("landcover_path", None) is None:
            if cfg is None:
                self.topog_args["landcover_path"] = None
            elif cfg.landcover_datadir is not None:
                self.topog_args["landcover_path"] = Path(cfg.landcover_datadir)

    def _build_fp_pattern(self, passed_path, cfg_root, date, suffix):
        """
        Build the glob pattern for one footprint format. Uses passed_path if given,
        appending the date and suffix unless the path already names a filetype,
        otherwise builds cfg_root/DOMAIN/*REGION*DOMAIN_date*suffix.
        Returns None if neither a path nor a config root is available for the format.
        """
        if passed_path is not None:
            passed_path = str(passed_path)
            if passed_path.endswith((".nc", ".zarr")):
                return Path(passed_path)
            return Path(passed_path + f"*{str(date)}*{suffix}")

        if cfg_root is None:
            return None

        return Path(cfg_root) / self.domain / f"*{self.region}*{self.domain}_{str(date)}*{suffix}"

    def _resolve_fp_format(self, fp_format):
        """
        Decide which footprint format to load, returning (format, glob pattern).

        "auto" looks for Zarr stores first and falls back to NetCDF, so both formats
        can coexist and Zarr wins for a year that is available in both. "zarr" or "nc"
        force a format.
        """
        patterns = {"nc": self.fp_datadir_nc, "zarr": self.fp_datadir_zarr}

        if fp_format not in ("auto", "nc", "zarr"):
            raise ValueError(f"fp_format should be one of 'auto', 'nc' or 'zarr', got '{fp_format}'")

        if fp_format != "auto":
            if patterns[fp_format] is None:
                arg_name = "fp_zarr_datadir" if fp_format == "zarr" else "fp_datadir"
                raise ValueError(
                    f"fp_format='{fp_format}' was requested, but no {fp_format} footprint path is "
                    f"available. Pass {arg_name} as an argument, or set {arg_name} in the config file."
                )
            return fp_format, patterns[fp_format]

        for candidate in ("zarr", "nc"):
            if patterns[candidate] is not None and len(glob.glob(str(patterns[candidate]))) > 0:
                if self.verbose: print(f"found {candidate} footprints matching {patterns[candidate]}")
                return candidate, patterns[candidate]

        tried = "\n".join(f"  {fmt}: {patterns[fmt]}" for fmt in ("zarr", "nc") if patterns[fmt] is not None)
        raise ValueError(
            f"No footprints found for region {self.region}, date {self.date}, in either format.\n"
            f"Tried:\n{tried}\n"
            "Check that the paths are correct and that there are files matching the patterns."
        )

    def load_meteorology(self, met_datadir=None, met_levels = [], met_variables= [], lazy_load=True, parallel=False):
        """
        loads meteorology (yearly Zarr store) and selects the met levels and variables if required

        Inputs
            - met_datadir (str): glob pattern for the meteorology Zarr store(s). Default directs to the configured meteorology folder.
            - met_levels (list): met levels to select
            - met_variables (list) : met variables to select. Derived variables (wind_speed / wind_angle) are ignored here and computed later downstream.
            - lazy_load (bool): if True, does not load the met data into memory (lazy array),  False, loads the met data into memory.
        """
        if self.verbose: print("\n ---- LOADING MET")

        # 1) load from file — level/variable selection happens inside, using the
        #    tolerant select_met_* helpers.
        self.met_file = self._get_meteorology_file(
            met_datadir,
            parallel=parallel, met_levels=met_levels, met_variables=met_variables,
        )

        # 2) check domain overlap
        self._check_domain_overlap(self.fp_data_full, self.met_file, "footprint", "meteorology")

        if not lazy_load:
            print("Loading met data into memory. If you only want to lazy-load, pass load=False")
            self.met_file.load()

        return self.met_file

    def load_topog(self, topog_path=None, landcover_path=None):
        """
        load the topgoraphy and landcover files, and interpolate to the same resolution and domain as the footprints in self.fp_data_full
        args:
         - topog_path and landcover_path: str paths to each file, or "default" for default file
        
        uses objet attributes: self.padding (contains if any amount of padding is needed to the footprint domain, and in which direction)
        """
        #### load topography
        if self.verbose: print("\n---- LOADING TOPOG AND LANDCOVER")
        if topog_path is None:
            topog_path = self.topog_args.get("topog_path", None)
        # if topog file doesnt exist
        if not os.path.exists(topog_path):
            raise ValueError(f"Topography file not found at {topog_path}. Check that the path is correct and that the file exists.")
        elif self.verbose:
             print(f"Loading topography from {topog_path}")
        with xr.load_dataset(topog_path) as topog_dataset:
            topog_file = topog_dataset.copy()
    
        if landcover_path is None:
            landcover_path = self.topog_args.get("landcover_path", None)

        if landcover_path is None:
            if self.verbose: print("no landcover path was passed, and no default landcover path found in config. skipping loading landcover")
            landcover_file = None
        
        else:
            if not os.path.exists(landcover_path):
                raise ValueError(f"Landcover file not found at {landcover_path}. Check that the path is correct and that the file exists.")
            if self.verbose: print(f"Loading landcover from {landcover_path}")
            with xr.load_dataset(landcover_path) as landcover_dataset:
                landcover_file = landcover_dataset.copy()

        if not hasattr(self, "padded_domain_coords"):
            self.padded_domain_coords = None

        topog_file = self._interp_topog(topog_file, padding=self.padded_domain_coords)

        print("loading both into memory")
        topog_file.load()

        if landcover_file is not None:
            landcover_file = self._interp_landcover(landcover_file, padding=self.padded_domain_coords)
            landcover_file.load()
        
        

        return topog_file, landcover_file

    def _get_meteorology_file(self, met_datadir, lazy_load=True, parallel=False, met_levels=[], met_variables=[]):
        """
        Load the meteorology from one or more yearly Zarr stores, concatenating
        along time. ``met_datadir`` is a glob pattern (e.g. DOMAIN_Met_2016*.zarr)
        resolved to a list of stores; usually a single year, but a year pattern
        such as "201[4-5]" resolves to several stores opened together.

        The Zarr stores are written by data_utils/convert_met_to_zarr.py, which
        already renames latitude/longitude -> lat/lon and model_level_number ->
        levels, drops the unused UM variables, removes duplicate/unsorted
        timestamps, and writes native chunks {time:1, lat:-1, lon:-1, levels:3}.
        So none of the old NetCDF-era preprocessing/renaming/rechunking is needed.
        """

        if self.verbose: print("Loading meteorology from " + str(met_datadir))

        met_stores = sorted(glob.glob(str(met_datadir)))
        if len(met_stores) == 0:
            raise ValueError(
                f"No meteorology Zarr stores found matching:\n {met_datadir}"
            )

        met_file = xr.open_mfdataset(
            met_stores,
            engine="zarr",
            concat_dim="time",
            combine="nested",
            consolidated=True,
        )

        # Slice to a single month when one was requested (whole-year loads leave
        # self.month unset). The whole-year store means cross-month-boundary
        # time_deltas resolve for the year path; a single-month load keeps the
        # month's own timestamps only.
        if getattr(self, "month", None) is not None:
            met_file = met_file.sel(time=met_file.time.dt.month == int(self.month))

        # Tolerant selection: skips missing levels and derived variables
        # (wind_speed / wind_angle), which are computed later downstream. Copy the
        # lists because these helpers mutate them in place (.remove()), and the
        # caller's met_args may be reused across years.
        met_file = select_met_levels(met_file, levels=list(met_levels))
        met_file = select_met_variables(met_file, variables=list(met_variables))

        if self.verbose:
            #print("Met file chunk stats:")
            #print("Chunks:", met_file.chunks)
            print("Dataset size (GB):", met_file.nbytes/1e9)

        self.met_file = met_file
        return self.met_file


    def _get_domain(self, region, cfg=None):
        #### check domains
        # TODO make domains dict importable
        if cfg is not None and hasattr(cfg, "domains"):
            domains = cfg.domains
            try:
                domain = domains[region]["domain_name"]   
            except: 
                raise ValueError("The region that you passed does not have an associated domain in the config file. Check your input, or add the region-domain pair to the config file.")   
        else:
            raise ValueError("No config file found, or no domains dict was found in the config file. Check your input!")
        
        return domain   

    def _check_domain_overlap(self, data1, data2, data1_name="footprints", data2_name="data2"):
        """
        Check that data2 (e.g., meteorology) has sufficient spatial coverage of data1 (e.g., footprint).
        Raises an error if there is no overlap, and warns if data2 is smaller than data1.

        Parameters
        ----------
        data1 : xarray.Dataset or DataArray
            Reference dataset (typically footprint). Expected to have release_lat and release_lon.
        data2 : xarray.Dataset or DataArray
            Dataset to check (typically meteorology, topography, or landcover).
        data1_name : str
            Name of data1 for messages. Default is "footprints".
        data2_name : str
            Name of data2 for messages (e.g., "meteorology").

        Raises
        ------
        ValueError
            If the spatial domains do not overlap at all.

        Warns
        -----
        If data2 domain is noticeably smaller than data1, suggests alignment may be needed.
        """
        import warnings

        # Get lat/lon coordinate names (assume it has been renamed already)
        lat1_min, lat1_max = float(data1['lat'].min()), float(data1['lat'].max())
        lon1_min, lon1_max = float(data1['lon'].min()), float(data1['lon'].max())
        lat2_min, lat2_max = float(data2['lat'].min()), float(data2['lat'].max())
        lon2_min, lon2_max = float(data2['lon'].min()), float(data2['lon'].max())

        # Check for overlap
        lat_overlap = not (lat1_max < lat2_min or lat1_min > lat2_max)
        lon_overlap = not (lon1_max < lon2_min or lon1_min > lon2_max)

        if not (lat_overlap and lon_overlap):
            raise ValueError(
                f"No spatial overlap between {data1_name} and {data2_name}!\n"
                f"  {data1_name}: lat [{lat1_min:.2f}, {lat1_max:.2f}], lon [{lon1_min:.2f}, {lon1_max:.2f}]\n"
                f"  {data2_name}: lat [{lat2_min:.2f}, {lat2_max:.2f}], lon [{lon2_min:.2f}, {lon2_max:.2f}]"
            )

        # Calculate margin based on max release latitude/longitude in the footprint
        max_release_lat = float(data1['release_lat'].max())
        min_release_lat = float(data1['release_lat'].min())
        margin_lat = [lat2_max - max_release_lat, min_release_lat - lat2_min]
        max_release_lon = float(data1['release_lon'].max())
        min_release_lon = float(data1['release_lon'].min())
        margin_lon = [lon2_max - max_release_lon, min_release_lon - lon2_min]   

        margin_threshold = 0.5  # degrees, can adjust based on typical footprint spread
        # warn if the lat and lon dont cover the area where there are releases
        if margin_lat[0] < margin_threshold or margin_lat[1] < margin_threshold or margin_lon[0] < margin_threshold or margin_lon[1] < margin_threshold:
            warnings.warn(
                f"{data2_name} domain does not fully cover the area where {data1_name} releases occur, or is close to it.\n"
                f"  {data1_name}: lat range {lat1_max - lat1_min:.2f}°, lon range {lon1_max - lon1_min:.2f}°\n"
                f"  {data2_name}: lat range {lat2_max - lat2_min:.2f}°, lon range {lon2_max - lon2_min:.2f}°\n",
                UserWarning
            )

    def _load_footprints(self, fp_datadir, load_fps_in_mem=True):
        """
        Load footprint from fp_datadir and applies subsampling according to the freq and sampling_mode parameters. 
        
        The footprints are stored in self.fp_data_full as an xarray dataset, with dimensions time, lat and lon. 
        """
        #### load footprint (fp) data from file
        if self.verbose: print("Loading footprint data from " + str(fp_datadir) )

        self.fp_data_full = load_fps(fp_datadir, verbose=self.verbose, parallel_loading=self.parallel_loading, drop_variables_except=["fp", "release_lat", "release_lon"])

        # Zarr stores are yearly, so a single-month load has to be sliced after opening.
        # The NetCDF files are monthly, so there the month is already selected by the filename.
        if self.fp_format == "zarr" and getattr(self, "month", None) is not None:
            self.fp_data_full = self.fp_data_full.sel(
                time=self.fp_data_full.time.dt.month == int(self.month))
            if self.fp_data_full.time.size == 0:
                raise ValueError(
                    f"No footprints found for month {self.month} in the zarr store(s) matching "
                    f"{fp_datadir}. Check that the month is covered by the store."
                )

        self.fp_data_full = self.fp_data_full.drop_duplicates(dim="time")

        # reshape onto the canonical sample_id layout (satellite: relabel the time
        # axis; receptor subclass: flatten (time, receptor)) — see _prepare_samples
        self.fp_data_full = self._prepare_samples(self.fp_data_full)

        ## reduce data frequency with regular sampling 9eg keep only 1 in every 3 timesteps
        # uses the sampling_mode and freq parameters
        self._subsample_frequency(**self.subsample_parameters)

        #print(self.fp_data_full)
        if self.verbose: print(f"Loading {self.fp_data_full.sizes['sample_id']} footprints")
        if load_fps_in_mem:
            self.fp_data_full.fp.load()
            print("loaded fp variable into mem")
        #print(self.fp_data_full)
        #self.fp_data_full = self.fp_data_full.chunk({"lat": -1, "lon": -1, "time": "auto"})



        
    
    def _prepare_samples(self, fp_data_full):
        """
        Reshape freshly-loaded footprints onto the canonical per-sample layout: a
        unique integer ``sample_id`` index. Satellite data has one footprint per
        timestamp, so this just relabels the time axis (keeping time as a searchable
        coordinate); LoadReceptorData overrides it to flatten (time, receptor).
        """
        return _relabel_time_to_sample_id(fp_data_full)

    def _subsample_frequency(self, freq=1, sampling_mode="regular",freq_offset=0):
        """
        Subsample the footprint data by selecting every freq-th sample from the original dataset, starting from the sample specified by freq_offset (if sampling_mode is "regular"), or by randomly selecting N/freq samples from the original dataset (if sampling_mode is "random"). If freq=1, no subsampling is done and all footprints are loaded. Operates along the sample_id index.
        """
        # subsample the footprint data according to a particular sampling mode,
        # along the sample_id index
        sample_values = self.fp_data_full["sample_id"].values
        self.original_fp_time_length = len(sample_values)
        if freq>1 and sampling_mode=="regular":
            print(f"reduced the number of datapoints by frequency {freq}")
            self.fp_data_full = self.fp_data_full.sel(sample_id=sample_values[freq_offset::freq])

        elif freq>1 and sampling_mode=="random":
            print(f"reduced the number of datapoints by frequency {freq}, chosen at random")
            self.fp_data_full = self.fp_data_full.sel(sample_id=np.random.choice(sample_values, size=np.shape(sample_values[::freq]), replace=False))
        else:
            if self.verbose: print("no sampling was done because you didnt pass a valid sampling mode, or freq=1")

    def _interp_topog(self, topog_file, padding=None):
        """
        loads the topography, interpolates to fp res
        # assumes the same resolution and domain as the footprints, unless padding is passed as padded coordinates (tuple with shape (lat_values, lon_values)) 
        """

        lat_values = list(self.fp_data_full.lat.values)
        lon_values = list(self.fp_data_full.lon.values)

        if padding is not None:
            lat_values = padding[0]
            lon_values = padding[1]
        
        """
        if padding is not None and padding != {"lat":(0,0), "lon":(0,0)}:
            delta_lon = lon_values[1]-lon_values[0]
            delta_lat = lat_values[1]-lat_values[0]               
            lat_values = np.array(sorted(lat_values + [np.max(lat_values)+delta_lat*i for i in range(5+padding["lat"][1])]+ [np.min(lat_values)-delta_lat*i for i in range(5+padding["lat"][0])]))
            lon_values = np.array(sorted(lon_values + [np.max(lon_values)+delta_lon*i for i in range(5+padding["lon"][1])]+ [np.min(lon_values)-delta_lon*i for i in range(5+padding["lon"][0])]))       

        """
        topog_file = _rename_latlon(topog_file)
        topog_file = _wrap_longitudes(topog_file)
        
        # Check domain overlap after renaming and wrapping
        self._check_domain_overlap(self.fp_data_full, topog_file, "footprint", "topography")
        
        topog_file = topog_file.interp(lat=lat_values, lon=lon_values)

        return topog_file

    def _interp_landcover(self, landcover_file, padding=None):
        """
        loads the landcover file, interpolates
        # assumes the same resolution and domain as the footprints, unless padding is passed as padded coordinates (tuple with shape (lat_values, lon_values))
        """
        lat_values = list(self.fp_data_full.lat.values)
        lon_values = list(self.fp_data_full.lon.values)

        if padding is not None:
            lat_values = padding[0]
            lon_values = padding[1]
        """
        if padding is not None and padding != {"lat":(0,0), "lon":(0,0)}:
            delta_lon = lon_values[1]-lon_values[0]
            delta_lat = lat_values[1]-lat_values[0]               
            lat_values = np.array(sorted(lat_values + [np.max(lat_values)+delta_lat*i for i in range(5+padding["lat"][1])]+ [np.min(lat_values)-delta_lat*i for i in range(5+padding["lat"][0])]))
            lon_values = np.array(sorted(lon_values + [np.max(lon_values)+delta_lon*i for i in range(5+padding["lon"][1])]+ [np.min(lon_values)-delta_lon*i for i in range(5+padding["lon"][0])]))   
        """
        landcover_file = _rename_latlon(landcover_file)
        landcover_file = _wrap_longitudes(landcover_file)
        
        # Check domain overlap after renaming and wrapping
        self._check_domain_overlap(self.fp_data_full, landcover_file, "footprint", "landcover")
        
        landcover_file = landcover_file.interp(lat=lat_values, lon=lon_values, method="nearest")

        landcover_file = landcover_file.transpose("lat", "lon","pseudo_level")

        return landcover_file

    def align_domains(
            self,
            crop_to_intersection=True,
            include_topo_and_landcover=True
        ):
        """
        Aligns domains for the meteorology, footprints, and optionally topography and landcover.

        Parameters
        ----------
        crop_to_intersection : bool
            If False:
                Interpolate met_file to the full footprint grid using nearest neighbour,
                even if met_file is smaller. This preserves all footprint pixels but 
                risks artefacts in the interpolated meteorology.

            If True (default):
                Crop BOTH datasets to the spatial intersection BEFORE interpolating.
                This removes footprint pixels outside the met domain but avoids artefacts.

        include_topo_and_landcover: bool
            Optionally include alignment of topography and landcover files. Included as default.
        
        """
        if not hasattr(self, "met_file"):
            print("met file has not been loaded yet, cannot align domains. Please load met file first)")
            return

        met = self.met_file
        fp = self.fp_data_full

        met_lat_min, met_lat_max = float(met.lat.min()), float(met.lat.max())
        met_lon_min, met_lon_max = float(met.lon.min()), float(met.lon.max())

        fp_lat_min, fp_lat_max = float(fp.lat.min()), float(fp.lat.max())
        fp_lon_min, fp_lon_max = float(fp.lon.min()), float(fp.lon.max())

        if not crop_to_intersection:
            if (met_lat_max < fp_lat_max or met_lat_min > fp_lat_min or met_lon_max < fp_lon_max or met_lon_min > fp_lon_min):
                warnings.warn("met file domain is smaller than the footprint domain! interpolating to the same grid as the footprint, but this will create artifacts in the meteorology! \n You may want to set crop_to_intersection=True.")

            self.met_file = self.met_file.interp(lat=self.fp_data_full.lat.values, lon=self.fp_data_full.lon.values, method="nearest")
            return

        # Determine intersection box to crop to
        inter_lat_min = max(met_lat_min, fp_lat_min)
        inter_lat_max = min(met_lat_max, fp_lat_max)
        inter_lon_min = max(met_lon_min, fp_lon_min)
        inter_lon_max = min(met_lon_max, fp_lon_max)

        print(
            f"Cropping to intersection:\n"
            f"  lat: {inter_lat_min:.3f} → {inter_lat_max:.3f}\n"
            f"  lon: {inter_lon_min:.3f} → {inter_lon_max:.3f}"
        )

        fp_cropped = fp.sel(
            lat=slice(inter_lat_min, inter_lat_max),
            lon=slice(inter_lon_min, inter_lon_max)
        )
        met_cropped = met.sel(
            lat=slice(inter_lat_min, inter_lat_max),
            lon=slice(inter_lon_min, inter_lon_max)
        )

        met_interp = met_cropped.interp(lat=fp_cropped.lat.values, lon=fp_cropped.lon.values, method="nearest")

        self.fp_data_full = fp_cropped
        self.met_file = met_interp

        if include_topo_and_landcover:
            topo = self.topog_file
            land = self.landcover_file

            if topo is not None:
                topo_cropped = topo.sel(
                    lat=slice(inter_lat_min, inter_lat_max),
                    lon=slice(inter_lon_min, inter_lon_max)
                )
                self.topog_file = topo_cropped
            if land is not None:
                land_cropped = land.sel(
                    lat=slice(inter_lat_min, inter_lat_max),
                    lon=slice(inter_lon_min, inter_lon_max)
                )
                self.landcover_file = land_cropped

    def get_country_masks(self, countrymask_path="default"):
        """
        Loads country mask for the domain, and creates a land-sea mask. Interpolates both to the same resolution and domain as the footprints in self.fp_data_full. Stores the country mask in self.countries.country_mask and the land-sea mask in self.countries.land_mask.
        """
        ## get land-sea mass and country mask, can be used for filtering out footprints/data and during plotting
        if countrymask_path=="default": 
            countrymask_path = "/group/chem/acrg/LPDM/countries/country_"+self.domain+".nc"
        if self.verbose: print(f"trying to load country mask from {countrymask_path}")
        with xr.load_dataset(countrymask_path) as country_dataset:
            country_ds = country_dataset.copy()
        
        try:
            country_ds = country_ds.interp(lat=self.fp_data_full.lat.values, lon=self.fp_data_full.lon.values, method="nearest")
            country_indices = xr.DataArray(np.arange(len(country_ds.name)), coords={'ncountries': country_ds.name.values}, dims='ncountries')

            country_ds['country_mask'] = country_ds.country == country_indices
            country_ds = country_ds.drop_vars("name").rename({"ncountries": "name"})
            self.countries = country_ds

            landmask = (self.countries.country != 0).astype(int)
            self.countries['land_mask'] = landmask

            if self.verbose: print("country mask loaded successfully at self.countries.country_mask and landmask at self.countries.land_mask")

        except Exception as e:
            warnings.warn(f"Error occurred while processing country mask: {e} \n Returning original country dataset without processing")
            self.countries = country_ds


    def plot_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, return_fig=False, plot_marker=False, dpi=100, figsize=(6,6), coastlines_res="110m"):
        """
        plot a footprint for a particular timestamp or index

        inputs:
            - idx: int position along sample_id of the footprint to plot. If timestamp is also passed, timestamp will be used instead of idx
            - timestamp: timestamp of the footprint to plot, as a string in format "YYYY-MM-DDTHH:MM:SS" (eg "2016-01-01T12:00:00"). If idx is also passed, timestamp will be used instead of idx
            - return_fig: if True, returns the fig and ax objects instead of showing the plot.
        """
        import matplotlib.pyplot as plt

        # a timestamp no longer identifies a single sample (receptor data has several
        # samples per time), so look up the matching positions along sample_id
        if timestamp is not None:
            matches = np.flatnonzero(self.fp_data_full.time.values == np.datetime64(timestamp))
            if len(matches) > 1:
                print("there are multiple footprints for the timestamp you passed, check the timestamp and try again! plotting the first one")
            idx = matches[0]

        fp_to_plot = self.fp_data_full.isel(sample_id=idx).copy()

        f = np.copy(fp_to_plot.fp.values)

        extent = (fp_to_plot.lon.values[0], fp_to_plot.lon.values[-1], fp_to_plot.lat.values[0], fp_to_plot.lat.values[-1])

        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize, dpi=dpi)
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        ax.coastlines(resolution=coastlines_res, color='black', linewidth=1, alpha=0.5)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.stock_img()

        cmap = plt.cm.Reds
        cmap.set_over = "k"
        plot_params = {"transform":cartopy.crs.PlateCarree(), "cmap":cmap, "vmin":vmin_vmax[0], "vmax":vmin_vmax[1]}
        background_alpha=0.4
        if levels is None:
            levels = [-4, -3.5, -3,  -2.5, -2, -1.5]

        cb = ax.contourf(fp_to_plot.lon.values, fp_to_plot.lat.values, np.log10(f), **plot_params, levels=levels, extend="both", alpha=background_alpha)
        
        levels=cb.levels

        f[f<background_threshold] = 0
        cb = ax.contourf(fp_to_plot.lon.values, fp_to_plot.lat.values,np.log10(f), **plot_params, levels=levels, extend="both")
        formatted_time = fp_to_plot.time.values.astype('datetime64[ms]').astype('O').strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]
        ax.set_title(f"{formatted_time}\nsample_id {int(fp_to_plot.sample_id)}")

        if plot_marker:
            ax.scatter(fp_to_plot.release_lon.values, fp_to_plot.release_lat.values, marker="x", color="white",s=25, lw=1,transform=cartopy.crs.PlateCarree(), zorder=10)

        if add_cbar:
            cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both", shrink=0.55).set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)

        if return_fig:
            return fig, ax
        else:
            plt.show()

class LoadSquareSatelliteData(LoadBaseSatelliteData):
    """
    Load footprint and meteorological data for a particular domain and time period, outputting all data cut to a square centered around the measurement point for each timestamp. 

    Inherites the loading functions from the general LoadBaseSatelliteData

    ## udpate these using above

    main inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - month: str in format "01" for January etc, None if loading a whole year
        - region: region identifyer, as a string. 
        - domain: Spatial domain related to the region. It can be extracted automatically from the config file if the region-domain pair is specified there, or it can be passed directly as an argument. The domain is used to find the files to load.
        - size: size for footprint to be cut to, as an int. Resolution of the footprint is maintained, cut to a sizexsize square around the release point. 
    
    Paths:
        The paths to the files are by default loaded from the config file, where the file structure is expected to be path/to/footprints/domain/region_*domain*_yearmonth.nc for the NetCDF footprints, path/to/footprints_zarr/domain/region_*domain*_year.zarr for the Zarr footprints, and path/to/meteorology/domain/domain_Met_year.zarr for the meteorology.
        The config paths are superceded by passing the paths as arguments, as strings or path objects:
        - fp_datadir for the NetCDF footprints, which should point to the folder containing the files for each month/year with format example_name_yearmonth.nc (eg brazil_201601.nc), so that the date can be automatically added.
        - fp_zarr_datadir for the Zarr footprints, which should point to the folder containing the yearly stores with format example_name_year.zarr (eg brazil_2016.zarr), so that the year can be automatically added.
        - fp_format: "auto" (default), "zarr" or "nc". "auto" looks for Zarr stores first and falls back to NetCDF, so both formats can coexist and Zarr wins for a year available in both. Passing only one of fp_datadir/fp_zarr_datadir also selects that format. "zarr" or "nc" force a format.
        - met_args = {"met_datadir": "path/to/meteorology/domain/domain_Met_"}, which should point to the folder containing the files for each month/year with format example_name_yearmonth.nc (eg brazil_201601.nc), so that the date can be automatically added.
        - topog_args = {"topog_path": "path/to/topography/file.nc", "landcover_path": "path/to/landcover/file.nc"}, which should point to the specific files for topography and landcover. These files will be interpolated to the same resolution and domain as the footprints, so they can be from a different source and with a different original resolution.

    other inputs
        - freq: int, frequency of the data to load. 
        freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset for testing
        - sampling_mode: str, default "regular".
        if "regular", subsamples footprints regularly (e.g. one in every two, sequentially with freq=2). if random, subsamples N/freq footprints randomly (where N is the total number of footprints)
        - freq_offset: int
        if using sampling_mode="regular", offsets the start of the regular sampling, e.g. freq=2 and freq_offset=0 will sample even footprints, and freq_offset=1 will sample uneven footprints
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. 
        Applied after sampling with freq (or pass freq=1 to load all footprints)
        - verbose: if True, prints out the steps throughout the data loading process
        - lazy_load: if True, does not load the met data into memory (lazy array), False, loads the met data into memory.
        - fill_outofdomain_with: str, out of "nans" and "zeros". Determines what to do if any part of the square cut around the footprint is outside of the domain. "nans" and "zeros" fill only the out of domain areas with nans and zeros respectively. 
        - delete_outofdomain: bool, if True delete all footprints (and associated datapoints) where the extracted area size x size escapes the domain. Default is False, which means that the cut footprints will be kept and the out of domain areas will be filled according to fill_outofdomain_with.
        - verbose: if True, prints out the steps throughout the data loading process
        - load_everything: bool, if True, loads all data (footprints, meteorology and topography) at once when initializing the class. If False, only loads footprints, and meteorology and topography can be loaded later with the load_meteorology() and load_topog() functions. Default is False
        - cfg: a gates.config.Config object containing the config values. If None, the config will be loaded from the config file. If passing as an argument, this supercedes loading from the config file, and allows you to pass a custom config object with custom paths and settings.
        - crop_met: bool, if True, crops the meteorology to a square of SxS around the emasurement location, following the format of the footprint and storing as data.met . If false, the meteorology is only loaded and preprocessed, but not cropped. The inputs function calculates the domain directly from the loaded (uncropped) .met_file, so cropping the meteorology is redundant when using the inputs function.
    
    met_args:
        see load_meteorology()
    topog_args:
        see load_topog()
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, verbose = False, fill_outofdomain_with="nans", delete_outofdomain=False, check_for_nans=False, sampling_mode="regular", fp_datadir = None, fp_zarr_datadir = None, fp_format="auto", load_everything=True, lazy_load=True, met_args={}, topog_args={}, cfg=None, parallel_loading=False, crop_met=True, load_fps_in_mem=True):

        print(dask.__version__)

        self.dataset_format = "square"
        self.size = size
        self.fill_outofdomain_with = fill_outofdomain_with
        self.delete_outofdomain = delete_outofdomain

        self._common_init(
            year, region=region, month=month, domain=domain,
            freq=freq, freq_offset=freq_offset, sampling_mode=sampling_mode,
            fp_datadir=fp_datadir, fp_zarr_datadir=fp_zarr_datadir, fp_format=fp_format,
            met_args=met_args, topog_args=topog_args, cfg=cfg,
            verbose=verbose, parallel_loading=parallel_loading,
        )
        
        #### load footprint (fp) data, subsample, crop
        if verbose: print("---- LOADING FOOTPRINTS") 
        self._load_footprints(self.fp_datadir, load_fps_in_mem=load_fps_in_mem)
        self._process_footprints(lazy_load)

        self.met_processed = False

        if load_everything:
            self.met_file = self.load_meteorology(**self.met_args,lazy_load=lazy_load, parallel=parallel_loading)
            self.topog_file, self.landcover_file = self.load_topog(**self.topog_args)
            self.topog = self._process_topog_and_landcover()
            if crop_met:
                self.met = self._process_meteorology(lazy_load=True)
            else:
                print("NOT cropping met!")
        """
        if check_for_nans:
            print("\n Checking if there are any nans in the data")
            self._remove_fp_nans()
            self._remove_met_nans()
        """
        if verbose: print("---- All done!")       

    def _process_footprints(self, lazy_load):
        """
        cut data around release point
        fp data returned is array of shape (time, size*size) with each footprint centered around its release point AND as a full xarray dataset of coordinates time, lat lon where lat and lon are artificial coordinates with range (0,size) and the measurement point is in the center at size//2, size//2
        
        If the square to extract escapes the footprint domain, the padded space is filled with nans or zeros or deleted according to the fill_outofdomain_with and delete_outofdomain parameters.
        """
        if self.verbose: print(f"----- Cutting footprints to square of size {self.size}") 
        self.fp_xr, self.fp_data_full, self.release_idxs, padded_domain_coords = cut_satellite_data(self.fp_data_full, self.size, fill_bads_with=self.fill_outofdomain_with, delete_outofdomain = self.delete_outofdomain, verbose=self.verbose, load=not lazy_load)
        ## for now!
        self.padded_domain_coords = padded_domain_coords

    def _process_meteorology(self,rechunk=0,lazy_load=True, pad_mode="edge"):
        """
        Call cut_satellite_met to cut the meteorology data around the release point, to the same size as the cut footprints.
        Pads with mode="edge" 
        """
        if self.verbose: print("----- Cutting met")
        self.metsize=self.size
        #if self.fill_outofdomain_with=="nans" or self.delete_outofdomain:
        #    pad_mode = "nans"
        #if self.fill_outofdomain_with=="zeros":

        self.met, nan_idxs = cut_satellite_met(self.met_file, self.fp_data_full, metsize=self.size, time_delta=0, pad_mode=pad_mode, load=not lazy_load, add_wind_direction=True, return_nan_idxs=True)

        # samples whose met time could not be matched (met_timestamps is NaT) would
        # otherwise silently carry the met of the first timestamp, so drop them from
        # every object at once — self.met is assigned above so it is dropped too
        if len(nan_idxs) > 0:
            warnings.warn(
                f"dropping {len(nan_idxs)} samples with no meteorology within the matching tolerance")
            self.remove_indeces(nan_idxs)

        if rechunk>0:
            self.met.chunk({"sample_id":rechunk})
        
        self.met_processed = True
        return self.met

    
    def _process_topog_and_landcover(self):
        """
        Obsolete as standalone logic: delegates to cut_topog_data.

        Cuts topography and landcover to a size x size square centred on each
        footprint's release point.
        """
        if self.verbose:
            print("----- Cutting topog")
        self.topog = cut_topog_data(
            self.topog_file, self.landcover_file, self.fp_data_full,
            self.size, pad_mode="zeros")
        return self.topog

    def get_flux(self, year=None, append_to_fp=True, search_others=True, convert_units=False, convert_units_args={}):
        if year is None:
            year = self.year
        ems = load_flux_data(self.domain, year=year, search_others=search_others)
        if convert_units:
            if len(convert_units_args) == 0:
                warnings.warn("convert_units is True but no convert_units_args were passed!")
            ems = convert_flux_units(ems, **convert_units_args)
        cropped_flux, nan_idxs = cut_flux_data(ems, self.fp_data_full, size=self.size)
        self.fluxes = cropped_flux
        self.remove_indeces(nan_idxs)
        if append_to_fp:
            self.fp_xr["flux"] = self.fluxes.flux
        return self.fluxes

    def calculate_modelled_mf(self):
        """
        Calculate the modelled mole fraction for each sample, using the fluxes and the footprints. Returns a 1D array of shape (n_samples,) with the modelled mole fraction for each sample.
        """
        if not hasattr(self, "fluxes"):
            raise ValueError("Fluxes have not been loaded yet. Please run get_flux() first.")
        if not hasattr(self, "fp_xr"):
            raise ValueError("Footprints have not been loaded yet. Please run _load_footprints() first.")
        if self.fluxes.flux.shape[0] != self.fp_xr.fp.shape[0]:
            raise ValueError("Fluxes and footprints have different number of samples. Please check your data.")
        # calculate the modelled mole fraction for each sample
        modelled_mf = self.fp_xr.fp * self.fluxes.flux
        modelled_mf = modelled_mf.sum(axis=(1, 2))
        self.fluxes["modelled_mf"] = modelled_mf
        self.fluxes["release_lat"] = self.fp_xr.release_lat
        self.fluxes["release_lon"] = self.fp_xr.release_lon
        return modelled_mf

    def remove_indeces(self, nan_idxs):
        """
        removes any set of samples passed as nan_idxs from all the objects in the dataset

        nan_idxs can be sample_id values, or timestamps — in which case every sample at
        those times is removed (for receptor data, all the receptors sharing the timestamp)
        """
        sample_ids = np.asarray(nan_idxs)
        if np.issubdtype(sample_ids.dtype, np.datetime64):
            times = self.fp_data_full["time"].values
            sample_ids = self.fp_data_full["sample_id"].values[np.isin(times, sample_ids)]

        self.fp_data_full = self.fp_data_full.drop_sel(sample_id=sample_ids)
        if hasattr(self, "fp_xr"):
            self.fp_xr = self.fp_xr.drop_sel(sample_id=sample_ids)
        if hasattr(self, "met"):
            self.met = self.met.drop_sel(sample_id=sample_ids)
        if hasattr(self, "topog"):
            self.topog = self.topog.drop_sel(sample_id=sample_ids)
        if hasattr(self, "fluxes"):
            self.fluxes = self.fluxes.drop_sel(sample_id=sample_ids)

        if self.verbose: print(f"Length after removing indeces: {self.fp_data_full.sample_id.size}")
        # if the len is zero, raise an error
        if self.fp_data_full.sample_id.size == 0:
            raise ValueError("All data has been removed after removing nans, cannot continue!")
    """
    def _remove_fp_nans(self):
        ## check if any of the fp entries are nans, and if so remove from met and others
        # can take a long time to run!
        if np.sum(np.isnan(self.fp_data_full.fp.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.fp_data_full.fp.values))[2])
            print(f"There are {len(nan_idxs)} nans in the fp data. finding and deleting from met and fp (only on axis time)")
            self.remove_indeces(nan_idxs)
            self.fp_nan_idxs = nan_idxs
        else:
            self.fp_nan_idxs = []
      
    def _remove_met_nans(self):
        # check if any of the met entries are nans, and if so remove from all objects
        # this shouldnt be hardcoded !!
        # needs a revision
        if np.sum(np.isnan(self.met[[list(self.met.data_vars)[0]]].values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.met[[list(self.met.data_vars)[0]]].values[0,0,0,:])))
            print(f"There are {len(nan_idxs)} nans in the met data. finding and deleting from met and fp (only on axis time)")
            
            self.remove_indeces(nan_idxs)
            self.met_nan_idxs = nan_idxs
        else:
            self.met_nan_idxs=[]
        """  
    
    def plot_cropped_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, plot_wind=False, return_fig=False, coastlines_res="110m"):
        """
        plot a footprint for a particular timestamp or index, with the option to also plot the topography and landcover if they have been loaded. 

        inputs:
            - idx: int position along sample_id of the footprint to plot. If timestamp is also passed, timestamp will be used instead of idx
            - timestamp: timestamp of the footprint to plot, as a string in format "YYYY-MM-DDTHH:MM:SS" (eg "2016-01-01T12:00:00"). If idx is also passed, timestamp will be used instead of idx
        """
        import matplotlib.pyplot as plt

        # a timestamp no longer identifies a single sample (receptor data has several
        # samples per time), so look up the matching positions along sample_id
        if timestamp is not None:
            matches = np.flatnonzero(self.fp_xr.time.values == np.datetime64(timestamp))
            if len(matches) > 1:
                print("there are multiple footprints for the timestamp you passed, check the timestamp and try again! plotting the first one")
            idx = matches[0]

        fp_to_plot = self.fp_xr.isel(sample_id=idx).copy()

        f = np.copy(fp_to_plot.fp.values)
        fp_lats = fp_to_plot.lat_coords.values
        fp_lons = fp_to_plot.lon_coords.values

        extent = (fp_lons[0], fp_lons[-1], fp_lats[0], fp_lats[-1])

        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        ax.coastlines(resolution=coastlines_res, color='black', linewidth=1, alpha=0.5)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)

        cmap = plt.cm.Reds
        cmap.set_over = "k"
        plot_params = {"transform":cartopy.crs.PlateCarree(), "cmap":cmap, "vmin":vmin_vmax[0], "vmax":vmin_vmax[1]}
        background_alpha=0.4
        if levels is None:
            levels = [-4, -3.5, -3,  -2.5, -2, -1.5]

        cb = ax.contourf(fp_lons, fp_lats, np.log10(f), **plot_params, levels=levels, extend="both", alpha=background_alpha)
        levels=cb.levels
        f[f<background_threshold] = 0
        cb = ax.contourf(fp_lons, fp_lats, np.log10(f), **plot_params, levels=levels, extend="both")
        formatted_time = fp_to_plot.time.values.astype('datetime64[ms]').astype('O').strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]
        ax.set_title(f"{formatted_time}\nsample_id {int(fp_to_plot.sample_id)}")

        if add_cbar:
            cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both").set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)

        if plot_wind:
            u_arrow = -self.met.x_wind.sel(levels=3, lat=self.size//2, lon=self.size//2).isel(sample_id=idx).values
            v_arrow = -self.met.y_wind.sel(levels=3, lat=self.size//2, lon=self.size//2).isel(sample_id=idx).values

            # Position arrow at centre of domain
            arrow_lat = fp_to_plot.lat_coords.sel(lat=self.size//2).values
            arrow_lon = fp_to_plot.lon_coords.sel(lon=self.size//2).values
            print(f"plotting wind arrow at lat {arrow_lat} and lon {arrow_lon} with u {u_arrow} and v {v_arrow}")
            ax.quiver(arrow_lon, arrow_lat, u_arrow, v_arrow,
                    transform=cartopy.crs.PlateCarree(),
                    scale=10, scale_units="inches",
                    color='black', width=0.005,
                    zorder=10)



        if return_fig:
            return fig, ax

    def plot_footprint_mean(self,levels = [-4, -3.5, -3,  -2.5, -2, -1.5], vmin_vmax=[-4,-2], add_cbar=False):
        """
        Plot the mean of the cropped and aligned footprints. Levels and vmin_vmax adjust the colour scale. If add_cbar is True, adds a colorbar
        """
        import matplotlib.pyplot as plt

        f = self.fp_xr.fp.mean(dim="sample_id").values
        #f[f<5e-5] = 0
        vmin, vmax = vmin_vmax

        fig, ax = plt.subplots(1,1)

        cmap = plt.cm.Reds
        cmap.set_over = "k"
        plot_params = {"cmap":cmap, "vmin":vmin, "vmax":vmax}
        alpha =0.4
        cb = ax.contourf(np.log10(f), **plot_params, levels=levels, alpha=0.6, extend="both")
        if add_cbar:
            cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both").set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)

        # set ticks at the middle data.size//2 and every 10 units from there
        tick_interval =10
        ticks = np.arange(tick_interval//2, self.size, tick_interval)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        #ax.set_xticklabels(ticks - self.size//2)
        #ax.set_yticklabels(ticks - self.size//2)
        ax.set_xlabel("Longitude (in grid-cells)")
        ax.set_ylabel("Latitude (in grid-cells)")

        fig.suptitle("Mean footprint, centered around release point")
        plt.show()


class LoadReceptorData(LoadSquareSatelliteData):
    """
    Load receptor-format footprints and cut them to a square around each receptor's
    release point, in the same layout produced by LoadSquareSatelliteData.

    The satellite footprints have one release per timestamp, so ``time`` uniquely
    labels each datapoint. Receptor footprints instead come as yearly Zarr stores
    with dimensions (time, receptor, lat, lon): the same set of receptors is
    recorded at many timesteps, so a datapoint is a (time, receptor) pair. On load
    these are flattened into a single ``sample_id`` index (see
    _stack_receptors_to_sample_id). The physical footprint time and receptor id are
    preserved as coordinates along ``sample_id`` — add an index on the object you
    are inspecting (``ds.set_xindex("time")``) to select on them.

    Only the footprint reshaping differs; the square cropping and the rest of the
    machinery are inherited and operate on the shared ``sample_id`` index. This is
    done by overriding the ``_prepare_samples`` hook rather than the whole
    ``_load_footprints`` method, so the shared loading/subsampling logic (zarr
    opening, month slicing, frequency subsampling) is reused as-is.

    Footprint paths: pass ``fp_zarr_datadir`` pointing at the receptor Zarr stores
    (a glob resolving to one or more *_YYYY.zarr stores that open together along
    time), or rely on the config's ``fp_zarr_datadir``. If the receptor stores
    follow a different naming convention to the satellite Zarr, override
    ``_resolve_paths``. Other arguments match LoadSquareSatelliteData.

    Pass load_everything=True to also load and crop the meteorology and topography
    onto the same (sample_id, lat, lon) grid as fp_xr.

    NOTE: flux and the input/dataloader machinery are not yet updated for the
    sample_id layout, so load_everything defaults to False.
    """
    def __init__(self, year, region="Canterbury", subregion=None, month=None, domain=None, size=10,
                 fp_zarr_datadir=None, fp_format="zarr",
                 load_everything=False, **kwargs):

        self.super_region=str(region).upper()
        self.subregion=subregion
        if subregion is not None:
            region = f"{region}{subregion}"
        # if there is a number in the super_region, keep the string ONLY before the number
        if any(char.isdigit() for char in self.super_region):
            self.super_region = re.split(r"\d", self.super_region)[0]

        super().__init__(
            year, region=region, month=month, domain=domain, size=size,
            fp_zarr_datadir=fp_zarr_datadir, fp_format=fp_format,
            load_everything=load_everything, **kwargs,
        )



    def _prepare_samples(self, fp_data_full):
        """Flatten (time, receptor) footprints into a unique sample_id index; see _stack_receptors_to_sample_id."""
        if self.verbose:
            print("----- Flattening (time, receptor) footprints into sample_id index")
        return _stack_receptors_to_sample_id(fp_data_full)

    def split_samples(self, mode="random", split_fractions={"train":0.75, "val":0.20, "test":0.05}, return_fps=False, random_seed=None):
        """
        Split the sample_id index into train, val and test sets. Returns a dict with keys "train", "val" and "test" and values as lists of sample_id values for each set. The split is done randomly by default, but can be changed to "sequential" to split the sample_id index sequentially (eg first 75% for train, next 20% for val, last 5% for test). The split fractions can be changed by passing a dict with keys "train", "val" and "test" and values as the fractions to use for each set. The fractions should sum to 1.0.
        """
        if mode=="random":
            sample_ids = self.fp_data_full.sample_id.values
            np.random.shuffle(sample_ids)
            n_samples = len(sample_ids)
            train_end = int(split_fractions["train"] * n_samples)
            val_end = train_end + int(split_fractions["val"] * n_samples)
            self.data_split = {
                "train": sample_ids[:train_end],
                "val": sample_ids[train_end:val_end],
                "test": sample_ids[val_end:],
            }
        elif mode=="sequential":
            sample_ids = self.fp_data_full.sample_id.values
            n_samples = len(sample_ids)
            train_end = int(split_fractions["train"] * n_samples)
            val_end = train_end + int(split_fractions["val"] * n_samples)
            self.data_split = {
                "train": sample_ids[:train_end],
                "val": sample_ids[train_end:val_end],
                "test": sample_ids[val_end:],
            }
        else:
            raise ValueError(f"mode {mode} not recognized. Use 'random' or 'sequential'.")
        if self.verbose:
            print(f"Data split into {len(self.data_split['train'])} train, {len(self.data_split['val'])} val and {len(self.data_split['test'])} test samples.")
        
        if not return_fps:
            return self.data_split
        
        elif return_fps:
            return {k: self.fp_xr.sel(sample_id=v) for k, v in self.data_split.items()}
    
    def plot_modelled_mf(self):
        """
        Plot the modelled mole fraction for each sample as scatter points on a map, calculated from the fluxes and footprints.
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, "fluxes"):
            raise ValueError("Fluxes have not been loaded yet. Please run get_flux() first.")
        # if modelled_mf is not in fluxes, calculate it
        if "modelled_mf" not in self.fluxes:
            self.calculate_modelled_mf()
        
        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines(resolution="50m", color='black', linewidth=1, alpha=0.5)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        cb = ax.scatter(self.fluxes.release_lon.values, self.fluxes.release_lat.values, c=self.fluxes.modelled_mf.values, cmap="viridis", s=10, transform=cartopy.crs.PlateCarree())
        cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both").set_label(label=r'Modelled mole fraction', size=12)
        plt.show()


        


def _get_release_idxs(fp, domain_lats=None, domain_lons=None):
    """
    Returns array of shape (n_times, 2) with (lat_idx, lon_idx) of the nearest
    grid point to each footprint's release location, on the grid defined by
    domain_lats and domain_lons. Uses the footprint's domain lat and domain lon if not passed as arguments.

    Vectorized replacement for get_release_idxs: uses a broadcast argmin instead
    of a Python loop, so it is O(n_times + n_grid_cells) rather than
    O(n_times * n_grid_cells). Still loads release_lat/lon into memory (they are
    small 1-D arrays).
    """
    fp.release_lat.load()
    fp.release_lon.load()
    if domain_lats is None:
        domain_lats = fp.lat.values
    if domain_lons is None:
        domain_lons = fp.lon.values
    rlats = fp.release_lat.values   # (n_times,)
    rlons = fp.release_lon.values   # (n_times,)
    lat_idxs = np.argmin(np.abs(domain_lats[:, None] - rlats[None, :]), axis=0)  # (n_times,)
    lon_idxs = np.argmin(np.abs(domain_lons[:, None] - rlons[None, :]), axis=0)  # (n_times,)
    return np.stack([lat_idxs, lon_idxs], axis=1)  # (n_times, 2)



def _open_flux_file(path):
    """
    Open a flux file and return its flux DataArray, carrying the file's global
    attributes across so they survive into cut_flux_data's output. Attributes set on
    the flux variable itself take precedence over global ones of the same name.
    """
    ds = xr.open_dataset(path)
    flux = ds.flux
    flux.attrs = {**ds.attrs, **flux.attrs}
    return flux


def load_flux_data(domain, year=2016, species="ch4", flux_path=None, cfg=None, search_others=False):
    """
    Load emissions for the given domain and year. Returns a lazy xarray DataArray (time, lat, lon) with all months in the file. Loads the file in flux_path if provided, otherwise looks up the file based on the config, domain and species. 
    Inputs:
    - domain: str, domain to load emissions for. Should match the domain_name in the config file. Case-insensitive. If not found, will try to match against the region name in the config file. If still not found, raises an error with the available domain names in the config file.
    - year: int or str, year to load emissions for. If str, should be in the format "2016" or "201[4-5]" to match multiple years. Default is 2016.
    - species: str, species to load emissions for. Default is "ch4". Should match the species used in the flux_suffix in the config file if flux_suffix is a dict.
    - flux_path: str, optional path to the flux file to load. If provided, this will be used instead of looking up the file based on the config, domain and species.
    - cfg: gates.config.Config instance or None. If None, the config will be loaded from the config file. If provided, should be an instance of gates.config.Config. Used to look up the flux file if flux_path is not provided.
    """
    if flux_path is not None:
        # assert that there is a file at flux path
        if not os.path.isfile(flux_path):
            raise ValueError(f"flux_path {flux_path} does not exist or is not a file")
        return _open_flux_file(flux_path)
    
    if cfg is None:
        cfg = get_config()
    elif not isinstance(cfg, gates.config.Config):
        raise ValueError("cfg should be either None or a gates.config.Config instance. If None, the config will be loaded from the config file. Check your input!")
    
    domains = cfg.domains

    domain_match = str(domain).upper()
    matched_entry = None

    for region_name, region_cfg in domains.items():
        cfg_domain_name = str(region_cfg.get("domain_name", "")).upper()
        # Match against the configured domain_name
        if cfg_domain_name == domain_match or str(region_name).upper() == domain_match:
            matched_entry = region_cfg
            break

    if matched_entry is None:
        available_domains = sorted( {str(region_cfg.get("domain_name", "")).upper()   for region_cfg in domains.values()   })
        raise ValueError(
            f"Domain '{domain}' not recognized in cfg.domains domain_name values. "
            f"Available: {available_domains}")

    resolved_domain_name = str(matched_entry["domain_name"]).upper()
    flux_suffix_cfg = matched_entry.get("flux_suffix", matched_entry.get("flux_sufflix", ""))

    if isinstance(flux_suffix_cfg, dict):
        flux_suffix = str(flux_suffix_cfg.get(species, ""))
    elif flux_suffix_cfg is None:
        flux_suffix = ""
    else:
        flux_suffix = str(flux_suffix_cfg)

    path = Path(cfg.flux_datadir) / resolved_domain_name / f"{species}_{resolved_domain_name}_{year}{flux_suffix}.nc"
    # make the sorted list a list of strings
    if not path.is_file():
        if search_others:
            other_files = sorted(str(f) for f in path.parent.glob(f"{species}*{resolved_domain_name}*{year}*.nc"))
            if len(other_files) ==1:
                print(f"Using another file for this domain and species: \n{other_files[0]}")
                path = other_files[0]
            else:
                raise ValueError(f"Flux file not found for domain '{domain}', species '{species}' and year '{year}' \nat {path}. \nThe existing files are: {sorted(str(f) for f in path.parent.glob(f'{species}_{resolved_domain_name}_{year}_*.nc'))}")
        else:
        
            raise ValueError(f"Flux file not found for domain '{domain}', species '{species}' and year '{year}' \nat {path}. \nThe existing files are: {sorted(str(f) for f in path.parent.glob(f"{species}_{resolved_domain_name}_{year}_*.nc"))}")
    else:
        print(f"Loading flux data from {path}")
    return _open_flux_file(path)


def load_default_brazil_emissions(year=2016):
    print("Obsolete: use load_emissions(domain='brazil', year=2016) instead")


def load_default_sahara_emissions(year=2016):
    print("Obsolete: use load_emissions(domain='sahara', year=2016) instead")


_FLUX_DENSITY_WARNING = (
    "WARNING — interpolation assumes the flux is a density (e.g. mol/m2/s). If it is a mass "
    "per grid cell (e.g. kT/cell), interpolating does not conserve the total and the result "
    "will be wrong; convert to a density first."
)


def cut_flux_data(flux, fp, size=None, tolerance="32D", interp_method="linear", verbose=True):
    """
    Crops flux data to a size x size square centred on each footprint's release
    point, after matching each footprint to the nearest monthly flux snapshot.

    Parameters
    ----------
    flux : xr.DataArray, dims (time, lat, lon)
        Monthly flux snapshots (e.g. from load_default_brazil_emissions).
    fp : xr.Dataset
        Footprint dataset indexed by sample_id, with release_lat, release_lon and a
        time coordinate along sample_id.
    size : int
        Side length of crop square. Must be even.
    tolerance : str
        Maximum time distance for matching a footprint to a flux snapshot.
        Default '32D' (32 days) ensures each footprint matches at most one month.
    interp_method : str or None
        Method used to regrid the flux onto the footprint grid when the two do not
        already share coordinates: 'linear' or 'nearest', passed to
        xr.DataArray.interp. None skips regridding and only warns that the
        resolutions differ. Interpolation is only valid for flux *densities*: a
        flux given as mass per grid cell must be converted to a density first, or
        the regridded totals will be wrong.
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple of (xr.Dataset, np.ndarray)
        Dataset with variables:
            - flux       : cropped flux (sample_id, lat, lon)
            - lat_coords : actual latitudes  (sample_id, lat)
            - lon_coords : actual longitudes (sample_id, lon)
        lat/lon are artificial 0..size coordinates; release point is at size//2.
        Array of sample_id values that had no flux match within tolerance.
    """

    # cropped footprints keep release_lat/release_lon, but their lat/lon dims are
    # artificial 0..size indices and the real coordinates live in lat_coords/lon_coords
    fp_is_cropped = hasattr(fp, "lat_coords") and hasattr(fp, "lon_coords")
    if not fp_is_cropped and size is None:
        raise ValueError("size must be passed when cutting uncropped footprints")
    elif fp_is_cropped and size is not None and size != fp.sizes["lat"]:
        # raise a warning and use size of fp
        warnings.warn(f"size {size} does not match footprint size {fp.sizes['lat']}. Using footprint size.")
    if fp_is_cropped:
        size = fp.sizes["lat"]

    if size % 2 != 0:
        raise ValueError(f"size must be even, got {size}")
    half = size // 2


    # --- 1. Regrid onto the footprint grid ---
    # Uncropped footprints carry the domain grid on fp.lat/fp.lon, so the flux can be
    # regridded once here, before the time axis is expanded to one snapshot per sample.
    # Cropped footprints have a different target grid per sample, so they are regridded
    # onto lat_coords/lon_coords in step 4 instead.
    if not fp_is_cropped:
        same_grid = (np.array_equal(flux.lat.values, fp.lat.values)
                     and np.array_equal(flux.lon.values, fp.lon.values))
        if not same_grid:
            em_dlat = flux.lat.values[1] - flux.lat.values[0]
            em_dlon = flux.lon.values[1] - flux.lon.values[0]
            fp_dlat = fp.lat.values[1] - fp.lat.values[0]
            fp_dlon = fp.lon.values[1] - fp.lon.values[0]
            grid_str = (f"flux grid ({flux.sizes['lat']}, {flux.sizes['lon']}) at "
                        f"({em_dlat:.4f}, {em_dlon:.4f}) deg vs fp grid "
                        f"({fp.sizes['lat']}, {fp.sizes['lon']}) at ({fp_dlat:.4f}, {fp_dlon:.4f}) deg")
            if interp_method is None:
                if verbose:
                    print(f"cut_flux_data: WARNING — {grid_str}, and interp_method is None. "
                          f"Spatial alignment may be off.")
            else:
                if verbose:
                    print(f"cut_flux_data: regridding onto the fp grid with "
                          f"method='{interp_method}': {grid_str}.\n{_FLUX_DENSITY_WARNING}")
                flux = flux.interp(lat=fp.lat.values, lon=fp.lon.values, method=interp_method)

    # --- 2. Time matching ---
    tol = pd.Timedelta(tolerance)
    nearest = flux.indexes["time"].get_indexer(
        pd.DatetimeIndex(fp.time.values), method="nearest", tolerance=tol)
    # report failed matches as sample_id values (nearest is per-sample), so
    # remove_indeces drops the right samples rather than every sample that
    # happens to share a timestamp
    nan_idxs = fp["sample_id"].values[nearest == -1]
    if verbose and len(nan_idxs):
        print(f"cut_flux_data: {len(nan_idxs)} footprints had no "
              f"flux snapshot within {tolerance}. These will be NaN in the output.")
    nearest_safe = np.where(nearest != -1, nearest, 0)
    # picking one flux snapshot per sample turns the flux time axis into a
    # per-sample axis: make sample_id the index and keep the footprint's own time
    flux_matched = flux.isel(time=xr.DataArray(nearest_safe, dims="sample_id"))
    flux_matched = flux_matched.assign_coords({
        "sample_id": fp["sample_id"].values,
        "time": ("sample_id", fp.time.values),
    })
    if len(nan_idxs):
        flux_matched = flux_matched.where(
            xr.DataArray(nearest != -1, dims="sample_id"), np.nan)

    domain_lats = flux_matched.lat.values
    domain_lons = flux_matched.lon.values

    if fp_is_cropped:
        # --- 3+4. Sample onto each footprint's own coordinates ---
        em_dlat = domain_lats[1] - domain_lats[0]
        em_dlon = domain_lons[1] - domain_lons[0]
        fp_dlat = float(fp.lat_coords[0, 1] - fp.lat_coords[0, 0])
        fp_dlon = float(fp.lon_coords[0, 1] - fp.lon_coords[0, 0])
        grid_str = (f"flux grid at ({em_dlat:.4f}, {em_dlon:.4f}) deg vs cropped fp "
                    f"lat_coords/lon_coords at ({fp_dlat:.4f}, {fp_dlon:.4f}) deg")
        if interp_method is None:
            print(f"Using lat_coords and lon_coords to cut flux data, assuming they are aligned "
                  f"and have the same resolution as the flux data: {grid_str}.")
            cropped = flux_matched.sel(lat=fp.lat_coords, lon=fp.lon_coords, method="nearest")
        else:
            print(f"Interpolating flux onto the footprints' lat_coords and lon_coords with "
                  f"method='{interp_method}': {grid_str}.\n{_FLUX_DENSITY_WARNING}")
            cropped = flux_matched.interp(lat=fp.lat_coords, lon=fp.lon_coords, method=interp_method)
        lat_coords = fp.lat_coords
        lon_coords = fp.lon_coords

    elif hasattr(fp, "release_lat") and hasattr(fp, "release_lon"):
        # --- 3. Release indices + padding ---
        release_idxs = _get_release_idxs(fp, domain_lats, domain_lons)
        flux_matched, domain_lats, domain_lons, release_idxs = _pad_domain(
            flux_matched, fp, release_idxs, half, pad_mode="nans", verbose=verbose)

        # --- 4. Vectorised isel ---
        lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
        lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]
        lat_da = xr.DataArray(lat_indices, dims=["sample_id", "lat"], coords={"sample_id": fp["sample_id"]})
        lon_da = xr.DataArray(lon_indices, dims=["sample_id", "lon"], coords={"sample_id": fp["sample_id"]})
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            cropped = flux_matched.isel(lat=lat_da, lon=lon_da)

        lat_coords = xr.DataArray(
                domain_lats[lat_indices], dims=["sample_id", "lat"],
                coords={"sample_id": fp["sample_id"]})
        lon_coords=  xr.DataArray(
                domain_lons[lon_indices], dims=["sample_id", "lon"],
                coords={"sample_id": fp["sample_id"]})

    # --- 5. Assign coordinates + return ---
    cropped = (
        cropped
        .assign_coords(lat=np.arange(size), lon=np.arange(size))
        .to_dataset(name="flux")
        .assign({
            "lat_coords": lat_coords,
            "lon_coords": lon_coords,
        })
    )
    cropped

    # if ems has attributes, add them to cropped
    if hasattr(flux, "attrs"):
        cropped.attrs = flux.attrs
    cropped = cropped.astype("float32", copy=False)
    return cropped, nan_idxs


def _relabel_time_to_sample_id(fp_full):
    """
    Put satellite-format footprints (one footprint per timestamp, dims time, lat,
    lon) onto the canonical sample layout: a unique integer ``sample_id`` index,
    with the timestamps kept as a ``time`` coordinate along it. release_lat/
    release_lon come along as (sample_id,) coordinates.

    ``time`` is a plain coordinate, not an index — a secondary index on the sample
    dimension breaks xarray's alignment in the cropping/batching steps. To query by
    time, add the index on the object you are inspecting: ``ds.set_xindex("time")``.
    """
    fp_full = fp_full.sortby("time")
    times = fp_full["time"].values
    fp_full = fp_full.rename({"time": "sample_id"})
    fp_full = fp_full.assign_coords(
        sample_id=np.arange(1, len(times) + 1),
        time=("sample_id", times),
    )
    return fp_full


def _stack_receptors_to_sample_id(fp_full):
    """
    Flatten receptor-format footprints (dims time, receptor, lat, lon) onto the
    canonical sample layout: a unique integer ``sample_id`` index, with ``time`` and
    ``receptor`` kept as coordinates along it. release_lat/release_lon are stored
    per (time, receptor) — e.g. shape (1, 40) for one timestamp and 40 receptors —
    so they flatten straight to (sample_id,), one release location per sample.

    ``time``/``receptor`` are plain coordinates, not indexes — a secondary index on
    the sample dimension breaks xarray's alignment in the cropping/batching steps.
    To query by them, add the index on the object you are inspecting:
    ``ds.set_xindex("time")`` / ``ds.set_xindex("receptor")``.
    """
    if "receptor" not in fp_full.dims:
        raise ValueError(
            "_stack_receptors_to_sample_id expects a 'receptor' dimension to flatten; "
            f"got dims {tuple(fp_full.dims)}. Is this receptor-format footprint data?"
        )

    fp_full = fp_full.sortby(["time", "receptor"])
    fp_full = fp_full.stack(sample_id=("time", "receptor"))
    fp_full = fp_full.reset_index("sample_id")
    fp_full = fp_full.assign_coords(
        sample_id=("sample_id", np.arange(1, fp_full.sizes["sample_id"] + 1)))
    return fp_full


def cut_satellite_data(fp_full, size, fill_bads_with="nans", delete_outofdomain=False,
                           load=False, verbose=True):
    """
    Cuts footprints to a size x size square centred on each release point, returning
    an xarray Dataset with artificial lat/lon coordinates 0..size and the actual
    coordinates stored as lat_coords (time, lat) and lon_coords (time, lon) variables.

    Parameters
    ----------
    fp_full : xarray.Dataset
        Full footprint dataset with variables fp, release_lat, release_lon and
        dimensions (time, lat, lon).
    size : int
        Side length of the square crop. Must be even.
    fill_bads_with : str
        'nans' or 'zeros' — fill value for out-of-domain padding.
    delete_outofdomain : bool
        If True, drop footprints whose crop square escapes the domain rather than padding.
    load : bool
        If True, load the result into memory immediately.
    verbose : bool
        Print progress messages.
    """
    if size % 2 != 0:
        raise ValueError("size must be even so the release point is centred")
    half = size // 2

    release_idxs = _get_release_idxs(fp_full)

    if delete_outofdomain:
        domain_lats = fp_full.lat.values
        domain_lons = fp_full.lon.values
        south = release_idxs[:, 0] < half
        north = (len(domain_lats) - release_idxs[:, 0]) < half
        west  = release_idxs[:, 1] < half
        east  = (len(domain_lons) - release_idxs[:, 1]) < half
        oob = np.where(np.any([south, north, west, east], axis=0))[0]
        if len(oob) > 0:
            if verbose:
                print(f"Dropping {len(oob)} footprints that escape the domain when cut to size {size}")
            fp_full = fp_full.drop_sel(sample_id=fp_full["sample_id"].values[oob])
            release_idxs = _get_release_idxs(fp_full)

    before_padding_coords = (fp_full.lat.values.copy(), fp_full.lon.values.copy())

    fp_full, domain_lats, domain_lons, release_idxs = _pad_domain(
        fp_full, fp_full, release_idxs, half, pad_mode=fill_bads_with)

    padded_domain_coords = (domain_lats, domain_lons)
    # integer index arrays: shape (n_times, size)
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]

    lat_da = xr.DataArray(lat_indices, dims=["sample_id", "lat"], coords={"sample_id": fp_full["sample_id"]})
    lon_da = xr.DataArray(lon_indices, dims=["sample_id", "lon"], coords={"sample_id": fp_full["sample_id"]})

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        cropped_fp = fp_full.isel(lat=lat_da, lon=lon_da)

    cropped_fp = (
        cropped_fp
        .assign_coords(lat=np.arange(size), lon=np.arange(size))
        .assign({
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["sample_id", "lat"],
                coords={"sample_id": cropped_fp["sample_id"]}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["sample_id", "lon"],
                coords={"sample_id": cropped_fp["sample_id"]}),
        })
    )

    cropped_fp = cropped_fp[["fp", "lat_coords", "lon_coords", "release_lat", "release_lon"]]
    cropped_fp = cropped_fp.transpose("sample_id", "lat", "lon")
    #cropped_fp = cropped_fp.chunk({"time": 100, "lat": -1, "lon": -1})

    fp_full = fp_full.sel(lat=slice(before_padding_coords[0][0], before_padding_coords[0][-1]),
                          lon=slice(before_padding_coords[1][0], before_padding_coords[1][-1]))

    if load:
        if verbose:
            print("loading cropped footprint dataset into memory")
        cropped_fp.load()
    # return the same outputs as cut_satellite_data, except without the option to return as an array (we can add this later if needed, but it can be done easily with .values and reshape on the returned xarray)
    #, fp_lats, fp_lons, release_idxs, padding, fp_full.sel(lat=slice(original_fp_domain[0], original_fp_domain[1]), lon=slice(original_fp_domain[2], original_fp_domain[3])), cropped_fp
    return cropped_fp, fp_full, release_idxs, padded_domain_coords


def _match_met_times_to_fp(met, fp, time_delta, interp_method, closest_tolerance="4h"):
    """
    Matches each footprint sample to a met timestamp (at fp_time - time_delta), without
    expanding the met onto the samples.
    Returns (met_at_used_times, time_positions, sample_coords, nan_idxs).

    The met keeps its own time axis, selected down to just the timestamps actually used
    (deduplicated, so receptor data — where many samples share a timestamp — selects each
    timestep once). time_positions gives, per sample, the position of its timestamp on
    that axis: the caller applies it as an indexer alongside the spatial crop, so the
    per-sample expansion and the crop happen in one step and the full-domain per-sample
    array is never built.

    sample_coords holds the coordinates to attach once the data is on the sample axis:
    'sample_id', 'fp_time' (the original footprint observation time, so that
    fp_time - time_delta is the met time used) and, when interp_method='closest',
    'met_timestamps' (the actual met timestamp used for each sample, NaT where no match
    was found within closest_tolerance). nan_idxs holds the sample_id values that had no
    match within the tolerance.
    """
    fp_original_times = fp.time.values
    sample_vals = fp["sample_id"].values
    tol = pd.Timedelta(closest_tolerance)

    target_times = (fp_original_times if time_delta == 0
                    else pd.DatetimeIndex(fp_original_times) - pd.Timedelta(f"{time_delta}h"))

    # failed matches are reported as sample_id values, so that remove_indeces drops
    # the right samples rather than every sample sharing a timestamp
    sample_coords = {
        "sample_id": sample_vals,
        "fp_time": ("sample_id", fp_original_times),
    }

    if interp_method == "closest":
        nearest = met.indexes["time"].get_indexer(
            pd.DatetimeIndex(target_times), method="nearest", tolerance=tol)
        nan_mask = nearest == -1
        nearest_safe = np.where(~nan_mask, nearest, 0)
        nearest_timestamps = met.indexes["time"].values[nearest_safe]

        used_times, time_positions = np.unique(nearest_timestamps, return_inverse=True)
        met = met.sel(time=used_times)

        sample_coords["met_timestamps"] = ("sample_id", pd.DatetimeIndex(
            np.where(~nan_mask, nearest_timestamps, pd.NaT)))
        nan_idxs = sample_vals[nan_mask]
    else:
        # interpolate to the unique targets only, for the same reason: several samples
        # can ask for the same target time, and interpolating each one separately
        # produces a full-domain field per sample
        used_times, time_positions = np.unique(target_times, return_inverse=True)
        met = met.interp(time=used_times, method=interp_method)
        nan_idxs = sample_vals[:0]

    return met, time_positions, sample_coords, nan_idxs


def _pad_domain(data, fp, release_idxs, half, pad_mode, verbose=True):
    """
    Extends the spatial domain of an xarray Dataset/DataArray so that a
    (2*half) x (2*half) crop is possible for every footprint release point.
    Works for any xarray object with lat/lon dimensions (met, fp, topog, etc.).
    Returns (data_padded, domain_lats, domain_lons, updated_release_idxs).

    For pad_mode='nans': uses xr.reindex with fill_value nan — lazy, 
    For pad_mode='zeros': uses xr.reindex with fill_value 0 — lazy, but be careful if your data has valid zeros!
    For pad_mode='edge': uses xr.pad(mode='edge') then assigns the correct
    extended coordinate values.
    """
    if pad_mode not in ["nans", "zeros", "edge"]:
        raise ValueError("pad_mode should be one of 'nans', 'zeros', or 'edge'")
    domain_lats = data.lat.values.copy()
    domain_lons = data.lon.values.copy()
    delta_lat = domain_lats[1] - domain_lats[0]
    delta_lon = domain_lons[1] - domain_lons[0]
    padding_needed = False

    need_S = np.sum(release_idxs[:, 0] < half)
    need_N = np.sum((len(domain_lats) - release_idxs[:, 0]) < half)
    need_W = np.sum(release_idxs[:, 1] < half)
    need_E = np.sum((len(domain_lons) - release_idxs[:, 1]) < half)


    if need_S > 0 or need_N > 0:
        pad_S = int(np.max([0, half - np.min(release_idxs[:, 0])]))
        pad_N = int(np.max([0, half - (len(domain_lats) - np.max(release_idxs[:, 0]))]))
        if verbose: print(f"Padding lat by ({pad_S}, {pad_N}) cells (S, N) with mode='{pad_mode}'")
        extended_lats = (
            sorted([domain_lats[0] - (i + 1) * delta_lat for i in range(pad_S)])
            + list(domain_lats)
            + [domain_lats[-1] + (i + 1) * delta_lat for i in range(pad_N)]
        )
        if pad_mode == "edge":
            data = data.pad(pad_width={"lat": (pad_S, pad_N)}, mode="edge")
            data = data.assign_coords({"lat": extended_lats})
        elif pad_mode == "zeros":
            data = data.reindex(lat=extended_lats, fill_value=0)
        elif pad_mode == "nans":
            data = data.reindex(lat=extended_lats, fill_value=np.nan)
        domain_lats = data.lat.values.copy()
        padding_needed = True

    if need_W > 0 or need_E > 0:
        pad_W = int(np.max([0, half - np.min(release_idxs[:, 1])]))
        pad_E = int(np.max([0, half - (len(domain_lons) - np.max(release_idxs[:, 1]))]))
        if verbose: print(f"Padding lon by ({pad_W}, {pad_E}) cells (W, E) with mode='{pad_mode}'")
        extended_lons = (
            sorted([domain_lons[0] - (i + 1) * delta_lon for i in range(pad_W)])
            + list(domain_lons)
            + [domain_lons[-1] + (i + 1) * delta_lon for i in range(pad_E)]
        )
        if pad_mode == "edge":
            data = data.pad(pad_width={"lon": (pad_W, pad_E)}, mode="edge")
            data = data.assign_coords({"lon": extended_lons})
        elif pad_mode == "zeros":
            data = data.reindex(lon=extended_lons, fill_value=0)
        elif pad_mode == "nans":
            data = data.reindex(lon=extended_lons, fill_value=np.nan)
        domain_lons = data.lon.values.copy()
        padding_needed = True

    if padding_needed:
        release_idxs = _get_release_idxs(fp, domain_lats, domain_lons)

    return data, domain_lats, domain_lons, release_idxs


def cut_satellite_met(met, fp, metsize, time_delta=0, relevant_levels=None,
                          relevant_variables=None, verbose=True, pad_mode="nans",
                          load=True, add_wind_direction=True, save=False,
                          savepath=None, attrs_dict=None, interp_method="closest",
                          closest_tolerance="4h", return_nan_idxs=False):
    """
    Cuts meteorology to a metsize x metsize square around each footprint release
    point, interpolated to footprint times (or t-time_delta).

    Uses xarray and dask to produce one coherent lazy dask graph.

    The result is indexed by 'sample_id' to match the footprints, with the footprint time kept as an 'fp_time' coordinate (fp_time - time_delta = the met time used). lat_coords and lon_coords are stored as (sample_id, lat) / (sample_id, lon) variables

    Parameters:
    - met: xarray dataset with meteorological data, with dimensions including 'time', 'lat', 'lon', and possibly 'levels'. Should have variables for the relevant meteorological fields
    - fp: xarray dataset with footprint data, indexed by 'sample_id', with a 'time' coordinate and variables 'release_lat' and 'release_lon' for the release locations of each footprint
    - metsize: int, the size of the square to cut around each release point. Must be even to ensure the release point is centered.
    - time_delta: int, hours to shift the footprint times backwards for interpolation. Default is 0 (no shift).
    - relevant_levels: list, the levels of atmospheric variables to extract. If None, uses all levels in met.
    - relevant_variables: list, the meteorological variables to extract. If None, uses all variables in met.
    - verbose: bool, whether to print progress messages
    - pad_mode: str, either "nans" or "edge". If "nans", pads with NaNs when the cut square extends beyond the met domain. If "edge", pads by extending the edge values of the met domain.
    - load: bool, whether to load the resulting cropped met into memory at the end. Default is False (keep as lazy dask array).
    - add_wind_direction: bool, whether to calculate and add wind direction and speed from x_wind and y_wind. Default is True.
    - save: bool, whether to save the resulting cropped met to a NetCDF file. Default is False.
    - savepath: str, the path to save the NetCDF file if save is True. Must end with .nc.
    - attrs_dict: dict, additional attributes to add to the resulting cropped met dataset. The original met attributes will be stored under "original_met_attrs".
    - interp_method: str, method to use for time interpolation. Options are "nearest" (with tolerance specified by closest_tolerance) or any method supported by xarray's interp (e.g. "linear", "nearest", "zero", "slinear", "quadratic", "cubic"). Default and most efficient is "closest".
    - closest_tolerance: str or pandas Timedelta specifying the maximum allowed distance for the "nearest" interpolation method. Default is "4h". Ignored if interp_method is not "nearest".
    - return_nan_idxs: bool, whether to return the indices of footprints for which no met timestamp was found within closest_tolerance when using interp_method="nearest". Default is False. If True, the function returns a tuple (cropped_met, nan_idxs)
    """

    if metsize % 2 != 0:
        raise ValueError("metsize must be even so the release point is centred")
    half = metsize // 2

    met = select_met_levels(met, levels=relevant_levels)
    met = select_met_variables(met, variables=relevant_variables)

    first_var = list(met.data_vars)[0]
    if met[first_var].dtype != "float32":
        if verbose:
            print(f"casting met to float32 from {met[first_var].dtype}")
        met = met.astype("float32")

    assert time_delta >= 0, "time_delta must be zero or positive"
    met_times = met.indexes["time"]
    met, time_positions, sample_coords, nan_idxs = _match_met_times_to_fp(
        met, fp, time_delta, interp_method, closest_tolerance)

    # if nothing matched there is no point cropping — the footprints and the met do
    # not overlap in time
    if len(nan_idxs) == fp.sizes["sample_id"]:
        raise ValueError(
            "No temporal overlap between the footprints and the meteorology: no met data "
            f"could be matched to any footprint at time_delta={time_delta}h.\n"
            f"  footprints:  {pd.Timestamp(fp.time.values.min())} to {pd.Timestamp(fp.time.values.max())}\n"
            f"  meteorology: {met_times.min()} to {met_times.max()}"
        )

    met = met.assign_coords({"time_delta": ("time_delta", [time_delta])})

    domain_lats = met.lat.values.copy()
    domain_lons = met.lon.values.copy()
    delta_lat = domain_lats[1] - domain_lats[0]
    delta_lat_fp = fp.lat.values[1] - fp.lat.values[0]
    delta_lon = domain_lons[1] - domain_lons[0]
    delta_lon_fp = fp.lon.values[1] - fp.lon.values[0]
    if abs(delta_lat - delta_lat_fp) > 0.01 or abs(delta_lon - delta_lon_fp) > 0.01:
        print("Warning: met and fp resolutions differ — cropping may be inaccurate")

    release_idxs = _get_release_idxs(fp, domain_lats, domain_lons)
    met, domain_lats, domain_lons, release_idxs = _pad_domain(
        met, fp, release_idxs, half, pad_mode, verbose=verbose)

    # build integer index arrays: shape (n_samples, metsize)
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(metsize)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(metsize)[None, :]

    # all three indexers carry the sample_id dim, so xarray broadcasts them into a single
    # pointwise selection: every sample reads its own metsize x metsize block straight out
    # of its own met timestep. Selecting the time first would materialise a full-domain
    # field per sample before cropping it away, which for receptor data is one field per
    # (time, receptor) pair rather than per timestep.
    sample_vals = sample_coords["sample_id"]
    time_da = xr.DataArray(time_positions, dims=["sample_id"], coords={"sample_id": sample_vals})
    lat_da = xr.DataArray(lat_indices, dims=["sample_id", "lat"], coords={"sample_id": sample_vals})
    lon_da = xr.DataArray(lon_indices, dims=["sample_id", "lon"], coords={"sample_id": sample_vals})

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        cropped_met = met.isel(time=time_da, lat=lat_da, lon=lon_da)

    # indexing the time axis leaves the met timestamps behind as a 'time' coordinate along
    # sample_id; met_timestamps holds the same thing with NaT for the unmatched samples
    cropped_met = (
        cropped_met
        .drop_vars("time")
        .assign_coords(lat=np.arange(metsize), lon=np.arange(metsize))
        .assign_coords(sample_coords)
        .assign({
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["sample_id", "lat"],
                coords={"sample_id": sample_vals}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["sample_id", "lon"],
                coords={"sample_id": sample_vals}),
        })
    )

    if add_wind_direction:
        try:
            if relevant_variables is None or "wind_angle" in relevant_variables:
                cropped_met["wind_angle"] = np.arctan2(
                    -cropped_met.x_wind, -cropped_met.y_wind)
            if relevant_variables is None or "wind_speed" in relevant_variables:
                cropped_met["wind_speed"] = np.sqrt(
                    cropped_met.x_wind ** 2 + cropped_met.y_wind ** 2)
            if verbose:
                print("calculated wind_angle and wind_speed from x_wind and y_wind")
        except Exception as e:
            print(f"Error adding wind variables: {e}")

    if attrs_dict is not None:
        attrs_dict.update({"original_met_attrs": cropped_met.attrs})
        cropped_met.attrs = attrs_dict

    #print("LOADING")
    #load=True
    if load:
        if verbose:
            print("!!!!!!!!")
            print("loading cropped met into memory")
        cropped_met.load()

    if save:
        assert savepath is not None, "pass a savepath to save the file"
        assert savepath.endswith(".nc"), "savepath must end with .nc"
        if verbose:
            print("saving met at", savepath)
        cropped_met.to_netcdf(savepath)
        if verbose:
            print("met saved")

    if return_nan_idxs:
        return cropped_met, nan_idxs
    return cropped_met


def getint(name):
    num = name.split('_')[-1]
    num = num.split('.')[0]
    return int(num)

"""
def grid_coordinates(side):
    xx, yy = np.meshgrid(np.array(list(range(side)), dtype=np.float32), np.array(list(range(side)), dtype=np.float32))
    z = np.empty((side**2, 2), np.float32)
    z[:, 1] = xx.reshape(side**2)
    z[:, 0] = yy.reshape(side**2)
    return z
"""

def cut_topog_data(topog_file, landcover_file, fp, size, pad_mode="zeros"):
    """
    Crops topography and landcover to a size x size square centred on each
    footprint's release point.  Returns an xarray Dataset with dimensions
    (sample_id, lat, lon) and variables:
        - topog: surface altitude, shape (sample_id, lat, lon)
        - landcover: integer landcover type, shape (sample_id, lat, lon)
        - disaggregated_landcover: (sample_id, lat, lon, landcover_level) with
          land_binary_mask inverted (sea=1, land=0) in level 0 and 9 fractional
          landcover types in levels 1–9
    lat and lon are artificial coordinates 0..size; actual geographic coordinates
    are stored in lat_coords (sample_id, lat) and lon_coords (sample_id, lon) variables.

    Parameters
    ----------
    topog_file : xarray.Dataset
        Topography dataset with variable surface_altitude and dimensions (lat, lon).
    landcover_file : xarray.Dataset
        Landcover dataset with variables landcover_type, land_binary_mask, and
        landcover_fraction (lat, lon, pseudo_level),with coordinates lat, lon, pseudo_level
    fp : xarray.Dataset
        Footprint dataset providing release_lat, release_lon, and time coordinates.
    size : int
        Side length of the square crop. Must be even.
    pad_mode : str
        How to pad if a crop escapes the topog domain: 'zeros' (default) or 'edge'.
    verbose : bool
        Print progress messages.
    """
    if size % 2 != 0:
        raise ValueError("size must be even so the release point is centred")
    half = size // 2

    domain_lats = topog_file.lat.values
    domain_lons = topog_file.lon.values
    release_idxs = _get_release_idxs(fp, domain_lats=domain_lats, domain_lons=domain_lons)

    # pad both topog and landcover using the same release indices
    topog_file, domain_lats, domain_lons, release_idxs = _pad_domain(
        topog_file, fp, release_idxs, half, pad_mode=pad_mode, verbose=False)
    if landcover_file is not None:
        landcover_file, _, _, _ = _pad_domain(
            landcover_file, fp,
            _get_release_idxs(fp, domain_lats=landcover_file.lat.values, domain_lons=landcover_file.lon.values),
            half, pad_mode=pad_mode, verbose=False)

    # integer index arrays: shape (n_times, size)
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]

    lat_da = xr.DataArray(lat_indices, dims=["sample_id", "lat"], coords={"sample_id": fp["sample_id"]})
    lon_da = xr.DataArray(lon_indices, dims=["sample_id", "lon"], coords={"sample_id": fp["sample_id"]})

    # isel on static (lat, lon) arrays — DataArray indexers introduce the sample_id dim
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        topog_crop  = topog_file.surface_altitude.isel(lat=lat_da, lon=lon_da)
        if landcover_file is not None:
            landcover_crop  = landcover_file.landcover_type.isel(lat=lat_da, lon=lon_da)
            sea_mask_crop   = landcover_file.land_binary_mask.isel(lat=lat_da, lon=lon_da)
            lc_frac_crop    = landcover_file.landcover_fraction.isel(lat=lat_da, lon=lon_da)

            # assemble disaggregated_landcover: level 0 = inverted sea mask, levels 1-9 = fractions
            # rename whatever the fractional landcover's last dim is called to "landcover_level"
            # sea level mask should be, like topog, not having a landcover_level dimension, so we expand it and concat along the new landcover_level dim

            frac_levels = lc_frac_crop.fillna(0.0).rename({"pseudo_level": "landcover_level"})
            sea_level = (1 - sea_mask_crop)
            sea_level = sea_level.assign_coords(landcover_level=0)
            #.expand_dims({"landcover_level": 1}, axis=-1)
            disagg = xr.concat([sea_level, frac_levels], dim="landcover_level")
    
    if landcover_file is None:
        result = xr.Dataset({
        "topog":                    topog_crop,
        "lat_coords": xr.DataArray(
            domain_lats[lat_indices], dims=["sample_id", "lat"], coords={"sample_id": fp["sample_id"]}),
        "lon_coords": xr.DataArray(
            domain_lons[lon_indices], dims=["sample_id", "lon"], coords={"sample_id": fp["sample_id"]}),
    })
    else:
        result = xr.Dataset({
            "topog":                    topog_crop,
            "landcover":                landcover_crop,
            "disaggregated_landcover":  disagg,
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["sample_id", "lat"], coords={"sample_id": fp["sample_id"]}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["sample_id", "lon"], coords={"sample_id": fp["sample_id"]}),
        })

    result = result.assign_coords(lat=np.arange(size), lon=np.arange(size))

    return result


def get_grid(fp_xr, reference_fp=0):
    """
    produce reference grid and node indeces.
    
    the grid is made from the lat/lon coordinates of the reference footprint, and the indices are centred around the reference footprint (i.e. the release point is at index (size//2, size//2)).

    Inputs:
    - fp_xr: a xarray Dataset containing the cropped footprint data, with variables lat_coords and lon_coords, and dimensions sample_id, lat, lon
    - reference_fp: the position along sample_id of the reference footprint to use for grid generation. Default is 0 (the first footprint).
    """
    if reference_fp is None:
        reference_fp = 0

    single_meshgrid = np.meshgrid(fp_xr.lat_coords.isel(sample_id=reference_fp), fp_xr.lon_coords.isel(sample_id=reference_fp))

    latlons = [(single_meshgrid[0][i,j], single_meshgrid[1][i,j]) for i in range(fp_xr.lon.size) for j in range(fp_xr.lat.size)]

    idx_meshgrid = np.meshgrid(fp_xr.lat.values, fp_xr.lon.values)
    idx_latlons = [(idx_meshgrid[0][i,j], idx_meshgrid[1][i,j]) for i in range(fp_xr.lon.size) for j in range(fp_xr.lat.size)]

    return latlons, idx_latlons





