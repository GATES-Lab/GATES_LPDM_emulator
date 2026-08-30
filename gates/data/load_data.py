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
import time

import cartopy.crs as ccrs
import cartopy

from .load_data_helper_funs import *

from gates.config import get_config


def _rename_latlon(ds):
    """Rename latitude→lat and longitude→lon if those names are present as dims.

    Args:
        ds (xr.Dataset or xr.DataArray): Dataset/DataArray to rename dims on.

    Returns:
        xr.Dataset or xr.DataArray: ``ds`` with "latitude"/"longitude" dims renamed
        to "lat"/"lon" where present; unchanged otherwise.
    """
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds


def _wrap_longitudes(ds):
    """Convert longitudes from 0-360 to -180 to 180 if any values exceed 180.

    Detects the coordinate name automatically (lon or longitude). Sorts by the
    longitude coordinate after wrapping so the grid stays monotonic.

    Args:
        ds (xr.Dataset or xr.DataArray): Dataset/DataArray with a "lon" or
            "longitude" coordinate.

    Returns:
        xr.Dataset or xr.DataArray: ``ds`` with longitudes wrapped to [-180, 180]
        and sorted, or unchanged if no longitude coordinate is found or none exceed 180.
    """
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords), None)
    if lon_name is None or float(ds[lon_name].max()) <= 180:
        return ds
    ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
    return ds.sortby(lon_name)


def _round_time_to_seconds(ds):
    """Preprocess function: round time coordinate to nearest second to avoid sub-millisecond floating-point jitter."""
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds.time.dt.round("s"))
    return ds


def load_fps(fp_datadir, verbose=False, chunk=True, parallel_loading=False, drop_variables_except=None, bad_files_list=None):
    """Load footprints from datadir, using workaround if problematic files are encountered.

    Will throw an error if ANY of the specified files is problematic and NOT on the
    bad_files list. Note that the list of problematic files is currently updated manually!

    Args:
        fp_datadir (str or Path): String or path pointing to the directory with footprints
            to load, including special characters (e.g. "/path/to/footprints/*.nc", or
            "/path/to/footprints/*2020*.nc"). Note that ``fp_datadir`` is passed directly
            to glob, so it needs to specify filetype (i.e. finish with .nc).
        verbose (bool, optional): If True, prints out the steps throughout the data
            loading process. Defaults to False.
        chunk (bool, optional): If True, loads the data with dask chunking, which can
            help with memory issues but can cause some problems with certain files.
            Defaults to True.
        parallel_loading (bool, optional): If True, uses dask to load the files in
            parallel, which can speed up loading but can cause some problems with
            certain files. Parallel loading can only be used if ``chunk=True``.
            Defaults to False.
        drop_variables_except (list, optional): If specified, only keeps the listed
            variables in the dataset. Defaults to None.
        bad_files_list (list, optional): If specified, a list of files that are known
            to be problematic and should be skipped. If None, fetches list from config.
            Pass an empty list to not skip any files. Defaults to None.

    Returns:
        xr.Dataset: The footprints specified in ``fp_datadir``, opened correctly.

    Note:
        Potential improvement: add capability to ignore any files that couldn't be
        opened, and return only the successful files.
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
            fp_data_full = xr.open_mfdataset(sorted(glob.glob(str(fp_datadir))), combine='by_coords', preprocess=_round_time_to_seconds, **chunk_args)

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
                most = xr.open_mfdataset(sorted(without_bad_files), preprocess=_round_time_to_seconds, **chunk_args)
                bad_arrays = []
                for badfile in bad_files_list:
                    if badfile in fp_files:
                        print("loading bad file with workaround:", badfile)
                        # load each bad file separately
                        f_bad = xr.open_mfdataset(badfile, preprocess=_round_time_to_seconds, **chunk_args)
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

    if drop_variables_except is not None:
        vars_to_drop = [var for var in fp_data_full.data_vars if var not in drop_variables_except]
        fp_data_full = fp_data_full.drop(vars_to_drop)

    return fp_data_full


def remove_duplicates(ds, dim="longitude"):
    """Remove duplicate values along a specified dimension in an xarray Dataset.

    Args:
        ds (xr.Dataset): Dataset to remove duplicates from.
        dim (str or list, optional): Dimension(s) along which to check for duplicates.
            Defaults to "longitude".

    Returns:
        xr.Dataset: ``ds`` with duplicates removed along ``dim``.
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
    """Preprocess meteorology data by renaming latitude/longitude coordinates and removing duplicates.

    Args:
        ds (xr.Dataset): Met dataset to preprocess.
        duplicate_dim (str, optional): Dimension along which to check for duplicates,
            after renaming latitude/longitude to lat/lon. Defaults to "longitude".

    Returns:
        xr.Dataset: ``ds`` with latitude/longitude renamed to lat/lon (if present) and
        duplicates removed along ``duplicate_dim``.
    """
    ds = _rename_latlon(ds)
    if duplicate_dim in ds.dims:
        ds = remove_duplicates(ds, dim=duplicate_dim)
    elif duplicate_dim == "longitude" and "lon" in ds.dims:
        ds = remove_duplicates(ds, dim="lon")
    return ds


class LoadBaseSatelliteData:
    """Parent class for loading satellite data.

    Loads footprint, meteorology, and topography/landcover data for a given domain
    and time period.

    Paths:
        The paths to the files are by default loaded from the config file, where the
        file structure is expected to be ``path/to/footprints/domain/region_*domain*_yearmonth.nc``
        for the footprints (one NetCDF file per month/year), and
        ``path/to/meteorology/domain/domain_Met_year.zarr`` for the meteorology (one
        Zarr store per year). The config paths are superseded by passing the paths as
        arguments, as strings or path objects:

        - ``fp_datadir`` for the footprints, which should point to the folder
          containing the files for each month/year with format
          example_name_yearmonth.nc (e.g. brazil_201601.nc), so that the date can be
          automatically added.
        - ``met_args = {"met_datadir": "path/to/meteorology/domain/domain_Met_"}``,
          which should point to the folder containing the yearly Zarr stores with
          format example_name_year.zarr (e.g. brazil_2016.zarr), so that the year can
          be automatically added.
        - ``topog_args = {"topog_path": "path/to/topography/file.nc", "landcover_path": "path/to/landcover/file.nc"}``,
          which should point to the specific files for topography and landcover.
          These files will be interpolated to the same resolution and domain as the
          footprints, so they can be from a different source and with a different
          original resolution.

    Attributes:
        fp_data_full (xr.Dataset): The footprints specified in ``fp_datadir``, opened
            correctly, and subsampled according to the ``freq`` and ``sampling_mode``
            parameters. Dimensions are time, lat and lon.
        met_file (xr.Dataset): The meteorology data, opened correctly and with
            selected levels and variables if specified. Dimensions are time, levels,
            lat and lon. Only set if ``load_everything=True`` or after calling
            ``load_meteorology()``.
        topog_file (xr.Dataset): The topography data, opened correctly and
            interpolated to the same grid as the footprints. Dimensions are lat and
            lon. Only set if ``load_everything=True`` or after calling ``load_topog()``.
        landcover_file (xr.Dataset): The landcover data, opened correctly and
            interpolated to the same grid as the footprints. Dimensions are lat and
            lon. Only set if ``load_everything=True`` or after calling ``load_topog()``.
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, freq=1, freq_offset=0, verbose = False, sampling_mode="regular", fp_datadir = None, load_everything=False, met_args={}, topog_args={}, cfg=None, parallel_loading=False):
        """Initialize the class and load footprint data (and optionally meteorology/topography).

        Args:
            year (int or str): Year to load, e.g. 2016, or a string including
                combinations of years (e.g. "2016", "201[4-5]").
            region (str, optional): Region identifier. Current set-up has regions
                "BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA" (note - Brazil is a
                subset of South America!). Defaults to "BRAZIL".
            month (str, optional): Month in format "01" for January etc. None to load
                a whole year. Defaults to None.
            domain (str, optional): Domain related to the region, used for file search
                (due to existing filenaming conventions). Set-up regions ("BRAZIL",
                "SOUTHAMERICA", "SAHARA" and "INDIA") have a default domain, all
                others need domain passed. Defaults to None.
                Note: update dict of region-domain using config rather than being hard-coded.
            freq (int, optional): Frequency of the data to load. freq=1 loads all
                datapoints, freq=2 loads one in every two, etc. Useful to reduce
                memory usage. Many datapoints are very close in time and space (and
                therefore very similar) so using freq particularly at low values
                (<10) does not affect the quality of the dataset much for testing.
                Defaults to 1.
            freq_offset (int, optional): If ``sampling_mode="regular"``, offsets the
                start of the regular sampling, e.g. freq=2 and freq_offset=0 will
                sample even footprints, and freq_offset=1 will sample uneven
                footprints. Defaults to 0.
            verbose (bool, optional): If True, prints out the steps throughout the
                data loading process. Defaults to False.
            sampling_mode (str, optional): If "regular", subsamples footprints
                regularly (e.g. one in every two, sequentially with freq=2). If
                "random", subsamples N/freq footprints randomly (where N is the total
                number of footprints). Defaults to "regular".
            fp_datadir (str or Path, optional): Path to the footprints; see the class
                "Paths" docs. If None, constructed from the config file. Defaults to None.
            load_everything (bool, optional): If True, loads all data (footprints,
                meteorology and topography) at once when initializing the class. If
                False, only loads footprints, and meteorology and topography can be
                loaded later with ``load_meteorology()`` and ``load_topog()``.
                Defaults to False.
            met_args (dict, optional): Arguments for ``load_meteorology()``; see that
                method and the class "Paths" docs. Defaults to {}.
            topog_args (dict, optional): Arguments for ``load_topog()``; see that
                method and the class "Paths" docs. Defaults to {}.
            cfg (gates.config.Config, optional): Config object containing the config
                values. If None, the config will be loaded from the config file. If
                passed, this supersedes loading from the config file, and allows you
                to pass a custom config object with custom paths and settings.
                Defaults to None.
            parallel_loading (bool, optional): Whether to load fp data in parallel using dask. Defaults to False.

        Raises:
            FileNotFoundError: If no config is available and one is needed to resolve
                missing paths.
            ValueError: If ``cfg`` is neither None nor a ``gates.config.Config`` instance.

        """
        self.dataset_format = "base"
        self.data_type="satellite"

        self.parallel_loading = parallel_loading

        needs_cfg = (
             domain is None
             or fp_datadir is None
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
            self.domain=domain

        self.year = year
        self.date = self.year
        self.verbose=verbose

        if month is not None:
            if not isinstance(month, str):
                month = str(month).zfill(2)
            self.month = month
            self.date = str(self.year)+month
        
        # prepare the paths to load the data, using the config values as default and superceded by any arguments passed to the function
        self._resolve_paths(self.cfg, fp_datadir=fp_datadir, met_args=met_args, topog_args=topog_args)


        self.subsample_parameters = {"freq":freq, "sampling_mode":sampling_mode, "freq_offset":freq_offset}
        
        #### load footprint (fp) data     
        if verbose: print("---- LOADING FOOTPRINTS")  
        self._load_footprints(self.fp_datadir)

        self.met_processed = False

        
        self.padding=None

        if load_everything:
            self.met_file = self.load_meteorology(**self.met_args)
            self.topog_file, self.landcover_file = self.load_topog(**self.topog_args)

        
        if verbose: print("---- All done!")

    def _resolve_paths(self, cfg, fp_datadir=None, met_args={}, topog_args={}):
        """Prepare the necessary loading paths.

        If paths are not passed as arguments, they will be constructed from the
        config file values. For the footprint and the meteorology paths, the config
        data is expected to point at a folder, which contains a folder for each
        domain, which in turn contains the files for each month/year with format
        example_name_yearmonth.nc (e.g. brazil_201601.nc). The function will
        construct the path to point directly to the files, including the date.

        If paths are passed as arguments, they will be used directly (but the date
        will still be added automatically, so the files should have format
        example_name_yearmonth.nc (e.g. brazil_201601.nc) and you should pass
        ``met_datadir="/path/example_name_"``).

        The topography and landcover paths should point to a specific file, either
        through the config file or through the arguments.

        Args:
            cfg (gates.config.Config or None): Config object to fall back to for any
                paths not passed as arguments.
            fp_datadir (str or Path, optional): Footprint directory/prefix. Defaults to None.
            met_args (dict, optional): Must contain "met_datadir" if overriding the
                config value. Defaults to {}.
            topog_args (dict, optional): May contain "topog_path" and/or
                "landcover_path" to override the config values. Defaults to {}.
        """
        if fp_datadir is None:
            self.fp_datadir = Path(cfg.fp_datadir) / self.domain / f"*{self.region}*{self.domain}_{str(self.date)}*.nc"
        else:
            self.fp_datadir=Path(str(fp_datadir)+ f"*{str(self.date)}*.nc")

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
    


    def load_meteorology(self, met_datadir=None, met_levels = [], met_variables= [], lazy_load=True):
        """Load zarr meteorology and select the met levels and variables if required.
        
        Args:
            met_datadir (str, optional): Prefix/path for the meteorology Zarr store(s).
                Default directs to the configured meteorology folder. If passing as an
                argument, the year is automatically appended (as ``*year*.zarr``), so the
                stores should have format name_of_your_choice_year.zarr (e.g.
                brazil_2016.zarr) and you should pass
                ``met_datadir="/path/name_of_your_choice_"``. Defaults to None.
            met_levels (list, optional): Met levels to select. Defaults to [].
            met_variables (list, optional): Met variables to select. Defaults to [].
                Derived variables (wind_speed / wind_angle) are ignored here and computed later downstream.
            lazy_load (bool, optional): If True, does not load the met data into
                memory (lazy array); if False, loads the met data into memory.
                Defaults to True.

        Returns:
            xr.Dataset: The meteorology as an xarray Dataset, assigned to ``self.met_file``.

        """
        if self.verbose: print("\n ---- LOADING MET")

        # 1) load from file — level/variable selection happens inside, using the
        #    tolerant select_met_* helpers.
        self.met_file = self._get_meteorology_file(
            met_datadir,
             met_levels=met_levels, met_variables=met_variables,
        )

        # 2) check domain overlap
        self._check_domain_overlap(self.fp_data_full, self.met_file, "footprint", "meteorology")

        if not lazy_load:
            print("Loading met data into memory. If you only want to lazy-load, pass load=False")
            self.met_file.load()

        return self.met_file

    def load_topog(self, topog_path=None, landcover_path=None):
        """Load the topography and landcover files, and interpolate to the same resolution and domain as the footprints in self.fp_data_full.

        Uses object attribute ``self.padding`` (contains if any amount of padding is
        needed to the footprint domain, and in which direction).

        Args:
            topog_path (str, optional): Path to the topography file, or "default" for
                the default file. If None, uses ``self.topog_args["topog_path"]``.
                Defaults to None.
            landcover_path (str, optional): Path to the landcover file, or "default"
                for the default file. If None, uses
                ``self.topog_args["landcover_path"]``. Defaults to None.

        Returns:
            tuple:
                - topog_file (xr.Dataset): Topography interpolated to the footprint grid.
                - landcover_file (xr.Dataset or None): Landcover interpolated to the
                  footprint grid, or None if no landcover path is available.

        Raises:
            ValueError: If the topography or landcover file does not exist at the
                resolved path.
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

    def _get_meteorology_file(self, met_datadir, met_levels=[], met_variables=[]):
        """Load the meteorology from one or more yearly Zarr stores, concatenating along time.

        ``met_datadir`` is a glob pattern (e.g. ``DOMAIN_Met_2016*.zarr``) resolved
        to a sorted list of stores. Usually this is a single year, but a year
        pattern such as "201[4-5]" resolves to several stores opened together. If
        a single month was requested (``self.month`` is set), the loaded met is
        sliced to that month; whole-year loads keep every timestamp. Level and
        variable selection is delegated to the tolerant ``select_met_levels`` /
        ``select_met_variables`` helpers, which skip missing levels and the
        derived variables (wind_speed / wind_angle) that are computed later
        downstream.

        Args:
            met_datadir (str or Path or None): Glob pattern for the meteorology
                Zarr store(s).
            met_levels (list, optional): Met levels to select. Missing levels are
                skipped. Defaults to [].
            met_variables (list, optional): Met variables to select. Missing
                variables and derived wind_speed/wind_angle are skipped. Defaults to [].

        Returns:
            xr.Dataset: The loaded meteorology, assigned to ``self.met_file``.

        Raises:
            ValueError: If no meteorology Zarr stores match ``met_datadir``.
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
        """Look up the domain name associated with ``region`` in the config.

        Args:
            region (str): Region identifier to look up.
            cfg (gates.config.Config, optional): Config object with a ``domains``
                dict mapping region to domain info. Defaults to None.

        Returns:
            str: Domain name associated with ``region``.

        Raises:
            ValueError: If ``cfg`` is None / has no ``domains`` dict, or if ``region``
                has no associated domain in the config.
        """
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
        """Check that data2 (e.g., meteorology) has sufficient spatial coverage of data1 (e.g., footprint).

        Raises an error if there is no overlap, and warns if data2 is smaller than data1.

        Args:
            data1 (xr.Dataset or xr.DataArray): Reference dataset (typically
                footprint). Expected to have ``release_lat`` and ``release_lon``.
            data2 (xr.Dataset or xr.DataArray): Dataset to check (typically
                meteorology, topography, or landcover).
            data1_name (str, optional): Name of data1 for messages. Defaults to "footprints".
            data2_name (str, optional): Name of data2 for messages (e.g., "meteorology").
                Defaults to "data2".

        Raises:
            ValueError: If the spatial domains do not overlap at all.

        Warns:
            UserWarning: If data2's domain is noticeably smaller than data1's,
                suggesting alignment may be needed.
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
        """Load footprints from ``fp_datadir`` and apply subsampling according to the freq and sampling_mode parameters.

        The footprints are stored in ``self.fp_data_full`` as an xarray dataset, with
        dimensions time, lat and lon.

        Args:
            fp_datadir (str or Path): Directory/glob pattern to load footprints from.
            load_fps_in_mem (bool, optional): If True, loads the ``fp`` variable into
                memory. Defaults to True.
        """
        #### load footprint (fp) data from file
        if self.verbose: print("Loading footprint data from " + str(fp_datadir) )

        self.fp_data_full = load_fps(fp_datadir, verbose=self.verbose, parallel_loading=self.parallel_loading, drop_variables_except=["fp", "release_lat", "release_lon"])

        self.fp_data_full = self.fp_data_full.drop_duplicates(dim="time")

        ## reduce data frequency with regular sampling 9eg keep only 1 in every 3 timesteps
        # uses the sampling_mode and freq parameters
        self._subsample_frequency(**self.subsample_parameters)

        #print(self.fp_data_full)
        if self.verbose: print(f"Loading {len(self.fp_data_full.time.values)} footprints")
        if load_fps_in_mem:
            self.fp_data_full.fp.load()
            print("loaded fp variable into mem")
        #print(self.fp_data_full)
        #self.fp_data_full = self.fp_data_full.chunk({"lat": -1, "lon": -1, "time": "auto"})



        
    
    def _subsample_frequency(self, freq=1, sampling_mode="regular",freq_offset=0):
        """Subsample the footprint data (in place, on ``self.fp_data_full``).

        Selects every freq-th timestamp from the original dataset, starting from the
        timestamp specified by ``freq_offset`` (if ``sampling_mode`` is "regular"), or
        by randomly selecting N/freq timestamps from the original dataset (if
        ``sampling_mode`` is "random"). If ``freq=1``, no subsampling is done and all
        footprints are loaded.

        Args:
            freq (int, optional): Subsampling frequency. Defaults to 1.
            sampling_mode (str, optional): "regular" or "random"; see above. Defaults
                to "regular".
            freq_offset (int, optional): Offset for the start of regular sampling.
                Defaults to 0.
        """
        # subsample the footprint data according to a particular sampling mode
        self.original_fp_time_length = len(self.fp_data_full.time.values)
        if freq>1 and sampling_mode=="regular":
            print(f"reduced the number of datapoints by frequency {freq}")
            self.fp_data_full = self.fp_data_full.sel(time=self.fp_data_full.time.values[freq_offset::freq])
        
        elif freq>1 and sampling_mode=="random":
            print(f"reduced the number of datapoints by frequency {freq}, chosen at random")
            self.fp_data_full = self.fp_data_full.sel(time=np.random.choice(self.fp_data_full.time.values, size=np.shape(self.fp_data_full.time.values[::freq]), replace=False))
        else:
            if self.verbose: print("no sampling was done because you didnt pass a valid sampling mode, or freq=1")

    def _interp_topog(self, topog_file, padding=None):
        """Interpolate topography to the footprint resolution.

        Assumes the same resolution and domain as the footprints, unless ``padding``
        is passed as padded coordinates (tuple with shape (lat_values, lon_values)).

        Args:
            topog_file (xr.Dataset): Topography dataset to interpolate.
            padding (tuple, optional): ``(lat_values, lon_values)`` to interpolate to
                instead of the footprint grid. Defaults to None.

        Returns:
            xr.Dataset: Topography interpolated onto the footprint (or padded) grid.
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
        """Interpolate landcover to the footprint resolution.

        Assumes the same resolution and domain as the footprints, unless ``padding``
        is passed as padded coordinates (tuple with shape (lat_values, lon_values)).

        Args:
            landcover_file (xr.Dataset): Landcover dataset to interpolate.
            padding (tuple, optional): ``(lat_values, lon_values)`` to interpolate to
                instead of the footprint grid. Defaults to None.

        Returns:
            xr.Dataset: Landcover interpolated onto the footprint (or padded) grid,
            transposed to (lat, lon, pseudo_level).
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
        """Align domains for the meteorology, footprints, and optionally topography and landcover.

        Args:
            crop_to_intersection (bool, optional): If False, interpolates
                ``met_file`` to the full footprint grid using nearest neighbour, even
                if ``met_file`` is smaller. This preserves all footprint pixels but
                risks artefacts in the interpolated meteorology. If True (default),
                crops BOTH datasets to the spatial intersection BEFORE interpolating.
                This removes footprint pixels outside the met domain but avoids
                artefacts. Defaults to True.
            include_topo_and_landcover (bool, optional): Whether to also align
                topography and landcover files. Defaults to True.
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
        """Load country mask for the domain, and create a land-sea mask.

        Interpolates both to the same resolution and domain as the footprints in
        ``self.fp_data_full``. Stores the resulting dataset in ``self.countries``,
        with a ``country_mask`` variable and a ``land_mask`` variable.

        Args:
            countrymask_path (str, optional): Path to the country mask file, or
                "default" to use the default ACRG path for ``self.domain``.
                Defaults to "default".
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


    def plot_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, return_fig=False, plot_marker=False, dpi=100, figsize=(6,6)):
        """Plot a footprint for a particular timestamp or index.

        Args:
            idx (int, optional): Index of the footprint to plot. If ``timestamp`` is
                also passed, ``timestamp`` is used instead of ``idx``. Defaults to 0.
            timestamp (str, optional): Timestamp of the footprint to plot, as a string
                in format "YYYY-MM-DDTHH:MM:SS" (e.g. "2016-01-01T12:00:00"). If
                passed, takes precedence over ``idx``. Defaults to None.
            vmin_vmax (list, optional): ``[vmin, vmax]`` color scale limits passed to
                the plot. Defaults to [None, None].
            levels (list, optional): Contour levels (in log10 space) to plot.
                Defaults to None, which uses ``[-4, -3.5, -3, -2.5, -2, -1.5]``.
            background_threshold (float, optional): Values below this threshold are
                set to 0 for the foreground (non-transparent) contour layer.
                Defaults to 1e-4.
            add_cbar (bool, optional): If True, adds a colorbar to the plot.
                Defaults to False.
            return_fig (bool, optional): If True, returns the fig and ax objects
                instead of showing the plot. Defaults to False.
            plot_marker (bool, optional): If True, marks the release point on the plot.
                Defaults to False.
            dpi (int, optional): Figure resolution. Defaults to 100.
            figsize (tuple, optional): Figure size. Defaults to (6, 6).

        Returns:
            tuple or None: ``(fig, ax)`` if ``return_fig`` is True, otherwise None
            (the plot is shown directly).
        """
        import matplotlib.pyplot as plt

        if timestamp is not None:
            fp_to_plot = self.fp_data_full.sel(time=np.datetime64(timestamp)).copy()
            if len(fp_to_plot.time.values)>1:
                print("there are multiple footprints for the timestamp you passed, check the timestamp and try again! plotting the first one")
                fp_to_plot = fp_to_plot.isel(time=0)
        
        else:
            fp_to_plot = self.fp_data_full.isel(time=idx).copy()

        f = np.copy(fp_to_plot.fp.values)

        extent = (fp_to_plot.lon.values[0], fp_to_plot.lon.values[-1], fp_to_plot.lat.values[0], fp_to_plot.lat.values[-1])

        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize, dpi=dpi)
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        ax.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
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
        f[f<background_threshold] = 0
        cb = ax.contourf(fp_to_plot.lon.values, fp_to_plot.lat.values,np.log10(f), **plot_params, levels=levels, extend="both")
        formatted_time = fp_to_plot.time.values.astype('datetime64[ms]').astype('O').strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]
        ax.set_title(formatted_time)

        if plot_marker:
            ax.scatter(fp_to_plot.release_lon.values, fp_to_plot.release_lat.values, marker="x", color="white",s=25, lw=1,transform=cartopy.crs.PlateCarree(), zorder=10)

        if add_cbar:
            cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both", shrink=0.55).set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)

        if return_fig:
            return fig, ax
        else:
            plt.show()

class LoadSquareSatelliteData(LoadBaseSatelliteData):
    """Load footprint and meteorological data for a particular domain and time period.

    Outputs all data cut to a square centered around the measurement point for each
    timestamp. Inherits the loading functions from the general ``LoadBaseSatelliteData``.

    Paths:
        The paths to the files are by default loaded from the config file, where the
        file structure is expected to be ``path/to/footprints/domain/region_*domain*_yearmonth.nc``
        for the footprints (one NetCDF file per month/year), and
        ``path/to/meteorology/domain/domain_Met_year.zarr`` for the meteorology (one
        Zarr store per year). The config paths are superseded by passing the paths as
        arguments, as strings or path objects:

        - ``fp_datadir`` for the footprints, which should point to the folder
          containing the files for each month/year with format
          example_name_yearmonth.nc (e.g. brazil_201601.nc), so that the date can be
          automatically added.
        - ``met_args = {"met_datadir": "path/to/meteorology/domain/domain_Met_"}``,
          which should point to the folder containing the yearly Zarr stores with
          format example_name_year.zarr (e.g. brazil_2016.zarr), so that the year can
          be automatically added.
        - ``topog_args = {"topog_path": "path/to/topography/file.nc", "landcover_path": "path/to/landcover/file.nc"}``,
          which should point to the specific files for topography and landcover.
          These files will be interpolated to the same resolution and domain as the
          footprints, so they can be from a different source and with a different
          original resolution.
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, verbose = False, fill_outofdomain_with="nans", delete_outofdomain=False, sampling_mode="regular", fp_datadir = None, load_everything=True, lazy_load=True, met_args={}, topog_args={}, cfg=None, parallel_loading=False, crop_met=True, load_fps_in_mem=True):
        """Initialize the class and load (and, by default, process) footprint, meteorology, and topography data.

        Args:
            year (int or str): Year to load, e.g. 2016, or a string including
                combinations of years (e.g. "2016", "201[4-5]").
            region (str, optional): Region identifier. Defaults to "BRAZIL".
            month (str, optional): Month in format "01" for January etc. None to load
                a whole year. Defaults to None.
            domain (str, optional): Spatial domain related to the region. It can be
                extracted automatically from the config file if the region-domain
                pair is specified there, or it can be passed directly as an argument.
                The domain is used to find the files to load. Defaults to None.
            size (int, optional): Size for footprint to be cut to. Resolution of the
                footprint is maintained, cut to a size x size square around the
                release point. Defaults to 10.
            freq (int, optional): Frequency of the data to load. freq=1 loads all
                datapoints, freq=2 loads one in every two, etc. Useful to reduce
                memory usage. Many datapoints are very close in time and space (and
                therefore very similar) so using freq particularly at low values
                (<10) does not affect the quality of the dataset much for testing.
                Defaults to 1.
            freq_offset (int, optional): If ``sampling_mode="regular"``, offsets the
                start of the regular sampling, e.g. freq=2 and freq_offset=0 will
                sample even footprints, and freq_offset=1 will sample uneven
                footprints. Defaults to 0.
            verbose (bool, optional): If True, prints out the steps throughout the
                data loading process. Defaults to False.
            fill_outofdomain_with (str, optional): One of "nans" or "zeros".
                Determines what to do if any part of the square cut around the
                footprint is outside of the domain — fills only the out-of-domain
                areas with NaNs or zeros respectively. Defaults to "nans".
            delete_outofdomain (bool, optional): If True, deletes all footprints (and
                associated datapoints) where the extracted size x size area escapes
                the domain. If False, the cut footprints are kept and the
                out-of-domain areas are filled according to
                ``fill_outofdomain_with``. Defaults to False.
            sampling_mode (str, optional): If "regular", subsamples footprints
                regularly (e.g. one in every two, sequentially with freq=2). If
                "random", subsamples N/freq footprints randomly (where N is the total
                number of footprints). Defaults to "regular".
            fp_datadir (str or Path, optional): Path to the footprints; see class
                "Paths" docs. If None, constructed from the config file. Defaults to None.
            load_everything (bool, optional): If True, loads all data (footprints,
                meteorology and topography) at once when initializing the class. If
                False, only loads footprints, and meteorology and topography can be
                loaded later with ``load_meteorology()`` and ``load_topog()``.
                Defaults to True.
            lazy_load (bool, optional): If True, does not load the met data into
                memory (lazy array); if False, loads the met data into memory.
                Defaults to True.
            met_args (dict, optional): Arguments for ``load_meteorology()``; see that
                method and the class "Paths" docs. Defaults to {}.
            topog_args (dict, optional): Arguments for ``load_topog()``; see that
                method and the class "Paths" docs. Defaults to {}.
            cfg (gates.config.Config, optional): Config object containing the config
                values. If None, the config will be loaded from the config file. If
                passed, this supersedes loading from the config file, and allows you
                to pass a custom config object with custom paths and settings.
                Defaults to None.
            parallel_loading (bool, optional): Whether to load fp data in parallel using dask. Defaults to False.
            crop_met (bool, optional): If True, crops the meteorology to a square of
                size x size around the measurement location, following the format of
                the footprint, and stores it as ``self.met``. If False, the
                meteorology is only loaded and preprocessed, but not cropped. The
                inputs function calculates the domain directly from the loaded
                (uncropped) ``self.met_file``, so cropping the meteorology is
                redundant when using the inputs function. Defaults to True.
            load_fps_in_mem (bool, optional): If True, loads the footprint ``fp``
                variable into memory. Defaults to True.

        Raises:
            FileNotFoundError: If no config is available and one is needed to resolve
                missing paths.
            ValueError: If ``cfg`` is neither None nor a ``gates.config.Config`` instance.
        """
        print(dask.__version__)

        self.dataset_format = "square" 
        self.data_type = "satellite"

        self.parallel_loading = parallel_loading
        
        needs_cfg = (
             domain is None
             or fp_datadir is None
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
            self.domain=domain

        self.size = size

        self.fill_outofdomain_with = fill_outofdomain_with
        self.delete_outofdomain = delete_outofdomain

        self.date = year
        ### todo need to implement multiple years! 
        if len(str(year)) >4 and "[" not in str(year):
            self.year = int(str(year)[:4])
        elif "[" in str(year):
            self.year = str(year)
        else:
            self.year = int(year)

        self.verbose=verbose


        if month is not None:
            if not isinstance(month, str):
                month = str(month).zfill(2)
            self.month = month
            self.date = str(self.year)+month

        # prepare the paths to load the data, using the config values as default and superceded by any arguments passed to the function
        self._resolve_paths(self.cfg, fp_datadir=fp_datadir, met_args=met_args, topog_args=topog_args)

        self.subsample_parameters = {"freq":freq, "sampling_mode":sampling_mode, "freq_offset":freq_offset}
        
        #### load footprint (fp) data, subsample, crop
        if verbose: print("---- LOADING FOOTPRINTS") 
        self._load_footprints(self.fp_datadir, load_fps_in_mem=load_fps_in_mem)
        self._process_footprints(lazy_load)

        self.met_processed = False

        if load_everything:
            self.met_file = self.load_meteorology(**self.met_args,lazy_load=lazy_load)
            self.topog_file, self.landcover_file = self.load_topog(**self.topog_args)
            self.topog = self._process_topog_and_landcover()
            if crop_met:
                self.met = self._process_meteorology(lazy_load=True)
            else:
                print("NOT cropping met!")
        if verbose: print("---- All done!")       

    def _process_footprints(self, lazy_load):
        """Cut footprint data to a square around each release point.

        Sets ``self.fp_xr`` (a full xarray dataset with coordinates time, lat, lon,
        where lat and lon are artificial coordinates with range (0, size) and the
        measurement point is in the center at size//2, size//2) and updates
        ``self.fp_data_full``. If the square to extract escapes the footprint domain,
        the padded space is filled with NaNs or zeros, or the datapoint is deleted,
        according to ``self.fill_outofdomain_with`` and ``self.delete_outofdomain``.

        Args:
            lazy_load (bool): If False, loads the cut data into memory immediately.
        """
        if self.verbose: print(f"----- Cutting footprints to square of size {self.size}") 
        self.fp_xr, self.fp_data_full, self.release_idxs, padded_domain_coords = cut_satellite_data(self.fp_data_full, self.size, fill_bads_with=self.fill_outofdomain_with, delete_outofdomain = self.delete_outofdomain, verbose=self.verbose, load=not lazy_load) 
        ## for now!
        self.padded_domain_coords = padded_domain_coords

    def _process_meteorology(self,rechunk=0,lazy_load=True, pad_mode="edge"):
        """Cut the meteorology data around the release point, to the same size as the cut footprints.

        Calls ``cut_satellite_met``. Pads with ``pad_mode="edge"`` by default.

        Args:
            rechunk (int, optional): If > 0, rechunks ``self.met`` along the time
                dimension to this chunk size. Defaults to 0 (no rechunking).
            lazy_load (bool, optional): If False, loads the cut met data into memory.
                Defaults to True.
            pad_mode (str, optional): Padding mode passed to ``cut_satellite_met``.
                Defaults to "edge".

        Returns:
            xr.Dataset: The cropped meteorology, also stored in ``self.met``.
        """
        if self.verbose: print("----- Cutting met")
        self.metsize=self.size
        #if self.fill_outofdomain_with=="nans" or self.delete_outofdomain:
        #    pad_mode = "nans"
        #if self.fill_outofdomain_with=="zeros":

        self.met = cut_satellite_met(self.met_file, self.fp_data_full, metsize=self.size, time_delta=0, pad_mode=pad_mode, load=not lazy_load, add_wind_direction=True)

        if rechunk>0:
            self.met.chunk({"time":rechunk})
        
        self.met_processed = True
        return self.met

    
    def _process_topog_and_landcover(self):
        """Cut topography and landcover to a size x size square centred on each footprint's release point.

        Obsolete as standalone logic: delegates to ``cut_topog_data``.

        Returns:
            xr.Dataset: The cropped topography/landcover, also stored in ``self.topog``.
        """
        if self.verbose:
            print("----- Cutting topog")
        self.topog = cut_topog_data(
            self.topog_file, self.landcover_file, self.fp_data_full,
            self.size, pad_mode="zeros")
        return self.topog

    def get_flux(self, year=None, append_to_fp=True, search_others=True, convert_units=False, convert_units_args={}):
        """Load flux/emissions data for this domain and year, and crop it to match the footprints.

        Args:
        year (int, optional): Year to load. If None, uses ``self.year``. Defaults to None.
            append_to_fp (bool, optional): If True, adds the cropped flux as a "flux"
                variable on ``self.fp_xr``. Defaults to True.
            search_others (bool, optional): If True, searches for flux data in other
                directories if not found in the default directory. Defaults to True.
            convert_units (bool, optional): If True, converts the flux units with function ``transform_flux`` to match the footprint units. Defaults to False.
            convert_units_args (dict, optional): Arguments for ``transform_flux`` if ``convert_units`` is True. Defaults to {}.

        Returns:
            xr.Dataset: The cropped flux data, also stored in ``self.fluxes``.
        """
        if year is None:
            year = self.year
        ems = load_flux_data(self.domain, year=year, search_others=search_others)
        if convert_units:
            if len(convert_units_args) == 0:
                warnings.warn("convert_units is True but no convert_units_args were passed!")
            ems = transform_flux(ems, **convert_units_args)



        cropped_flux, nan_idxs = cut_flux_data(ems, self.fp_data_full, size=self.size)
        self.fluxes = cropped_flux
        self.remove_indeces(nan_idxs)
        if append_to_fp:
            self.fp_xr["flux"] = self.fluxes.flux
        return self.fluxes

    def remove_indeces(self, nan_idxs):
        """Remove a set of timestamps from all the objects in the dataset.

        Args:
            nan_idxs (array-like): Timestamps to drop from ``self.fp_data_full`` and,
                if present, ``self.fp_xr``, ``self.met``, ``self.topog``, and
                ``self.fluxes``.

        Raises:
            ValueError: If no data remains after removing ``nan_idxs``.
        """
        self.fp_data_full = self.fp_data_full.drop_sel(time=nan_idxs)
        if hasattr(self, "fp_xr"):
            self.fp_xr = self.fp_xr.drop_sel(time=nan_idxs)
        if hasattr(self, "met"): 
            self.met = self.met.drop_sel(time=nan_idxs)
        if hasattr(self, "topog"):
            self.topog = self.topog.drop_sel(time=nan_idxs)
        if hasattr(self, "fluxes"):
            self.fluxes = self.fluxes.drop_sel(time=nan_idxs)

        if self.verbose: print(f"Length after removing indeces: {self.fp_data_full.time.size}")
        # if the len is zero, raise an error
        if self.fp_data_full.time.size == 0:
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
    
    def plot_cropped_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, plot_wind=False, return_fig=False):
        """Plot a cropped footprint for a particular timestamp or index, optionally overlaying a wind arrow.

        Args:
            idx (int, optional): Index of the footprint to plot. If ``timestamp`` is
                also passed, ``timestamp`` is used instead of ``idx``. Defaults to 0.
            timestamp (str, optional): Timestamp of the footprint to plot, as a string
                in format "YYYY-MM-DDTHH:MM:SS" (e.g. "2016-01-01T12:00:00"). If
                passed, takes precedence over ``idx``. Defaults to None.
            vmin_vmax (list, optional): ``[vmin, vmax]`` color scale limits passed to
                the plot. Defaults to [None, None].
            levels (list, optional): Contour levels (in log10 space) to plot.
                Defaults to None, which uses ``[-4, -3.5, -3, -2.5, -2, -1.5]``.
            background_threshold (float, optional): Values below this threshold are
                set to 0 for the foreground (non-transparent) contour layer.
                Defaults to 1e-4.
            add_cbar (bool, optional): If True, adds a colorbar to the plot.
                Defaults to False.
            plot_wind (bool, optional): If True, overlays an arrow showing the wind
                direction/speed at the domain centre (requires ``self.met`` to be
                loaded). Defaults to False.
            return_fig (bool, optional): If True, returns the fig and ax objects instead of showing the plot. Defaults to False.
        """
        import matplotlib.pyplot as plt

        if timestamp is not None:
            fp_to_plot = self.fp_xr.sel(time=np.datetime64(timestamp)).copy()
            if len(fp_to_plot.time.values)>1:
                print("there are multiple footprints for the timestamp you passed, check the timestamp and try again! plotting the first one")
                fp_to_plot = fp_to_plot.isel(time=0)
            
            idx = np.where(self.fp_xr.time.values == fp_to_plot.time.values)[0][0]
        
        else:
            fp_to_plot = self.fp_xr.isel(time=idx).copy()
            timestamp = fp_to_plot.time.values

        f = np.copy(fp_to_plot.fp.values)
        fp_lats = self.fp_lats[idx]
        fp_lons = self.fp_lons[idx]

        extent = (fp_lons[0], fp_lons[-1], fp_lats[0], fp_lats[-1])

        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
        ax.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)

        cmap = plt.cm.Reds
        cmap.set_over = "k"
        plot_params = {"transform":cartopy.crs.PlateCarree(), "cmap":cmap, "vmin":vmin_vmax[0], "vmax":vmin_vmax[1]}
        background_alpha=0.4
        if levels is None:
            levels = [-4, -3.5, -3,  -2.5, -2, -1.5]

        cb = ax.contourf(fp_lons, fp_lats, np.log10(f), **plot_params, levels=levels, extend="both", alpha=background_alpha)
        f[f<background_threshold] = 0
        cb = ax.contourf(fp_lons, fp_lats, np.log10(f), **plot_params, levels=levels, extend="both")
        formatted_time = fp_to_plot.time.values.astype('datetime64[ms]').astype('O').strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]
        ax.set_title(formatted_time)

        if add_cbar:
            cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both").set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)

        if plot_wind:
            u_arrow = -self.met.x_wind.sel(levels=3, lat=self.size//2, lon=self.size//2).isel(time=idx).values
            v_arrow = -self.met.y_wind.sel(levels=3, lat=self.size//2, lon=self.size//2).isel(time=idx).values

            # Position arrow at centre of domain
            arrow_lat = self.fp_xr.lat_coords.sel(lat=self.size//2, time=timestamp).values
            arrow_lon = self.fp_xr.lon_coords.sel(lon=self.size//2, time=timestamp).values
            print(f"plotting wind arrow at lat {arrow_lat} and lon {arrow_lon} with u {u_arrow} and v {v_arrow}")
            ax.quiver(arrow_lon, arrow_lat, u_arrow, v_arrow,
                    transform=cartopy.crs.PlateCarree(),
                    scale=10, scale_units="inches",
                    color='black', width=0.005,
                    zorder=10)



        if return_fig:
            return fig, ax

    def plot_footprint_mean(self,levels = [-4, -3.5, -3,  -2.5, -2, -1.5], vmin_vmax=[-4,-2], add_cbar=False):
        """Plot the mean of the cropped and aligned footprints.

        Args:
            levels (list, optional): Contour levels (in log10 space) to plot.
                Defaults to [-4, -3.5, -3, -2.5, -2, -1.5].
            vmin_vmax (list, optional): ``[vmin, vmax]`` color scale limits.
                Defaults to [-4, -2].
            add_cbar (bool, optional): If True, adds a colorbar to the plot.
                Defaults to False.
        """
        import matplotlib.pyplot as plt

        f = self.fp_xr.fp.mean(dim="time").values
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


def _get_release_idxs(fp, domain_lats=None, domain_lons=None):
    """Find the nearest grid index to each footprint's release location.

    Vectorized replacement for a per-timestamp Python loop: uses a broadcast argmin
    instead, so it is O(n_times + n_grid_cells) rather than O(n_times * n_grid_cells).
    Still loads release_lat/lon into memory (they are small 1-D arrays).

    Args:
        fp (xr.Dataset): Footprint dataset with ``release_lat`` and ``release_lon``.
        domain_lats (array-like, optional): Grid latitudes to match against. If None,
            uses ``fp.lat.values``. Defaults to None.
        domain_lons (array-like, optional): Grid longitudes to match against. If None,
            uses ``fp.lon.values``. Defaults to None.

    Returns:
        np.ndarray: Array of shape (n_times, 2) with (lat_idx, lon_idx) of the
        nearest grid point to each footprint's release location.
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



def load_flux_data(domain, year=2016, species="ch4", flux_path=None, cfg=None, search_others=False):
    """Load emissions for the given domain and year.

    Loads the file in ``flux_path`` if provided, otherwise looks up the file based on
    the config, domain and species.

    Args:
        domain (str): Domain to load emissions for. Should match the ``domain_name``
            in the config file. Case-insensitive. If not found, will try to match
            against the region name in the config file. If still not found, raises
            an error with the available domain names in the config file.
        year (int or str, optional): Year to load emissions for. If str, should be in
            the format "2016" or "201[4-5]" to match multiple years. Defaults to 2016.
        species (str, optional): Species to load emissions for. Should match the
            species used in the ``flux_suffix`` in the config file if
            ``flux_suffix`` is a dict. Defaults to "ch4".
        flux_path (str, optional): Path to the flux file to load. If provided, this
            is used instead of looking up the file based on the config, domain and
            species. Defaults to None.
        cfg (gates.config.Config, optional): Config instance. If None, the config
            will be loaded from the config file. Used to look up the flux file if
            ``flux_path`` is not provided. Defaults to None.
        search_others (bool, optional): If the expected file is not found, search the
            same directory for a uniquely-matching file with a different suffix
            instead of raising immediately. Defaults to False.

    Returns:
        xr.DataArray: Lazy DataArray (time, lat, lon) with all months in the file.

    Raises:
        ValueError: If ``flux_path`` is given but does not exist, if ``cfg`` is
            neither None nor a ``gates.config.Config`` instance, if ``domain`` is not
            recognized, or if no matching flux file is found.
    """
    if flux_path is not None:
        # assert that there is a file at flux path
        if not os.path.isfile(flux_path):
            raise ValueError(f"flux_path {flux_path} does not exist or is not a file")
        return xr.open_dataset(flux_path).flux
    
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
            other_files = sorted(str(f) for f in path.parent.glob(f"{species}_{resolved_domain_name}_{year}_*.nc"))
            if len(other_files) ==1:
                print(f"Using another file for this domain and species: \n{other_files[0]}")
                path = other_files[0]
            else:
                raise ValueError(f"Flux file not found for domain '{domain}', species '{species}' and year '{year}' \nat {path}. \nThe existing files are: {sorted(str(f) for f in path.parent.glob(f'{species}_{resolved_domain_name}_{year}_*.nc'))}")
        else:
        
            raise ValueError(f"Flux file not found for domain '{domain}', species '{species}' and year '{year}' \nat {path}. \nThe existing files are: {sorted(str(f) for f in path.parent.glob(f"{species}_{resolved_domain_name}_{year}_*.nc"))}")
    else:
        print(f"Loading flux data from {path}")
    return xr.open_dataset(path).flux


def load_default_brazil_emissions(year=2016):
    """Obsolete. Use ``load_flux_data(domain='brazil', year=2016)`` instead.

    Args:
        year (int, optional): Unused. Defaults to 2016.
    """
    print("Obsolete: use load_emissions(domain='brazil', year=2016) instead")


def load_default_sahara_emissions(year=2016):
    """Obsolete. Use ``load_flux_data(domain='sahara', year=2016)`` instead.

    Args:
        year (int, optional): Unused. Defaults to 2016.
    """
    print("Obsolete: use load_emissions(domain='sahara', year=2016) instead")


def cut_flux_data(flux, fp, size, tolerance="32D", verbose=True):
    """Crop flux data to a size x size square centred on each footprint's release point.

    Each footprint is first matched to the nearest monthly flux snapshot.

    Args:
        flux (xr.DataArray): Monthly flux snapshots, dims (time, lat, lon) (e.g. from
            ``load_flux_data``).
        fp (xr.Dataset): Footprint dataset with ``release_lat``, ``release_lon``, and
            ``time`` coordinates.
        size (int): Side length of crop square. Must be even.
        tolerance (str, optional): Maximum time distance for matching a footprint to
            a flux snapshot. Defaults to "32D" (32 days), which ensures each
            footprint matches at most one month.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        tuple:
            - xr.Dataset: Dataset with variables ``flux`` (cropped flux,
              (time, lat, lon)), ``lat_coords`` (actual latitudes, (time, lat)), and
              ``lon_coords`` (actual longitudes, (time, lon)). ``lat``/``lon`` are
              artificial 0..size coordinates; the release point is at size//2.
            - pd.DatetimeIndex: fp timestamps that had no flux match within ``tolerance``.

    Raises:
        ValueError: If ``size`` is not even.
    """
    if size % 2 != 0:
        raise ValueError(f"size must be even, got {size}")
    half = size // 2

    # --- 1. Time matching ---
    tol = pd.Timedelta(tolerance)
    nearest = flux.indexes["time"].get_indexer(
        pd.DatetimeIndex(fp.time.values), method="nearest", tolerance=tol)
    nan_idxs = pd.DatetimeIndex(fp.time.values)[nearest == -1]
    if verbose and len(nan_idxs):
        print(f"cut_flux_data: {len(nan_idxs)} footprint timestamps had no "
              f"flux snapshot within {tolerance}. These will be NaN in the output.")
    nearest_safe = np.where(nearest != -1, nearest, 0)
    flux_matched = flux.isel(time=xr.DataArray(nearest_safe, dims="time"))
    flux_matched["time"] = fp.time.values
    if len(nan_idxs):
        flux_matched = flux_matched.where(
            xr.DataArray(nearest != -1, dims="time"), np.nan)

    # --- 2. Resolution check ---
    domain_lats = flux_matched.lat.values
    domain_lons = flux_matched.lon.values
    fp_lats = fp.lat.values
    fp_lons = fp.lon.values
    em_dlat = domain_lats[1] - domain_lats[0]
    em_dlon = domain_lons[1] - domain_lons[0]
    fp_dlat = fp_lats[1] - fp_lats[0]
    fp_dlon = fp_lons[1] - fp_lons[0]
    if verbose and (not np.isclose(em_dlat, fp_dlat) or not np.isclose(em_dlon, fp_dlon)):
        print(f"cut_flux_data: WARNING — flux resolution "
              f"({em_dlat:.4f}, {em_dlon:.4f}) differs from fp resolution "
              f"({fp_dlat:.4f}, {fp_dlon:.4f}). Spatial alignment may be off.")

    if hasattr(fp, "release_lat") and hasattr(fp, "release_lon"):
        # --- 3. Release indices + padding ---
        release_idxs = _get_release_idxs(fp, domain_lats, domain_lons)
        flux_matched, domain_lats, domain_lons, release_idxs = _pad_domain(
            flux_matched, fp, release_idxs, half, pad_mode="nans", verbose=verbose)

        # --- 4. Vectorised isel ---
        lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
        lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]
        lat_da = xr.DataArray(lat_indices, dims=["time", "lat"], coords={"time": fp.time})
        lon_da = xr.DataArray(lon_indices, dims=["time", "lon"], coords={"time": fp.time})
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            cropped = flux_matched.isel(lat=lat_da, lon=lon_da)
        
        lat_coords = xr.DataArray(
                domain_lats[lat_indices], dims=["time", "lat"],
                coords={"time": fp.time})
        lon_coords=  xr.DataArray(
                domain_lons[lon_indices], dims=["time", "lon"],
                coords={"time": fp.time})
    
    elif hasattr(fp, "lat_coords") or hasattr(fp, "lon_coords"):
        print("Using lat_coords and lon_coords to cut flux data, assuming they are aligned and have the same resolution as the flux data.")
        cropped = flux_matched.sel(lat=fp.lat_coords, lon=fp.lon_coords, method="nearest") 
        lat_coords = fp.lat_coords
        lon_coords = fp.lon_coords 

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


def cut_satellite_data(fp_full, size, fill_bads_with="nans", delete_outofdomain=False,
                           load=False, verbose=True):
    """Cut footprints to a size x size square centred on each release point.

    Returns an xarray Dataset with artificial lat/lon coordinates 0..size and the
    actual coordinates stored as ``lat_coords`` (time, lat) and ``lon_coords``
    (time, lon) variables.

    Args:
        fp_full (xr.Dataset): Full footprint dataset with variables ``fp``,
            ``release_lat``, ``release_lon`` and dimensions (time, lat, lon).
        size (int): Side length of the square crop. Must be even.
        fill_bads_with (str, optional): "nans" or "zeros" — fill value for
            out-of-domain padding. Defaults to "nans".
        delete_outofdomain (bool, optional): If True, drop footprints whose crop
            square escapes the domain rather than padding. Defaults to False.
        load (bool, optional): If True, load the result into memory immediately.
            Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        tuple:
            - cropped_fp (xr.Dataset): Cropped footprint dataset with variables
              ``fp``, ``lat_coords``, ``lon_coords``, ``release_lat``,
              ``release_lon``, dims (time, lat, lon), lat/lon being the artificial
              0..size coordinates.
            - fp_full (xr.Dataset): The (possibly domain-trimmed) full-resolution
              footprint dataset the crop was taken from.
            - release_idxs (np.ndarray): (lat_idx, lon_idx) of the release point for
              each timestamp, on the (possibly padded) domain grid.
            - padded_domain_coords (tuple): ``(domain_lats, domain_lons)`` of the
              (possibly padded) domain grid used for cropping.

    Raises:
        ValueError: If ``size`` is not even.
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
            fp_full = fp_full.drop_sel(time=fp_full.time.values[oob])
            release_idxs = _get_release_idxs(fp_full)

    before_padding_coords = (fp_full.lat.values.copy(), fp_full.lon.values.copy())

    fp_full, domain_lats, domain_lons, release_idxs = _pad_domain(
        fp_full, fp_full, release_idxs, half, pad_mode=fill_bads_with)

    padded_domain_coords = (domain_lats, domain_lons)
    # integer index arrays: shape (n_times, size)
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]

    lat_da = xr.DataArray(lat_indices, dims=["time", "lat"], coords={"time": fp_full.time})
    lon_da = xr.DataArray(lon_indices, dims=["time", "lon"], coords={"time": fp_full.time})

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        cropped_fp = fp_full.isel(lat=lat_da, lon=lon_da)

    cropped_fp = (
        cropped_fp
        .assign_coords(lat=np.arange(size), lon=np.arange(size))
        .assign({
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["time", "lat"],
                coords={"time": cropped_fp.time}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["time", "lon"],
                coords={"time": cropped_fp.time}),
        })
    )

    cropped_fp = cropped_fp[["fp", "lat_coords", "lon_coords", "release_lat", "release_lon"]]
    cropped_fp = cropped_fp.transpose("time", "lat", "lon")
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


def _interp_met_to_fp_times(met, fp, time_delta, interp_method, closest_tolerance="4h"):
    """Reindex/interpolate met to footprint times (or time-shifted versions).

    Adds an ``fp_time`` variable (original footprint observation times) and, when
    ``interp_method="closest"``, a ``met_timestamps`` variable (the actual met
    timestamp used for each footprint, NaT where no match was found within
    ``closest_tolerance``).

    Args:
        met (xr.Dataset): Meteorology dataset with a ``time`` dimension.
        fp (xr.Dataset): Footprint dataset with a ``time`` dimension.
        time_delta (int): Hours to shift the target times backwards (0 = no shift).
        interp_method (str): "closest" to snap to the nearest met timestamp, or
            use any method supported by ``xarray.interp`` (e.g. "linear", "nearest").
        closest_tolerance (str, optional): Max allowed distance for nearest-timestamp
            lookup when ``interp_method="closest"``. Defaults to "4h".

    Returns:
        tuple:
            - met_interpolated (xr.Dataset): Met reindexed/interpolated to the target times.
            - nan_idxs (pd.DatetimeIndex): Target times that could not be matched/interpolated.
    """
    fp_original_times = fp.time.values
    nan_idxs = []
    tol = pd.Timedelta(closest_tolerance)

    target_times = (fp_original_times if time_delta == 0
                    else pd.DatetimeIndex(fp_original_times) - pd.Timedelta(f"{time_delta}h"))

    if interp_method == "closest":
        nearest = met.indexes["time"].get_indexer(
            pd.DatetimeIndex(target_times), method="nearest", tolerance=tol)
        nan_idxs = pd.DatetimeIndex(target_times)[nearest == -1]
        nearest_safe = np.where(nearest != -1, nearest, 0)
        nearest_timestamps = met.indexes["time"].values[nearest_safe]
        unique_met_times, inverse_idx = np.unique(
            nearest_timestamps, return_inverse=True
        )
        t0 = time.perf_counter()
        met_unique = met.sel(time=unique_met_times)
        print(f"Graph build: {time.perf_counter()-t0:.2f}s")
        # print weight and chunks of unique
        print(f"met_unique has chunks {met_unique.chunks} and size {met_unique.nbytes / 1e6:.2f} MB")
        
        print("Tasks in graph:", len(met_unique.__dask_graph__()))
        #met_unique = met_unique.chunk({"time": -1})  # merge into one chunk before compute
        met_unique = met_unique.chunk({"time": -1, "lat": -1, "lon": -1, "levels": -1})
        print(f"Tasks after rechunk: {len(met_unique.__dask_graph__())}")
        print("computing met unique!")
        #met_unique = met_unique.compute()
        met = met_unique.isel(time=inverse_idx)
        met = met.assign_coords(time=target_times)
        nearest_timestamps_full = pd.DatetimeIndex(
            np.where(nearest != -1, nearest_timestamps, pd.NaT)
        )
        print(met)
        met["met_timestamps"] = ("time", nearest_timestamps_full)
        #v1 reindex
        #nearest_timestamps = pd.DatetimeIndex(
        #    np.where(nearest != -1, met.indexes["time"].values[nearest_safe], pd.NaT))
        #met = met.reindex(time=target_times, method="nearest", tolerance=tol, fill_value=np.nan)
        #met["met_timestamps"] = ("time", nearest_timestamps)
    else:
        met = met.interp(time=target_times, method=interp_method)

    met = met.assign({"fp_time": (("time",), fp_original_times)})
    # convert nan_idxs to the original fp time values for clarity by adding the time_delta back, and then to numpy array for easier use later
    nan_idxs = (pd.to_datetime(nan_idxs) + pd.Timedelta(f"{time_delta}h")).to_numpy()

    return met, nan_idxs


def _pad_domain(data, fp, release_idxs, half, pad_mode, verbose=True):
    """Extend the spatial domain of an xarray Dataset/DataArray so a crop is possible for every release point.

    Extends the domain so that a (2*half) x (2*half) crop is possible for every
    footprint release point. Works for any xarray object with lat/lon dimensions
    (met, fp, topog, etc.).

    For ``pad_mode="nans"``: uses ``xr.reindex`` with ``fill_value=nan`` — lazy.
    For ``pad_mode="zeros"``: uses ``xr.reindex`` with ``fill_value=0`` — lazy, but be
    careful if your data has valid zeros!
    For ``pad_mode="edge"``: uses ``xr.pad(mode="edge")`` then assigns the correct
    extended coordinate values.

    Args:
        data (xr.Dataset or xr.DataArray): Data to pad; must have lat/lon dimensions.
        fp (xr.Dataset): Footprint dataset used to recompute release indices after padding.
        release_idxs (np.ndarray): (lat_idx, lon_idx) release indices on the current grid.
        half (int): Half of the crop size (crop is ``2*half`` per side).
        pad_mode (str): One of "nans", "zeros", or "edge".
        verbose (bool, optional): Whether to print padding amounts. Defaults to True.

    Returns:
        tuple:
            - data (xr.Dataset or xr.DataArray): Padded data.
            - domain_lats (np.ndarray): Latitude values of the (possibly padded) domain.
            - domain_lons (np.ndarray): Longitude values of the (possibly padded) domain.
            - release_idxs (np.ndarray): Release indices recomputed on the padded domain.

    Raises:
        ValueError: If ``pad_mode`` is not one of "nans", "zeros", or "edge".
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
    """Cut meteorology to a metsize x metsize square around each footprint release point.

    Meteorology is interpolated to footprint times (or t-time_delta). Uses xarray and
    dask to produce one coherent lazy dask graph. ``lat_coords`` and ``lon_coords``
    are stored as (time, lat) / (time, lon) variables.

    Args:
        met (xr.Dataset): Meteorological data, with dimensions including "time",
            "lat", "lon", and possibly "levels". Should have variables for the
            relevant meteorological fields.
        fp (xr.Dataset): Footprint data, with dimensions including "time", and
            variables "release_lat" and "release_lon" for the release locations of
            each footprint.
        metsize (int): Size of the square to cut around each release point. Must be
            even to ensure the release point is centered.
        time_delta (int, optional): Hours to shift the footprint times backwards for
            interpolation. Defaults to 0 (no shift).
        relevant_levels (list, optional): Levels of atmospheric variables to extract.
            If None, uses all levels in ``met``. Defaults to None.
        relevant_variables (list, optional): Meteorological variables to extract. If
            None, uses all variables in ``met``. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        pad_mode (str, optional): "nans" or "edge". If "nans", pads with NaNs when
            the cut square extends beyond the met domain. If "edge", pads by
            extending the edge values of the met domain. Defaults to "nans".
        load (bool, optional): Whether to load the resulting cropped met into memory
            at the end (keep as lazy dask array if False). Defaults to True.
        add_wind_direction (bool, optional): Whether to calculate and add wind
            direction and speed from ``x_wind`` and ``y_wind``. Defaults to True.
        save (bool, optional): Whether to save the resulting cropped met to a NetCDF
            file. Defaults to False.
        savepath (str, optional): Path to save the NetCDF file if ``save`` is True.
            Must end with .nc. Defaults to None.
        attrs_dict (dict, optional): Additional attributes to add to the resulting
            cropped met dataset. The original met attributes are stored under
            "original_met_attrs". Defaults to None.
        interp_method (str, optional): Method to use for time interpolation. Options
            are "closest" (nearest met timestamp within ``closest_tolerance``) or any
            method supported by xarray's ``.interp`` (e.g. "linear", "nearest",
            "zero", "slinear", "quadratic", "cubic"). Default and most efficient is
            "closest".
        closest_tolerance (str, optional): Maximum allowed distance for the "closest"
            interpolation method, as a string or pandas Timedelta. Ignored if
            ``interp_method`` is not "closest". Defaults to "4h".
        return_nan_idxs (bool, optional): Whether to return the indices of footprints
            for which no met timestamp was found within ``closest_tolerance`` when
            using ``interp_method="closest"``. If True, the function returns a tuple
            ``(cropped_met, nan_idxs)``. Defaults to False.

    Returns:
        xr.Dataset or tuple: Cropped meteorology dataset, or ``(cropped_met, nan_idxs)``
        if ``return_nan_idxs`` is True.

    Raises:
        ValueError: If ``metsize`` is not even.
        AssertionError: If ``time_delta`` is negative.
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
    met, nan_idxs = _interp_met_to_fp_times(
        met, fp, time_delta, interp_method, closest_tolerance)
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

    # build integer index arrays: shape (n_times, metsize)
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(metsize)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(metsize)[None, :]

    lat_da = xr.DataArray(lat_indices, dims=["time", "lat"], coords={"time": met.time})
    lon_da = xr.DataArray(lon_indices, dims=["time", "lon"], coords={"time": met.time})

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        cropped_met = met.isel(lat=lat_da, lon=lon_da)

    cropped_met = (
        cropped_met
        .assign_coords(lat=np.arange(metsize), lon=np.arange(metsize))
        .assign({
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["time", "lat"],
                coords={"time": cropped_met.time}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["time", "lon"],
                coords={"time": cropped_met.time}),
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
    """\Potentially deprecated. Extract the last underscore-separated, extension-stripped number from a filename.

    Args:
        name (str): <FILL IN> — a filename/string ending in ``..._<number>.<ext>``.

    Returns:
        int: The last underscore-separated, extension-stripped number in ``name``.
    """
    num = name.split('_')[-1]
    num = num.split('.')[0]
    return int(num)


def cut_topog_data(topog_file, landcover_file, fp, size, pad_mode="zeros"):
    """Crop topography and landcover to a size x size square centred on each footprint's release point.

    ``lat`` and ``lon`` are artificial coordinates 0..size; actual geographic
    coordinates are stored in ``lat_coords`` (time, lat) and ``lon_coords``
    (time, lon) variables.

    Args:
        topog_file (xr.Dataset): Topography dataset with variable
            ``surface_altitude`` and dimensions (lat, lon).
        landcover_file (xr.Dataset or None): Landcover dataset with variables
            ``landcover_type``, ``land_binary_mask``, and ``landcover_fraction``
            (lat, lon, pseudo_level), with coordinates lat, lon, pseudo_level. If
            None, only the cropped topography is included in the output.
        fp (xr.Dataset): Footprint dataset providing ``release_lat``, ``release_lon``,
            and ``time`` coordinates.
        size (int): Side length of the square crop. Must be even.
        pad_mode (str, optional): How to pad if a crop escapes the topog domain:
            "zeros" or "edge". Defaults to "zeros".

    Returns:
        xr.Dataset: Dataset with dimensions (time, lat, lon) and variables:

            - ``topog``: surface altitude, shape (time, lat, lon).
            - ``landcover``: integer landcover type, shape (time, lat, lon).
            - ``disaggregated_landcover``: (time, lat, lon, landcover_level) with
              ``land_binary_mask`` inverted (sea=1, land=0) in level 0 and 9
              fractional landcover types in levels 1-9.

            (``landcover``/``disaggregated_landcover`` are omitted if
            ``landcover_file`` is None.)

    Raises:
        ValueError: If ``size`` is not even.
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

    lat_da = xr.DataArray(lat_indices, dims=["time", "lat"], coords={"time": fp.time})
    lon_da = xr.DataArray(lon_indices, dims=["time", "lon"], coords={"time": fp.time})

    # isel on static (lat, lon) arrays — DataArray indexers introduce the time dim
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
            domain_lats[lat_indices], dims=["time", "lat"], coords={"time": fp.time}),
        "lon_coords": xr.DataArray(
            domain_lons[lon_indices], dims=["time", "lon"], coords={"time": fp.time}),
    })
    else:
        result = xr.Dataset({
            "topog":                    topog_crop,
            "landcover":                landcover_crop,
            "disaggregated_landcover":  disagg,
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["time", "lat"], coords={"time": fp.time}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["time", "lon"], coords={"time": fp.time}),
        })

    result = result.assign_coords(lat=np.arange(size), lon=np.arange(size))

    return result


def get_grid(fp_xr, reference_fp=0):
    """Produce a reference grid and node indices.

    The grid is made from the lat/lon coordinates of the reference footprint, and the
    indices are centred around the reference footprint (i.e. the release point is at
    index (size//2, size//2)).

    Args:
        fp_xr (xr.Dataset): Footprint data with variables ``lat_coords``/``lon_coords``,
            and coordinates time, lat, lon.
        reference_fp (int, optional): Index of the reference footprint to use for
            grid generation. Defaults to 0 (the first footprint).

    Returns:
        tuple:
            - latlons (list[tuple]): (lat, lon) pairs for every grid node.
            - idx_latlons (list[tuple]): (lat_idx, lon_idx) pairs for every grid node.
    """
    if reference_fp is None:
        reference_fp = 0
        
    single_meshgrid = np.meshgrid(fp_xr.lat_coords.isel(time=reference_fp), fp_xr.lon_coords.isel(time=reference_fp))  

    latlons = [(single_meshgrid[0][i,j], single_meshgrid[1][i,j]) for i in range(fp_xr.lon.size) for j in range(fp_xr.lat.size)]

    idx_meshgrid = np.meshgrid(fp_xr.lat.values, fp_xr.lon.values)
    idx_latlons = [(idx_meshgrid[0][i,j], idx_meshgrid[1][i,j]) for i in range(fp_xr.lon.size) for j in range(fp_xr.lat.size)]

    return latlons, idx_latlons





