"""
Data loading functions, loading footprints, meteorology and topography data.
Can be fed into the GATES model as a dataset, or used for other models

author: Elena Fillola @elenafillo
"""

from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os
import copy
import warnings

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

from .load_data_helper_funs import *


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


def load_fps(fp_datadir, verbose=False, chunk=False):
    """
    Load footprints from datadir, using workaround if problematic files are encountered. Will throw an error if ANY of the specified files is problematic and NOT on the bad_files list
    note that the list of problematic files is currently updated manually!

    Args:
        - fp_datadir (str): string pointing to the directory with footprints to load, 
        including special characters (eg "/path/to/footprints/*.nc", or "/path/to/footprints/*2020*.nc")
        Note that fp_datadir is passed directly to glob, so it needs to specify filetype (i.e. finish with .nc)
    
    Returns:
        - fp_data_full: xarray dataset with the footprints specified in the fp_datadir, opened correctly
    
    Potential Improvements:
        - Add capability to ignore any files that couldn't be opened, and return only the successful files 
    """
    try:           
        if chunk:
            time_chunk = 25
            chunk_args = {"chunks" : {"time": time_chunk}, "parallel": True}
            chunk_args = {"chunks":"auto", "parallel": True}
        else:
            chunk_args = {}
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # check that there are any files to open
            if len(glob.glob(fp_datadir))==0:
                raise ValueError(f"No files found in the specified directory:\n {fp_datadir} \nCheck that the path is correct and that there are files matching the pattern.")
            # attempt to load dataset of multiple files thfe standard way
            fp_data_full = xr.open_mfdataset(sorted(glob.glob(fp_datadir)), combine='by_coords', **chunk_args)

    except Exception as e:
        # some files have small errors in format that prevent xr from concatenating and opening together. This is a workaround to open those separately. This list only contains known files and could be more! can add manually whenever you encounter one 
        # the bad_files contains full paths, first the full path is checked 
        if verbose: print("there was an error opening the dataset. checking if any of the files are in the bad files list")
        fp_files = sorted(glob.glob(fp_datadir))
        path = os.path.split(fp_datadir)[0] + "/"
        filenames = [os.path.split(x)[1] for x in fp_files]

        bad_files = ["GOSAT-BRAZIL-column_SOUTHAMERICA_201511.nc", 
            "GOSAT-SAHARA-column_NORTHAFRICA_201409.nc", 
            'GOSAT-SAHARA-column_NORTHAFRICA_201501.nc',
            'GOSAT-SAHARA-column_NORTHAFRICA_201502.nc',
            'GOSAT-SAHARA-column_NORTHAFRICA_201503.nc',
            'GOSAT-SAHARA-column_NORTHAFRICA_201504.nc',
            'GOSAT-SAHARA-column_NORTHAFRICA_201609.nc',
            'GOSAT-SAHARA-column_NORTHAFRICA_201610.nc',
            'GOSAT-SAHARA-column_NORTHAFRICA_201612.nc']
        # remove any files from the list that were in the bad files list

        
        without_bad_files = list(set(filenames) - set(bad_files))
        without_bad_files = [path+f for f in without_bad_files]
        bad_files = [path+f for f in bad_files]

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
                most = xr.open_mfdataset(sorted(without_bad_files))
                bad_arrays = []
                for badfile in bad_files:
                    if badfile in fp_files:
                        # load each bad file separately
                        f_bad = xr.open_mfdataset(badfile)
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

    other inputs
        - freq: int, frequency of the data to load. 
        freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset for testing
        - sampling_mode: str, default "regular".
        if "regular", subsamples footprints regularly (e.g. one in every two, sequentially with freq=2). if random, subsamples N/freq footprints randomly (where N is the total number of footprints)
        - freq_offset: int
        if using sampling_mode="regular", offsets the start of the regular sampling, e.g. freq=2 and freq_offset=0 will sample even footprints, and freq_offset=1 will sample uneven footprints
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. 
        Applied after sampling with freq (or pass freq=1 to load all footprints)
        - fp_datadir: str, directory for footprints. default directs to ACRG folder. 
        If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass fp_datadir="/path/name_of_your_choice_"
        - verbose: if True, prints out the steps throughout the data loading process
        - lazy_load: if True, does not load the met data into memory (lazy array), False, loads the met data into memory.
    
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
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, freq=1, freq_offset=0, verbose = False, sampling_mode="regular", fp_datadir = None, load_everything=False, met_args={}, topog_args={}):
        
        self.dataset_format = "base" 
        self.data_type="satellite"

        #### check domains
        self.region = region
        if domain is None:
            self.domain = self._get_domain(region)
        else:
            self.domain=domain

        self.year = year
        self.date = self.year
        self.verbose=verbose


        if month != None:
            self.month = month
            self.date = str(self.year)+month

        self.subsample_parameters = {"freq":freq, "sampling_mode":sampling_mode, "freq_offset":freq_offset}
        
        #### load footprint (fp) data     
        if verbose: print("---- LOADING FOOTPRINTS")  
        self._load_footprints(fp_datadir)

        self.met_processed = False
        
        self.padding=None

        if load_everything:
            self.met_file = self.load_meteorology(**met_args)
            self.topog_file, self.landcover_file = self.load_topog(**topog_args)

        
        if verbose: print("---- All done!")



    def load_meteorology(self, met_datadir=None, met_levels = [], met_variables= [], lazy_load=True, met_time_chunk=24, parallel=False):
        """
        loads meteorology and selects the met levels and variables if required

        Inputs
            - met_datadir (str): directory for meteorology. default directs to ACRG meteorology folder. If passing as arg, the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/name_of_your_choice_"
            - met_levels (list): met levels to select
            - met_variables (list) : met variables to select
            - lazy_load (bool): if True, does not load the met data into memory (lazy array),  False, loads the met data into memory. 
        """
        if self.verbose: print("\n ---- LOADING MET")

        # 1) load from file
        self.met_file = self._get_meteorology_file(
            met_datadir,
            met_time_chunk=met_time_chunk,
            parallel=parallel,
        )

        # 2) check domain overlap
        self._check_domain_overlap(self.fp_data_full, self.met_file, "footprint", "meteorology")

        if len(met_levels)>0:
            try:
                self.met_file = self.met_file.sel(levels=met_levels)
            except KeyError:
                print(f"there was an error selecting the met levels you passed. Check! \n You passed  {met_levels} but met loaded has {self.met_file.levels}. \n Loading all levels")
        if len(met_variables)>0:
            try:
                self.met_file = self.met_file[met_variables]
            except KeyError:
                print(f"there was an error selecting the met variables you passed. Check! \n You passed  {met_variables} but met loaded has {list(self.met_file.keys())}. \n Loading all variables")

        if not lazy_load:
            print("Loading met data into memory. If you only want to lazy-load, pass load=False")
            self.met_file.load()

        return self.met_file

    def load_topog(self, topog_path="default", landcover_path="default"):
        """
        load the topgoraphy and landcover files, and interpolate to the same resolution and domain as the footprints in self.fp_data_full
        args:
         - topog_path and landcover_path: str paths to each file, or "default" for default file
        
        uses objet attributes: self.padding (contains if any amount of padding is needed to the footprint domain, and in which direction)
        """
        #### load topography
        if self.verbose: print("\n---- LOADING TOPOG")
        if topog_path=="default":
            topog_path="/group/chem/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc"
        if self.verbose: print(f"trying to load topography from {topog_path}")
        with xr.load_dataset(topog_path) as topog_dataset:
            topog_file = topog_dataset.copy()

        if landcover_path=="default":
            landcover_path = "/group/chem/acrg/LPDM/topog_NAME/land_cover.nc"
        with xr.load_dataset(landcover_path) as landcover_dataset:
            landcover_file = landcover_dataset.copy()

        if not hasattr(self, "padded_domain_coords"):
            self.padded_domain_coords = None

        topog_file = self._interp_topog(topog_file, padding=self.padded_domain_coords)

        landcover_file = self._interp_landcover(landcover_file, padding=self.padded_domain_coords)
        return topog_file, landcover_file

    def _get_meteorology_file(self, met_datadir, lazy_load=True, met_time_chunk=24, parallel=False):
        """
        Load the meteorology from the directory, concatenating files along the time dimension. If met_datadir is None, uses default directory and file format. If met_datadir is passed, the date will be automatically added, so the files should have format example_name_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/example_name_"
        """
        if met_datadir==None:
            met_datadir = "/group/chem/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if self.verbose: print("Loading meteorology from " + met_datadir)

        met_files = sorted(glob.glob(met_datadir))
        if len(met_files) == 0:
            raise ValueError(
                f"No meteorology files found in the specified directory:\n {met_datadir}"
            )

        chunk_args = {"chunks": {"time": met_time_chunk}} if met_time_chunk is not None else {}

        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            met_file = xr.open_mfdataset(
                met_files,
                concat_dim="time",
                combine="nested",
                data_vars="minimal",
                coords="minimal",
                parallel=parallel,
                join="inner",
                **chunk_args,
                drop_variables=["forecast_period", "forecast_reference_time", "level_height_0", "sigma_0"],
                compat="override",
                preprocess=preprocess_met_data,
            )

            if "model_level_number" in met_file.dims:
                met_file = met_file.rename({"model_level_number": "levels"})
            met_file = _rename_latlon(met_file)

            for dim in ("lat", "lon", "time"):
                if dim in met_file.dims and met_file.get_index(dim).has_duplicates:
                    met_file = met_file.drop_duplicates(dim=dim)

            first_var = list(met_file.data_vars)[0]
            if met_file[first_var].dtype != np.float32:
                met_file = met_file.astype(np.float32)

            self.met_file = met_file
        return self.met_file


    def _get_domain(self, region):
        #### check domains
        # TODO make domains dict importable
        domains = {"BRAZIL":"SOUTHAMERICA", "SOUTHAMERICA":"SOUTHAMERICA", "SAHARA":"NORTHAFRICA", "INDIA":"SOUTHASIA"} 
        try:
            domain = domains[region]   
        except: 
            raise ValueError("No domain was passed, and the region you passed is not associated to any domain!")   
        
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

    def _load_footprints(self, fp_datadir):
        """
        Load footprint from fp_datadir and applies subsampling according to the freq and sampling_mode parameters. 
        
        The footprints are stored in self.fp_data_full as an xarray dataset, with dimensions time, lat and lon. 
        """
        #### load footprint (fp) data from file
        if fp_datadir is None:
            fp_datadir = "/group/chem/acrg/LPDM/fp_NAME_pre20210701/"+self.domain+"/*"+self.region+"*"+self.domain+"_"+str(self.date)+"*.nc"
        else:
            fp_datadir=fp_datadir+str(self.date)+"*.nc"
            #fp_datadir = f"{fp_datadir}{self.domain}/*{self.region}*{self.domain}_{str(self.date)}*.nc"
        if self.verbose: print("Loading footprint data from " + fp_datadir) 

        self.fp_data_full = load_fps(fp_datadir, verbose=self.verbose)  

        self.fp_data_full = self.fp_data_full.drop_duplicates(dim="time")

        ## reduce data frequency with regular sampling 9eg keep only 1 in every 3 timesteps
        # uses the sampling_mode and freq parameters
        self._subsample_frequency(**self.subsample_parameters)

        self.fp_data_full = self.fp_data_full.chunk({"lat": -1, "lon": -1, "time": 500})

        if self.verbose: print(f"Loading {len(self.fp_data_full.time.values)} footprints")

        #self.fp_data_full.load()
    
    def _subsample_frequency(self, freq=1, sampling_mode="regular",freq_offset=0):
        """
        Subsample the footprint data by selecting every freq-th timestamp from the original dataset, starting from the timestamp specified by freq_offset (if sampling_mode is "regular"), or by randomly selecting N/freq timestamps from the original dataset (if sampling_mode is "random"). If freq=1, no subsampling is done and all footprints are loaded. 
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
        
        # Check domain overlap after renaming
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
        ## get land-sea mass and country mask, can be used for filtering out footprints/data and during fplotting
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


    def plot_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, return_fig=False, plot_marker=False, dpi=100):
        """
        plot a footprint for a particular timestamp or index

        inputs:
            - idx: int index of the footprint to plot. If timestamp is also passed, timestamp will be used instead of idx
            - timestamp: timestamp of the footprint to plot, as a string in format "YYYY-MM-DDTHH:MM:SS" (eg "2016-01-01T12:00:00"). If idx is also passed, timestamp will be used instead of idx
            - return_fig: if True, returns the fig and ax objects instead of showing the plot. 
        """

        if timestamp is not None:
            fp_to_plot = self.fp_data_full.sel(time=np.datetime64(timestamp)).copy()
            if len(fp_to_plot.time.values)>1:
                print("there are multiple footprints for the timestamp you passed, check the timestamp and try again! plotting the first one")
                fp_to_plot = fp_to_plot.isel(time=0)
        
        else:
            fp_to_plot = self.fp_data_full.isel(time=idx).copy()

        f = np.copy(fp_to_plot.fp.values)

        extent = (fp_to_plot.lon.values[0], fp_to_plot.lon.values[-1], fp_to_plot.lat.values[0], fp_to_plot.lat.values[-1])

        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(8,6), dpi=dpi)
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
            cbar = fig.colorbar(cb, ax=ax, location='bottom', extend="both").set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)

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
        - region: region identifyer, as a string. Default is Brazil. Current set-up has regions "BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA"
        (note - Brazil is a subset of South America!)
        - domain: Domain related to the region, used for file search (due to existing filenaming conventions). Set-up regions ("BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA") have a default domain, all others need domain passed
        NOTE: Update dict of region-domain using config rather than being hard-coded
        - size: size for footprint to be cut to, as an int. Resolution of the footprint is maintained, cut to a sizexsize square around the release point. 
    

    other inputs
        - freq: int, frequency of the data to load. 
        freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset for testing
        - sampling_mode: str, default "regular".
        if "regular", subsamples footprints regularly (e.g. one in every two, sequentially with freq=2). if random, subsamples N/freq footprints randomly (where N is the total number of footprints)
        - freq_offset: int
        if using sampling_mode="regular", offsets the start of the regular sampling, e.g. freq=2 and freq_offset=0 will sample even footprints, and freq_offset=1 will sample uneven footprints
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. 
        Applied after sampling with freq (or pass freq=1 to load all footprints)
        - fp_datadir: str, directory for footprints. default directs to ACRG folder. 
        If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass fp_datadir="/path/name_of_your_choice_"
        - verbose: if True, prints out the steps throughout the data loading process
        - lazy_load: if True, does not load the met data into memory (lazy array), False, loads the met data into memory.
        - fill_outofdomain_with: str, out of "nans" and "zeros". Determines what to do if any part of the square cut around the footprint is outside of the domain. "nans" and "zeros" fill only the out of domain areas with nans and zeros respectively. 
        - delete_outofdomain: bool, if True delete all footprints (and associated datapoints) where the extracted area size x size escapes the domain. Default is False, which means that the cut footprints will be kept and the out of domain areas will be filled according to fill_outofdomain_with.
        - verbose: if True, prints out the steps throughout the data loading process
        - load_everything: bool, if True, loads all data (footprints, meteorology and topography) at once when initializing the class. If False, only loads footprints, and meteorology and topography can be loaded later with the load_meteorology() and load_topog() functions. Default is False
    
    met_args:
        see load_meteorology()
    topog_args:
        see load_topog()
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, verbose = False, fill_outofdomain_with="nans", delete_outofdomain=False, check_for_nans=False, sampling_mode="regular", fp_datadir = None, load_everything=True, lazy_load=True, met_args={}, topog_args={}):

        self.dataset_format = "square" 
        #### check domains
        self.region = region
        if domain is None:
            self.domain = self._get_domain(region)
        else:
            self.domain=domain

        self.size = size

        self.fill_outofdomain_with = fill_outofdomain_with
        self.delete_outofdomain = delete_outofdomain

        self.year = year
        self.date = self.year
        self.verbose=verbose


        if month != None:
            self.month = month
            self.date = str(self.year)+month

        self.subsample_parameters = {"freq":freq, "sampling_mode":sampling_mode, "freq_offset":freq_offset}
        
        #### load footprint (fp) data, subsample, crop
        if verbose: print("---- LOADING FOOTPRINTS") 
        self._load_footprints(fp_datadir)
        self._process_footprints(lazy_load)

        self.met_args = met_args
        self.met_processed = False

        if load_everything:
            self.met_file = self.load_meteorology(**met_args,lazy_load=lazy_load)
            self.met = self._process_meteorology(lazy_load=True)
            self.topog_file, self.landcover_file = self.load_topog(**topog_args)
            self.topog = self._process_topog_and_landcover()

        if check_for_nans:
            print("\n Checking if there are any nans in the data")
            self._remove_fp_nans()
            self._remove_met_nans()
        
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

    def _process_meteorology(self,rechunk=0,lazy_load=True):
        """
        Call cut_satellite_met to cut the meteorology data around the release point, to the same size as the cut footprints.
        If the square to extract escapes the met domain, the padded space is filled with nans or zeros according to the fill_outofdomain_with parameter 
        """
        if self.verbose: print("----- Cutting met")
        self.metsize=self.size
        #if self.fill_outofdomain_with=="nans" or self.delete_outofdomain:
        #    pad_mode = "nans"
        #if self.fill_outofdomain_with=="zeros":
        pad_mode = "edge"
        self.met = cut_satellite_met(self.met_file, self.fp_data_full, metsize=self.size, time_delta=0, pad_mode=pad_mode, load=not lazy_load, add_wind_direction=True)

        if rechunk>0:
            self.met.chunk({"time":rechunk})
        
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

    def remove_indeces(self, nan_idxs):
        """
        removes any set of indeces passed as nan_idxs from all the objects in the dataset
        """
        self.fp_data_full = self.fp_data_full.drop_sel(time=nan_idxs)
        if hasattr(self, "fp_data"):
        # this should now be redundant and fail!
            print("this should be redundant")
            self.fp_lats = np.delete(self.fp_lats, nan_idxs, axis=0)
            self.fp_lons = np.delete(self.fp_lons, nan_idxs, axis=0)
            self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
            if self.verbose: print(f"current length: {len(self.release_idxs)}")
            self.release_idxs = np.delete(self.release_idxs, nan_idxs, axis=0)
        if hasattr(self, "fp_xr"):
            self.fp_xr = self.fp_xr.drop_sel(time=nan_idxs)
        if hasattr(self, "met"): 
            self.met = self.met.drop_sel(time=nan_idxs)
        if hasattr(self, "topog"):
            self.topog = self.topog.drop_sel(time=nan_idxs)

        if self.verbose: print(f"Length after removing indeces: {self.fp_data_full.time.size}")
        # if the len is zero, raise an error
        if self.fp_data_full.time.size == 0:
            raise ValueError("All data has been removed after removing nans, cannot continue!")

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

    
    def plot_cropped_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, plot_wind=False, return_fig=False):
        """
        plot a footprint for a particular timestamp or index, with the option to also plot the topography and landcover if they have been loaded. 

        inputs:
            - idx: int index of the footprint to plot. If timestamp is also passed, timestamp will be used instead of idx
            - timestamp: timestamp of the footprint to plot, as a string in format "YYYY-MM-DDTHH:MM:SS" (eg "2016-01-01T12:00:00"). If idx is also passed, timestamp will be used instead of idx
        """

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
        """
        Plot the mean of the cropped and aligned footprints. Levels and vmin_vmax adjust the colour scale. If add_cbar is True, adds a colorbar
        """
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



def load_flux_data(domain, year=2016):
    """
    Load emissions for the given domain and year. Returns a lazy xarray DataArray (time, lat, lon) with all months in the file. Hardcoded paths should be replaced by a config file .
    """
    if domain.upper() in ["BRAZIL", "SOUTHAMERICA"]:
        path = f"/group/chem/acrg/LPDM/emissions/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}_SWAMPS-v32-5_Saunois-Annual-Mean.nc"
    elif domain.upper() in ["SAHARA", "NORTHAFRICA"]:
        path = f"/group/chem/acrg/LPDM/emissions/NORTHAFRICA/ch4_NORTHAFRICA_{year}.nc"
    else:
        raise ValueError("Domain not recognized")

    return xr.open_dataset(path).flux


def load_default_brazil_emissions(year=2016):
    print("Obsolete: use load_emissions(domain='brazil', year=2016) instead")


def load_default_sahara_emissions(year=2016):
    print("Obsolete: use load_emissions(domain='sahara', year=2016) instead")


def cut_flux_data(flux, fp, size, tolerance="32D", verbose=True):
    """
    Crops flux data to a size x size square centred on each footprint's release
    point, after matching each footprint to the nearest monthly flux snapshot.

    Parameters
    ----------
    flux : xr.DataArray, dims (time, lat, lon)
        Monthly flux snapshots (e.g. from load_default_brazil_emissions).
    fp : xr.Dataset
        Footprint dataset with release_lat, release_lon, and time coordinates.
    size : int
        Side length of crop square. Must be even.
    tolerance : str
        Maximum time distance for matching a footprint to a flux snapshot.
        Default '32D' (32 days) ensures each footprint matches at most one month.
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple of (xr.Dataset, pd.DatetimeIndex)
        Dataset with variables:
            - flux       : cropped flux (time, lat, lon)
            - lat_coords : actual latitudes  (time, lat)
            - lon_coords : actual longitudes (time, lon)
        lat/lon are artificial 0..size coordinates; release point is at size//2.
        DatetimeIndex of fp timestamps that had no flux match within tolerance.
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

    # --- 3. Release indices + padding ---
    release_idxs = _get_release_idxs(fp, domain_lats, domain_lons)
    flux_matched, domain_lats, domain_lons, release_idxs = _pad_domain(
        flux_matched, fp, release_idxs, half, pad_mode="nans")

    # --- 4. Vectorised isel ---
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]
    lat_da = xr.DataArray(lat_indices, dims=["time", "lat"], coords={"time": fp.time})
    lon_da = xr.DataArray(lon_indices, dims=["time", "lon"], coords={"time": fp.time})
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        cropped = flux_matched.isel(lat=lat_da, lon=lon_da)

    # --- 5. Assign coordinates + return ---
    cropped = (
        cropped
        .assign_coords(lat=np.arange(size), lon=np.arange(size))
        .to_dataset(name="flux")
        .assign({
            "lat_coords": xr.DataArray(
                domain_lats[lat_indices], dims=["time", "lat"],
                coords={"time": fp.time}),
            "lon_coords": xr.DataArray(
                domain_lons[lon_indices], dims=["time", "lon"],
                coords={"time": fp.time}),
        })
    )
    return cropped, nan_idxs


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
    print("using new cut satellite data!!!")
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
    cropped_fp = cropped_fp.chunk({"time": 100, "lat": -1, "lon": -1})

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
    """
    Reindexes/interpolates met to footprint times (or time-shifted versions).
    Returns (met_interpolated, nan_idxs).

    Adds 'fp_time' variable (original footprint observation times) and, when interp_method='closest', 'met_timestamps' (the actual met timestamp used for each footprint, NaT where no match was found within closest_tolerance).
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
        nearest_timestamps = pd.DatetimeIndex(
            np.where(nearest != -1, met.indexes["time"].values[nearest_safe], pd.NaT))
        met = met.reindex(time=target_times, method="nearest", tolerance=tol, fill_value=np.nan)
        met["met_timestamps"] = ("time", nearest_timestamps)
    else:
        met = met.interp(time=target_times, method=interp_method)

    met = met.assign({"fp_time": (("time",), fp_original_times)})
    # convert nan_idxs to the original fp time values for clarity by adding the time_delta back, and then to numpy array for easier use later
    nan_idxs = (pd.to_datetime(nan_idxs) + pd.Timedelta(f"{time_delta}h")).to_numpy()

    return met, nan_idxs


def _pad_domain(data, fp, release_idxs, half, pad_mode):
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
        print(f"Padding lat by ({pad_S}, {pad_N}) cells (S, N) with mode='{pad_mode}'")
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
        print(f"Padding lon by ({pad_W}, {pad_E}) cells (W, E) with mode='{pad_mode}'")
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
                          load=False, add_wind_direction=True, save=False,
                          savepath=None, attrs_dict=None, interp_method="closest",
                          closest_tolerance="4h", return_nan_idxs=False):
    """
    Cuts meteorology to a metsize x metsize square around each footprint release
    point, interpolated to footprint times (or t-time_delta).

    Uses xarray and dask to produce one coherent lazy dask graph.

    lat_coords and lon_coords are stored as (time, lat) / (time, lon) variables

    Parameters:
    - met: xarray dataset with meteorological data, with dimensions including 'time', 'lat', 'lon', and possibly 'levels'. Should have variables for the relevant meteorological fields
    - fp: xarray dataset with footprint data, with dimensions including 'time', and variables 'release_lat' and 'release_lon' for the release locations of each footprint
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
        met, fp, release_idxs, half, pad_mode)

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

    if load:
        if verbose:
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

def get_grid(data, latlon_fp=0):
    """
    produce reference grid and node indeces
    """
    if latlon_fp is None:
        latlon_fp = 0
        
    print(f"getting grid for time {data.met.time.values[latlon_fp]}")
    #print(f"making grid for footprint at time {data}")

    if data.dataset_format == "square":
        single_meshgrid = np.meshgrid(data.fp_lats[latlon_fp,:], data.fp_lons[latlon_fp,:])
        latlons = [(single_meshgrid[0][i,j], single_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lons)[1]) for j in range(np.shape(data.fp_lats)[1])] 

        idx_meshgrid = np.meshgrid(list(range(len(data.fp_lats[latlon_fp,:]))), list(range(len(data.fp_lons[latlon_fp,:]))))
        idx_meshgrid = np.array(idx_meshgrid)-int(data.size/2)
        idx_latlons = [(idx_meshgrid[0][i,j], idx_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lons)[1]) for j in range(np.shape(data.fp_lats)[1])] 

    elif data.dataset_format == "domain":
        single_meshgrid = np.meshgrid(data.fp_lats[0], data.fp_lons[0])
        latlons = [(single_meshgrid[0][i,j], single_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lons)[1]) for j in range(np.shape(data.fp_lats)[1])] 

        # should I find a way of making sure this idx meshgrid is consistent across different domain sizes?
        idx_meshgrid = np.meshgrid(list(range(len(data.fp_lats[0]))), list(range(len(data.fp_lons[0]))))

        #idx_meshgrid = np.array(idx_meshgrid)-int(data.size/2)

        idx_latlons = [(idx_meshgrid[0][i,j], idx_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lons)[1]) for j in range(np.shape(data.fp_lats)[1])]    


    return latlons, idx_latlons

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
    (time, lat, lon) and variables:
        - topog: surface altitude, shape (time, lat, lon)
        - landcover: integer landcover type, shape (time, lat, lon)
        - disaggregated_landcover: (time, lat, lon, landcover_level) with
          land_binary_mask inverted (sea=1, land=0) in level 0 and 9 fractional
          landcover types in levels 1–9
    lat and lon are artificial coordinates 0..size; actual geographic coordinates
    are stored in lat_coords (time, lat) and lon_coords (time, lon) variables.

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
        topog_file, fp, release_idxs, half, pad_mode=pad_mode)
    landcover_file, _, _, _ = _pad_domain(
        landcover_file, fp,
        _get_release_idxs(fp, domain_lats=landcover_file.lat.values, domain_lons=landcover_file.lon.values),
        half, pad_mode=pad_mode)

    # integer index arrays: shape (n_times, size)
    lat_indices = (release_idxs[:, 0] - half)[:, None] + np.arange(size)[None, :]
    lon_indices = (release_idxs[:, 1] - half)[:, None] + np.arange(size)[None, :]

    lat_da = xr.DataArray(lat_indices, dims=["time", "lat"], coords={"time": fp.time})
    lon_da = xr.DataArray(lon_indices, dims=["time", "lon"], coords={"time": fp.time})

    # isel on static (lat, lon) arrays — DataArray indexers introduce the time dim
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        topog_crop      = topog_file.surface_altitude.isel(lat=lat_da, lon=lon_da)
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
    #disagg = disagg.assign_coords(landcover_level=np.arange(10))
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






