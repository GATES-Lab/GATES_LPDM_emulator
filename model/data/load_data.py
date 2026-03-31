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
#import warning

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

from .load_data_helper_funs import *

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
        else:
            chunk_args = {}
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # attempt to load dataset of multiple files the standard way
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

    fp_data_full= fp_data_full.sortby('time')

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
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = ds.drop_duplicates(dim)
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

    ds = ds.astype("float32")

    
    ds = remove_duplicates(ds, dim=duplicate_dim)

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



    def load_meteorology(self, met_datadir=None, met_levels = [], met_variables= [], lazy_load=True):
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
        self._get_meteorology_file(met_datadir)

        if len(met_levels)>0:
            try:
                self.met_file = self.met_file.sel(levels=met_levels)
            except KeyError:
                print(f"there was an error selecting the met levels you passed. Check! \n You passed  {met_levels} but met loaded has {self.met_file.levels}. \n Loading all levels")
        if len(met_variables)>0:
            try:
                self.met_file = self.met[met_variables]
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
            

        topog_file = self._interp_topog(topog_file, padding=self.padding)

        landcover_file = self._interp_landcover(landcover_file, padding=self.padding)

        return topog_file, landcover_file

    def _get_meteorology_file(self, met_datadir, lazy_load=True):
        """
        Load the meteorology from the directory, concatenating files along the time dimension. If met_datadir is None, uses default directory and file format. If met_datadir is passed, the date will be automatically added, so the files should have format example_name_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/example_name_"
        """
        if met_datadir==None:
            met_datadir = "/group/chem/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if self.verbose: print("Loading meteorology from " + met_datadir)

        # each chunk should have around 1mill values,  - chunk per level and by time, rounded to the nearest hundred, 100MB-1GB
        # could calcualte this dynamically 
        chunk = False
        if chunk:
            time_chunk = 500 #round(1000000/(self.metsize*self.metsize), -2) #
            chunk_args = {"chunks" : {"time": time_chunk}}
        else:
            chunk_args = {}
        
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            with xr.open_mfdataset(sorted(glob.glob(met_datadir)),  concat_dim="time", combine="nested", data_vars="minimal", coords="minimal", parallel=True, join="inner", **chunk_args, drop_variables=["forecast_period", "forecast_reference_time", "level_height_0", "sigma_0"], compat="override", preprocess=remove_duplicates) as met_file:

                #) rename, select levels and variables
                if "model_level_number" in met_file.dims:
                    met_file = met_file.rename({"model_level_number": "levels", "latitude":"lat", "longitude":"lon"})

                met_file = met_file.drop_duplicates(dim=["lat", "lon", "time"])
                self.met_file = met_file.copy()


    def _get_domain(self, region):
        #### check domains
        # TODO make domains dict importable
        domains = {"BRAZIL":"SOUTHAMERICA", "SOUTHAMERICA":"SOUTHAMERICA", "SAHARA":"NORTHAFRICA", "INDIA":"SOUTHASIA"} 
        try:
            domain = domains[region]   
        except: 
            raise ValueError("No domain was passed, and the region you passed is not associated to any domain!")   
        
        return domain   

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
        # assumes the same resolution and domain as the footprints, unless padding is passed (as a dict of shape {"lat":(0,0), "lon":(0,0)})
        """

        lat_values = list(self.fp_data_full.lat.values)
        lon_values = list(self.fp_data_full.lon.values)

        if padding is not None and padding != {"lat":(0,0), "lon":(0,0)}:
            delta_lon = lon_values[1]-lon_values[0]
            delta_lat = lat_values[1]-lat_values[0]               
            lat_values = np.array(sorted(lat_values + [np.max(lat_values)+delta_lat*i for i in range(5+padding["lat"][1])]+ [np.min(lat_values)-delta_lat*i for i in range(5+padding["lat"][0])]))
            lon_values = np.array(sorted(lon_values + [np.max(lon_values)+delta_lon*i for i in range(5+padding["lon"][1])]+ [np.min(lon_values)-delta_lon*i for i in range(5+padding["lon"][0])]))       


        topog_file = topog_file.interp(latitude=lat_values, longitude=lon_values).rename({"latitude":"lat", "longitude":"lon"})

        return topog_file

    def _interp_landcover(self, landcover_file, padding=None):
        """
        loads the landcover file, interpolates
        # assumes the same resolution and domain as the footprints, unless padding is passed (as a dict of shape {"lat":(0,0), "lon":(0,0)})
        """
        lat_values = list(self.fp_data_full.lat.values)
        lon_values = list(self.fp_data_full.lon.values)

        if padding is not None and padding != {"lat":(0,0), "lon":(0,0)}:
            delta_lon = lon_values[1]-lon_values[0]
            delta_lat = lat_values[1]-lat_values[0]               
            lat_values = np.array(sorted(lat_values + [np.max(lat_values)+delta_lat*i for i in range(5+padding["lat"][1])]+ [np.min(lat_values)-delta_lat*i for i in range(5+padding["lat"][0])]))
            lon_values = np.array(sorted(lon_values + [np.max(lon_values)+delta_lon*i for i in range(5+padding["lon"][1])]+ [np.min(lon_values)-delta_lon*i for i in range(5+padding["lon"][0])]))   


        landcover_file = landcover_file.assign_coords(lon=(((landcover_file.lon + 180) % 360) - 180))
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
                print("met file domain is smaller than the footprint domain! interpolating to the same grid as the footprint, but this will create artifacts in the meteorology! \n You may want to set crop_to_intersection=True.")

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
            print("Error occurred while processing country mask:", e)
            print("Returning original country dataset without processing")
            self.countries = country_ds


    def plot_footprint(self, idx=0, timestamp=None, vmin_vmax=[None,None], levels=None, background_threshold=1e-4, add_cbar=False, return_fig=False, plot_marker=False):
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

        fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
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
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, verbose = False, fill_outofdomain_with="nans", delete_outofdomain=False, check_for_nans=False, sampling_mode="regular", fp_datadir = None, load_everything=False, lazy_load=True, met_args={}, topog_args={}, load_fps_as="array"):

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
        self._process_footprints(lazy_load, fp_data_as=load_fps_as)

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

    def _process_footprints(self, lazy_load, fp_data_as="array"):
        """
        cut data around release point
        fp data returned is array of shape (time, size*size) with each footprint centered around its release point AND as a full xarray dataset of coordinates time, lat lon where lat and lon are artificial coordinates with range (0,size) and the measurement point is in the center at size//2, size//2
        
        If the square to extract escapes the footprint domain, the padded space is filled with nans or zeros or deleted according to the fill_outofdomain_with and delete_outofdomain parameters.
        """
        if self.verbose: print(f"----- Cutting footprints to square of size {self.size}") 
        self.fp_data, self.fp_lats, self.fp_lons, self.release_idxs, self.padding, self.fp_data_full, self.fp_xr = cut_satellite_data(self.fp_data_full, self.size, return_as=fp_data_as,fill_bads_with=self.fill_outofdomain_with, delete_outofdomain = self.delete_outofdomain, verbose=self.verbose, return_everything=True, load=not lazy_load) 


    def _process_meteorology(self,rechunk=0,lazy_load=True):
        """
        Call cut_satellite_met to cut the meteorology data around the release point, to the same size as the cut footprints.
        If the square to extract escapes the met domain, the padded space is filled with nans or zeros according to the fill_outofdomain_with parameter 
        """
        if self.verbose: print("----- Cutting met")
        self.metsize=self.size
        if self.fill_outofdomain_with=="nans" or self.delete_outofdomain:
            pad_mode = "nans"
        if self.fill_outofdomain_with=="zeros":
            pad_mode = "edge"
        self.met = cut_satellite_met(self.met_file, self.fp_data_full, metsize=self.size, time_delta=0, pad_mode=pad_mode, load=not lazy_load, add_wind_direction=True)

        if rechunk>0:
            self.met.chunk({"time":rechunk})
        
        self.met_processed = True
        return self.met

    
    def _process_topog_and_landcover(self):
        """
        cut topography to the same domain covered by the cut footprints (ie a sizexsize square centered around measurement point)
        inputs are the latitudes and longitudes that the topog was interpolated to (this is clunky)

        uses object attributes self.topog_file, self.fp_data_full, self.size, self.fp_data, returning self.topog as an xarray dataset with dimensions time, lat, lon, and data variables topog (surface altitude), landcover (landcover type) and disaggregated_landcover (sea mask + nine types of land cover fractions). The lat and lon coordinates are artificial coordinates with range (0,size) and the measurement point is in the center at size//2, size//2. If the square to extract escapes the topog domain, the padded space is filled with zeros.

        ENHANCEMENT: the topog dataset is often larger than the footprint dataset, so cutting it and padding with zeros is redundant!
        """
        if self.verbose: print("----- Cutting topog")

        topog_lats = list(self.topog_file.lat.values)
        topog_lons = list(self.topog_file.lon.values)

        topog_release_idxs = _get_release_idxs(self.fp_data_full, domain_lats=topog_lats, domain_lons=topog_lons)

        half = int(self.size/2)
        full_topog=np.zeros_like(self.fp_data)
        full_landcover=np.zeros_like(self.fp_data)
        
        full_topog=np.reshape(full_topog, (len(self.fp_data), self.size, self.size))
        full_landcover=np.reshape(full_landcover, (len(self.fp_data), self.size, self.size))

        n_disagg_landcover_types = 10
        disaggregated_landcover = np.zeros((len(self.fp_data), self.size, self.size, n_disagg_landcover_types)) # sea mask + nine types of land cover

        failed_idxs = []
        for rel_unique in np.unique(topog_release_idxs, axis=0):
            idxs = np.where((topog_release_idxs == rel_unique).all(axis=1))[0]  
            try:
                full_topog[idxs, :,:] = self.topog_file.surface_altitude.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half][np.newaxis, :]
                full_landcover[idxs, :,:] = self.landcover_file.landcover_type.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half][np.newaxis, :]  

                disaggregated_landcover[idxs, :,:,0] = self.landcover_file.land_binary_mask.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half][np.newaxis, :]   
                disaggregated_landcover[idxs, :,:,1:] = self.landcover_file.landcover_fraction.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,:][np.newaxis, :]   


            except IndexError:
                empty = np.empty_like(full_topog[idxs, :,:])
                empty[:] = np.nan
                full_topog[idxs, :,:] = empty
                full_landcover[idxs, :,:] = empty
                failed_idxs = failed_idxs + list(idxs)
            
        
        # reverse sea mask so that 1 is sea and zero is land
        disaggregated_landcover[:, :,:,0] = 1 - disaggregated_landcover[:, :,:,0] 
        # remove the nans that in the original file show the sea mask
        disaggregated_landcover[:, :,:,1:] = np.nan_to_num(disaggregated_landcover[:, :,:,1:], copy=False)
    
        coords = {"time":("time", self.fp_data_full.time.values), "lat":("lat", np.arange(self.size)), "lon":np.arange(self.size), "landcover_level":np.arange(n_disagg_landcover_types)}

        self.topog = xr.Dataset(
            data_vars = 
            {"topog": (["time", "lat", "lon"], full_topog),
            "landcover": (["time", "lat", "lon"], full_landcover),
             "disaggregated_landcover": (["time", "lat", "lon", "landcover_level"], disaggregated_landcover) },
             coords = coords
             )
        
        if len(failed_idxs)>0:
            if self.verbose: print(f"remove {len(failed_idxs)} failed idxs for topog")
            self.remove_indeces(failed_idxs)

        # returning as a netcdf dataset
        return self.topog

    def remove_indeces(self, nan_idxs):
        """
        removes any set of indeces passed as nan_idxs from all the objects in the dataset
        """
        self.fp_data_full = self.fp_data_full.sel(time=np.delete(self.fp_data_full.time.values, nan_idxs))
        if hasattr(self, "fp_data"):
            self.fp_lats = np.delete(self.fp_lats, nan_idxs, axis=0)
            self.fp_lons = np.delete(self.fp_lons, nan_idxs, axis=0)
            self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
            if self.verbose: print(f"current length: {len(self.release_idxs)}")
            self.release_idxs = np.delete(self.release_idxs, nan_idxs, axis=0)
        if hasattr(self, "fp_xr"):
            self.fp_xr = self.fp_xr.sel(time=np.delete(self.fp_xr.time.values, nan_idxs))
        if hasattr(self, "met"): 
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))
        if hasattr(self, "topog"):
            self.topog = self.topog.sel(time=np.delete(self.topog.time.values, nan_idxs))

        if self.verbose: print(f"Length after removing indeces: {self.fp_data_full.time.size}")

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


def load_default_brazil_emissions(year=2016, month_to_use=6):
    emissions = xr.open_dataset("/group/chem/acrg/LPDM/emissions/SOUTHAMERICA/ch4_SOUTHAMERICA_2016_SWAMPS-v32-5_Saunois-Annual-Mean.nc")
    emissions = emissions.sel(time=emissions.time[month_to_use-1])
    return emissions.flux.values

def load_default_sahara_emissions(year=2016, month_to_use=6):
    emissions = xr.open_dataset(f"/group/chem/acrg/LPDM/emissions/NORTHAFRICA/ch4_NORTHAFRICA_{year}.nc")
    emissions = emissions.sel(time=emissions.time[month_to_use-1])
    return emissions.flux.values

def cut_emissions_data(flux, fp_full, size):
    # this assumes flux is a 2D np array of the same resolution and size as the footprints! it also assumes that the data is cut to a square size

    release_idxs = _get_release_idxs(fp_full)

    flux_cut = np.zeros((size, size, len(fp_full.time)))
    half = int(size/2) 

    # release indeces aren't unique so to save memory, process all footprints with same release at once
    for rel_unique in np.unique(release_idxs, axis=0):
        # find indeces across the time axis of footprints that have rel_unique as their release coordinates
        idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]
        try:
            emitted = flux[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half]
            flux_cut[:,:,idxs] = emitted[:,:,None]
            
        except ValueError:
            # footprint is pratially outside of domain, cut  
            lower_lat = np.max((0, rel_unique[0]-half))
            lower_lon = np.max((0, rel_unique[1]-half))
            upper_lat = np.min((len(fp_full.lat.values), rel_unique[0]+half))
            upper_lon = np.min((len(fp_full.lon.values), rel_unique[1]+half))

            # cut the part of the footprint within domain area
            emitted = flux[lower_lat:upper_lat,lower_lon:upper_lon]     

            # upper_cut and lower_cut are the coordinates of the area that has been cut within the full area the cut footprint should cover
            lower_cut_lat = np.max((0, -(rel_unique[0]-half)))
            lower_cut_lon = np.max((0, -(rel_unique[1]-half)))
            upper_cut_lat = np.min(((rel_unique[0]+half)-len(fp_full.lat.values), size))
            if (rel_unique[0]+half)-len(fp_full.lat.values)>0: 
                upper_cut_lat=size-((rel_unique[0]+half)-len(fp_full.lat.values))
            else: upper_cut_lat=size
            if (rel_unique[1]+half)-len(fp_full.lon.values)>0: 
                upper_cut_lon=size-((rel_unique[1]+half)-len(fp_full.lon.values))
            else: upper_cut_lon=size                

            # save the part of the footprint that is within the domain
            flux_cut[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon, idxs] = emitted[:,:,None]     

    return flux_cut


def cut_satellite_data(fp_full, size, fill_bads_with="nans", delete_outofdomain=False, load=True, return_as="netcdf", verbose=True,  return_everything=False):
    """
    cuts footprint to square of size size x size gridcells around the footprint's release point, returning with artificial lat-lon coordinates where the release point is at the middle of the grid [size/2, size/2]

    Sometimes the square of size size x size around the release point escapes the footprint domain in at least one direction. This is more likely to happen for bigger sizes, and for release points near the edge of the domain.
    If size is bigger than the original footprint domain, the function will pad the cut footprint with either nans or zeros, depending on fill_bads_with, and if delete_outofdomain is False. If delete_outofdomain is True, the function will drop the footprints that escape the domain when cut to size.


    Inputs:
    - fp_full - full xr array with footprints. Should have variables .fp, .release_lat and .release_lon 

    - size - size of square to cut footprints to. Must be even to ensure that the release point is in the middle of the cut footprint.

    - fill_bads_with - str, options are "nans" and "zeros". If "nans" or "zeros", only the parts out of the domain are set to "nans" or "zeros" respectively. 
    
    - delete_outofdomain - bool, if True, deletes footprints that need to be padded (i.e. that escape the domain in at least one direction when cut to size). If False, keeps them and fills the part of the cut footprint that escapes the domain with either nans or zeros, depending on fill_bads_with. 
    
    the returns are a bit convoluted at the moment. Ideally, we move towards using only return_as="netcdf" and returning everything as an xarray with coordinates, but for now, to avoid breaking existing code, we have the following options:
    - return_as - str, options are "array", "netcdf". If "array" returns as array of shape (time, size*size), if "netcdf" returns as an xarray with coordinates
    - return_everything - bool, if True, returns extra variables together with the footprints.
        - if return_as="array", returns
            fp_data (array of size time x (size*size)), fp_lats (array of size time x lat), fp_lons (array of size time x lon), release_idxs (array of size time x 2 with the grid-indeces of the release point for each footprint with respect to the original domain), padding (dict with the number of gridcells that were padded in each direction, if any), something else i need to check
        - if return_as="netcdf", returns
            cropped_fp (xarray of the cropped footprints with coordinates time, lat, lon where lat and lon are artificial coordinates centered on the release point 0-size. the footprints are stored in variable .fp , the original lat and lon coordinates of the cut footprints are stored in variables .lat_coords and .lon_coords) release_idxs, padding (as above)
        
        if return_as = "both", returns all of the above, in the order of return_as="array" followed by the cropped xarray. Using this while transitioning
         
    """

    if size%2!=0:
        raise ValueError("size should be even to ensure that the release point is in the middle of the cut footprint")
    
    half = int(size/2)
    release_idxs = _get_release_idxs(fp_full)

    padding_needed = False

    if fill_bads_with not in ["nans", "zeros"]:
        raise ValueError("fill_bads_with should be either 'nans' or 'zeros'")

    if delete_outofdomain or fill_bads_with == "nans":
        constant_values = np.nan
    if fill_bads_with == "zeros":
        constant_values = 0
    
    domain_lats = np.copy(fp_full.lat.values)
    domain_lons = np.copy(fp_full.lon.values)

    original_fp_domain = (fp_full.lat.values[0], fp_full.lat.values[-1], fp_full.lon.values[0], fp_full.lon.values[-1])

    # this dict stores how many footprints needed to be expanded in each direction
    # a footprint that needs padding in multiple directions is counted multiple times
    fp_needed_padding_direction = {"N":0, "S":0, "E":0, "W":0}
    # check if we need to pad in any direction
    south_padding = release_idxs[:,0] < half
    north_padding = (len(domain_lats) - release_idxs[:,0]) < half
    west_padding = release_idxs[:,1] < half
    east_padding = (len(domain_lons) - release_idxs[:,1]) < half
    fp_needed_padding_direction["S"] = np.sum(south_padding)
    fp_needed_padding_direction["N"] = np.sum(north_padding)
    fp_needed_padding_direction["E"] = np.sum(east_padding)
    fp_needed_padding_direction["W"] = np.sum(west_padding)
    n_unique_padded_fps = np.sum(np.any([south_padding, north_padding, east_padding, west_padding], axis=0))
    
    padded_fps_idxs = np.where(np.any([south_padding, north_padding, east_padding, west_padding], axis=0))[0]
    padding = {"lat":(0,0), "lon":(0,0)}

    if delete_outofdomain and len(padded_fps_idxs)>0:
        if verbose: print(f"for {len(padded_fps_idxs)} footprints, a square of size x size escapes the footprint domain in directions {fp_needed_padding_direction}. \n dropping these! if you want to keep them anyway, pass delete_outofdomain=False")
        fp_full = fp_full.drop_sel(time=fp_full.time[padded_fps_idxs])
        fp_needed_padding_direction = {"N":0, "S":0, "E":0, "W":0}
        release_idxs = _get_release_idxs(fp_full)  

    ## TODO make padding its own function
    # 1) check if any footprints, when cut to size, will escape the domain
    # if so, pad the array with either zeros or nans 
    if fp_needed_padding_direction["S"]>0  or fp_needed_padding_direction["N"]>0:

        delta_lat = domain_lats[1] -domain_lats[0]
        to_pad = (np.max([0, half - np.min(release_idxs[:,0])]) , np.max([0, half - (len(domain_lats) - np.max(release_idxs[:,0]))]))
        padding["lat"] = to_pad
        print(f"careful! We had to pad the footprints along the latitude dimension to extract size {size} ({to_pad[0]} and {to_pad[1]} idxs on either side) Padding with {constant_values}") 
        
        fp_full = fp_full.pad(pad_width={"lat":to_pad}, constant_values=constant_values)
    
        padding_needed = True

        # reassign coordinates to ensure that padded values have the right coordinate spacing
        updated_lats = sorted([np.min(domain_lats)-(i+1)*delta_lat for i in range(to_pad[0])]) + list(domain_lats) + sorted([np.max(domain_lats)+(i+1)*delta_lat for i in range(to_pad[1])])
        fp_full = fp_full.assign_coords({"lat":updated_lats})  
          

    if fp_needed_padding_direction["E"]>0  or fp_needed_padding_direction["W"]>0:
        delta_lon = domain_lons[1] - domain_lons[0]
        to_pad = (np.max([0, half - np.min(release_idxs[:,1])]) , np.max([0, half - (len(domain_lons) - np.max(release_idxs[:,1]))]))
        padding["lon"] = to_pad
        print(f"careful! We had to pad the footprints along the longitude dimension to extract size {size} ({to_pad[0]} and {to_pad[1]} idxs on either side) Padding with {constant_values}") 
        
        fp_full = fp_full.pad(pad_width={"lon":to_pad}, constant_values=constant_values)
    
        padding_needed = True
        
        # reassign coordinates to ensure that padded values have the right spacing
        updated_lons = sorted([np.min(domain_lons)-(i+1)*delta_lon for i in range(to_pad[0])]) + list(domain_lons) + sorted([np.max(domain_lons)+(i+1)*delta_lon for i in range(to_pad[1])])
        fp_full = fp_full.assign_coords({"lon":updated_lons}) 
        

    if padding_needed:
        # recalculate the release indeces to account for the new padding that was just added
        domain_lats = np.copy(fp_full.lat.values)
        domain_lons = np.copy(fp_full.lon.values)
        release_idxs = _get_release_idxs(fp_full)  

        if verbose: # print some stats 
            print(f"{n_unique_padded_fps} footprints were at least partially filled with {fill_bads_with} because they were cutting outside of the footprint file domain (this is {round(100*n_unique_padded_fps/len(fp_full.time.values), 2)}% of samples)")
            print(f"Padding was needed for the following number of footprints along each direction: {fp_needed_padding_direction}")
    
    cropped_arrays = []

    coords_array = np.arange(size)

    
    # 2) crop the data to an array of sizexsize, for each unique coordinate. store as a list of small arrays with artificial lat-lon coordinates 0-size, concatenate at the end along the time dimension 
    for rel_unique in np.unique(release_idxs, axis=0):
        # find the corresponding timestamps
        idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]

        # crop the meteorology around the releasepoint
        cutfp = fp_full.sel(time=fp_full.time.values[idxs], lat=domain_lats[rel_unique[0]-half:rel_unique[0]+half], lon=domain_lons[rel_unique[1]-half:rel_unique[1]+half])          

        # copy the latitude/longitude values for this specific cropped square
        lats = cutfp.lat.values.copy()
        lons = cutfp.lon.values.copy()
        #print(cutfp.fp)
        # store them as variables to use as inputs and assign artificial coordinates
        cutfp = cutfp.assign_coords({"lat":coords_array, "lon":coords_array}).assign({"lat_coords":(("lat"), lats), "lon_coords":(("lon"), lons)})

        cropped_arrays.append(cutfp)

    # concatenate all of the cropped arrays
    cropped_fp = xr.concat(cropped_arrays, dim="time")
    cropped_fp = cropped_fp.sortby("time") 
    cropped_fp = cropped_fp[["fp", "lat_coords", "lon_coords", "release_lat", "release_lon"]]
    cropped_fp = cropped_fp.transpose("time", "lat", "lon") # make sure fp is in the right order of dimensions
    cropped_fp = cropped_fp.chunk({"time":100, "lat":-1, "lon":-1}) # chunk along time to avoid memory issues, can be loaded into memory later if needed

    # load into memory, if required
    if load:
        print("loading cropped footprint dataset into memory. If you only want to lazy-load, pass load=False")
        cropped_fp.load()    

    if return_as=="netcdf":
        if return_everything:
            return cropped_fp, release_idxs, padding
        else:
            return cropped_fp
    elif return_as=="array":
        fp_data = cropped_fp.fp.transpose("time","lat", "lon").values
        fp_data = np.reshape(fp_data, (len(cropped_fp.time), size**2))

        if return_everything:
            fp_lats = cropped_fp.lat_coords.values
            fp_lons = cropped_fp.lon_coords.values  
            return fp_data, fp_lats, fp_lons, release_idxs, padding, fp_full.sel(lat=slice(original_fp_domain[0], original_fp_domain[1]), lon=slice(original_fp_domain[2], original_fp_domain[3]))               
        else:
            return fp_data     
    elif return_as=="both":
        fp_data = cropped_fp.fp.transpose("time","lat", "lon").values
        fp_data = np.reshape(fp_data, (len(cropped_fp.time), size**2))

        if return_everything:
            fp_lats = cropped_fp.lat_coords.values
            fp_lons = cropped_fp.lon_coords.values  
            return fp_data, fp_lats, fp_lons, release_idxs, padding, fp_full.sel(lat=slice(original_fp_domain[0], original_fp_domain[1]), lon=slice(original_fp_domain[2], original_fp_domain[3])), cropped_fp


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
    return met, nan_idxs


def _pad_met_domain(met, fp, release_idxs, half, pad_mode):
    """
    Extends the met domain so that a (2*half) x (2*half) crop is possible for
    every release point.  Returns (met_padded, domain_lats, domain_lons,
    updated_release_idxs).

    For pad_mode='nans': uses xr.reindex with fill_value=np.nan — lazy, and
    coordinates are defined upfront so no post-hoc patching is needed.
    For pad_mode='edge': uses xr.pad(mode='edge') then assigns the correct
    extended coordinate values.
    """
    domain_lats = met.lat.values.copy()
    domain_lons = met.lon.values.copy()
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
        if pad_mode == "nans":
            met = met.reindex(lat=extended_lats, fill_value=np.nan)
        else:  # "edge"
            met = met.pad(pad_width={"lat": (pad_S, pad_N)}, mode="edge")
            met = met.assign_coords({"lat": extended_lats})
        domain_lats = met.lat.values.copy()
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
        if pad_mode == "nans":
            met = met.reindex(lon=extended_lons, fill_value=np.nan)
        else:  # "edge"
            met = met.pad(pad_width={"lon": (pad_W, pad_E)}, mode="edge")
            met = met.assign_coords({"lon": extended_lons})
        domain_lons = met.lon.values.copy()
        padding_needed = True

    if padding_needed:
        release_idxs = _get_release_idxs(fp, domain_lats, domain_lons)

    return met, domain_lats, domain_lons, release_idxs


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
    met, domain_lats, domain_lons, release_idxs = _pad_met_domain(
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





