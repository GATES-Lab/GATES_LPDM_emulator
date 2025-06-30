import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os
import copy
#import warning

from .load_data_helper_funs import *

def load_fps(fp_datadir, verbose=False):
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
        time_chunk = 25
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # attempt to load dataset of multiple files the standard way
            with xr.open_mfdataset(sorted(glob.glob(fp_datadir)), combine='by_coords', chunks = {"time":time_chunk}) as ds:
                fp_data_full = ds.copy()
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
                with xr.open_mfdataset(sorted(without_bad_files)) as ds:
                    most = ds.copy()
                bad_arrays = []
                for badfile in bad_files:
                    if badfile in fp_files:
                        # load each bad file separately
                        with xr.open_mfdataset(badfile) as f_bad:
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




class LoadBaseSatelliteData:
    """
    Parent class for loading Satellite Data. Loads footprint, meteorological and topography data for a particular domain and time period. 

    main inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - month: str in format "01" for January etc, None if loading a whole year
        - region: region identifyer, as a string. Default is Brazil. Current set-up has regions "BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA"
        (note - Brazil is a subset of South America!)
        - domain: Domain related to the region, used for file search (due to existing filenaming conventions). Set-up regions ("BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA") have a default domain, all others need domain passed
    - 

    other inputs
        - freq: int, frequency of the data to load. freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset
        - sampling_mode: if "regular", subsamples footprints regularly (e.g. one in every two, sequentially with freq=2). if random, subsamples N/freq footprints randomly (where N is the total number of footprints). default is "regular"
        - freq_offset: if using sampling_mode="regular", offsets the start of the regular sampling, e.g. freq=2 and freq_offset=0 will sample even footprints, and freq_offset=1 will sample uneven footprints
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. Applied after sampling with freq (or pass freq=1 to load all footprints)
        - fp_datadir: str, directory for footprints. default directs to ACRG folder. If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass fp_datadir="/path/name_of_your_choice_"
        - verbose: if True, prints out the steps throughout the data loading process
    
    met_args:
        see load_meteorology()
    topog_args:
        see load_topogs()
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
        load the meteorology from the directory, select the met levels and variables if required

        Inputs
            - met_datadir (str): directory for meteorology. default directs to ACRG meteorology folder. If passing as arg, the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/name_of_your_choice_"
            - met_levels (list): met levels to select
            - met_variables (list) : met variables to select
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
        print("\n---- LOADING TOPOG")
        if topog_path=="default":
            topog_path="/group/chemistry/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc"
        print(f"trying to load topography from {topog_path}")
        with xr.load_dataset(topog_path) as topog_dataset:
            topog_file = topog_dataset.copy()

        if landcover_path=="default":
            landcover_path = "/group/chemistry/acrg/LPDM/topog_NAME/land_cover.nc"
        with xr.load_dataset(landcover_path) as landcover_dataset:
            landcover_file = landcover_dataset.copy()
            

        topog_file = self._interp_topog(topog_file, padding=self.padding)

        landcover_file = self._interp_landcover(landcover_file, padding=self.padding)

        return topog_file, landcover_file

    def _get_meteorology_file(self, met_datadir, lazy_load=True):
        if met_datadir==None:
            met_datadir = "/group/chemistry/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if self.verbose: print("Loading meteorology from " + met_datadir)

        # each chunk should have around 1mill values,  - chunk per level and by time, rounded to the nearest hundred, 100MB-1GB
        # could calcualte this dynamically 
        time_chunk = 500 #round(1000000/(self.metsize*self.metsize), -2) #
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            with xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords', data_vars="minimal", coords="minimal", parallel=True, join="inner", chunks = {"level":1, "time":time_chunk}) as met_file:

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
        #### load footprint (fp) data from file
        if fp_datadir is None:
            fp_datadir = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/"+self.domain+"/*"+self.region+"*"+self.domain+"_"+str(self.date)+"*.nc"
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
    


class LoadSquareSatelliteData(LoadBaseSatelliteData):
    """
    Load footprint and meteorological data for a particular domain and time period, outputting all data cut to a square centered around the measurement point for each timestamp. 

    Inherites the loading functions from the general LoadBaseSatelliteData

    main inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - month: str in format "01" for January etc, None if loading a whole year
        - region: region identifyer, as a string. Default is Brazil. Current set-up has regions "BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA"
        (note - Brazil is a subset of South America!)
        - domain: Domain related to the region, used for file search (due to existing filenaming conventions). Set-up regions ("BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA") have a default domain, all others need domain passed
        - size: size for footprint to be cut to, as an int. Resolution of the footprint is maintained, cut to a sizexsize square around the release point. 
    - 

    other inputs
        - freq: int, frequency of the data to load. freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset
        - sampling_mode: if "regular", subsamples footprints regularly (e.g. one in every two, sequentially with freq=2). if random, subsamples N/freq footprints randomly (where N is the total number of footprints). default is "regular"
        - freq_offset: if using sampling_mode="regular", offsets the start of the regular sampling, e.g. freq=2 and freq_offset=0 will sample even footprints, and freq_offset=1 will sample uneven footprints
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. Applied after sampling with freq (or pass freq=1 to load all footprints)
        - fp_datadir: str, directory for footprints. default directs to ACRG folder. If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass fp_datadir="/path/name_of_your_choice_"
        - fill_outofdomain_with: str, out of "all_nans", "nans" and "zeros". Determines what to do if any part of the square cut around the footprint is outside of the domain. "nans" and "zeros" fill only the out of domain areas with nans and zeros respectively. 
        - delete_outofdomain: bool, if True delete all footprints (and associated datapoints) where the extracted area size x size escapes the domain


        - verbose: if True, prints out the steps throughout the data loading process
    
    met_args:
        see load_meteorology()
    topog_args:
        see load_topog()
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, verbose = False, fill_outofdomain_with="nans", delete_outofdomain=False, check_for_nans=False, sampling_mode="regular", fp_datadir = None, load_everything=False, lazy_load=True, met_args={}, topog_args={}):

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
        
        #### load footprint (fp) data    
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
        fp data returned is array of shape (time, size*size) with each footprint centered around its release point
        """
        if self.verbose: print(f"----- Cutting footprints to square of size {self.size}") 
        self.fp_data, self.fp_lats, self.fp_lons, self.release_idxs, self.padding, self.fp_data_full = cut_satellite_data(self.fp_data_full, self.size, returnlatlons = True, return_as="array",fill_bads_with=self.fill_outofdomain_with, delete_outofdomain = self.delete_outofdomain, verbose=self.verbose, return_everything=True, load=not lazy_load) 

        # if self.fill_outofdomain_with == "all_nans":
            # remove here all indeces from self.idxs_out_of_domain! so we only keep the footprints that were fully inside the domain

    def _process_meteorology(self,rechunk=0,lazy_load=True):
        if self.verbose: print("----- Cutting met")
        self.metsize=self.size
        if self.fill_outofdomain_with=="nans" or self.delete_outofdomain:
            pad_mode = "nans"
        if self.fill_outofdomain_with=="zeros":
            pad_mode = "edge"
        self.met = cut_satellite_met_v4(self.met_file, self.fp_data_full, metsize=self.size, time_delta=0, pad_mode=pad_mode, load=not lazy_load, add_wind_direction=True)

        if rechunk>0:
            self.met.chunk({"time":rechunk})
        
        self.met_processed = True
        return self.met

    
    def _process_topog_and_landcover(self):
        """
        cut topography to the same domain covered by the cut footprints (ie a sizexsize square centered around measurement point)
        inputs are the latitudes and longitudes that the topog was interpolated to (this is clunky)

        uses object attributes self.topog_file, self.fp_data_full, self.size, self.fp_data
        """
        if self.verbose: print("----- Cutting topog")

        topog_lats = list(self.topog_file.lat.values)
        topog_lons = list(self.topog_file.lon.values)

        topog_release_idxs = get_release_idxs(self.fp_data_full, domain_lats=topog_lats, domain_lons=topog_lons)

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
            
        if hasattr(self, "met"): 
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))
        if hasattr(self, "topog"):
            self.topog = self.topog.sel(time=np.delete(self.topog.time.values, nan_idxs))

        if self.verbose: print(f"Length after removing indeces: {self.fp_data_full.time.size}")

    def _remove_fp_nans(self):
        ## check if any of the fp entries are nans, and if so remove from met and others
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

class LoadDomainSatelliteData(LoadBaseSatelliteData):
    """
    Cuts the dataset to a common fixed domain. By default, cuts to the biggest domain that is shared by the footprints and the met

    to specify the area to cut, pass domain_to_cut as a dict of format {"lat":[start_lat, end_lat], "lon":[start_lon, end_lon]} in degrees. If either lat or lon is missing, they will be the largest possible

    all other inputs are the same
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, domain_to_cut=None, freq=1, freq_offset=0, verbose = False, sampling_mode="regular", fp_datadir = None, met_args={}, topog_args={}):
        
        ## INITIALISE THE BASE OBJECT, TO LOAD THE FOOTPRINTS
        super().__init__(year, region, month=month, domain=domain, freq=freq, freq_offset=freq_offset, verbose=verbose, sampling_mode=sampling_mode, fp_datadir=fp_datadir, load_everything=True, met_args=met_args, topog_args=topog_args)

        self.dataset_format = "domain"

        if verbose: print("-----CROPPING TO A FIXED DOMAIN")

        # calculate the max possible domain, given by the footprint and the met files
        lats = [np.max([self.fp_data_full.lat.values[0], self.met_file.lat.values[0]]), np.min([self.fp_data_full.lat.values[-1], self.met_file.lat.values[-1]])]
        lons = [np.max([self.fp_data_full.lon.values[0], self.met_file.lon.values[0]]), np.min([self.fp_data_full.lon.values[-1], self.met_file.lon.values[-1]])]
        max_allowed_domain = {"lat":lats, "lon":lons}
        

        if domain_to_cut is None:
            self.domain_to_cut = max_allowed_domain
            rounded_dom = {key : [round(self.domain_to_cut[key][i], 3) for i in range(2)] for key in self.domain_to_cut}
            if verbose: print(f"cutting everything to the minimum possible domain: {rounded_dom}" )
        else:
            self.domain_to_cut = self._check_domain_sizes(domain_to_cut, max_allowed_domain)
        
        # slice to that domain
        self._slice_to_domain(self.domain_to_cut)

        ## after cropping, need to arrange the datasets in the same way as the square ones!
        self._process_footprints()
        self.met = process_domain_met(self.met_file, self.fp_data_full)
        self.met_processed=True
        self.topog = self._process_topog_and_landcover()



    def _check_domain_sizes(self, domain_to_cut, max_allowed_domain):
        # fill in lat or lon in case only one was passed
        if not "lat" in domain_to_cut.keys():
            domain_to_cut["lat"] = max_allowed_domain["lat"]
        if not "lon" in domain_to_cut.keys():
            domain_to_cut["lon"] = max_allowed_domain["lon"]
        
        self.domain_to_cut = copy.deepcopy(domain_to_cut)

        self.domain_to_cut["lat"][0] = np.max([domain_to_cut["lat"][0], max_allowed_domain["lat"][0]])
        
        self.domain_to_cut["lat"][-1] = np.min([domain_to_cut["lat"][-1], max_allowed_domain["lat"][-1]])
    
        self.domain_to_cut["lon"][0] = np.max([domain_to_cut["lon"][0], max_allowed_domain["lon"][0]])
        
        self.domain_to_cut["lon"][-1] = np.min([domain_to_cut["lon"][-1], max_allowed_domain["lon"][-1]])

        if self.domain_to_cut != domain_to_cut:
            rounded_dom = {key : [round(self.domain_to_cut[key][i], 3) for i in range(2)] for key in self.domain_to_cut}
            print(f"the domain you passed is bigger than the domain of the data in at least one direction. cropping to {rounded_dom}")

        return self.domain_to_cut

    def _slice_to_domain(self, domain_to_cut):
        ## slice all arrays to the passed domain
        self.fp_data_full = self.fp_data_full.sel(lat=slice(domain_to_cut["lat"][0]-0.0001, domain_to_cut["lat"][1]+0.0001), lon=slice(domain_to_cut["lon"][0]-0.0001, domain_to_cut["lon"][1]+0.0001))

        # removing footprints that arent within the slice domain
        lat_min, lat_max = self.fp_data_full.lat.min().item(), self.fp_data_full.lat.max().item()
        lon_min, lon_max = self.fp_data_full.lon.min().item(), self.fp_data_full.lon.max().item()
        valid_times = (
        (self.fp_data_full.release_lat.values >= lat_min) & (self.fp_data_full.release_lat.values <= lat_max) &
        (self.fp_data_full.release_lon.values >= lon_min) & (self.fp_data_full.release_lon.values <= lon_max))
        
        self.fp_data_full = self.fp_data_full.sel(time=self.fp_data_full.time[valid_times])

        if self.verbose and np.sum(valid_times)<len(valid_times): print(f"keeping only the {np.sum(valid_times)} footprints where the release point is within the defined domain")


        if hasattr(self, "met_file"):
            self.met_file = self.met_file.sel(lat=slice(domain_to_cut["lat"][0]-0.0001, domain_to_cut["lat"][1]+0.0001), lon=slice(domain_to_cut["lon"][0]-0.0001, domain_to_cut["lon"][1]+0.0001))

        if hasattr(self, "topog_file"):
            self.topog_file = self.topog_file.sel(lat=slice(domain_to_cut["lat"][0], domain_to_cut["lat"][1]), lon=slice(domain_to_cut["lon"][0], domain_to_cut["lon"][1]))

        if hasattr(self, "landcover_file"):
            self.landcover_file = self.landcover_file.sel(lat=slice(domain_to_cut["lat"][0], domain_to_cut["lat"][1]), lon=slice(domain_to_cut["lon"][0], domain_to_cut["lon"][1]))

    def _process_footprints(self):
        self.fp_data = self.fp_data_full.fp.transpose("time","lat", "lon").values
        self.fp_data = np.reshape(self.fp_data, (self.fp_data_full.time.size, self.fp_data_full.lat.size*self.fp_data_full.lon.size))

        self.fp_lats = self.fp_data_full.lat.values
        self.fp_lats = self.fp_lats[np.newaxis, :]
        self.fp_lons = self.fp_data_full.lon.values
        self.fp_lons = self.fp_lons[np.newaxis, :]

        self.domain_size = [self.fp_data_full.lat.size, self.fp_data_full.lon.size]


    def _process_topog_and_landcover(self):
        stacked_landcover = xr.concat([self.landcover_file.land_binary_mask.assign_coords(pseudo_level=0).rename("disaggregated_landcover"), self.landcover_file.landcover_fraction.rename("disaggregated_landcover")], dim="pseudo_level").rename({"pseudo_level":"landcover_level"})

        topog = xr.merge([self.topog_file.surface_altitude.rename("topog"), self.landcover_file.landcover_type.rename("landcover"), stacked_landcover])
        return topog

    def remove_indeces(self, nan_idxs):
        """
        removes any set of indeces passed as nan_idxs from all the objects in the dataset
        """
        if self.verbose: print(f"Length before removing indeces: {self.fp_data_full.time.size}")

        self.fp_data_full = self.fp_data_full.sel(time=np.delete(self.fp_data_full.time.values, nan_idxs))
        if hasattr(self, "fp_data"):
            self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
            
        if hasattr(self, "met"): 
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))

        if self.verbose: print(f"Length after removing indeces: {self.fp_data_full.time.size}")


class LoadDomainSiteData(LoadDomainSatelliteData):
    def __init__(self, year, site = "MHD", month=None, size=None, freq=1, domain_to_cut=None, domain=None, verbose = False, fp_datadir = None, lazy_load=True, met_args={}, topog_args={}):  

        #### check domains
        self.site = site
        if domain is None:
            self.domain = self._get_domain(site)
        else:
            self.domain=domain

        super().__init__(year=year, month=month, region=site, domain=self.domain, fp_datadir=fp_datadir, freq=freq, met_args=met_args, topog_args=topog_args, verbose=verbose)

        self.data_type="site"
        self.site = site

        # sites have fixed release coordinates, so can just extract the release lat and lon from the first timestep
        # self.release_coords = [site_lat, site_lon]
        self.release_coords = [self.fp_data_full.sel(time=self.fp_data_full.time.values[0]).release_lat.values, self.fp_data_full.sel(time=self.fp_data_full.time.values[0]).release_lon.values]


    def _get_release_idxs(self):
        idx_release_lat = np.argmin(abs(self.fp_data_full.lat.values - self.release_coords[0]))
        idx_release_lon = np.argmin(abs(self.fp_data_full.lon.values - self.release_coords[1]))
        release_idxs = [idx_release_lat, idx_release_lon]
        return release_idxs
    
    def _get_domain(self, site):
        #### check domains
        # TODO make domains dict importable
        domains = {"MHD":"EUROPE", "GSN":"EASTASIA"} 
        try:
            domain = domains[site]   
        except: 
            raise ValueError("No domain was passed, and the region you passed is not associated to any domain!")   
        
        return domain   


class LoadSquareSiteData(LoadSquareSatelliteData):
    def __init__(self, year, site = "MHD", month=None, domain=None, freq=1, size=10, freq_offset=0, verbose = False, fp_datadir = None, lazy_load=True, met_args={}, topog_args={}):  
        #### check domains
        self.site = site
        if domain is None:
            self.domain = self._get_domain(site)
        else:
            self.domain=domain

        super().__init__(year=year, month=month, region=site, freq=freq, domain=self.domain, size=size, fp_datadir=fp_datadir, met_args=met_args, topog_args=topog_args, verbose=verbose, load_everything=True)

        self.data_type="site"

        # sites have fixed release coordinates, so can just extract the release lat and lon from the first timestep
        # self.release_coords = [site_lat, site_lon]
        self.release_coords = [self.fp_data_full.sel(time=self.fp_data_full.time.values[0]).release_lat.values, self.fp_data_full.sel(time=self.fp_data_full.time.values[0]).release_lon.values]


    def _get_release_idxs(self):
        idx_release_lat = np.argmin(abs(self.fp_data_full.lat.values - self.release_coords[0]))
        idx_release_lon = np.argmin(abs(self.fp_data_full.lon.values - self.release_coords[1]))
        release_idxs = [idx_release_lat, idx_release_lon]
        return release_idxs

    def _get_domain(self, site):
        #### check domains
        # TODO make domains dict importable
        domains = {"MHD":"EUROPE", "GSN":"EASTASIA"} 
        try:
            domain = domains[site]   
        except: 
            raise ValueError("No domain was passed, and the region you passed is not associated to any domain!")   
        
        return domain   


class LoadBaseSiteData(LoadBaseSatelliteData):
    def __init__(self, year, site = "MHD", month=None, domain=None, freq=1, freq_offset=0, verbose = False, fp_datadir = None, lazy_load=True, met_args={}, topog_args={}):  
        #### check domains
        self.site = site
        if domain is None:
            self.domain = self._get_domain(site)
        else:
            self.domain=domain

        super().__init__(year=year, month=month, region=site, domain=domain, fp_datadir=fp_datadir, met_args=met_args, topog_args=topog_args, verbose=verbose, load_everything=True)

        del self.region
        self.data_type="site"

        # sites have fixed release coordinates, so can just extract the release lat and lon from the first timestep
        # self.release_coords = [site_lat, site_lon]
        self.release_coords = [self.fp_data_full.sel(time=self.fp_data_full.time.values[0]).release_lat.values, self.fp_data_full.sel(time=self.fp_data_full.time.values[0]).release_lon.values]


    def _get_release_idxs(self):
        idx_release_lat = np.argmin(abs(self.fp_data_full.lat.values - self.release_coords[0]))
        idx_release_lon = np.argmin(abs(self.fp_data_full.lon.values - self.release_coords[1]))
        release_idxs = [idx_release_lat, idx_release_lon]
        return release_idxs

    def _get_domain(self, site):
        #### check domains
        # TODO make domains dict importable
        domains = {"MHD":"EUROPE", "GSN":"EASTASIA"} 
        try:
            domain = domains[site]   
        except: 
            raise ValueError("No domain was passed, and the region you passed is not associated to any domain!")   
        
        return domain   


"""
class LoadSiteData(LoadBaseSiteData):
    def __init__(self, year, site = "MHD", month=None, size=None, domain_to_cut=None, domain=None, verbose = False, fp_datadir = None, lazy_load=True, met_args={}, topog_args={}):  
    
        super().__init__(year=year, month=month, site=site, domain=domain, fp_datadir=fp_datadir, met_args=met_args, topog_args=topog_args, verbose=verbose)

        max_allowed_domain = self._get_max_allowed_domain()

        if size is None and domain_to_cut is None:
            if verbose: print("cutting everything to the largest possible domain")
            self.domain_to_cut = max_allowed_domain

        elif size is not None and domain_to_cut is None:
            if verbose: print("cutting to a square of size size x size centered on the site")
            release_idxs = self._get_release_idxs()
            half = int(size/2)
            domain_lats = self.fp_data_full.lat.values
            domain_lons = self.fp_data_full.lon.values
            self.domain_to_cut = {"lat":[domain_lats[release_idxs[0]-half], domain_lats[release_idxs[0]+half-1]], "lon":[domain_lons[release_idxs[1]-half], domain_lons[release_idxs[1]+half-1]]}

        elif size is None and domain_to_cut is not None:
            assert type(domain_to_cut) is dict, "domain_to_cut should be a dict of format {'lat':[start_lat, end_lat], 'lon':[start_lon, end_lon]}"
            if verbose: print(f"cutting everything to the domain passed {domain_to_cut}")
            self.domain_to_cut = self._check_domain_sizes(domain_to_cut, max_allowed_domain)
        elif size is not None and domain_to_cut is not None:
            if verbose: print("you passed both size and domain to cut! following domain to cut and ignoring size")
            self.domain_to_cut = self._check_domain_sizes(domain_to_cut, max_allowed_domain)

        self._slice_to_domain(self.domain_to_cut)


    def _check_domain_sizes(self, domain_to_cut, max_allowed_domain):
        # check that the passed domain actually fits, correct otherwise

        # fill in lat or lon in case only one was passed
        if not "lat" in domain_to_cut.keys():
            domain_to_cut["lat"] = max_allowed_domain["lat"]
        if not "lon" in domain_to_cut.keys():
            domain_to_cut["lon"] = max_allowed_domain["lon"]
        
        self.domain_to_cut = copy.deepcopy(domain_to_cut)

        self.domain_to_cut["lat"][0] = np.max([domain_to_cut["lat"][0], max_allowed_domain["lat"][0]])
        
        self.domain_to_cut["lat"][-1] = np.min([domain_to_cut["lat"][-1], max_allowed_domain["lat"][-1]])
    
        self.domain_to_cut["lon"][0] = np.max([domain_to_cut["lon"][0], max_allowed_domain["lon"][0]])
        
        self.domain_to_cut["lon"][-1] = np.min([domain_to_cut["lon"][-1], max_allowed_domain["lon"][-1]])

        if self.domain_to_cut != domain_to_cut:
            print(f"the domain you passed is bigger than the domain of the data in at least one direction. cropping to {self.domain_to_cut}")

        return self.domain_to_cut
    
    def _get_max_allowed_domain(self):
        # calculate the max possible domain, given by the footprint and the met files
        print("2")
        lats = [np.max([self.fp_data_full.lat.values[0], self.met_file.lat.values[0]]), np.min([self.fp_data_full.lat.values[-1], self.met_file.lat.values[-1]])]
        lons = [np.max([self.fp_data_full.lon.values[0], self.met_file.lon.values[0]]), np.min([self.fp_data_full.lon.values[-1], self.met_file.lon.values[-1]])]
        max_allowed_domain = {"lat":lats, "lon":lons}
        return max_allowed_domain 

    def _slice_to_domain(self, domain_to_cut):
        self.fp_data_full = self.fp_data_full.sel(lat=slice(domain_to_cut["lat"][0], domain_to_cut["lat"][1]), lon=slice(domain_to_cut["lon"][0], domain_to_cut["lon"][1]))

        if hasattr(self, "met_file"):
            self.met_file = self.met_file.sel(lat=slice(domain_to_cut["lat"][0], domain_to_cut["lat"][1]), lon=slice(domain_to_cut["lon"][0], domain_to_cut["lon"][1]))

        if hasattr(self, "topog_file"):
            self.topog_file = self.topog_file.sel(lat=slice(domain_to_cut["lat"][0], domain_to_cut["lat"][1]), lon=slice(domain_to_cut["lon"][0], domain_to_cut["lon"][1]))

        if hasattr(self, "landcover_file"):
            self.landcover_file = self.landcover_file.sel(lat=slice(domain_to_cut["lat"][0], domain_to_cut["lat"][1]), lon=slice(domain_to_cut["lon"][0], domain_to_cut["lon"][1]))
"""



def get_release_idxs(fp_full, domain_lats=None, domain_lons=None):
    """
    Returns array of shape (time, 2) with the indeces of the measurement point for each footprint, for either the footprint's own grid (do not pass domain_lats and domain_lons) or for another grid defined by domain_lats and domain_lons. Requires fp_full has variables release_lat and release_lon

    this function could be more general if we passed the release indeces directly!
    """
    if domain_lats is None:
        domain_lats=fp_full.lat.values
    if domain_lons is None:
        domain_lons = fp_full.lon.values
    release_idxs = []
    # get release indeces for each footprint
    fp_full.release_lat.load()
    fp_full.release_lon.load()
    for rlat, rlon in zip(fp_full.release_lat.values, fp_full.release_lon.values):
        release_lat, release_lon = min(domain_lats, key=lambda x:abs(x-rlat)), min(domain_lons, key=lambda x:abs(x-rlon))
        idx_release_lat = np.where(domain_lats == release_lat)[0][0]
        idx_release_lon = np.where(domain_lons == release_lon)[0][0]    
        release_idxs.append((idx_release_lat, idx_release_lon))
    release_idxs = np.array(release_idxs)
    return release_idxs


def load_default_brazil_emissions(year=2016, month_to_use=6):
    emissions = xr.open_dataset("/group/chemistry/acrg/LPDM/emissions/SOUTHAMERICA/ch4_SOUTHAMERICA_2016_SWAMPS-v32-5_Saunois-Annual-Mean.nc")
    emissions = emissions.sel(time=emissions.time[month_to_use-1])
    return emissions.flux.values

def load_default_sahara_emissions(year=2016, month_to_use=6):
    emissions = xr.open_dataset(f"/group/chemistry/acrg/LPDM/emissions/NORTHAFRICA/ch4_NORTHAFRICA_{year}.nc")
    emissions = emissions.sel(time=emissions.time[month_to_use-1])
    return emissions.flux.values

def cut_emissions_data(flux, fp_full, size):
    # this assumes flux is a 2D np array of the same resolution and size as the footprints! it also assumes that the data is cut to a square size

    release_idxs = get_release_idxs(fp_full)

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


def cut_satellite_data(fp_full, size, returnlatlons = False,fill_bads_with="nans", delete_outofdomain=False, load=True, return_as="netcdf", verbose=True, fill_latlons=True, return_everything=False):
    """
    cuts footprint to square of size "size" around release point and returns as a netcdf with artificial lat-lon coordinates 0-size

    Inputs:
    fp_full - full xr array with footprints. Should have variables .fp, .release_lat and .release_lon
    size - size of square to cut footprints to
    
    fill_bads_with - str, options are "all_nans", "nans" and "zeros". If "all_nans", any footprint with part of the cutting area outside of the fp_full domain is filled fully with nans. If "nans" or "zeros", only the parts out of the domain are set to "nans" or "zeros" respectively
    
    ## return_as - str, options are "array", "netcdf". If "array" returns as array of shape (time, size*size), if "netcdf" returns as an xarray with coordinates
    ## returnlatlons - bool, if True return the cut footprints, and the lats, lons and release_idxs arrays. If false, return only the cut footprints
    """
    half = int(size/2)
    release_idxs = get_release_idxs(fp_full)    

    padding_needed = False

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
        release_idxs = get_release_idxs(fp_full)  

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
        

    #print("padding", padding_needed, padding)
    if padding_needed:
        # recalculate the release indeces to account for the new padding that was just added
        domain_lats = np.copy(fp_full.lat.values)
        domain_lons = np.copy(fp_full.lon.values)
        release_idxs = get_release_idxs(fp_full)  

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

    # load into memory, if required
    if load:
        print("loading cropped footprint dataset into memory. If you only want to lazy-load, pass load=False")
        cropped_fp.load()    

    if return_as=="netcdf":
        if return_everything:
            return cropped_fp, release_idxs, padding
        else:
            return cropped_fp
    if return_as=="array":
        fp_data = cropped_fp.fp.transpose("time","lat", "lon").values
        fp_data = np.reshape(fp_data, (len(cropped_fp.time), size**2))

        if return_everything:
            fp_lats = cropped_fp.lat_coords.values
            fp_lons = cropped_fp.lon_coords.values  
            return fp_data, fp_lats, fp_lons, release_idxs, padding, fp_full.sel(lat=slice(original_fp_domain[0], original_fp_domain[1]), lon=slice(original_fp_domain[2], original_fp_domain[3]))               
        else:
            return fp_data            
 
def process_domain_met(met, fp, time_delta=0,relevant_levels=None, relevant_variables=None, verbose=True, add_wind_direction=True):

    met = select_met_levels(met, levels=relevant_levels)

    met = select_met_variables(met, variables=relevant_variables)  

    fp_times = np.copy(fp.time.values)

    assert time_delta>=0, "time_delta needs to be zero or positive!!"

    if time_delta==0:
        met = met.interp(time=fp_times)
        met = met.assign({"fp_time":(("time"), fp_times)})
    else:

        fp_times = (pd.DatetimeIndex(fp_times) - pd.Timedelta(f"{time_delta}h"))
        met = met.interp(time=fp_times)

        # store the original footprint times as a separate value
        met = met.assign({"fp_time":(("time"),fp.time.values)})

    

    domain_lats = np.copy(met.lat.values)
    domain_lons = np.copy(met.lon.values)

    delta_lat = domain_lats[1]-domain_lats[0]
    delta_lat_fp = fp.lat.values[1]-fp.lat.values[0]
    delta_lon = domain_lons[1]-domain_lons[0]
    delta_lon_fp = fp.lon.values[1]-fp.lon.values[0]

    if abs(delta_lat - delta_lat_fp) > 0.01 or abs(delta_lon - delta_lon_fp)>0.01:
        print("the resolution is different! this doesnt work yet?")    

    if add_wind_direction and (relevant_variables is None or "wind_speed" in relevant_variables):
        try:
            met["wind_angle"]=np.arctan2(-met.x_wind,-met.y_wind)
            met["wind_speed"]=np.sqrt(met.x_wind**2 + met.y_wind**2)
        except Exception as e:
            print(f"Error {e} happened when adding wind direction and speed to met. Could be a naming error!")
    
    met = met.assign_coords({"time_delta":("time_delta",[time_delta])})

    return met



def cut_satellite_met_v4(met, fp, metsize, time_delta=0, relevant_levels=None, relevant_variables=None, verbose=True, pad_mode="nans", load=False, add_wind_direction=True, save=False, savepath=None, delete_nans=False, attrs_dict=None):
    """
    make into smaller functions!
    
    cuts the meteorology from fixed domain and regular timesteps to match the footprint dataset:
        - in time: interpolated to the time of the footprints if time_delta=0, or to t-time_delta hours otherwise
        - in space: cropped to a square of size metsize x metsize around the coordinates of the satellite measurement for each footprint 

    Main Inputs:
        - met: meteorology array
        - fp: footprint array
        - metsize (int): size of the square to crop the meteorology to
        - time_delta (int, 0 or positive): If time_delta==0, the meteorology will be interpolated to the times of the footprints. Otherwise, the met will be interpolated to t-time_delta, where t is the time of the footprint
        - relevant_levels and relevant_variables: lists of levels (as ints) and variables (as str) to keep. If None, all levels/variables are kept respectively
    Other Inputs:
        - pad_mode: if the area to be extracted (of size metsize x metsize) escapes the domain of the met file, the met file is extended. if pad_mode="nans", it's extended with nans (and can be deleted later), if pad_mode="edge", it's extended using the edges of the domain
        - load (bool): load array into memory
        - add_wind_direction (bool): if True calculate wind_angle and wind_speed from the two horizontal wind vectors and add as variables
        - delete_nans: bool, if True delete timestamps where there were nans
        - attrs_dict: dictionary of attributes to add to the file before saving/returning
        - save (bool): save to file. requires a savepath to be passed
        - savepath (str): full path to save file to

    Returns:
        - met_cut: xarray with cropped and interpolated meteorology (ie interpolated to the correct times, and cropped to a square of size metsize around the footprint release point)
    """

    # subset the right levels and variables, as specified in the inputs
    met = select_met_levels(met, levels=relevant_levels)

    met = select_met_variables(met, variables=relevant_variables)       

    #if verbose: print("loading data")
    
    half = int(metsize/2)

    fp_times = np.copy(fp.time.values)

    # interpolate the meteorology to the correct timestamps
    ###
    # TO DO - add here capability to interpolate to every X minutes, then interpolate timestamps with mode="nearest"
    ###
    assert time_delta>=0, "time_delta needs to be zero or positive!!"


    if time_delta==0:
        met = met.interp(time=fp_times)
        met = met.assign({"fp_time":(("time"), fp_times)})
    else:

        fp_times = (pd.DatetimeIndex(fp_times) - pd.Timedelta(f"{time_delta}h"))
        met = met.interp(time=fp_times)

        # store the original footprint times as a separate value
        met = met.assign({"fp_time":(("time"),fp.time.values)})

    met = met.assign_coords({"time_delta":("time_delta",[time_delta])})


    domain_lats = np.copy(met.lat.values)
    domain_lons = np.copy(met.lon.values)

    delta_lat = domain_lats[1]-domain_lats[0]
    delta_lat_fp = fp.lat.values[1]-fp.lat.values[0]
    delta_lon = domain_lons[1]-domain_lons[0]
    delta_lon_fp = fp.lon.values[1]-fp.lon.values[0]

    if abs(delta_lat - delta_lat_fp) > 0.01 or abs(delta_lon - delta_lon_fp)>0.01:
        print("the resolution is different! this doesnt work yet?")
    
    #met = self.met.interp({"lat":self.domain_lats, "lon":self.domain_lons})

    met_release_idxs = get_release_idxs(fp, domain_lats = domain_lats, domain_lons = domain_lons)
    padding = {"lat":[0,0], "lon":[0,0]}

    met_needed_padding_direction = {"N":0, "S":0, "E":0, "W":0}
    # check if we need to pad in any direction
    met_needed_padding_direction["S"] = np.sum(met_release_idxs[:,0] < half)
    met_needed_padding_direction["N"] = np.sum((len(domain_lats) - met_release_idxs[:,0]) < half)
    met_needed_padding_direction["E"] = np.sum(met_release_idxs[:,1] < half)
    met_needed_padding_direction["W"] = np.sum((len(domain_lons) - met_release_idxs[:,1]) < half)

    #if padding is np.any(np.array(met_needed_padding_direction.values))

    padding_needed = False

    
    if met_needed_padding_direction["S"]>0  or met_needed_padding_direction["N"]>0:
        delta_lat = domain_lats[1] - domain_lats[0]

        to_pad = (np.max([0, half - np.min(met_release_idxs[:,0])]) , np.max([0, half - (len(domain_lats) - np.max(met_release_idxs[:,0]))]))
        padding["lat"] = to_pad

        print(f"careful! the meteorology is smaller than the domain you are trying to extract along the latitude dimension.  We need to pad {to_pad[0]} and {to_pad[1]} idxs on either side! padding with {pad_mode}")

        if pad_mode == "nans":
            met = met.pad(pad_width={"lat":to_pad})
        
        if pad_mode == "edge":
            met = met.pad(pad_width={"lat":to_pad}, mode="edge")

        padding_needed = True

        # reassign coordinates to ensure that padded values have the right spacing
        updated_lats = sorted([np.min(domain_lats)-(i+1)*delta_lat for i in range(to_pad[0])]) + list(domain_lats) + sorted([np.max(domain_lats)+(i+1)*delta_lat for i in range(to_pad[1])])
        met = met.assign_coords({"lat":updated_lats})
    
   

    if met_needed_padding_direction["E"]>0  or met_needed_padding_direction["W"]>0:
        delta_lon = domain_lats[1] - domain_lats[0]

        to_pad = (np.max([0, half - np.min(met_release_idxs[:,1])]) , np.max([0, half - (len(domain_lons) - np.max(met_release_idxs[:,1]))]))


        print(f"careful! the meteorology is smaller than the domain you are trying to extract along the longitude dimension. We need to pad {to_pad[0]} and {to_pad[1]} idxs on either side! padding with {pad_mode}")


        if pad_mode == "nans":
            met = met.pad(pad_width={"lon":to_pad})
        
        if pad_mode == "edge":
            met = met.pad(pad_width={"lon":to_pad}, mode="edge")

        padding_needed = True
        # reassign coordinates to ensure that padded values have the right spacing
        updated_lons = sorted([np.min(domain_lons)-(i+1)*delta_lon for i in range(to_pad[0])]) + list(domain_lons) + sorted([np.max(domain_lons)+(i+1)*delta_lon for i in range(to_pad[1])])
        met = met.assign_coords({"lon":updated_lons})


    if padding_needed:
        # recalculate the release indeces to account for the new padding that was just added
        domain_lats = np.copy(met.lat.values)
        domain_lons = np.copy(met.lon.values)
        met_release_idxs = get_release_idxs(fp, domain_lats = domain_lats, domain_lons = domain_lons)

    cropped_met_arrays = []

    coords_array = np.arange(metsize)
    # crop the data as a small array for each unique release index
    for rel_unique in np.unique(met_release_idxs, axis=0):
        # find the corresponding timestamps
        idxs = np.where((met_release_idxs == rel_unique).all(axis=1))[0]
        # crop the meteorology around the releasepoint

        #cutmet = met.sel(time=fp_times[idxs], lat=domain_lats[rel_unique[0]-half:rel_unique[0]+half], lon=domain_lons[rel_unique[1]-half:rel_unique[1]+half])

        cutmet = met.sel(time=fp_times[idxs])
        cutmet = cutmet.interp({"lat":domain_lats[rel_unique[0]-half:rel_unique[0]+half], "lon":domain_lons[rel_unique[1]-half:rel_unique[1]+half]}, method="nearest")
        # copy the latitude/longitude values for this specific cropped square
        #lats = cutmet.lat.values.copy()
        #lons = cutmet.lon.values.copy()
        
        # replace the latitude/longitude coordinates with grid-like coords (0-metsize), and save the actual coordinates as variables
        #.rename({"latitude":"lat","longitude":"lon"})
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            cutmet = cutmet.assign_coords({"lat":coords_array, "lon":coords_array}).assign({"lat_coords":(("lat"), domain_lats[rel_unique[0]-half:rel_unique[0]+half]), "lon_coords":(("lon"), domain_lons[rel_unique[1]-half:rel_unique[1]+half])})

        cropped_met_arrays.append(cutmet)
    # concatenate all of the cropped arrays
    cropped_met = xr.concat(cropped_met_arrays, dim="time")
    cropped_met = cropped_met.sortby("time")
    # add any passed attributes
    if attrs_dict is not None:
        cropped_met.attrs = attrs_dict.update({"original_met_attrs":cropped_met.attrs})
    else:
        cropped_met.attrs = {"original_met_attrs":cropped_met.attrs}
    
    # load into memory, if required
    if load:
        print("loading cropped met dataset into memory. If you only want to lazy-load, pass load=False")
        cropped_met.load()

    # add additional wind variables
    if add_wind_direction:# and (relevant_variables is None or "wind_speed" in relevant_variables or "wind_angle" in relevant_variables):
        try:
            if relevant_variables is None or "wind_angle" in relevant_variables:
                cropped_met["wind_angle"]=np.arctan2(-cropped_met.x_wind,-cropped_met.y_wind)
            if relevant_variables is None or "wind_speed" in relevant_variables:
                cropped_met["wind_speed"]=np.sqrt(cropped_met.x_wind**2 + cropped_met.y_wind**2)
            print("calculated wind angle and/or speed from x_wind and y_wind")
        except Exception as e:
            print(f"Error {e} happened when adding wind direction and speed to met. Could be a naming error!")

    # this needs implementing
    # need to delete nans 
    #   1) in time (e.g. when t-jump is outside of the meteorology file )
    #   2) in space (if pad_mode="nans", identify and delete the whole timepoint? this could also be done later in the LoadData object
            
    if delete_nans:
        raise NotImplementedError
        """
        nan_idxs = np.unique(np.where(np.isnan(met_cut.x_wind.values[0,0,0,:])))
        met_cut = met_cut.sel(time=np.delete(met_cut.time.values, nan_idxs))
        print(f"removed {len(nan_idxs)} invalid indeces")
        """
    if save:
        assert savepath is not None, "pass a savepath to save the file to!"
        assert savepath[-3:] == ".nc", "ensure you passed a full savepath, including filename and .nc"
        print("saving met at", savepath)
        cropped_met.to_netcdf(savepath)
        print("met saved")

    # cropped met will have some nans!!!
    return cropped_met



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

    """
    assert hasattr(data, "dataset_format"), "It doesn't seem this is a SatelliteData object"
    #assert data.dataset_format == "square", "At the moment this only works for LoadSquareSatelliteData objects"
    assert data.met_processed == True, "Make sure that you have loaded and cut the meteorology in the SatelliteData object"

    assert type(met_variables) is dict, "met_variables should be a dict of shape {'variable_name':levels_to_extract, 'surface_variable':[], ...}. For each atmospheric variable with levels, pass the levels to extract as a list. For each surface variable, pass an empty list"

    if verbose: print("---Preparing met")

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
                    print(met_variables_needed)
                    met = cut_satellite_met_v4(data.met_file, data.fp_data_full, metsize=data.metsize, time_delta=delta, relevant_levels = min_levels_needed, relevant_variables = met_variables_needed, pad_mode=data.fill_outofdomain_with, load=False, add_wind_direction=True)
                    print("met", met)
                if data.dataset_format == "domain":
                    met = process_domain_met(data.met_file, data.fp_data_full,time_delta=delta, relevant_levels = min_levels_needed, relevant_variables = met_variables_needed, add_wind_direction=True)

                met = met.swap_dims({"time":"fp_time"})
                #met = met.reset_coords(["time"])
                met = met.drop_vars("time")
                

                
            all_met_files[delta] = met.copy()

            del met 

    
    # concatenate all met datasets, which should have the same coordinates except the time_delta dimension
    full_met = xr.concat(list(all_met_files.values()), dim="time_delta", data_vars =met_variables_needed).transpose("fp_time", "lat", "lon", ..., "time_delta")

    # if the time_delta is large, cut met might have interpolated to t-time_delta outside of the known met. check and if so remove indeces
    if data.met.time.values[0] - pd.Timedelta(f"{max(time_deltas)}h") < data.met_file.time.values[0]:
        badly_interpolated = data.met.time.values - pd.Timedelta(f"{max(time_deltas)}h") < data.met_file.time.values[0]
        full_met = full_met.drop_sel(fp_time=full_met.fp_time.values[badly_interpolated])

        # this updates any indeces that couldnt be interpolated
        # i think it updates data without needing to return it as a new object
        data.remove_indeces(np.where(badly_interpolated)[0])


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

        input_arrays.append(stacked_static_inputs)

    else:
        stacked_static_inputs=None

    concatenated_inputs = xr.concat(input_arrays, dim="variable_name")
    idx = pd.MultiIndex.from_tuples(concatenated_inputs.variable_name.values, names=["variable", "levels", "time_delta"])
    concatenated_inputs = concatenated_inputs.assign_coords(variable_name=pd.MultiIndex.from_tuples(concatenated_inputs.variable_name.values, names=["variable", "levels", "time_delta"]))

    #latlons, idx_latlons = get_grid(data, latlon_fp)

    if not return_asarray:
        if return_variable_names:
            return concatenated_inputs, varnames_dict
        else:
            return concatenated_inputs
        
    if return_asarray:
        if return_variable_names:
            return np.reshape(concatenated_inputs.values, (concatenated_inputs.fp_time.size, concatenated_inputs.lat.size*concatenated_inputs.lon.size, concatenated_inputs.variable_name.size)), varnames_dict
        else:
            return np.reshape(concatenated_inputs.values, (concatenated_inputs.fp_time.size, concatenated_inputs.lat.size*concatenated_inputs.lon.size, concatenated_inputs.variable_name.size))



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





