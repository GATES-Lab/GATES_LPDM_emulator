import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os
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
            # attempt to load dataset of multiple files the standard wayf
            fp_data_full = xr.open_mfdataset(sorted(glob.glob(fp_datadir)), combine='by_coords', chunks = {"time":time_chunk})
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

        #### check domains
        self.region = region
        if domain is None:
            self.domain = self._get_domain(region)

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
        

        if load_everything:
            self.met_file = self.load_meteorology(**met_args)
            self.padding=None
            self.topog_file, self.landcover_file = self.load_topog(**topog_args)

        
        if verbose: print("---- All done!")



    def load_meteorology(self, met_datadir=None, met_levels = [], met_variables= []):
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
        topog_file = xr.load_dataset(topog_path)

        if landcover_path=="default":
            landcover_path = "/group/chemistry/acrg/LPDM/topog_NAME/land_cover.nc"
        landcover_file = xr.load_dataset(landcover_path)
            

        topog_file = self._interp_topog(topog_file, padding=self.padding)

        landcover_file = self._interp_landcover(landcover_file, padding=self.padding)

        return topog_file, landcover_file

    def _get_meteorology_file(self, met_datadir):
        if met_datadir==None:
            met_datadir = "/group/chemistry/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if self.verbose: print("Loading meteorology from " + met_datadir)

        # each chunk should have around 1mill values,  - chunk per level and by time, rounded to the nearest hundred, 100MB-1GB
        # could calcualte this dynamically 
        time_chunk = 500 #round(1000000/(self.metsize*self.metsize), -2) #
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.met_file = xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords', data_vars="minimal", coords="minimal", parallel=True, join="inner", chunks = {"level":1, "time":time_chunk})

        #) rename, select levels and variables
        if "model_level_number" in self.met_file.dims:
            self.met_file = self.met_file.rename({"model_level_number": "levels", "latitude":"lat", "longitude":"lon"})

        self.met_file = self.met_file.drop_duplicates(dim=["lat", "lon", "time"])

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
            #fp_datadir=fp_datadir+str(self.date)+"*.nc"
            fp_datadir = f"{fp_datadir}{self.domain}/*{self.region}*{self.domain}_{str(self.date)}*.nc"
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
            print("no sampling was done because you didnt pass a valid sampling mode, or freq=1")

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
            self.met_file = self.load_meteorology(**met_args)
            self.met = self._process_meteorology(lazy_load=lazy_load)
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
        self.fp_data, self.fp_lats, self.fp_lons, self.release_idxs, self.padding, self.fp_data_full = cut_satellite_data_v2(self.fp_data_full, self.size, returnlatlons = True, return_as="array",fill_bads_with=self.fill_outofdomain_with, delete_outofdomain = self.delete_outofdomain, verbose=self.verbose, return_everything=True, load=not lazy_load) 

        # if self.fill_outofdomain_with == "all_nans":
            # remove here all indeces from self.idxs_out_of_domain! so we only keep the footprints that were fully inside the domain

    def _process_meteorology(self,rechunk=0,lazy_load=True):
        if self.verbose: print("----- Cutting met")
        self.metsize=self.size
        if self.fill_outofdomain_with=="nans" or self.delete_outofdomain:
            pad_mode = "nans"
        if self.fill_outofdomain_with=="zeros":
            pad_mode = "edge"
        self.met = cut_satellite_met_v4(self.met_file, self.fp_data_full, metsize=self.size, time_delta=0, pad_mode=pad_mode, load=not lazy_load)

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
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, freq=1, freq_offset=0, verbose = False, sampling_mode="regular", fp_datadir = None, load_everything=False, met_args={}, topog_args={}):
        
        ## INITIALISE THE BASE OBJECT, TO LOAD THE FOOTPRINTS
        super().__init__(self, year, region, month=month, domain=domain, freq=freq, freq_offset=freq_offset, verbose=verbose, sampling_mode=sampling_mode, fp_datadir=fp_datadir, load_everything=False, met_args=met_args, topog_args=topog_args)

        self.dataset_format = "domain"

    def _process_footprints(self):
        raise NotImplementedError

    def _process_met(self):
        raise NotImplementedError

    def _process_topog(self):
        raise NotImplementedError
    


class LoadSatelliteDataNew:
    """
    Load footprint and meteorological data for a particular domain and time period, outputting all data cut to a square centered around the measurement point for each timestamp. 

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
            - fill_outofdomain_with: str, out of "all_nans", "nans" and "zeros". Determines what to do if any part of the square cut around the footprint is outside of the domain. "all_nans" fills that whole footprint with nans, "nans" and "zeros" fill only the out of domain areas with nans and zeros respectively. If "all_nans" or "nans", that footprint will be eliminated from the dataset
        - verbose: if True, prints out the steps throughout the data loading process
    
    met_args:
        see load_meteorology()
    """
    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, verbose = False, fill_outofdomain_with="nans", check_for_nans=False, sampling_mode="regular", fp_datadir = None, load_everything=False, lazy_load=True, met_args={}):
        


        #### check domains
        self.region = region
        if domain is None:
            self.domain = self._get_domain(region)

        self.size = size

        self.fill_outofdomain_with = fill_outofdomain_with

        self.year = year
        self.date = self.year
        self.verbose=verbose


        if month != None:
            self.month = month
            self.date = str(self.year)+month

        self.subsample_parameters = {"freq":freq, "sampling_mode":sampling_mode, "freq_offset":freq_offset}
        
        #### load footprint (fp) data     
        if verbose: print("---- PREPARING FOOTPRINTS")  
        self._load_footprints(fp_datadir)
        self._process_footprints(lazy_load)

        self.met_args = met_args
        self.met_processed = False

        if load_everything:
            self.met = self.load_meteorology(**met_args, lazy_load=lazy_load)
            self.load_topog()


        if check_for_nans:
            self._remove_fp_nans()
            self._remove_met_nans()
        
        if verbose: print("---- All done!")



    def load_meteorology(self, met_datadir=None, met_levels = [], met_variables= [], metsize=None, lazy_load=True, rechunk=0):
        """
        load the meteorology from the directory, select the met levels and variables if required, interpolates and crop to size.

        Inputs
            - met_datadir (str): directory for meteorology. default directs to ACRG meteorology folder. If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/name_of_your_choice_".- the function will reshape it as long as size of loaded meteorology >= metsize
            - met_levels (list): met levels to select
            - met_variables (list) : met variables to select
            - metsize: side of square of size metsize x metsize to crop meteorology to. defaults to the same as the footprints (so that metsize=size)
            - rechunk: number of chunks to rechunk met to. might help with memory?
            - lazy_load: if true, do a shallow load (does not load the array into memory)

        """
        assert self.met_processed is False, "It seems like this met stored in this object (as self.met) has already been cropped to a square of size metsize x metsize!"

        if self.verbose: print("\n ---- PREPARING MET")
        if metsize is None or metsize == self.size:
            self.metsize = self.size
        if self.metsize != self.size:
            print("the met loaded will be cropped to a different size than the footprints! make sure this is what you want")
            self.metsize=metsize

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

        if self.verbose: print("Cutting Met")
        self.met = cut_satellite_met_v4(self.met_file, self.fp_data_full, metsize=self.metsize, time_delta=0, pad_mode=self.fill_outofdomain_with, load=not lazy_load)

        if rechunk>0:
            self.met.chunk({"time":rechunk})
        
        self.met_processed = True
        return self.met

    def load_topog(self, topog_path="default", landcover_path="default"):
        """
        load the topgoraphy and landcover files, and crop to the same shape as the footprints
        args:
         - topog_path and landcover_path: str paths to each file, or "default" for default file
        
        uses objet attributes: self.padding (contains if any amount of padding is needed to the footprint domain, and in which direction)
        """
        #### load and cut topography
        print("---- PREPARING TOPOG")
        if topog_path=="default":
            topog_path="/group/chemistry/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc"
        print(f"trying to load topography from {topog_path}")
        topog_file = xr.load_dataset(topog_path)

        if landcover_path=="default":
            landcover_path = "/group/chemistry/acrg/LPDM/topog_NAME/land_cover.nc"
        landcover_file = xr.load_dataset(landcover_path)
            

        self.topog_file, topog_lats, topog_lons = self._interp_topog(topog_file, padding=self.padding, return_latlons=True)

        self.landcover_file = self._interp_landcover(landcover_file, padding=self.padding)


        #self.topog, self.land_cover, self.disaggregated_land_cover, failed_idxs = self._crop_topog_and_landcover(topog_lats=topog_lats, topog_lons=topog_lons )
        self.topog, failed_idxs = self._crop_topog_and_landcover(topog_lats=topog_lats, topog_lons=topog_lons)

        if len(failed_idxs)>0:
            if self.verbose: print(f"remove {len(failed_idxs)} failed idxs for topog")
            self.remove_indeces(failed_idxs)

        print("all topog loaded!")


    def _get_meteorology_file(self, met_datadir):
        if met_datadir==None:
            met_datadir = "/group/chemistry/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if self.verbose: print("Loading meteorology from " + met_datadir)

        # each chunk should have around 1mill values,  - chunk per level and by time, rounded to the nearest hundred, 100MB-1GB
        # could calcualte this dynamically 
        time_chunk = 500 #round(1000000/(self.metsize*self.metsize), -2) #
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.met_file = xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords', data_vars="minimal", coords="minimal", parallel=True, join="inner", chunks = {"level":1, "time":time_chunk})

        #) rename, select levels and variables
        if "model_level_number" in self.met_file.dims:
            self.met_file = self.met_file.rename({"model_level_number": "levels", "latitude":"lat", "longitude":"lon"})

        self.met_file = self.met_file.drop_duplicates(dim=["lat", "lon", "time"])

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
            #fp_datadir=fp_datadir+str(self.date)+"*.nc"
            fp_datadir = f"{fp_datadir}{self.domain}/*{self.region}*{self.domain}_{str(self.date)}*.nc"
        if self.verbose: print("Loading footprint data from " + fp_datadir) 

        self.fp_data_full = load_fps(fp_datadir, verbose=self.verbose)  

        self.fp_data_full = self.fp_data_full.drop_duplicates(dim="time")

        ## reduce data frequency with regular sampling 9eg keep only 1 in every 3 timesteps
        # uses the sampling_mode and freq parameters
        self._subsample_frequency(**self.subsample_parameters)

        if self.verbose: print(f"Loading {len(self.fp_data_full.time.values)} footprints")

        #self.fp_data_full.load()
    
    def _process_footprints(self, lazy_load):
        # cut data around release point
        # fp data returned is array of shape (time, size*size) with each footprint centered around its release point
        if self.verbose: print("Cutting footprints to size") 
        self.fp_data, self.fp_lats, self.fp_lons, self.release_idxs, self.padding, _ = cut_satellite_data_v2(self.fp_data_full, self.size, returnlatlons = True, return_as="array",fill_bads_with=self.fill_outofdomain_with, verbose=self.verbose, return_everything=True, load=not lazy_load)

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
            print("no sampling was done because you didnt pass a valid sampling mode, or freq=1")

    def _interp_topog(self, topog_file, padding=None, return_latlons=False):
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


        topog_file = topog_file.interp(latitude=lat_values, longitude=lon_values)

        if return_latlons:
            return topog_file, lat_values, lon_values
        else:
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
    
    def _crop_topog_and_landcover(self, topog_lats=None, topog_lons=None):
        """
        cut topography to the same domain covered by the cut footprints (ie a sizexsize square centered around measurement point)
        inputs are the latitudes and longitudes that the topog was interpolated to (this is clunky)

        uses object attributes self.fp_data_full, self.size, self.fp_data
        """
        if topog_lats is None:
            topog_lats = list(self.fp_data_full.lat.values)
        if topog_lons is None:
            topog_lons = list(self.fp_data_full.lon.values)

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
                full_topog[idxs, :,:] = np.zeros_like(full_topog[idxs, :,:])
                full_landcover[idxs, :,:] = np.zeros_like(full_landcover[idxs, :,:])
                failed_idxs = failed_idxs + list(idxs)
            
        
        # reverse sea mask so that 1 is sea and zero is land
        disaggregated_landcover[:, :,:,0] = 1 - disaggregated_landcover[:, :,:,0] 
        # remove the nans that in the original file show the sea mask
        disaggregated_landcover[:, :,:,1:] = np.nan_to_num(disaggregated_landcover[:, :,:,1:], copy=False)


        #return full_topog, full_landcover, disaggregated_landcover, failed_idxs
    
        coords = {"time":("time", self.fp_data_full.time.values), "lat":("lat", np.arange(self.size)), "lon":np.arange(self.size), "landcover_level":np.arange(n_disagg_landcover_types)}

        topog_ds = xr.Dataset(
            data_vars = 
            {"topog": (["time", "lat", "lon"], full_topog),
            "landcover": (["time", "lat", "lon"], full_landcover),
             "disaggregated_landcover": (["time", "lat", "lon", "landcover_level"], disaggregated_landcover) },
             coords = coords
             )

        # returning as an array instead
        return topog_ds, failed_idxs


    def remove_indeces(self, nan_idxs):
        """
        removes any set of indeces passed as nan_idxs from all the objects in the dataset
        """
        self.fp_data_full = self.fp_data_full.sel(time=np.delete(self.fp_data_full.time.values, nan_idxs))
        self.fp_lats = np.delete(self.fp_lats, nan_idxs, axis=0)
        self.fp_lons = np.delete(self.fp_lons, nan_idxs, axis=0)
        self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
        if self.verbose: print(f"current length: {len(self.release_idxs)}")
        self.release_idxs = np.delete(self.release_idxs, nan_idxs, axis=0)
        if self.verbose: print(f"Length after removing indeces: {len(self.release_idxs)}")
        if hasattr(self, "met"):
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))
        if hasattr(self, "topog"):
            self.topog = self.topog.sel(time=np.delete(self.topog.time.values, nan_idxs))
        """
        if hasattr(self, "topog"):
            self.topog = np.delete(self.topog, nan_idxs, axis=0)
        if hasattr(self, "land_cover"):
            self.land_cover = np.delete(self.land_cover, nan_idxs, axis=0)
        """

    def _check_for_nans(self):
        raise NotImplementedError

    def _remove_fp_nans(self):
        ## check if any of the fp entries are nans, and if so remove from met and others
        if np.sum(np.isnan(self.fp_data_full.fp.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.fp_data_full.fp.values))[2])
            print(f"There are {len(nan_idxs)} nans in the fp data. finding and deleting those timestamps from object attributes")
            self.remove_indeces(nan_idxs)
            self.fp_nan_idxs = nan_idxs
        else:
            self.fp_nan_idxs = []
    
    def _remove_met_nans(self):
        # check if any of the met entries are nans, and if so remove from all objects
        if np.sum(np.isnan(self.met.x_wind.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.met.x_wind.values[0,0,0,:])))
            print(f"There are {len(nan_idxs)} nans in the met data. finding and deleting those timestamps from object attributes")
            
            self.remove_indeces(nan_idxs)
            self.met_nan_idxs = nan_idxs
        else:
            self.met_nan_idxs=[]
    



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


def cut_satellite_data_v2(fp_full, size, returnlatlons = False,fill_bads_with="nans", delete_outofdomain=False, load=True, return_as="netcdf", verbose=True, fill_latlons=True, return_everything=False):
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
        if verbose: print(f"for {len(padded_fps_idxs)} footprints, a square of size x size escapes the footprint domain in directions {fp_needed_padding_direction}. dropping these! if you want to keep them anyway, pass delete_outofdomain=False")
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
        print("loading array into memory. If you only want to lazy-load, pass load=False")
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
        print("loading array into memory. If you only want to lazy-load, pass load=False")
        cropped_met.load()

    # add additional wind variables
    if add_wind_direction and (relevant_variables is None or "wind_speed" in relevant_variables):
        try:
            cropped_met["wind_angle"]=np.arctan2(-cropped_met.x_wind,-cropped_met.y_wind)
            cropped_met["wind_speed"]=np.sqrt(cropped_met.x_wind**2 + cropped_met.y_wind**2)
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



def get_all_inputs_graphnet_satellite_v7(data, met_variables, time_deltas=[], static_variables=[], verbose=True, return_variable_names=False, return_asarray=False):
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
    assert data.dataset_format == "square", "At the moment this only works for LoadSquareSatelliteData objects"
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

    if len(time_deltas)>1:
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
                met = cut_satellite_met_v4(data.met_file, data.fp_data_full, metsize=data.metsize, time_delta=delta, relevant_levels = min_levels_needed, relevant_variables = met_variables_needed, pad_mode=data.fill_outofdomain_with, load=False)
                met = met.swap_dims({"time":"fp_time"})
                #met = met.reset_coords(["time"])
                met = met.drop_vars("time")
                

                
            all_met_files[delta] = met.copy()

            del met 
    
    # concatenate all met datasets, which should have the same coordinates except the time_delta dimension
    full_met = xr.concat(list(all_met_files.values()), dim="time_delta", data_vars =met_variables_needed).transpose("fp_time", "lat", "lon", "levels", "time_delta")

    # if the time_delta is large, cut met might have interpolated to t-time_delta outside of the known met. check and if so remove indeces
    if data.met.time.values[0] - pd.Timedelta(f"{max(time_deltas)}h") < data.met_file.time.values[0]:
        badly_interpolated = data.met.time.values - pd.Timedelta(f"{max(time_deltas)}h") < data.met_file.time.values[0]
        print(badly_interpolated)
        full_met = full_met.drop_sel(fp_time=full_met.fp_time.values[badly_interpolated])


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
        (static_ds, ) = xr.broadcast(full_met[["lat_coords", "lon_coords"]])

        for var in static_variables:
            # add each input as a variable in the static_ds
            print(var)
            if var in ["topog", "landcover", "landcover_disaggregated"]:
                assert hasattr(data, "topog"), "Load topog on the data object with data.load_topog() before trying to extract this as an input!"
                static_ds = static_variables_functions[var](data.topog, static_ds)
            elif var in list(static_variables_functions.keys()) and var not in ["lat_coords", "lon_coords"]:
                static_ds = static_variables_functions[var](static_ds)
            else:
                print(f"variable {var} was not found in the list of known functions!")

        if "lat_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lat_coords"])
        if "lon_coords" not in static_variables:
            static_ds = static_ds.drop_vars(["lon_coords"])

        # add empty variable levels and time_delta so it aligns with the met dataset
        print("new format")
        stacked_static_inputs = static_ds.assign_coords(levels=0, time_delta=0).expand_dims("levels").expand_dims("time_delta").transpose("fp_time", "lat", "lon", "levels", "time_delta").to_stacked_array(new_dim="variable_name", sample_dims=["fp_time", "lat", "lon"], name="stacked_static_inputs")  

        
        varnames_dict = varnames_dict + [{"var":tup[0], "type":"static"} for tup in stacked_static_inputs.variable_name.values]
        stacked_static_inputs = stacked_static_inputs.drop_vars({'variable_name', 'variable'}).assign_coords({"variable_name":stacked_static_inputs.variable_name.values})

        input_arrays.append(stacked_static_inputs)

    else:
        stacked_static_inputs=None

    print("concatenate!")
    concatenated_inputs = xr.concat(input_arrays, dim="variable_name")
    idx = pd.MultiIndex.from_tuples(concatenated_inputs.variable_name.values, names=["variable", "levels", "time_delta"])
    concatenated_inputs = concatenated_inputs.assign_coords(variable_name=pd.MultiIndex.from_tuples(concatenated_inputs.variable_name.values, names=["variable", "levels", "time_delta"]))

    if not return_asarray:
        if return_variable_names:
            return concatenated_inputs, varnames_dict
        else:
            return concatenated_inputs
        
    if return_asarray:
        if return_variable_names:
            return concatenated_inputs.values, varnames_dict
        else:
            return concatenated_inputs.values




def get_all_inputs_graphnet_satellite_v6(data, variables_past, jumps, variables_nopast, others=[], jumps_reduced=[], variables_reduced=[], topog=True, latlon_fp=0, transform=True, add_current_time=True, centered_coords=False, return_idx=False, relative_met=False):
    """
    LATEST VERSION - get inputs from LoadSatelliteData object and format as array of size (time, variables)

    Inputs:
        - data - LoadSatelliteData object
        - variables_past - dictionary of variables to load for all times in jumps with format {"varname":levels to load} eg {{"x_wind":[3,9]} 
        - jumps - list of ints. Met for vars in variables_past will be used as inputs for t-jump for all jumps, where t is the timestamp of the footprint
        - add_current_time - wether to add the meteorology at the timestamps of the footprint (just adds 0 to jumps if it isnt there already)
        - variables_nopast - dictionary of variables to load only at t=0 with format {"varname":levels to load} eg {{"x_wind":[3,9]} 
        - others - list of other non-meterological variables to load. see below for names and explanation
        - topog - bool, if True add topography at each lat/lon as a feature
        - latlon_fp - int, index of footprint to use as reference to create the latlon grid. default is 0, first footprint in the dataset
        - transform - bool, if False return data with shape (size,size, time, features), if True return with shape (time, size*size, features) 
        - centered_coords - bool, only used if x_coords and/or y_coords are in others. If True, x/y coords have the release point as (0,0), if false the south-west corner is (0,0) and the release point is (size/2, size/2) 
        - return_idx - bool, if True return node indeces

        - relative_met: if false, return met at t-xh for each x in jumps, if true return met relative to t0 (ie t0, tx - t0 ...)

    Returns:
        - latlons - latlon grid to pass to model
        - idx_latlons - node indeces - only returned if return_idx=True
        - all_vars - features, shape depends on transform
        - var_names - list containing information about each of the features, of length len(features). Each entry is a dictionary with attributes "var" (name), "level", "type" (3D if it's a variable with levels, 2D otherwise, "not met" if it's topography or others) and "time" ("t-0" indicates the meteorology is for the time of the footprint, "t-6" six hours before and so on)
        - data - the LoadSatelliteData object, with any updates to the datasets if there are nans in the past data 

    Accepted values in others:
        - lat_coords/lon_coords: lat/lon coordinate for each node
        - sin_lat_coords/sin_lon_coords/cos_lat_coords/cos_lon_coords: useful mostly if working with a big/whole world domain to encode the sphere
        - x_coords/y_coords: x/y index of each node (see centered_coords above)
        - binary_centre: zero for each node except 1 for release node
        - distance_centre: euclidean distance to the release node (using x/y coords, not lat/lon coords for speed of calculation)
        - normalised_time_of_day and normalised_time_of_year: sin and cos of the time of the normalised time of the day and year. Meant to encode cyclical patterns. For some reason it really messes up with training, do not use!
        - relative_time - ignore too
    """

    all_vars = []
    var_names = []
    og_jumps = jumps.copy()

    jumps = og_jumps+jumps_reduced

    if relative_met:
        assert (0 in jumps) or add_current_time, "jump 0 is needed to do relative met!"

    if not (0 in jumps) and add_current_time:
        jumps.append(0) # append 0 to get present met too
    jumps=list(sorted(set(jumps)))
    print(f"hours back in time: {jumps}")

    # mets contains all of the met objects, the key is the jump
    mets={}
    """
    Met files should already be 
        1) cut in space, so that it has size NxX and centered around measurement point
        2) interpolated in time, to align with t (time of the measurements) or t-H

    data.met contains the meteorological data at the time of the measurement. the rest of the times are loaded below. 
    the met data is stored in a dictionary with format mets[H] = data array for that H (so that mets = {0: met at time=t, 6: met at time=t-6 etc})
    """
    were_there_nans=False
    """
    
    print("y_wind" in data.met.variables)
    if "y_wind" in data.met.variables:
        print("checking for nans")
        if np.any(np.isnan(data.met.y_wind.values)):
            data.met["y_wind"] = data.met.y_wind.interpolate_na(dim="lat", max_gap = 5)
            data.met["y_wind"] = data.met.y_wind.interpolate_na(dim="lon", max_gap = 5)
            #data = data.y_wind.interpolate_na(dim="lon", max_gap = 5)
            were_there_nans = True
    """
    
    mets[0] = data.met
    valid_timestamps = np.copy(data.met.time.values)
    for j in jumps:
        if j!=0:
            met_files = glob.glob(f"{data.met_datadir}{j}h_{data.date}*")
            if len(met_files)==0:
                print(f"couldnt find files for jump {j} at {data.met_datadir}{j}h_")
            else:
                time_chunk = round(1000000/(data.size*data.size), -2)
                with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                    #mets[j] = xr.open_mfdataset(met_files, parallel=True, chunks = {"level":1,"time":time_chunk})
                    mets[j] = xr.open_mfdataset(sorted(met_files), combine='by_coords', data_vars="minimal", compat="override", coords="minimal", parallel=True, join="inner", chunks = {"level":1, "time":time_chunk})

                if len(mets[j].lat) > data.metsize:
                    print(f"file for jump {j}h is bigger than metsize, cutting down")
                    diff = int((len(mets[j].lat) - data.metsize)/2)
                    cut_idxs =  mets[j].lat.values[diff:-diff]
                    mets[j] = mets[j].sel(lat=cut_idxs, lon=cut_idxs)
                    mets[j] = mets[j].assign_coords({"lat":list(range(data.metsize)), "lon":list(range(data.metsize))})
                    print(mets[j])

                try:
                    # select met
                    ## making sure that only the right indices are selected
                    #print(j, len(mets[j].time), len(mets[j].lat))
                    mets[j] = mets[j].sel(time=(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H")))
                    #print(j, len(mets[j].time), len(mets[j].lat))
                except KeyError:
                    print("in here")
                    # there was a problem with the time indeces - likely because the first datapoints are outside of the range
                    intersect, idxs1, idxs2 = np.intersect1d(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H"), pd.DatetimeIndex(mets[j].time.values), return_indices=True)
                    mets[j] = mets[j].sel(time=intersect)
                    valid_timestamps = pd.DatetimeIndex(np.copy(intersect))+pd.Timedelta(f"{j}H")
                print(j, len(mets[j].time), len(mets[j].lat), len(valid_timestamps))

                if np.any(mets[j].x_wind.isnull()):
                    print(f"there are some nans in the met for jump {j}")
                    mets[j] = mets[j].dropna(dim="time")
                    valid_timestamps = pd.DatetimeIndex(np.copy(mets[j].time.values))+pd.Timedelta(f"{j}H")
                
                if were_there_nans:
                    print("there were nans?!")
                    """
                    print(mets[j].dims)
                    print(mets[j].dims)
                    #print(f"before xwind nans: {np.any(np.isnan(mets[j].x_wind.values))}, ywind nans:  {np.any(np.isnan(mets[j].y_wind.values))}")
                    mets[j]["y_wind"] = mets[j].y_wind.interpolate_na(dim="lat", max_gap = 5)
                    mets[j]["y_wind"] = mets[j].y_wind.interpolate_na(dim="lon", max_gap = 5)
                    #print(f"after xwind nans: {np.any(np.isnan(mets[j].x_wind.values))}, ywind nans:  {np.any(np.isnan(mets[j].y_wind.values))}")
                    print(f"dropping nans fully? {len(mets[j].time.values)}")
                    #mets[j] = mets[j].dropna(dim="time", subset=["y_wind"])
                    mets[j] = mets[j].sel(time=mets[j].time.values[np.where(~np.isnan(mets[j].y_wind.sel(levels=mets[j].levels.values[0], lat=mets[j].lat.values[0],lon=mets[j].lon.values[0]).values))[0]])
                    """

                    print(f"after {len(mets[j].time.values)}")


        # drops and recalculates 
        mets[j] = mets[j].drop_vars(["wind_angle", "wind_speed"])
        
        if "wind_speed" not in list(mets[j].keys()):
            print(f"wind speed isnt present in {j}h data, adding now")
            mets[j]["wind_angle"]=np.arctan2(-mets[j].x_wind,-mets[j].y_wind)
            mets[j]["wind_speed"]=np.sqrt(mets[j].x_wind**2 + mets[j].y_wind**2)
            print("done?")

    # this makes sure that all the footprint data and all the met files are aligned in time 
            # met data is still a dictionary of independent arrays!
    print(len(valid_timestamps), len(data.met.time.values))
    if len(valid_timestamps) < len(data.met.time.values):
        print(f"deleting {len(mets[0].time.values) - len(valid_timestamps)} nan indeces (in the time axis) from jump mets and from the data object")
        # indeces that are nan wrt the original data:
        time_idx_nan = []
        for n,t in enumerate(data.met.time.values):
            if t not in valid_timestamps:
                time_idx_nan.append(n)

        ## keep from here

        for j in jumps:
            print(j, len(mets[j].time), len(mets[j].lat))
            mets[j] = mets[j].sel(time=pd.DatetimeIndex(valid_timestamps) - pd.Timedelta(f"{j}H"))
            print(j, len(mets[j].time), len(mets[j].lat))
        
        data.fp_data = np.delete(data.fp_data, np.unique(time_idx_nan), axis=0)
        data.fp_lats = np.delete(data.fp_lats, np.unique(time_idx_nan), axis=0)
        data.fp_lons = np.delete(data.fp_lons, np.unique(time_idx_nan), axis=0)
        data.topog = np.delete(data.topog, np.unique(time_idx_nan), axis=0)
    
        if hasattr(data, "land_cover"):
            data.land_cover = np.delete(data.land_cover, time_idx_nan, axis=0)

        data.met = data.met.sel(time=valid_timestamps)
        if hasattr(data, "fp_binary"):
            data.fp_binary = np.delete(data.fp_binary, np.unique(time_idx_nan), axis=0)


    print(len(data.met.time))

    for j in jumps:
        print(j, len(mets[j].time), len(mets[j].lat))
        

    n_vars_with_past =(len(jumps))*np.sum([len(variables_past[var]) for var in variables_past])
    if len([len(variables_nopast[var]) for var in variables_nopast])==0:
        n_vars_no_past=0
    else:
        n_vars_no_past =np.sum([len(variables_nopast[var]) for var in variables_nopast]) 

    
    n_vars_reduced =(len(jumps_reduced))*np.sum([len(variables_reduced[var]) for var in variables_reduced])
    
    n_variables = n_vars_with_past+n_vars_no_past+len(others)+topog+n_vars_reduced

    if "full_land_cover" in others or "binary_land_cover" in others:
        # there are nine types of land cover, plus sea-land mask
        # ignoring the last type of land cover (ICE) as this doesn't apply in most domains!
        n_variables = n_variables + 8

    if "relative_time" in others:
        n_variables = n_variables + len(jumps) -  1 
            ## as relative time adds one uniform variable to all nodes to signpost time of the inputs with respect to release - ie met at release will have variable with value 0, six hours before will have value 6 etc
            # this messes up the training big time! do not pass
    
    if "normalised_time_of_year" in others:
        n_variables = (n_variables-1) + 2*len(jumps)   
    if "normalised_time_of_day" in others:
        n_variables = (n_variables-1) + 2*len(jumps)  
        # this messes up the training big time! do not pass
    print(n_variables)

    ## all_vars is a numpy array, in the right shape, that will host the input variables
    # the for loop below "fills up" this numpy array, per variable and timestep, extracting the variable from each corresponding array in the mets dictionary
    all_vars = np.zeros((np.shape(data.fp_lats)[1], np.shape(data.fp_lons)[1], len(data.met.time), int(n_variables)))

    
    col = 0
    for v in variables_past:
        for njump, jump in enumerate(og_jumps):
            #print(v, jump)
            mets[jump][v].load()
            for lev in variables_past[v]:
                #print(col, v, jump, lev)
                if hasattr(mets[jump][v], "levels"):
                    print(jump, v, lev)
                    cutmet = mets[jump][v].sel(levels=lev).values
                    vartype="3D"
                else:
                    print(jump, v)
                    cutmet = mets[jump][v].values
                    vartype="2D"
                #print(mets[jump][v])
                #print(np.shape(cutmet))
                if relative_met and njump>0:
                    all_vars[:,:,:,col] = cutmet - all_vars[:,:,:,col-njump]

                    var_names.append({"var":v, "level":lev, "type": vartype, "time":f"t-{jump} - t0"})
                else:
                    all_vars[:,:,:,col] = cutmet
                
                    var_names.append({"var":v, "level":lev, "type": vartype, "time":f"t-{jump}"})
                col = col+1

    for v in variables_nopast:
        for lev in variables_nopast[v]:
            if hasattr(mets[0][v], "levels"):
                cutmet = mets[0][v].sel(levels=lev).values
                var_names.append({"var":v, "level":lev, "time":"present", "type": "3D"})
            else:
                cutmet = data.met[v].values
                var_names.append({"var":v, "level":lev, "time":"present", "type": "2D"})
                if lev!=0:
                    print("Careful! This varible has no levels but you passed a level different from 0")
            
            all_vars[:,:,:,col] = cutmet
            col=col+1
    #print("brrr", len(variables_reduced), len(jumps_reduced))
    ## ignore these below!
    if len(variables_reduced)>0 and len(jumps_reduced)>0:
        for v in variables_reduced:
            for njump, jump in enumerate(jumps_reduced):
                print("here", jump)
                #print(v, jump)
                mets[jump][v].load()
                for lev in variables_past[v]:
                    #print(col, v, jump, lev)
                    if hasattr(mets[jump][v], "levels"):
                        print(jump, v, lev)
                        cutmet = mets[jump][v].sel(levels=lev).values
                        vartype="3D"
                    else:
                        print(jump, v)
                        cutmet = mets[jump][v].values
                        vartype="2D"
                    #print(mets[jump][v])
                    #print(np.shape(cutmet))
                    if relative_met and njump>0:
                        all_vars[:,:,:,col] = cutmet - all_vars[:,:,:,col-njump]

                        var_names.append({"var":v, "level":lev, "type": vartype, "time":f"t-{jump} - t0"})
                    else:
                        all_vars[:,:,:,col] = cutmet
                    
                        var_names.append({"var":v, "level":lev, "type": vartype, "time":f"t-{jump}"})
                    col = col+1

    ## adds the variables that are not meteorological
    if len(others) > 0:
        for oth in others:
            ## add option for  solar radiation, orography, land-sea mask
            var_names.append({"var":oth, "type": "not met"})

            numRows, numCols = np.shape(data.fp_lats)[1], np.shape(data.fp_lons)[1]
            if oth == "lat_coords":
                x =np.copy(data.fp_lats)
                x = x.reshape(1,len(x), numCols)
                x = x.repeat(numRows, axis=0)
                coord = np.transpose(x, [0,2,1])                
                all_vars[:,:,:,col] = coord
                del x, coord
                col+=1
                
            elif oth == "lon_coords":
                y =np.copy(data.fp_lons)
                y = y.reshape(len(y), numRows,1)
                coord = y.repeat(numCols, axis=2)
                coord = np.transpose(y, [1,2,0]) 
                all_vars[:,:,:,col] = coord
                del y, coord
                col+=1
            elif oth == "sin_lat_coords":
                x = np.sin(np.copy(data.fp_lats))*np.pi / 180
                x = x.reshape(1,len(x), numCols)
                x = x.repeat(numRows, axis=0)
                coord = np.transpose(x, [0,2,1])   
                print(np.shape(coord))
                all_vars[:,:,:,col] = coord
                del x, coord
                col+=1
                
            elif oth == "sin_lon_coords":
                y = np.sin(np.copy(data.fp_lons)) * np.pi / 180
                y = y.reshape(len(y), numRows,1)
                coord = y.repeat(numCols, axis=2)
                coord = np.transpose(y, [1,2,0]) 
                all_vars[:,:,:,col] = coord
                del y, coord
                col+=1

            elif oth == "cos_lat_coords":
                x = np.cos(np.copy(data.fp_lats))*np.pi / 180
                x = x.reshape(1,len(x), numCols)
                x = x.repeat(numRows, axis=0)
                coord = np.transpose(x, [0,2,1])   
                all_vars[:,:,:,col] = coord
                del x, coord
                col+=1
                
            elif oth == "cos_lon_coords":
                y = np.cos(np.copy(data.fp_lons)) * np.pi / 180
                y = y.reshape(len(y), numRows,1)
                coord = y.repeat(numCols, axis=2)
                coord = np.transpose(y, [1,2,0]) 
                all_vars[:,:,:,col] = coord
                del y, coord
                col+=1

            elif oth == "x_coords":
                grid_coords = np.meshgrid(np.arange(np.shape(data.fp_lats)[1]), np.arange(np.shape(data.fp_lons)[1]))     
                coord = np.dstack([grid_coords[0]]*np.shape(all_vars)[2])  
                if centered_coords:
                    coord = coord - int(data.size/2)    
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth == "y_coords":
                grid_coords = np.meshgrid(np.arange(np.shape(data.fp_lats)[1]), np.arange(np.shape(data.fp_lons)[1]))            
                coord = np.dstack([grid_coords[1]]*np.shape(all_vars)[2])    
                if centered_coords:
                    coord = coord - int(data.size/2)    
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth=="binary_centre":
                centre = int(data.size/2)
                grid = np.zeros((numRows, numCols)) -1
                grid[centre, centre] = 1
                coord = np.dstack([grid]*np.shape(all_vars)[2])               
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth=="distance_centre":
                ## distance centre is not real km currently, it's in arbitrary units
                centre = int(np.shape(data.fp_lats)[1]/2)
                grid_coords = np.meshgrid(np.arange(numRows), np.arange(numCols)) 
                distance = np.sqrt(np.abs(grid_coords[0]-centre)**2 + np.abs(grid_coords[1]-centre)**2)
                coord = np.dstack([distance]*np.shape(all_vars)[2])               
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth=="normalised_time_of_day":
                _ = var_names.pop() 
                for j in jumps:
                    t = pd.DatetimeIndex(mets[j].time.values)
                    seconds = (t.hour * 60 + t.minute) * 60 + t.second
                    seconds = np.tile(seconds, (np.shape(all_vars)[0],np.shape(all_vars)[1],1))
                    seconds_in_day = 86400
                    all_vars[:,:,:,col] = np.sin(2*np.pi*seconds/seconds_in_day)
                    col+=1
                    all_vars[:,:,:,col] = np.cos(2*np.pi*seconds/seconds_in_day)
                    col+=1
                    var_names.append({"var":"sin_time_day", "type": "not met", "time":f"t-{j}"})
                    var_names.append({"var":"cos_time_day", "type": "not met", "time":f"t-{j}"})


            elif oth=="normalised_day_of_year":
                _ = var_names.pop() 
                t = pd.DatetimeIndex(mets[0].time.values)
                hours = (t - pd.to_datetime(t.year, format='%Y')).days/365
                hours = np.tile(hours, (np.shape(all_vars)[0],np.shape(all_vars)[1],1))
                all_vars[:,:,:,col] = np.sin(2*np.pi*hours)
                col+=1
                all_vars[:,:,:,col] = np.cos(2*np.pi*hours)
                col+=1
                var_names.append({"var":"sin_time_year", "type": "not met", "time":f"0"})
                var_names.append({"var":"cos_time_year", "type": "not met", "time":f"0"})

            elif oth=="normalised_time_of_year":
                _ = var_names.pop() 
                for j in jumps:
                    t = pd.DatetimeIndex(mets[j].time.values)
                    hours = (t - pd.to_datetime(t.year, format='%Y')).days*24 + (t - pd.to_datetime(t.year, format='%Y')).seconds/3600
                    hours_in_year = 8760
                    hours = np.tile(hours, (np.shape(all_vars)[0],np.shape(all_vars)[1],1))
                    all_vars[:,:,:,col] = np.sin(2*np.pi*hours/hours_in_year)
                    col+=1
                    all_vars[:,:,:,col] = np.cos(2*np.pi*hours/hours_in_year)
                    col+=1
                    var_names.append({"var":"sin_time_year", "type": "not met", "time":f"t-{j}"})
                    var_names.append({"var":"cos_time_year", "type": "not met", "time":f"t-{j}"})


            elif oth=="relative_time":
                _ = var_names.pop() 
                for j in jumps:
                    grid = np.zeros((numRows, numCols)) + j
                    coord = np.dstack([grid]*np.shape(all_vars)[2])  
                    all_vars[:,:,:,col] = coord
                    col+=1
                    var_names.append({"var":oth, "type": "not met", "time":f"t-{j}"})
            elif "land_cover" in oth:
                _ = var_names.pop() 
                continue
            else:
                print(f"{oth} not recognised!")   
                _ = var_names.pop() 

    ## adds topography and land cover to the output array         
    if topog:
        if hasattr(data, "topog"):
            all_vars[:,:,:,-1] = np.transpose(data.topog, [1,2,0])
        else:
            print("topog was passed as true but there is no topography in the data class. Load the data again using custom topog or 'default'. Last column of input will be empty ")

        if "sea_mask" in others:
            if hasattr(data, "disaggregated_land_cover"):
                # ignoring the last type of land cover (ICE) as this doesn't apply in most domains!
                all_vars[:,:,:,col] = np.transpose(data.disaggregated_land_cover[:,:,:,0], [1,2,0])
                var_names.append({"var":f"binary sea mask", "type": "not met"})
            else:
                print("sea_mask was passed as true but there is no disaggregated_land_cover in the data class. the corresponding column in the inputs will be empty!")        

        if "land_cover" in others:
            if hasattr(data, "land_cover"):
                all_vars[:,:,:,-2] = np.transpose(data.land_cover, [1,2,0])
            else:
                print("land_cover was passed as true but there is no land_cover in the data class. Second to last column of input will be empty ")                                 
            
            var_names.append({"var":"land_cover", "type": "not met"}) 

        if "full_land_cover" in others:
            if hasattr(data, "disaggregated_land_cover"):
                # ignoring the last type of land cover (ICE) as this doesn't apply in most domains!
                for pseudolevel in range(9):
                    all_vars[:,:,:,col] = np.transpose(data.disaggregated_land_cover[:,:,:,pseudolevel], [1,2,0])
                    col = col+1

                    var_names.append({"var":f"land_cover pseudolevel {pseudolevel}", "type": "not met"})
            else:
                print("full_land_cover was passed as true but there is no disaggregated_land_cover in the data class. all land cover related cols of returned inputs will be empty!")                                 
        print(others)
        print("binary_land_cover" in others)
        if "binary_land_cover" in others:
            if hasattr(data, "disaggregated_land_cover"):
                # ignoring the last type of land cover (ICE) as this doesn't apply in most domains!
                for pseudolevel in range(9):
                    all_vars[:,:,:,col] = 1*np.transpose(data.land_cover==pseudolevel, [1,2,0])
                    col = col+1

                    var_names.append({"var":f"binary_land_cover pseudolevel {pseudolevel}", "type": "not met"})
            else:
                print("full_land_cover was passed as true but there is no disaggregated_land_cover in the data class. all land cover related cols of returned inputs will be empty!")                 

             


        var_names.append({"var":"topog", "type": "not met"}) 

    latlons, idx_latlons = get_grid(data, latlon_fp)

    if transform:
        all_vars = np.reshape(all_vars, (np.shape(all_vars)[0]*np.shape(all_vars)[1], np.shape(all_vars)[2], np.shape(all_vars)[3]))
        all_vars = np.transpose(all_vars, [1,0,2])
    
    print(f"there are {np.sum(np.isnan(all_vars))} nans in the inputs")
    print(np.where(np.isnan(all_vars)))

    if return_idx:
        return latlons, idx_latlons, all_vars, var_names, data
    else:
        return latlons, all_vars, var_names, data




def get_all_inputs_graphnet_satellite_v5(data, variables_past, jumps, variables_nopast, others=[], topog=True, latlon_fp=0, transform=True, add_current_time=True, centered_coords=False, return_idx=False, relative_met=False):
    """
    get inputs from LoadSatelliteData object and format as array of size (time, variables)

    Inputs:
        - data - LoadSatelliteData object
        - variables_past - dictionary of variables to load for all times in jumps with format {"varname":levels to load} eg {{"x_wind":[3,9]} 
        - jumps - list of ints. Met for vars in variables_past will be used as inputs for t-jump for all jumps, where t is the timestamp of the footprint
        - add_current_time - wether to add the meteorology at the timestamps of the footprint (just adds 0 to jumps if it isnt there already)
        - variables_nopast - dictionary of variables to load only at t=0 with format {"varname":levels to load} eg {{"x_wind":[3,9]} 
        - others - list of other non-meterological variables to load. see below for names and explanation
        - topog - bool, if True add topography at each lat/lon as a feature
        - latlon_fp - int, index of footprint to use as reference to create the latlon grid. default is 0, first footprint in the dataset
        - transform - bool, if False return data with shape (size,size, time, features), if True return with shape (time, size*size, features) 
        - centered_coords - bool, only used if x_coords and/or y_coords are in others. If True, x/y coords have the release point as (0,0), if false the south-west corner is (0,0) and the release point is (size/2, size/2) 
        - return_idx - bool, if True return node indeces

        - relative_met: if false, return met at t-xh for each x in jumps, if true return met relative to t0 (ie t0, tx - t0 ...)

    Returns:
        - latlons - latlon grid to pass to model
        - idx_latlons - node indeces - only returned if return_idx=True
        - all_vars - features, shape depends on transform
        - var_names - list containing information about each of the features, of length len(features). Each entry is a dictionary with attributes "var" (name), "level", "type" (3D if it's a variable with levels, 2D otherwise, "not met" if it's topography or others) and "time" ("t-0" indicates the meteorology is for the time of the footprint, "t-6" six hours before and so on)
        - data - the LoadSatelliteData object, with any updates to the datasets if there are nans in the past data 

    Accepted values in others:
        - lat_coords/lon_coords: lat/lon coordinate for each node
        - sin_lat_coords/sin_lon_coords/cos_lat_coords/cos_lon_coords: useful mostly if working with a big/whole world domain to encode the sphere
        - x_coords/y_coords: x/y index of each node (see centered_coords above)
        - binary_centre: zero for each node except 1 for release node
        - distance_centre: euclidean distance to the release node (using x/y coords, not lat/lon coords for speed of calculation)
        - normalised_time_of_day and normalised_time_of_year: sin and cos of the time of the normalised time of the day and year. Meant to encode cyclical patterns. For some reason it really messes up with training, do not use!
        - relative_time - ignore too
    """

    all_vars = []
    var_names = []

    if relative_met:
        assert (0 in jumps) or add_current_time, "jump 0 is needed to do relative met!"

    if not (0 in jumps) and add_current_time:
        jumps.append(0) # append 0 to get present met too
    jumps=list(sorted(set(jumps)))
    print(f"hours back in time: {jumps}")

    # mets contains all of the met objects, the key is the jump
    mets={}
    mets[0] = data.met
    valid_timestamps = np.copy(data.met.time.values)
    for j in jumps:
        if j!=0:
            met_files = glob.glob(f"{data.met_datadir}{j}h_{data.date}*")
            if len(met_files)==0:
                print(f"couldnt find files for jump {j} at {data.met_datadir}{j}h_")
            else:
                time_chunk = round(1000000/(data.size*data.size), -2)
                with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                    #mets[j] = xr.open_mfdataset(met_files, parallel=True, chunks = {"level":1,"time":time_chunk})
                    mets[j] = xr.open_mfdataset(sorted(met_files), combine='by_coords', data_vars="minimal", compat="override", coords="minimal", parallel=True, join="inner", chunks = {"level":1, "time":time_chunk})

                if len(mets[j].lat) > data.metsize:
                    print(f"file for jump {j}h is bigger than metsize, cutting down")
                    diff = int((len(mets[j].lat) - data.metsize)/2)
                    cut_idxs =  mets[j].lat.values[diff:-diff]
                    mets[j] = mets[j].sel(lat=cut_idxs, lon=cut_idxs)
                    mets[j] = mets[j].assign_coords({"lat":list(range(data.metsize)), "lon":list(range(data.metsize))})
                    print(mets[j])

                try:
                    # select met
                    print(j, len(mets[j].time), len(mets[j].lat))
                    mets[j] = mets[j].sel(time=(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H")))
                    print(j, len(mets[j].time), len(mets[j].lat))
                except KeyError:
                    print("in here")
                    # there was a problem with the time indeces - likely because the first datapoints are outside of the range
                    intersect, idxs1, idxs2 = np.intersect1d(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H"), pd.DatetimeIndex(mets[j].time.values), return_indices=True)
                    mets[j] = mets[j].sel(time=intersect)
                    valid_timestamps = pd.DatetimeIndex(np.copy(intersect))+pd.Timedelta(f"{j}H")
                print(j, len(mets[j].time), len(mets[j].lat), len(valid_timestamps))

                if np.any(mets[j].x_wind.isnull()):
                    print(f"there are some nans in the met for jump {j}")
                    mets[j] = mets[j].dropna(dim="time")
                    valid_timestamps = pd.DatetimeIndex(np.copy(mets[j].time.values))+pd.Timedelta(f"{j}H")
        
        if "wind_speed" not in list(mets[j].keys()):
            print(f"wind speed isnt present in {j}h data, adding now")
            mets[j]["wind_angle"]=np.arctan2(-mets[j].x_wind,-mets[j].y_wind)
            mets[j]["wind_speed"]=np.sqrt(mets[j].x_wind**2 + mets[j].y_wind**2)
            print("done?")

    print(len(valid_timestamps), len(data.met.time.values))
    if len(valid_timestamps) < len(data.met.time.values):
        print(f"deleting {len(mets[0].time.values) - len(valid_timestamps)} nan indeces (in the time axis) from jump mets and from the data object")
        # indeces that are nan wrt the original data:
        time_idx_nan = []
        for n,t in enumerate(data.met.time.values):
            if t not in valid_timestamps:
                time_idx_nan.append(n)

        ## keep from here

        for j in jumps:
            print(j, len(mets[j].time), len(mets[j].lat))
            mets[j] = mets[j].sel(time=pd.DatetimeIndex(valid_timestamps) - pd.Timedelta(f"{j}H"))
            print(j, len(mets[j].time), len(mets[j].lat))
        
        data.fp_data = np.delete(data.fp_data, np.unique(time_idx_nan), axis=0)
        data.fp_lats = np.delete(data.fp_lats, np.unique(time_idx_nan), axis=0)
        data.fp_lons = np.delete(data.fp_lons, np.unique(time_idx_nan), axis=0)
        data.topog = np.delete(data.topog, np.unique(time_idx_nan), axis=0)
        data.met = data.met.sel(time=valid_timestamps)
        if hasattr(data, "fp_binary"):
            data.fp_binary = np.delete(data.fp_binary, np.unique(time_idx_nan), axis=0)


    print(len(data.met.time))

    for j in jumps:
        print(j, len(mets[j].time), len(mets[j].lat))
        

    n_vars_with_past =(len(jumps))*np.sum([len(variables_past[var]) for var in variables_past])
    if len([len(variables_nopast[var]) for var in variables_nopast])==0:
        n_vars_no_past=0
    else:
        n_vars_no_past =np.sum([len(variables_nopast[var]) for var in variables_nopast]) 
    
    n_variables = n_vars_with_past+n_vars_no_past+len(others)+topog

    if "full_land_cover" in others:
        # there are nine types of land cover, plus sea-land mask
        # ignoring the last type of land cover (ICE) as this doesn't apply in most domains!
        n_variables = n_variables + 8

    if "relative_time" in others:
        n_variables = n_variables + len(jumps) -  1 
            ## as relative time adds one uniform variable to all nodes to signpost time of the inputs with respect to release - ie met at release will have variable with value 0, six hours before will have value 6 etc
            # this messes up the training big time! do not pass
    
    if "normalised_time_of_year" in others:
        n_variables = (n_variables-1) + 2*len(jumps)   
    if "normalised_time_of_day" in others:
        n_variables = (n_variables-1) + 2*len(jumps)  
        # this messes up the training big time! do not pass

    all_vars = np.zeros((np.shape(data.fp_lats)[1], np.shape(data.fp_lons)[1], len(data.met.time), n_variables))

    
    col = 0
    for v in variables_past:
        for njump, jump in enumerate(jumps):
            #print(v, jump)
            mets[jump][v].load()
            for lev in variables_past[v]:
                #print(col, v, jump, lev)
                if hasattr(mets[jump][v], "levels"):
                    print(jump, v, lev)
                    cutmet = mets[jump][v].sel(levels=lev).values
                    vartype="3D"
                else:
                    print(jump, v)
                    cutmet = mets[jump][v].values
                    vartype="2D"
                #print(mets[jump][v])
                #print(np.shape(cutmet))
                if relative_met and njump>0:
                    all_vars[:,:,:,col] = cutmet - all_vars[:,:,:,col-njump]

                    var_names.append({"var":v, "level":lev, "type": vartype, "time":f"t-{jump} - t0"})
                else:
                    all_vars[:,:,:,col] = cutmet
                
                    var_names.append({"var":v, "level":lev, "type": vartype, "time":f"t-{jump}"})
                col = col+1

    for v in variables_nopast:
        for lev in variables_nopast[v]:
            if hasattr(mets[0][v], "levels"):
                cutmet = mets[0][v].sel(levels=lev).values
                var_names.append({"var":v, "level":lev, "time":"present", "type": "3D"})
            else:
                cutmet = data.met[v].values
                var_names.append({"var":v, "level":lev, "time":"present", "type": "2D"})
                if lev!=0:
                    print("Careful! This varible has no levels but you passed a level different from 0")
            
            all_vars[:,:,:,col] = cutmet
            col=col+1


    if len(others) > 0:
        for oth in others:
            ## add option for  solar radiation, orography, land-sea mask
            var_names.append({"var":oth, "type": "not met"})

            numRows, numCols = np.shape(data.fp_lats)[1], np.shape(data.fp_lons)[1]
            if oth == "lat_coords":
                x =np.copy(data.fp_lats)
                x = x.reshape(1,len(x), numCols)
                x = x.repeat(numRows, axis=0)
                coord = np.transpose(x, [0,2,1])                
                all_vars[:,:,:,col] = coord
                del x, coord
                col+=1
                
            elif oth == "lon_coords":
                y =np.copy(data.fp_lons)
                y = y.reshape(len(y), numRows,1)
                coord = y.repeat(numCols, axis=2)
                coord = np.transpose(y, [1,2,0]) 
                all_vars[:,:,:,col] = coord
                del y, coord
                col+=1
            elif oth == "sin_lat_coords":
                x = np.sin(np.copy(data.fp_lats))*np.pi / 180
                x = x.reshape(1,len(x), numCols)
                x = x.repeat(numRows, axis=0)
                coord = np.transpose(x, [0,2,1])   
                print(np.shape(coord))
                all_vars[:,:,:,col] = coord
                del x, coord
                col+=1
                
            elif oth == "sin_lon_coords":
                y = np.sin(np.copy(data.fp_lons)) * np.pi / 180
                y = y.reshape(len(y), numRows,1)
                coord = y.repeat(numCols, axis=2)
                coord = np.transpose(y, [1,2,0]) 
                all_vars[:,:,:,col] = coord
                del y, coord
                col+=1

            elif oth == "cos_lat_coords":
                x = np.cos(np.copy(data.fp_lats))*np.pi / 180
                x = x.reshape(1,len(x), numCols)
                x = x.repeat(numRows, axis=0)
                coord = np.transpose(x, [0,2,1])   
                all_vars[:,:,:,col] = coord
                del x, coord
                col+=1
                
            elif oth == "cos_lon_coords":
                y = np.cos(np.copy(data.fp_lons)) * np.pi / 180
                y = y.reshape(len(y), numRows,1)
                coord = y.repeat(numCols, axis=2)
                coord = np.transpose(y, [1,2,0]) 
                all_vars[:,:,:,col] = coord
                del y, coord
                col+=1

            elif oth == "x_coords":
                grid_coords = np.meshgrid(np.arange(np.shape(data.fp_lats)[1]), np.arange(np.shape(data.fp_lons)[1]))     
                coord = np.dstack([grid_coords[0]]*np.shape(all_vars)[2])  
                if centered_coords:
                    coord = coord - int(data.size/2)    
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth == "y_coords":
                grid_coords = np.meshgrid(np.arange(np.shape(data.fp_lats)[1]), np.arange(np.shape(data.fp_lons)[1]))            
                coord = np.dstack([grid_coords[1]]*np.shape(all_vars)[2])    
                if centered_coords:
                    coord = coord - int(data.size/2)    
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth=="binary_centre":
                centre = int(data.size/2)
                grid = np.zeros((numRows, numCols)) -1
                grid[centre, centre] = 1
                coord = np.dstack([grid]*np.shape(all_vars)[2])               
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth=="distance_centre":
                ## distance centre is not real km currently, it's in arbitrary units
                centre = int(np.shape(data.fp_lats)[1]/2)
                grid_coords = np.meshgrid(np.arange(numRows), np.arange(numCols)) 
                distance = np.sqrt(np.abs(grid_coords[0]-centre)**2 + np.abs(grid_coords[1]-centre)**2)
                coord = np.dstack([distance]*np.shape(all_vars)[2])               
                all_vars[:,:,:,col] = coord
                col+=1
            elif oth=="normalised_time_of_day":
                _ = var_names.pop() 
                for j in jumps:
                    t = pd.DatetimeIndex(mets[j].time.values)
                    seconds = (t.hour * 60 + t.minute) * 60 + t.second
                    seconds = np.tile(seconds, (np.shape(all_vars)[0],np.shape(all_vars)[1],1))
                    seconds_in_day = 86400
                    all_vars[:,:,:,col] = np.sin(2*np.pi*seconds/seconds_in_day)
                    col+=1
                    all_vars[:,:,:,col] = np.cos(2*np.pi*seconds/seconds_in_day)
                    col+=1
                    var_names.append({"var":"sin_time_day", "type": "not met", "time":f"t-{j}"})
                    var_names.append({"var":"cos_time_day", "type": "not met", "time":f"t-{j}"})


            elif oth=="normalised_day_of_year":
                _ = var_names.pop() 
                t = pd.DatetimeIndex(mets[0].time.values)
                hours = (t - pd.to_datetime(t.year, format='%Y')).days/365
                hours = np.tile(hours, (np.shape(all_vars)[0],np.shape(all_vars)[1],1))
                all_vars[:,:,:,col] = np.sin(2*np.pi*hours)
                col+=1
                all_vars[:,:,:,col] = np.cos(2*np.pi*hours)
                col+=1
                var_names.append({"var":"sin_time_year", "type": "not met", "time":f"0"})
                var_names.append({"var":"cos_time_year", "type": "not met", "time":f"0"})

            elif oth=="normalised_time_of_year":
                _ = var_names.pop() 
                for j in jumps:
                    t = pd.DatetimeIndex(mets[j].time.values)
                    hours = (t - pd.to_datetime(t.year, format='%Y')).days*24 + (t - pd.to_datetime(t.year, format='%Y')).seconds/3600
                    hours_in_year = 8760
                    hours = np.tile(hours, (np.shape(all_vars)[0],np.shape(all_vars)[1],1))
                    all_vars[:,:,:,col] = np.sin(2*np.pi*hours/hours_in_year)
                    col+=1
                    all_vars[:,:,:,col] = np.cos(2*np.pi*hours/hours_in_year)
                    col+=1
                    var_names.append({"var":"sin_time_year", "type": "not met", "time":f"t-{j}"})
                    var_names.append({"var":"cos_time_year", "type": "not met", "time":f"t-{j}"})


            elif oth=="relative_time":
                _ = var_names.pop() 
                for j in jumps:
                    grid = np.zeros((numRows, numCols)) + j
                    coord = np.dstack([grid]*np.shape(all_vars)[2])  
                    all_vars[:,:,:,col] = coord
                    col+=1
                    var_names.append({"var":oth, "type": "not met", "time":f"t-{j}"})
            elif oth=="land_cover" or oth=="full_land_cover":
                _ = var_names.pop() 
                continue
            else:
                print(f"{oth} not recognised!")   
                _ = var_names.pop() 

              
    if topog:
        if hasattr(data, "topog"):
            all_vars[:,:,:,-1] = np.transpose(data.topog, [1,2,0])
        else:
            print("topog was passed as true but there is no topography in the data class. Load the data again using custom topog or 'default'. Last column of input will be empty ")

        if "land_cover" in others:
            if hasattr(data, "land_cover"):
                all_vars[:,:,:,-2] = np.transpose(data.land_cover, [1,2,0])
            else:
                print("land_cover was passed as true but there is no land_cover in the data class. Second to last column of input will be empty ")                                 
            
            var_names.append({"var":"land_cover", "type": "not met"}) 

        if "full_land_cover" in others:
            print("here")
            if hasattr(data, "disaggregated_land_cover"):
                # ignoring the last type of land cover (ICE) as this doesn't apply in most domains!
                for pseudolevel in range(9):
                    all_vars[:,:,:,col] = np.transpose(data.disaggregated_land_cover[:,:,:,pseudolevel], [1,2,0])
                    col = col+1

                    var_names.append({"var":f"land_cover pseudolevel {pseudolevel}", "type": "not met"})
        
            else:
                print("full_land_cover was passed as true but there is no disaggregated_land_cover in the data class. all land cover related cols of returned inputs will be empty!")                                 

             


        var_names.append({"var":"topog", "type": "not met"}) 


    latlons, idx_latlons = get_grid(data, latlon_fp)

    if transform:
        all_vars = np.reshape(all_vars, (np.shape(all_vars)[0]*np.shape(all_vars)[1], np.shape(all_vars)[2], np.shape(all_vars)[3]))
        all_vars = np.transpose(all_vars, [1,0,2])
    
    if return_idx:
        return latlons, idx_latlons, all_vars, var_names, data
    else:
        return latlons, all_vars, var_names, data



def getint(name):
    num = name.split('_')[-1]
    num = num.split('.')[0]
    return int(num)

def get_grid(data, latlon_fp):
    """
    produce reference grid and node indeces
    """
    print(f"getting grid for time {data.met.time.values[latlon_fp]}")
    #print(f"making grid for footprint at time {data}")
    single_meshgrid = np.meshgrid(data.fp_lats[latlon_fp,:], data.fp_lons[latlon_fp,:])
    latlons = [(single_meshgrid[0][i,j], single_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lats)[1]) for j in range(np.shape(data.fp_lats)[1])] 

    idx_meshgrid = np.meshgrid(list(range(len(data.fp_lats[latlon_fp,:]))), list(range(len(data.fp_lons[latlon_fp,:]))))
    idx_meshgrid = np.array(idx_meshgrid)-int(data.size/2)
    idx_latlons = [(idx_meshgrid[0][i,j], idx_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lats)[1]) for j in range(np.shape(data.fp_lats)[1])] 

    return latlons, idx_latlons

"""
def grid_coordinates(side):
    xx, yy = np.meshgrid(np.array(list(range(side)), dtype=np.float32), np.array(list(range(side)), dtype=np.float32))
    z = np.empty((side**2, 2), np.float32)
    z[:, 1] = xx.reshape(side**2)
    z[:, 0] = yy.reshape(side**2)
    return z
"""





