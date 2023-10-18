import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys


def load_fps(fp_datadir):
    """
    Load footprints from datadir, using workaround if problematic files are encountered
    note that the list of problematic files is currently updated manually!
    """
    try:           
        time_chunk = 25
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            fp_data_full = xr.open_mfdataset(sorted(glob.glob(fp_datadir)), combine='by_coords', chunks = {"time":time_chunk})
    except Exception as e:
        # some files have small errors in format that prevent xr from concatenating and opening together. This is a workaround to open those separately. This list only contains known files and could be more! can add manually whenever you encounter one
        print("checking bad files")
        fp_files = sorted(glob.glob(fp_datadir))
        bad_files = ["/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/SOUTHAMERICA/GOSAT-BRAZIL-column_SOUTHAMERICA_201511.nc", 
            "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201409.nc", 
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201501.nc',
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201502.nc',
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201503.nc',
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201504.nc',
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201609.nc',
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201610.nc',
            '/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_201612.nc']
        without_bad_files = list(set(fp_files) - set(bad_files))
        if len(without_bad_files) < len(fp_files):
            print("at least one of the files was in the bad files list, opening with workaround")
            # error arises because fp file for Brazil Nov 2015 has non-monotonic timestamps, use workaround
            # error arises because fp file for Sahara Nov 2014 has non-monotonic timestamps, use workaround
            # also some Sahara 2015 files are missing mean_age_particles_ variable - drop and concat
            # Add clauses here to catch other known exceptions
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                most = xr.open_mfdataset(sorted(without_bad_files))
                bad_arrays = []
                for badfile in bad_files:
                    if badfile in fp_files:
                        f_bad = xr.open_mfdataset(badfile)
                        if "NORTHAFRICA_2015" in badfile:
                            try:
                                f_bad = f_bad.drop(["mean_age_particles_n", "mean_age_particles_e", "mean_age_particles_w", "mean_age_particles_s"])        
                            except Exception as e:
                                print(e)
                        bad_arrays.append(f_bad)
                fp_data_full = xr.concat([most]+bad_arrays, dim="time")
        else:
            print("there was a problem", e)

    fp_data_full= fp_data_full.sortby('time')

    return fp_data_full
    

def cut_and_save_met_data(date, 
                 region="BRAZIL", 
                 met_jump=[0], size=50,
                 met_levels=[3,9,15,21,30,42,51], met_variables=["air_pressure", "air_temperature", "atmosphere_boundary_layer_thickness", "surface_air_pressure", "upward_air_velocity", "x_wind", "y_wind"], 
                 verbose=True, 
                 met_datadir=None, fp_datadir=None, 
                 savemetpath=[]):
    """
    Cuts meteorology from default format (full lat-lon grid, regular time steps) to match footprint domain and timesteps, as well as to match levels and variables specified as inputs. Can be used for multiple timesteps back in the past, saving the meteorology in the same domain at t-x for each x in met_jump.
    Saves extracted met as .nc file to use in LoadSatelliteData object.
        - met_jump: int or list of ints. Determines the time of the meteorology with respect to the time of the footprint - eg met_jump=0 will interpolate the meteorology to the time of the footprint, met_jump=6 will interpolate the met to t-6h where t is time of the footprint etc. If a list, a corresponding list of saving paths must be passed to savemetpath  
        - savemetpath: str or list of strs with path/filename to save the cut met
        
        - met_variables: list of variables to keep from the original meteorology file (must be present in the meteorology)
        - met_levels: list of levels to keep from the original meteorology file
    """
    domains = {"BRAZIL":"SOUTHAMERICA", "SOUTHAMERICA":"SOUTHAMERICA", "SAHARA":"NORTHAFRICA", "INDIA":"SOUTHASIA"}

    # Load reference footprints
    if fp_datadir==None:
        fp_datadir = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/"+domains[region]+"/*"+region+"*"+domains[region]+"_"+str(date)+"*.nc"
    else:
        fp_datadir=fp_datadir+str(date)+"*.nc"

    fp_data_full = load_fps(fp_datadir)

    fp_data_full.lat.load()
    fp_data_full.lon.load()
    fp_data_full.release_lat.load()
    fp_data_full.release_lon.load()
    fp_data_full.time.load()

    if met_datadir==None:
        met_datadir = "/group/chemistry/acrg/met_archive/UM/"+domains[region]+"/"+domains[region]+"_Met_"+str(date)+"*.nc"
    else:
        met_datadir = met_datadir+str(date)+"*.nc"
    
    if verbose: print("Loading meteorology from " + met_datadir)

    # each chunk should have around 1mill values - chunk per level and by time, rounded to the nearest hundred
    time_chunk = round(1000000/(size*size), -2)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        met = xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords', parallel=True, chunks = {"level":1, "time":time_chunk})    

    assert "lat_coords" not in met.coords, "this meteorology seems to have already been cut. are you sure you are choosing the right file?"

    """
    checking if there is extramet (ie met outside of the fp_data_full domain) that can be appended.
    this is only needed if all of these are true
        a) The size to cut footprints is so big (>100 for Brazil) that many footprints are removed due to being partially out-of-domain
        b) partially out-of-domain footprints are valid and needed
        c) the met files have the same domain as the footprint files (which is default right now)
    
    TODO improve/replace this - could be avoided if met loaded is already of right size (ie bigger than fp domain)
    """
    dims = {"ExtraE":"longitude", "ExtraW":"longitude", "ExtraN":"latitude",  "ExtraS":"latitude"}
    for extra in list(dims.keys()):
        if len(glob.glob(met_datadir+extra+"_"+date+"*"))>0:
            extra_met = xr.open_mfdataset(sorted(glob.glob(met_datadir+extra+"_"+date+"*")), combine='by_coords', parallel=True, chunks = {"level":1})

            met = xr.concat([met, extra_met], dim=dims[extra])

            if verbose: print(f"loaded extra meteorology for {extra}")

    if verbose: print("Cutting met to size")     

    met_release_idxs = get_release_idxs(fp_data_full, domain_lats = met.latitude.values, domain_lons = met.longitude.values)

    # cut the satellite meteorology for each jump in list and save at corresponding path
    if type(met_jump) == list:
        assert len(met_jump)==len(savemetpath), "pass as many savemetpaths as met_jumps!"
        for jump, savep in zip(met_jump, savemetpath):
            print(f"saving met for {region} and {date} for time jump t-{jump} hours, at path {savep}")
            _ = cut_satellite_met_v3(met, fp_data_full, size, met_release_idxs, jump, met_levels, met_variables, verbose=verbose, save=True, savepath=savep, delete_nans=True)

    if type(met_jump) == int:
        _  = cut_satellite_met_v3(met, fp_data_full, size, met_release_idxs, met_jump, met_levels, met_variables, verbose=verbose, save=True, savepath=savemetpath, delete_nans=True)


class LoadSatelliteData:
    """
    Main use: Load footprint and meteorological data for a particular domain and time period

    Secondary use: cut and save meteorological data to the right shape and format to speed up main use (this could maybe be split into a separate function!)


    main inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - month: str in format "01" for January etc, None if loading a whole year
        - region: region identifyer, as a string. Default is Brazil. Current set-up has regions "BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA"
        (note - Brazil is a subset of South America!)
        - domain: Domain related to the region, used for file search (due to existing filenaming conventions). Set-up regions ("BRAZIL", "SOUTHAMERICA", "SAHARA" and "INDIA") have a default domain, all others need domain passed
        - size: size for footprint to be cut to, as an int. Resolution of the footprint is maintained, cut to a sizexsize square around the release point. 
        Have only tested with even numbers!
        - metsize: size for the meteorology to be cut to, as an int. In most occasions metsize should be equal to size
        - freq: int, frequency of the data to load. freq=1 will load all the datapoints, freq=2 will load one in every two etc. Useful to reduce memory usage. Many datapoints are very close in time and space (and therefore very similar) so using freq particularly in low values (<10) does not affect much the quality of the dataset
        - select_time_index: list or 1D np array of timestamps to be selected as datapoints. Applied after sampling with freq (or pass freq=1 to load all footprints)
        - fp_datadir: str, directory for footprints. default directs to ACRG folder. If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass fp_datadir="/path/name_of_your_choice_"
        - met_datadir: str, directory for meteorology. default directs to ACRG meteorology folder. If passing the date will be automatically added, so the files should have format name_of_your_choice_yearmonth.nc (eg brazil_201601.nc) and you should pass met_datadir="/path/name_of_your_choice_". If passing met that has already been processed and cut, pass cut_met as false. Note that if passing processed meteorology, it does not need to be exactly the same size as metsize - the function will reshape it as long as size of loaded meteorology >= metsize
        - topog: if True, loads the topography and cuts it in the same way as the footprints
        - fill_outofdomain_with: str, out of "all_nans", "nans" and "zeros". Determines what to do if any part of the square cut around the footprint is outside of the domain. "all_nans" fills that whole footprint with nans, "nans" and "zeros" fill only the out of domain areas with nans and zeros respectively. If "all_nans" or "nans", that footprint will be eliminated from the dataset
        - verbose: if True, prints out the steps throughout the data loading process
 
    output - LoadData object with attributes:
        - input-related attributes:
            - domain,year, month, region, met_datadir, fp_datadir, size, metsize, freq
            - date combines year and month
        - footprint-related attributes:
            - fp_data_full: xr dataset with the footprints in the same format as loaded from fp_datadir, ie full domain (not cut around release point). Any invalid timestamps are deleted
            - fp_data: array of shape (time, size*size) with the footprints cut as described above. If reshaped to (time, size,size), the measurement point is always at coordinates [int(size/2), int(size/2)]
            - fp_lats and fp_lons: arrays of shape (time, size) with the latitudes and longitudes for each footprint respectively - eg footprint at fp_data[0] sits on a grid defined by fp_lats[0] and fp_lons[0]
            - release_idxs: array of shape (time, 2) with the index of the cell where the measurement location is within the full-domain array (in format latitude, longitude)
        - met-related attributes:
            - met: xr array with meteorology cut to the right size and shape. It has coordinates lat and lon with values 0-metsize, common to all footprints, with the release point at [int(metsize/2), int(metsize/2)]. coordinates lat_coords and lon_coords contain the actual lat/lon coordinates for each footprint. If size==metsize, lat_coords and lon_coords equal fp_lats and fp_lons
        - nan-related attributes: these are used to clean the data if any of the files/arrays contains nans. The following attributes are modified if there are any nans so that they all align across the time dimension: fp_data, fp_data_full, fp_lats, fp_lons, release_idxs, met
            - fp_nan_idxs: indeces in the time dimension where the footprints contains some nans
            - met_nan_idxs: indeces in the time dimension where the meteorology contains some nans

    functions:
        - get_binary_threshold(threshold=0.001, zero=0, one=1): adds attribute fp_binary, of the same shape as fp_data with a "binarisation" applied - all values above threshold are assigned value one, and all values below value zero.
    """

    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, met_jump=0, metsize=None, met_levels = [], met_variables= [], freq=1, verbose = False, met_datadir = None, fp_datadir = None, topog=None,fill_outofdomain_with="nans", select_time_index=[]):
        
        #### check domains
        if domain==None:
            domains = {"BRAZIL":"SOUTHAMERICA", "SOUTHAMERICA":"SOUTHAMERICA", "SAHARA":"NORTHAFRICA", "INDIA":"SOUTHASIA"}
            try:
                self.domain = domains[region]   
            except: 
                print("No domain was passed and there is no default domain for this site.")
                if met_datadir != None: 
                    print("Domain is not needed because met_datadir was passed")
                else:    
                    print("If domain is not passed custom path for met_datadir should be passed.")
                    self.domain=""
                   
        self.year = year
        self.date = self.year
        if month != None:
            self.month = month
            self.date = str(self.year)+month

        self.region = region
        self.met_datadir=met_datadir
        
        self.size = size
        if metsize == None:
            self.metsize = size
        else:
            self.metsize=metsize

        #### load footprint (fp) data
        if fp_datadir==None:
            fp_datadir = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/"+self.domain+"/*"+region+"*"+self.domain+"_"+str(self.date)+"*.nc"
        else:
            fp_datadir=fp_datadir+str(self.date)+"*.nc"
        if verbose: print("Loading footprint data from " + fp_datadir) 

        self.fp_data_full = load_fps(fp_datadir)

        ## reduce data frequency with regular sampling (freq parameter)
        self.freq = freq
        self.original_fp_time_length = len(self.fp_data_full.time.values)
        if freq>1:
            print(f"reduced the number of datapoints by frequency {freq}")
            self.fp_data_full = self.fp_data_full.sel(time=self.fp_data_full.time.values[::freq])

        # select only timepoints present in select_time_index list if any
        if len(select_time_index)>0:
            # TODO test 
            intersect = np.intersect1d(self.fp_data_full.time.values, select_time_index, return_indices=False)
            print(f"there are {len(select_time_index)} indeces in select_time_index, and {len(intersect)} are in the time variable of fp_data_full. Keeping those only!")
            self.fp_data_full = self.fp_data_full.sel(time=intersect)

        self.fp_data_full.load()
        
        # cut data around release point
        # fp data returned is array of shape (time, size*size) with each footprint centered around its release point
        if verbose: print("Cutting footprints to size") 

        self.fp_data, self.fp_lats, self.fp_lons, self.release_idxs = cut_satellite_data(self.fp_data_full, size, returnlatlons = True, fill_bads_with=fill_outofdomain_with, verbose=verbose)
        self.fp_data_full.close()

        #### load meteorology data
        if met_datadir==None:
            met_datadir = "/group/chemistry/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if verbose: print("Loading meteorology from " + met_datadir)
        

        # each chunk should have around 1mill values - chunk per level and by time, rounded to the nearest hundred
        time_chunk = round(1000000/(self.metsize*self.metsize), -2)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.met = xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords', parallel=True, chunks = {"level":1, "time":time_chunk})

        # load already processed met data
        assert "lat_coords" in self.met.coords, "this meteorology does not seem to have been cut! use the cut_met_data function first"

        print("met levels", met_levels)
        if len(met_levels)>0:
            try:
                self.met = self.met.sel(levels=met_levels)
            except KeyError:
                print(f"there was an error selecting the met levels you passed. Check! \n You passed  {met_levels} but met loaded has {self.met.levels}. \n Loading all levels")
        if len(met_variables)>0:
            try:
                self.met = self.met[met_variables]
            except KeyError:
                print(f"there was an error selecting the met variables you passed. Check! \n You passed  {met_variables} but met loaded has {list(self.met.keys())}. \n Loading all variables")

        # align the time dimension of footprints and met. This could be needed because freq was used, and/or because either of the original files are missing some indeces
        if len(self.met.time) != len(self.fp_data_full.time):
            self._align_time_dimensions()

        # check domain size in metfiles and match to metsize if needed
        if len(self.met.lat) > self.metsize:
            self._cut_met_to_metsize()

        elif len(self.met.lat) < self.metsize:
            print("passed met is smaller than passed metsize, so cannot cut to shape. Leaving as is - but check if this is what you want to do!")


        #### checking for nans and aligning all datasets

        ## check if any of the fp entries are nans, and if so remove from met and others
        if np.sum(np.isnan(self.fp_data_full.fp.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.fp_data_full.fp.values))[2])
            print(f"There are {len(nan_idxs)} nans in the fp data. finding and deleting from met and fp (only on axis time)")
            self.remove_indeces(nan_idxs)
            self.fp_nan_idxs = nan_idxs
        else:
            self.fp_nan_idxs = []
            

        # check if any of the met entries are nans, and if so remove from fp
        if np.sum(np.isnan(self.met.x_wind.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.met.x_wind.values[0,0,0,:])))
            print(f"There are {len(nan_idxs)} nans in the met data. finding and deleting from met and fp (only on axis time)")
            
            self.remove_indeces(nan_idxs)
            self.met_nan_idxs = nan_idxs
        else:
            self.met_nan_idxs=[]

        #### load and cut topography
        if topog is not None:
            if topog=="default":
                topog="/group/chemistry/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc"
            print(f"loading topography from {topog}")
            topog_file = xr.load_dataset(topog)

            if "onlyvalid" in self.met_datadir:
                expand_topog=True
            else:
                expand_topog=False

            self.topog = self._cut_topog(topog_file, expand_topog)

        if verbose: print("All data loaded")

    def _cut_met_to_metsize(self):
        # if the met passed (note - has to be an already processed file!) is bigger than metsize, cut to size
        # this is easy to do as processed met is centered around release point - just need to keep the square centered around this point of size metsize, metsize
        if len(self.met.lat) > self.metsize:
            print(f"passed met is bigger than the passed metsize (sizes {self.metsize, len(self.met.lat)}). Cutting met to match metsize.")
            diff = int((len(self.met.lat) - self.metsize)/2)
            cut_idxs =  self.met.lat.values[diff:-diff]
            self.met = self.met.sel(lat=cut_idxs, lon=cut_idxs)
            self.met = self.met.assign_coords({"lat":list(range(self.metsize)), "lon":list(range(self.metsize))})

    def _align_time_dimensions(self):
        # keep only time indices that are present in both the met and the footprints
        intersect, idxs1, idxs2 = np.intersect1d(self.met.time.values, self.fp_data_full.time.values, return_indices=True)
        print("met and fp_data have different timeframes! could be because either is already frequency-reduced or due to selected time indeces.")
        print(f"there are {len(self.met.time.values)} met timepoints, {len(self.fp_data_full.time.values)} footprint timepoints, and {len(intersect)} are in common. reducing both to the common points - could go wrong but shouldnt!")
        self.met = self.met.sel(time=intersect)
        self.fp_data_full = self.fp_data_full.sel(time=intersect)
        self.fp_lats = self.fp_lats[idxs2,:]
        self.fp_lons = self.fp_lons[idxs2,:]
        self.fp_data = self.fp_data[idxs2,:]
        self.release_idxs = self.release_idxs[idxs2,:]

    def _cut_topog(self, topog_file, expand_topog=False):
        # cut topography to the same domain covered by the cut footprints (ie a sizexsize square centered around measurement point). 
        # If expand_topog=False, assumes all cut footprints are within the fp domain. If expand_topog=True, loads a bigger domain of the topography to allow for footprints that escape the fp domain
        if expand_topog:
            print("expanding topography to out-of-footprint domain! careful, this is very case-specific")
            delta_lon = 0.352
            delta_lat = 0.234
            expand_by = 50
            lat_values = list(self.fp_data_full.lat.values)
            lon_values = list(self.fp_data_full.lon.values)
            lat_values = np.array(sorted(lat_values + [np.max(lat_values)+delta_lat*i for i in range(expand_by)]+ [np.min(lat_values)-delta_lat*i for i in range(expand_by)]))
            lon_values = np.array(sorted(lon_values + [np.max(lon_values)+delta_lon*i for i in range(expand_by)]+ [np.min(lon_values)-delta_lon*i for i in range(expand_by)]))
        else:
            lat_values = self.fp_data_full.lat.values
            lon_values = self.fp_data_full.lon.values
        
        # interpolate topography to domain (either fp_domain if not expand_topog, or wider if expand_topog
        self.topog_file = topog_file.interp(latitude=lat_values, longitude=lon_values)
        topog_release_idxs = get_release_idxs(self.fp_data_full, domain_lats=self.topog_file.latitude.values, domain_lons=self.topog_file.longitude.values)
        half = int(self.size/2)
        full_topog=np.zeros_like(self.fp_data)
        full_topog=np.reshape(full_topog, full_topog.shape[:-1] + (self.size, self.size))
        for rel_unique in np.unique(topog_release_idxs, axis=0):
            idxs = np.where((topog_release_idxs == rel_unique).all(axis=1))[0]  
            full_topog[idxs, :,:] = self.topog_file.surface_altitude.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half][np.newaxis, :]
        return full_topog

    def remove_indeces(self, nan_idxs):
        self.fp_data_full = self.fp_data_full.sel(time=np.delete(self.fp_data_full.time.values, nan_idxs))
        self.fp_lats = np.delete(self.fp_lats, nan_idxs, axis=0)
        self.fp_lons = np.delete(self.fp_lons, nan_idxs, axis=0)
        self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
        if self.verbose: print(f"current length: {len(self.release_idxs)}")
        self.release_idxs = np.delete(self.release_idxs, nan_idxs, axis=0)
        if self.verbose: print(f"Length after removing indeces: {len(self.release_idxs)}")
        self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))

        if hasattr(self, "topog"):
            self.topog = np.delete(self.topog, nan_idxs, axis=0)

    def get_binary_threshold(self, threshold=0.001, zero=0, one=1):
        # add binary footprint as attribute
        self.fp_binary = np.copy(self.fp_data)
        self.fp_binary[self.fp_binary<threshold] = zero
        self.fp_binary[self.fp_binary>0] = one



def intersection_over_union(fps, preds, zero=-1):
    # calculates metric intersection over union (IoU) for a binary footprint 
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero=lower number"
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    intersection = np.sum(np.logical_and(fps==1, preds==1, where=1), axis=-1)
    union = np.sum(np.logical_or(fps==1, preds==1, where=1), axis=-1)
    IoU = intersection/union
    return IoU

def accuracy(fps, preds, zero=-1):
    # calculates metric intersection over union (IoU) for a binary footprint 
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero=lower number"
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    accuracy = np.sum(fps==preds, axis=-1)/((np.shape(preds)[-1]))
    return accuracy



def dice_similarity(fps, preds, zero=-1):
    # calculates metric intersection over union (IoU) for a binary footprint 
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero="
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    TP = np.sum(np.logical_and(fps==1, preds==1), axis=-1) 
    FP = np.sum(np.logical_and(fps==zero, preds==1), axis=-1) 
    FN = np.sum(np.logical_and(fps==1, preds==zero), axis=-1)

    dice = 2*TP/(2*TP + FP + FN)
    return dice



def get_release_idxs(fp_full, domain_lats=None, domain_lons=None):
    """
    Returns array of shape (time, 2) with the indeces of the measurement point for each footprint, for either the footprint's own grid (do not pass domain_lats and domain_lons) or for another grid defined by domain_lats and domain_lons. Requires fp_full has variables release_lat and release_lon
    """
    if domain_lats is None:
        domain_lats=fp_full.lat.values
    if domain_lons is None:
        domain_lons = fp_full.lon.values
    release_idxs = []
    # get release indeces for each footprint
    for rlat, rlon in zip(fp_full.release_lat.values, fp_full.release_lon.values):
        release_lat, release_lon = min(domain_lats, key=lambda x:abs(x-rlat)), min(domain_lons, key=lambda x:abs(x-rlon))
        idx_release_lat = np.where(domain_lats == release_lat)[0][0]
        idx_release_lon = np.where(domain_lons == release_lon)[0][0]    
        release_idxs.append((idx_release_lat, idx_release_lon))
    release_idxs = np.array(release_idxs)
    return release_idxs


def cut_satellite_data(fp_full, size, returnlatlons = False, fill_bads_with="all_nans", return_as="array", verbose=True):
    """
    cuts footprint to size around release point and returns as a flattened np array (n_samples, size**2)

    Inputs:
    fp_full - full xr array with footprints. Should have variables .fp, .release_lat and .release_lon
    size - size of square to cut footprints to
    returnlatlons - bool, if True return the cut footprints, and the lats, lons and release_idxs arrays. If false, return only the cut footprints
    fill_bads_with - str, options are "all_nans", "nans" and "zeros". If "all_nans", any footprint with part of the cutting area outside of the fp_full domain is filled fully with nans. If "nans" or "zeros", only the parts out of the domain are set to "nans" or "zeros" respectively
    return_as - str, options are "array", "netcdf". If "array" returns as array of shape (time, size*size), if "netcdf" returns as an xarray with coordinates
    """
    fp_cut = np.zeros((size*size, len(fp_full.time)))
    lats = np.zeros((len(fp_full.time), size))
    lons = np.zeros((len(fp_full.time), size))

    half = int(size/2)

    release_idxs = get_release_idxs(fp_full)

    filled = 0 
    filled_sides = {"N":0, "S":0, "E":0, "W":0}
    max_sides = {"N":0, "S":0, "E":0, "W":0}

    # release indeces aren't unique so to save memory, process all footprints with same release at once
    for rel_unique in np.unique(release_idxs, axis=0):
        # find indeces across the time axis of footprints that have rel_unique as their release coordinates
        idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]
        try:
            f = fp_full.fp.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs]
            f = np.squeeze(f).reshape((size*size, len(idxs)))
            fp_cut[:,idxs] = f
            lats[idxs, :] = fp_full.lat.values[rel_unique[0]-half:rel_unique[0]+half]
            lons[idxs, :] = fp_full.lon.values[rel_unique[1]-half:rel_unique[1]+half]
        except (IndexError, ValueError): 
            # triggered if the index cannot be retrieved due to part of the cutting area being outside of the fp_full domain
            # gather stats on which direction the footprint is out of the domain and by how many cells
            if rel_unique[0]-half < 0:
                filled_sides["S"] = filled_sides["S"]+len(idxs)
            if rel_unique[0]+half > 356:
                filled_sides["N"] = filled_sides["N"]+len(idxs)
                max_sides["N"] = np.max((max_sides["N"], rel_unique[0]+half))
            if rel_unique[1]+half > 189:
                filled_sides["E"] = filled_sides["E"]+len(idxs)
                max_sides["E"] = np.max((max_sides["E"], rel_unique[1]+half))
            if rel_unique[1]-half < 0:
                filled_sides["W"] = filled_sides["W"]+len(idxs)
                max_sides["W"] = np.min((max_sides["W"], rel_unique[1]-half))
            filled+=len(idxs)

            if fill_bads_with=="all_nans":
                # make all values nan so index is removed later
                fp_cut[:,idxs] = np.nan
                fp_full.fp.values[:,:,idxs] = np.nan
            if fill_bads_with=="zeros" or fill_bads_with=="nans":
                # fill the area outside the domain with zeros/nans 
                # lower_ and upper_ are the are footprint coordinates within the domain boundaries
                lower_lat = np.max((0, rel_unique[0]-half))
                lower_lon = np.max((0, rel_unique[1]-half))
                upper_lat = np.min((len(fp_full.lat.values), rel_unique[0]+half))
                upper_lon = np.min((len(fp_full.lon.values), rel_unique[1]+half))

                # cut the part of the footprint within domain area
                f = fp_full.fp.values[lower_lat:upper_lat,lower_lon:upper_lon,idxs]

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

                f_empty = np.zeros((size, size, len(idxs)))
                if fill_bads_with=="nans":
                    f_empty[:] = np.nan

                # save the part of the footprint that is within the domain
                f_empty[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon,:] = f


                f = np.reshape(f_empty, (size*size, len(idxs)))
                fp_cut[:,idxs] = f
                lats_empty = np.zeros_like(lats[idxs, :])
                lons_empty = np.zeros_like(lons[idxs, :])

                # do the same for the latitudes and longitudes of the areas outside the domain - fill with zeros/nans
                if fill_bads_with=="nans":
                    lats_empty[:] = np.nan
                    lons_empty[:] = np.nan       

                lats_empty[:,lower_cut_lat:upper_cut_lat] = fp_full.lat.values[lower_lat:upper_lat]  
                lons_empty[:,lower_cut_lon:upper_cut_lon] =  fp_full.lon.values[lower_lon:upper_lon]
                lats[idxs, :] = lats_empty
                lons[idxs, :] = lons_empty
        
    if filled>0 and verbose:
        print(f"{filled} footprints were at least partially filled with {fill_bads_with} because they were cutting outside of the footprint file domain (this is {round(100*filled/np.shape(fp_cut)[-1], 2)}% of samples)")
        print(f"footprints that are partially out of the domain in each direction: {filled_sides}")
        print(f"maximum out-of-domain index in each direction: {max_sides}")


    fp_cut = np.transpose(fp_cut, [1,0])

    if return_as=="netcdf":
        print("preparing to return as netcdf")
        coords = {"lat":np.arange(size), "lon":np.arange(size), "time":fp_full.time.values}
        data_vars = {}
        data_vars["fp"] = (["time","lat", "lon"], np.reshape(fp_cut, (np.shape(fp_cut)[0], size, size)), fp_full.fp.attrs)
        data_vars["lat_coords"] = (["time", "lat"], lats)
        data_vars["lon_coords"] = (["time", "lon"], lons)

        fp_cut = xr.Dataset(data_vars=data_vars, coords=coords)

    if returnlatlons:
        return fp_cut, lats, lons, release_idxs
    else:
        return fp_cut 



def cut_satellite_met_v3(met, fp, metsize, release_idxs, jump=0, relevant_levels=None, relevant_variables=None, save=False, savepath=None, verbose=False, add_wind_direction=True, delete_nans=False):
    """
    cuts the meteorology to right shape and format, and either saves or returns as xarray

    Inputs:
        - met: unprocessed meteorology file
        - fp: full footprint xr array
        - metsize
        - release_idxs (as obtained with met_release_idxs = get_release_idxs(self.fp_data_full, domain_lats = met.latitude.values, domain_lons = met.longitude.values) )
        - jump: zero or positive int. If jump==0, the meteorology will be interpolated to the times of the footprints. Otherwise, the met will be interpolated to t-jump, where t is the time of the footprint
        - relevant_levels and relevant_variables: lists of levels (as ints) and variables (as str) to keep from the unprocessed met data
        - save: bool
        - savepath: str to save file to
        - add_wind_direction: bool, if True calculate wind_angle and wind_speed from the two horizontal wind vectors and add as variables
        - delete_nans: bool, if True delete timestamps where there were nans
    Returns:
        - met_cut: xarray with cut and processed meteorology
    """
    assert jump>=0, "jump needs to be zero or positive!!"

    if relevant_levels != None:
        for lev in relevant_levels:
            if lev not in met.model_level_number.values: 
                print("level ", lev, "cannot be found in the met file")
        if len(relevant_levels)==1:
            print("this is not ready for selecting only one level!")
        if verbose: print("selecting levels and loading met")
    
        met = met.sel(model_level_number=relevant_levels)
    if relevant_variables != None:
        if verbose: print("dropping irrelevant vars")
        for v in relevant_variables:
            if v not in met.data_vars:
                print("variable ", v, " not found in met file")
        vars_to_drop = list(set(list(met.data_vars))- set(relevant_variables))
        met = met.drop_vars(vars_to_drop)
    if verbose: print("loading data")


    half = int(metsize/2)
    data_vars = {}

    # interpolate the meteorology to the correct timestamps
    if jump==0:
        met = met.interp(time=fp.time.values)
    else:
        #print("before", met.time.values)
        met = met.interp(time=(pd.DatetimeIndex(fp.time.values) - pd.Timedelta(f"{jump}H")))
        #print("after", met.time.values)

    met = met.transpose("model_level_number", "latitude", "longitude", "time")

    metlats = np.zeros(( len(met.time), metsize))
    metlons = np.zeros((len(met.time), metsize))
    metlatlon = "Not Done"
    if verbose: print("creating dict for each variable")
    for varname in met.data_vars:
        if verbose: print("starting ", varname)
        var = met[varname]
        var.load()
        var = var.transpose(..., "latitude", "longitude", "time")
        if len(np.shape(var)) == 4:
            empty_arr = np.zeros((len(relevant_levels),metsize, metsize, len(met.time)))
            if metlatlon == "Not Done":
                metlatlon = "On It"
        elif len(np.shape(var)) == 3:
            empty_arr = np.zeros((metsize, metsize, len(met.time)))
        else:
            print("something is wrong?")
            print(np.shape(var))

        for rel_unique in np.unique(release_idxs, axis=0):
            idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]
            #print(rel_unique)
            if len(np.shape(var)) == 4:
                try:
                    m = var.values[:, rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs] ## assumes level, lat, lon, time
                    empty_arr[:, :, :, idxs] = m
                except Exception as e: 
                    # if it's too close to the edge, or other problems, make nan so index is removed later
                    empty_arr[:, :, :, idxs] = np.nan 
                    continue
                if metlatlon == "On It":
                    metlats[idxs, :] = met.latitude.values[rel_unique[0]-half:rel_unique[0]+half]
                    metlons[idxs, :] = met.longitude.values[rel_unique[1]-half:rel_unique[1]+half]   
            elif len(np.shape(var)) == 3: 
                try:   
                    m = var.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs]  ## assumes lat, lon, time
                    empty_arr[:,:, idxs] = m
                except Exception as e: 
                    empty_arr[:, :, idxs] = np.nan 
        if len(np.shape(var)) == 4:
            data_vars[met[varname].name] = (["levels", "lat", "lon", "time"], empty_arr, met[varname].attrs)
            if metlatlon == "On It":
                metlatlon == "Done"
        elif len(np.shape(var)) == 3:
            data_vars[met[varname].name] = (["lat", "lon", "time"], empty_arr, met[varname].attrs)           
        if verbose: print("done with ", varname)
    
    met.close()
        
    coords = {"lat":np.arange(metsize), "lon":np.arange(metsize), "time":met.time.values, "levels":relevant_levels}
    data_vars["lat_coords"] = (["time", "lat"], metlats)
    data_vars["lon_coords"] = (["time", "lon"], metlons)
    if verbose: print("creating new array with cut vars")
    met_cut = xr.Dataset(data_vars=data_vars, coords=coords)
    met_cut = met_cut.set_coords(("lat_coords", "lon_coords"))

    if add_wind_direction:
        try:
            met_cut["wind_angle"]=np.arctan2(-met_cut.x_wind,-met_cut.y_wind)
            met_cut["wind_speed"]=np.sqrt(met_cut.x_wind**2 + met_cut.y_wind**2)
        except Exception as e:
            print(f"Error {e} happened when adding wind direction and speed to met. Could be a naming error!")

    if delete_nans:
        nan_idxs = np.unique(np.where(np.isnan(met_cut.x_wind.values[0,0,0,:])))
        met_cut = met_cut.sel(time=np.delete(met_cut.time.values, nan_idxs))
        print(f"removed {len(nan_idxs)} invalid indeces")

    if save:
        print("saving met at", savepath)
        met_cut.to_netcdf(savepath)
        print("met saved")
    return met_cut


def get_all_inputs_graphnet_satellite_v4(data, variables_past, jumps, variables_nopast, others=[], topog=True, latlon_fp=0, transform=True, add_current_time=True, centered_coords=False, return_idx=False):
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
    if not (0 in jumps) and add_current_time:
        jumps.append(0) # append 0 to get present met too
    jumps=list(sorted(set(jumps)))
    print(f"hours back in time: {jumps}")

    # mets contains all of the met objects, the key is the jump
    mets={}
    mets[0] = data.met
    time_idx_nan = []
    already_updated_jumps=[]
    for j in jumps:
        if j!=0:
            met_files = glob.glob(f"{data.met_datadir}{j}h_{data.date}*")
            if len(met_files)==0:
                print(f"couldnt find files for jump {j} at {data.met_datadir}{j}h_")
            else:
                time_chunk = round(1000000/(data.size*data.size), -2)
                with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                    mets[j] = xr.open_mfdataset(met_files, parallel=True, chunks = {"level":1,"time":time_chunk})

                if len(mets[j].lat) > data.metsize:
                    print(f"file for jump {j}h is bigger than metsize, cutting down")
                    diff = int((len(mets[j].lat) - data.metsize)/2)
                    cut_idxs =  mets[j].lat.values[diff:-diff]
                    mets[j] = mets[j].sel(lat=cut_idxs, lon=cut_idxs)
                    mets[j] = mets[j].assign_coords({"lat":list(range(data.metsize)), "lon":list(range(data.metsize))})
                    print(mets[j])

                try:
                    # select met
                    mets[j] = mets[j].sel(time=(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H")))
                except KeyError:
                    print("in here")
                    # there was a problem with the time indeces - likely because the first datapoints are outside of the range
                    intersect, idxs1, idxs2 = np.intersect1d(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H"), pd.DatetimeIndex(mets[j].time.values), return_indices=True)
                    mets[j] = mets[j].sel(time=intersect)
                    already_updated_jumps.append(j)
                    time_idx_nan.append(list(set(list(range(len(data.fp_data_full.time.values)))) - set(idxs1)))

                if np.any(mets[j].x_wind.isnull()):
                    print(f"there are some nans in the met for jump {j}")
                    time_idx_nan.append(np.unique(np.where(mets[j].x_wind.isnull())[0])) 

        if "wind_speed" not in list(mets[j].keys()):
            print(f"wind speed isnt present in {j}h data, adding now")
            mets[j]["wind_angle"]=np.arctan2(-mets[j].x_wind,-mets[j].y_wind)
            mets[j]["wind_speed"]=np.sqrt(mets[j].x_wind**2 + mets[j].y_wind**2)
            print("done?")

    if len(time_idx_nan)>0:
        print(f"deleting {len(np.unique(time_idx_nan))} nan indeces (in the time axis) from jump mets and from the data object")
        for j in jumps:
            if j not in already_updated_jumps:
                mets[j] = mets[j].sel(time=np.delete(mets[j].time, np.unique(time_idx_nan)))
        
        data.fp_data = np.delete(data.fp_data, np.unique(time_idx_nan), axis=0)
        data.fp_lats = np.delete(data.fp_lats, np.unique(time_idx_nan), axis=0)
        data.fp_lons = np.delete(data.fp_lons, np.unique(time_idx_nan), axis=0)
        data.topog = np.delete(data.topog, np.unique(time_idx_nan), axis=0)
        data.met = data.met.sel(time=np.delete(data.met.time, np.unique(time_idx_nan)))
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
        for jump in jumps:
            #print(v, jump)
            mets[jump][v].load()
            for lev in variables_past[v]:
                #print(col, v, jump, lev)
                if hasattr(mets[jump][v], "levels"):
                    cutmet = mets[jump][v].sel(levels=lev).values
                    vartype="3D"
                else:
                    cutmet = mets[jump][v].values
                    vartype="2D"
                #print(mets[jump][v])
                #print(np.shape(cutmet))
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
            else:
                print(f"{oth} not recognised!")   
                _ = var_names.pop() 

              

    if topog:
        if hasattr(data, "topog"):
            all_vars[:,:,:,-1] = np.transpose(data.topog, [1,2,0])
                                             
        else:
            print("topog was passed as true but there is no topography in the data class. Load the data again using custom topog or 'default'. Last column of input will be empty ")
        
        var_names.append({"var":"topog", "type": "not met"})  


    latlons, idx_latlons = get_grid(data, latlon_fp)

    if transform:
        all_vars = np.reshape(all_vars, (np.shape(all_vars)[0]*np.shape(all_vars)[1], np.shape(all_vars)[2], np.shape(all_vars)[3]))
        all_vars = np.transpose(all_vars, [1,0,2])
    
    if return_idx:
        return latlons, idx_latlons, all_vars, var_names, data
    else:
        return latlons, all_vars, var_names, data

def get_grid(data, latlon_fp):
    """
    produce reference grid and node indeces
    """
    single_meshgrid = np.meshgrid(data.fp_lats[latlon_fp,:], data.fp_lons[latlon_fp,:])
    latlons = [(single_meshgrid[0][i,j], single_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lats)[1]) for j in range(np.shape(data.fp_lats)[1])] 

    idx_meshgrid = np.meshgrid(list(range(len(data.fp_lats[latlon_fp,:]))), list(range(len(data.fp_lons[latlon_fp,:]))))
    idx_meshgrid = np.array(idx_meshgrid)-int(data.size/2)
    idx_latlons = [(idx_meshgrid[0][i,j], idx_meshgrid[1][i,j]) for i in range(np.shape(data.fp_lats)[1]) for j in range(np.shape(data.fp_lats)[1])] 

    return latlons, idx_latlons


def grid_coordinates(side):
    xx, yy = np.meshgrid(np.array(list(range(side)), dtype=np.float32), np.array(list(range(side)), dtype=np.float32))
    z = np.empty((side**2, 2), np.float32)
    z[:, 1] = xx.reshape(side**2)
    z[:, 0] = yy.reshape(side**2)
    return z

def align_datasets_with_xr(ds1, ds2):
    """
    removes any datapoints that are not common to both DataObject ds1 and xarray ds2. returns synced ds1 and ds2
    """
    assert type(ds2) == xr.Dataset, "use this function to align data object with an array dataset. Use align_datasets for two datasets"
    try:
        intersect, idxs1, idxs2 = np.intersect1d(ds1.met.time.values, ds2.time.values, return_indices=True)
    except ValueError:
        print("There are no overlapping timestamps between there two arrays!")


    if len(intersect) == np.max((len(ds1.met.time.values), len(ds2.time.values))): # ds are already aligned, 
            print("it seems both datasets are already synced!")
            return ds1, ds2


    ds1.aligned_nan_idx = list(set(list(range(len(ds1.fp_data_full.time)))) - set(idxs1))

    ds1.fp_data_full = ds1.fp_data_full.sel(time=intersect)
    ds1.met = ds1.met.sel(time=intersect)
    ds1.fp_lats = ds1.fp_lats[idxs1,:]
    ds1.fp_lons = ds1.fp_lons[idxs1,:]
    ds1.fp_data = ds1.fp_data[idxs1,:]

    if hasattr(ds1, "topog"):
        ds1.topog = ds1.topog[idxs1,:]

    ds2 = ds2.sel(time=intersect)

    return ds1, ds2



def align_datasets(ds1, ds2, inputs1=None, inputs2=None):
    """
    removes any datapoints that are not common to both ds1 and ds2. returns synced ds1 and ds2
    """
    try:
        intersect, idxs1, idxs2 = np.intersect1d(ds1.met.time.values, ds2.met.time.values, return_indices=True)
    except ValueError:
        print("There are no overlapping timestamps between there two arrays!")


    if len(intersect) == np.max((len(ds1.met.time.values), len(ds2.met.time.values))): # ds are already aligned, are inputs?
        if inputs1 is not None and inputs2 is not None:
            if len(intersect) == np.max((len(inputs1), len(inputs2))):
                raise Exception("it seems both datasets and inputs are already synced!")
            else:
                assert hasattr(ds1, "aligned_nan_idx") and hasattr(ds2, "aligned_nan_idx"), "the two datasets are aligned but not the inputs. to align the inputs both datasets should have the aligned_nan_idx attribute, but they dont right now! something went wrong"

                inputs1 = np.delete(inputs1, ds1.aligned_nan_idx, axis=0)
                inputs2 = np.delete(inputs2, ds2.aligned_nan_idx, axis=0)
                print("datasets were aligned, aligned inputs too")

        else:
            print("it seems both datasets are already synced!")
            if inputs1 is not None:
                return ds1, ds2, inputs1, inputs2
            else:
                return ds1, ds2

    for dataset, indeces, inputs in zip([ds1, ds2], [idxs1, idxs2], [inputs1, inputs2]):

        dataset.aligned_nan_idx = list(set(list(range(len(dataset.fp_data_full.time)))) - set(indeces))


        dataset.fp_data_full = dataset.fp_data_full.sel(time=intersect)
        dataset.met = dataset.met.sel(time=intersect)
        dataset.fp_lats = dataset.fp_lats[indeces,:]
        dataset.fp_lons = dataset.fp_lons[indeces,:]
        dataset.fp_data = dataset.fp_data[indeces,:]

        if hasattr(dataset, "topog"):
            dataset.topog = dataset.topog[indeces,:]

        
        if inputs is not None:
            inputs = inputs[indeces]

    if inputs1 is not None:
        return ds1, ds2, inputs1, inputs2
    else:
        return ds1, ds2



