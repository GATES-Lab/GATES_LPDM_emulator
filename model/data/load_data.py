import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys

class LoadSatelliteData:
    """
    Load data for training and testing, for a particular site

    inputs:
        - year: can be an int (eg 2016) or a string, including combinations of years (eg "2016", "201[4-5]")
        - site: site identifyer, as a string. Default is Mace Head ("MHD")
        - siteheight: height of footprints as a string of numbers (eg "100magl"). If empty, default height is used.
        - metheight: height for met data as a string of numbers (eg "100magl"). Default is 10magl.
        - size: size for footprint to be cut to, as an int. Resolution of initial footprint is maintained, cut to a sizexsize square around the release point. 
            Should be even for computational purposes.
        - verbose: if True, prints out the steps throughout the data loading process.
        - met_datadir: Directory for .nc met data as a string, including wildcards if needed (eg "data/MHD_*"). If empty, uses default folder and naming. 
        - extramet_datadir: Directory for .nc extramet data (used for gradients, needs to be preprocessed to have some time and space resolution as met) as a string. 
            including wildcards if needed (eg "data/MHD_*"). If empty, uses default folder and naming.
        - fp_datadir: Directory for .nc footprint data as a string, including wildcards if needed (eg "data/MHD_*"). If empty, uses default folder and naming.
        Note for all three directories: If not empty, only appends year (ie need to specify or use wildcards for rest of name, including site or domain) 
        If passing wildcards from command line, need to do so in quotes.
    outputs:
        LoadData object with attributes:
        - year, site, metheight, size, metsize: details about inputs
        - met: meteorology input files, cut to size
        - fp_data_full: footprint input files
        - fp_data: flattened np array with footprint cut to size, with shape (n_samples, size**2)
        - release_lat, release_lon: coordinates of site
        - fp_lats, fp_lons: latitudes and longitudes for each cell in the cut footprint
        - temp_grad, x_wind_grad, y_wind_grad: vertical gradients extracted from extramet input files, each as 
            a flattened np array with shape (n_samples, size**2). Note last three items (timewise) are nan due to interpolation
        - y_wind and x_wind: horizontal wind vectors, transformed from input data's wind direction and speed.
    """

    def __init__(self, year, region = "BRAZIL", month=None, domain=None, size=10, met_jump=0, metsize=None, met_levels = [1], met_variables= None, freq=1, verbose = False, met_datadir = None, cut_met=True, fp_datadir = None, topog=None, savemet=False, savemetpath=None, fill_outofdomain_with="nans"):
 
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
            self.metsize = size+6
        else:
            self.metsize=metsize
        ## change this above too




        #### load footprint (fp) data
        if fp_datadir==None:
            fp_datadir = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/"+self.domain+"/*"+region+"*"+self.domain+"_"+str(self.date)+"*.nc"
        else:
            fp_datadir=fp_datadir+str(self.date)+"*.nc"
        if verbose: print("Loading footprint data from " + fp_datadir) 
        try:           
            time_chunk = round(1000000/(self.size*self.size), -2)
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                self.fp_data_full = xr.open_mfdataset(sorted(glob.glob(fp_datadir)), combine='by_coords', chunks = {"time":time_chunk})
        
        except Exception as e:
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
                                print("dropping")
                                try:
                                    f_bad = f_bad.drop(["mean_age_particles_n", "mean_age_particles_e", "mean_age_particles_w", "mean_age_particles_s"])        
                                except Exception as e:
                                    print(e)
                            bad_arrays.append(f_bad)
                    self.fp_data_full = xr.concat([most]+bad_arrays, dim="time")
            else:
                print("there was a problem", e)

        self.fp_data_full= self.fp_data_full.sortby('time')

        print(freq, len(self.fp_data_full.time.values))
        self.freq = freq
        self.original_fp_time_length = len(self.fp_data_full.time.values)
        if freq>1:
            print(f"reduced the number of datapoints by frequency {freq}")
            self.fp_data_full = self.fp_data_full.sel(time=self.fp_data_full.time.values[::freq])

        if not cut_met:
            self.fp_data_full.load()
            print(len(self.fp_data_full.time.values))
            
            # cut data around release point
            # fp data returned is array of shape (time, size*size) with each footprint centered around its release point
            if verbose: print("Cutting footprints to size") 
            # reduce number of samples 


            self.fp_data, self.fp_lats, self.fp_lons, self.release_idxs = cut_satellite_data(self.fp_data_full, size, returnlatlons = True, fill_bads_with=fill_outofdomain_with)
            self.fp_data_full.close()
            print(np.shape(self.fp_data), len(self.fp_data_full.time.values))
        
        if cut_met:
            self.fp_data_full.lat.load()
            self.fp_data_full.lon.load()
            self.fp_data_full.release_lat.load()
            self.fp_data_full.release_lon.load()
            self.fp_data_full.time.load()
            self.release_idxs = get_release_idxs(self.fp_data_full)


        #### load meteorology data
        if met_datadir==None:
            met_datadir = "/group/chemistry/acrg/met_archive/UM/"+self.domain+"/"+self.domain+"_Met_"+str(self.date)+"*.nc"
        else:
            met_datadir = met_datadir+str(self.date)+"*.nc"
        if verbose: print("Loading meteorology from " + met_datadir)
        

        # each chunk should have around 1mill values - chunk per level and by time, rounded to the nearest hundred
        time_chunk = round(1000000/(self.size*self.size), -2)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            met = xr.open_mfdataset(sorted(glob.glob(met_datadir)), combine='by_coords', parallel=True, chunks = {"level":1, "time":time_chunk})

        if not cut_met:
            # load already processed met data, check that size matches and fix/warn if not
            print("not cutting met, just checking")

            # if not cut met, met should have already been processed and saved 
            assert "lat_coords" in met.coords, "cut_met was passed as false but passed met does not have the right format!"
            self.met = met
            

            if len(self.met.time) != len(self.fp_data_full.time):
                if len(self.met.time) == self.original_fp_time_length:
                    print("reducing time frequency of met too")
                    self.met = self.met.interp(time=self.fp_data_full.time, method="nearest")
                else:
                    intersect, idxs1, idxs2 = np.intersect1d(self.met.time.values, self.fp_data_full.time.values, return_indices=True)
                    print("met and fp_data have different timeframes! could be because either is already frequency-reduced.")
                    print(f"there are {len(self.met.time.values)} met timepoints, {len(self.fp_data_full.time.values)} footprint timepoints, and {len(intersect)} are in common. reducing both to the common points - could go wrong!")
                    self.met = self.met.sel(time=intersect)
                    self.fp_data_full = self.fp_data_full.sel(time=intersect)
                    self.fp_lats = self.fp_lats[idxs2,:]
                    self.fp_lons = self.fp_lons[idxs2,:]
                    self.fp_data = self.fp_data[idxs2,:]
                    self.release_idxs = self.release_idxs[idxs2,:]

            if len(self.met.lat) == self.metsize:
                print("cut_met was passed as false. Met will be used as is")
                print("Note that because met has been passed pre-cut, if fp and original met had different sizes fp will not be cut!")


            elif len(self.met.lat) > self.metsize:
                print(f"passed met does not match the passed metsize (sizes {self.metsize, len(self.met.lat)}). Cutting to match.")
                diff = int((len(self.met.lat) - self.metsize)/2)
                cut_idxs =  self.met.lat.values[diff:-diff]
                self.met = self.met.sel(lat=cut_idxs, lon=cut_idxs)
                self.met = self.met.assign_coords({"lat":list(range(self.metsize)), "lon":list(range(self.metsize))})

            elif len(self.met.lat) < self.metsize:
                print("passed met is smaller than passed metsize, so cannot cut to shape. Leaving as is")
            


        """
        elif cut_met:
            # if using original met, check that domain is same as fp and cut fp if not
            if len(met.latitude.values) < len(self.fp_data_full.lat.values) or len(met.longitude.values) < len(self.fp_data_full.lon.values):
                print("met is smaller than footprint, likely because the footprint's domain is unncecessarily big. Cutting fp")
                self.fp_data_full = self.fp_data_full.interp(lat=met.latitude.values, lon=met.longitude.values, method="nearest")
        """

        # cut met around release points + only correct levels and variables
        if cut_met:
            if verbose: print("Cutting met to size") 
            if type(met_jump) == list:
                for jump, savep in zip(met_jump, savemetpath):
                    print(jump, savep)
                    self.met = cut_satellite_met_v3(met, self.fp_data_full, self.metsize, self.release_idxs, jump, met_levels, met_variables, verbose=verbose, save=savemet, savepath=savep, delete_nans=True)
                print("exiting the program after saving met for different jumps!")
                #print("no invalid indeces (if any) were removed")
                # just generating met
                return None
            else:
                ## here load met with variables one by one
                #self.met = cut_satellite_met_v2(met, self.fp_data_full, self.metsize, self.release_idxs, met_jump, met_levels, met_variables, verbose=verbose, save=savemet, savepath=savemetpath)

                self.met = cut_satellite_met_v3(met, self.fp_data_full, self.metsize, self.release_idxs, met_jump, met_levels, met_variables, verbose=verbose, save=savemet, savepath=savemetpath, delete_nans=False)


        ## check if any of the fp entries are nans, and if so remove from met
        print(np.shape(self.fp_data), len(self.met.time))
        if np.sum(np.isnan(self.fp_data_full.fp.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.fp_data_full.fp.values))[2])
            print(f"There are {len(nan_idxs)} nans in the fp data. finding and deleting from met and fp (only on axis time)")
            self.fp_data_full = self.fp_data_full.sel(time=np.delete(self.fp_data_full.time.values, nan_idxs))
            self.fp_lats = np.delete(self.fp_lats, nan_idxs, axis=0)
            self.fp_lons = np.delete(self.fp_lons, nan_idxs, axis=0)
            self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
            print(len(self.release_idxs))
            self.release_idxs = np.delete(self.release_idxs, nan_idxs, axis=0)
            print(len(self.release_idxs))
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))

            self.fp_nan_idxs = nan_idxs
        else:
            self.fp_nan_idxs = []
            
        print(np.shape(self.fp_data), len(self.fp_data_full.time.values))
        # check if any of the met entries are nans, and if so remove from fp
        if np.sum(np.isnan(self.met.x_wind.values)) != 0:
            nan_idxs = np.unique(np.where(np.isnan(self.met.x_wind.values[0,0,0,:])))
            print(f"There are {len(nan_idxs)} nans in the met data. finding and deleting from met and fp (only on axis time)")
            
            self.met = self.met.sel(time=np.delete(self.met.time.values, nan_idxs))
            self.fp_data_full = self.fp_data_full.sel(time=np.delete(self.fp_data_full.time.values, nan_idxs))
            self.fp_lats = np.delete(self.fp_lats, nan_idxs, axis=0)
            self.fp_lons = np.delete(self.fp_lons, nan_idxs, axis=0)
            self.fp_data = np.delete(self.fp_data, nan_idxs, axis=0)
            self.release_idxs = np.delete(self.release_idxs, nan_idxs, axis=0)
            self.met_nan_idxs = nan_idxs
        else:
            self.met_nan_idxs=[]
        print(np.shape(self.fp_data), len(self.fp_data_full.time.values))
        print(len(self.release_idxs))

        if topog is not None:
            if topog=="default":
                topog="/group/chemistry/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc"
            print(f"loading topography from {topog}")
            topog_file = xr.load_dataset(topog)
            self.topog_file = topog_file.interp(latitude=self.fp_data_full.lat.values, longitude=self.fp_data_full.lon.values)
            half = int(self.size/2)
            full_topog=np.zeros_like(self.fp_data)
            full_topog=np.reshape(full_topog, full_topog.shape[:-1] + (self.size, self.size))
            for rel_unique in np.unique(self.release_idxs, axis=0):
                idxs = np.where((self.release_idxs == rel_unique).all(axis=1))[0]  
                full_topog[idxs, :,:] = self.topog_file.surface_altitude.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half][np.newaxis, :]
            self.topog=np.copy(full_topog)



        if verbose: print("All data loaded")

    def get_binary_threshold(self, threshold=0.001, zero=0, one=1):
        self.fp_binary = np.copy(self.fp_data)
        self.fp_binary[self.fp_binary<threshold] = zero
        self.fp_binary[self.fp_binary>0] = one

def intersection_over_union(fps, preds, zero=-1):
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero="
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    intersection = np.sum(np.logical_and(fps==1, preds==1, where=1), axis=-1)
    union = np.sum(np.logical_or(fps==1, preds==1, where=1), axis=-1)
    IoU = intersection/union

    return IoU

def dice_similarity(fps, preds, zero=-1):
    assert (np.unique(fps) == np.array([zero,1])).all(), "pass binary footprints, or if they arent -1/1, pass parameter zero="
    assert len(np.shape(fps))<=2, "currently this only supports flattened arrays (of shape (samples x pixels))"
    TP = np.sum(np.logical_and(fps==1, preds==1), axis=-1) 
    FP = np.sum(np.logical_and(fps==zero, preds==1), axis=-1) 
    FN = np.sum(np.logical_and(fps==1, preds==zero), axis=-1)

    dice = 2*TP/(2*TP + FP + FN)
    return dice


def cut_met(met, release_lat, release_lon, size):
    ## cuts meteorology to size around release point and returns as a smaller xarray
    release_lat, release_lon = min(met.lat.values, key=lambda x:abs(x-release_lat)), min(met.lon.values, key=lambda x:abs(x-release_lon))
    idx_release_lat = np.where(met.lat.values == release_lat)[0][0]
    idx_release_lon = np.where(met.lon.values == release_lon)[0][0]
    half = int(size/2)
    lats = met.lat.values[idx_release_lat-half:idx_release_lat+half]
    lons = met.lon.values[idx_release_lon-half:idx_release_lon+half]
    met = met.sel({"lat":lats, "lon":lons}).compute()
    return met

def cut_data(fp_full, release_lat, release_lon, size, returnlatlons = False):
    ## cuts footprint to size around release point and returns as a flattened np array (n_samples, size**2)
    release_lat, release_lon = min(fp_full.lat.values, key=lambda x:abs(x-release_lat)), min(fp_full.lon.values, key=lambda x:abs(x-release_lon))
    idx_release_lat = np.where(fp_full.lat.values == release_lat)[0][0]
    idx_release_lon = np.where(fp_full.lon.values == release_lon)[0][0]    
    half = int(size/2)
    lats = fp_full.lat.values[idx_release_lat-half:idx_release_lat+half]
    lons = fp_full.lon.values[idx_release_lon-half:idx_release_lon+half]
    
    data = fp_full.sel({"lat":lats, "lon":lons}).fp.values

    data = data.reshape((np.shape(data)[0]*np.shape(data)[1], np.shape(data)[2]))
    data = np.transpose(data, [1,0])
        
    if returnlatlons:
        return data, lats, lons
    else:
        return data      

def get_release_idxs(fp_full):
    release_idxs = []
    # get release indeces for each footprint
    for rlat, rlon in zip(fp_full.release_lat.values, fp_full.release_lon.values):
        release_lat, release_lon = min(fp_full.lat.values, key=lambda x:abs(x-rlat)), min(fp_full.lon.values, key=lambda x:abs(x-rlon))
        idx_release_lat = np.where(fp_full.lat.values == release_lat)[0][0]
        idx_release_lon = np.where(fp_full.lon.values == release_lon)[0][0]    
        release_idxs.append((idx_release_lat, idx_release_lon))
    release_idxs = np.array(release_idxs)
    return release_idxs


def cut_satellite_data(fp_full, size, returnlatlons = False, fill_bads_with="all_nans", return_as="array"):
    ## cuts footprint to size around release point and returns as a flattened np array (n_samples, size**2)
    fp_cut = np.zeros((size*size, len(fp_full.time)))
    lats = np.zeros(( len(fp_full.time), size))
    lons = np.zeros((len(fp_full.time), size))

    half = int(size/2)

    release_idxs = get_release_idxs(fp_full)

    filled = 0 
    filled_sides = {"N":0, "S":0, "E":0, "W":0}
    max_sides = {"N":0, "S":0, "E":0, "W":0}
    # release indeces aren't unique so to save memory, process all footprints with same release at once
    for rel_unique in np.unique(release_idxs, axis=0):
        idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]
        try:
            f = fp_full.fp.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs]
            f = np.squeeze(f).reshape((size*size, len(idxs)))
            fp_cut[:,idxs] = f
            lats[idxs, :] = fp_full.lat.values[rel_unique[0]-half:rel_unique[0]+half]
            lons[idxs, :] = fp_full.lon.values[rel_unique[1]-half:rel_unique[1]+half]
        except Exception as e: 
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
                # if it's too close to the edge, or other problems, make all values nan so index is removed later
                fp_cut[:,idxs] = np.nan
                fp_full.fp.values[:,:,idxs] = np.nan
            if fill_bads_with=="zeros" or fill_bads_with=="nans":
                #print(f"indeces: {rel_unique}, lat release cords {rel_unique[0]-half,rel_unique[0]+half},lon release coords {rel_unique[1]-half,rel_unique[1]+half}, size: {len(fp_full.lat.values), len(fp_full.lon.values)}")
                # if it's too close to the edge, make zeros/nans the area outside the edge 
                # lower_ and upper_ are the are footprint coordinates within the domain boundaries
                lower_lat = np.max((0, rel_unique[0]-half))
                lower_lon = np.max((0, rel_unique[1]-half))
                upper_lat = np.min((len(fp_full.lat.values), rel_unique[0]+half))
                upper_lon = np.min((len(fp_full.lon.values), rel_unique[1]+half))

                # cut footprint within domain area
                f = fp_full.fp.values[lower_lat:upper_lat,lower_lon:upper_lon,idxs]
                #print(f"coordinates to cut within domain {lower_lat, upper_lat, lower_lon, upper_lon}, shape of cut area {np.shape(f)}")
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
                #print(f"coordinates within footprint domain {lower_cut_lat, upper_cut_lat, lower_cut_lon, upper_cut_lon}")
                f_empty = np.zeros((size, size, len(idxs)))
                if fill_bads_with=="nans":
                    f_empty[:] = np.nan

                #print(f"sshape of empty: {np.shape(f_empty)}, shape of empty to replace with f {np.shape(f_empty[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon,:])}")
                f_empty[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon,:] = f
                #print(np.shape(np.squeeze(f_empty)))

                f = np.reshape(f_empty, (size*size, len(idxs)))
                fp_cut[:,idxs] = f
                lats_empty = np.zeros_like(lats[idxs, :])
                lons_empty = np.zeros_like(lons[idxs, :])

                if fill_bads_with=="nans":
                    lats_empty[:] = np.nan
                    lons_empty[:] = np.nan       

                lats_empty[:,lower_cut_lat:upper_cut_lat] = fp_full.lat.values[lower_lat:upper_lat]  
                #print(np.shape(lons_empty), lower_cut_lon, upper_cut_lon, np.shape(lons_empty[lower_cut_lon:upper_cut_lon]), np.shape( fp_full.lon.values[lower_lon:upper_lon] ))
                lons_empty[:,lower_cut_lon:upper_cut_lon] =  fp_full.lon.values[lower_lon:upper_lon]
                lats[idxs, :] = lats_empty
                lons[idxs, :] = lons_empty
        
    if filled>0:
        print(f"{filled} footprints were at least partially filled with {fill_bads_with} because they were cutting outside of the footprint file domain (this is {round(100*filled/np.shape(fp_cut)[-1], 2)}% of samples)")
        print(filled_sides)
        print(max_sides)



    
    fp_cut = np.transpose(fp_cut, [1,0])

    if return_as=="netcdf":
        print("preparing to return as netcdf")
        #print(np.shape(fp_cut))
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


def cut_satellite_data_full_domain(fp_full, size, returnlatlons = False, fill_bads_with="nans", return_as="array"):
    print("in here")
    ## cuts footprint to size around release point and returns as a flattened np array (n_samples, size**2)
    fp_cut = xr.zeros_like(fp_full)
    lats = np.zeros(( len(fp_full.time), size))
    lons = np.zeros((len(fp_full.time), size))
    release_idxs = []
    # get release indeces for each footprint
    for rlat, rlon in zip(fp_full.release_lat.values, fp_full.release_lon.values):
        release_lat, release_lon = min(fp_full.lat.values, key=lambda x:abs(x-rlat)), min(fp_full.lon.values, key=lambda x:abs(x-rlon))
        idx_release_lat = np.where(fp_full.lat.values == release_lat)[0][0]
        idx_release_lon = np.where(fp_full.lon.values == release_lon)[0][0]    
        release_idxs.append((idx_release_lat, idx_release_lon))
    release_idxs =np.array(release_idxs)
    half = int(size/2)

    filled = 0 
    # release indeces aren't unique so to save memory, process all footprints with same release at once
    for rel_unique in np.unique(release_idxs, axis=0):
        idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]
        try:
            f = fp_full.fp.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs]
            fp_cut.fp.values[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs] = f
        except Exception as e: 
            filled+=len(idxs)
            if fill_bads_with=="zeros" or fill_bads_with=="nans":
                #print(f"indeces: {rel_unique}, lat release cords {rel_unique[0]-half,rel_unique[0]+half},lon release coords {rel_unique[1]-half,rel_unique[1]+half}, size: {len(fp_full.lat.values), len(fp_full.lon.values)}")
                # if it's too close to the edge, make zeros/nans the area outside the edge 
                # lower_ and upper_ are the are footprint coordinates within the domain boundaries
                lower_lat = np.max((0, rel_unique[0]-half))
                lower_lon = np.max((0, rel_unique[1]-half))
                upper_lat = np.min((len(fp_full.lat.values), rel_unique[0]+half))
                upper_lon = np.min((len(fp_full.lon.values), rel_unique[1]+half))

                # cut footprint within domain area
                f = fp_full.fp.values[lower_lat:upper_lat,lower_lon:upper_lon,idxs]
                #print(f"coordinates to cut within domain {lower_lat, upper_lat, lower_lon, upper_lon}, shape of cut area {np.shape(f)}")
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
                #print(f"coordinates within footprint domain {lower_cut_lat, upper_cut_lat, lower_cut_lon, upper_cut_lon}")
                f_empty = np.zeros((np.shape(fp_full[:,:,0])[0], np.shape(fp_full[:,:,0])[1], len(idxs)))

                if fill_bads_with=="nans":
                    f_empty[:] = np.nan

                #print(f"sshape of empty: {np.shape(f_empty)}, shape of empty to replace with f {np.shape(f_empty[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon,:])}")
                f_empty[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon,:] = f
                #print(np.shape(np.squeeze(f_empty)))

                fp_cut[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs] = f
        

    #print(f"{filled} footprints were at least partially filled with {fill_bads_with} because they were cutting outside of the footprint file domain (this is {round(100*filled/np.shape(fp_cut)[-1], 2)}% of samples)")


    if returnlatlons:
        return fp_cut, lats, lons, release_idxs
    else:
        return fp_cut 


def cut_satellite_met_v3(met, fp, metsize, release_idxs, jump=0, relevant_levels=None, relevant_variables=None, save=False, savepath=None, verbose=False, add_wind_direction=True, delete_nans=False):
    # it is NOT assumed that the relevant levels == levels the met has been cut to

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

    # method 1: interpolating met to fp times (alternative method: take met for timestep closest to fp)
    if jump==0:
        met = met.interp(time=fp.time.values)
    else:
        met = met.interp(time=(pd.DatetimeIndex(fp.time.values) - pd.Timedelta(f"{jump}H")))

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
    Returns all input variables stacked and ready to pass to regressors, with shape (n_samples-jump-3, (2*#variables_past+#variables_nopast)**2)

    Takes a dict of variables that are passed at time of footprint and A LIST OF INTS WITH THE JUMP hours before, with their chosen levels
    and a list of variables that are passed only at time of footprint with their chosen levels. 
    Last three items of all variables are removed, due to interpolation setup.
    If var has no levels, pass 0 as level
    equivalent to the above if passing only a single jump in the list (eg [6])
    """
    all_vars = []
    var_names = []
    if not (0 in jumps) and add_current_time:
        jumps.append(0) # to do present
    jumps=list(sorted(set(jumps)))
    print(f"hours back in time: {jumps}")

    print(len(data.met.time))


    mets={}
    mets[0] = data.met
    time_idx_nan = []
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

                mets[j] = mets[j].sel(time=(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H")))
                """
                if data.freq != 1:
                    # interpolate here just removes any time indeces not present in time data
                    # as the processed files already have the correct timestamps
                    mets[j] = mets[j].interp(time=(pd.DatetimeIndex(data.fp_data_full.time.values) - pd.Timedelta(f"{j}H")), method="nearest")
                     # could probably replace this below with an interpolation too
                else:
                    if hasattr(data, "fp_nan_idxs") and len(data.fp_nan_idxs)>0:
                        print(f"deleting met entries that have fp nans in jump {j}h")
                        print(np.max(data.fp_nan_idxs), len(data.fp_nan_idxs))
                        mets[j] = mets[j].sel(time=np.delete(mets[j].time.values, data.fp_nan_idxs))
                        print(len(mets[j].time))
                    if hasattr(data, "met_nan_idxs") and len(data.met_nan_idxs)>0:
                        print(f"deleting met entries that have met nans in jump {j}h")
                        print(np.max(data.met_nan_idxs))
                        mets[j] = mets[j].sel(time=np.delete(mets[j].time.values, data.met_nan_idxs))
                    if hasattr(data, "aligned_nan_idx") and len(data.aligned_nan_idx)>0:
                        print(f"deleting met and fp entries entries in jump {j}h that were removed during aligning datasets")
                        mets[j] = mets[j].sel(time=np.delete(mets[j].time.values, data.aligned_nan_idx))
                        print(len(mets[j].time))
                """
                if np.any(mets[j].x_wind.isnull()):
                    print(f"there are some nans in the met for jump {j}")
                    time_idx_nan.append(np.unique(np.where(mets[j].x_wind.isnull())[0])) 

        #print(mets[j].keys())
        if "wind_speed" not in list(mets[j].keys()):
            print(f"wind speed isnt present in {j}h data, adding now")
            mets[j]["wind_angle"]=np.arctan2(-mets[j].x_wind,-mets[j].y_wind)
            mets[j]["wind_speed"]=np.sqrt(mets[j].x_wind**2 + mets[j].y_wind**2)
            print("done?")
            #print(mets[j]["wind_speed"])

    if len(time_idx_nan)>0:
        print(f"deleting {len(np.unique(time_idx_nan))} nan indeces (in the time axis) from jump mets and from the data object")
        for j in jumps:
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
        print(j, len(mets[j].time))
        

    n_vars_with_past =(len(jumps))*np.sum([len(variables_past[var]) for var in variables_past])
    if len([len(variables_nopast[var]) for var in variables_nopast])==0:
        n_vars_no_past=0
    else:
        n_vars_no_past =np.sum([len(variables_nopast[var]) for var in variables_nopast]) 
    
    n_variables = n_vars_with_past+n_vars_no_past+len(others)+topog

    if "relative_time" in others:
        n_variables = n_variables + len(jumps) -  1 
            ## as relative time adds one uniform variable to all nodes to signpost time of the inputs with respect to release - ie met at release will have variable with value 0, six hours before will have value 6 etc
    
    if "normalised_time_of_year" in others:
        n_variables = (n_variables-1) + 2*len(jumps)   
    if "normalised_time_of_day" in others:
        n_variables = (n_variables-1) + 2*len(jumps)  

    all_vars = np.zeros((np.shape(data.fp_lats)[1], np.shape(data.fp_lons)[1], len(data.met.time), n_variables))


    
    col = 0
    for v in variables_past:
        for jump in jumps:
            mets[jump][v].load()
            for lev in variables_past[v]:
                #print(col, v, jump, lev)
                if hasattr(mets[jump][v], "levels"):
                    cutmet = mets[jump][v].sel(levels=lev).values
                    vartype="3D"
                else:
                    cutmet = mets[jump][v].values
                    vartype="2D"
                
                #print(np.shape(cutmet), np.shape(all_vars[:,:,:,col]))
                #print(np.shape(cutmet), np.shape(all_vars), v, jump, lev)
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
            ## add option for  solar radiation, orography, land-sea mask, the day-of-year
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



