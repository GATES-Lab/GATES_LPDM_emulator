import sys

#sys.path.insert(0,"/software/local/languages/miniforge3/envs/tensorflow/lib/python3.12/site-packages")
sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")
sys.path.insert(0,"/user/work/ef17148/oldstuff/ef17148/.conda/envs/new_graphnet/lib/python3.12/site-packages")

#import cartopy
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import einops
import numpy as np
import torch
import os
import pickle
import random

from model.evaluation import *
from model.data.dataloader_graphnet import *
from model.data.load_data import *
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import argparse
import json


"""
After footprints are predicted on a square of size sizexsize, they should be "integrated" back into the footprint domain so they can be used in the inversion. we also apply the bias correction at this stage

this file can probably do with some improvements!
"""

parser = argparse.ArgumentParser(description="Load parameters")

parser.add_argument("--months", help="model to use to generate fps", default="all")
parser.add_argument("--name_footprint_path", help="path to original NAME footprints", default=None)
parser.add_argument("--emulated_footprints_path", help="path to folder where to save fps", default=None)
parser.add_argument("--model_name", help="model to use to generate fps", default=None)


# set up paths

args = parser.parse_args()
months=args.months 

name_footprint_path = args.name_footprint_path
if name_footprint_path is None:
    name_footprint_path = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/SOUTHAMERICA/GOSAT-BRAZIL-column_SOUTHAMERICA_"

emulated_footprints_path = args.emulated_footprints_path
if emulated_footprints_path is None:
    emulated_footprints_path = "/group/chemistry/acrg/LPDM/fp_Elena/satellite_emulated_logv4/def_lc_biascorrected/SOUTHAMERICA/GOSAT-BRAZIL-column_SOUTHAMERICA_"
    # DID YOU CREATE A SOUTHAMERICA FOLDER???

model_name = args.model_name
if model_name is None:
    model_name = "satellite_clever2_200_[6,12]_vB_logv4_weightedbytruth_B_landcover"
    #model_name = "satellite_clever2_200_[6,12]_vB_logv4_relu_weightedbytrutha05_B"

month="08"
test_year="2016"

size=200
half = 100

bias_correct = True
fixed_bias=(False, 1.05)

#months = [ "04", "05", "06", "07", "08", "09", "10", "11", "12"]
if type(months)==str and months=="all":
    months = ["01","02", "03", "04", "05", "06", "07", "08","09", "10", "11", "12"]
elif type(months)==str:
    months = int(months)
if type(months) == int:
    months=[f"{months:02}"]

for month in months:
    try:
        # load preds and bias correct
        ## currently validating on Jan-March, but using all for emissions inference
        preds_processed = ModelEv(model_name, size=200, test_set=[month], check_validation_overlap=False,test_year=test_year, path_to_files="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_NORTHAFRICA/")
        
        preds_processed.get_adjusted_models(which="thr_only") 

        preds = preds_processed.preds_test.copy()
        

        if bias_correct:
            q = 5000
            thr = 1
            print("bias correcting")
            biascorrected_preds = quantile_mapping_interp(preds_processed.preds_validation.fp.values, preds_processed.preds_validation.trans_predictions.values, to_correct=preds_processed.preds_test.trans_predictions.values, mode="replace", n_quantiles=q, mult_factor=thr)
            preds["corrected"] = (("time", "lat", "lon"), biascorrected_preds)


        elif fixed_bias[0]:
            preds["corrected"] = (("time", "lat", "lon"), fixed_bias[1]*preds_processed.variations["thr"])

        else:
            preds["corrected"] = (("time", "lat", "lon"), preds_processed.variations["thr"])

        preds = preds.transpose("lat", "lon", "time")
        print("loading original footprints and replacing")
        # load original footprints to copy and align datasets
        original_fps = xr.open_dataset(f"{name_footprint_path}{test_year}{month}.nc")

        # match the two in time - assume emulated is a subset
        try:
            original_fps = original_fps.sel(time=preds.time)
        except KeyError: # if not a subset for whatever reason...
            inters = np.intersect1d(original_fps.time.values, preds.time.values)
            original_fps = original_fps.sel(time=inters)

            print(f"not all timesteps in model are found in the original footprints, so cutting both. losing {len(preds.time) - len(inters)} timepoints from model and predicted fps!")
            preds = preds.sel(time=inters)     



        release_idxs = get_release_idxs(original_fps)

        emulated_fps = np.zeros_like(original_fps.fp.values) # lat lon time
        #emulated_fps = np.copy(original_fps.fp.values)  ## to do integrated footprints

        for rel_unique in np.unique(release_idxs, axis=0):
            # find indeces across the time axis of footprints that have rel_unique as their release coordinates
            idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]

            fps_here = preds.corrected.values[:,:,idxs]
            if np.shape(emulated_fps[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half, idxs[0]]) == (size,size): # changed 0 for idxs[0] above... think that's fine?
                emulated_fps[rel_unique[0]-half:rel_unique[0]+half, rel_unique[1]-half:rel_unique[1]+half,idxs] = fps_here
            else:
                # footprint is pratially outside of domain
                # 1. define the area of the emulated footprints to keep
                lower_cut_lat = np.max((0, -(rel_unique[0]-half)))
                lower_cut_lon = np.max((0, -(rel_unique[1]-half)))
                upper_cut_lat = np.min(((rel_unique[0]+half)-len(original_fps.lat.values), size))
                if (rel_unique[0]+half)-len(original_fps.lat.values)>0: 
                    upper_cut_lat=size-((rel_unique[0]+half)-len(original_fps.lat.values))
                else: upper_cut_lat=size
                if (rel_unique[1]+half)-len(original_fps.lon.values)>0: 
                    upper_cut_lon=size-((rel_unique[1]+half)-len(original_fps.lon.values))
                else: upper_cut_lon=size  

                fps_here = preds.corrected.values[lower_cut_lat:upper_cut_lat,lower_cut_lon:upper_cut_lon, idxs]

                # 2. define area of the original footprints to replace 
                lower_lat = np.max((0, rel_unique[0]-half))
                lower_lon = np.max((0, rel_unique[1]-half))
                upper_lat = np.min((len(original_fps.lat.values), rel_unique[0]+half))
                upper_lon = np.min((len(original_fps.lon.values), rel_unique[1]+half)) 

                # 3. replace
                emulated_fps[lower_lat:upper_lat,lower_lon:upper_lon, idxs] = fps_here

        # replace 
        full_emulated_footprints = original_fps.copy()

        full_emulated_footprints["fp"] = (("lat", "lon", "time"), emulated_fps)

        # save
        full_emulated_footprints.attrs["author"] = "ef17148"
        full_emulated_footprints.attrs["emulation_model"] = model_name
        full_emulated_footprints.attrs["met"] = "all fps - met outside of domain extended"
        full_emulated_footprints.attrs["bias_corrected"] = str(bias_correct)
        full_emulated_footprints.to_netcdf(f"{emulated_footprints_path}{test_year}{month}.nc")
        print("done! saved at ", f"{emulated_footprints_path}{test_year}{month}.nc")

    except Exception as e:
        print(f"month {month} failed with error {e}")