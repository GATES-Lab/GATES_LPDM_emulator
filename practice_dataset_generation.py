import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import einops
import numpy as np
import torch
import os
import pickle
import random

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import *
from model.data.load_data import *
from model.utils import parse_years, baseline_mol_updated, write_to_file

from model.forecast import GraphSatelliteForecaster, GraphSatelliteForecasterClassifier, GraphSatelliteForecasterConvClassifier
from model.loss_functions import *

import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse

import random
from datetime import date

def baseline_mol_updated_times(desired_data,months,years):
    '''
    Function used to get the value for the input and output
    '''
    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]                                                                      
    # Load the CSV file
    #df = pd.read_csv('Analysis/CH4_Semihemispheric_modelled_mole_fractions.csv')
    df = pd.read_csv('/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv')
    baseline_list = np.zeros((total_data_points,4))

    '''
    # Specify the year and month you're interested in
    # Filter the data for the specific year and month
    filtered_data = df[(df['Year'] == specific_year) & (df['Month'] == specific_month)]

    # Extract the 4th, 5th, 6th, and 7th columns
    selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values

    # Display the results
    print(selected_columns)
    '''

    # Iterate through the different numbers 
    #months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    #year = 2016
    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)
    specific_years = np.array([np.datetime64(date, 'Y').astype(int) + 1970 for date in datetime_array])
    specific_months = np.array([np.datetime64(date, 'M').astype(int) % 12 + 1 for date in datetime_array])

    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]
    north_list = np.zeros(total_data_points) # Creates a list of size N with None values
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)
    specific_times_list = np.zeros(total_data_points)
    for year in years:
        print('year',year)
        for month in months:
            # Cams field for a particular month
            if year > 2017:
                # Change made due to the naming of the data
                cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc")
            else:
                cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc")    
            # Extract year and month
            #import ipdb; ipdb.set_trace()
            

            # Desired year and month
            #desired_year = 2016
            desired_year = year
            desired_month = month
            print('desired month',desired_month)
            
            '''
            coarse_cams  = cams.coarsen(lon=desired_data.coarsening_factor, lat=desired_data.coarsening_factor, boundary="pad").mean()
            '''
            # Find the index of the first occurrence
            indices = np.where((specific_years == desired_year) & (specific_months == int(desired_month)))[0]
            print(indices)
            if len(indices)>0:
                # Get the inputs
                filtered_data = df[(df['Year'] == desired_year) & (df['Month'] == int(desired_month))]
                # Extract the 4th, 5th, 6th, and 7th columns
                selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values
                baseline_list[indices] = selected_columns/1000 # Convert from parts per trillion to parts per million

                print(indices[0])
                # Multply the first value with all the other values of the array
                print(cams.vmr_n.shape)
                # CAMS field should be stationary over the period of a month
                #import ipdb; ipdb.set_trace()
                import ipdb; ipdb.set_trace()
                north_mol = np.sum(cams.vmr_n * desired_data.fp_data_full.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(cams.vmr_s * desired_data.fp_data_full.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(cams.vmr_e * desired_data.fp_data_full.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(cams.vmr_w * desired_data.fp_data_full.particle_locations_w[:,:,indices], axis=(0,1))
                specific_times = datetime_array[indices]
                
                #import ipdb; ipdb.set_trace()
                north_list[indices] = north_mol
                south_list[indices] = south_mol
                east_list[indices] = east_mol
                west_list[indices] = west_mol
                specific_times_list[indices] = specific_times

        #print(baseline_list)
        #cams = xr.open_dataset("/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_201611_CAMS-inversion.nc")
        # Making the assumption that the values in the month are not different, get the first value
        # Multiple the different values
    
    return baseline_list, north_list, south_list, east_list, west_list, specific_times_list
    
parameters = {
    "model_name" : "test_run",
    "train_load_data" : {
        "year":"2013",
        "freq":100,
        "region":"BRAZIL",
        "size":50,
        "verbose":True,
        "met_args":{"met_levels":[3, 15,21]},
        "load_everything":True,
    },

    "test_load_data" : {
        "year":"2016",
        "freq":100
    },

    "variables" : {
        #"met_variables":{"x_wind":[3,15], "y_wind":[3,15], "upward_air_velocity":[3,15], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]},
        "met_variables":{"x_wind":[3,15], "y_wind":[3,15],"surface_air_pressure":[]},
        "static_variables":["lat_coords", "lon_coords", "x_coords", "y_coords", "topog", "landcover"],
        "time_deltas":[0]
        
    },

    "dataloader_parameters":{
        "input_transforms":["clever_transform_3"],
        "output_transforms":["logv4"]  
    },

    "model_parameters":{
        "num_blocks":4, 
        "node_dim":64, 
        "edge_dim":64, 
        "hidden_layers_processor_node":2, 
        "hidden_layers_processor_edge":2,  
        "hidden_layers_decoder":1, 
        "hidden_dim_processor_node":16, 
        "hidden_dim_processor_edge":16, 
        "hidden_dim_decoder":16, 
        "resolution":4, 
        "output_dim":1, 
        "residuals":False, 
        "attention":False
    },

    "learning_rate":5e-5,

    "loss_functions" : {
        "criterion": "torch.nn.MSELoss()",
        "criterion_test": "torch.nn.MSELoss()"
    }    
}

import copy
import numpy as np
import pandas as pd
import xarray as xr

train_load_data = copy.deepcopy(parameters["train_load_data"])
train_load_data["freq"] = 3

all_years = [ "2012","2013","2014","2015","2016","2017"]
all_months = ['01','02','03','04','05','06','07','08','09','10','11','12']

# Temporary storage
all_outputs = []
all_baselines = []
all_specific_times = []

for year in all_years:
    train_load_data['year'] = year
    parsed_year = parse_years(year)

    for month in all_months:
        data = LoadSquareSatelliteData(month=month, **train_load_data)

        baseline_list, north_list, south_list, east_list, west_list, specific_times_list = \
            baseline_mol_updated_times(data, [month], parsed_year)

        outputs = np.stack((north_list, south_list, east_list, west_list), axis=1)

        # NaN check
        has_nan_outputs = np.isnan(outputs).any()
        print(f"Year {year}, Month {month} -> Contains NaNs outputs:", has_nan_outputs)

        # Save results
        all_outputs.append(outputs)
        all_baselines.append(baseline_list)
        all_specific_times.extend(specific_times_list)

# Concatenate all results
all_outputs = np.concatenate(all_outputs, axis=0)       # (time, 4)
all_baselines = np.concatenate(all_baselines, axis=0)   # (time,)

all_specific_times = pd.to_datetime(all_specific_times)
years = np.array([t.year for t in all_specific_times])
months = np.array([t.month for t in all_specific_times])
import ipdb; ipdb.set_trace()
ds = xr.Dataset(
    data_vars=dict(
        outputs=(["time", "direction"], all_outputs),    # (N, 4)
        baseline=(["time", "direction"], all_baselines), # (N, 4)
    ),
    coords=dict(
        time=("time", all_specific_times),               # (N,)
        year=("time", years),                            # (N,)
        month=("time", months),                          # (N,)
        direction=["north", "south", "east", "west"],    # (4,)
    ),
    attrs=dict(description="Satellite outputs & baselines with times"),
)
ds.to_netcdf("satellite_outputs_old.nc")

# This gets the summed output
outputs_sum = ds.outputs.sum(dim="direction")  # shape: (time,)

print(ds)

