import sys

#import cartopy
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import einops
import numpy as np
import torch
import os
import pickle

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import *
from model.data.load_data import *
from model.forecast import GraphSatelliteForecaster
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime


torch.manual_seed(33)

path="/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/trained_models"

# name of trained model used to predict
model_name = "satellite_50x50_default"
# name of model under which to save the predictions
model_name_data = "satellite_100x100_with_default"



with open(f"{path}{model_name_data}/grid_{model_name_data}.pickle", 'rb') as f:
    grid = pickle.load(f)
with open(f"{path}{model_name_data}/transforms_{model_name_data}.pickle", 'rb') as f:
    test_mode = pickle.load(f)

#os.mkdir(f"{path}{model_name_data}/predictions")
#"01", "02", "03", "04", 
for month in ["05", "06", "07", "08", "09", "10", "11", "12"]:
    print("loading month " + month)
    test_data = LoadSatelliteData("2016"+month, region="BRAZIL", freq=4, metsize=100, size =100, topog="default", verbose=True, cut_met = False, met_datadir="/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_onlyvalid_100_", fill_outofdomain_with="zeros")

    print("getting inputs")
    others = ["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "distance_centre", "x_coords", "y_coords"]
    topog=True
    _, inputs, names, test_data = get_all_inputs_graphnet_satellite_v4(test_data, {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[0], "surface_air_pressure":[0]}, [6,12], {}, topog=True, others=others, centered_coords=True)



    test_params = {"inputs":np.copy(inputs), "fp":np.copy(test_data.fp_data), "transform_output":"boxcox", "feature_dim":np.shape(inputs)[-1]-len(others)-topog, "clever_transform_2":True, "aux_dim":len(others)+topog, "zeroing":True, "input_names":names, "test_mode":test_mode}

    test_dataset = FootprintsDataset(**test_params)

    fps = np.copy(np.reshape(test_data.fp_data, (len(test_data.fp_data), 100,100)))
    time =  test_data.met.time.values

    del test_data
    del inputs 

    def getint(name):
        num = name.split('_')[-1]
        num = num.split('.')[0]
        return int(num)

    print("creating model")
    model_params = {"whole_world":False, "feature_dim":np.shape(test_dataset.inputs)[-1]-len(others)-topog, "aux_dim":len(others)+topog,"num_blocks":4, "node_dim":64, "edge_dim":64, "hidden_layers_processor_node":2, "hidden_layers_processor_edge":2,  "hidden_layers_decoder":1, "hidden_dim_processor_node":16, "hidden_dim_processor_edge":16, "hidden_dim_decoder":16, "resolution":4, "output_dim":1, "residuals":False}

    model = GraphSatelliteForecaster(grid, **model_params)

    files = sorted(glob.glob(glob.escape(f"{path}{model_name}/{model_name}_")+"*.pt"), key=getint)
    checkpoint_to_load = files[-1] 
    checkpoint = torch.load(checkpoint_to_load, map_location=torch.device('cpu'))

    checkpoint["model_state_dict"]["encoder.h3_nodes"] = torch.zeros((5316, 139))


    optimizer = optim.AdamW(model.parameters(), lr=checkpoint["learning_rate"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("predicting")
    preds = model(test_dataset.inputs).detach().numpy()
    transformed_preds = test_dataset.inverse_transform(np.squeeze(preds))

    preds = np.reshape(np.squeeze(preds), (len(preds), 100,100))
    transformed_preds = np.reshape(transformed_preds, (len(transformed_preds), 100,100))

    data_vars = {'predictions':(['time', "lat", "lon"], preds, 
                            {'space': 'transformed', 'type':"prediction", 'emulated_with': model_name}),
                'trans_predictions':(['time', "lat", "lon"], transformed_preds, 
                            {'space': 'original', 'type':"prediction",'emulated_with': model_name}),
                'fp':(['time', "lat", "lon"], fps, 
                            {'space': 'original', 'type':"truth"}),
                'trans_fp':(['time', "lat", "lon"], np.reshape(test_dataset.fp, (len(test_dataset.fp), 100,100)), 
                            {'space': 'transformed', 'type':"truth"})}

    # define coordinates
    coords = {'time': (['time'], time),
            'lat': (['lat'], list(range(100))),
            'lon': (['lon'],  list(range(100)))}

    # define global attributes
    attrs = {'creation_date':str(datetime.now())}
    
    print("saving")
    # create dataset
    ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs)

    date="2016"+month
    ds.to_netcdf(f"{path}{model_name_data}/predictions/preds_{date}.nc")

    del model, checkpoint, preds, transformed_preds, data_vars, ds 