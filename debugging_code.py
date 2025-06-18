''''
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")

sys.path.insert(0,"/user/work/yl18410/miniconda3/envs/new_graphnet_v2/lib/python3.12/site-packages")

print(sys.path)

import torch
import os
import pickle
import einops

#import plenoptic as po

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import *
from model.data.load_data import *
from model.forecast import GraphSatelliteForecaster
from model.loss_functions import *
from model.evaluation import *

import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse

import random



import timeit

met_args = {"met_levels":[3, 15]}
data_dom = LoadDomainSatelliteData(year="2016", region="BRAZIL", month="06", freq=40, verbose=True, met_args=met_args, domain_to_cut={"lat":[-20,20]})

variables = {"x_wind":[3,15], "y_wind":[3], "surface_air_pressure":[]}
static_variables=["lat_coords", "lon_coords", "x_coords", "y_coords", "topog", "landcover"]
inputs_array, input_names = get_square_satellite_inputs(data_dom, variables, static_variables=static_variables, time_deltas=[24], return_variable_names=True, return_asarray=True)

#inputs, names = get_square_satellite_inputs(data_dom, {"x_wind":[3,15], "y_wind":[3], "surface_air_pressure":[]}, static_variables=["lat_coords", "topog", "landcover", "landcover_disaggregated"], time_deltas=[100], verbose=True, return_asarray=False, return_variable_names=True)

grid,_  = get_grid(data_dom)


dataset = FootprintsDataset(inputs=inputs_array, fp=data_dom.fp_data, input_names=input_names, input_transforms=["clever_transform_3"], output_transforms= ["logv4"])


train_loader = DataLoader(dataset, batch_size=5, shuffle=True)


aux_dim = 0
feature_dim=np.shape(inputs_array)[-1]-aux_dim


model = GraphSatelliteForecaster(grid, whole_world=False,aux_dim=aux_dim, feature_dim=feature_dim)

for i, batch in enumerate(train_loader):
    # get the inputs; data is a list of [inputs, labels]
    ins, labels = batch[0], batch[1]
    # zero the parameter gradient
    # forward + backward + optimize
    outputs = model(ins)
'''


import sys

# delete before use!!
sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")
sys.path.insert(0,"/user/work/yl18410/miniconda3/envs/new_graphnet_v2/lib/python3.12/site-packages")

#import cartopy
#import cartopy.crs as ccrs
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
from model.forecast import GraphSatelliteForecaster
from model.loss_functions import *


import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse

import random



#### 1 Set up


parameters = {
    "model_name" : "test_run",
    "train_load_data" : {
        "year":"2015",
        "freq":100,
        "region":"BRAZIL",
        "size":50,
        "verbose":True,
        "met_args":{"met_levels":[3, 15,21]},
        "load_everything":True,
    },

    "test_load_data" : {
        "year":"2016",
        "freq":300
    },

    "variables" : {
        "met_variables":{"x_wind":[3,15], "y_wind":[3,15], "upward_air_velocity":[3,15], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]},
        "static_variables":["lat_coords", "lon_coords", "x_coords", "y_coords", "topog", "landcover"],
        "time_deltas":[6,12]
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
        "criterion": "MSE_weighted_by_truth_nans(multiply_by_add, w=1000)",
        "criterion_test": "torch.nn.MSELoss()"
    }    
}

# TODO make this an argument


print("PARAMETERS:")
print(parameters)

model_name = parameters["model_name"]
print(model_name)


if "seed" in (parameters.keys()):
    print(parameters["seed"])
    np.random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.cuda.manual_seed(parameters["seed"])
    random.seed(parameters["seed"])

else:
    print("34")
    np.random.seed(34)
    torch.manual_seed(34)
    torch.cuda.manual_seed(34)
    random.seed(34)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### 2 Load Data


train_load_data = copy.deepcopy(parameters["train_load_data"])
# load train parameters and upload with any changes to test data
test_load_data = copy.deepcopy(parameters["train_load_data"])
test_load_data.update(parameters["test_load_data"])
print(train_load_data)
print(test_load_data)

data = LoadSquareSatelliteData(**train_load_data)
test_data = LoadSquareSatelliteData(**test_load_data)

# extract inputs
import ipdb; ipdb.set_trace()
input_variables = parameters["variables"]

inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)

test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)

# the model gets built with respect to a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
grid, _ = get_grid(data, parameters.get("grid_reference_fp"))
print("setting up model")

test_batch_size=10

# transform data - 
train_dataset = FootprintsDatasetV3(inputs, data.fp_data, input_names=names, **parameters["dataloader_parameters"])

print(train_dataset.transform_parameters)

test_dataset = FootprintsDatasetV3(test_inputs, test_data.fp_data, input_names=names, test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

size = data.size

# all the necessary data is already in the loaders, so we can delete the objects
del data, test_data

lr = parameters["learning_rate"]
print(lr)
#### 3 Make model

# this is leftover from the previous model and actually shouldnt make a difference
aux_dim = len(input_variables["others"]) 
feature_dim=np.shape(inputs)[-1]-aux_dim

# Should probably update the name!!
model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_dim, **parameters["model_parameters"])

criterion = eval(parameters["loss_functions"]["criterion"])

criterion_test = eval(parameters["loss_functions"]["criterion_test"])

optimizer = optim.AdamW(model.parameters(), lr=lr)

losses = {"train":[], "test":[], "NMAE_test":[], "MSE_test_transformed":[], "NMAE_test_transformed":[], "accuracy":[], "IoU":[]}

NMAE_function = NMAE


#### 3 Dump info
print("saving grids etc")

epoch_so_far = 0
if torch.cuda.is_available():
    model.cuda()

#### 4 Train loop
for epoch in range(302):
    epoch=epoch+epoch_so_far
    running_loss = 0.0
    print(f"Start Epoch: {epoch}")
    start = time.time()
    for i, batch in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(ins)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = criterion_test(outputs, labels)
        running_loss += loss.item()
        end = time.time()
        del outputs 
        if i % 10==0:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec")
    
    test_error = 0.0
    test_out = np.zeros((test_dataset.inputs.size()[0], test_dataset.inputs.size()[1]))   
    for i_test, batch in enumerate(test_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels = batch[0].to(device), batch[1].to(device)
        test_error += criterion_test(model(ins), labels).item()
        test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = np.squeeze(model(ins).detach().cpu().numpy())
    
    
    losses["train"].append(running_loss/(i+1))
    losses["test"].append(test_error/(i_test+1))

    #evaluate
    truths = torch.squeeze(test_dataset.fp).detach().numpy()
    print(f"NMAE: {NMAE(test_out,truths)}")
    losses["NMAE_test"].append(NMAE(test_out,truths))
    transformed_preds = test_dataset.inverse_transform(test_out)
    evaluation_metrics = test_dataset.evaluate()
    losses["NMAE_test_transformed"].append(evaluation_metrics["NMAE"])
    losses["MSE_test_transformed"].append(evaluation_metrics["MSE"])
    losses["accuracy"].append(evaluation_metrics["Accuracy"])
    losses["IoU"].append(evaluation_metrics["IOU"])
