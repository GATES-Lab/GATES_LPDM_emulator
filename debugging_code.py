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