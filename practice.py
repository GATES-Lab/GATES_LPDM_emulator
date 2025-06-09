import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

#sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")

#sys.path.insert(0,"/user/work/ef17148/oldstuff/ef17148/.conda/envs/new_graphnet/lib/python3.12/site-packages")

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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import timeit


original_data = LoadDomainSatelliteData(year="2016", region="BRAZIL", month="06", freq=40, verbose=True, load_everything=True)

variables = {"x_wind":[3,15], "y_wind":[3], "surface_air_pressure":[]}
static_variables=["lat_coords", "lon_coords", "x_coords", "y_coords"]
#existatic_variables=[]
inputs, input_names = get_fixed_satellite_inputs(original_data, variables, static_variables=static_variables, time_deltas=[24], return_variable_names=True, return_asarray=True)
import ipdb; ipdb.set_trace()