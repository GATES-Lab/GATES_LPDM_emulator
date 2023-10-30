import sys

#import cartopy
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import einops
import numpy as np
import torch
import os
import pickle
sys.path.insert(0, "/user/work/yl18410/")
sys.path.insert(0, "/user/work/yl18410/graphnet_LPDM_emulator/")
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
import argparse
import json
import scipy
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import power_transform




def normalize_footprints(footprint_data,pca=None, normalization_parameters=None):
    dim=64 
    footprint_data = power_transform(footprint_data+1e-10, method='box-cox')
    if normalization_parameters==None:
        mean = np.mean(footprint_data)
        std = np.std(footprint_data)
    else:
        mean, std = normalization_parameters[0], normalization_parameters[1]
    normalized_footprint_data = (footprint_data -mean)/std
    if pca==None:
        pca = PCA(n_components=dim)
        pca.fit(normalized_footprint_data)
        
    footprint_features = pca.transform(normalized_footprint_data)
    return footprint_features, mean, std, pca


def basic_prototype_assignment(prototype_features,data_features):
    prototype_distance_matrix = scipy.spatial.distance.cdist(prototype_features, data_features,'euclidean')
    prototype_assignments = np.argmin(prototype_distance_matrix,axis=0)
    return prototype_assignments
'''
parser = argparse.ArgumentParser(description="Load parameters")
parser.add_argument("file_name", help="parameter file name")
parser.add_argument("--file_path", help="parameter file path")

args = parser.parse_args()
file_name = args.file_name
file_path = args.file_path
'''


#print(file_name, file_path)

def load_file(file_name, file_path):
    # file_path=False if no argument was passed to the parser
    if not file_path:
       # edit this to your default filepath for comfort
       file_path ="/user/work/ef17148/GCN/graphnet/graph_weather/train_satellite_files/"
    file_path = f"{file_path}{file_name}"
    try: # Nawid - Open up the file
        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                data = json.load(file)
            else:
                data = file.read()
                data = json.loads(data)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None
    

"""
Creates predictions for a particular model and saves them as a .nc file

- If you want predictions for a model on the same domain size that model was trained, just pass the parameter file 

- If you want predictions for a model on a different domain size:
    - create a folder named as model_name that will contain the predictions
    - add the prediction grid (copy from another model of the same size, should have name f"grid_{model_name}.pickle")
    - add a section in the parameters file with the following info
        "reference_model" : {
            "_comment" : "use reference model to generate predictions of a particular domainsize that is different that the training size",
            "model_name" : "smaller_test_run",
            "domain_size" : 30
        }    
    where model_name is the name of the trained model to use for predictions, and domain_size is the size this model was trained on

"""


torch.manual_seed(33)

path="/user/work/yl18410/graphnet_LPDM_emulator/graph_weather/trained_satellite_models_fixedmet/"
file_name = 'prototype_parameter_template.txt'
file_path = '/user/work/yl18410/graphnet_LPDM_emulator/'
parameters = load_file(file_name, file_path) 


model_name = parameters["model_name"]
print(model_name)

assert os.path.isdir(f"{path}{model_name}"), f"There should exist a folder called {path}{model_name} that contains a grid of the right size!"

assert os.path.isfile(f"{path}{model_name}/grid_{model_name}.pickle"), f"There should exist a file with the grid of the right size, named {path}{model_name}/grid_{model_name}.pickle"
# Nawid - Make a folder for predictions
os.makedirs(f"{path}{model_name}/predictions", exist_ok=True)

if hasattr(parameters, "reference_model"):
    reference_model=True
    reference_model_name = parameters["reference_model"]["model_name"]
    print(f"using reference model {reference_model_name} to make predictions! Note this only works if transforms are transferrable across domain sizes (ie not boxcox)")
else:
    reference_model_name = model_name
    reference_model=False

# Naiwd - open the grid size I believe
with open(f"{path}{model_name}/grid_{model_name}.pickle", 'rb') as f:
    grid = pickle.load(f)


with open(f"{path}{reference_model_name}/transform_parameters_{reference_model_name}.pickle", 'rb') as f:
    test_mode = pickle.load(f)

train_load_data = copy.deepcopy(parameters["train_load_data"])
data = LoadSatelliteData(**train_load_data)

prototype_load_data = copy.deepcopy(parameters["train_load_data"])
prototype_load_data.update(parameters["prototype_load_data"])
prototype_data = LoadSatelliteData(**prototype_load_data)



prototype_fps = np.copy(prototype_data.fp_data)
prototype_fps = np.reshape(prototype_fps, (len(prototype_fps), prototype_data.size*prototype_data.size))
prototype_indices = [9,10,50,3,13,110,94]

train_fps = np.copy(data.fp_data)
train_fps = np.reshape(train_fps, (len(train_fps), data.size*data.size))

footprint_train_features, mean, std, pca = normalize_footprints(train_fps)
normalization_parameters = [mean, std]
footprint_prototype_features, _, _, pca = normalize_footprints(prototype_fps,pca=pca, normalization_parameters=normalization_parameters)

test_load_data = copy.deepcopy(parameters["train_load_data"])
test_load_data.update(parameters["test_load_data"])

test_year = copy.deepcopy(test_load_data["year"])
del test_load_data["year"]

variables = parameters["variables"]
met_variables = variables["met_variables"]
others = variables["others"]
topog=variables["topog"]
jumps = variables["jumps"]

# "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"
for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
    print("loading month " + month)
    date="2016"+month
    test_data = LoadSatelliteData(date, **test_load_data)
    # Nawid - Get the test prototypes
    test_fps = np.copy(test_data.fp_data)
    test_fps = np.reshape(test_fps, (len(test_fps), data.size*data.size))
    footprint_test_features, _, _, pca = normalize_footprints(test_fps,pca=pca, normalization_parameters=normalization_parameters)
    desired_prototype_test_assignments = basic_prototype_assignment(footprint_prototype_features[prototype_indices],footprint_test_features)
    test_prototypes = prototype_fps[prototype_indices][desired_prototype_test_assignments]

    _, inputs, names, test_data = get_all_inputs_graphnet_satellite_v4(test_data, met_variables, jumps, {}, topog=topog, others=others, centered_coords=variables["centered_coords"])

    # Nawid - get the test dataset
    test_dataset = FootprintsDatasetV2(inputs, test_data.fp_data, input_names=names, test_mode=test_mode, **parameters["dataloader_parameters"])
    test_dataset.add_prototypes(test_prototypes)
    # Nawid - get the footprints
    fps = np.copy(np.reshape(test_data.fp_data, (len(test_data.fp_data), test_data.size,test_data.size)))
    time_vals = test_data.met.time.values
    # Nawid - Cant delete these as they are used as inputs into functions later on
    #del test_data
    #del inputs 

    def getint(name):
        num = name.split('_')[-1]
        num = num.split('.')[0]
        return int(num)

    print("creating model")
    # Nawid - Add an extra dimension for the approach
    model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=np.shape(inputs)[-1]-len(others)-topog+1, aux_dim=len(others)+topog, **parameters["model_parameters"])

    # load reference model
    files = sorted(glob.glob(glob.escape(f"{path}{reference_model_name}/{reference_model_name}_")+"*.pt"), key=getint)
    checkpoint_to_load = files[-1] 
    checkpoint = torch.load(checkpoint_to_load, map_location=torch.device('cpu'))
    optimizer = optim.AdamW(model.parameters(), lr=checkpoint["learning_rate"])

    if reference_model:
        # this h3 parameter means nothing, it's just an array of zeros. But when loading different model sizes it complains. I should get rid of it but in the meantime this is fine.
        checkpoint["model_state_dict"]["encoder.h3_nodes"] = model.encoder.h3_nodes
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Nawid -Makes a prediction and performs an inverse transform
    print("predicting")
    preds = model(test_dataset.inputs).detach().numpy()
    transformed_preds = test_dataset.inverse_transform(np.squeeze(preds))
    # Naiwd - reshape into correct size
    preds = np.reshape(np.squeeze(preds), (len(preds), test_data.size,test_data.size))
    transformed_preds = np.reshape(transformed_preds, (len(transformed_preds), test_data.size,test_data.size))

    data_vars = {'predictions':(['time', "lat", "lon"], preds, 
                            {'space': 'transformed', 'type':"prediction", 'emulated_with': model_name}),
                'trans_predictions':(['time', "lat", "lon"], transformed_preds, 
                            {'space': 'original', 'type':"prediction",'emulated_with': model_name}),
                'fp':(['time', "lat", "lon"], fps, 
                            {'space': 'original', 'type':"truth"}),
                'trans_fp':(['time', "lat", "lon"], np.reshape(test_dataset.fp, (len(test_dataset.fp), test_data.size,test_data.size)), 
                            {'space': 'transformed', 'type':"truth"})}

    # define coordinates
    coords = {'time': (['time'], time_vals),
            'lat': (['lat'], list(range(test_data.size))),
            'lon': (['lon'],  list(range(test_data.size)))}

    # define global attributes
    attrs = {'creation_date':str(datetime.now())}
    
    print("saving")
    # create dataset
    ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs)


    ds.to_netcdf(f"{path}{model_name}/predictions/preds_{date}.nc")

    del model, checkpoint, preds, transformed_preds, data_vars, ds 