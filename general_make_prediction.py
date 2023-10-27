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
import argparse
import json

parser = argparse.ArgumentParser(description="Load parameters")
parser.add_argument("file_name", help="parameter file name")
parser.add_argument("--file_path", help="parameter file path")

args = parser.parse_args()
file_name = args.file_name
file_path = args.file_path

#print(file_name, file_path)

def load_file(file_name, file_path):
    # file_path=False if no argument was passed to the parser
    if not file_path:
       # edit this to your default filepath for comfort
       file_path ="/user/work/ef17148/GCN/graphnet/graph_weather/train_satellite_files/"
    file_path = f"{file_path}{file_name}"
    try:
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

path="/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/trained_models"

parameters = load_file(file_name, file_path) 


model_name = parameters["model_name"]
print(model_name)

assert os.path.isdir(f"{path}{model_name}"), f"There should exist a folder called {path}{model_name} that contains a grid of the right size!"
assert os.path.isfile(f"{path}{model_name}/grid_{model_name}.pickle"), f"There should exist a file with the grid of the right size, named {path}{model_name}/grid_{model_name}.pickle"

os.mkdirs(f"{path}{model_name}/predictions", exist_ok=True)

if hasattr(parameters, "reference_model"):
    reference_model=True
    reference_model_name = parameters["reference_model"]["model_name"]
    print(f"using reference model {reference_model_name} to make predictions! Note this only works if transforms are transferrable across domain sizes (ie not boxcox)")
else:
    reference_model_name = model_name
    reference_model=False


with open(f"{path}{model_name}/grid_{model_name}.pickle", 'rb') as f:
    grid = pickle.load(f)


with open(f"{path}{reference_model_name}/transform_parameters_{reference_model_name}.pickle", 'rb') as f:
    test_mode = pickle.load(f)


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

    _, inputs, names, test_data = get_all_inputs_graphnet_satellite_v4(test_data, met_variables, jumps, {}, topog=topog, others=others, centered_coords=variables["centered_coords"])


    test_dataset = FootprintsDatasetV2(inputs, test_data.fp_data, input_names=names, test_mode=test_mode, **parameters["dataloader_parameters"])

    fps = np.copy(np.reshape(test_data.fp_data, (len(test_data.fp_data), test_data.size,test_data.size)))
    time_vals = test_data.met.time.values

    del test_data
    del inputs 

    def getint(name):
        num = name.split('_')[-1]
        num = num.split('.')[0]
        return int(num)

    print("creating model")
    


    model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=np.shape(inputs)[-1]-len(others)-topog, aux_dim=len(others)+topog, **parameters["model_parameters"])

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

    print("predicting")
    preds = model(test_dataset.inputs).detach().numpy()
    transformed_preds = test_dataset.inverse_transform(np.squeeze(preds))

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