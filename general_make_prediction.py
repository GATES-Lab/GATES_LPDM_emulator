import sys

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

"""
This file makes monthly footprint predictions based on a reference model. As datasets can be too large to fit into memory, we load N independent chunks, save the intermediate files, and join at the end. Although this work, we should test smarter ways of integrating the new data loading functions, e.g. doing the data loading only once, but only bringing subsets of it into memory?


- If you want predictions for a model on the same domain size that model was trained, just pass the parameter file 

- If you want predictions from model_name, but on a different domain size:
    - create a folder named new_model_name that will contain the predictions
    - add the prediction grid (copy from another model of the same size, should have name f"grid_{new_model_name}.pickle") 
    - copy the parameter file from the reference model into the folder, rename to "training_settings_new_model_name.json"
    - edit the size in train_load_data
    - add a section in the parameters file with the following info
        "reference_model" : {
            "_comment" : "use reference model to generate predictions of a particular domainsize that is different that the training size",
            "model_name" : "model_name",
            "domain_size" : 50
        }    

    where model_name is the name of the trained model to use for predictions, and domain_size is the size this model was trained on

"""

#### 1 Set up
parser = argparse.ArgumentParser(description="Load parameters")
parser.add_argument("file_name", help="parameter file name")
parser.add_argument("--file_path", help="parameter file path")
parser.add_argument("--month", help="month number", default="all")
parser.add_argument("--grid_file", help="name id for grid file to use in the prediction", default="grid")
parser.add_argument("--saving_folder", help="folder to save predictions to", default="predictions")


args = parser.parse_args()
file_name = args.file_name
file_path = args.file_path
month=args.month

grid_file = args.grid_file
saving_folder = args.saving_folder



def load_file(file_name, file_path):
    # file_path=False if no argument was passed to the parser
    if not file_path:
       # edit this to your default filepath
       file_path ="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/" 

    print(f"opening {file_path}{file_name}")
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
    
# make this importable!
path="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/"

parameters = load_file(file_name, file_path) 


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


assert os.path.isdir(f"{path}{model_name}"), f"There should exist a folder called {path}{model_name} that contains a grid of the right size!"
assert os.path.isfile(f"{path}{model_name}/{grid_file}_{model_name}.pickle"), f"There should exist a file with the grid of the right size, named {path}{model_name}/{grid_file}_{model_name}.pickle"


os.makedirs(f"{path}{model_name}/{saving_folder}", exist_ok=True)

## checking if prediction will be done with the same model or with a reference model
if hasattr(parameters, "reference_model") or "reference_model" in parameters.keys():
    reference_model=True
    reference_model_name = parameters["reference_model"]["model_name"]
    print(f"using reference model {reference_model_name} to make predictions! Note this only works if transforms are transferrable across domain sizes (ie not boxcox)")
else:
    reference_model_name = model_name
    reference_model=False


with open(f"{path}{model_name}/{grid_file}_{model_name}.pickle", 'rb') as f:
    grid = pickle.load(f)

try: 
    with open(f"{path}{model_name}/transform_parameters_{model_name}.pickle", 'rb') as f:
        test_mode = pickle.load(f)
    print("test mode for model name successfully loaded!")
except:
    print("no test mode was found for model name - attempting reference model")
    with open(f"{path}{reference_model_name}/transform_parameters_{reference_model_name}.pickle", 'rb') as f:
        test_mode = pickle.load(f)

parameters["train_load_data"]["fill_outofdomain_with"] = "nans"
test_load_data = copy.deepcopy(parameters["train_load_data"])
test_load_data.update(parameters["test_load_data"])
test_year = copy.deepcopy(test_load_data["year"])
del test_load_data["year"]


test_load_data["sampling_mode"] = "regular"

input_variables = parameters["variables"]


# loading the model into memory is an expensive step
# we only do it once, then it stays loaded
model_loaded = False

# "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"
all_months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

if month != "all":
    all_months = [all_months[int(month)-1]]

for month in all_months:

    # these could be chosen more cleverly!
    if test_load_data["size"]==200:
        if month in ["07", "08", "09", "12"]:
            test_load_data["freq"] = 20
        else:
            test_load_data["freq"] = 15

    else:
        if month in ["07", "08", "09"]:
            test_load_data["freq"] = 15
        else:
            test_load_data["freq"] = 10
    
    
    generated_files = []
    # we sample 1 in every freq footprints regularly along the time axis. offset 
    for offset in list(range(test_load_data["freq"])):
        print(f"loading month {month} w offset {offset} (set {offset/test_load_data["freq"]}) at {datetime.now().strftime('%d/%m %H:%M:%S')}")

        date=str(test_year)+month
        test_data = LoadSquareSatelliteData(month=month, year=test_year, freq_offset=offset, **test_load_data)

        inputs, names = get_square_satellite_inputs(test_data, **input_variables, return_variable_names=True, return_asarray=True)

        fps = np.copy(test_data.fp_data)
        size = test_data.size
        time_vals = test_data.met.time.values

        del test_data

        aux_dim = len(input_variables["others"]) 
        feature_dim=np.shape(inputs)[-1]-aux_dim 

        test_dataset = FootprintsDataset(inputs, fps, input_names=names, test_mode=test_mode, **parameters["dataloader_parameters"])
    
        del inputs

        if not model_loaded:
            grid, idx_latlon = get_grid(test_data, parameters.get("grid_reference_fp"))

            model = GraphSatelliteForecaster(grid, whole_world=False, idx_latlon=idx_latlon, feature_dim=feature_dim, aux_dim=aux_dim, **parameters["model_parameters"])
    
            print("loading reference model model")
            # load reference model
            print(f"path: {path}{reference_model_name}/{reference_model_name}_*.pt")
            print(glob.glob(glob.escape(f"{path}{reference_model_name}/{reference_model_name}_")+"*.pt"))
            files = sorted(glob.glob(glob.escape(f"{path}{reference_model_name}/{reference_model_name}_")+"*.pt"), key=getint)
            checkpoint_to_load = files[-1] 
            checkpoint = torch.load(checkpoint_to_load, map_location=torch.device('cpu'))
            optimizer = optim.AdamW(model.parameters(), lr=checkpoint["learning_rate"])
    

            if reference_model:
                # this h3 parameter means nothing, it's just an array of zeros. But when loading different model sizes it complains. I should get rid of it but in the meantime this is fine.
                checkpoint["model_state_dict"]["encoder.h3_nodes"] = model.encoder.h3_nodes
    
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.eval()

            del checkpoint 
            model_loaded = True

        preds = model(test_dataset.inputs).detach().numpy()
        transformed_preds = test_dataset.inverse_transform(np.squeeze(preds))

        preds = np.reshape(np.squeeze(preds), (len(preds), size,size))
        transformed_preds = np.reshape(transformed_preds, (len(transformed_preds), size,size))

        data_vars = {'predictions':(['time', "lat", "lon"], preds, 
                                {'space': 'transformed', 'type':"prediction", 'emulated_with': model_name}),
                    'trans_predictions':(['time', "lat", "lon"], transformed_preds, 
                                {'space': 'original', 'type':"prediction",'emulated_with': model_name}),
                    'fp':(['time', "lat", "lon"], np.reshape(fps, (len(test_dataset.fp), size,size)), 
                                {'space': 'original', 'type':"truth"}),
                    'trans_fp':(['time', "lat", "lon"], np.reshape(test_dataset.fp, (len(test_dataset.fp), size,size)), 
                                {'space': 'transformed', 'type':"truth"})}

        # define coordinates
        coords = {'time': (['time'], time_vals),
                'lat': (['lat'], list(range(size))),
                'lon': (['lon'],  list(range(size)))}

        # define global attributes
        attrs = {'creation_date':str(datetime.now())}
        
        print("saving")
        # create dataset
        ds = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)


        ds.to_netcdf(f"{path}{model_name}/{saving_folder}/preds_{date}_{offset}.nc")
        generated_files.append(f"{path}{model_name}/{saving_folder}/preds_{date}_{offset}.nc")
        print("saved")
        del preds, transformed_preds, data_vars, ds 

        # merge into one file
    print(f"ACCOUNTING: {datetime.now().strftime('%d/%m %H:%M:%S')}")
    print("merging all subfiles")

    all_files = []
    for f in generated_files:
        all_files.append(xr.open_dataset(f))

    joint = xr.concat(all_files, dim="time")
    joint = joint.sortby("time")
    joint.to_netcdf(f"{path}{model_name}/{saving_folder}/preds_{test_year}{month}.nc")   

    for file, path in zip(all_files, generated_files):
        file.close()
        try:
            os.remove(path)
        except Exception as e:
            print(f"there was a problem removing file {path}")
            print("leaving it here and continuing!")



