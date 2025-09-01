import sys

'''
# delete before use!!
sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")
sys.path.insert(0,"/user/work/yl18410/miniconda3/envs/new_graphnet_v2/lib/python3.12/site-packages")
'''
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



def getint_updated(filename):
    basename = os.path.basename(filename)
    num_part = basename.split('_')[-1].split('.')[0]
    return int(num_part) if num_part.isdigit() else -1  # give non-numeric ones a low sort order

def get_checkpoint_to_load_updated(model_name, directory, prefer_best=True):
    print(f"loading checkpoint for {model_name}")
    
    model_dir = os.path.join(directory, model_name)
    best_path = os.path.join(model_dir, f"{model_name}_best.pt")
    
    if prefer_best and os.path.exists(best_path):
        print(f"Found best checkpoint: {best_path}")
        return best_path

    # Fallback: load the latest numbered checkpoint
    pattern = os.path.join(model_dir, f"{model_name}_*.pt")
    files = sorted(
        [f for f in glob.glob(pattern) if getint(f) != -1],
        key=getint
    )
    
    assert len(files) > 0, f"No numeric checkpoint files found for model name {model_name}"
    checkpoint_to_load = files[-1]
    print(f"Loading latest checkpoint: {checkpoint_to_load}")
    
    return checkpoint_to_load

def get_checkpoint_to_load(model_name,directory):
    print(f"loading last checkpoint for {model_name}")
    #directory="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/"
    files = sorted(glob.glob(glob.escape(f"{directory}{model_name}/{model_name}_")+"*.pt"), key=getint)
    assert len(files)>0, f"no files found for model name {model_name}"
    checkpoint_to_load = files[-1] 
        
    return checkpoint_to_load 


# Load the models
# Load the data for a particular year
# Go through the different months
# Get the test predictions
# Get the MAE
# Make it so that I can put the device somewhere


# Make it so that I can load the parameters effectively.
    # Load the information using wandb
# Make it so that I can save the information in a particular location

def perform_inference(_run_path,_practice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize W&B API
    api = wandb.Api()
    
    # 2. Get the run object
    '''
    run = api.run(f"your_entity/your_project/{run_id}")  # e.g. "username/projectname/runid"
    '''
    run = api.run('nerdk312/BoundaryCondition-Prediction/7s441nbt')
    # 3. Pull hyperparameters
    parameters = run.config  # this is a dict with your hparams
    folder_name = 'boundary_condition'
    # Nawid - Model loading
    model_name = run.name #'num_classes-4_baselines-False_size-50_decoder-conv_normalization-all_trainyear-2014-5_trainfreq-3_epochs-150_lr-5e-05_seed-42_date-Aug-09-2025'

    inference_path=f"/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/graph_weather/{folder_name}/"
    data_save_name = f"{inference_path}{model_name}/{model_name}_final_epoch_data.pkl"

    with open(f'{data_save_name}', 'rb') as f:
        loaded_filtered_data_dict = pickle.load(f)
    #loaded_filtered_data_dict = np.load(data_save_name)
    predictions = loaded_filtered_data_dict['test_dataset_predictions']
    truths = loaded_filtered_data_dict['test_dataset_truths']
    # Nawid - Look at loading the best model
    checkpoint_to_load = get_checkpoint_to_load_updated(model_name,inference_path, prefer_best=False)
    # Nawid - Make sure it has the right location
    checkpoint = torch.load(checkpoint_to_load, map_location=device)

    output_mean, output_std = checkpoint['normalization']['outputs_mean'], checkpoint['normalization']['outputs_std']
    # Nawid - data loading
    if _practice:
        with open("practice_data_2014_6_updated.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        data, test_data, grid, inputs, names, test_inputs =loaded_data["data"],loaded_data["test_data"],loaded_data["grid"], loaded_data["inputs"], loaded_data["names"], loaded_data["test_inputs"]

    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])
    
    # Nawid - Making it so that the different value can be used for the case where there is a single value for the approach
    print(train_load_data)
    print(test_load_data)
    if not _practice:
        if "size" in (parameters["train_load_data"].keys()):
            print('Using square domain')
            data = LoadSquareSatelliteData(**train_load_data)
            test_data = LoadSquareSatelliteData(**test_load_data)    
        else:    
            print('Using fixed domain')
            data = LoadDomainSatelliteData(**train_load_data)
            test_data = LoadDomainSatelliteData(**test_load_data)
        
    train_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # TODO: Nawid- get the train year and the test year from the trainload data 
    train_year = parse_years(train_load_data['year'])
    
    test_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    test_year = parse_years(test_load_data['year'])

    baseline_list, north_list, south_list, east_list, west_list = baseline_mol_updated(data,train_months, train_year)
    test_baseline_list, test_north_list, test_south_list, test_east_list, test_west_list = baseline_mol_updated(test_data,test_months, test_year)
    outputs = np.stack((north_list,south_list, east_list,west_list),axis=1)
    test_outputs = np.stack((test_north_list,test_south_list, test_east_list,test_west_list),axis=1)

    if parameters['normalization'] =='separate':
        outputs_mean_values, outputs_std_values = np.mean(outputs,axis=0), np.std(outputs,axis=0)
        baseline_mean_values, baseline_std_values = np.mean(baseline_list,axis=0), np.std(baseline_list,axis=0)
        outputs =  (outputs-outputs_mean_values)/outputs_std_values        
        baseline_list =  (baseline_list-baseline_mean_values)/baseline_std_values

        test_outputs = (test_outputs-outputs_mean_values)/outputs_std_values
        test_baseline_list = (test_baseline_list-baseline_mean_values)/baseline_std_values

    elif parameters['normalization'] =='all':
        outputs_mean_values, outputs_std_values = np.mean(outputs), np.std(outputs)
        baseline_mean_values, baseline_std_values = np.mean(baseline_list), np.std(baseline_list)
        outputs =  (outputs-outputs_mean_values)/outputs_std_values
        baseline_list =  (baseline_list-baseline_mean_values)/baseline_std_values

        test_outputs = (test_outputs-outputs_mean_values)/outputs_std_values
        test_baseline_list = (test_baseline_list-baseline_mean_values)/baseline_std_values

    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))
    input_variables = parameters["variables"]
    if not _practice:
        inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)
        test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
    
    use_baselines = parameters['use_baselines']
    if use_baselines:
        aux_index= 4
    else:
        aux_index = 0

    train_batch_size = 5
    test_batch_size=5
    print(train_load_data)
    print(test_load_data)
    write_to_file(inference_path,model_name,"Before loading data")
    #grid, _ = get_grid(data, parameters.get("grid_reference_fp"))
    #dummy_dataset = BoundaryDatasetXR(dummy_inputs,dummy_outputs,input_names=names, **parameters["dataloader_parameters"])
    # Nawid - loading to get the parameters of interest
    train_dataset = BoundaryDataset(inputs,baseline_list,outputs,use_baselines=use_baselines,input_names=names, **parameters["dataloader_parameters"])
    test_dataset = BoundaryDataset(test_inputs,test_baseline_list,test_outputs,use_baselines= use_baselines,input_names=names,test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    #aux_dim = len(input_variables["static_variables"]) 
    feature_dim=np.shape(inputs)[-1]
    
    num_classes = parameters['num_classes']
    # Should probably update the name!!
    num_lat, num_lon = len(data.met.lat.values), len(data.met.lon.values)

    if parameters['network_decoder'] =='conv':
        print('Using conv network')
        model = GraphSatelliteForecasterConvClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes,input_height = num_lat, input_width=num_lon, **parameters["model_parameters"])
    else:
        print('Using normal network')
        model = GraphSatelliteForecasterClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes, **parameters["model_parameters"])

    # 3. Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Put in eval mode
    model.eval()

    for i_test, batch in enumerate(test_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels = batch[0].to(device), batch[1].to(device)
        test_model_outputs = model(ins)

        test_error += criterion_test(test_model_outputs, labels).item()
        test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = test_model_outputs.detach().cpu().numpy()
            
        test_loss = test_error / (i_test+1)               
        losses["train"].append(running_loss/(i_train+1))
        losses["test"].append(test_error/(i_test+1))

        # Nawid - test predictions data
        denormalized_test_predictions = (test_out*outputs_std_values) + outputs_mean_values
        denormalized_test_predictions_summed = np.sum(denormalized_test_predictions,axis=1)
        
        test_mae = np.mean(np.abs(denormalized_test_truths_summed - denormalized_test_predictions_summed))


    test_mae = np.mean(np.abs(denormalized_test_truths_summed - denormalized_test_predictions_summed))
    '''
        # 👇 Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / (i_train + 1),
            "test_loss": test_loss,
            "summed_test_MAE": test_mae
        })
    wandb.finish()  # 👈 End wandb session
    '''



perform_inference(True, True)