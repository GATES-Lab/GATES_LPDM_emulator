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
import wandb


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
    #run = api.run('nerdk312/BoundaryCondition-Prediction/yxwypg3s')
    #run = api.run('nerdk312/BoundaryCondition-Prediction/8b5jglg6')
    run = api.run('nerdk312/BoundaryCondition-Prediction/608nfpxb')   # Nawid - Current best model
    # num_classes-4_baselines-False_size-50_decoder-conv_normalization-all_trainyear-2014-6_trainfreq-3_epochs-200_lr-5e-06_seed-42_date-Aug-17-2025
    # 3. Pull hyperparameters
    parameters = run.config  # this is a dict with your hparams
    folder_name = 'boundary_condition'
    # Nawid - Model loading
    model_name = run.name #'num_classes-4_baselines-False_size-50_decoder-conv_normalization-all_trainyear-2014-5_trainfreq-3_epochs-150_lr-5e-05_seed-42_date-Aug-09-2025'

    inference_path=f"/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/graph_weather/{folder_name}/"
    
    with open(f"{inference_path}{model_name}/transform_parameters_{model_name}.pickle", 'rb') as f:
        test_mode = pickle.load(f)
        

    # Nawid - Look at loading the best model
    checkpoint_to_load = get_checkpoint_to_load_updated(model_name,inference_path, prefer_best=True)
    # Nawid - Make sure it has the right location
    checkpoint = torch.load(checkpoint_to_load, map_location=device)

    output_mean, output_std = checkpoint['normalization']['outputs_mean'], checkpoint['normalization']['outputs_std']
    # Nawid - data loading
    '''
    if _practice:
        with open("practice_data_2014_6_updated.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        data, test_data, grid, inputs, names, test_inputs =loaded_data["data"],loaded_data["test_data"],loaded_data["grid"], loaded_data["inputs"], loaded_data["names"], loaded_data["test_inputs"]

    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])
    '''
    
    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])
    test_load_data['year'] = '2018' 
    test_year = copy.deepcopy(test_load_data["year"])
    test_load_data["freq"] = 2
    test_batch_size=5
    # Nawid - Making it so that the different value can be used for the case where there is a single value for the approach
    print(test_load_data)
    # Nawid - decide to use baseline or not
    use_baselines = parameters['use_baselines']
    if use_baselines:
        aux_index= 4
    else:
        aux_index = 0
    input_variables = parameters["variables"]
    num_classes = parameters['num_classes']
    collated_test_maes = []
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    all_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    all_predictions = []
    all_truths = []
    for month in all_months:
        if "size" in (parameters["train_load_data"].keys()):
            print('Using square domain')
            
            test_data = LoadSquareSatelliteData(month = month, **test_load_data)    
        else:    
            print('Using fixed domain')
            test_data = LoadDomainSatelliteData(month = month, **test_load_data)   
        
        test_year = parse_years(test_load_data['year'])
        test_baseline_list, test_north_list, test_south_list, test_east_list, test_west_list = baseline_mol_updated(test_data,[month], test_year)
        grid, _ = get_grid(test_data, parameters.get("grid_reference_fp"))

        
        test_outputs = np.stack((test_north_list,test_south_list, test_east_list,test_west_list),axis=1)
        has_nan_outputs = np.isnan(test_outputs).any()
        print("Contains NaNs outputs:", has_nan_outputs)
        # Nawid - Normalizing the outputs
        test_outputs = (test_outputs - output_mean)/output_std   
        # Nawid- true outputs of the test data (not using the train valeus as the training data is shuffled)
        denormalized_test_truths = (test_outputs*output_std) + output_mean
        denormalized_test_truths_summed = np.sum(denormalized_test_truths, axis=1)

        test_inputs, names = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True, return_variable_names=True)
        has_nan_inputs = np.isnan(test_inputs).any()
        print("Contains NaNs inputs:", has_nan_inputs)
        feature_dim=np.shape(test_inputs)[-1]
        num_lat, num_lon = len(test_data.met.lat.values), len(test_data.met.lon.values)
        
        test_dataset = BoundaryDataset(test_inputs,test_baseline_list,test_outputs,use_baselines= use_baselines,input_names=names,test_mode=test_mode, **parameters["dataloader_parameters"])
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        
        if parameters['network_decoder'] =='conv':
            print('Using conv network')
            model = GraphSatelliteForecasterConvClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes,input_height = num_lat, input_width=num_lon, **parameters["model_parameters"])
        else:
            print('Using normal network')
            model = GraphSatelliteForecasterClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes, **parameters["model_parameters"])

        # 3. Load weights
        checkpoint["model_state_dict"]["encoder.h3_nodes"] = model.encoder.h3_nodes
        model.load_state_dict(checkpoint['model_state_dict'])

        # 4. Put in eval mode
        model.eval()

        test_error = 0.0
        test_out = np.zeros((len(test_dataset), num_classes))
        for i_test, batch in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            ins, labels = batch[0].to(device), batch[1].to(device)
            test_model_outputs = model(ins)


            test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = test_model_outputs.detach().cpu().numpy()
                

        # Nawid - test predictions data
        denormalized_test_predictions = (test_out*output_std) + output_mean
        denormalized_test_predictions_summed = np.sum(denormalized_test_predictions,axis=1)
        
        all_predictions.append(denormalized_test_predictions_summed)
        all_truths.append(denormalized_test_truths_summed)

        test_mae = np.mean(np.abs(denormalized_test_truths_summed - denormalized_test_predictions_summed))
        collated_test_maes.append(test_mae)
        print(f"Month {month}: MAE = {test_mae}")
    

    # ---- After loop ----
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_truths = np.concatenate(all_truths, axis=0)

    overall_mae = np.mean(np.abs(all_truths - all_predictions))
    print(f"Overall MAE across all months = {overall_mae:.4f}")

    # log to wandb
    wandb.init(project=run.project, entity=run.entity, name=f"{run.name}_inference_eval")

    wandb.log({
        "monthly_mae": {m: mae for m, mae in zip(all_months, collated_test_maes)},
        "avg_monthly_mae": np.mean(collated_test_maes),
        "overall_mae": overall_mae
    })

    '''
    # --- Compute and log average MAE ---
    avg_test_mae = np.mean(collated_test_maes)
    print(f"Average MAE across months = {avg_test_mae:.4f}")

    # Log into wandb
    wandb.init(project=run.project, entity=run.entity, name=f"{run.name}_inference_eval")
    wandb.log({
        "monthly_mae": {m: mae for m, mae in zip(all_months, collated_test_maes)},
        "avg_test_mae": avg_test_mae
    })
    wandb.finish()
    '''

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