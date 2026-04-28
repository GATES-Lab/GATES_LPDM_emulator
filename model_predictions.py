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

from full_bc_prediction_pipeline import baseline_mol, write_to_file,  parse_years, additional_data, normalize_boundary_data
from full_bc_prediction_practice import baseline_mol_correction


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



def perform_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    api = wandb.Api()
    run = api.run('nerdk312/BoundaryCondition-Prediction/wc2vux78')
    parameters = run.config

    folder_name = 'boundary_condition'
    model_name = run.name
    inference_path = f"/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/graph_weather/{folder_name}/"

    with open(f"{inference_path}{model_name}/transform_parameters_{model_name}.pickle", 'rb') as f:
        test_mode = pickle.load(f)

    with open(f"{inference_path}{model_name}/grid_{model_name}.pickle", 'rb') as f:
        saved_grid = pickle.load(f)

    checkpoint_to_load = get_checkpoint_to_load_updated(model_name, inference_path, prefer_best=True)
    checkpoint = torch.load(checkpoint_to_load, map_location=device)

    output_mean = checkpoint['normalization']['outputs_mean']
    output_std = checkpoint['normalization']['outputs_std']
    baselines_mean = checkpoint['baselines_normalization']['baselines_mean']
    baselines_std = checkpoint['baselines_normalization']['baselines_std']

    train_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    
    train_load_data["freq"] = 100
    test_load_data.update(parameters["test_load_data"])
    test_load_data['year'] = '2018'
    test_load_data["freq"] = 12
    test_batch_size = 5

    use_baselines = parameters['use_baselines']
    input_variables = parameters["variables"]
    num_classes = parameters['num_classes']
    name_output_format = parameters['output_format']
    height_indices = [4, 5, 6, 7] if parameters['auxiliary'] == 'multiple' else [4]
    import ipdb; ipdb_set_trace()
    # ---------------------------------------------------------------
    # Recompute auxiliary normalisation parameters from training data
    # These aren't stored in the checkpoint so we need to recompute them
    # ---------------------------------------------------------------
    print("Loading training data to compute auxiliary normalisation parameters...")
    train_data = LoadSquareSatelliteData(**train_load_data)
    train_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    train_year = parse_years(train_load_data['year'])

    _, _, train_auxiliary_cams, _ = baseline_mol_correction(
        train_data, train_months, train_year,
        output_format=name_output_format,
        height_indices=height_indices
    )
    if use_baselines:
        aux_index= train_auxiliary_cams.shape[1] # Nawid - gets the sape of the data
    else:
        aux_index = 0
    # Compute and store auxiliary normalisation parameters from training data
    train_auxiliary_cams, (auxiliary_mean_values,auxiliary_std_values) = normalize_boundary_data(train_auxiliary_cams)
    print('cams mean', auxiliary_mean_values)
    print('cams std', auxiliary_std_values)
    del train_data, train_auxiliary_cams  # free memory

    # ---------------------------------------------------------------
    # Load reference month to get feature_dim and spatial dimensions
    # ---------------------------------------------------------------
    print("Loading reference month to build model architecture...")
    ref_data = LoadSquareSatelliteData(month='01', **test_load_data)
    ref_inputs, ref_names = get_square_satellite_inputs(
        ref_data, **input_variables, return_asarray=True, return_variable_names=True
    )
    feature_dim = np.shape(ref_inputs)[-1]
    num_lat = len(ref_data.met.lat.values)
    num_lon = len(ref_data.met.lon.values)
    names = ref_names
    del ref_data, ref_inputs

    # ---------------------------------------------------------------
    # Build and load model once
    # ---------------------------------------------------------------
    print("Building model from saved training grid...")
    if parameters['network_decoder'] == 'conv':
        model = GraphSatelliteForecasterConvClassifier(
            saved_grid, whole_world=False, feature_dim=feature_dim,
            aux_dim=aux_index, num_classes=num_classes,
            input_height=num_lat, input_width=num_lon,
            **parameters["model_parameters"]
        )
    else:
        model = GraphSatelliteForecasterClassifier(
            saved_grid, whole_world=False, feature_dim=feature_dim,
            aux_dim=aux_index, num_classes=num_classes,
            **parameters["model_parameters"]
        )

    checkpoint["model_state_dict"]["encoder.h3_nodes"] = model.encoder.h3_nodes
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    print("Model loaded successfully.")

    # ---------------------------------------------------------------
    # Monthly inference loop
    # ---------------------------------------------------------------
    all_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    all_predictions = []
    all_truths = []
    all_meta = []
    for month in all_months:
        print(f"\nRunning inference for month {month}...")
        test_data = LoadSquareSatelliteData(month=month, **test_load_data)
        test_year = parse_years(test_load_data['year'])

        test_baseline_list, test_outputs, test_auxiliary_cams, test_correction_values = baseline_mol_correction(
            test_data, [month], test_year,
            output_format=name_output_format,
            height_indices=height_indices
        )

        # Normalise outputs and baselines using checkpoint parameters
        test_outputs, _ = normalize_boundary_data(test_outputs,outputs_norm_vals=(output_mean,output_std))
        test_baseline_list, _ = normalize_boundary_data(test_outputs,outputs_norm_vals=(baselines_mean,baselines_std))

        # Normalise auxiliary cams using training-derived parameters
        test_auxiliary_cams, _ = normalize_boundary_data(test_auxiliary_cams,outputs_norm_vals=((auxiliary_mean_values,auxiliary_std_values)))

        denormalized_test_truths = (test_outputs * output_std) + output_mean
        denormalized_test_truths_summed = np.sum(denormalized_test_truths, axis=1)

        test_inputs = get_square_satellite_inputs(
            test_data, **input_variables, return_asarray=True
        )

        test_dataset = BoundaryDataset(
            test_inputs, test_auxiliary_cams, test_outputs,
            use_baselines=use_baselines, input_names=names,
            test_mode=test_mode, **parameters["dataloader_parameters"]
        )
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        test_out = np.zeros((len(test_dataset), num_classes))
        with torch.no_grad():
            for i_test, batch in enumerate(test_loader):
                ins, labels = batch[0].to(device), batch[1].to(device)
                test_model_outputs = model(ins)
                test_out[i_test * test_batch_size:(i_test + 1) * test_batch_size, :] = \
                    test_model_outputs.detach().cpu().numpy()

        denormalized_test_predictions = (test_out * output_std) + output_mean
        denormalized_test_predictions_summed = np.sum(denormalized_test_predictions, axis=1)

        all_predictions.append(denormalized_test_predictions_summed)
        all_truths.append(denormalized_test_truths_summed)

        # --- 1. Extract Meta with Coordinates ---
        # This keeps the release variables AND the spatial grid context
        test_subset = test_data.fp_data_full[["release_lon", "release_lat"]].assign_coords(
            lat=test_data.fp_data_full["lat"],
            lon=test_data.fp_data_full["lon"]
        )
        all_meta.append(test_subset)


    all_predictions = np.concatenate(all_predictions, axis=0)
    all_truths = np.concatenate(all_truths, axis=0)

    # --- 5. Concatenation & NetCDF Export ---
    # Concatenate NumPy results
    all_predictions_final = np.concatenate(all_predictions, axis=0)
    all_truths_final = np.concatenate(all_truths, axis=0)

    # Concatenate Xarray metadata along the 'time' dimension
    combined_ds = xr.concat(all_meta, dim="time")

    # Attach the inference results as new variables
    combined_ds["predictions"] = (("time"), all_predictions_final)
    combined_ds["truths"] = (("time"), all_truths_final)

    # Save the complete object
    save_path = f"full_year_inference_with_coords_2018.nc"
    combined_ds.to_netcdf(save_path)

    print(f"\nSaved full year results with grid coordinates to: {save_path}")
    wandb.finish()

    

perform_inference()
