import sys

# delete before use!!
'''
sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")
sys.path.insert(0,"/user/work/ef17148/oldstuff/ef17148/.conda/envs/new_graphnet/lib/python3.12/site-packages")
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


sys.path.insert(0, "/user/work/ef17148/GCN/graphnet/")
sys.path.insert(1, "/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/")
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
import wandb
import yaml

import re
from general_train_nawid import write_to_file, load_file, set_reproducibility, log_object_as_artifact, EarlyStopping



def parse_years(year_str):
    # Case 1: Range like '201[4-5]'
    match = re.fullmatch(r'201\[(\d)-(\d)\]', year_str)
    if match:
        start, end = map(int, match.groups())
        return [2010 + i for i in range(start, end + 1)]
    
    # Case 2: Exact year like '2014' or '2015'
    if re.fullmatch(r'20\d{2}', year_str):
        return [int(year_str)]
    
    # Fallback
    raise ValueError(f"Invalid year format: {year_str}")


def train_one_epoch(model, loader, optimizer, criterion, criterion_test, device, epoch):
    """
    Runs a single training epoch, iterating over all batches, computing the loss, and updating model weights.
    A separate display loss (criterion_test) is tracked for monitoring without affecting gradients.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader providing batches of (inputs, labels, true_fp).
        optimizer (torch.optim.Optimizer): The optimiser used to update model weights.
        criterion (callable): The training loss function, called as criterion(outputs, labels, true_fp).
        criterion_test (callable): A secondary loss function used for display purposes only, called as criterion_test(outputs, labels).
        device (torch.device): The device (CPU or GPU) on which to run computation.
        epoch (int): The current epoch number, used for progress logging.

    Returns:
        float: The mean display loss across all batches in the epoch.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        ins, labels, true_fp = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        optimizer.zero_grad()
        outputs = model(ins)
        loss = criterion(outputs, labels, true_fp)
        loss.backward()
        optimizer.step()

        # Metrics tracking
        with torch.no_grad():
            display_loss = criterion_test(outputs, labels)
            running_loss += display_loss.item()
        
        if i % 10 == 0:
            print(f"[{epoch}, {i:5d}] Loss: {running_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s")
            
    return running_loss / len(loader)


@torch.no_grad()
def validate_and_predict(model, loader, criterion_test, device):
    """
    Evaluates the model on a validation or test set, collecting predictions and computing the mean loss.
    Runs under torch.no_grad() to disable gradient computation for efficiency.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing batches of (inputs, labels).
        criterion_test (callable): The loss function used to score predictions against labels.
        device (torch.device): The device (CPU or GPU) on which to run computation.

    Returns:
        tuple:
            - float: The mean loss across all batches.
            - np.ndarray: A 2D array of shape (total_samples, features) containing all model predictions.
    """
    model.eval()
    test_error = 0.0
    preds_list = []
    
    for batch in loader:
        ins, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(ins)
        
        test_error += criterion_test(outputs, labels).item()
        
        # Convert to numpy and ensure it is 2D (Batch, Features)
        # Reshape to (Batch_Size, Everything Else)
        pred_np = outputs.cpu().numpy().reshape(outputs.shape[0], -1)
        preds_list.append(pred_np)
        
    # vstack will now reliably return (Total_Samples, Features)
    return test_error / len(loader), np.vstack(preds_list)


    

def export_results_to_netcdf(test_out, transformed_preds, test_dataset, test_data, size, path, model_name, use_wandb=True):
    """
    Reshapes model predictions and ground truth arrays into (time, lat, lon) format, writes them to
    a NetCDF file via Xarray, and logs the file to Weights & Biases as a versioned dataset artifact.

    Args:
        test_out (np.ndarray): Raw model predictions in transformed space, shape (N, flat_spatial).
        transformed_preds (np.ndarray): Predictions mapped back to the original space, shape (N, flat_spatial).
        test_dataset: A dataset object exposing fp (transformed truth) and fp_untransformed attributes.
        test_data: An object exposing met.time.values, providing the time coordinates for the NetCDF file.
        size (tuple of int): The spatial dimensions (height, width) used to reshape flat arrays into (lat, lon) grids.
        path (str): The base directory path for saving the NetCDF file.
        model_name (str): The model name used to locate the output subfolder and tag the artifact.
        use_wandb (bool): If True, logs the NetCDF file to W&B as a versioned artifact. Defaults to True.
    Returns:
        None
    """
    fps = test_dataset.fp.detach().numpy()

    test_out_reshaped = np.reshape(test_out, (len(test_out), size[0], size[1]))
    trans_preds_reshaped = np.reshape(transformed_preds, (len(transformed_preds), size[0], size[1]))
    fps_reshaped = np.reshape(fps, (len(fps), size[0], size[1]))
    trans_fp_reshaped = np.reshape(test_dataset.fp, (len(test_dataset.fp), size[0], size[1]))

    data_vars = {
        'predictions': (['time', "lat", "lon"], test_out_reshaped, {'space': 'transformed', 'emulated_with': model_name}),
        'trans_predictions': (['time', "lat", "lon"], trans_preds_reshaped, {'space': 'original', 'emulated_with': model_name}),
        'fp': (['time', "lat", "lon"], fps_reshaped, {'type': "truth"}),
        'trans_fp': (['time', "lat", "lon"], trans_fp_reshaped, {'space': 'transformed'})
    }

    coords = {
        'time': (['time'], test_data.met.time.values),
        'lat': (['lat'], list(range(size[0]))),
        'lon': (['lon'], list(range(size[1])))
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={'creation_date': str(datetime.now()), "model": model_name})
    netcdf_save_path = f"{path}{model_name}/sample_predictions_training.nc"
    ds.to_netcdf(netcdf_save_path)
    print("NetCDF file saved.")

    if use_wandb:
        preds_artifact = wandb.Artifact(
            name=f"{model_name}-predictions",
            type="dataset",
            description="Sample predictions saved during training"
        )
        preds_artifact.add_file(netcdf_save_path)
        wandb.log_artifact(preds_artifact)

def run_full_training(model,parameters, train_loader, test_loader, optimizer, criterion, criterion_test, 
                      test_dataset, test_data, device, epoch_so_far, losses, 
                      flux_evaluation, image_plots, image_dates, size, path, model_name, NMAE_function):
    """
    Executes the full training loop for a given number of epochs, including training and validation passes,
    metric logging, early stopping, periodic visualisation, and model checkpointing. At the end of training,
    predictions are exported to a NetCDF file and logged to Weights & Biases.

    Args:
        model (torch.nn.Module): The model to train.
        parameters (dict): A dictionary of training configuration values. Expected keys include:
            - 'learning_rate' (float): The learning rate used for logging in checkpoints.
            - 'epochs' (dict): Sub-keys 'training' (int), 'visualize' (int), 'model_saving' (int),
              and 'patience' (int) controlling loop behaviour.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation/test set.
        optimizer (torch.optim.Optimizer): The optimiser used to update model weights.
        criterion (callable): The training loss function, called as criterion(outputs, labels, true_fp).
        criterion_test (callable): A secondary loss function used for validation and display metrics.
        test_dataset: A dataset object exposing fp, inverse_transform(), and evaluate() / evaluate_flux() methods.
        test_data: An object exposing met.time.values, used when exporting results to NetCDF.
        device (torch.device): The device (CPU or GPU) on which to run computation.
        epoch_so_far (int): The epoch count to start from, allowing training to resume from a checkpoint.
        losses (dict): A dictionary of lists used to accumulate per-epoch metrics across the run.
        flux_evaluation (list of str): Flux evaluation mode names (e.g. "uniform", "checkerboard_10") passed to evaluate_flux().
        image_plots (list of int): Indices of test samples to visualise at each visualisation epoch.
        image_dates (list of str): Date strings corresponding to each index in image_plots.
        size (tuple of int): Spatial dimensions (height, width) used to reshape predictions for plotting and export.
        path (str): Base directory path for saving logs, plots, and checkpoints.
        model_name (str): The model name used for subfolder paths, filenames, and W&B artifact names.
        NMAE_function (callable): A function to compute the Normalised Mean Absolute Error,
            called as NMAE_function(predictions, truths).

    Returns:
        None
    """
    lr = parameters['learning_rate']
    num_epochs = parameters['epochs']['training']
    saving_epochs = parameters['epochs']['model_saving']
    patience_epochs = parameters['epochs']['patience']
    use_wandb = parameters.get('use_wandb', True)  # defaults to True if not set

    best_model_path = f"{path}{model_name}/{model_name}_best.pt"
    early_stopping = EarlyStopping(
    patience=patience_epochs,
    verbose=True,
    path=best_model_path,
    use_wandb=use_wandb,   
    model_name=model_name
    )

    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} ---")

        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, criterion_test, device, epoch)
        avg_test_loss, test_out = validate_and_predict(model, test_loader, criterion_test, device)

        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss
            }, step=epoch)

        early_stopping(avg_test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        log_text = f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, NMAE: {nmae_val:.4f}"
        write_to_file(log_text, path, model_name)


        if epoch % saving_epochs == 0:
            checkpoint_path = f"{path}{model_name}/{model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                'learning_rate': lr,
            }, checkpoint_path)

            if use_wandb:
                checkpoint_artifact = wandb.Artifact(
                    name=f"{model_name}-checkpoint",
                    type="model",
                    description="Model checkpoint saved during training"
                )
                checkpoint_artifact.add_file(checkpoint_path)
                wandb.log_artifact(checkpoint_artifact)

    print("Finished Training.") 

def train_and_save_model(parameters, path):
    """
    Top-level entry point for a full model training run. Handles all setup steps — including directory
    creation, W&B initialisation, data loading, dataset construction, model instantiation, and
    loss/optimiser configuration — before delegating to run_full_training() to execute the training loop.
    The model name is automatically timestamped to keep each run uniquely identifiable.

    Args:
        parameters (dict): A dictionary of all training configuration values. Expected top-level keys include:
            - 'model_name' (str): Base name for the model; a timestamp is appended at runtime.
            - 'env' (str): Environment identifier used to select data paths from config.yml.
            - 'learning_rate' (float): Learning rate passed to the AdamW optimiser.
            - 'epochs' (dict): Epoch control settings (see run_full_training for sub-keys).
            - 'variables' (dict): Keyword arguments forwarded to get_square_satellite_inputs().
            - 'dataloader_parameters' (dict): Keyword arguments forwarded to FootprintsDatasetV3().
            - 'model_parameters' (dict): Keyword arguments forwarded to GraphSatelliteForecaster().
            - 'loss_functions' (dict): Contains 'criterion' and 'criterion_test' as eval-able strings.
            - 'train_load_data' (dict): Keyword arguments for loading the training dataset.
            - 'test_load_data' (dict): Overrides applied on top of train_load_data for the test dataset.
        path (str): Base directory path under which all model output folders and files will be created.

    Returns:
        None
    """
    use_wandb = parameters.get('use_wandb', True)  # defaults to True if not set

    env = parameters['env']
    
    name_output_format = parameters['output_format']
    normalization_type = parameters['normalization']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    print(model_name)

    set_reproducibility(parameters)

    if use_wandb:
        wandb.init(
            project="BoundaryCondition-Prediction",
            config=parameters,
            tags=[]
        )

    os.makedirs(f"{path}{model_name}", exist_ok=True)
    os.mkdir(f"{path}{model_name}/training_imgs")
    training_outputs_path = f"{path}{model_name}/training_outputs"
    os.mkdir(training_outputs_path)
    f = open(f"{path}{model_name}/{model_name}_updates.txt", "x")
    f.close()

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    env_paths = config["data_paths"][env]
    base_data_path = env_paths["base_data_path"]
    fp_datadir = os.path.join(base_data_path, env_paths["fp_datadir"].lstrip("/"))
    met_datadir = os.path.join(base_data_path, env_paths["met_datadir"].lstrip("/"))
    topog_datadir = os.path.join(base_data_path, env_paths["topog_datadir"].lstrip("/"))
    landcover_datadir = os.path.join(base_data_path, env_paths["landcover_datadir"].lstrip("/"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"), path, model_name)
    write_to_file("loading data", path, model_name)

    train_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])

    shared_data_args = dict(
        load_everything=True,
        base_data_path=base_data_path,
        fp_datadir=fp_datadir,
        met_datadir=met_datadir,
        topog_args={
            "topog_path": topog_datadir,
            "landcover_path": landcover_datadir
        }
    )

    print("Load training met and fp data")
    write_to_file("Load training met and fp data", path, model_name)
    data = LoadSquareSatelliteData(**train_load_data, **shared_data_args)
    write_to_file("Successfully load training met and fp data", path, model_name)
    write_to_file("Now load test met and fp data", path, model_name)
    print("Load test met and fp data")
    test_data = LoadSquareSatelliteData(**test_load_data, **shared_data_args)
    write_to_file("Successfully load test met and fp data", path, model_name)
    write_to_file("setting up data", path, model_name)

    input_variables = parameters["variables"]
    inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)

    nan_indices_train = np.argwhere(np.isnan(inputs))
    if nan_indices_train.size > 0:
        print(f"NaNs found in training set at indices: {nan_indices_train[:10]}")

    test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
    nan_indices = np.argwhere(np.isnan(test_inputs))
    if nan_indices.size > 0:
        print(f"NaNs found in test set at indices: {nan_indices[:10]}")


    train_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # TODO: Nawid- get the train year and the test year from the trainload data 
    train_year = parse_years(train_load_data['year'])
    
    test_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    test_year = parse_years(test_load_data['year'])
    height_indices = [4,5,6,7] if parameters['auxilliary'] == 'multiple' else [4]
    

    baseline_list, outputs, auxiliary_cams = baseline_mol_correction(data,train_months, train_year,output_format=name_output_format,height_indices=height_indices)
    test_baseline_list, test_outputs, test_auxiliary_cams = baseline_mol_correction(test_data,test_months, test_year,output_format =name_output_format,height_indices=height_indices)

    outputs,(outputs_mean_values,outputs_std_values) = normalize_boundary_data(outputs)
    baseline_list, (baseline_mean_values,baseline_std_values) = normalize_boundary_data(baseline_list)
    auxiliary_cams, (auxiliary_mean_values,auxiliary_std_values) = normalize_boundary_data(auxiliary_cams)

    test_outputs, _ = normalize_boundary_data(test_outputs,outputs_norm_vals=(outputs_mean_values,outputs_std_values))
    test_baseline_list, _ = normalize_boundary_data(test_outputs,outputs_norm_vals=(baseline_mean_values,baseline_std_values))
    test_auxiliary_cams, _ = normalize_boundary_data(test_auxiliary_cams,outputs_norm_vals=((auxiliary_mean_values,auxiliary_std_values)))

    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    train_dataset = FootprintsDatasetV3(inputs, data.fp_data, input_names=names, **parameters["dataloader_parameters"])
    print(train_dataset.transform_parameters)
    test_dataset = FootprintsDatasetV3(test_inputs, test_data.fp_data, input_names=names, test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10)

    if data.dataset_format == "square":
        size = [data.size, data.size]
    if data.dataset_format == "domain":
        size = data.domain_size

    log_object_as_artifact(grid, "grid", training_outputs_path, model_name,
                           description="Grid object used during training", use_wandb=use_wandb)
    log_object_as_artifact(train_dataset.transform_parameters, "transform_parameters",
                           training_outputs_path, model_name, description="Transform parameters used in training", use_wandb=use_wandb)
    log_object_as_artifact(parameters, "training_settings", training_outputs_path, model_name,
                           file_type="json", description="Training settings and hyperparameters", use_wandb=use_wandb)

    write_to_file("setting up model", path, model_name)
    print("setting up model")

    image_plots = random.sample(list(range(len(test_inputs))), k=4)
    image_dates = np.datetime_as_string(test_data.fp_data_full.time.values[image_plots])

    lr = parameters["learning_rate"]
    print(lr)

    feature_dim = np.shape(inputs)[-1]
    aux_dim = 0

    model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_dim, **parameters["model_parameters"])
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    flux_evaluation = ["uniform", "checkerboard_10", "checkerboard_5"]
    losses = {"train": [], "test": [], "NMAE_test": [], "MSE_test_transformed": [], "NMAE_test_transformed": [], "accuracy": [], "IoU": []}
    losses.update({f"flux_{f}": {"MAE": [], "R2": []} for f in flux_evaluation})

    print("saving grids etc")

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    epoch_so_far = 0
    if torch.cuda.is_available():
        model.cuda()

    run_full_training(model, parameters, train_loader, test_loader, optimizer, criterion, criterion_test,
                      test_dataset, test_data, device, epoch_so_far, losses,
                      flux_evaluation, image_plots, image_dates, size, path, model_name, NMAE_function=NMAE_function)

    if use_wandb:
        wandb.finish()

    ## save checkpoint every 50 epochs

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "11d787a211e05ca01c50131c5724e375cd5d3364"  # <<-- REPLACE THIS
    wandb.login()

    ## make this importable!
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    path=config['user_paths']['save_models_dir']
    file_path = config['user_paths']['parameter_files_dir']

    #### 1 Set up
    parameters = load_file(file_path)
    
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    

    # Train the model with the loaded parameters
    train_and_save_model(parameters, path=path)