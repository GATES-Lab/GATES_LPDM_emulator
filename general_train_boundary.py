import sys
import copy
import os
import pickle
import random
import re
import time
import json
import argparse
from datetime import datetime

import einops
import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import BoundaryDataset
from model.data.load_data import *
from model.forecast import GraphSatelliteForecasterClassifier, GraphSatelliteForecasterConvClassifier
from model.loss_functions import *

from general_train_nawid import write_to_file, load_file, set_reproducibility, log_object_as_artifact, EarlyStopping


# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------

def normalize_boundary_data(outputs, outputs_norm_vals=None):
    """
    Normalise an array using provided or computed mean and std.

    Args:
        outputs (np.ndarray): Array to normalise.
        outputs_norm_vals (tuple, optional): (mean, std) to reuse from training.
            If None, mean and std are computed from outputs.

    Returns:
        tuple:
            - np.ndarray: Normalised array.
            - tuple: (mean, std) used for normalisation.
    """
    if outputs_norm_vals is None:
        outputs_norm_vals = (np.mean(outputs), np.std(outputs))

    outputs_mean, outputs_std = outputs_norm_vals
    normalized_outputs = (outputs - outputs_mean) / outputs_std

    return normalized_outputs, outputs_norm_vals


def denormalize(array, mean, std):
    """
    Reverse normalisation.

    Args:
        array (np.ndarray): Normalised array.
        mean (float or np.ndarray): Mean used during normalisation.
        std (float or np.ndarray): Std used during normalisation.

    Returns:
        np.ndarray: Denormalised array.
    """
    return (array * std) + mean


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def parse_years(year_str):
    """
    Parse a year string into a list of integer years.

    Args:
        year_str (str): Year string, either an exact year (e.g. '2014') or a
            bracketed range (e.g. '201[4-5]').

    Returns:
        list of int: List of years.

    Raises:
        ValueError: If the format is not recognised.
    """
    match = re.fullmatch(r'201\[(\d)-(\d)\]', year_str)
    if match:
        start, end = map(int, match.groups())
        return [2010 + i for i in range(start, end + 1)]

    if re.fullmatch(r'20\d{2}', year_str):
        return [int(year_str)]

    raise ValueError(f"Invalid year format: {year_str}")


# ---------------------------------------------------------------
# CAMS boundary condition computation
# ---------------------------------------------------------------

def baseline_mol_correction(desired_data, months, years, output_format, height_indices=None):
    """
    Compute CAMS-based boundary condition outputs and auxiliary features.

    Args:
        desired_data: Data object with fp_data_full containing particle locations and times.
        months (list of str): Months to process, e.g. ['01', '02'].
        years (list of int): Years to process.
        output_format (str): One of 'sum' or 'corrected'.
        height_indices (list of int, optional): Height levels to extract. Defaults to [4].

    Returns:
        tuple:
            - np.ndarray: Baseline list, shape (N, 4).
            - np.ndarray: Outputs, shape (N, 1).
            - np.ndarray: Auxiliary CAMS values, shape (N, 4 * len(height_indices)).
            - np.ndarray: Correction values, shape (N,).
    """
    if height_indices is None:
        height_indices = [4]

    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]

    df = pd.read_csv(
        '/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv'
    )

    baseline_list = np.zeros((total_data_points, 4))
    north_list = np.zeros(total_data_points)
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)
    total_list = np.zeros(total_data_points)
    auxiliary_cams = np.zeros((total_data_points, 4 * len(height_indices)))
    collated_corrections = np.zeros(total_data_points)

    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)
    specific_years = np.array([np.datetime64(date, 'Y').astype(int) + 1970 for date in datetime_array])
    specific_months = np.array([np.datetime64(date, 'M').astype(int) % 12 + 1 for date in datetime_array])

    for year in years:
        print(f"Processing year {year}")
        for month in months:
            cams_path = (
                f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc"
                if year > 2017
                else f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc"
            )
            cams = xr.open_dataset(cams_path)

            indices = np.where(
                (specific_years == year) & (specific_months == int(month))
            )[0]

            if len(indices) == 0:
                continue

            filtered_data = df[(df["Year"] == year) & (df["Month"] == int(month))]
            baseline_list[indices] = filtered_data.iloc[:, [3, 4, 5, 6]].values / 1000

            north_mol = np.sum(
                cams.vmr_n * desired_data.fp_data_full.particle_locations_n[:, :, indices], axis=(0, 1)
            )
            south_mol = np.sum(
                cams.vmr_s * desired_data.fp_data_full.particle_locations_s[:, :, indices], axis=(0, 1)
            )
            east_mol = np.sum(
                cams.vmr_e * desired_data.fp_data_full.particle_locations_e[:, :, indices], axis=(0, 1)
            )
            west_mol = np.sum(
                cams.vmr_w * desired_data.fp_data_full.particle_locations_w[:, :, indices], axis=(0, 1)
            )

            north_list[indices] = north_mol
            south_list[indices] = south_mol
            east_list[indices] = east_mol
            west_list[indices] = west_mol

            mid_n_index = cams.vmr_n.shape[1] // 2
            mid_e_index = cams.vmr_e.shape[1] // 2

            all_vals = []
            for h in height_indices:
                all_vals.extend([
                    cams.vmr_n.values[h, mid_n_index],
                    cams.vmr_s.values[h, mid_n_index],
                    cams.vmr_e.values[h, mid_e_index],
                    cams.vmr_w.values[h, mid_e_index],
                ])
            auxiliary_cams[indices, :] = np.array(all_vals)

            correction = np.mean(cams.vmr_s[1].values)
            total_list[indices] = (north_mol + south_mol + east_mol + west_mol).values - correction
            collated_corrections[indices] = correction

    outputs = np.stack((north_list, south_list, east_list, west_list), axis=1)

    if output_format == "sum":
        outputs = np.sum(outputs, axis=1, keepdims=True)
    elif output_format == "corrected":
        outputs = np.array(total_list).reshape(-1, 1)

    return baseline_list, outputs, auxiliary_cams, collated_corrections


# ---------------------------------------------------------------
# Data loading and normalisation
# ---------------------------------------------------------------

def load_and_normalise_boundary_data(data, months, years, output_format, height_indices,
                                     norm_vals=None):
    """
    Run baseline_mol_correction and normalise all outputs.

    Args:
        data: LoadSquareSatelliteData object.
        months (list of str): Months to process.
        years (list of int): Years to process.
        output_format (str): 'sum' or 'corrected'.
        height_indices (list of int): Height indices to extract.
        norm_vals (dict, optional): Dict with keys 'outputs', 'baselines', 'auxiliary',
            each a (mean, std) tuple. If None, normalisation values are computed from data.

    Returns:
        tuple:
            - np.ndarray: Normalised outputs.
            - np.ndarray: Normalised baseline list.
            - np.ndarray: Normalised auxiliary CAMS.
            - np.ndarray: Raw correction values.
            - dict: Normalisation values used, with keys 'outputs', 'baselines', 'auxiliary'.
    """
    baseline_list, outputs, auxiliary_cams, corrections = baseline_mol_correction(
        data, months, years, output_format=output_format, height_indices=height_indices
    )

    if norm_vals is None:
        outputs, outputs_norm = normalize_boundary_data(outputs)
        baseline_list, baselines_norm = normalize_boundary_data(baseline_list)
        auxiliary_cams, auxiliary_norm = normalize_boundary_data(auxiliary_cams)
    else:
        outputs, outputs_norm = normalize_boundary_data(outputs, norm_vals['outputs'])
        baseline_list, baselines_norm = normalize_boundary_data(baseline_list, norm_vals['baselines'])
        auxiliary_cams, auxiliary_norm = normalize_boundary_data(auxiliary_cams, norm_vals['auxiliary'])

    norm_vals_out = {
        'outputs': outputs_norm,
        'baselines': baselines_norm,
        'auxiliary': auxiliary_norm,
    }

    return outputs, baseline_list, auxiliary_cams, corrections, norm_vals_out


# ---------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------

def build_model(parameters, grid, feature_dim, aux_dim, num_lat, num_lon):
    """
    Construct the appropriate model variant based on parameters.

    Args:
        parameters (dict): Training parameters including 'network_decoder', 'num_classes',
            and 'model_parameters'.
        grid: Lat/lon grid passed to the model encoder.
        feature_dim (int): Number of input features per node.
        aux_dim (int): Number of auxiliary input features.
        num_lat (int): Number of latitude points (used by conv decoder).
        num_lon (int): Number of longitude points (used by conv decoder).

    Returns:
        torch.nn.Module: The constructed model.
    """
    num_classes = parameters['num_classes']

    if parameters['network_decoder'] == 'conv':
        print('Using conv network')
        model = GraphSatelliteForecasterConvClassifier(
            grid, whole_world=False, feature_dim=feature_dim,
            aux_dim=aux_dim, num_classes=num_classes,
            input_height=num_lat, input_width=num_lon,
            **parameters["model_parameters"]
        )
    else:
        print('Using normal network')
        model = GraphSatelliteForecasterClassifier(
            grid, whole_world=False, feature_dim=feature_dim,
            aux_dim=aux_dim, num_classes=num_classes,
            **parameters["model_parameters"]
        )

    return model


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, criterion_test, device, epoch):
    """
    Runs a single training epoch.

    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): Training data loader providing (inputs, labels) batches.
        optimizer (torch.optim.Optimizer): Optimiser for weight updates.
        criterion (callable): Training loss function.
        criterion_test (callable): Display loss function (no gradient).
        device (torch.device): Compute device.
        epoch (int): Current epoch number, used for logging.

    Returns:
        float: Mean display loss across all batches.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(loader):
        ins, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(ins)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            display_loss = criterion_test(outputs, labels)
            running_loss += display_loss.item()

        if i % 10 == 0:
            print(f"[{epoch}, {i:5d}] Loss: {running_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s")

    return running_loss / len(loader)


@torch.no_grad()
def validate_and_predict(model, loader, criterion_test, device):
    """
    Evaluates the model on a validation or test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader providing (inputs, labels) batches.
        criterion_test (callable): Loss function for scoring.
        device (torch.device): Compute device.

    Returns:
        tuple:
            - float: Mean loss across all batches.
            - np.ndarray: Predictions of shape (total_samples, features).
    """
    model.eval()
    test_error = 0.0
    preds_list = []

    for batch in loader:
        ins, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(ins)

        test_error += criterion_test(outputs, labels).item()
        pred_np = outputs.cpu().numpy().reshape(outputs.shape[0], -1)
        preds_list.append(pred_np)

    return test_error / len(loader), np.vstack(preds_list)


def run_full_training(model, parameters, train_loader, test_loader, optimizer, criterion,
                      criterion_test, device, epoch_so_far, losses, path, model_name):
    """
    Executes the full training loop with early stopping and checkpointing.

    Args:
        model (torch.nn.Module): The model to train.
        parameters (dict): Training configuration. Expected keys: 'learning_rate',
            'epochs' (with sub-keys 'training', 'model_saving', 'patience'), 'use_wandb'.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimiser.
        criterion (callable): Training loss function.
        criterion_test (callable): Validation/display loss function.
        device (torch.device): Compute device.
        epoch_so_far (int): Starting epoch, for resuming from a checkpoint.
        losses (dict): Accumulator dict with 'train' and 'test' lists.
        path (str): Base directory for saving checkpoints and logs.
        model_name (str): Model identifier used in file paths and W&B artifact names.

    Returns:
        None
    """
    lr = parameters['learning_rate']
    num_epochs = parameters['epochs']['training']
    saving_epochs = parameters['epochs']['model_saving']
    patience_epochs = parameters['epochs']['patience']
    use_wandb = parameters.get('use_wandb', True)

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

        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, criterion_test, device, epoch
        )
        avg_test_loss, _ = validate_and_predict(model, test_loader, criterion_test, device)

        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss,
            }, step=epoch)

        early_stopping(avg_test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        write_to_file(
            f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}",
            path, model_name
        )

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


# ---------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------

def train_and_save_model(parameters, path):
    """
    Top-level entry point for a full training run.

    Args:
        parameters (dict): Full training configuration. Expected top-level keys:
            'model_name', 'env', 'learning_rate', 'epochs', 'variables',
            'dataloader_parameters', 'model_parameters', 'loss_functions',
            'train_load_data', 'test_load_data', 'output_format', 'normalization',
            'use_baselines', 'num_classes', 'network_decoder', 'auxiliary', 'use_wandb'.
        path (str): Base directory for saving all outputs.

    Returns:
        None
    """
    use_wandb = parameters.get('use_wandb', True)
    name_output_format = parameters['output_format']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    print(model_name)

    set_reproducibility(parameters)

    if use_wandb:
        wandb.init(project="BoundaryCondition-Prediction", config=parameters, tags=[])

    # Create output directories
    os.makedirs(f"{path}{model_name}", exist_ok=True)
    os.makedirs(f"{path}{model_name}/training_imgs", exist_ok=True)
    training_outputs_path = f"{path}{model_name}/training_outputs"
    os.makedirs(training_outputs_path, exist_ok=True)
    open(f"{path}{model_name}/{model_name}_updates.txt", "x").close()

    # Load environment paths from config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    env = parameters['env']
    env_paths = config["data_paths"][env]
    base_data_path = env_paths["base_data_path"]
    fp_datadir = os.path.join(base_data_path, env_paths["fp_datadir"].lstrip("/"))
    met_datadir = os.path.join(base_data_path, env_paths["met_datadir"].lstrip("/"))
    topog_datadir = os.path.join(base_data_path, env_paths["topog_datadir"].lstrip("/"))
    landcover_datadir = os.path.join(base_data_path, env_paths["landcover_datadir"].lstrip("/"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(
        f"using device {device}, starting at {datetime.now().strftime('%d/%m/%y %H:%M:%S')}",
        path, model_name
    )

    # Build data loading args
    train_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])

    shared_data_args = dict(
        load_everything=True,
        base_data_path=base_data_path,
        fp_datadir=fp_datadir,
        met_datadir=met_datadir,
        topog_args={"topog_path": topog_datadir, "landcover_path": landcover_datadir}
    )

    # Load data
    write_to_file("Load training met and fp data", path, model_name)
    data = LoadSquareSatelliteData(**train_load_data, **shared_data_args)
    write_to_file("Successfully loaded training data. Now loading test data.", path, model_name)
    test_data = LoadSquareSatelliteData(**test_load_data, **shared_data_args)
    write_to_file("Successfully loaded test data.", path, model_name)

    # Get inputs
    input_variables = parameters["variables"]
    inputs, names = get_square_satellite_inputs(
        data, **input_variables, return_variable_names=True, return_asarray=True
    )
    test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)

    for label, arr in [("training", inputs), ("test", test_inputs)]:
        nan_indices = np.argwhere(np.isnan(arr))
        if nan_indices.size > 0:
            print(f"NaNs found in {label} set at indices: {nan_indices[:10]}")

    # Compute boundary condition outputs and normalise
    # Note: 'auxiliary' key used here — check parameter file uses this spelling
    height_indices = [4, 5, 6, 7] if parameters.get('auxiliary') == 'multiple' else [4]
    all_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    train_year = parse_years(train_load_data['year'])
    test_year = parse_years(test_load_data['year'])

    outputs, baseline_list, auxiliary_cams, _, norm_vals = load_and_normalise_boundary_data(
        data, all_months, train_year, name_output_format, height_indices
    )
    test_outputs, test_baseline_list, test_auxiliary_cams, _, _ = load_and_normalise_boundary_data(
        test_data, all_months, test_year, name_output_format, height_indices,
        norm_vals=norm_vals
    )

    # Save auxiliary normalisation values into checkpoint-compatible format
    outputs_mean_values, outputs_std_values = norm_vals['outputs']
    baseline_mean_values, baseline_std_values = norm_vals['baselines']
    auxiliary_mean_values, auxiliary_std_values = norm_vals['auxiliary']

    # Build grid and datasets
    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    use_baselines = parameters['use_baselines']
    aux_dim = auxiliary_cams.shape[1] if use_baselines else 0

    train_dataset = BoundaryDataset(
        inputs, auxiliary_cams, outputs,
        use_baselines=use_baselines, input_names=names,
        **parameters["dataloader_parameters"]
    )
    test_dataset = BoundaryDataset(
        test_inputs, test_auxiliary_cams, test_outputs,
        use_baselines=use_baselines, input_names=names,
        test_mode=train_dataset.transform_parameters,
        **parameters["dataloader_parameters"]
    )
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5)

    # Log artifacts
    log_object_as_artifact(grid, "grid", training_outputs_path, model_name,
                           description="Grid used during training", use_wandb=use_wandb)
    log_object_as_artifact(train_dataset.transform_parameters, "transform_parameters",
                           training_outputs_path, model_name,
                           description="Transform parameters used in training", use_wandb=use_wandb)
    log_object_as_artifact(parameters, "training_settings", training_outputs_path, model_name,
                           file_type="json", description="Training settings", use_wandb=use_wandb)

    # Save auxiliary normalisation values so inference can reuse them without reloading train data
    auxiliary_norm_vals = {
        "auxiliary_mean": auxiliary_mean_values,
        "auxiliary_std": auxiliary_std_values
    }
    log_object_as_artifact(auxiliary_norm_vals, "auxiliary_norm_vals", training_outputs_path,
                           model_name, file_type="json",
                           description="Auxiliary normalisation values", use_wandb=use_wandb)

    # Build model
    feature_dim = np.shape(inputs)[-1]
    num_lat, num_lon = len(data.met.lat.values), len(data.met.lon.values)

    model = build_model(parameters, grid, feature_dim, aux_dim, num_lat, num_lon)
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    optimizer = optim.AdamW(model.parameters(), lr=parameters["learning_rate"])

    normalization_vals = {"outputs_mean": outputs_mean_values, "outputs_std": outputs_std_values}
    baseline_normalization_vals = {"baselines_mean": baseline_mean_values, "baselines_std": baseline_std_values}
    losses = {"train": [], "test": []}

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    if torch.cuda.is_available():
        model.cuda()

    run_full_training(
        model, parameters, train_loader, test_loader, optimizer,
        criterion, criterion_test, device, epoch_so_far=0,
        losses=losses, path=path, model_name=model_name
    )

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "11d787a211e05ca01c50131c5724e375cd5d3364"
    wandb.login()

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    path = config['user_paths']['save_models_dir']
    file_path = config['user_paths']['parameter_files_dir']

    parameters = load_file(file_path)
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    train_and_save_model(parameters, path=path)