import sys
import os
import copy
import time
import json
import pickle
import random
import re
import argparse
from datetime import datetime
 
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
 
import einops
import yaml
import wandb
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from sklearn.metrics import mean_squared_error, r2_score
 
sys.path.insert(0, "/user/work/ef17148/GCN/graphnet/")
sys.path.insert(1, "/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/")
 
from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import BoundaryDataset
from model.data.load_data import *  # provides LoadSquareSatelliteData, get_square_satellite_inputs, get_grid
from model.forecast import (
    GraphSatelliteForecasterClassifier,
    GraphSatelliteForecasterConvClassifier,
)
from model.loss_functions import *
 
from general_train_nawid import (
    write_to_file,
    load_file,
    set_reproducibility,
    log_object_as_artifact,
    EarlyStopping,
)
 
 
def normalize_boundary_data(outputs, outputs_norm_vals=None):
    """
    Normalize an array using a (mean, std) pair.
 
    If `outputs_norm_vals` is provided, it is reused (typical for test data
    normalised with training statistics). Otherwise mean/std are computed
    from `outputs`.
 
    Returns
    -------
    normalized_outputs : np.ndarray
    outputs_norm_vals  : tuple(mean, std)
    """
    if outputs_norm_vals is None:
        outputs_norm_vals = (np.mean(outputs), np.std(outputs))
 
    outputs_mean, outputs_std = outputs_norm_vals
    normalized_outputs = (outputs - outputs_mean) / outputs_std
    return normalized_outputs, outputs_norm_vals
 
 
def baseline_mol_correction(desired_data, months, years, output_format, height_indices=None):
    """
    Compute CAMS-based baseline & boundary mole-fraction inputs and outputs.
 
    Parameters
    ----------
    desired_data : object
        Input data object containing particle locations and times.
    months : list[int|str]
        Months to process.
    years : list[int]
        Years to process.
    output_format : str
        Either 'sum' or 'corrected'.
    height_indices : list[int], optional
        Height indices to extract (default = [4]).
 
    Returns
    -------
    baseline_list, outputs, auxiliary_cams, collated_corrections
    """
    if height_indices is None:
        height_indices = [4]
 
    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]
 
    df = pd.read_csv(
        "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/"
        "CH4_Semihemispheric_modelled_mole_fractions.csv"
    )
    baseline_list = np.zeros((total_data_points, 4))
 
    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)
    specific_years = np.array([np.datetime64(date, "Y").astype(int) + 1970 for date in datetime_array])
    specific_months = np.array([np.datetime64(date, "M").astype(int) % 12 + 1 for date in datetime_array])
 
    north_list = np.zeros(total_data_points)
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)
    total_list = np.zeros(total_data_points)
 
    num_heights = len(height_indices)
    auxiliary_cams = np.zeros((total_data_points, 4 * num_heights))
    collated_corrections = np.zeros(total_data_points)
 
    for year in years:
        print("year", year)
        for month in months:
            if year > 2017:
                cams = xr.open_dataset(
                    f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/"
                    f"ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc"
                )
            else:
                cams = xr.open_dataset(
                    f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/"
                    f"ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc"
                )
 
            desired_year = year
            desired_month = month
            print("desired month", desired_month)
 
            indices = np.where(
                (specific_years == desired_year) & (specific_months == int(desired_month))
            )[0]
 
            if len(indices) == 0:
                continue
 
            filtered_data = df[(df["Year"] == desired_year) & (df["Month"] == int(desired_month))]
            selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values
            baseline_list[indices] = selected_columns / 1000  # ppt -> ppm
 
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
                n_val = cams.vmr_n.values[h, mid_n_index]
                s_val = cams.vmr_s.values[h, mid_n_index]
                e_val = cams.vmr_e.values[h, mid_e_index]
                w_val = cams.vmr_w.values[h, mid_e_index]
                all_vals.extend([n_val, s_val, e_val, w_val])
 
            auxiliary_cams[indices, :] = np.array(all_vals)
 
            correction = np.mean(cams.vmr_s[1].values)
            total_list[indices] = (
                (north_mol + south_mol + east_mol + west_mol).values - correction
            )
            collated_corrections[indices] = correction
 
    outputs = np.stack((north_list, south_list, east_list, west_list), axis=1)
 
    if output_format == "sum":
        outputs = np.sum(outputs, axis=1, keepdims=True)
    elif output_format == "corrected":
        outputs = np.array(total_list).reshape(-1, 1)
 
    return baseline_list, outputs, auxiliary_cams, collated_corrections
 
 
def parse_years(year_str):
    """Parse a year string of the form '2014' or '201[4-5]' into a list of ints."""
    match = re.fullmatch(r"201\[(\d)-(\d)\]", year_str)
    if match:
        start, end = map(int, match.groups())
        return [2010 + i for i in range(start, end + 1)]
 
    if re.fullmatch(r"20\d{2}", year_str):
        return [int(year_str)]
 
    raise ValueError(f"Invalid year format: {year_str}")
 
 
def train_one_epoch(model, loader, optimizer, criterion, criterion_test, device, epoch):
    """Run a single training epoch and return the mean display loss."""
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
            print(
                f"[{epoch}, {i:5d}] Loss: {running_loss/(i+1):.3f} "
                f"Time: {time.time()-start_time:.1f}s"
            )
 
    return running_loss / len(loader)
 
 
@torch.no_grad()
def validate_and_predict(model, loader, criterion_test, device):
    """Evaluate the model and collect predictions as a 2D (N, F) numpy array."""
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
 
 
def run_full_training(
    model,
    parameters,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    criterion_test,
    device,
    epoch_so_far,
    losses,
    path,
    model_name,
):
    """Execute the full training loop with early stopping and periodic checkpointing."""
    lr = parameters["learning_rate"]
    num_epochs = parameters["epochs"]["training"]
    saving_epochs = parameters["epochs"]["model_saving"]
    patience_epochs = parameters["epochs"]["patience"]
    use_wandb = parameters.get("use_wandb", True)
 
    best_model_path = f"{path}{model_name}/{model_name}_best.pt"
    early_stopping = EarlyStopping(
        patience=patience_epochs,
        verbose=True,
        path=best_model_path,
        use_wandb=use_wandb,
        model_name=model_name,
    )
 
    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} ---")
 
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, criterion_test, device, epoch
        )
        avg_test_loss, _test_out = validate_and_predict(
            model, test_loader, criterion_test, device
        )
 
        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)
 
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "test/loss": avg_test_loss,
                },
                step=epoch,
            )
 
        early_stopping(avg_test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break
 
        log_text = (
            f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {avg_test_loss:.4f}"
        )
        write_to_file(log_text, path, model_name)
 
        if epoch % saving_epochs == 0:
            checkpoint_path = f"{path}{model_name}/{model_name}_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": losses,
                    "learning_rate": lr,
                },
                checkpoint_path,
            )
 
            if use_wandb:
                checkpoint_artifact = wandb.Artifact(
                    name=f"{model_name}-checkpoint",
                    type="model",
                    description="Model checkpoint saved during training",
                )
                checkpoint_artifact.add_file(checkpoint_path)
                wandb.log_artifact(checkpoint_artifact)
 
    print("Finished Training.")
 
 
def train_and_save_model(parameters, path):
    """Top-level entry point: set up env, data, model, and run training."""
    use_wandb = parameters.get("use_wandb", True)
 
    env = parameters["env"]
    name_output_format = parameters["output_format"]
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    print(model_name)
 
    set_reproducibility(parameters)
 
    if use_wandb:
        wandb.init(
            project="BoundaryCondition-Prediction",
            config=parameters,
            tags=[],
        )
 
    os.makedirs(f"{path}{model_name}", exist_ok=True)
    os.mkdir(f"{path}{model_name}/training_imgs")
    training_outputs_path = f"{path}{model_name}/training_outputs"
    os.mkdir(training_outputs_path)
    with open(f"{path}{model_name}/{model_name}_updates.txt", "x"):
        pass
 
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
 
    env_paths = config["data_paths"][env]
    base_data_path = env_paths["base_data_path"]
    fp_datadir = os.path.join(base_data_path, env_paths["fp_datadir"].lstrip("/"))
    met_datadir = os.path.join(base_data_path, env_paths["met_datadir"].lstrip("/"))
    topog_datadir = os.path.join(base_data_path, env_paths["topog_datadir"].lstrip("/"))
    landcover_datadir = os.path.join(base_data_path, env_paths["landcover_datadir"].lstrip("/"))
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(
        f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"),
        path,
        model_name,
    )
    write_to_file("loading data", path, model_name)
 
    train_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])
 
    shared_data_args = dict(
        load_everything=True,
        base_data_path=base_data_path,
        fp_datadir=fp_datadir,
        met_datadir=met_datadir,
        topog_args={"topog_path": topog_datadir, "landcover_path": landcover_datadir},
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
    inputs, names = get_square_satellite_inputs(
        data, **input_variables, return_variable_names=True, return_asarray=True
    )
 
    nan_indices_train = np.argwhere(np.isnan(inputs))
    if nan_indices_train.size > 0:
        print(f"NaNs found in training set at indices: {nan_indices_train[:10]}")
 
    test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
    nan_indices = np.argwhere(np.isnan(test_inputs))
    if nan_indices.size > 0:
        print(f"NaNs found in test set at indices: {nan_indices[:10]}")
 
    train_months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    train_year = parse_years(train_load_data["year"])
 
    test_months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    test_year = parse_years(test_load_data["year"])
    height_indices = [4, 5, 6, 7] if parameters["auxilliary"] == "multiple" else [4]
 
    baseline_list, outputs, auxiliary_cams, _train_corrections = baseline_mol_correction(
        data, train_months, train_year, output_format=name_output_format, height_indices=height_indices
    )
    test_baseline_list, test_outputs, test_auxiliary_cams, _test_corrections = baseline_mol_correction(
        test_data, test_months, test_year, output_format=name_output_format, height_indices=height_indices
    )
 
    outputs, (outputs_mean_values, outputs_std_values) = normalize_boundary_data(outputs)
    baseline_list, (baseline_mean_values, baseline_std_values) = normalize_boundary_data(baseline_list)
    auxiliary_cams, (auxiliary_mean_values, auxiliary_std_values) = normalize_boundary_data(auxiliary_cams)
 
    test_outputs, _ = normalize_boundary_data(
        test_outputs, outputs_norm_vals=(outputs_mean_values, outputs_std_values)
    )
    test_baseline_list, _ = normalize_boundary_data(
        test_baseline_list, outputs_norm_vals=(baseline_mean_values, baseline_std_values)
    )
    test_auxiliary_cams, _ = normalize_boundary_data(
        test_auxiliary_cams, outputs_norm_vals=(auxiliary_mean_values, auxiliary_std_values)
    )
 
    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))
    use_baselines = parameters["use_baselines"]
    aux_dim = auxiliary_cams.shape[1] if use_baselines else 0
    print("aux index", aux_dim)
 
    train_batch_size = 5
    test_batch_size = 5
 
    train_dataset = BoundaryDataset(
        inputs,
        auxiliary_cams,
        outputs,
        use_baselines=use_baselines,
        input_names=names,
        **parameters["dataloader_parameters"],
    )
    test_dataset = BoundaryDataset(
        test_inputs,
        test_auxiliary_cams,
        test_outputs,
        use_baselines=use_baselines,
        input_names=names,
        test_mode=train_dataset.transform_parameters,
        **parameters["dataloader_parameters"],
    )
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
 
    if data.dataset_format == "square":
        size = [data.size, data.size]
    elif data.dataset_format == "domain":
        size = data.domain_size
 
    log_object_as_artifact(
        grid, "grid", training_outputs_path, model_name,
        description="Grid object used during training", use_wandb=use_wandb,
    )
    log_object_as_artifact(
        train_dataset.transform_parameters, "transform_parameters",
        training_outputs_path, model_name,
        description="Transform parameters used in training", use_wandb=use_wandb,
    )
    log_object_as_artifact(
        parameters, "training_settings", training_outputs_path, model_name,
        file_type="json", description="Training settings and hyperparameters",
        use_wandb=use_wandb,
    )
 
    write_to_file("setting up model", path, model_name)
    print("setting up model")
 
    lr = parameters["learning_rate"]
    print(lr)
 
    feature_dim = np.shape(inputs)[-1]
    num_classes = parameters["num_classes"]
    num_lat, num_lon = len(data.met.lat.values), len(data.met.lon.values)
 
    if parameters["network_decoder"] == "conv":
        print("Using conv network")
        model = GraphSatelliteForecasterConvClassifier(
            grid,
            whole_world=False,
            feature_dim=feature_dim,
            aux_dim=aux_dim,
            num_classes=num_classes,
            input_height=num_lat,
            input_width=num_lon,
            **parameters["model_parameters"],
        )
    else:
        print("Using normal network")
        model = GraphSatelliteForecasterClassifier(
            grid,
            whole_world=False,
            feature_dim=feature_dim,
            aux_dim=aux_dim,
            num_classes=num_classes,
            **parameters["model_parameters"],
        )
 
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
 
    losses = {"train": [], "test": [], "individual_summed_test_MAE": []}
 
    print("saving grids etc")
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
 
    epoch_so_far = 0
    if torch.cuda.is_available():
        model.cuda()
 
    run_full_training(
        model,
        parameters,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        criterion_test,
        device,
        epoch_so_far,
        losses,
        path,
        model_name,
    )
 
    if use_wandb:
        wandb.finish()
 
 
if __name__ == "__main__":
    # NOTE: Removed hardcoded WANDB key. Set WANDB_API_KEY in your environment, or run
    # `wandb login` once on the machine. Hardcoding secrets in source is a security risk.
    '''
    if "WANDB_API_KEY" not in os.environ:
        print(
            "Warning: WANDB_API_KEY is not set in the environment. "
            "Either export it or run `wandb login` before training."
        )
    '''
    os.environ["WANDB_API_KEY"] = "11d787a211e05ca01c50131c5724e375cd5d3364"
    wandb.login()
 
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
 
    path = config["user_paths"]["save_models_dir"]
    file_path = config["user_paths"]["parameter_files_dir"]
 
    parameters = load_file(file_path)
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)
 
    print("PARAMETERS:")
    print(parameters)
 
    train_and_save_model(parameters, path=path)