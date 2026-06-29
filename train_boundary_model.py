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



import sys


import matplotlib.pyplot as plt

import numpy as np
import torch
import xarray as xr

import random
import copy
import wandb


sys.path.insert(0, "/user/work/yl18410/new_graphnet")
sys.path.insert(1, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")
from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
#from model.data.dataloader_graphnet import *
#from model.data.load_data import *
from model.loss_functions import *

from gates.training.training_background import format_aux_data, normalize_boundary_data, denormalize

import time
from datetime import datetime

import argparse

import random
from pathlib import Path

import gates.training.training as gates_training
import gates.training.training_background as gates_training_background
import gates
from gates.data.load_data import get_grid
from gates.training.training_dataclasses import PathContext, TrainingContext, BoundaryTrainingContext

from gates.training.training_helperfuns import load_parameter_file, save_object, write_to_file, save_training_plots, export_results_to_netcdf, save_wandb_artifact, set_reproducibility


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def _denormalized_mae_per_class(outputs, labels, output_norm):
    """Per-class MAE in denormalised (physical) units; returns shape (num_classes,).

    Averages the absolute error over the batch, keeping the output (class) dimension,
    so each boundary direction (e.g. north/south/east/west when num_classes=4) is
    reported separately. The overall MAE is just the mean of these.
    """
    mean, std = output_norm
    diff = torch.abs(denormalize(outputs, mean, std) - denormalize(labels, mean, std))
    return diff.reshape(diff.shape[0], -1).mean(dim=0)


def train_one_epoch(model, loader, optimizer, criterion, criterion_test, device, epoch, output_norm=None):
    """Run one training epoch.

    Returns ``(avg_loss, avg_mae_denorm, avg_mae_denorm_per_class)``. ``avg_loss`` is
    the mean normalised criterion_test loss. When ``output_norm`` (the (mean, std)
    used to normalise the boundary outputs) is provided, ``avg_mae_denorm`` is the
    overall MAE in denormalised (physical) units and ``avg_mae_denorm_per_class`` is
    a numpy array of the per-class (per-direction) MAE; both are ``None`` otherwise.
    """
    model.train()
    running_loss = 0.0
    running_mae_denorm = 0.0
    running_mae_per_class = None
    start_time = time.time()
    n_batches = 0

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
            if output_norm is not None:
                per_class = _denormalized_mae_per_class(outputs, labels, output_norm).cpu().numpy()
                running_mae_per_class = per_class if running_mae_per_class is None else running_mae_per_class + per_class
                running_mae_denorm += float(per_class.mean())

        n_batches += 1

        if i % 10 == 0:
            msg = f"[{epoch}, {i:5d}] Loss: {running_loss/(i+1):.3f}"
            if output_norm is not None:
                msg += f" MAE(denorm): {running_mae_denorm/(i+1):.3e}"
            print(f"{msg} Time: {time.time()-start_time:.1f}s")

    avg_loss = running_loss / max(n_batches, 1)
    if output_norm is not None:
        avg_mae_denorm = running_mae_denorm / max(n_batches, 1)
        avg_mae_denorm_per_class = running_mae_per_class / max(n_batches, 1)
    else:
        avg_mae_denorm = None
        avg_mae_denorm_per_class = None
    return avg_loss, avg_mae_denorm, avg_mae_denorm_per_class


@torch.no_grad()
def validate_and_predict(model, loader, criterion_test, device, output_norm=None):
    """Validate the model.

    Returns ``(avg_loss, avg_mae_denorm, avg_mae_denorm_per_class, preds)``. ``avg_loss``
    is the mean normalised criterion_test loss; ``avg_mae_denorm`` is the overall MAE in
    denormalised (physical) units and ``avg_mae_denorm_per_class`` is the per-class
    (per-direction) MAE as a numpy array (both ``None`` if ``output_norm`` is not
    provided); ``preds`` are the (normalised) predictions stacked across batches.
    """
    model.eval()
    test_error = 0.0
    test_mae_denorm = 0.0
    test_mae_per_class = None
    preds_list = []
    n_batches = 0

    for batch in loader:
        ins, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(ins)

        test_error += criterion_test(outputs, labels).item()
        if output_norm is not None:
            per_class = _denormalized_mae_per_class(outputs, labels, output_norm).cpu().numpy()
            test_mae_per_class = per_class if test_mae_per_class is None else test_mae_per_class + per_class
            test_mae_denorm += float(per_class.mean())
        pred_np = outputs.cpu().numpy().reshape(outputs.shape[0], -1)
        preds_list.append(pred_np)
        n_batches += 1

    avg_error = test_error / max(n_batches, 1)
    if output_norm is not None:
        avg_mae_denorm = test_mae_denorm / max(n_batches, 1)
        avg_mae_denorm_per_class = test_mae_per_class / max(n_batches, 1)
    else:
        avg_mae_denorm = None
        avg_mae_denorm_per_class = None
    return avg_error, avg_mae_denorm, avg_mae_denorm_per_class, np.vstack(preds_list)


def run_full_training(model, train_loader, test_loader, model_ctx, training_ctx, paths_ctx, losses, output_norm=None, class_names=None):
    """
    Executes the full training loop with early stopping and checkpointing,
    using context dataclasses for configuration.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation data loader.
        model_ctx (ModelContext): Model configuration including optimizer, criterion,
            early stopping, and epoch settings.
        training_ctx (BoundaryTrainingContext): Training context including device,
            use_wandb flag, and other training parameters.
        paths_ctx (PathContext): Path context including model_path, model_name,
            and updates_path.
        losses (dict): Accumulator dict with 'train' and 'test' lists.
        output_norm (tuple, optional): The (mean, std) used to normalise the boundary
            outputs. When provided, the train/test MAE in denormalised (physical) units
            is also computed and logged under the 'train_mae_denorm'/'test_mae_denorm' keys.
        class_names (list[str], optional): Names of the output classes/directions
            (e.g. ["north","south","east","west"] for num_classes=4). When given with
            more than one class, the per-direction denormalised MAE is also logged.

    Returns:
        None
    """
    # Track the denormalised (physical-unit) MAE alongside the normalised loss.
    log_per_class = output_norm is not None and class_names is not None and len(class_names) > 1
    if output_norm is not None:
        losses.setdefault("train_mae_denorm", [])
        losses.setdefault("test_mae_denorm", [])
    if log_per_class:
        for name in class_names:
            losses.setdefault(f"train_mae_denorm_{name}", [])
            losses.setdefault(f"test_mae_denorm_{name}", [])

    updates_path = paths_ctx.model_path / f"{paths_ctx.model_name}_updates.txt"
    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx
        print(f"\n--- Start Epoch: {epoch} ---")

        avg_train_loss, avg_train_mae_denorm, avg_train_mae_per_class = train_one_epoch(
            model, train_loader, model_ctx.optimizer,
            model_ctx.criterion, model_ctx.criterion_test,
            model_ctx.device, epoch, output_norm=output_norm
        )
        avg_test_loss, avg_test_mae_denorm, avg_test_mae_per_class, _ = validate_and_predict(
            model, test_loader, model_ctx.criterion_test, model_ctx.device, output_norm=output_norm
        )

        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)
        if output_norm is not None:
            losses["train_mae_denorm"].append(avg_train_mae_denorm)
            losses["test_mae_denorm"].append(avg_test_mae_denorm)
        if log_per_class:
            for c, name in enumerate(class_names):
                losses[f"train_mae_denorm_{name}"].append(float(avg_train_mae_per_class[c]))
                losses[f"test_mae_denorm_{name}"].append(float(avg_test_mae_per_class[c]))

        if training_ctx.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss,
            }
            if output_norm is not None:
                log_dict["train/mae_denorm"] = avg_train_mae_denorm
                log_dict["test/mae_denorm"] = avg_test_mae_denorm
            if log_per_class:
                for c, name in enumerate(class_names):
                    log_dict[f"train/mae_denorm_{name}"] = float(avg_train_mae_per_class[c])
                    log_dict[f"test/mae_denorm_{name}"] = float(avg_test_mae_per_class[c])
            wandb.log(log_dict, step=epoch)

        model_ctx.early_stopping(avg_test_loss, model)

        if model_ctx.early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        update_msg = f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        if output_norm is not None:
            update_msg += (
                f", Train MAE (denorm): {avg_train_mae_denorm:.4e}, "
                f"Test MAE (denorm): {avg_test_mae_denorm:.4e}"
            )
        if log_per_class:
            per_dir = ", ".join(
                f"{name}: {float(avg_test_mae_per_class[c]):.4e}" for c, name in enumerate(class_names)
            )
            update_msg += f" | Test MAE (denorm) per direction -> {per_dir}"
        write_to_file(update_msg, updates_path)

        if epoch % model_ctx.epochs_save == 0:
            checkpoint_path = paths_ctx.model_path / f"{paths_ctx.model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_ctx.optimizer.state_dict(),
                'loss': losses,
                'learning_rate': model_ctx.lr,
            }, checkpoint_path)

            if training_ctx.use_wandb:
                checkpoint_artifact = wandb.Artifact(
                    name=f"{paths_ctx.model_name}-checkpoint",
                    type="model",
                    description="Model checkpoint saved during training"
                )
                checkpoint_artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(checkpoint_artifact)

    print("Finished Training.")


# ---------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------

def train_and_save_model(parameters, model_save_dir):
    """
    Top-level entry point for a full training run.
    """
    use_wandb = parameters.get('use_wandb', True)
    name_output_format = 'corrected'
    cfg = gates.config.get_config()
    verbose = parameters.get("verbose", True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    print(f"Initialising model run for model_name: {model_name}")

    paths_ctx = PathContext(
        model_save_dir=model_save_dir,
        model_name=model_name,
        model_path=model_path
    )
    paths_ctx.make_dirs()

    seed = parameters.get("seed", 34)
    set_reproducibility(seed)

    if use_wandb:
        wandb_project = parameters.get("wandb", {}).get("project", None)
        wandb_entity = parameters.get("wandb", {}).get("entity", None)
        wandb_tags = parameters.get("wandb", {}).get("tags", [])
        if wandb_project is None or wandb_entity is None:
            print("Warning: 'use_wandb' is True but no 'wandb.project' or 'wandb.entity' specified. W&B will not be initialised.")
            use_wandb = False
            parameters["use_wandb"] = False
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config=parameters,
                tags=wandb_tags
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at " + datetime.now().strftime("%d/%m/%y %H:%M:%S"), paths_ctx.updates_path)
    write_to_file("loading data", paths_ctx.updates_path)

    train_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params.update(parameters["test_load_data"])
    input_variables = parameters["variables"]

    datapath_args = paths_ctx.resolve_datapath_args(parameters)

    print("Loading met, fp AND BACKGROUND data for model", model_name)
    write_to_file("Load training and testing met, fp and background data", paths_ctx.updates_path)

    client, cluster = gates_training.make_cluster()
    
    load_monthly = True # moving towards always using monthly loading for memory efficiency
    #load_monthly = parameters.get("load_data_monthly", False)

    background_setup = parameters.get("background_setup", {})
    
    default_bg_params = {
        "detrend": True,
        "use_auxiliary_bc": True,
        "auxilary_bc_levels":[4, 5, 6, 7]}

    # update
    background_params = default_bg_params.copy()
    background_params.update(background_setup)
    parameters["background_setup"] = background_params

    num_classes = parameters["model_parameters"].get("num_classes", 1)
    if num_classes not in [1, 4]:
        print("Warning: num_classes is set to a value other than 1 or 4. Defaulting to 1 (summed output).")
        num_classes = 1
        parameters["model_parameters"]["num_classes"] = 1
        

    if num_classes == 1:
        if background_params["detrend"]:
            print("extracting background mole fraction, summed across the four boundaries, and detrended")
        else:
            print("extracting background mole fraction, summed across the four boundaries, without detrending")
    elif num_classes == 4:
        print("extracting background mole fraction for each boundary separately, without detrending")


    train_fp_data, train_inputs, train_bgs, train_aux_cams_data = gates_training_background.load_GATES_data_with_bg(
            train_load_data_params, input_variables=input_variables,
            datapath_args=datapath_args, verbose=verbose,
            load_into_memory=parameters.get("load_into_memory", False), detrend=background_params["detrend"], use_aux_bc=background_params["use_auxiliary_bc"], aux_indeces=background_params["auxilary_bc_levels"]
        )

    

    
    write_to_file(f"Successfully loaded training data with {len(train_fp_data.time)} time samples", paths_ctx.updates_path)
    print("Successfully load training met and fp data with", len(train_fp_data.time), "time samples")


    test_fp_data, test_inputs, test_bgs, test_aux_cams_data = gates_training_background.load_GATES_data_with_bg(
            test_load_data_params, input_variables=input_variables,
            datapath_args=datapath_args, verbose=verbose,
            load_into_memory=parameters.get("load_into_memory", False), detrend=background_params["detrend"], use_aux_bc=background_params["use_auxiliary_bc"], aux_indeces=background_params["auxilary_bc_levels"]
        )

    if num_classes == 1:
        train_bgs = train_bgs[["summed"]].to_dataarray()
        test_bgs = test_bgs[["summed"]].to_dataarray()
        class_names = ["summed"]
    elif num_classes == 4:
        train_bgs = train_bgs[["north", "south", "east", "west"]].to_dataarray()
        test_bgs = test_bgs[["north", "south", "east", "west"]].to_dataarray()
        class_names = ["north", "south", "east", "west"]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    write_to_file(f"Successfully loaded test data with {len(test_fp_data.time)} time samples", paths_ctx.updates_path)
    print("Successfully load test met and fp data with", len(test_fp_data.time), "time samples")


    write_to_file("scaling data and setting up dataloaders", paths_ctx.updates_path)
    if cluster is not None:
        cluster.close()
        client.close()

    use_auxiliary_bc = background_params["use_auxiliary_bc"]
    if use_auxiliary_bc:
        train_aux_cams_data = format_aux_data(train_aux_cams_data, time_coord=train_fp_data.time)
        test_aux_cams_data = format_aux_data(test_aux_cams_data, time_coord=test_fp_data.time)
    
    norm_train_bgs, norm_train_aux_data, norm_vals = normalize_boundary_data(train_bgs, aux_data=train_aux_cams_data)
    norm_test_bgs, norm_test_aux_data, norm_vals = normalize_boundary_data(test_bgs, aux_data=test_aux_cams_data, norm_vals=norm_vals)


    save_object(norm_vals, "norm_vals", paths_ctx.training_outputs_path, model_name,
            file_type="json", description=f"Normalisation values for boundary condition outputs in model {model_name}",
            use_wandb=use_wandb)

    
    #aux_dim = auxiliary_cams.sizes["aux"] if use_baselines and auxiliary_cams is not None else 0
    #parameters["aux_dim"] = aux_dim

    num_features = train_inputs.shape[-1]
    feature_dim = num_features
    if use_auxiliary_bc:
        num_features = num_features+ train_aux_cams_data.aux.shape[0]
        aux_dim = train_aux_cams_data.aux.shape[0]
    else:
        aux_dim = 0
    parameters["num_features"] = num_features
    print("using num_features =", num_features, "out of which aux_dim =", aux_dim)
    
    if use_wandb:
        wandb.summary.update({
            "num_training_samples": len(train_fp_data.time),
            "num_testing_samples": len(test_fp_data.time),
            "num_features": num_features,
        })
    
    train_loader, test_loader, scalers = gates_training_background.setup_boundary_dataloaders(
        parameters, train_inputs, norm_train_bgs, test_inputs, norm_test_bgs,
        norm_train_aux_data, norm_test_aux_data
    )

    save_object(scalers, "scalers", paths_ctx.training_outputs_path, model_name,
                description=f"Input and output scaler objects used in model {model_name}",
                use_wandb=use_wandb)

    image_plots = random.sample(list(range(len(test_inputs))), k=4)
    image_dates = np.datetime_as_string(test_fp_data.time.values[sorted(image_plots)])
    parameters["plotted_dates"] = image_dates.tolist()

    save_object(parameters, "training_settings", paths_ctx.training_outputs_path, model_name,
                file_type="json", description=f"Training settings for model {model_name}",
                use_wandb=use_wandb)

    grid, _ = get_grid(train_fp_data, parameters.get("grid_reference_fp"))
    save_object(grid, "grid", paths_ctx.training_outputs_path, model_name,
                description="Grid object used during training", use_wandb=use_wandb)

    training_ctx = BoundaryTrainingContext(
        parameters, device, use_wandb, image_dates, image_plots, grid,
        None, scalers, feature_dim,
        len(train_fp_data.lat.values), aux_dim
    )

    print("Successfully set up dataloaders!")
    write_to_file("setting up model", paths_ctx.updates_path)

    model, model_ctx = gates_training_background.setup_boundary_model(parameters, training_ctx, paths_ctx)

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    losses = gates_training_background.initialise_boundary_losses()

    if torch.cuda.is_available():
        model.cuda()

    run_full_training(
        model,
        train_loader,
        test_loader,
        model_ctx=model_ctx,
        training_ctx=training_ctx,
        paths_ctx=paths_ctx,
        losses=losses,
        output_norm=norm_vals["outputs"],
        class_names=class_names
    )

    if use_wandb:
        wandb.finish()

# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters. Example usage: python train_GATES_model.py parameters.json")
    parser.add_argument("file_name", help="Parameter file name")
    parser.add_argument("--file_path", help="Parameter file path. By default, the path in config.yml will be used.", default=None)

    args = parser.parse_args()
    file_name = args.file_name
    file_path = args.file_path

    cfg = gates.config.get_config()

    if file_path is None:
        file_path = cfg.parameter_files_dir

    parameter_path = Path(file_path) / file_name

    parameters = load_parameter_file(parameter_path)

    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    if parameters.get("use_wandb", False):
        wandb.login()

    if parameters.get("model_save_dir", None) is None:
        model_saving_dir = cfg.save_models_dir
    else:
        model_saving_dir = Path(parameters["model_save_dir"])

    train_and_save_model(parameters, model_save_dir=model_saving_dir)
