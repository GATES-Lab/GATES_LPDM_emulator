import faulthandler
faulthandler.enable()

import sys


import matplotlib.pyplot as plt

import numpy as np
import torch
import xarray as xr

import random
import copy
import wandb



import time
from datetime import datetime

import argparse

import random
from pathlib import Path

import gates.training.training as gates_training

#import gates
from gates.config import get_config
from gates.data.load_data import get_grid
from gates.training.training_dataclasses import PathContext, TrainingContext

from gates.training.training_helperfuns import load_parameter_file, save_object, write_to_file, save_training_plots, export_results_to_netcdf, save_wandb_artifact, set_reproducibility



def train_one_epoch(model, loader, model_ctx, epoch, paths_ctx=None):
    """
    Runs a single training epoch, iterating over all batches, computing the loss, and updating model weights.
    A separate display loss (criterion_test) is tracked for monitoring without affecting gradients.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader providing batches of (inputs, fp).
        model_ctx (ModelContext): Context object containing optimizer, loss functions, device, and epoch settings.
        epoch (int): The current epoch number, used for progress logging.
        paths_ctx (PathContext, optional): Context object containing file paths for logging.

    Returns:
        tuple:
            - float: The mean display loss (criterion_test) across all batches.
            - float: The mean training loss (criterion) across all batches.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()
    mean_loss = 0.0
    #write_to_file(f"Starting epoch {epoch}", paths_ctx.updates_path)
    for i, batch in enumerate(loader):
        #write_to_file(f"Loaded batch {i} in epoch {epoch}", paths_ctx.updates_path)
        
        features_batch, fp_batch = batch[0].to(model_ctx.device), batch[1].to(model_ctx.device)
        #write_to_file(f"Sent to device", paths_ctx.updates_path)

        if len(fp_batch.shape) == 3:
            true_values = fp_batch[:,:,0].unsqueeze(-1)
        else:
            true_values = fp_batch.unsqueeze(-1)
        
        model_ctx.optimizer.zero_grad()

        #write_to_file(f"Inferring", paths_ctx.updates_path)
        outputs = model(features_batch)
        #write_to_file(f"Calculating loss", paths_ctx.updates_path)
        loss = model_ctx.criterion(outputs, true_values, fp_batch)
        mean_loss += loss.item()
        #write_to_file(f"Backward pass", paths_ctx.updates_path)
        loss.backward()
        #write_to_file(f"Optimizer step", paths_ctx.updates_path)
        model_ctx.optimizer.step()

        # Metrics tracking
        with torch.no_grad():
            display_loss = model_ctx.criterion_test(outputs, true_values, fp_batch)
            running_loss += display_loss.item()
        
        if i % 50 == 0:
            print(f"[{epoch}, {i:5d}] Training Loss: {mean_loss/(i+1):.3f} Loss: {running_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s")
            write_to_file(f"[{epoch}, {i:5d}] Training Loss: {mean_loss/(i+1):.3f} Time: {time.time()-start_time:.1f}s", paths_ctx.updates_path)
            
    return running_loss / len(loader), mean_loss / len(loader)  # return both the display loss and the actual training loss for logging


@torch.no_grad()
def validate_and_predict(model, model_ctx, loader):
    """
    Evaluates the model on a validation or test set, collecting predictions and computing the mean loss.
    Runs under torch.no_grad() to disable gradient computation for efficiency.

    Args:
        model (torch.nn.Module): The model to evaluate.
        model_ctx (ModelContext): Context object providing criterion_test and device.
        loader (torch.utils.data.DataLoader): DataLoader providing batches of (inputs, fp).

    Returns:
        tuple:
            - float: The mean loss (criterion_test) across all batches.
            - np.ndarray: Array of shape (total_samples, flat_lat_lon) containing all model predictions.
    """
    model.eval()
    test_error = 0.0
    preds_list = []

    print("validating and predicting")
    
    for i, batch in enumerate(loader):
        features_batch, fp_batch = batch[0].to(model_ctx.device), batch[1].to(model_ctx.device)

        if len(fp_batch.shape) == 3:
            true_values = fp_batch[:,:,0].unsqueeze(-1)
        else:
            true_values = fp_batch.unsqueeze(-1)

        #ins, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(features_batch)
        
        test_error += model_ctx.criterion_test(outputs, true_values, fp_batch).item()
        

        preds_list.append(outputs)
    
    all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
        
    # vstack will now reliably return (Total_Samples, Features)
    return test_error / len(loader), all_preds #np.vstack(preds_list)


 
def run_full_training(model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader, test_fp_dataset, losses, epoch_so_far=0):
    #optimizer, criterion, criterion_test, 
                    #   test_dataset, test_data, epoch_so_far, 
                    #   flux_evaluation, image_plots, image_dates, size, path, model_name, NMAE_function):
    """
    Executes the full training loop for a given number of epochs, including training and validation passes,
    metric logging, early stopping, periodic visualisation, and model checkpointing. At the end of training,
    predictions are exported to a NetCDF file and logged to Weights & Biases.

    Args:
        model (torch.nn.Module): The model to train.
        model_ctx (ModelContext): Context object containing optimizer, loss functions, device, epoch
            counts, early stopping, and W&B flag.
        training_ctx (TrainingContext): Context object containing scalers, grid, image plot indices,
            dates, and other training-time metadata.
        paths_ctx (PathContext): Context object containing all file paths for logs, plots, and checkpoints.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation/test set.
        test_fp_dataset (xr.Dataset): Dataset holding the test footprints; predictions are written
            into this object each epoch for evaluation and plotting.
        losses (dict): A dictionary of lists used to accumulate per-epoch metrics across the run.
        epoch_so_far (int): The epoch count to start from, allowing training to resume from a checkpoint.

    Returns:
        None
    """

    #best_model_path =  paths_ctx.model_save_dir / f"{model_ctx.model_name}_best.pt"

    #for epoch_idx in range(model_ctx.epochs_num):
    write_to_file("starting training loop", paths_ctx.updates_path)

    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} ---")

        avg_train_loss, avg_train_transformed_loss = train_one_epoch(model, train_loader, model_ctx, epoch, paths_ctx=paths_ctx)
        print(f"Finished training epoch {epoch} with average training loss: {avg_train_loss:.4f}")
        
        #with dask.config.set(scheduler='synchronous'):
        print("NOT synchronous dask config set for validation and prediction")
        avg_test_loss, test_out = validate_and_predict(model, model_ctx, test_loader)

        #print(avg_train_loss, avg_test_loss, test_out.shape)

    
        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)

        #test_dataset["predicted_fp"] = test_out.reshape(-1, *training_ctx.size, 1)  # reshape to (samples, H, W, 1) for evaluation and plotting

        print("inversing outputs")
        write_to_file("inversing outputs", paths_ctx.updates_path)

        outputs_original_space = training_ctx.scalers["fp_scaler"].inverse_transform(test_out) 

        print("adding outputs to dataset for evaluation")
        write_to_file("adding outputs to dataset for evaluation", paths_ctx.updates_path)

        test_fp_dataset["fp_transformed_pred"] = (("time", "lat", "lon"), test_out.reshape(*test_fp_dataset.fp_original.shape))  # reshape to (samples, H, W, 1) for evaluation and plotting
        test_fp_dataset["fp_pred"] = (("time", "lat", "lon"), outputs_original_space.reshape(*test_fp_dataset.fp_original.shape)) 



        print("calculating losses and metrics") 
        write_to_file("calculating losses and metrics", paths_ctx.updates_path)           
        
        losses, computed_metrics = gates_training.calculate_losses(losses, test_fp_dataset)  

        
        


        # for flux_mode in flux_evaluation:
        #     flux_metrics = test_dataset.evaluate_flux(mode=flux_mode)
        #     losses[f"flux_{flux_mode}"]["MAE"].append(flux_metrics["MAE"])
        #     losses[f"flux_{flux_mode}"]["R2"].append(flux_metrics["R2"])

        list_of_metrics = {f"metrics_transformed-{k}": v for k, v in computed_metrics["transformed_eval_metrics"].items()}
        list_of_metrics.update({f"metrics_original-{k}": v for k, v in computed_metrics["eval_metrics"].items()})
        for flux_mode, metrics in computed_metrics["static_mf_eval_metrics"].items():
            list_of_metrics.update({f"metrics_fluxes_static/{flux_mode}/{k}": v for k, v in metrics.items()})
        if "flux_eval_metrics" in computed_metrics:
            list_of_metrics.update({f"metrics_fluxes/{k}": v for k, v in computed_metrics["flux_eval_metrics"].items()})


        if model_ctx.use_wandb:
            logging_dict = {
                "epoch": epoch + 1,
                "MSE/train":avg_train_loss,
                "MSE/test": avg_test_loss,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss,
                "LossFn/train": avg_train_transformed_loss,
                **list_of_metrics,}
            
            wandb.log(logging_dict)

            # wandb.log({
            #     "epoch": epoch + 1,
            #     "train/loss": avg_train_loss,
            #     "test/loss": avg_test_loss,
            #     "test/NMAE": nmae_val,
            #     "test/NMAE_transformed": eval_metrics['NMAE'],
            #     "test/MSE": eval_metrics['MSE'],
            #     "test/IoU": eval_metrics['IOU'],
            #     **{f"flux/{k}": v for k, v in flux_metrics.items()}
            # }, step=epoch)

        model_ctx.early_stopping(avg_test_loss, model)

        if model_ctx.early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        log_text = f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        write_to_file(log_text, paths_ctx.updates_path)
        log_text = f"    metrics: {str(losses['metrics_transformed'])}, {str(losses['metrics_original'])}, {str(losses['metrics_fluxes_static'])}"
        write_to_file(log_text, paths_ctx.updates_path)

        if epoch % model_ctx.epochs_visualise == 0:
            img_save_path = save_training_plots(epoch, test_fp_dataset, training_ctx, paths_ctx.model_path, paths_ctx.model_name)
        if epoch % model_ctx.epochs_visualise == 0:
            if model_ctx.use_wandb:
                wandb.log({"epoch": epoch, "training_plots": wandb.Image(img_save_path)})



        if epoch % model_ctx.epochs_save == 0:
            checkpoint_path = paths_ctx.model_path / f"{model_ctx.model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_ctx.optimizer.state_dict(),
                'loss': losses,
                'learning_rate': model_ctx.lr,
            }, checkpoint_path)

            if model_ctx.use_wandb:
                save_wandb_artifact(model_ctx.model_name, f"checkpoint_epoch_{epoch}", "model", f"Model checkpoint at epoch {epoch} with loss {avg_test_loss:.4f}, saved during training", checkpoint_path)
        
        write_to_file(f"Finished one loop!", paths_ctx.updates_path)
        #exit()  # <<-- TEMPORARY!

    export_results_to_netcdf(test_fp_dataset, paths_ctx.model_path, model_ctx.model_name, use_wandb=model_ctx.use_wandb)
    print("Finished Training.") 

def train_and_save_model(parameters, model_save_dir):
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
            - 'variables' (dict): Keyword arguments forwarded to get_square_satellite_inputs_v2().
            - 'dataloader_parameters' (dict): Keyword arguments forwarded to FootprintDataset().
            - 'model_parameters' (dict): Keyword arguments forwarded to GraphSatelliteForecaster().
            - 'loss_functions' (dict): Contains 'criterion' and 'criterion_test' as eval-able strings.
            - 'train_load_data' (dict): Keyword arguments for loading the training dataset.
            - 'test_load_data' (dict): Overrides applied on top of train_load_data for the test dataset.
        model_save_dir (str): Base directory path under which all model output folders and files will be created.

    Returns:
        None
    """
    #NMAE_function = NMAE
    #NMAE_function = NMAE

    ### setting up
    use_wandb = parameters.get('use_wandb', False)  
    
    #cfg = get_config()

    verbose = parameters.get("verbose", True)

    # `load_before_training` (formerly the top-level `load_into_memory`) controls whether
    # the inputs/footprints are materialised into RAM before the dataloader is built.
    # Defaults to True. The old `load_into_memory` key is still honoured for back-compat.
    load_before_training = parameters.get("load_before_training", parameters.get("load_into_memory", True))
    parameters["load_before_training"] = load_before_training


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    parameters["start_time"] = timestamp
    print(f"Initialising model run for model_name: {model_name}")

    # the paths are all stored in this dataclass (e.g. paths_ctx.model_path)
    paths_ctx = PathContext(
        model_save_dir=model_save_dir,
        model_name=model_name,
        model_path=model_path # model_save_dir / model_name
    )

    paths_ctx.make_dirs()

    seed = parameters.get("seed", 34)
    set_reproducibility(seed)
    if verbose: print(f"Set random seed to {seed} for reproducibility.")

    if use_wandb:
        wandb_project = parameters.get("wandb", {}).get("project", None)
        wandb_entity = parameters.get("wandb", {}).get("entity", None)
        wandb_tags = parameters.get("wandb", {}).get("tags", [])
        if use_wandb and wandb_project is None or wandb_entity is None:
            print("Warning: 'use_wandb' is True but no 'wandb.project' or 'wandb.entity' specified in parameters. W&B will not be initialised.")
            use_wandb = False
            parameters["use_wandb"] = False

        if use_wandb:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project, 
                config=parameters,
                tags=wandb_tags
            )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"), paths_ctx.updates_path)
    write_to_file("loading data", paths_ctx.updates_path)

    train_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params.update(parameters["test_load_data"])

    input_variables = parameters["variables"]

    # the data is extracted from the config file, unless it is superced from parameters. resolve_datapath_args returns the correct path in a dictionary passed to the data objects 
    datapath_args = paths_ctx.resolve_datapath_args(parameters)
    flux_args = parameters.get("flux", None)

    print("Loading met and fp data for model", model_name)
    write_to_file("Load training and testing met and fp data", paths_ctx.updates_path)

    client, cluster = gates_training.make_cluster()

    # Shared loading state so W&B loading metrics form one continuous series across
    # the train and test loads instead of the counter restarting at 1 for each.
    wandb_loading_state = gates_training.initialise_wandb_loading() if use_wandb else None

    data, train_inputs = gates_training.load_GATES_data_v2(train_load_data_params, input_variables=input_variables, datapath_args=datapath_args, flux_args=flux_args, verbose=verbose, load_into_memory=load_before_training, use_wandb=use_wandb, wandb_state=wandb_loading_state)
    train_fp_data = data

    write_to_file(f"Successfully loaded training met and fp data with {len(train_fp_data.time)} time samples. Loading test data", paths_ctx.updates_path)
    print("Successfully loaded training met and fp data with", len(train_fp_data.time), "time samples")
    print("Loading test data")

    test_data, test_inputs = gates_training.load_GATES_data_v2(test_load_data_params, input_variables=input_variables, datapath_args=datapath_args, flux_args=flux_args, verbose=verbose, load_into_memory=load_before_training, use_wandb=use_wandb, wandb_state=wandb_loading_state)  # if load_before_training is True, this will load the test data into memory immediately; if False, it will remain as dask arrays until needed
    test_fp_data = test_data


    write_to_file(f"Successfully load test met and fp data with {len(test_fp_data.time)} time samples", paths_ctx.updates_path)
    print("Successfully load test met and fp data with", len(test_fp_data.time), "time samples")

    if use_wandb:
        # save the number of testing and training samples to wandb config for reference
        wandb.summary.update({
            "num_training_samples": len(train_fp_data.time),
            "num_testing_samples": len(test_fp_data.time),
        })
    
    ## add the number of features to the parameter file, and to wandb
    num_features = train_inputs.shape[-1]
    parameters["num_features"] = num_features
    if use_wandb:
        wandb.summary.update({"num_features": num_features})



    if load_before_training:
        write_to_file("Loading data into memory as specified in parameters - FIXED", paths_ctx.updates_path)
        print("Loading data into memory as specified in parameters. - FIXED")
        print(f"    computing train inputs, with size {train_inputs.nbytes / 1e9:.2f} GB")
        train_inputs = train_inputs.compute()  # if using dask arrays, this will load into memory; if already numpy arrays, this does nothing
        print(f"    computing test inputs, with size {test_inputs.nbytes / 1e9:.2f} GB")
        test_inputs = test_inputs.compute()
        print("computing training fp data (although it should already be in mem)")
        #data.fp_xr = data.fp_xr.compute()
        train_fp_data = train_fp_data.compute()
        print("computing test fp data (although it should already be in mem)")
        #test_data.fp_xr = test_data.fp_xr.compute()
        test_fp_data = test_fp_data.compute()

    write_to_file("scaling data and setting up dataloaders", paths_ctx.updates_path)

    if cluster is not None:
        print("closing client before cluster!")
        client.close()    # drain and disconnect first
        cluster.close()   # then shut down the workers
        # closes the dask clients and frees up the memory for other workers

    #inputs_dataset = gates_training.setup_input_dataset(parameters, train_inputs)
    #fp_dataset = gates_training.setup_fp_dataset(parameters, data.fp_xr)

    train_loader, test_loader, fp_labels, test_scaled_fp, scalers = gates_training.setup_GATES_dataloaders(parameters, train_inputs, train_fp_data, test_inputs, test_fp_data)

    # save the scalers
    save_object(scalers, "scalers",
    paths_ctx.training_outputs_path, model_name, description=f"Input and output scaler objects used in model {model_name}", use_wandb=use_wandb)
    
    # images will get plotted and saved for a random selection of 4 dates from the test set - these indeces are saved to parameters for reference 
    image_plots = random.sample(list(range(len(test_inputs))), k=4)
    image_dates = np.datetime_as_string(test_fp_data.time.values[sorted(image_plots)])

    parameters["plotted_dates"] = image_dates.tolist()

    save_object(parameters, "training_settings", paths_ctx.training_outputs_path, model_name,  file_type="json", description=f"Training settings and hyperparameters for model {model_name}", use_wandb=use_wandb)

    # create the grid object to make the mesh with and save
    grid, _ = get_grid(train_fp_data, parameters.get("grid_reference_fp"))
    save_object(grid, "grid", paths_ctx.training_outputs_path, model_name,
                           description="Grid object used during training", use_wandb=use_wandb)
    print("dyanmic edges parameters:", parameters.get("dynamic_edges", None))

    
    # parameters["dynamic_edges"] can be None (default), a dict (with specific dynamic edge settings), or True (which defaults to dynamic wind edges)

    if parameters.get("dynamic_edges", None) is not None:
        # if its a dict
        if isinstance(parameters["dynamic_edges"], dict):
            dynamic_edges_params = gates_training.setup_dynamic_edges(input_names=scalers["input_names"], **parameters["dynamic_edges"])

        elif parameters.get("dynamic_edges") is True:
            dynamic_edges_params = gates_training.setup_dynamic_edges(input_names=scalers["input_names"])
        
        else:
            dynamic_edges_params = {}
    else:
        dynamic_edges_params = {}
            

    training_ctx = TrainingContext(parameters, device, use_wandb, image_dates, image_plots, grid, fp_labels, scalers, train_inputs.variable_name.size, len(train_fp_data.lat.values), dynamic_edges_params) # get size from train params

 
    ########################


    print("Successfully set up dataloaders!!! using the contexts!!!")

    write_to_file("setting up model", paths_ctx.updates_path)


    model, model_ctx = gates_training.setup_GATES_model(parameters, training_ctx, paths_ctx)

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    losses = gates_training.initialise_losses()

    if use_wandb:
        wandb.define_metric("epoch")
        wandb.define_metric("MSE/*", step_metric="epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("LossFn/*", step_metric="epoch")
        wandb.define_metric("metrics_*", step_metric="epoch")
        wandb.define_metric("training_plots", step_metric="epoch")

        if "metrics_fluxes_static" in losses:
            for flux_mode, flux_metrics in losses["metrics_fluxes_static"].items():
                for metric_name in flux_metrics:
                    wandb.define_metric(f"metrics_fluxes_static/{flux_mode}/{metric_name}", step_metric="epoch")

    print("successfully set up the model!!! starting training loop")
    
    run_full_training(model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader, test_scaled_fp, losses, epoch_so_far=0)


if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Load parameters. Example usage: python train_GATES_model.py parameters.json")
    parser.add_argument("file_name", help="Parameter file name")
    parser.add_argument("--file_path", help="Parameter file path. By default, the path in config.yml will be used.", default=None)

    args = parser.parse_args()
    file_name = args.file_name
    file_path = args.file_path

    cfg = get_config()

    if file_path is None:
        file_path = cfg.parameter_files_dir
    
    parameter_path = Path(file_path) / file_name
    
    #### 1 Set up
    parameters = load_parameter_file(parameter_path)
    
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    if parameters.get("use_wandb", False):
        wandb.login()
    
    if parameters.get("model_save_dir", None) is None: 
        model_saving_dir=cfg.save_models_dir
    else:
        model_saving_dir = Path(parameters["model_save_dir"])

    #TODO write a function that checks minimum parameters exist
    

    # Train the model with the loaded parameters
    train_and_save_model(parameters, model_save_dir=model_saving_dir)

