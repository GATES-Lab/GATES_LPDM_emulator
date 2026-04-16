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

def write_to_file(message, path, model_name):
    """
    Appends a timestamped message to a model-specific log file.

    Args:
        message (str): The message to log.
        path (str): The base directory path where the model folder is located.
        model_name (str): The name of the model, used to locate the subfolder and name the log file.

    Returns:
        None
    """
    f = open(f"{path}{model_name}/{model_name}_updates.txt", "a")
    f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
    f.close()


def load_file(file_path):
    """
    Loads and parses a JSON file from a given path, falling back to a default directory if no path is provided.

    Args:
        file_name (str): The name of the file to load (including extension).
        file_path (str or bool): The directory path containing the file. If False, a hardcoded default path is used.

    Returns:
        dict or list: The parsed JSON contents of the file, or None if the file was not found or an error occurred.
    """
    # file_path=False if no argument was passed to the parser
    if not file_path:
       file_path ="/user/work/ef17148/GCN/graphnet/graph_weather/train_satellite_files/"
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

def log_object_as_artifact(obj, name, folder, model_name, file_type="pickle", description="", use_wandb=True):
    """
    Serialises an object to disk as either a pickle or JSON file, then logs it to Weights & Biases as a versioned artifact.

    Args:
        obj (any): The Python object to save and log.
        name (str): A descriptive name for the artifact, used in both the filename and the W&B artifact name.
        folder (str): The local directory in which to save the file.
        model_name (str): The name of the model, included in the filename and W&B artifact name.
        file_type (str): Serialisation format — either "pickle" (default) or "json".
        description (str): An optional description string attached to the W&B artifact.
        use_wandb (bool): If True, logs the saved file to W&B as a versioned artifact. Defaults to True.

    Returns:
        str: The local file path where the object was saved.
    """
    ext = "pickle" if file_type == "pickle" else "json"
    filename = f"{name}_{model_name}.{ext}"
    path = os.path.join(folder, filename)

    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    if use_wandb:
        artifact = wandb.Artifact(
            name=f"{model_name}-{name.replace('_', '-')}",
            type=file_type,
            description=description
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    return path

def set_reproducibility(parameters, default_seed=34):
    """
    Sets random seeds across Python, NumPy, and PyTorch to ensure reproducible training runs.
    If a GPU is available, CUDA seeds are also set and cuDNN is switched to deterministic mode.

    Args:
        parameters (dict): A dictionary of training parameters; expected to optionally contain a "seed" key.
        default_seed (int): The seed value to use if "seed" is not present in parameters. Defaults to 34.

    Returns:
        int: The seed value that was applied.
    """
    seed = parameters.get("seed", default_seed)
    print(f"Using seed: {seed}")

    # Standard CPU-based seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU-specific seeds and settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Forces cuDNN to use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        # Disables the auto-tuner that selects the fastest (but non-deterministic) algorithms
        torch.backends.cudnn.benchmark = False
        print("CUDA seeds set and cuDNN configured to deterministic mode.")
    else:
        print("CUDA not available; skipping GPU seed initialization.")

    return seed


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


class EarlyStopping:
    """
    Monitors validation loss during training and halts training when no improvement is seen
    for a given number of consecutive epochs. Also saves the best model checkpoint to disk,
    and optionally logs it to Weights & Biases as a versioned artifact.
    """
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', use_wandb=True, model_name="model"):
        """
        Args:
            patience (int): Number of epochs with no improvement to wait before stopping. Defaults to 20.
            verbose (bool): If True, prints a message each time the counter increments or the model is saved. Defaults to False.
            delta (float): Minimum change in validation loss to qualify as an improvement. Defaults to 0.
            path (str): File path at which to save the best model checkpoint. Defaults to 'checkpoint.pt'.
            use_wandb (bool): If True, logs the best model checkpoint to W&B as a versioned artifact each time it improves. Defaults to True.
            model_name (str): The model name used for the W&B artifact name. Defaults to "model".
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.use_wandb = use_wandb
        self.model_name = model_name

    def __call__(self, val_loss, model):
        """
        Evaluates the current validation loss and updates the early-stopping state.

        Args:
            val_loss (float): The validation loss for the current epoch.
            model (torch.nn.Module): The model to checkpoint if validation loss has improved.

        Returns:
            None
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model's state dict to disk when validation loss reaches a new minimum,
        and optionally logs it to W&B as a versioned artifact.

        Args:
            val_loss (float): The new best validation loss.
            model (torch.nn.Module): The model whose weights should be saved.

        Returns:
            None
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        if self.use_wandb:
            best_artifact = wandb.Artifact(
                name=f"{self.model_name}-best",
                type="model",
                description=f"Best model checkpoint (val_loss: {val_loss:.6f})"
            )
            best_artifact.add_file(self.path)
            wandb.log_artifact(best_artifact)

def save_training_plots(epoch, test_dataset, transformed_preds, image_plots, image_dates, size, path, model_name):
    """
    Generates and saves a 4×4 grid of images comparing model predictions against ground truth
    at a selection of test samples, for visual inspection during training.

    Args:
        epoch (int): The current epoch number, used in the output filename.
        test_dataset: A dataset object exposing fp, fp_untransformed, and predictions attributes.
        transformed_preds (np.ndarray): Model predictions mapped back into the original (untransformed) space.
        image_plots (list of int): Indices of the test samples to visualise, should contain exactly 4 values.
        image_dates (list of str): Date strings corresponding to each sample index in image_plots.
        size (tuple of int): The spatial dimensions (height, width) used to reshape flat arrays into images.
        path (str): The base directory path for saving output images.
        model_name (str): The model name used to locate the output subfolder and name the saved file.

    Returns:
        None
    """
    og_fps = test_dataset.fp_untransformed
    fps = test_dataset.fp.detach().numpy()
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    
    for axis, fn in enumerate(image_plots):
        # Reshape for visualization
        pred_img = np.reshape(test_dataset.predictions[fn, :], (size[0], size[1]))
        truth_img = np.reshape(fps[fn, :], (size[0], size[1]))
        trans_pred_img = np.reshape(transformed_preds[fn, :], (size[0], size[1]))
        og_truth_img = np.reshape(og_fps[fn, :], (size[0], size[1]))

        ax[0, axis].imshow(pred_img, origin="lower")
        ax[1, axis].imshow(truth_img, origin="lower")
        ax[2, axis].imshow(trans_pred_img, origin="lower")
        ax[3, axis].imshow(og_truth_img, origin="lower")

        ax[0, axis].set_title(f"Pred\nSample {fn}")
        ax[1, axis].set_title(f"Truth\n({str(image_dates[axis][:10])})")
        ax[2, axis].set_title(f"Trans Pred")
        ax[3, axis].set_title(f"Orig Truth")

        for a in range(4):
            ax[a, axis].set_xticks([])
            ax[a, axis].set_yticks([])

    plt.tight_layout()
    save_path = f"{path}{model_name}/training_imgs/{model_name}_{epoch}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

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
    visualize_epochs = parameters['epochs']['visualize']
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

        truths = test_dataset.fp.squeeze(-1).detach().numpy()
        nmae_val = NMAE_function(test_out, truths)
        losses["NMAE_test"].append(nmae_val)

        transformed_preds = test_dataset.inverse_transform(test_out)
        eval_metrics = test_dataset.evaluate()

        losses["NMAE_test_transformed"].append(eval_metrics["NMAE"])
        losses["MSE_test_transformed"].append(eval_metrics["MSE"])
        losses["accuracy"].append(eval_metrics["Accuracy"])
        losses["IoU"].append(eval_metrics["IOU"])

        for flux_mode in flux_evaluation:
            flux_metrics = test_dataset.evaluate_flux(mode=flux_mode)
            losses[f"flux_{flux_mode}"]["MAE"].append(flux_metrics["MAE"])
            losses[f"flux_{flux_mode}"]["R2"].append(flux_metrics["R2"])

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss,
                "test/NMAE": nmae_val,
                "test/NMAE_transformed": eval_metrics['NMAE'],
                "test/MSE": eval_metrics['MSE'],
                "test/IoU": eval_metrics['IOU'],
                **{f"flux/{k}": v for k, v in flux_metrics.items()}
            }, step=epoch)

        early_stopping(avg_test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        log_text = f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, NMAE: {nmae_val:.4f}"
        write_to_file(log_text, path, model_name)

        if epoch % visualize_epochs == 0:
            save_training_plots(epoch, test_dataset, transformed_preds, image_plots, image_dates, size, path, model_name)

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
                '''
                # 7. Special Epoch Conditions
                if epoch == 102 and NMAE_nans(test_out, truths) == 1:
                print("No learning detected. Early stopping.")
                break
                '''

    export_results_to_netcdf(test_out, transformed_preds, test_dataset, test_data, size, path, model_name, use_wandb=use_wandb)
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
    #NMAE_function = NMAE
    NMAE_function = NMAE_nans
    use_wandb = parameters.get('use_wandb', True)  # defaults to True if not set

    env = parameters['env']
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

