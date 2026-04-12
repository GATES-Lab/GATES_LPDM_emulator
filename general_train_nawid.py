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
    f = open(f"{path}{model_name}/{model_name}_updates.txt", "a")
    f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
    f.close()


def load_file(file_name, file_path):
    # file_path=False if no argument was passed to the parser
    if not file_path:
       file_path ="/user/work/ef17148/GCN/graphnet/graph_weather/train_satellite_files/"
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

def log_object_as_artifact(obj, name, folder, model_name, file_type="pickle", description=""):
    """
    Saves an object locally and logs it to WandB as an artifact.
    """
    ext = "pickle" if file_type == "pickle" else "json"
    filename = f"{name}_{model_name}.{ext}"
    path = os.path.join(folder, filename)

    # 1. Save locally
    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    # 2. Log to WandB
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
    Sets seeds for reproducibility, with safety checks for CUDA availability.
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
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            print((score,self.best_score))
            print('counter increased',self.counter)
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print((score,self.best_score))
            print('counter reset',self.counter)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def save_training_plots(epoch, test_dataset, transformed_preds, image_plots, image_dates, size, path, model_name):
    """Handles the 4x4 grid plotting logic."""
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
    

def export_results_to_netcdf(test_out, transformed_preds, test_dataset, test_data, size, path, model_name):
    """Handles the Xarray dataset creation and NetCDF export."""
    fps = test_dataset.fp.detach().numpy()
    
    # Reshape arrays for NetCDF (Time, Lat, Lon)
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
    
    # Log to W&B as a versioned artifact with explicit name
    preds_artifact = wandb.Artifact(
        name=f"{model_name}-predictions",   # e.g. myModel-predictions:v0
        type="dataset",
        description="Sample predictions saved during training"
        )
    preds_artifact.add_file(netcdf_save_path)
    wandb.log_artifact(preds_artifact)

def run_full_training(model,parameters, train_loader, test_loader, optimizer, criterion, criterion_test, 
                      test_dataset, test_data, device, epoch_so_far, losses, 
                      flux_evaluation, image_plots, image_dates, size, path, model_name, NMAE_function):
    print(parameters['epochs'])
    lr = parameters['learning_rate']
    num_epochs = parameters['epochs']['training']
    visualize_epochs = parameters['epochs']['visualize']
    saving_epochs = parameters['epochs']['model_saving']
    patience_epochs = parameters['epochs']['patience']

    # Initialize Early Stopping
    best_model_path = f"{path}{model_name}/{model_name}_best.pt"
    early_stopping = EarlyStopping(patience=patience_epochs, verbose=True, path=best_model_path)
    

    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} ---")
        
        # 1. Training Pass
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, criterion_test, device, epoch)
        
        # 2. Validation Pass
        avg_test_loss, test_out = validate_and_predict(model, test_loader, criterion_test, device)
        
        # 3. Store Primary Metrics
        losses["train"].append(avg_train_loss)
        losses["test"].append(avg_test_loss)
        
        # 4. Evaluation & Logging
        #truths = torch.squeeze(test_dataset.fp).detach().numpy()
        truths = test_dataset.fp.squeeze(-1).detach().numpy()
        
        print(truths.shape)
        print(test_out.shape)
        
        nmae_val = NMAE_function(test_out, truths)
        losses["NMAE_test"].append(nmae_val)
        
        transformed_preds = test_dataset.inverse_transform(test_out)
        eval_metrics = test_dataset.evaluate()
        
        losses["NMAE_test_transformed"].append(eval_metrics["NMAE"])
        losses["MSE_test_transformed"].append(eval_metrics["MSE"])
        losses["accuracy"].append(eval_metrics["Accuracy"])
        losses["IoU"].append(eval_metrics["IOU"])

        # Flux Evaluation
        for flux_mode in flux_evaluation:
            flux_metrics = test_dataset.evaluate_flux(mode=flux_mode)
            losses[f"flux_{flux_mode}"]["MAE"].append(flux_metrics["MAE"])
            losses[f"flux_{flux_mode}"]["R2"].append(flux_metrics["R2"])
        
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

        # 5. EARLY STOPPING CHECK
        # We pass the validation loss to the class. It handles the counter and saving.
        early_stopping(avg_test_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        # Write to Log File
        log_text = f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, NMAE: {nmae_val:.4f}"
        write_to_file(log_text, path, model_name)

        # 5. Visualizations (Every 5 epochs)
        if epoch % visualize_epochs == 0:
            save_training_plots(epoch, test_dataset, transformed_preds, image_plots, image_dates, size, path, model_name)

        # 6. Checkpoints (Every 50 epochs)
        if epoch % saving_epochs == 0:
            checkpoint_path = f"{path}{model_name}/{model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                'learning_rate': lr,
            }, checkpoint_path)

            # Log to W&B as a versioned artifact
            checkpoint_artifact = wandb.Artifact(
                name=f"{model_name}-checkpoint",   # e.g. myModel-checkpoint:v0
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
        
        
    export_results_to_netcdf(test_out, transformed_preds, test_dataset, test_data, size, path, model_name)

    print("Finished Training.")


def train_and_save_model(parameters, path):
    """
    Train GATES model. Check readme.md for more information on the parameters.
    """
    NMAE_function = NMAE
    #NMAE_function = NMAE_nans
    
    env = parameters['env']
    # 1. Get the current time
    # Format: YYYYMMDD_HHMMSS (e.g., 20260412_114530)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Append it to your model name
    model_name = f"{parameters['model_name']}_{timestamp}"
    print(model_name)

    set_reproducibility(parameters)

    #### wandb
    wandb.init(
        project="BoundaryCondition-Prediction",
        config=parameters,
        tags=[
            #"experiment",
            #"baseline",
            #"reproducibility",
            #"lr_0.01"
            #"levels",
            #"time_deltas",
            #"uncertainty",
            #"grid_size",
            ]

    )
    # make files
    os.makedirs(f"{path}{model_name}", exist_ok=True)
    os.mkdir(f"{path}{model_name}/training_imgs")
    training_outputs_path = f"{path}{model_name}/training_outputs"
    os.mkdir(training_outputs_path)
    f = open(f"{path}{model_name}/{model_name}_updates.txt", "x")
    f.close()


    # Load config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Select the HPC environment you're using
    env_paths = config["data_paths"][env]
    # Join paths
    base_data_path = env_paths["base_data_path"]
    fp_datadir = os.path.join(base_data_path, env_paths["fp_datadir"].lstrip("/"))
    met_datadir = os.path.join(base_data_path, env_paths["met_datadir"].lstrip("/"))
    topog_datadir = os.path.join(base_data_path, env_paths["topog_datadir"].lstrip("/"))
    landcover_datadir = os.path.join(base_data_path, env_paths["landcover_datadir"].lstrip("/"))
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"), path, model_name)
    #write_to_file("loading data")
    write_to_file("loading data", path, model_name)

    #### 2 Load Data
    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
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
    write_to_file("Load training met and fp data",path, model_name)
    data = LoadSquareSatelliteData(**train_load_data, **shared_data_args)
    write_to_file("Successfully load training met and fp data",path, model_name)
    write_to_file("Now load test met and fp data",path, model_name)
    print("Load test met and fp data")
    test_data = LoadSquareSatelliteData(**test_load_data, **shared_data_args)
    write_to_file("Successfully load test met and fp data",path, model_name)
    '''
    data = LoadSquareSatelliteData(**train_load_data, load_everything=True)
    test_data = LoadSquareSatelliteData(**test_load_data,load_everything=True)
    '''
    write_to_file("setting up data", path, model_name)

    # extract inputs

    input_variables = parameters["variables"]

    inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)
    # Checking for NaNs
    nan_indices_train = np.argwhere(np.isnan(inputs))
    if nan_indices_train.size > 0:
        print(f"NaNs found in training set at indices: {nan_indices_train[:10]}")  # show just first 10 for now

    test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
    nan_indices = np.argwhere(np.isnan(test_inputs))
    if nan_indices.size > 0:
        print(f"NaNs found in test set at indices: {nan_indices[:10]}")  # show just first 10 for now
    
    # the model gets built with respect to a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    
    test_batch_size=10

    # transform data - 
    train_dataset = FootprintsDatasetV3(inputs, data.fp_data, input_names=names, **parameters["dataloader_parameters"])

    print(train_dataset.transform_parameters)

    test_dataset = FootprintsDatasetV3(test_inputs, test_data.fp_data, input_names=names, test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10)

    if data.dataset_format == "square":
        size = [data.size, data.size]
    if data.dataset_format == "domain":
        size = data.domain_size
    
    # Save and log the Grid
    log_object_as_artifact(grid, "grid", training_outputs_path, model_name, 
                        description="Grid object used during training")

    # Save and log Transform Parameters
    log_object_as_artifact(train_dataset.transform_parameters, "transform_parameters", 
                        training_outputs_path, model_name, description="Transform parameters used in training")

    # Save and log Settings
    log_object_as_artifact(parameters, "training_settings", training_outputs_path, model_name, 
                        file_type="json", description="Training settings and hyperparameters")
    
    write_to_file("setting up model", path, model_name)
    print("setting up model")

    # all the necessary data is already in the loaders, so we can delete the objects
    image_plots = random.sample(list(range(len(test_inputs))), k=4)
    image_dates = np.datetime_as_string(test_data.fp_data_full.time.values[image_plots])

    #del data, test_data

    lr = parameters["learning_rate"]
    print(lr)
    #### 3 Make model

    # this is leftover from the previous model and actually shouldnt make a difference
    #aux_dim = len(input_variables["others"]) 
    feature_dim=np.shape(inputs)[-1]
    aux_dim = 0

    # Should probably update the name!!
    model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_dim, **parameters["model_parameters"])
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    flux_evaluation=["uniform", "checkerboard_10", "checkerboard_5"]
    losses = {"train":[], "test":[], "NMAE_test":[], "MSE_test_transformed":[], "NMAE_test_transformed":[], "accuracy":[], "IoU":[]}
    losses.update({f"flux_{f}":{"MAE":[], "R2":[]} for f in flux_evaluation})

    

    #### 3 Dump info
    print("saving grids etc")
    '''
    # save transform parameters, grid and training settings
    with open(f"{path}{model_name}/grid_{model_name}.pickle", 'wb') as handle:
        pickle.dump(grid, handle)

    with open(f"{path}{model_name}/transform_parameters_{model_name}.pickle", 'wb') as handle:
        pickle.dump(train_dataset.transform_parameters, handle)

    with open(f"{path}{model_name}/training_settings_{model_name}.json", 'w') as handle:
        json.dump(parameters, handle)
    '''
    wandb.watch(model, log="all", log_freq=100)  # 👈 Track gradients and weights
    epoch_so_far = 0
    if torch.cuda.is_available():
        model.cuda()
    
    #### 4 Train loop
    run_full_training(model,parameters, train_loader, test_loader, optimizer, criterion, criterion_test, 
    test_dataset, test_data, device, epoch_so_far, losses, 
    flux_evaluation, image_plots, image_dates, size, path, model_name, NMAE_function=NMAE_function)
    wandb.finish()

    ## save checkpoint every 50 epochs

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "11d787a211e05ca01c50131c5724e375cd5d3364"  # <<-- REPLACE THIS
    wandb.login()

    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("file_name", help="parameter file name")
    parser.add_argument("--file_path", help="parameter file path")

    args = parser.parse_args()
    file_name = args.file_name
    file_path = args.file_path

    print(file_name, file_path)

    #### 1 Set up

    parameters = load_file(file_name, file_path)
    
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    ## make this importable!
   
    path="/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/"

    # Train the model with the loaded parameters
    train_and_save_model(parameters, path=path)

