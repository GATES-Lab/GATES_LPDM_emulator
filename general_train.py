import sys

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

#### 1 Set up

parser = argparse.ArgumentParser(description="Load parameters")
parser.add_argument("file_name", help="parameter file name")
parser.add_argument("--file_path", help="parameter file path")
parser.add_argument("--output_dir", type=str, default="training_output")
parser.add_argument("--seed", type=int, default=34)

args = parser.parse_args()
file_name = args.file_name
file_path = args.file_path

print(file_name, file_path)

def write_to_file(message):
    with open(log_file_path, "a") as f:
        f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
        f.flush()
        os.fsync(f.fileno())


parameters = load_file(file_name, file_path) 

print("PARAMETERS:")
print(parameters)

model_name = parameters["model_name"]
print(model_name)

# Ensure output dir and subfolders exist
os.makedirs(args.output_dir, exist_ok=True)
training_img_dir = os.path.join(args.output_dir, "training_imgs")
os.makedirs(training_img_dir, exist_ok=True)

log_file_path = os.path.join(args.output_dir, f"{model_name}_updates.txt")
with open(log_file_path, "a") as f:
    pass

if "seed" in (parameters.keys()):
    seed = parameters["seed"]

else:
    seed = args.seed

#### wandb
wandb.init(
        project="Oracle",
        name=os.environ.get("WANDB_NAME", "local-run"),
        config=parameters,
        notes=os.environ.get("WANDB_NOTES", ""),
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
    


np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

print(seed)
print("Random check:", random.randint(0, 10000), np.random.randint(0, 10000), torch.randint(0, 10000, (1,)))

# Load config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Select the HPC environment you're using
env = "oracle"
env_paths = config["data_paths"][env]

# Join paths
base_data_path = env_paths["base_data_path"]
fp_datadir = os.path.join(base_data_path, env_paths["fp_datadir"].lstrip("/"))
met_datadir = os.path.join(base_data_path, env_paths["met_datadir"].lstrip("/"))
topog_datadir = os.path.join(base_data_path, env_paths["topog_datadir"].lstrip("/"))
landcover_datadir = os.path.join(base_data_path, env_paths["landcover_datadir"].lstrip("/"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
write_to_file(f"using device {device}, starting at " + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
write_to_file("loading data")

#### 2 Load Data
train_load_data = copy.deepcopy(parameters["train_load_data"])
# load train parameters and upload with any changes to test data
test_load_data = copy.deepcopy(parameters["train_load_data"])
test_load_data.update(parameters["test_load_data"])

write_to_file("loading parameters")
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
write_to_file("Load training met and fp data")
data = LoadSquareSatelliteData(**train_load_data, **shared_data_args)
write_to_file("Successfully load training met and fp data")
write_to_file("Now load test met and fp data")
print("Load test met and fp data")
test_data = LoadSquareSatelliteData(**test_load_data, **shared_data_args)
write_to_file("Successfully load test met and fp data")

write_to_file("setting up data")

# extract inputs

input_variables = parameters["variables"]

print("Load training satellite data")

inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)
# Checking for NaNs
nan_indices_train = np.argwhere(np.isnan(inputs))
if nan_indices_train.size > 0:
    print(f"NaNs found in training set at indices: {nan_indices_train[:10]}")  # show just first 10 for now

print("Load test satellite data")
test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
# Checking for NaNs
nan_indices = np.argwhere(np.isnan(test_inputs))
if nan_indices.size > 0:
    print(f"NaNs found in test set at indices: {nan_indices[:10]}")  # show just first 10 for now

# the model gets built with respect to a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

write_to_file("setting up model")
print("setting up model")

test_batch_size=10

# transform data - 
train_dataset = FootprintsDatasetV3(inputs, data.fp_data, input_names=names, **parameters["dataloader_parameters"])

print(train_dataset.transform_parameters)

test_dataset = FootprintsDatasetV3(test_inputs, test_data.fp_data, input_names=names, test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])




g = torch.Generator()
g.manual_seed(seed)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, generator=g, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=10, generator=g, num_workers=0)

if data.dataset_format == "square":
    size = [data.size, data.size]
if data.dataset_format == "domain":
    size = data.domain_size

image_plots = random.sample(list(range(len(test_inputs))), k=4)
image_dates = np.datetime_as_string(test_data.fp_data_full.time.values[image_plots])

time_values = test_data.met.time.values
# all the necessary data is already in the loaders, so we can delete the objects
del data, test_data

lr = parameters["learning_rate"]
print(lr)
#### 3 Make model

# this is leftover from the previous model and actually shouldnt make a difference
#aux_dim = len(input_variables["others"]) 
feature_dim=np.shape(inputs)[-1]
aux_dim = 0

image_plots = random.sample(list(range(len(test_inputs))), k=4)

# Should probably update the name!!
model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_dim, **parameters["model_parameters"])

criterion = eval(parameters["loss_functions"]["criterion"])

criterion_test = eval(parameters["loss_functions"]["criterion_test"])

optimizer = optim.AdamW(model.parameters(), lr=lr)

flux_evaluation=["uniform", "checkerboard_10", "checkerboard_5"]

losses = {"train":[], "test":[], "NMAE_test":[], "MSE_test_transformed":[], "NMAE_test_transformed":[], "accuracy":[], "IoU":[]}

losses.update({f"flux_{f}":{"MAE":[], "R2":[]} for f in flux_evaluation})


NMAE_function = NMAE
NMAE_function = NMAE_nans


#### 3 Dump info
print("saving grids etc")

# save transform parameters, grid and training settings
grid_path = os.path.join(args.output_dir, f"grid_{model_name}.pickle")
with open(grid_path, 'wb') as handle:
    pickle.dump(grid, handle)

grid_artifact = wandb.Artifact(
    name=f"{model_name}-grid",           # e.g. myModel-grid:v0
    type="pickle",
    description="Grid object used during training"
)
grid_artifact.add_file(grid_path)
wandb.log_artifact(grid_artifact)

# -------------------------------------------------------------

transform_path = os.path.join(args.output_dir, f"transform_parameters_{model_name}.pickle")
with open(transform_path, 'wb') as handle:
    pickle.dump(train_dataset.transform_parameters, handle)

transform_artifact = wandb.Artifact(
    name=f"{model_name}-transform-parameters",   # e.g. myModel-transform-parameters:v0
    type="pickle",
    description="Transform parameters used in training"
)
transform_artifact.add_file(transform_path)
wandb.log_artifact(transform_artifact)

# -------------------------------------------------------------

settings_path = os.path.join(args.output_dir, f"training_settings_{model_name}.json")
with open(settings_path, 'w') as handle:
    json.dump(parameters, handle, indent=2)

settings_artifact = wandb.Artifact(
    name=f"{model_name}-training-settings",   # e.g. myModel-training-settings:v0
    type="json",
    description="Training settings and hyperparameters"
)
settings_artifact.add_file(settings_path)
wandb.log_artifact(settings_artifact)

# ------------

epoch_so_far = 0
if torch.cuda.is_available():
    model.cuda()

wandb.watch(model, log="all", log_freq=100)  # 👈 Track gradients and weights

num_epochs = parameters.get("epochs", 350)  # fallback to 350 if not set
write_to_file(f"Training for {num_epochs} epochs")
#### 4 Train loop
for epoch in range(num_epochs):
    epoch=epoch+epoch_so_far
    running_loss = 0.0
    print(f"Start Epoch: {epoch}")
    start = time.time()
    for i, batch in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels, true_fp = batch[0].to(device), batch[1].to(device), batch[2].to(device)        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(ins)
        loss = criterion(outputs, labels, true_fp)
        loss.backward()
        optimizer.step()
        loss = criterion_test(outputs, labels)
        running_loss += loss.item()
        end = time.time()
        del outputs 
        if i % 10==0:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec")
    
    test_error = 0.0
    test_out = np.zeros((test_dataset.inputs.size()[0], test_dataset.inputs.size()[1]))   
    for i_test, batch in enumerate(test_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels = batch[0].to(device), batch[1].to(device)

        # check if NaN
        if torch.isnan(ins).any():
            print(f"NaN detected in inputs at batch {i_test}")
        if torch.isnan(labels).any():
            print(f"NaN detected in labels at batch {i_test}")


        test_error += criterion_test(model(ins), labels).item()
        test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = np.squeeze(model(ins).detach().cpu().numpy())
    
    
    losses["train"].append(running_loss/(i+1))
    losses["test"].append(test_error/(i_test+1))

    #evaluate
    truths = torch.squeeze(test_dataset.fp).detach().numpy()
    print(f"NMAE: {NMAE_function(test_out,truths)}")
    losses["NMAE_test"].append(NMAE_function(test_out,truths))
    transformed_preds = test_dataset.inverse_transform(test_out)
    evaluation_metrics = test_dataset.evaluate()
    losses["NMAE_test_transformed"].append(evaluation_metrics["NMAE"])
    losses["MSE_test_transformed"].append(evaluation_metrics["MSE"])
    losses["accuracy"].append(evaluation_metrics["Accuracy"])
    losses["IoU"].append(evaluation_metrics["IOU"])

    #write_to_file(f"{epoch + 1}, loss: {running_loss/(i+1)}, test loss: {test_error/(i_test+1)}, NMAE test: {NMAE(test_out,truths)}, NMAE test trasformed: {evaluation_metrics['NMAE']}, MSE test transformed: {evaluation_metrics['MSE']}, IoU {evaluation_metrics['IOU']}")

    for flux_mode in flux_evaluation:
        flux_metrics = test_dataset.evaluate_flux(mode=flux_mode)
        losses[f"flux_{flux_mode}"]["MAE"].append(flux_metrics["MAE"])
        losses[f"flux_{flux_mode}"]["R2"].append(flux_metrics["R2"])

    write_to_file(f"{epoch + 1}, loss: {running_loss/(i+1)}, test loss: {test_error/(i_test+1)}, NMAE test: {NMAE_nans(test_out,truths)}, NMAE test trasformed: {evaluation_metrics['NMAE']}, MSE test transformed: {evaluation_metrics['MSE']}, IoU {evaluation_metrics['IOU']}, flux metrics (checkerboard 5): {flux_metrics}")
    
    wandb.log({
        "epoch": epoch + 1,
        "train/loss": running_loss / (i + 1),
        "test/loss": test_error / (i_test + 1),
        "test/NMAE": NMAE_nans(test_out, truths),
        "test/NMAE_transformed": evaluation_metrics['NMAE'],
        "test/MSE": evaluation_metrics['MSE'],
        "test/IoU": evaluation_metrics['IOU'],
        **{f"flux/{k}": v for k, v in flux_metrics.items()}
    }, step=epoch)


    # every five epochs plot and save 
    if epoch % 5 == 0:
        n=0
        og_fps = test_dataset.fp_untransformed
        fps = test_dataset.fp.detach().numpy()
        fig, ax = plt.subplots(4,4, figsize=(10, 10))
        for axis, fn in enumerate(image_plots):
            ax[0,axis].imshow(np.reshape(test_dataset.predictions[fn+n,:], (size[0],size[1])), origin="lower")
            ax[1,axis].imshow(np.reshape(fps[fn+n,:], (size[0],size[1])), origin="lower") 
            ax[2,axis].imshow(np.reshape(transformed_preds[fn+n,:], (size[0],size[1])), origin="lower")
            ax[3,axis].imshow(np.reshape(og_fps[fn+n,:], (size[0],size[1])), origin="lower")   
            ax[0,axis].set_title(f"prediction, \n sample {fn}")
            ax[1,axis].set_title(f"truth, \n sample {fn} \n ({str(image_dates[axis][:-10])})")
            ax[2,axis].set_title(f"transformed prediction, \n sample {fn}")
            ax[3,axis].set_title(f"original truth, \n sample {fn}")
            for a in range(4):
                ax[a,axis].xaxis.set_ticks([])
                ax[a,axis].yaxis.set_ticks([])

        plt.tight_layout()
        savename = os.path.join(training_img_dir, f"{model_name}_{epoch}.png")
        wandb.log({
            "overview": wandb.Image(fig, caption=f"{model_name} epoch {epoch + 1}")
        }, step=epoch + 1)
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.close()

    ## save checkpoint every 50 epochs
    checkpoint_path = os.path.join(args.output_dir, f"{model_name}_{epoch}.pt")

    if epoch % 50 ==0:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,
                    'learning_rate':lr,
                    }, checkpoint_path)
        
        # Log to W&B as a versioned artifact
        checkpoint_artifact = wandb.Artifact(
            name=f"{model_name}-checkpoint",   # e.g. myModel-checkpoint:v0
            type="model",
            description="Model checkpoint saved during training"
        )
        checkpoint_artifact.add_file(checkpoint_path)
        wandb.log_artifact(checkpoint_artifact)
            


    if epoch == num_epochs - 1:
        test_out = np.reshape(np.squeeze(test_out), (len(test_out), size[0],size[1]))
        transformed_preds = np.reshape(np.squeeze(transformed_preds), (len(transformed_preds), size[0],size[1]))
        fps = np.reshape(fps, (len(test_dataset.fp), size[0],size[1]))
        fp_trans = np.reshape(test_dataset.fp, (len(test_dataset.fp), size[0],size[1]))
        
        data_vars = {'predictions':(['time', "lat", "lon"], test_out, 
                                {'space': 'transformed', 'type':"prediction", 'emulated_with': model_name}),
                    'trans_predictions':(['time', "lat", "lon"], transformed_preds, 
                                {'space': 'original', 'type':"prediction",'emulated_with': model_name}),
                    'fp':(['time', "lat", "lon"], fps, 
                                {'space': 'original', 'type':"truth"}),
                    'trans_fp':(['time', "lat", "lon"], fp_trans, 
                                {'space': 'transformed', 'type':"truth"})}

        # define coordinates
        coords = {'time': (['time'], time_values),
                'lat': (['lat'], list(range(size[0]))),
                'lon': (['lon'],  list(range(size[1])))}

        # define global attributes
        attrs = {'creation_date':str(datetime.now()), "model":model_name}
        

        # create dataset
        ds = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)
        
        netcdf_save_path = os.path.join(args.output_dir, "sample_predictions_training.nc")
        ds.to_netcdf(netcdf_save_path)

        print(f"Saved sample_predictions_training.nc to {netcdf_save_path}")

        # Log to W&B as a versioned artifact with explicit name
        preds_artifact = wandb.Artifact(
            name=f"{model_name}-predictions",   # e.g. myModel-predictions:v0
            type="dataset",
            description="Sample predictions saved during training"
        )
        preds_artifact.add_file(netcdf_save_path)
        wandb.log_artifact(preds_artifact)

    if epoch == 102:
        ## replaced NMAE with NMAE_nans in the whole file!
        if NMAE_nans(test_out,truths) == 1:
            print("no learning is happening! early stopping")
            break
wandb.finish()  # End wandb session

print("Finished Training")