import sys

# delete before use!!
sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")
sys.path.insert(0,"/user/work/ef17148/oldstuff/ef17148/.conda/envs/new_graphnet/lib/python3.12/site-packages")

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

parser = argparse.ArgumentParser(description="Load parameters")
parser.add_argument("file_name", help="parameter file name")
parser.add_argument("--file_path", help="parameter file path")

args = parser.parse_args()
file_name = args.file_name
file_path = args.file_path

print(file_name, file_path)

#### 1 Set up

## make this importable!
path="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_newversion/"

def write_to_file(message):
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
    

parameters = load_file(file_name, file_path) 

print("PARAMETERS:")
print(parameters)

model_name = parameters["model_name"]
print(model_name)


if "seed" in (parameters.keys()):
    print(parameters["seed"])
    np.random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.cuda.manual_seed(parameters["seed"])
    random.seed(parameters["seed"])

else:
    print("34")
    np.random.seed(34)
    torch.manual_seed(34)
    torch.cuda.manual_seed(34)
    random.seed(34)



# make files
os.makedirs(f"{path}{model_name}", exist_ok=True)
os.mkdir(f"{path}{model_name}/training_imgs")
f = open(f"{path}{model_name}/{model_name}_updates.txt", "x")
f.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
#write_to_file("loading data")
write_to_file("loading data")

#### 2 Load Data
train_load_data = copy.deepcopy(parameters["train_load_data"])
# load train parameters and upload with any changes to test data
test_load_data = copy.deepcopy(parameters["train_load_data"])
test_load_data.update(parameters["test_load_data"])
print(train_load_data)
print(test_load_data)

data = LoadSquareSatelliteData(**train_load_data, load_everything=True)
test_data = LoadSquareSatelliteData(**test_load_data,load_everything=True)

write_to_file("setting up data")

# extract inputs

input_variables = parameters["variables"]

inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)

test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)

# the model gets built with respect to a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

write_to_file("setting up model")
print("setting up model")

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

# all the necessary data is already in the loaders, so we can delete the objects

image_plots = random.sample(list(range(len(test_inputs))), k=4)
image_dates = np.datetime_as_string(test_data.fp_data_full.time.values[image_plots])

del data, test_data

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


NMAE_function = NMAE


#### 3 Dump info
print("saving grids etc")

# save transform parameters, grid and training settings
with open(f"{path}{model_name}/grid_{model_name}.pickle", 'wb') as handle:
    pickle.dump(grid, handle)

with open(f"{path}{model_name}/transform_parameters_{model_name}.pickle", 'wb') as handle:
    pickle.dump(train_dataset.transform_parameters, handle)

with open(f"{path}{model_name}/training_settings_{model_name}.json", 'w') as handle:
    json.dump(parameters, handle)


epoch_so_far = 0
if torch.cuda.is_available():
    model.cuda()

#### 4 Train loop
for epoch in range(302):
    epoch=epoch+epoch_so_far
    running_loss = 0.0
    print(f"Start Epoch: {epoch}")
    start = time.time()
    for i, batch in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels, true_fp = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # zero the parameter gradients
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
        test_error += criterion_test(model(ins), labels).item()
        test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = np.squeeze(model(ins).detach().cpu().numpy())
    
    
    losses["train"].append(running_loss/(i+1))
    losses["test"].append(test_error/(i_test+1))

    #evaluate
    truths = torch.squeeze(test_dataset.fp).detach().numpy()
    print(f"NMAE: {NMAE(test_out,truths)}")
    losses["NMAE_test"].append(NMAE(test_out,truths))
    transformed_preds = test_dataset.inverse_transform(test_out)
    evaluation_metrics = test_dataset.evaluate()
    losses["NMAE_test_transformed"].append(evaluation_metrics["NMAE"])
    losses["MSE_test_transformed"].append(evaluation_metrics["MSE"])
    losses["accuracy"].append(evaluation_metrics["Accuracy"])
    losses["IoU"].append(evaluation_metrics["IOU"])

    write_to_file(f"{epoch + 1}, loss: {running_loss/(i+1)}, test loss: {test_error/(i_test+1)}, NMAE test: {NMAE(test_out,truths)}, NMAE test trasformed: {evaluation_metrics['NMAE']}, MSE test transformed: {evaluation_metrics['MSE']}, IoU {evaluation_metrics['IOU']}")

    for flux_mode in flux_evaluation:
        flux_metrics = test_dataset.evaluate_flux(mode=flux_mode)
        losses[f"flux_{flux_mode}"]["MAE"].append(flux_metrics["MAE"])
        losses[f"flux_{flux_mode}"]["R2"].append(flux_metrics["R2"])

    write_to_file(f"{epoch + 1}, loss: {running_loss/(i+1)}, test loss: {test_error/(i_test+1)}, NMAE test: {NMAE_nans(test_out,truths)}, NMAE test trasformed: {evaluation_metrics['NMAE']}, MSE test transformed: {evaluation_metrics['MSE']}, IoU {evaluation_metrics['IOU']}, flux metrics (checkerboard 5): {flux_metrics}")
    
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
        savename = f"{path}{model_name}/training_imgs/{model_name}_{epoch}.png"
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.close()

    ## save checkpoint every 50 epochs
    if epoch % 50 ==0:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,
                    'learning_rate':lr,
                    }, f"{path}{model_name}/{model_name}_{epoch}.pt")

    if epoch == 350:
        test_out = np.reshape(np.squeeze(test_out), (len(test_out), size[0],size[1]))
        data_vars = {'predictions':(['time', "lat", "lon"], test_out, 
                                {'space': 'transformed', 'type':"prediction", 'emulated_with': model_name}),
                    'trans_predictions':(['time', "lat", "lon"], transformed_preds, 
                                {'space': 'original', 'type':"prediction",'emulated_with': model_name}),
                    'fp':(['time', "lat", "lon"], np.reshape(fps, (len(test_dataset.fp),size[0],size[1])), 
                                {'space': 'original', 'type':"truth"}),
                    'trans_fp':(['time', "lat", "lon"], np.reshape(test_dataset.fp, (len(test_dataset.fp), size[0],size[1])), 
                                {'space': 'transformed', 'type':"truth"})}

        # define coordinates
        coords = {'time': (['time'], test_data.met.time.values),
                'lat': (['lat'], list(range(size[0]))),
                'lon': (['lon'],  list(range(size[1])))}

        # define global attributes
        attrs = {'creation_date':str(datetime.now()), "model":model_name}
        

        # create dataset
        ds = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)
        

        ds.to_netcdf(f"{path}{model_name}/sample_predictions_training.nc")
        
    if epoch == 102:
        ## replaced NMAE with NMAE_nans in the whole file!
        if NMAE_nans(test_out,truths) == 1:
            print("no learning is happening! early stopping")
            break


print("Finished Training")