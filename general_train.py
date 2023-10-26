import sys
#sys.path.append("/user/work/ef17148/oldstuff/ef17148/.conda/envs/GCN_env/lib/python3.8/site-packages/")
#sys.path.insert(0, "/user/work/ef17148/oldstuff/ef17148/.conda/envs/graphnet/lib/python3.8/site-packages/")


sys.path.append('/user/work/ef17148/parallel_models/parallel_baselines/')
from trees_emulator.load_data import *


#import cartopy
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import einops
import numpy as np
sys.path.insert(0, "/user/work/ef17148/oldstuff/ef17148/.conda/envs/GCN_env/lib/python3.8/site-packages/")
import torch
import os
import pickle

sys.path.insert(0, "/user/work/ef17148/GCN/graphnet/")
sys.path.insert(1, "/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/")
from graphnet_LPDM_emulator.model.layers.encoder import *
from graphnet_LPDM_emulator.model.layers.decoder import *
from graphnet_LPDM_emulator.model.layers.processor import *
from graphnet_LPDM_emulator.model.layers.graph_net_block import *
from graphnet_LPDM_emulator.model.data.dataloader_graphnet import *
from graphnet_LPDM_emulator.model.data.load_data import *
from graphnet_LPDM_emulator.model.forecast import GraphSatelliteForecaster
from graphnet_LPDM_emulator.model.loss_functions import *

import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse

parser = argparse.ArgumentParser(description="Load parameters")
parser.add_argument("file_name", help="parameter file name")
parser.add_argument("--file_path", help="parameter file path")

args = parser.parse_args()
file_name = args.file_name
file_path = args.file_path

print(file_name, file_path)



torch.manual_seed(33)

path="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/"

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
    

# TODO make this an argument

parameters = load_file(file_name, file_path) 


model_name = parameters["model_name"]
print(model_name)


# make files
os.mkdir(f"/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/{model_name}")
os.mkdir(f"/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/{model_name}/training_imgs")
f = open(f"/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/{model_name}/{model_name}_updates.txt", "x")
f.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
#write_to_file("loading data")
write_to_file("loading data")


train_load_data = copy.deepcopy(parameters["train_load_data"])
# load train parameters and upload with any changes to test data
test_load_data = copy.deepcopy(parameters["train_load_data"])
test_load_data.update(parameters["test_load_data"])
print(train_load_data)
print(test_load_data)

data = LoadSatelliteData(**train_load_data)
test_data = LoadSatelliteData(**test_load_data)

write_to_file("setting up data")

variables = parameters["variables"]
met_variables = variables["met_variables"]
others = variables["others"]
topog=variables["topog"]
jumps = variables["jumps"]

grid, inputs, names, data = get_all_inputs_graphnet_satellite_v4(data, met_variables, jumps, {}, topog=topog, others=others, centered_coords=variables["centered_coords"])

_, test_inputs, _, test_data = get_all_inputs_graphnet_satellite_v4(test_data, met_variables, jumps, {}, topog=topog, others=others, centered_coords=variables["centered_coords"])

write_to_file("setting up model")
print("setting up model")

test_batch_size=10


train_dataset = FootprintsDatasetV2(inputs, data.fp_data, input_names=names, **parameters["dataloader_parameters"])

print(train_dataset.transform_parameters)

test_dataset = FootprintsDatasetV2(test_inputs, test_data.fp_data, input_names=names, test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

lr = parameters["learning_rate"]
print(lr)

model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=np.shape(inputs)[-1]-len(others)-topog, aux_dim=len(others)+topog, **parameters["model_parameters"])

criterion = eval(parameters["loss_functions"]["criterion"])

criterion_test = eval(parameters["loss_functions"]["criterion_test"])

optimizer = optim.AdamW(model.parameters(), lr=lr)


losses = {"train":[], "test":[], "NMAE_test":[], "MSE_test_transformed":[], "NMAE_test_transformed":[], "accuracy":[], "IoU":[]}


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

for epoch in range(300):
    epoch=epoch+epoch_so_far
    running_loss = 0.0
    print(f"Start Epoch: {epoch}")
    start = time.time()
    for i, batch in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels = batch[0].to(device), batch[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(ins)
        loss = criterion(outputs, labels)
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

    # every five epochs plot and save 
    if epoch % 5 == 0:
        n=0
        og_fps = test_dataset.fp_untransformed
        fps = test_dataset.fp.detach().numpy()
        fig, ax = plt.subplots(4,4, figsize=(10, 10))
        for axis, fn in enumerate([10,50,190,600]):
            ax[0,axis].imshow(np.reshape(test_dataset.predictions[fn+n,:], (data.size,data.size)), origin="lower")
            ax[1,axis].imshow(np.reshape(fps[fn+n,:], (data.size,data.size)), origin="lower") 
            ax[2,axis].imshow(np.reshape(transformed_preds[fn+n,:], (data.size,data.size)), origin="lower")
            ax[3,axis].imshow(np.reshape(og_fps[fn+n,:], (data.size,data.size)), origin="lower")   
            ax[0,axis].set_title(f"prediction, \n sample {fn+n}")
            ax[1,axis].set_title(f"truth, \n sample {fn+n}")
            ax[2,axis].set_title(f"transformed prediction, \n sample {fn+n}")
            ax[3,axis].set_title(f"original truth, \n sample {fn+n}")
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




print("Finished Training")