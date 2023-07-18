import sys

#import cartopy
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import einops
import numpy as np
import torch
import os
import pickle

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import *
from model.data.load_data import *
from model.forecast import GraphSatelliteForecaster
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime


torch.manual_seed(33)

path="/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/trained_models"

def write_to_file(message):
    f = open(f"{path}/{model_name}/{model_name}_updates.txt", "a")
    f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
    f.close()

model_name="satellite_50x50_default"
print(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


continue_training = False
if not continue_training:
    try:
        os.mkdir(f"{path}/{model_name}")
        os.mkdir(f"{path}/{model_name}/training_imgs")
        f = open(f"{path}/{model_name}/{model_name}_updates.txt", "x")
        f.close()
    except OSError:
        print("Folder already exists! Consider changing model name if you are training from scratch or making continue_training=True to continue training this model")
else:
    print("attempting to load model to continue training")
    def getint(name):
        num = name.split('_')[-1]
        num = num.split('.')[0]
        return int(num)

    try:
        files = sorted(glob.glob(f"{path}/{model_name}/{model_name}_*.pt"), key=getint)
        checkpoint_to_load = files[-1] 
    except IndexError:
        print(f"No files were found at path {path}/{model_name}/{model_name}_*.pt")

    try:
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
    except Exception as e:
        print(f"checkpoint loading failed with error {e}")


write_to_file(f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
write_to_file("loading data")

## load training and testing data
data = LoadSatelliteData("201[4-5]", region="BRAZIL", metsize=50, size = 50, verbose=True, topog="default", cut_met = False, met_datadir="/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_onlyvalid_50_")
test_data = LoadSatelliteData(2016, region="BRAZIL", metsize=50, size =50, topog="default", verbose=True, cut_met = False, met_datadir="/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_onlyvalid_50_")


write_to_file("setting up data")

## set up inputs
others=["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "distance_centre", "x_coords", "y_coords"]
topog=True

jumps = [6,12]

variables_past = {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[0], "surface_air_pressure":[0]}

variables_nopast = {}


grid, inputs, names, data = get_all_inputs_graphnet_satellite_v4(data, variables_past, jumps, variables_nopast, topog=topog, others=others, centered_coords=True)
_, test_inputs, _, test_data = get_all_inputs_graphnet_satellite_v4(test_data, variables_past, jumps, variables_nopast, topog=topog, others=others, centered_coords=True)



write_to_file("setting up model")

class MSE_weighted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss(reduction="none")
        loss = torch.mean(criterion(output, target), dim=1)
        fp_sum = torch.abs(torch.sum(target, dim=(1,2)))
        return torch.mean(torch.mul(torch.squeeze(loss),0.1*fp_sum))

class MSE_assymetric(nn.Module):
    def __init__(self, alpha=4):
        self.alpha=alpha
        super().__init__()
        

    def forward(self, output, target):
        loss=target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        return loss
    


test_batch_size=10

## set up dataset (transform inputs/outputs) and data loader

train_dataset = FootprintsDataset(np.copy(inputs), np.copy(data.fp_data), standardise=False, transform_output="boxcox", feature_dim=np.shape(inputs)[-1]-len(others)-topog, aux_dim=len(others)+topog, zeroing=True,  clever_transform_2=True, input_names=names)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

dataset_params = {"standardise":train_dataset.standardise_inputs, "transform_output":train_dataset.transform_output, "feature_dim":np.shape(inputs)[-1]-len(others)-topog, "aux_dim":len(others)+topog, "zeroing":train_dataset.zeroing,  "clever_transform_2":True, "input_names":names}


test_dataset = FootprintsDataset(np.copy(test_inputs[:,:,:]), np.copy(test_data.fp_data[:,:]), **dataset_params, test_mode={"clever_transformers":train_dataset.transformers,  "boxcox":train_dataset.boxcox})
test_loader = DataLoader(test_dataset, batch_size=10)


## save reference grid, transformers and parameters 
with open(f"{path}{model_name}/grid_{model_name}.pickle", 'wb') as handle:
    pickle.dump(grid, handle)

with open(f"{path}{model_name}/transformers_{model_name}.pickle", 'wb') as handle:
    pickle.dump({"clever_transformers":train_dataset.transformers,  "boxcox":train_dataset.boxcox}, handle)

train_dataloader_params = {"year":data.date, "region":data.region, "freq":data.freq, "metsize":data.metsize, "size":data.size, "topog":"default", "verbose":True, "cut_met":False, "met_datadir":"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_"}

input_params={"variables_past":variables_past, "jumps":jumps, "variables_nopast":variables_nopast, "others":others, "topog":topog, "centered_coords":True}

model_params = {"whole_world":False, "feature_dim":np.shape(inputs)[-1]-len(others)-topog, "aux_dim":len(others)+topog,"num_blocks":4, "node_dim":64, "edge_dim":64, "hidden_layers_processor_node":2, "hidden_layers_processor_edge":2,  "hidden_layers_decoder":1, "hidden_dim_processor_node":16, "hidden_dim_processor_edge":16, "hidden_dim_decoder":16, "resolution":4, "output_dim":1, "residuals":False}

params = {"train_dataloader_params":train_dataloader_params, "input_params":input_params, "dataset_params":dataset_params, "model_params":model_params}

with open(f"{path}{model_name}/params_{model_name}.pickle", 'wb') as handle:
    pickle.dump(params, handle)

## set up model, loss, criterion
lr=5e-5
model = GraphSatelliteForecaster(grid, **model_params)

criterion=MSE_assymetric(alpha=2)
# NN is trailed with MSE_assymetric but stats are outputted with normal MSE so it's more interpretable
criterion_test = torch.nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)


losses = {"train":[], "test":[], "NMAE_test":[], "MSE_test_transformed":[], "NMAE_test_transformed":[]}

epoch_so_far = 0
if torch.cuda.is_available():
    model.cuda()

if continue_training:
    # load model and attributes
    optimizer = optim.AdamW(model.parameters(), lr=checkpoint["learning_rate"])
    lr = checkpoint["learning_rate"]

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_so_far = checkpoint['epoch']
    losses = checkpoint['loss']
    model.train()
    print("model loaded")
    



write_to_file("starting training")

for epoch in range(301):  # loop over the dataset multiple times
    epoch=epoch+epoch_so_far
    running_loss = 0.0
    print(f"Start Epoch: {epoch}")
    start = time.time()
    for i, batch in enumerate(train_loader):

        ins, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

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

        ins, labels = batch[0].to(device), batch[1].to(device)
        test_error += criterion_test(model(ins), labels).item()
        test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = np.squeeze(model(ins).detach().cpu().numpy())
    
    
    losses["train"].append(running_loss/(i+1))
    losses["test"].append(test_error/(i_test+1))

    truths = torch.squeeze(test_dataset.fp).detach().numpy()
    print(f"NMAE: {NMAE(test_out,truths)}")
    losses["NMAE_test"].append(NMAE(test_out,truths))
    transformed_preds = test_dataset.inverse_transform(test_out)
    NMAE_trans, mse = test_dataset.prediction_error()
    losses["NMAE_test_transformed"].append(NMAE_trans)
    losses["MSE_test_transformed"].append(mse)

    write_to_file(f"{epoch + 1}, loss: {running_loss/(i+1)}, test loss: {test_error/(i_test+1)}, NMAE test: {NMAE(test_out,truths)}, NMAE test trasformed: {NMAE_trans}, MSE test transformed: {mse}")

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
        savename = f"{path}/{model_name}/training_imgs/{model_name}_{epoch}.png"
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.close()

    if epoch % 50 ==0:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,
                    'learning_rate':lr,
                    }, f"{path}/{model_name}/{model_name}_{epoch}.pt")




print("Finished Training")




























