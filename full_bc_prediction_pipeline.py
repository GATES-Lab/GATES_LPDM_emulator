import sys

'''
# delete before use!!
sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")
sys.path.insert(0,"/user/work/yl18410/miniconda3/envs/new_graphnet_v2/lib/python3.12/site-packages")
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

from model.layers.encoder import *
from model.layers.decoder import *
from model.layers.processor import *
from model.layers.graph_net_block import *
from model.data.dataloader_graphnet import *
#from model.data.load_data import *
from model.data.load_data_coarsening import *
from model.forecast import GraphSatelliteForecaster, GraphSatelliteForecasterClassifier, GraphSatelliteForecasterConvClassifier
from model.loss_functions import *


import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse

import random
from datetime import date

import wandb
import re

# Set your W&B API key to log in automatically
os.environ["WANDB_API_KEY"] = "11d787a211e05ca01c50131c5724e375cd5d3364"  # <<-- REPLACE THIS
wandb.login()


def baseline_mol(desired_data,months,desired_year):
    '''
    Function used to get the value for the input and output
    '''
    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]                                                                      
    # Load the CSV file
    #df = pd.read_csv('Analysis/CH4_Semihemispheric_modelled_mole_fractions.csv')
    df = pd.read_csv('/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv')
    baseline_list = np.zeros((total_data_points,4))

    '''
    # Specify the year and month you're interested in
    # Filter the data for the specific year and month
    filtered_data = df[(df['Year'] == specific_year) & (df['Month'] == specific_month)]

    # Extract the 4th, 5th, 6th, and 7th columns
    selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values

    # Display the results
    print(selected_columns)
    '''

    # Iterate through the different numbers 
    #months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    #year = 2016
    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)

    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]
    north_list = np.zeros(total_data_points) # Creates a list of size N with None values
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)

    for month in months:
        # Cams field for a particular month
        if desired_year > 2017:
            # Change made due to the naming of the data
            cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{desired_year}{month}_CAMS-inversion_climatology.nc")
        else:
            cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{desired_year}{month}_CAMS-inversion.nc")    
        # Extract year and month
        #import ipdb; ipdb.set_trace()
        years = np.array([np.datetime64(date, 'Y').astype(int) + 1970 for date in datetime_array])
        months = np.array([np.datetime64(date, 'M').astype(int) % 12 + 1 for date in datetime_array])

        # Desired year and month
        #desired_year = 2016
        desired_month = month
        print('desired month',desired_month)
        
        '''
        coarse_cams  = cams.coarsen(lon=desired_data.coarsening_factor, lat=desired_data.coarsening_factor, boundary="pad").mean()
        '''
        # Find the index of the first occurrence
        indices = np.where((years == desired_year) & (months == int(desired_month)))[0]
        print(indices)
        if len(indices)>0:
            # Get the inputs
            filtered_data = df[(df['Year'] == desired_year) & (df['Month'] == int(desired_month))]
            # Extract the 4th, 5th, 6th, and 7th columns
            selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values
            baseline_list[indices] = selected_columns/1000 # Convert from parts per trillion to parts per million

            print(indices[0])
            # Multply the first value with all the other values of the array
            print(cams.vmr_n.shape)
            # CAMS field should be stationary over the period of a month
            #import ipdb; ipdb.set_trace()

            north_mol = np.sum(cams.vmr_n * desired_data.locs.particle_locations_n[:,:,indices], axis=(0,1))
            south_mol = np.sum(cams.vmr_s * desired_data.locs.particle_locations_s[:,:,indices], axis=(0,1))
            east_mol = np.sum(cams.vmr_e * desired_data.locs.particle_locations_e[:,:,indices], axis=(0,1))
            west_mol = np.sum(cams.vmr_w * desired_data.locs.particle_locations_w[:,:,indices], axis=(0,1))
            '''
            north_mol = np.sum(coarse_cams.vmr_n * desired_data.locs.particle_locations_n[:,:,indices], axis=(0,1))
            south_mol = np.sum(coarse_cams.vmr_s * desired_data.locs.particle_locations_s[:,:,indices], axis=(0,1))
            east_mol = np.sum(coarse_cams.vmr_e * desired_data.locs.particle_locations_e[:,:,indices], axis=(0,1))
            west_mol = np.sum(coarse_cams.vmr_w * desired_data.locs.particle_locations_w[:,:,indices], axis=(0,1))
            '''
            '''
            north_mol = np.sum(cams.vmr_n * desired_data.fp_data_full.particle_locations_n[:,:,indices], axis=(0,1))
            south_mol = np.sum(cams.vmr_s * desired_data.fp_data_full.particle_locations_s[:,:,indices], axis=(0,1))
            east_mol = np.sum(cams.vmr_e * desired_data.fp_data_full.particle_locations_e[:,:,indices], axis=(0,1))
            west_mol = np.sum(cams.vmr_w * desired_data.fp_data_full.particle_locations_w[:,:,indices], axis=(0,1))
            '''
            #import ipdb; ipdb.set_trace()
            north_list[indices] = north_mol
            south_list[indices] = south_mol
            east_list[indices] = east_mol
            west_list[indices] = west_mol


        print(baseline_list)
        #cams = xr.open_dataset("/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_201611_CAMS-inversion.nc")
        # Making the assumption that the values in the month are not different, get the first value
        # Multiple the different values
    
    return baseline_list, north_list, south_list, east_list, west_list

def write_to_file(path, model_name,message):
    f = open(f"{path}{model_name}/{model_name}_updates.txt", "a")
    f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
    f.close()




'''
def train_bc_prediction_pipeline(_hparams,_practice):
    folder_name = 'boundary_condition'
    inference_path=f"/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/graph_weather/{folder_name}/"
    
    if not os.path.exists(inference_path):
        os.makedirs(inference_path)
    
    parameters = _hparams
    name_train_freq =parameters["train_load_data"]["freq"]
    #name_output_transforms = parameters["dataloader_parameters"]["output_transforms"][0].replace('"', '').replace('[', '').replace(']', '')
    name_train_data = str(parameters['train_load_data']['year'])
    name_lr = parameters['learning_rate']
    name_coarsening = str(parameters["train_load_data"]["coarsening_factor"])
    #name_alpha = parameters['loss_weight']
    name_normalization = parameters['normalization']
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")

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
    
    

    name_train_data = name_train_data.replace("[", "").replace("]", "")    

    name_seed = parameters['seed']
    name_num_classes = str(parameters['num_classes'])
    
    name_inference_epochs = parameters["epochs"]
    name_baselines = str(parameters['use_baselines'])
    if "size" in (parameters["train_load_data"].keys()):
        name_size = str(parameters["train_load_data"]['size'])
        model_name = f"num_classes-{name_num_classes}_baselines-{name_baselines}_size-{name_size}_coarsening-{name_coarsening}_normalization-{name_normalization}_trainyear-{name_train_data}_trainfreq-{name_train_freq}_epochs-{name_inference_epochs}_lr-{name_lr}_seed-{name_seed}_date-{d4}"
    else:
        model_name = f"num_classes-{name_num_classes}_baselines-{name_baselines}_coarsening-{name_coarsening}_normalization-{name_normalization}_trainyear-{name_train_data}_trainfreq-{name_train_freq}_epochs-{name_inference_epochs}_lr-{name_lr}_seed-{name_seed}_date-{d4}"

    if _practice:
        model_name = 'practice_run'
        
        with open("Analysis/quickload_data.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        data, test_data, grid, inputs, names, test_inputs =loaded_data["data"],loaded_data["test_data"],loaded_data["grid"], loaded_data["inputs"], loaded_data["names"], loaded_data["test_inputs"]
        

    else:
        print('Not using practice')
        model_folder = f"{inference_path}{model_name}"
        if os.path.exists(model_folder):
            print('Exiting simulation')
            return None
    
     # make files
    model_folder = f"{inference_path}{model_name}"
    img_folder =  f"{model_folder}/training_imgs"
    model_folder_Exist = os.path.exists(model_folder)
    if not model_folder_Exist:
       # Create a new directory because it does not exist
       os.makedirs(model_folder)

    img_folder_Exist = os.path.exists(img_folder)
    if not img_folder_Exist:
       # Create a new directory because it does not exist
       os.makedirs(img_folder)
    text_path = f"{model_folder}/{model_name}_updates.txt"
    text_path_Exist = os.path.exists(text_path)

    if not text_path_Exist:
        f = open(f"{inference_path}/{model_name}/{model_name}_updates.txt", "x")
        f.close()
    
    # Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(inference_path,model_name,f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
    #write_to_file("loading data")
    write_to_file(inference_path,model_name,"loading data")
    write_to_file(inference_path,model_name,"setting up data")

    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])
    print(train_load_data)
    print(test_load_data)
    
    
    if "size" in (parameters["train_load_data"].keys()):
        print('Using square domain')
        data = LoadSquareSatelliteData(**train_load_data)
        test_data = LoadSquareSatelliteData(**test_load_data)
        
    else:    
        print('Using fixed domain')
        data = LoadDomainSatelliteData(**train_load_data)
        test_data = LoadDomainSatelliteData(**test_load_data)
        
    

    
    train_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # TODO: Nawid- get the train year and the test year from the trainload data 
    train_year = 2015
    
    test_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    test_year = 2016
    
    baseline_list, north_list, south_list, east_list, west_list = baseline_mol(data,train_months, train_year)
    test_baseline_list, test_north_list, test_south_list, test_east_list, test_west_list = baseline_mol(test_data,test_months, test_year)

    outputs = np.stack((north_list,south_list, east_list,west_list),axis=1)
    test_outputs = np.stack((test_north_list,test_south_list, test_east_list,test_west_list),axis=1)

    if parameters['normalization'] =='separate':
        outputs_mean_values, outputs_std_values = np.mean(outputs,axis=0), np.std(outputs,axis=0)
        baseline_mean_values, baseline_std_values = np.mean(baseline_list,axis=0), np.std(baseline_list,axis=0)
        outputs =  (outputs-outputs_mean_values)/outputs_std_values        
        baseline_list =  (baseline_list-baseline_mean_values)/baseline_std_values

        test_outputs = (test_outputs-outputs_mean_values)/outputs_std_values
        test_baseline_list = (test_baseline_list-baseline_mean_values)/baseline_std_values

    elif parameters['normalization'] =='all':
        outputs_mean_values, outputs_std_values = np.mean(outputs), np.std(outputs)
        baseline_mean_values, baseline_std_values = np.mean(baseline_list), np.std(baseline_list)
        outputs =  (outputs-outputs_mean_values)/outputs_std_values
        baseline_list =  (baseline_list-baseline_mean_values)/baseline_std_values

        test_outputs = (test_outputs-outputs_mean_values)/outputs_std_values
        test_baseline_list = (test_baseline_list-baseline_mean_values)/baseline_std_values
    
    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    input_variables = parameters["variables"]

    inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)
    
    test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
    
    use_baselines = parameters['use_baselines']
    if use_baselines:
        aux_index= 4
    else:
        aux_index = 0

    train_batch_size = 5
    test_batch_size=5
    print(train_load_data)
    print(test_load_data)
    write_to_file(inference_path,model_name,"Before loading data")
    #grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    train_dataset = BoundaryDataset(inputs,baseline_list,outputs,use_baselines=use_baselines,input_names=names, **parameters["dataloader_parameters"])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = BoundaryDataset(test_inputs,test_baseline_list,test_outputs,use_baselines= use_baselines,input_names=names, **parameters["dataloader_parameters"])
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    #aux_dim = len(input_variables["static_variables"]) 
    feature_dim=np.shape(inputs)[-1]
    
    num_classes = parameters['num_classes']
    # Should probably update the name!!
    model = GraphSatelliteForecasterClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes, **parameters["model_parameters"])

    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    lr = parameters["learning_rate"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    normalization_vals = {"outputs_mean":outputs_mean_values,"outputs_std": outputs_std_values}
    losses = {"train":[], "test":[],"individual_summed_test_MAE":[]}
    #losses = {"train":[], "test":[],"individual_mol_test_MAE":[],"individual_summed_test_MAE":[]}
    print("saving grids etc")

    # save transform parameters, grid and training settings
    with open(f"{inference_path}{model_name}/grid_{model_name}.pickle", 'wb') as handle:
        pickle.dump(grid, handle)

    with open(f"{inference_path}{model_name}/transform_parameters_{model_name}.pickle", 'wb') as handle:
        pickle.dump(train_dataset.transform_parameters, handle)

    with open(f"{inference_path}{model_name}/training_settings_{model_name}.json", 'w') as handle:
        json.dump(parameters, handle)
    write_to_file(inference_path,model_name,"Before training model")
    epoch_so_far = 0
    if torch.cuda.is_available():
        model.cuda()
    # Nawid- true outputs of the test data (not using the train valeus as the training data is shuffled)
    denormalized_test_truths = (test_outputs*outputs_std_values) + outputs_mean_values
    denormalized_test_truths_summed = np.sum(denormalized_test_truths ,axis=1)


    best_test_loss = float("inf")
    best_epoch = -1

    num_epochs = parameters["epochs"]
    for epoch in range(num_epochs):
        epoch=epoch+epoch_so_far
        running_loss = 0.0
        #individual_mol_train_errors = np.zeros(output_num)
        
        print(f"Start Epoch: {epoch}")
        start = time.time()
        #import ipdb; ipdb.set_trace()
        # Nawid - Looking at saving the outputs of the data

        train_out = np.zeros((train_dataset.inputs.size()[0], num_classes))
        for i_train, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            ins, labels = batch[0].to(device), batch[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model_outputs = model(ins)
            loss = criterion(model_outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #train_out[i_train*test_batch_size:(i_test+1)*test_batch_size,:] = test_model_outputs.detach().cpu().numpy()
            #individual_mol_train_errors += np.mean(np.abs(((labels-model_outputs)*outputs_std_values).detach().cpu().numpy()),axis=0)

            end = time.time()
            del model_outputs 
            if i_train % 10==0:
                print(f"[{epoch + 1}, {i_train + 1:5d}] loss: {running_loss / (i_train + 1):.3f} Time: {end - start} sec")
        
                
        test_error = 0.0
        test_out = np.zeros((test_dataset.inputs.size()[0], num_classes))
        # Nawid- 
        
        for i_test, batch in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            ins, labels = batch[0].to(device), batch[1].to(device)
            test_model_outputs = model(ins)

            test_error += criterion_test(test_model_outputs, labels).item()
            test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = test_model_outputs.detach().cpu().numpy()
            
        test_loss = test_error / (i_test+1)               
        losses["train"].append(running_loss/(i_train+1))
        losses["test"].append(test_error/(i_test+1))

        # Nawid - test predictions data
        denormalized_test_predictions = (test_out*outputs_std_values) + outputs_mean_values
        denormalized_test_predictions_summed = np.sum(denormalized_test_predictions,axis=1)
        
        test_mae = np.mean(np.abs(denormalized_test_truths_summed - denormalized_test_predictions_summed))
        
        losses["individual_summed_test_MAE"].append(test_mae)
        write_to_file(inference_path, model_name,f"{epoch + 1}, loss: {running_loss/(i_train+1)}, test loss: {test_error/(i_test+1)}, denormzalised summed test loss:{test_mae}")
        
        # Save best model and predictions
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

            model_save_path = f"{inference_path}{model_name}/{model_name}_best.pt"
            prediction_save_path = f"{inference_path}{model_name}/{model_name}_best_predictions.npz"

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                "normalization": normalization_vals,
                'learning_rate': lr,
            }, model_save_path)

            # Save predictions and truths
            np.savez(
                prediction_save_path,
                predictions=test_out,
                truths=test_outputs
            )

            write_to_file(
                inference_path, model_name,
                f"Best model saved at epoch {epoch+1} with Test Loss: {best_test_loss:.6f}"
            )


        ## save checkpoint every 50 epochs
        if epoch % 50 ==0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,
                        "normalization":normalization_vals, 
                        'learning_rate':lr,
                        }, f"{inference_path}{model_name}/{model_name}_{epoch}.pt")

    # Nawid - Save for the final epoch
    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,
                        "normalization":normalization_vals, 
                        'learning_rate':lr,
                        }, f"{inference_path}{model_name}/{model_name}_{epoch}.pt")

    print("Finished Training")
    data_savename = f"{inference_path}{model_name}/{model_name}_final_epoch_data.pkl"
    data_dict = {'test_dataset_predictions':test_out,'test_dataset_truths':test_outputs}
    with open(data_savename, 'wb') as f:
        pickle.dump(data_dict, f)

'''
    
def train_bc_prediction_pipeline(_hparams,_practice):
    folder_name = 'boundary_condition'
    inference_path=f"/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/graph_weather/{folder_name}/"
    
    if not os.path.exists(inference_path):
        os.makedirs(inference_path)
    
    parameters = _hparams
    name_train_freq =parameters["train_load_data"]["freq"]
    #name_output_transforms = parameters["dataloader_parameters"]["output_transforms"][0].replace('"', '').replace('[', '').replace(']', '')
    name_train_data = str(parameters['train_load_data']['year'])
    name_lr = parameters['learning_rate']
    #name_coarsening = str(parameters["train_load_data"]["coarsening_factor"])
    #name_alpha = parameters['loss_weight']
    name_normalization = parameters['normalization']
    name_coarsening = parameters['train_load_data']['coarsening_factor']
    name_decoder = parameters['network_decoder']
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")

    # Safely get the 'domain_to_cut' dictionary or None if missing
    domain_to_cut = parameters["train_load_data"].get("domain_to_cut")

    # Initialize size
    name_size = 'full'

    if isinstance(domain_to_cut, dict):
        lat_domain = domain_to_cut.get("lat")
        lon_domain = domain_to_cut.get("lon")

        if lat_domain == [-29.5, -6] and lon_domain == [-60, -10]:
            name_size = 100
        elif lat_domain == [-35.5, -0.5] and lon_domain == [-77.5, 7.5]:
            name_size = 150
        elif lat_domain == [-39.7, 4.7] and lon_domain is None:
            name_size = 190



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
    
    name_train_data = name_train_data.replace("[", "").replace("]", "")    

    name_seed = parameters['seed']
    name_num_classes = str(parameters['num_classes'])
    
    name_inference_epochs = parameters["epochs"]
    name_baselines = str(parameters['use_baselines'])
    if "size" in (parameters["train_load_data"].keys()):
        # Version where i use the square domain
        name_size = str(parameters["train_load_data"]['size'])
        model_name = f"num_classes-{name_num_classes}_baselines-{name_baselines}_size-{name_size}_normalization-{name_normalization}_trainyear-{name_train_data}_trainfreq-{name_train_freq}_epochs-{name_inference_epochs}_lr-{name_lr}_seed-{name_seed}_date-{d4}"
    else:
        # Version where I use the fixed domain
        model_name = f"num_classes-{name_num_classes}_baselines-{name_baselines}__size-{name_size}_coarsening-{name_coarsening}_decoder-{name_decoder}_normalization-{name_normalization}_trainyear-{name_train_data}_trainfreq-{name_train_freq}_epochs-{name_inference_epochs}_lr-{name_lr}_seed-{name_seed}_date-{d4}"

    
    
    if _practice:
        model_name = 'practice_run'
        model_folder = f"{inference_path}{model_name}"
    else:
        print('Not using practice')
        # Check if the folder exists and make a new one if it does
        base_model_folder = f"{inference_path}{model_name}"
        model_folder = base_model_folder
        counter = 1
        while os.path.exists(model_folder):
            model_folder = f"{base_model_folder}_{counter}"
            model_name = os.path.basename(model_folder)  # update model_name as well
            counter += 1
        print(f"Using model folder: {model_folder}")
     # make files
    
    img_folder =  f"{model_folder}/training_imgs"
    model_folder_Exist = os.path.exists(model_folder)
    if not model_folder_Exist:
       # Create a new directory because it does not exist
       os.makedirs(model_folder)

    img_folder_Exist = os.path.exists(img_folder)
    if not img_folder_Exist:
       # Create a new directory because it does not exist
       os.makedirs(img_folder)
    text_path = f"{model_folder}/{model_name}_updates.txt"
    text_path_Exist = os.path.exists(text_path)

    if not text_path_Exist:
        f = open(f"{inference_path}/{model_name}/{model_name}_updates.txt", "x")
        f.close()
    
    wandb.init(
        project="BoundaryCondition-Prediction",
        name=model_name,
        config=parameters
    )
    
    # Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(inference_path,model_name,f"using device {device}, starting at" + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
    #write_to_file("loading data")
    write_to_file(inference_path,model_name,"loading data")
    write_to_file(inference_path,model_name,"setting up data")

    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
    test_load_data = copy.deepcopy(parameters["train_load_data"])
    test_load_data.update(parameters["test_load_data"])
    print(train_load_data)
    print(test_load_data)
    
    
    if "size" in (parameters["train_load_data"].keys()):
        print('Using square domain')
        data = LoadSquareSatelliteData(**train_load_data)
        test_data = LoadSquareSatelliteData(**test_load_data)
        
    else:    
        print('Using fixed domain')
        data = LoadDomainSatelliteData(**train_load_data)
        test_data = LoadDomainSatelliteData(**test_load_data)
        
        
    train_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # TODO: Nawid- get the train year and the test year from the trainload data 
    train_year = parse_years(train_load_data['year'])
    
    test_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    test_year = parse_years(test_load_data['year'])
    baseline_list, north_list, south_list, east_list, west_list = baseline_mol_updated(data,train_months, train_year)
    test_baseline_list, test_north_list, test_south_list, test_east_list, test_west_list = baseline_mol_updated(test_data,test_months, test_year)
    outputs = np.stack((north_list,south_list, east_list,west_list),axis=1)
    test_outputs = np.stack((test_north_list,test_south_list, test_east_list,test_west_list),axis=1)

    if parameters['normalization'] =='separate':
        outputs_mean_values, outputs_std_values = np.mean(outputs,axis=0), np.std(outputs,axis=0)
        baseline_mean_values, baseline_std_values = np.mean(baseline_list,axis=0), np.std(baseline_list,axis=0)
        outputs =  (outputs-outputs_mean_values)/outputs_std_values        
        baseline_list =  (baseline_list-baseline_mean_values)/baseline_std_values

        test_outputs = (test_outputs-outputs_mean_values)/outputs_std_values
        test_baseline_list = (test_baseline_list-baseline_mean_values)/baseline_std_values

    elif parameters['normalization'] =='all':
        outputs_mean_values, outputs_std_values = np.mean(outputs), np.std(outputs)
        baseline_mean_values, baseline_std_values = np.mean(baseline_list), np.std(baseline_list)
        outputs =  (outputs-outputs_mean_values)/outputs_std_values
        baseline_list =  (baseline_list-baseline_mean_values)/baseline_std_values

        test_outputs = (test_outputs-outputs_mean_values)/outputs_std_values
        test_baseline_list = (test_baseline_list-baseline_mean_values)/baseline_std_values
    
    grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    input_variables = parameters["variables"]

    inputs, names = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)
    
    test_inputs = get_square_satellite_inputs(test_data, **input_variables, return_asarray=True)
    
    use_baselines = parameters['use_baselines']
    if use_baselines:
        aux_index= 4
    else:
        aux_index = 0

    train_batch_size = 5
    test_batch_size=5
    print(train_load_data)
    print(test_load_data)
    write_to_file(inference_path,model_name,"Before loading data")
    #grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

    train_dataset = BoundaryDataset(inputs,baseline_list,outputs,use_baselines=use_baselines,input_names=names, **parameters["dataloader_parameters"])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    deterministic_train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    test_dataset = BoundaryDataset(test_inputs,test_baseline_list,test_outputs,use_baselines= use_baselines,input_names=names,test_mode=train_dataset.transform_parameters, **parameters["dataloader_parameters"])
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    #aux_dim = len(input_variables["static_variables"]) 
    feature_dim=np.shape(inputs)[-1]
    
    num_classes = parameters['num_classes']
    # Should probably update the name!!
    num_lat, num_lon = len(data.met.lat.values), len(data.met.lon.values)

    if parameters['network_decoder'] =='conv':
        print('Using conv network')
        model = GraphSatelliteForecasterConvClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes,input_height = num_lat, input_width=num_lon, **parameters["model_parameters"])
    else:
        print('Using normal network')
        model = GraphSatelliteForecasterClassifier(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_index,num_classes = num_classes, **parameters["model_parameters"])
    criterion = eval(parameters["loss_functions"]["criterion"])
    criterion_test = eval(parameters["loss_functions"]["criterion_test"])
    lr = parameters["learning_rate"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    normalization_vals = {"outputs_mean":outputs_mean_values,"outputs_std": outputs_std_values}
    losses = {"train":[], "test":[],"individual_summed_test_MAE":[]}
    #losses = {"train":[], "test":[],"individual_mol_test_MAE":[],"individual_summed_test_MAE":[]}
    print("saving grids etc")

    # save transform parameters, grid and training settings
    with open(f"{inference_path}{model_name}/grid_{model_name}.pickle", 'wb') as handle:
        pickle.dump(grid, handle)

    with open(f"{inference_path}{model_name}/transform_parameters_{model_name}.pickle", 'wb') as handle:
        pickle.dump(train_dataset.transform_parameters, handle)

    with open(f"{inference_path}{model_name}/training_settings_{model_name}.json", 'w') as handle:
        json.dump(parameters, handle)
    write_to_file(inference_path,model_name,"Before training model")
    epoch_so_far = 0
    if torch.cuda.is_available():
        model.cuda()
    
    wandb.watch(model, log="all", log_freq=10)  # 👈 Track gradients and weights


    # Nawid- true outputs of the test data (not using the train valeus as the training data is shuffled)
    denormalized_test_truths = (test_outputs*outputs_std_values) + outputs_mean_values
    denormalized_test_truths_summed = np.sum(denormalized_test_truths, axis=1)


    #best_test_loss = float("inf")
    #best_epoch = -1
    



    num_epochs = parameters["epochs"]
    n = 2  # Number of times to reload new data
    interval = num_epochs // n
    offset = 1
    # Nawid - need to use it earlier to make sure if runs correctly I believe before i reinitialise the data
    train_out = np.zeros((train_dataset.inputs.size()[0], num_classes))
    for epoch in range(num_epochs):
        epoch=epoch+epoch_so_far
        # Skip reloading on first interval (i.e., epoch 0)
        if epoch != 0 and epoch % interval == 0:
            print(f"🔄 Reloading training data at epoch {epoch} with offset {offset}")
            # Nawid - removing information to make it easier to load

            del inputs
            del data
            del train_dataset
            del train_loader
            train_dataset, train_loader = additional_data(
            parameters,
            offset,
            train_months,
            train_year,
            outputs_mean_values,
            outputs_std_values,
            input_variables,
            use_baselines,
            names,
            train_batch_size
        )
        offset += 1  # Move to next chunk next time
        
        running_loss = 0.0
        #individual_mol_train_errors = np.zeros(output_num)
        
        print(f"Start Epoch: {epoch}")
        start = time.time()
        #import ipdb; ipdb.set_trace()
        # Nawid - Looking at saving the outputs of the data

        
        for i_train, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            ins, labels = batch[0].to(device), batch[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model_outputs = model(ins)
            loss = criterion(model_outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #train_out[i_train*test_batch_size:(i_test+1)*test_batch_size,:] = test_model_outputs.detach().cpu().numpy()
            #individual_mol_train_errors += np.mean(np.abs(((labels-model_outputs)*outputs_std_values).detach().cpu().numpy()),axis=0)

            end = time.time()
            
            del model_outputs 
            if i_train % 10==0:
                print(f"[{epoch + 1}, {i_train + 1:5d}] loss: {running_loss / (i_train + 1):.3f} Time: {end - start} sec")
        
                
        test_error = 0.0
        test_out = np.zeros((test_dataset.inputs.size()[0], num_classes))
        # Nawid- 
        
        for i_test, batch in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            ins, labels = batch[0].to(device), batch[1].to(device)
            test_model_outputs = model(ins)

            test_error += criterion_test(test_model_outputs, labels).item()
            test_out[i_test*test_batch_size:(i_test+1)*test_batch_size,:] = test_model_outputs.detach().cpu().numpy()
            
        test_loss = test_error / (i_test+1)               
        losses["train"].append(running_loss/(i_train+1))
        losses["test"].append(test_error/(i_test+1))

        # Nawid - test predictions data
        denormalized_test_predictions = (test_out*outputs_std_values) + outputs_mean_values
        denormalized_test_predictions_summed = np.sum(denormalized_test_predictions,axis=1)
        
        test_mae = np.mean(np.abs(denormalized_test_truths_summed - denormalized_test_predictions_summed))
        
        losses["individual_summed_test_MAE"].append(test_mae)

         # 👇 Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / (i_train + 1),
            "test_loss": test_loss,
            "summed_test_MAE": test_mae
        })
        write_to_file(inference_path, model_name,f"{epoch + 1}, loss: {running_loss/(i_train+1)}, test loss: {test_error/(i_test+1)}, denormzalised summed test loss:{test_mae}")
        '''
        # Save best model and predictions
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

            model_save_path = f"{inference_path}{model_name}/{model_name}_best.pt"
            prediction_save_path = f"{inference_path}{model_name}/{model_name}_best_predictions.npz"

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                "normalization": normalization_vals,
                'learning_rate': lr,
            }, model_save_path)

            # Save predictions and truths
            np.savez(
                prediction_save_path,
                predictions=test_out,
                truths=test_outputs
            )

            write_to_file(
                inference_path, model_name,
                f"Best model saved at epoch {epoch+1} with Test Loss: {best_test_loss:.6f}"
            )
        '''

        ## save checkpoint every 50 epochs
        if epoch % 50 ==0:
            checkpoint_path = f"{inference_path}{model_name}/{model_name}_{epoch}.pt"
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,
                        "normalization":normalization_vals, 
                        'learning_rate':lr,
                        }, checkpoint_path)
            wandb.save(checkpoint_path)  # 👈 Save to W&B
            
    final_model_path = f"{inference_path}{model_name}/{model_name}_{epoch}.pt"
    # Nawid - Save for the final epoch
    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,
                        "normalization":normalization_vals, 
                        'learning_rate':lr,
                        }, final_model_path)
    wandb.save(final_model_path)

    
        
    # Nawid - Used to save the training outputs for the model at the end
    for i_det, batch in enumerate(deterministic_train_loader):
        # get the inputs; data is a list of [inputs, labels]
        ins, labels = batch[0].to(device), batch[1].to(device)
        det_outputs = model(ins)

        train_out[i_det*train_batch_size:(i_det+1)*train_batch_size,:] = det_outputs.detach().cpu().numpy()
            

    print("Finished Training")
    data_savename = f"{inference_path}{model_name}/{model_name}_final_epoch_data.pkl"
    data_dict = {'test_dataset_predictions':test_out,'test_dataset_truths':test_outputs, 'train_dataset_predictions':train_out,'train_dataset_truths':outputs}
    with open(data_savename, 'wb') as f:
        pickle.dump(data_dict, f)

    wandb.save(data_savename)
    wandb.finish()  # 👈 End wandb session





def additional_data(parameters,offset,train_months, train_year,outputs_mean_values, outputs_std_values, input_variables,use_baselines,names,train_batch_size):
    train_load_data = copy.deepcopy(parameters["train_load_data"])
    # load train parameters and upload with any changes to test data
    if "size" in (parameters["train_load_data"].keys()):
        print('Using square domain')
        data = LoadSquareSatelliteData(**train_load_data,freq_offset=offset)
        
    else:    
        print('Using fixed domain')
        data = LoadDomainSatelliteData(**train_load_data,freq_offset=offset)

    baseline_list, north_list, south_list, east_list, west_list = baseline_mol_updated(data,train_months, train_year)
    outputs = np.stack((north_list,south_list, east_list,west_list),axis=1)
    outputs =  (outputs-outputs_mean_values)/outputs_std_values
    inputs, _ = get_square_satellite_inputs(data, **input_variables, return_variable_names=True, return_asarray=True)

    train_dataset = BoundaryDataset(inputs,baseline_list,outputs,use_baselines=use_baselines,input_names=names, **parameters["dataloader_parameters"])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    return train_dataset,train_loader

def baseline_mol_updated(desired_data,months,years):
    '''
    Function used to get the value for the input and output
    '''
    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]                                                                      
    # Load the CSV file
    #df = pd.read_csv('Analysis/CH4_Semihemispheric_modelled_mole_fractions.csv')
    df = pd.read_csv('/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv')
    baseline_list = np.zeros((total_data_points,4))

    '''
    # Specify the year and month you're interested in
    # Filter the data for the specific year and month
    filtered_data = df[(df['Year'] == specific_year) & (df['Month'] == specific_month)]

    # Extract the 4th, 5th, 6th, and 7th columns
    selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values

    # Display the results
    print(selected_columns)
    '''

    # Iterate through the different numbers 
    #months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    #year = 2016
    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)
    specific_years = np.array([np.datetime64(date, 'Y').astype(int) + 1970 for date in datetime_array])
    specific_months = np.array([np.datetime64(date, 'M').astype(int) % 12 + 1 for date in datetime_array])

    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]
    north_list = np.zeros(total_data_points) # Creates a list of size N with None values
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)
    for year in years:
        for month in months:
            # Cams field for a particular month
            if year > 2017:
                # Change made due to the naming of the data
                cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc")
            else:
                cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc")    
            # Extract year and month
            #import ipdb; ipdb.set_trace()
            

            # Desired year and month
            #desired_year = 2016
            desired_year = year
            desired_month = month
            print('desired month',desired_month)
            
            '''
            coarse_cams  = cams.coarsen(lon=desired_data.coarsening_factor, lat=desired_data.coarsening_factor, boundary="pad").mean()
            '''
            # Find the index of the first occurrence
            indices = np.where((specific_years == desired_year) & (specific_months == int(desired_month)))[0]
            print(indices)
            if len(indices)>0:
                # Get the inputs
                filtered_data = df[(df['Year'] == desired_year) & (df['Month'] == int(desired_month))]
                # Extract the 4th, 5th, 6th, and 7th columns
                selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values
                baseline_list[indices] = selected_columns/1000 # Convert from parts per trillion to parts per million

                print(indices[0])
                # Multply the first value with all the other values of the array
                print(cams.vmr_n.shape)
                # CAMS field should be stationary over the period of a month
                #import ipdb; ipdb.set_trace()

                north_mol = np.sum(cams.vmr_n * desired_data.locs.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(cams.vmr_s * desired_data.locs.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(cams.vmr_e * desired_data.locs.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(cams.vmr_w * desired_data.locs.particle_locations_w[:,:,indices], axis=(0,1))
                '''
                north_mol = np.sum(coarse_cams.vmr_n * desired_data.locs.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(coarse_cams.vmr_s * desired_data.locs.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(coarse_cams.vmr_e * desired_data.locs.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(coarse_cams.vmr_w * desired_data.locs.particle_locations_w[:,:,indices], axis=(0,1))
                '''
                '''
                north_mol = np.sum(cams.vmr_n * desired_data.fp_data_full.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(cams.vmr_s * desired_data.fp_data_full.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(cams.vmr_e * desired_data.fp_data_full.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(cams.vmr_w * desired_data.fp_data_full.particle_locations_w[:,:,indices], axis=(0,1))
                '''
                #import ipdb; ipdb.set_trace()
                north_list[indices] = north_mol
                south_list[indices] = south_mol
                east_list[indices] = east_mol
                west_list[indices] = west_mol


        #print(baseline_list)
        #cams = xr.open_dataset("/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_201611_CAMS-inversion.nc")
        # Making the assumption that the values in the month are not different, get the first value
        # Multiple the different values
    
    return baseline_list, north_list, south_list, east_list, west_list



def parse_years(year_str):
    # Case 1: Range like '201[4-5]'
    match = re.fullmatch(r'201\[(\d)-(\d)\]', year_str)
    if match:
        start, end = map(int, match.groups())
        return [2010 + i for i in range(start, end + 1)]
    
    # Case 2: Exact year like '2014' or '2015'
    if re.fullmatch(r'20\d{2}', year_str):
        return [int(year_str)]
    
    # Fallback
    raise ValueError(f"Invalid year format: {year_str}")