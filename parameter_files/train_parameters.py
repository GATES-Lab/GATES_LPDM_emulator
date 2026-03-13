import copy
from itertools import product
from general_train import load_file

file_name = "parameter_file_base.json"
file_path = "/graphnet_LPDM_emulator/parameter_files/"

hparams = load_file(file_name, file_path)


sim_hparams = copy.deepcopy(hparams)
sim_hparams['normalization'] = 'all'
# sim_hparams["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_False_lr-5e-05_seed-31_date-Jan-18-2025"
sim_hparams["use_baselines"] = True
# sim_hparams['train_load_data']['year'] = "201[3-4]"
# sim_hparams["train_load_data"]["domain_to_cut"]= {"lat":[-29.5,-6], "lon":[-60,-10]} # Size 100 by 100

# sim_hparams["train_load_data"]["domain_to_cut"]={"lat":[-39.7,4.7]} # Size 190 by 190
sim_hparams['learning_rate'] = 5e-5
sim_hparams["train_load_data"]['size'] = 100
sim_hparams['network_decoder'] = 'conv'

# Added this for second set of simulations
sim_hparams['train_load_data']['year'] = "201[4-6]"
sim_hparams['test_load_data']['year'] = "2017"
# sim_hparams["train_load_data"]["coarsening_factor"] = 1
# sim_hparams['dataloader_parameters']['output_transforms'] = ['logv3']


# Second experiment - data which is cut up
sim_hparams_2 = copy.deepcopy(hparams)
sim_hparams_2['normalization'] = 'all'
sim_hparams_2['learning_rate'] = 5e-6
sim_hparams_2['train_load_data']['year'] = "201[4-6]"
sim_hparams_2['test_load_data']['year'] = "2017"
# sim_hparams_2['train_load_data']['year'] = "2014"
# sim_hparams_2["train_load_data"]["domain_to_cut"]= {"lat":[-29.5,-6], "lon":[-60,-10]} # Size 100 by 100
# sim_hparams_2["train_load_data"]["coarsening_factor"] = 1
# sim_hparams_2['variables']["met_variables"] = {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]}
sim_hparams_2["train_load_data"]['size'] = 100
sim_hparams_2['network_decoder'] = 'conv'

# sim_hparams_2['train_load_data']['year'] = "201[4-5]" # Added this for second set of simulations

# Experiment 3, full domain but high learning rate

sim_hparams_3 = copy.deepcopy(hparams)
sim_hparams_3['normalization'] = 'all'
sim_hparams_3['learning_rate'] = 5e-5
# sim_hparams_3['train_load_data']['year'] = "2014"
# sim_hparams_3["train_load_data"]["domain_to_cut"]= {"lat":[-29.5,-6], "lon":[-60,-10]} # Size 100 by 100
# sim_hparams_3["train_load_data"]["coarsening_factor"] = 1
sim_hparams_3['variables']["met_variables"] = {"x_wind": [3, 9, 15, 21, 30, 42, 51], "wind_speed": [3, 30, 51], "wind_angle": [3, 30, 51], "y_wind": [3, 9, 15, 21, 30, 42, 51], "upward_air_velocity": [
    3, 9, 15, 21, 30, 42, 51], "air_temperature": [3, 9, 15, 21, 30, 42, 51], "air_pressure": [3, 9, 15, 21, 30, 42, 51], "atmosphere_boundary_layer_thickness": [], "surface_air_pressure": []}
sim_hparams_3['variables']["static_variables"] = ["sin_lat_coords", "sin_lon_coords",
                                                  "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"]
sim_hparams_3['network_decoder'] = 'conv'

# Experiment 3, full domain but high learning rate
'''
sim_hparams_4 = copy.deepcopy(hparams)
sim_hparams_4['normalization'] = 'all'
#sim_hparams["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_False_lr-5e-05_seed-31_date-Jan-18-2025"
sim_hparams_4['learning_rate'] = 5e-5
sim_hparams_4['train_load_data']['year'] = "2014"
#sim_hparams_4["train_load_data"]["domain_to_cut"]= {"lat":[-29.5,-6], "lon":[-60,-10]} # Size 100 by 100
#sim_hparams_4["train_load_data"]["coarsening_factor"] = 1
sim_hparams_4['variables']["met_variables"] = {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]}
sim_hparams_4['variables']["static_variables"] = ["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"]
'''

# Experiment 3, full domain but high learning rate

sim_hparams_4 = copy.deepcopy(hparams)
sim_hparams_4['normalization'] = 'all'
sim_hparams_4['learning_rate'] = 5e-6
sim_hparams_4['train_load_data']['year'] = "201[4-6]"
sim_hparams_4['test_load_data']['year'] = "2017"
# sim_hparams_5['variables']["met_variables"] = {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]}
# sim_hparams_5['variables']["static_variables"] = ["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"]
sim_hparams_4['network_decoder'] = 'conv'
sim_hparams_4["train_load_data"]['size'] = 50
sim_hparams_4['seed'] = 42
sim_hparams_4['output_format'] = 'corrected'

sim_hparams_5 = copy.deepcopy(hparams)
sim_hparams_5['normalization'] = 'all'
sim_hparams_5['learning_rate'] = 5e-6
sim_hparams_5['train_load_data']['year'] = "201[4-6]"
sim_hparams_5['test_load_data']['year'] = "2017"
# sim_hparams_5['variables']["met_variables"] = {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]}
# sim_hparams_5['variables']["static_variables"] = ["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"]
sim_hparams_5['network_decoder'] = 'conv'
sim_hparams_5["train_load_data"]['size'] = 50
sim_hparams_5['seed'] = 31
sim_hparams_5['output_format'] = 'corrected'

# sim_hparams_5["train_load_data"]["domain_to_cut"]= {"lat":[-39.7,4.7]} # Lat 190 lon 190
# sim_hparams_5["train_load_data"]['coarsening_factor'] = 2


# Experiment 3, full domain but high learning rate
sim_hparams_6 = copy.deepcopy(hparams)
sim_hparams_6['normalization'] = 'all'
sim_hparams_6['learning_rate'] = 5e-6
sim_hparams_6['train_load_data']['year'] = "201[4-6]"
sim_hparams_6['test_load_data']['year'] = "2017"
# sim_hparams_6['variables']["met_variables"] = {"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]}
# sim_hparams_6['variables']["static_variables"] = ["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"]
sim_hparams_6['network_decoder'] = 'conv'
sim_hparams_6["train_load_data"]['size'] = 50
sim_hparams_6['seed'] = 28
sim_hparams_6['output_format'] = 'corrected'
# sim_hparams_2["train_load_data"]["coarsening_factor"] = 1
# sim_hparams_2["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_True_lr-5e-05_seed-31_date-Jan-17-2025"


hparams_list = [sim_hparams_4]

# hparams_list = [sim_hparams_4,sim_hparams_5, sim_hparams_6]


# hparams_list = [sim_hparams, sim_hparams_2,sim_hparams_3,sim_hparams_4, sim_hparams_5,sim_hparams_6]
