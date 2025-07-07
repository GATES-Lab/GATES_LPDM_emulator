import copy
from itertools import product



hparams = {
    "model_name" : "test_run",
    "train_load_data" : {
        "year":"2015",
        "freq":5,
        "region":"BRAZIL",
        "coarsening_factor":1,
        "verbose":True,
    },

    "test_load_data" : {
        "year":"2016",
        "freq":10
    },

    "variables" : {
        "met_variables":{"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]},
        "static_variables":["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"],
        "time_deltas":[6,12]
    },

    "dataloader_parameters":{
        "input_transforms":["clever_transform_3"],
    },

    "model_parameters":{
        "num_blocks":4, 
        "node_dim":64, 
        "edge_dim":64, 
        "hidden_layers_processor_node":2, 
        "hidden_layers_processor_edge":2,  
        "hidden_layers_decoder":1, 
        "hidden_dim_processor_node":16, 
        "hidden_dim_processor_edge":16, 
        "hidden_dim_decoder":16, 
        "resolution":4, 
        "output_dim":8, 
        "residuals":False, 
        "attention":False
    },
    'normalization':'all',
    "learning_rate":5e-7,
    "seed":42,
    "epochs":100,
    "num_classes":4,
    "use_baselines":False,

    "loss_functions" : {
        "criterion": "torch.nn.MSELoss()",
        "criterion_test": "torch.nn.MSELoss()"
    }    
}


sim_hparams = copy.deepcopy(hparams)
sim_hparams['normalization'] = 'all'
#sim_hparams["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_False_lr-5e-05_seed-31_date-Jan-18-2025"
sim_hparams["use_baselines"] = False
sim_hparams["train_load_data"]["domain_to_cut"]= {"lat":[-24.3,20]}
sim_hparams["train_load_data"]["coarsening_factor"] = 1
#sim_hparams['dataloader_parameters']['output_transforms'] = ['logv3']

sim_hparams['seed'] = 31

# Second experiment
sim_hparams_2 = copy.deepcopy(hparams)
sim_hparams_2['normalization'] = 'all'
sim_hparams_2["train_load_data"]["coarsening_factor"] = 1
#sim_hparams_2["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_True_lr-5e-05_seed-31_date-Jan-17-2025"



hparams_list = [sim_hparams,sim_hparams_2]