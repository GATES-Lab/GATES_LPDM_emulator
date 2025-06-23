import copy
from itertools import product



hparams = {
    "model_name" : "test_run",
    "train_load_data" : {
        "year":"2015",
        "freq":3,
        "region":"BRAZIL",
        "coarsening_factor":1,
        "verbose":True,
        "met_args":{"met_levels":[3, 15,21]},
    },

    "test_load_data" : {
        "year":"2016",
        "freq":50
    },

    "variables" : {
        "met_variables":{"x_wind":[3,15], "y_wind":[3,15], "upward_air_velocity":[3,15], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]},
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
sim_hparams["train_load_data"]["coarsening_factor"] = 4
#sim_hparams['dataloader_parameters']['output_transforms'] = ['logv3']

sim_hparams['seed'] = 31

# Second experiment
sim_hparams_2 = copy.deepcopy(sim_hparams)
sim_hparams_2['normalization'] = 'all'
sim_hparams_2["train_load_data"]["coarsening_factor"] = 2
#sim_hparams_2["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_True_lr-5e-05_seed-31_date-Jan-17-2025"

sim_hparams_3 = copy.deepcopy(sim_hparams)
sim_hparams_2["train_load_data"]["coarsening_factor"] = 1

hparams_list = [sim_hparams,sim_hparams_2,sim_hparams_3]