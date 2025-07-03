import copy
from itertools import product
'''
hparams = {
    
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
        "output_dim":4, 
        "residuals":False
    },
    "Class_evaluation":True,
    "loss_functions" : {
        "criterion": "torch.nn.MSELoss()",
        "criterion_test": "torch.nn.MSELoss()"
    },


    "train_load_data" : {
        "year":'201401',
        "freq":3,
        "region":"BRAZIL",
        "metsize":50,
        "size":50,
        "verbose":True,
        "topog":"default",
        "met_datadir":"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_200_all_"
    },
    "train_load_data_2" : {
        "freq_offset":1,
    },

    "test_load_data" : {
        "year":'201701',
        "freq_offset":1,
        "freq":1
    },

    "variables" : {
        "met_variables":{"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[0], "surface_air_pressure":[0]},
        "others":["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "distance_centre", "x_coords", "y_coords"],
        "jumps":[6,12],
        "topog":True,
        "centered_coords":True
    },

    "dataloader_parameters":{
        "input_transforms":["clever_transform_2"],  
        
    },
    'normalization':'all',
    "learning_rate":5e-5,
    "seed":42,
    "epochs":2,
    "num_classes":4,
    "use_baselines":False,


    "notes":"The purpose of this is to see the performance when predicting the boundary mols directly",
    "model_name":"trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_False_lr-5e-05_seed-31_date-Jan-18-2025"
         
}
'''

hparams = {
    "model_name" : "test_run",
    "train_load_data" : {
        "year":"2015",
        "freq":100,
        "region":"BRAZIL",
        "verbose":True,
        "coarsening_factor":2,
    },

    "test_load_data" : {
        "year":"2016",
        "freq":300
    },

    "variables" : {
        "met_variables":{"x_wind":[3,15], "y_wind":[3,15], "upward_air_velocity":[3,15],"atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]},
        "static_variables":["domain_binary_release", "domain_distance_release"],
        #"static_variables":["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog"],
        #"met_variables":{"x_wind":[3,9,15,21,30,42,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "upward_air_velocity":[3,9,15,21,30,42,51], "air_temperature":[3,9,15,21,30,42,51], "air_pressure":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[], "surface_air_pressure":[]},
        #"static_variables":["sin_lat_coords", "sin_lon_coords", "cos_lat_coords", "cos_lon_coords", "lat_coords", "lon_coords", "x_coords", "y_coords", "topog","domain_binary_release", "domain_distance_release"],
        "time_deltas":[6],
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
    "epochs":2,
    "num_classes":4,
    "use_baselines":False,

    "loss_functions" : {
        "criterion": "torch.nn.MSELoss()",
        "criterion_test": "torch.nn.MSELoss()"
    }    
}

practice_hparams_list = []
sim_hparams = copy.deepcopy(hparams)
sim_hparams['normalization'] = 'all'
sim_hparams["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_False_lr-5e-05_seed-31_date-Jan-18-2025"
#sim_hparams['dataloader_parameters']['output_transforms'] = ['logv3']

sim_hparams['seed'] = 31

# Second experiment
sim_hparams_2 = copy.deepcopy(sim_hparams)
sim_hparams_2['normalization'] = 'all'
sim_hparams_2['train_load_data']['coarsening_factor'] = 4
sim_hparams_2["model_name"] = "trainyear-2014_trainfreq-3_baselineyears-2014-2011_normalization-all_size-50-epochs-150_baselines_True_lr-5e-05_seed-31_date-Jan-17-2025"



practice_hparams_list = [sim_hparams,sim_hparams_2]

'''
practice_hparams_list = []
variables = {'output_transforms':[['logv3']],
			 'difference':['pca'],
			 'prototype_selection':['expert'],
			 'dim':[64],
             'seed':[31],
             "num_classes":[2,3]
             }

# Generate all permutations of the values 
permutations = list(product(*variables.values()))

for index,config in enumerate(permutations):
    sim_hparams = copy.deepcopy(hparams)
    sim_hparams['dataloader_parameters']['output_transforms'] = config[0]
    sim_hparams['classifier']['difference'] = config[1]
    sim_hparams['classifier']['prototype_selection'] = config[2]
    sim_hparams['classifier']['dim'] = config[3]
    sim_hparams['seed'] = config[4]
    sim_hparams['classifier']['model_parameters']['num_classes'] = config[5]

    practice_hparams_list.append(sim_hparams)
'''

'''
for key in variables.keys():
    sim_hparams = copy.deepcopy(hparams)
    for value in variables[key]:
        if key == 'output_transform':
            sim_hparams['classifier'][key] = value
        else:
            sim_hparams[key] = value

    
        
    import ipdb; ipdb.set_trace()

h_params_2 = copy.deepcopy(hparams)
h_params_2['classifier']['prototype_selection'] = 3
h_params_2['classifier']['model_parameters']['num_classes'] = 3
hparams_list = [hparams,h_params_2]
'''