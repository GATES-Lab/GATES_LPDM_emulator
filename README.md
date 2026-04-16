# GATES_LPDM_emulator v0.2.0

This repo implements a new, more user-friendly version of the model described at [Enabling Fast Greenhouse Gas Emissions Inference from Satellites with GATES: a Graph-Neural-Network Atmospheric Transport Emulation System (GMD, 2026)](https://gmd.copernicus.org/articles/19/1893/2026/gmd-19-1893-2026.html)  


## New file structure (currently in construction)
```
gates_LPDM_emulator/
├── gates/
│   └── data/
│   |   ├── load_data.py         # LoadSquareSatelliteData, LoadBaseSatelliteData
│   |   ├── datasets.py          # InputsDataset, FootprintDataset, 
│   |   └── HOW_TO_DATA.md       # Info on how to use the data files
|   └── evaluation/
│       ├── metrics.py         
│
├── notebooks
│       ├── data_tutorial.ipynb  

```





## Setting up

### Enviroment

The repo requires `python=3.12`, `xarray=2025.1`, and `pytorch=2.3`. Install the enviroment from [env_gates_pytorch.yml](.env_gates_pytorch.yml). 


You might need to make modifications to the enviroment if you want a CPU-only installation, or for different cuda versions.  If you are installing the enviroment on the login node of a cluster with GPUs, use the following to force a CUDA-install.

```
CONDA_OVERRIDE_CUDA=12.1 conda env create -f env_gates_pytorch.yml

```

### Installing the repo
To install an editable version of this package in your repository, run the following from the root of the repo

```
pip install --no-build-isolation --no-deps -e .
```



# -----------------To check!-------------------

## Loading data
#### `LoadBaseSatelliteData` loads data from the directories (provided or default). It does not crop or interpolate
``` 
original_data = LoadBaseSatelliteData(year=2016, region="SAHARA", freq=40,  verbose=True, load_everything=True)
```
original_data has attributes original_data.fp_data_full , original_data.met_file, original_data.topog_file and original_data.landcover_file

#### `LoadSquareSatelliteData` loads data, interpolating to the time of the footprints and cropping to a square of size size x size around the footprint's measurement point
``` 
cropped_data = LoadSquareSatelliteData(year=2016, region="SAHARA", freq=40,  size=200, verbose=True, load_everything=True)
```
cropped_data has attributes cropped_data.fp_data (np array), cropped_data.met (Dataset interpolated in time and cropped in space), and cropped_data.topog (Dataset cropped in space, with variables topog and landcover).

Notes:
- size should be even!
- `freq` reduces the time frequency of the data before loading to reduce computational expense (ie `freq=3` will only load one in every three footprints and corresponding data)
- sometimes the cropped area escapes the actual footprint domain.
  - if delete_outofdomain=True, these footprints are deleted from the dataset
  - if the footprints that escape the domain are kept, use fill_outofdomain_with to specify if the out-of-domain areas should be filled with "zeros" or "nans"



## Setting up data
### Extracting inputs with `get_square_satellite_inputs()`
Extract the inputs with shape (time, lat, lon, variable), for multiple atmospheric levels and times if passed. 
The parameter `time_deltas` allows extracting and interpolating meteorology further back in time, for t-delta Hours where t is the footprint measurement time. e.g. `time_deltas=[6]` means that the meteorological variables will be returned at t, and t-6h.

```
variables = {"x_wind":[3,15], "y_wind":[3], "surface_air_pressure":[]}
static_variables=["lat_coords", "lon_coords", "x_coords", "y_coords", "topog", "landcover"]
inputs, input_names = get_square_satellite_inputs(cropped_data, variables, static_variables=static_variables, time_deltas=[6], return_variable_names=True)

```

#### Variables
- Meteorological (time-dependent)
  - air_pressure
  - air_temperature
  - [atmosphere_boundary_layer_thickness](https://forecast.weather.gov/glossary.php?word=boundary%20layer#:~:text=Atmospheric%20Boundary%20Layer&text=For%20the%20earth%2C%20this%20layer,friction%20with%20the%20earth's%20surface.): height of the layer of the atmosphere within which the effects of friction are significant (roughly the lowest one or two kilometers of the atmosphere). 2D - no height component
  - surface_air_pressure: 2D - no height component
  - upward_air_velocity
  - x_wind
  - y_wind
  - wind_angle
  - wind_speed
- Static 
  - topog
  - landcover
  - sin_lat_coords/sin_lon_coords/cos_lat_coords/cos_lon_coords - sin and cos of coordinates. This is to encode cyclical variables when using a sphere (ie the whole world), but probably is not as useful when only using a reduced domain
  - lat_coords/lon_coords - coordinates at each node
  - distance_centre - Euclidean distance from the release point, calculated with x/y coords rather than actual distance - this is to speed up calculation, as the lat/lon frame of reference (and therefore distances) changes only slightly for each footprint
  - x_coords/y_coords - numerical indeces of each node in a x/y style, passing `centered_coords=True` returns 0,0 as the center (so negative x coordinates are west, negative y coordinates are south), otherwise 0,0 is the South-West corner and all x/y coords are positive, with the release point at int(size/2), int(size/2). For best practice and better inference across sizes pass centered_coords=True
  - binary_centre - zero for all nodes except the release point which is 1
  - 
### Preparing dataset
The `FootprintsDataset` object sets up the inputs and outputs to be loaded to the DataLoader, and makes any needed transformations.
The transformations I'm doing currently are:
- outputs:
  - logv4: Takes log of output data where non-zero, and offsets by minimum value so its above.
 `train_dataset.fp` contains the transformed data, and `train_dataset.fp_untransformed` the original footprint data
- inputs
  - `clever_transform_3` -  applies a sklearn `preprocessing.StandardScaler()` to each variable and level, across all time_deltas (eg all the x_wind data at level 3 is scaled together, so is at level 9 etc). Applies min-max transform to landcover. The transformers are stored in a dictionary of format "{"variable_name":{level_1:transformer, level_2:transformer...},...}" stored at train_dataset.transformers. The feature names need to be passed to `input_names` for this transform to work

The test dataset can be transformed using the trained transformers from the train dataset by passing a test_mode dictionary as shown below

```
train_dataset = FootprintsDataset(inputs=inputs, fp=cropped_data.fp_data, input_names=input_names, input_transforms=["clever_transform_3"], output_transforms= ["logv4"])

test_dataset = FootprintsDataset(inputs=inputs, fp=cropped_data.fp_data, input_names=input_names, input_names=names, test_mode=train_dataset.transform_parameters)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
train_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
```


## Model
Check `model_description.md` for more info on the architecture!

The model operates on two levels: lat-lon grid (of same resolution and shape for the inputs and the outputs) and an intermediate abstract layer, with nodes arranged in hexagons.
In the current setup, the model builts a grid and mesh pair using the location of a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

### Constructing a model
Create a model with the following:

```
aux_dim = len(input_variables["others"]) 
feature_dim=np.shape(inputs)[-1]-aux_dim

model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_dim, num_blocks=4, node_dim=64, edge_dim=64, hidden_layers_processor_node=3, hidden_layers_processor_edge=2,  hidden_layers_decoder=1, hidden_dim_processor_node=16, hidden_dim_processor_edge=16, hidden_dim_decoder=16, resolution=4, output_dim=1)

optimizer = optim.AdamW(model.parameters(), lr=lr)
```
Parameters:
- grid: list of lat-lon tuples for each of the nodes, outputted with get_all_inputs_graphnet_satellite_v4
- whole_world: wether to create a mesh graph that spans the whole globe, or only mesh nodes above the grid nodes passed in grid. Always False at this stage
- feature_dim and aux_dim: length of meteorological and non-met time inputs, respectively. Legacy from original code, right now they make no difference as long as feature_dim+aux_dim=total number of dims. feature_dim+aux_dim = length of the green node features in the Encoder in the diagram in model_description.md 
- resolution: resolution of the mesh grid, as determined by the [h3 library](https://h3geo.org/docs/core-library/restable). The lower the resolution, the bigger the hexagons are. Resolution of 4 (used throughout the models) covers between one and three grid nodes, resolution of 3 covers around 10-15 grid nodes, resolution of 5 covers none or one grid nodes. Resolution of 5 and below do not work (as the mesh needs to cover the domain completely and this resolution is too fine-grained to).
- NN parameters - please refer to diagram above which I need to label at some point:
  -  `num_blocks` - number of processor blocks (pink cubes). These update the mesh edges and nodes sequentially
  -  `node_dim` - size of mesh node feature array (ie length of yellow mesh node feature in diagram above)
  - `edge_dim` - size of mesh edge feature array (ie length of dashed yellow mesh edge feature above)
  - `hidden_layers_processor_node` - number of hidden layers in the Node Encoder and Node Updater, each of size `hidden_dim_processor_node` (blue blocks in Node Encoder and Node Updater)
  - `hidden_layers_processor_edge` - number of hidden layers in the Edge Encoder and Edge Updater, each of size `hidden_dim_processor_edge` (blue blocks in Edge Encoder and Edge Updater)
  - `hidden_layers_decoder` - number of hidden layers in the Decoder, each of size `hidden_dim_decoder` (blue block in Node Decoder)
  - `output_dim` - dimension of Decoder Output (flat green square in Decoder above)

### Training
Use the `general_train.py` file to train a model. Set up all your model parameters using the parameter_template_train.txt file (note that model_name should be unique!). A new folder will be created, and populated with the following:
```
.
└── model_name/
    ├── grid_model_name.pickle
    ├── training_settings_model_name.json
    ├── model_name_updates.txt
    ├── transform_parameters_model_name.pickle
    ├── training_imgs/
    │   ├── model_name_0.png
    │   ├── model_name_1.png
    │   └── ...
    ├── model_name_50.pt
    ├── model_name_100.pt
    └── ...
```

 ## Predicting: same model and size
 If you want to use a model model_name that you have trained to predict footprints, in the same domain and size, you can pass the parameter file directly to general_make_prediction.py
 
 ```python general_make_prediction.py training_settings_model_name.json --file_path /path/to/model ```

 This will create a new folder in the directory called `predictions`. Each file in `predictions` is a .nc array with
   - coords lat, lon and time
   - attributes:
     - fp - the true footprints, in the original dataspace
     - trans_fp: the true footprints, in the transformed dataspace
     - predictions: the predicted footprints, as outputted by the model in the transformed dataspace
     - trans_predictions: the predicted footprints, in the original dataspace

## Predicting: different size, different domain etc 
 To use a trained model `model_name` to predict but with a different set-up (e.g. predict at 200x200 when you trained at 50x50), you will need to create a folder new_model_name:
 - grid of lat/lon values (used during training for consistency)
 - trained input and output transformers
 - input, dataset and model parameters
```
   .
└── model_name/
    ├── grid_new_model_name.pickle # this is needed if the emulating size is different to the training size
    ├── training_settings_new_model_name.json # copy the parameter file from the reference model and add a section
```
Add the following to `training_settings_new_model_name.json`, to indicate which model should be used to emulate, and the size that it was trained on
```
    "reference_model": {
        "model_name":"model_name_to_predict_from", 
        "domain_size": 50
    }
```

## Inference and evaluation - WIP!
After inference, the footprints might need bias correcting. For the inversion, they also need to be integrated into the original domain shape. Use the `integrate_fps.py` file to apply bias correction, and save the footprints in the original footprint domain


## Inversion - TO DO! 



## See Also
- [HOW_TO_CONFIG.md](./gates/HOW_TO_CONFIG.md)
- [HOW_TO_DATA.md](./gates/data/HOW_TO_DATA.md)
- [HOW_TO_TESTS.md](./tests/HOW_TO_TESTS.md)

