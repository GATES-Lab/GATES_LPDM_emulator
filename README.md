# graphnet_LPDM_emulator
WIP!
This repo implements the model described at [add link!] 


## To Do - restructuring and updating
- [ ] Update data loading functions:
  - [x] Load footprints function
  - [ ] Data loader - update, comment, update documentation
  - [x] Add capability to cut and interpolate met directly from file, without needing to cut
  - [x] Input extracting - update, comment, udpate documentation
  - [ ] Add plotting function to data object?
- [ ] Update training/testing scripts to work with new data loading functions
- [ ] Small improvements to model code
- [ ] Improvements to evaluation code

## Environment - check this section! 

See environment_short.yml
This file does not contain torch and related packages - this is because you will need to install separately a CUDA-enabled version or not depending on where you are running the code. BluePebble has pytorch+cuda pre-installed, which you can load when you submit jobs to the queue (see the launch_train.sh file). To run notebooks or files on the login node, you will need torch and associated packages installed in a different environment, which I manually import when running notebooks with the following line. Alternatively you could have two parallel envs (graphnet to run on cluster, and graphnet+torch to run on login) but that might get more confusing if you need to install packages! 
```
sys.path.insert(0, "/path/to/environment_with_torch/env_name/lib/python3.8/site-packages/")
import torch
```

## Loading data
#### `LoadBaseSatelliteData` loads data from the directories (provided or default). It does not crop or interpolate
``` 
original_data = LoadBaseSatelliteData(year=2016, region="SAHARA", freq=40,  topog="default", verbose=True, load_everything=True)
```
original_data has attributes original_data.fp_data_full , original_data.met_file, original_data.topog_file and original_data.landcover_file

#### `LoadSquareSatelliteData` loads data, interpolating to the time of the footprints and cropping to a square of size size x size around the footprint's measurement point
``` 
cropped_data = LoadBaseSatelliteData(year=2016, region="SAHARA", freq=40,  size=200, topog="default", verbose=True, load_everything=True)
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
### Preparing dataset - this has changed, need to update!
The `FootprintsDataset` object sets up the inputs and outputs to be loaded to the DataLoader, and makes any needed transformations.
The transformations I'm doing currently are:
- outputs:
  - boxcox - the footprint data is very sparse and exponential (most of the domain is full of zeros, there are a few high values near the measurement point, and they decay very quickly as you move further out). To bring all of the data to a similar range, I standardise then apply a boxcox transform to each of the locations independently using sklearn's `PowerTransformer(method='box-cox', standardize=True)`. Applying this to each node separately means that the range of values to be transformed is within the same order of magnitude, and the data transformed is all within the same range (0-2 with a couple outliers). This transformation is definitely helpful but could be improved! The sparsity problem is still there. `train_dataset.fp` contains the transformed data, and `train_dataset.fp_untransformed` the original footprint data. The boxcox transformer is stored at `train_dataset.boxcox`
- inputs
  - `clever_transform_2` (not that clever!) -  applies a sklearn `preprocessing.StandardScaler()` to each variable and level, across all time jumps (eg all the x_wind data at level 3 is scaled together, so is at level 9 etc). The transformers are stored in a dictionary of format "{"variable_name":{level_1:transformer, level_2:transformer...},...}" stored at train_dataset.transformers. `clever_transform` does the same but across all levels rather than separately. The feature names need to be passed to `input_names` for this transform to work

The test dataset can be transformed using the trained transformers from the train dataset by passing a test_mode dictionary as shown below

```
train_dataset = FootprintsDataset(inputs=inputs, fp=np.copy(data.fp_data), transform_output="boxcox", feature_dim=np.shape(inputs)[-1]-len(others)-topog, aux_dim=len(others)+topog, clever_transform_2=True, input_names=names)
test_dataset = FootprintsDataset(inputs=test_inputs, fp=np.copy(test_data.fp_data), transform_output="boxcox", feature_dim=np.shape(inputs)[-1]-len(others)-topog, aux_dim=len(others)+topog, clever_transform_2=True, input_names=names, test_mode={"boxcox":train_dataset.boxcox, "clever_transformers":train_dataset.transformers})

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
train_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
```

## Other data functions
- You can align two LoadSatelliteData objects to have the same timestamps using `align_datasets(dataset1, dataset2)`. This is useful if you want to compare two sets of data, predictions etc but some datapoints have been removed in either dataset during loading, maybe due to freq, NaNs etc. 
  

## Model
The GNN paradigm are Graph Networks, described by [Deepmind, 2018](https://arxiv.org/pdf/1806.01261.pdf). 
![Deepmind paper - graph updates example](/readme_imgs/deepmind_updates.PNG?raw=true)

### Model literature/code
The model is based on the one described by [Deepmind,2022](https://arxiv.org/pdf/2212.12794.pdf) and particularly [Keisler, 2022](https://arxiv.org/pdf/2202.07575.pdf) and the code developed from the code [in the corresponding repo](https://github.com/openclimatefix/graph_weather). I have made some changes I will detail here at some point

### Model architecture
Here is an architecture diagram that could probably be a bit clearer
![Architecture diagram](/readme_imgs/diagram.jpg?raw=true)

### Constructing a model
Create a model with the following:
```
model = GraphSatelliteForecaster(grid, whole_world=False, feature_dim=np.shape(inputs)[-1]-len(others)-topog, aux_dim=len(others)+topog,num_blocks=4, node_dim=64, edge_dim=64, hidden_layers_processor_node=3, hidden_layers_processor_edge=2,  hidden_layers_decoder=1, hidden_dim_processor_node=16, hidden_dim_processor_edge=16, hidden_dim_decoder=16, "resolution=4, output_dim=1)
```
Parameters:
- grid: list of lat-lon tuples for each of the nodes, outputted with get_all_inputs_graphnet_satellite_v4
- whole_world: wether to create a mesh graph that spans the whole globe, or only mesh nodes above the grid nodes passed in grid. Always False at this stage
- feature_dim and aux_dim: length of meteorological and non-met time inputs, respectively. Legacy from original code, right now they make no difference as long as feature_dim+aux_dim=total number of dims. feature_dim+aux_dim = length of the green node features in the Encoder in the diagram above
- resolution: resolution of the mesh grid, as determined by the [h3 library](https://h3geo.org/docs/core-library/restable). The lower the resolution, the bigger the hexagons are. Resolution of 4 (used throughout the models) covers between one and three grid nodes, resolution of 3 covers around 10-15 grid nodes, resolution of 5 covers none or one grid nodes. Resolution of 5 and below do not work (as the mesh needs to cover the domain completely and this resolution is too fine-grained to).
- NN parameters - please refer to diagram above which I need to label at some point:
  -  `num_blocks` - number of processor blocks (pink cubes). These update the mesh edges and nodes sequentially
  -  `node_dim` - size of mesh node feature array (ie length of yellow mesh node feature in diagram above)
  - `edge_dim` - size of mesh edge feature array (ie length of dashed yellow mesh edge feature above)
  - `hidden_layers_processor_node` - number of hidden layers in the Node Encoder and Node Updater, each of size `hidden_dim_processor_node` (blue blocks in Node Encoder and Node Updater)
  - `hidden_layers_processor_edge` - number of hidden layers in the Edge Encoder and Edge Updater, each of size `hidden_dim_processor_edge` (blue blocks in Edge Encoder and Edge Updater)
  - `hidden_layers_decoder` - number of hidden layers in the Decoder, each of size `hidden_dim_decoder` (blue block in Node Decoder)
  - `output_dim` - dimension of Decoder Output (flat green square in Decoder above)


 ## Predicting
 To predict using a trained model, you will need 
 - grid of lat/lon values (used during training for consistency)
 - trained input and output transformers
 - input, dataset and model parameters

All three are pickled during training and can be loaded as shown in the `examples.ipynb`. They can then be used to create the test_inputs, test_dataset and model. Use the model to predict, and the dataset to invert the transform to the original space 
```
preds = model(test_dataset.inputs).detach().numpy()
transformed_preds = test_dataset.inverse_transform(np.squeeze(preds))
```
### Predicting for a different domain size than trained on
To do!
