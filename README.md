# graphnet_LPDM_emulator

## To Do
- [x] write readme section on model usage
- [x] write readme section on predicting
- [x] check/update all function docstrings!!
- [x] update example file with model loading, training and predicting
- [x] add training file
- [x] add example trained model
- [ ] add predicting file
- [ ] add info on evaluation metrics
- [x] clean up data loading
- [ ] clean up dataset creation
- [ ] Add science summary and ref papers

Nomenclature (needs tidying so it's less confusing but bear with me for now):
The model has a grid (square) and a mesh (hexagonal). The nodes in the grid are grid nodes and the nodes in the mesh, mesh nodes. I use grid node, pixel and location interchangeably - they all refer to a specific coordinate with a lat/lon. The release point and the measurement point are also the same thing.

The code works fine if used right but it's not robust and needs some cleaning up (there are some inconsistencies in formats, some parameters are redundant, chunks should be split into separate functions, some bits can definitely be parallelised and/or made more efficient)

## Loading data
Use the `LoadSatelliteData` object to load data for a date period.
``` 
data = LoadSatelliteData(year=2016, region="BRAZIL", freq=2, metsize=50, size =50, topog="default", verbose=True, met_datadir="/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_")
```

- All of the necessary data is in the ACRG folder (/group/chemistry/acrg/), the paths all default to this unless specifed
- Only valid `region`s are BRAZIL, SAHARA, INDIA (ie there are default domains and some footprint and met data)
- `year` can be an int (eg 2016) or a str (eg "201[4-5]"). There is a parameter `month` to load a specific month of data (so `year=2016, month="01"`). The month parameter isn't bulletproof, I have been using `year` directly, eg `year=201601` or `year="201601"` instead
- footprints:
  - The function takes footprint data and cuts it to a square of size *size* around the measurement point. The original footprint data is conserved in  `data.fp_data_full` and the cut footprint data is stored flattened in a np array of shape (time, size*size) in `data.fp_data`
  - The latitudes and longitudes of the cut square for each footprint is stored in `data.fp_lats` and `data.fp_lons` (each of these has size (time, size))
  - Size should be even 
- Met:
  - met_datadir should contain pre-cut meteorology (ie met that has already been cut to a square domain centered around the measurement point). You can pass any pre-cut met file as long as the size is bigger than metsize though best to past the exact if it exists (eg `Met_cut_v2_50_` will be properly resized for `metsize=50` and below, but not for bigger sizes.
  - The met data is stored in `data.met`.
  - See below to generate the cut meteorology files
  - Currently `size` should be equal to  `metsize`
- `freq` reduces the time frequency of the data before loading to reduce computational expense (ie `freq=3` will only load one in every three footprints and corresponding data).
- The function removes any datapoints where there are Nans in the met or fp data, be it because of a data problem or because the footprint is partially out of the domain.
- The topography for each footprint is stored at `data.topog` of size (time, size, size) 

### Preparing met files
See file `generate_sat_met.py` (and send to the cluster using `launch_cpu_job.sh`). Meteorology files are in `/group/chemistry/acrg/met_archive/UM/{domain}/{domain}_Met_`, at the same lat-lon resolution as the footprints and with hourly time resolution. Relevant pararameters:
- Here `size` needs to be the desired cutting size
- `met_levels` and `met_variables` define which levels and variables will be saved
- By default the meteorology is linearly interpolated to the timestamp of each footprint
- `met_jump` can be an int or a list. If a list, for each int `jump` in `met_jump`, the met is interpolated to `T - jump` where `T` is the timestamp of each footprint, and saved to the corresponding path in `savemetpath`. time "T" is added automatically

## Environment
See environment_short.yml, I think those are the main packages. environment.yml contains the raw output of saving the environment.
This file does not contain torch and related packages - this is because you will need to install separately a CUDA-enabled version or not depending on where you are running the code. BluePebble has pytorch+cuda pre-installed, which you can load when you submit jobs to the queue (see the launch_train.sh file). To run notebooks or files on the login node, you will need torch and associated packages installed in a different environment, which I manually import when running notebooks with the following line. Alternatively you could have two parallel envs (graphnet to run on cluster, and graphnet+torch to run on login) but that might get more confusing if you need to install packages! 
```
sys.path.insert(0, "/path/to/environment_with_torch/env_name/lib/python3.8/site-packages/")
import torch
```

## Setting up data
### Preparing inputs

Prepare the inputs using `get_all_inputs_graphnet_satellite_v4`. This function outputs:
- `grid` - a list of lat-lon tuples for each of the nodes, extracted from a reference footprint. The model assumes all footprints to be on this same grid, and the mesh will be constructed over this particular grid too. You can define which footprint is the reference one with parameter `latlon_fp` which defaults to 0 (ie use the first footprint in the dataset as reference) and will likely not need to modify this for now
- `idx_grid` - a list of (x,y) coordinate tuples for each node, where 0,0 is the measurement point
- `inputs` - a numpy array with the inputs, of shape (time, nodes, features)
- `names` - list of dictionaries of length `features` with info about each feature
- `data` - returns data object itself, in case any updates needed to be made (eg there are nans in the past data). Don't think it's actually needed to be returned explicitly

Parameters:
- data object
- `variables_past`: dict of variables to extract at each of the jumps passed. format is {"var name as it appears in data.met":[list of levels to extract]}. If a variable is 2D (ie it has no levels, like surface pressure) pass level 0.
- jumps: hours back to load (by default the time of the footprint, `jump=0`, is added automatically)
- variables_nopast: variables to be loaded only for jump=0, though I haven't used it in a while and could be deprecated?
- topog: whereas to add topography as a variable
- others: Other non-met variables that could be added to the inputs, eg lat/lon coords of each node, the euclidean distance... see below


```
others =["lat_coords", "lon_coords", "distance_centre", "x_coords", "y_coords"]
variables_past = {"x_wind":[3,30,51], "wind_speed":[3,30,51], "wind_angle":[3,30,51], "y_wind":[3,9,15,21,30,42,51], "atmosphere_boundary_layer_thickness":[0]}

grid, idx_grid, inputs, names, data = get_all_inputs_graphnet_satellite_v4(data, variables_past=variables_past, jumps=[6], variables_nopast={}, topog=True, others=others, return_idx=True, centered_coords=True)
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
- Not time dependent (all loaded under others parameter except topography) 
  - topography
  - sin_lat_coords/sin_lon_coords/cos_lat_coords/cos_lon_coords - sin and cos of coordinates. This is to encode cyclical variables when using a sphere (ie the whole world), but probably is not as useful when only using a reduced domain
  - lat_coords/lon_coords - coordinates at each node
  - distance_centre - Euclidean distance from the release point, calculated with x/y coords rather than actual distance - this is to speed up calculation, as the lat/lon frame of reference (and therefore distances) changes only slightly for each footprint
  - x_coords/y_coords - numerical indeces of each node in a x/y style, passing `centered_coords=True` returns 0,0 as the center (so negative x coordinates are west, negative y coordinates are south), otherwise 0,0 is the South-West corner and all x/y coords are positive, with the release point at int(size/2), int(size/2). For best practice and better inference across sizes pass centered_coords=True
  - binary_centre - zero for all nodes except the release point which is 1
- Not met but time-dependent - these seem to badly affect training and should not be used until properly tested!
  - normalised_time_of_day: sin and cos of the normalised time of the day in seconds (taking sin and cos to make it cyclical)
  - normalised_time_of_year: sin and cos of the normalised day of the year (taking sin and cos to make it cyclical)
  - relative_time: relative time of meteorology data with respect to release - eg 6 for met at t-6
  - 
### Preparing dataset
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
