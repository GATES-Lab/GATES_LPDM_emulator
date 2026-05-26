# GATES_LPDM_emulator v0.2.0

This repo implements a new, more user-friendly version of the model described at [Enabling Fast Greenhouse Gas Emissions Inference from Satellites with GATES: a Graph-Neural-Network Atmospheric Transport Emulation System (GMD, 2026)](https://gmd.copernicus.org/articles/19/1893/2026/gmd-19-1893-2026.html)  

Please note that this is the most up-to-date branch, and although it is functional, documentation and usability can be patchy! Keep an eye on this repo and on our website [https://gates-lab.github.io/](https://gates-lab.github.io/) for updates.

## New file structure (currently in construction)
```
gates_LPDM_emulator/
├── gates/
│   └── data/
│   |   ├── load_data.py         # LoadSquareSatelliteData, LoadBaseSatelliteData
│   |   ├── datasets.py          # InputsDataset, FootprintDataset, 
│   |   └── HOW_TO_DATA.md       # Info on how to use the data files
|   └── evaluation/
│   │   ├── metrics.py         
│   │   └── loss_functions.py
|   └── training/
│       └── training.py
├── notebooks/
│       ├── data_tutorial.ipynb
│       └── plotting_results.ipynb
├── train_GATES_model.py
├── train_GATES_model_multiregion.py
├── predict_GATES_model.py
```

Please check the following HowTos for info on different aspects!
- [HOWTO_CONFIG.md](How_Tos/HOWTO_CONFIG.md) for info on setting up the `config.yml` with your paths
- [HOWTO_DATA.md](How_Tos/HOWTO_DATA.md) for info on the data structures and loading
- [HOWTO_EVALUATION.md](How_Tos/HOWTO_EVALUATION.md) for info on the metrics and loss functions
- [HOWTO_WandB.md](How_Tos/HOWTO_WandB.md) for info on how to set up tracking and logging of your models with the Weights and Biases package

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

 
## Model
Check `model_description.md` for more info on the architecture!

The model operates on two levels: lat-lon grid (of same resolution and shape for the inputs and the outputs) and an intermediate abstract layer, with nodes arranged in hexagons.
In the current setup, the model builts a grid and mesh pair using the location of a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

## Parameter file
This section needs to be written up!

## Training
python train_GATES_model.py NEW_parameter_template_gpu_new.json
Use the `train_GATES_model.py` file to train a model. Set up all your model parameters using the NEW_parameter_template_gpu_new.json file. A new folder will be created, and populated with the following:
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

### Predicting: same model and size
This section needs to be written up!

### Predicting: different size, different domain etc 
This is still not implemented in the new version


## Inversion - TO DO! 



## See Also
- [HOW_TO_CONFIG.md](./gates/HOW_TO_CONFIG.md)
- [HOW_TO_DATA.md](./gates/data/HOW_TO_DATA.md)
- [HOW_TO_TESTS.md](./tests/HOW_TO_TESTS.md)

