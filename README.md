# GATES_LPDM_emulator v0.2.0

This repo implements a new, more user-friendly version of the model described at [Enabling Fast Greenhouse Gas Emissions Inference from Satellites with GATES: a Graph-Neural-Network Atmospheric Transport Emulation System (GMD, 2026)](https://gmd.copernicus.org/articles/19/1893/2026/gmd-19-1893-2026.html)  

Please note that this is the most up-to-date branch, and although it is functional, documentation and usability can be patchy! Keep an eye on this repo and on our website [https://gates-lab.github.io/](https://gates-lab.github.io/) for updates.

## File structure
```
graphnet_LPDM_emulator/
├── gates/                                    # Main GATES package
│   ├── config.py                             # Config loading (get_config, generates config.yml)
│   ├── data/
│   │   ├── datasets.py                       # InputsDataset, FootprintDataset, make_dataloader
│   │   ├── load_data.py                      # LoadSquareSatelliteData
│   │   ├── load_background_data.py           # Boundary condition utils
│   │   └── load_data_helper_funs.py
│   ├── evaluation/
│   │   ├── loss_functions.py
│   │   ├── metrics.py
│   │   └── post_processing.py
│   ├── model/
│   │   ├── forecast.py                       # GraphSatelliteForecaster — top-level model class
│   │   └── layers/
│   │       ├── encoder.py                    # Grid → Mesh encoder
│   │       ├── processor.py                  # Mesh message-passing blocks
│   │       ├── decoder.py                    # Mesh → Grid decoder
│   │       └── graph_net_block.py
│   ├── plotting/
│   │   └── plotting_predictions.py
│   ├── training/
│   │   ├── training.py
│   │   ├── training_background.py
│   │   ├── training_dataclasses.py
│   │   └── training_helperfuns.py
│   └── utils/
│       ├── config_defaults.yml
│       └── shape_utils.py
├── How_Tos/                                  # Reference guides
│   ├── HOW_TO_CONFIG.md
│   ├── HOW_TO_DATA.md
│   ├── HOW_TO_PARAMETER_FILE.md
│   ├── HOW_TO_WandB.md
│   ├── HOW_TO_BOUNDARIES.md
│   └── HOW_TO_evaluation.md
├── notebooks/
│   ├── data_tutorial.ipynb
│   ├── plotting_results.ipynb
│   └── uncertainty_fluxes.ipynb
├── parameter_files/                          # JSON parameter templates
├── tests/
├── trained_models/                           # Saved model checkpoints
├── train_GATES_model.py                      # Training entry point
├── train_GATES_model_multiregion.py
├── train_boundary_model.py
├── predict_GATES_model.py                    # Inference entry point
├── predict_boundary_model.py                 
├── launch_train.sh                           # SLURM job script
├── env_gates_pytorch.yml                     # Conda environment
├── model_description.md
└── pyproject.toml
```

Please check the following HowTos for info on different aspects!
- [HOW_TO_CONFIG.md](How_Tos/HOW_TO_CONFIG.md) for info on setting up the `config.yml` with your paths
- [HOW_TO_PARAMETER_FILE.md](How_Tos/HOW_TO_PARAMETER_FILE.md) for a full reference on the training parameter JSON
- [HOW_TO_DATA.md](How_Tos/HOW_TO_DATA.md) for info on the data structures and loading
- [HOW_TO_evaluation.md](How_Tos/HOW_TO_evaluation.md) for info on the metrics and loss functions
- [HOW_TO_WandB.md](How_Tos/HOW_TO_WandB.md) for info on how to set up tracking and logging of your models with the Weights and Biases package
- [HOW_TO_BOUNDARIES.md](How_Tos/HOW_TO_BOUNDARIES.md)

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

### Config

Generate a `config.yml` at the repo root with default local paths:

```bash
python gates/config.py
```

For HPC platforms (`bp`, `oracle`, `isambard_ai`), pass the `--platform` flag:

```bash
python gates/config.py --platform bp
```

Then edit `config.yml` to point to your data directories (`fp_datadir`, `met_datadir`, etc.). See [HOW_TO_CONFIG.md](How_Tos/HOW_TO_CONFIG.md) for full details.

## Model
Check `model_description.md` for more info on the architecture!

The model operates on two levels: lat-lon grid (of same resolution and shape for the inputs and the outputs) and an intermediate abstract layer, with nodes arranged in hexagons.
In the current setup, the model builts a grid and mesh pair using the location of a "reference footprint", and all predictions are done on this grid. An improvement would be to explore a way to select the best reference footrpint, or to find a way to do this dynamically for each footprint
grid, _ = get_grid(data, parameters.get("grid_reference_fp"))

## Parameter file

The parameter file is a JSON that controls data loading, variable selection, model architecture, and training schedule. A template is in `parameter_files/NEW_parameter_template_gpu_new3b.json`.

Key top-level fields to set for a new run:

| Field | Description |
|-------|-------------|
| `model_name` | Unique experiment identifier |
| `train_load_data.years` / `test_load_data.years` | Years used for training and evaluation |
| `train_load_data.region` | Geographic domain (must match a domain in `config.yml`) |
| `train_load_data.size` | Side length (grid cells) of the patch cut per footprint |
| `variables.met_variables` / `variables.met_levels` | Met variables and vertical levels used as inputs |
| `model_parameters` | GNN architecture (num_blocks, node_dim, edge_dim, resolution, …) |
| `epochs.training` | Total training epochs |
| `use_wandb` | Set to `true` to enable Weights & Biases logging |

See [HOW_TO_PARAMETER_FILE.md](How_Tos/HOW_TO_PARAMETER_FILE.md) for a full field reference.

## Training

```bash
python train_GATES_model.py parameter_file.json
# if the file is not in parameter_files_dir from config.yml:
python train_GATES_model.py --file_path /path/to/folder/ parameter_file.json
# on a SLURM cluster:
sbatch launch_train.sh
```

Use the `train_GATES_model.py` file to train a model. Set up all your model parameters using a parameter JSON (see [Parameter file](#parameter-file) above). A new folder will be created in the `save_models_dir` set in `config.yml`, populated with the following:
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

```bash
python predict_GATES_model.py training_settings_<model_name>.json --file_path /path/to/trained_models/
```

### Predicting: different size, different domain etc 
This is still not implemented in the new version


## Inversion - TO DO! 



## See Also
- [HOW_TO_CONFIG.md](How_Tos/HOW_TO_CONFIG.md)
- [HOW_TO_PARAMETER_FILE.md](How_Tos/HOW_TO_PARAMETER_FILE.md)
- [HOW_TO_DATA.md](How_Tos/HOW_TO_DATA.md)
- [HOW_TO_WandB.md](How_Tos/HOW_TO_WandB.md)
- [HOW_TO_evaluation.md](How_Tos/HOW_TO_evaluation.md)

