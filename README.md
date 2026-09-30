# GATES_LPDM_emulator v0.3.0

This repo implements a new, more user-friendly and memory-friendly version of the model described in [Enabling Fast Greenhouse Gas Emissions Inference from Satellites with GATES: a Graph-Neural-Network Atmospheric Transport Emulation System (GMD, 2026)](https://gmd.copernicus.org/articles/19/1893/2026/gmd-19-1893-2026.html).

GATES emulates a Lagrangian Particle Dispersion Model (LPDM). Given the meteorology around a satellite measurement, it predicts the **footprint**: a 2D (lat × lon) field describing how sensitive that measurement is to surface emissions. Footprints are used in atmospheric inversions to estimate greenhouse gas emissions. GATES produces them much faster than running the LPDM, in about one second per footprint instead of 10 minutes.

> **Note:** this is the most up-to-date branch.  You can read the documentation here [https://gates-lab.github.io/GATES_LPDM_emulator/](https://gates-lab.github.io/GATES_LPDM_emulator/).

## Repository structure

```
GATES_LPDM_emulator/
├── gates/                          # the GATES Python package
│   ├── config.py                   # generates and reads config.yml
│   ├── data/
│   │   ├── load_data.py            # LoadSquareSatelliteData
│   │   └── datasets.py             # datasets, scalers, dataloaders
│   ├── model/
│   │   ├── forecast.py             # GraphSatelliteForecaster (the full model)
│   │   └── layers/                 # encoder, processor, decoder
│   ├── training/                   # training loop, early stopping, helpers
│   ├── evaluation/                 # metrics, loss_functions
│   ├── plotting/
│   └── utils/
├── scripts/
│   ├── train_GATES_model.py        # train a model from a parameter file
│   ├── train_GATES_model_multiregion.py
│   ├── train_GATES_sweep.py        # submit a hyperparameter sweep
│   └── predict_GATES_model.py      # run a trained model
├── launch/                         # SLURM batch scripts
├── parameter_files/                # parameter file templates
├── notebooks/                      # data_tutorial.ipynb
├── How_Tos/                        # user guides (see below)
├── docs/                           # Sphinx documentation
└── env_gates_pytorch.yml           # conda environment
```

## Guides

The How_Tos cover each part of the workflow in detail:

- [HOW_TO_CONFIG.md](How_Tos/HOW_TO_CONFIG.md): setting up `config.yml` with your paths
- [HOW_TO_PARAMETER_FILE.md](How_Tos/HOW_TO_PARAMETER_FILE.md): full reference for the training parameter JSON
- [HOW_TO_LAUNCH.md](How_Tos/HOW_TO_LAUNCH.md): launching training runs and sweeps, locally or on SLURM
- [HOW_TO_DATA.md](How_Tos/HOW_TO_DATA.md): data structures and loading
- [HOW_TO_LOSSES.md](How_Tos/HOW_TO_LOSSES.md): loss functions and how to set them in the parameter file
- [HOW_TO_evaluation.md](How_Tos/HOW_TO_evaluation.md): evaluation metrics
- [HOW_TO_PREDICT.md](How_Tos/HOW_TO_PREDICT.md): running a trained model
- [HOW_TO_MULTIREGION.md](How_Tos/HOW_TO_MULTIREGION.md): training one model on several regions
- [HOW_TO_WandB.md](How_Tos/HOW_TO_WandB.md): tracking and logging models with Weights & Biases

## Setting up

### Environment

The repo requires `python=3.12`, `xarray=2025.1` and `pytorch=2.3`. Create the environment from [env_gates_pytorch.yml](env_gates_pytorch.yml):

```bash
conda env create -f env_gates_pytorch.yml
conda activate gates_env
```

The environment is set up for CUDA 12.1. For a CPU-only install or a different CUDA version, you will need to edit it. If you are creating the environment on the login node of a GPU cluster (which has no GPU itself), force a CUDA install with:

```bash
CONDA_OVERRIDE_CUDA=12.1 conda env create -f env_gates_pytorch.yml
```

### Installing the package

From the root of the repo, install an editable version of `gates`:

```bash
pip install --no-build-isolation --no-deps -e .
```

### Config

Generate a `config.yml` at the repo root with default local paths:

```bash
python gates/config.py
```

On a supported HPC platform (`bp`, `oracle`, `isambard-ai`), pass the `--platform` flag to start from that platform's paths:

```bash
python gates/config.py --platform bp
```

Then edit `config.yml` to point to your data directories (`fp_datadir`, `met_datadir`, etc.). See [HOW_TO_CONFIG.md](How_Tos/HOW_TO_CONFIG.md) for details.

## Model

GATES is an encode–process–decode graph neural network, based on Keisler (2022) and GraphCast. It works on two levels: a lat-lon grid (the same shape for inputs and outputs) and a coarser hexagonal mesh built with the `h3` library.

1. **Encoder:** grid nodes send their features to nearby mesh nodes.
2. **Processor:** several rounds (`num_blocks`) of message passing on the mesh.
3. **Decoder:** each grid node collects from its nearest mesh nodes to predict the footprint value.

The grid and mesh are built once from a "reference footprint" (set with `grid_reference_fp` in the parameter file, default: the first footprint), and all predictions use that grid. A possible improvement would be to choose the reference footprint more carefully, or to build the grid dynamically for each footprint.

## Parameter file

The parameter file is a JSON that controls data loading, variable selection, model architecture and the training schedule. Start from [parameter_files/NEW_parameter_template.json](parameter_files/NEW_parameter_template.json). For a quick smoke test that runs in a terminal or on a single CPU node, use `NEW_parameter_template_terminal.json`.

Key fields to set for a new run:

| Field | Description |
|-------|-------------|
| `model_name` | Base name for the experiment. A timestamp is appended at runtime. |
| `train_load_data.years` / `test_load_data.years` | Years used for training and evaluation |
| `train_load_data.region` | Geographic domain (must match a domain in `config.yml`) |
| `train_load_data.size` | Side length (grid cells) of the square cut around each footprint |
| `variables.met_variables` / `variables.met_levels` | Met variables and vertical levels used as inputs |
| `model_parameters` | GNN architecture (`num_blocks`, `node_dim`, `edge_dim`, `resolution`, …) |
| `epochs.training` | Total training epochs |
| `use_wandb` | Set to `true` to enable Weights & Biases logging |

See [HOW_TO_PARAMETER_FILE.md](How_Tos/HOW_TO_PARAMETER_FILE.md) for the full reference.

## Training

```bash
python scripts/train_GATES_model.py parameter_file.json
# if the file is not in parameter_files_dir from config.yml:
python scripts/train_GATES_model.py parameter_file.json --file_path /path/to/folder/
# on a SLURM cluster (edit the script first):
sbatch launch/launch_train.sh
```

See [HOW_TO_LAUNCH.md](How_Tos/HOW_TO_LAUNCH.md) for the SLURM scripts and for running sweeps, and [HOW_TO_MULTIREGION.md](How_Tos/HOW_TO_MULTIREGION.md) for training on several regions.

## Predicting


See [HOW_TO_PREDICT.md](How_Tos/HOW_TO_PREDICT.md) for all options (single months, choosing a checkpoint, changing the region, output location).

Predicting on a different domain size is not yet supported in this version.

## Inversion

Using the predicted footprints in an inversion is not documented yet.

## Citation

If you use GATES, please cite the GMD paper linked at the top. Citation metadata is also in [CITATION.cff](CITATION.cff).
