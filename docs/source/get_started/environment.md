# Environment Setup

The repo requires `python=3.12`, `xarray=2025.1`, and `pytorch=2.3`. Install the environment from [env_gates_pytorch.yml](https://github.com/GATES-Lab/GATES_LPDM_emulator/blob/main/env_gates_pytorch.yml):

```bash
conda env create -f env_gates_pytorch.yml
```

You might need to make modifications to the environment if you want a CPU-only installation, or for different CUDA versions. If you are installing the environment on the login node of a cluster with GPUs, use the following to force a CUDA install:

```bash
CONDA_OVERRIDE_CUDA=12.1 conda env create -f env_gates_pytorch.yml
```

## Installing the Repo

To install an editable version of this package into your environment, run the following from the root of the repo:

```bash
pip install --no-build-isolation --no-deps -e .
```
