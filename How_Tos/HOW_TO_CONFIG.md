# How-To: Generate and configure `config.yml`

## Overview
This guide explains how to generate, set up and use the `config.yml` file for the GATES LPDM emulator.

The purpose of `config.yml` is to centralise configuration such as data paths, domain mappings, and known problematic files, removing the need for hard‑coded values scattered throughout the codebase. 

---
## Basic Structure

The file is currently composed of four sections:
```yaml
data_paths: paths to fp, met, topog, landcover, and flux data directories
user_paths: placeholder in case defining a directory to save model outputs is desired
domains: name mapping between domains and regions
bad_files: list of problematic files
```

---
## Getting Started
### Step 1 — `config.yml` Generation
To generate a `config.yml` file in the repository with default paths for local usage, from the root directory, run:
```bash
python gates/config.py
```

GATES has been developed on several HPC platforms including `[bp, oracle, isambard_ai]`. To get set up with appropriate paths for one of these platforms, pass the appropriate arguement using the `--platform`` flag:
```bash
python gates/config.py --platform bp
```

Paths can now manually be edited if necessary.


---
### Step 2 - Validation

After generating your `config.yml`, validate it by [running tests, to be added]

---
### Step 3 — Loading the config file

When executing scripts, the repository config can be imported with

```python
from gates.config import get_config
cfg = get_config()
print(cfg.fp_datadir)
```
Initialising the config class Config loads `config.yml` and saves the paths. calling get_config initialises Config once and keeps it in the cache, so it can be called without needing to open the file each time.


## Troubleshooting
- **Invalid YAML**: If you've edited `config.yml`, ensure to use consistent indentation

## See Also
- [README.md](../README.md)