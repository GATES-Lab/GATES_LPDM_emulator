from pathlib import Path
import numpy as np
import json
import yaml
import warnings
import argparse

config_cache = None

# Path to the root directories of the project
root_dir = Path(__file__).parent.parent
package_dir = root_dir / "gates"

minimum_config_keys = ["data_paths", "domains", "bad_files"]

def get_config():
    """
    Get the global Config object, which holds the repository-wide configuration settings. 
    This function uses a simple caching mechanism to ensure that the Config object is only created once, and subsequent calls to get_config() will return the same Config instance.
    """
    global config_cache
    if config_cache is None:
        config_cache = Config()
    return config_cache

def _load_default_config():
    """Load the default configuration from a YAML file."""
    config_path = package_dir / "utils" / "config_defaults.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Default config file not found at {config_path}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup(platform="local"):
    """Create a config file with default paths, loaded from the YAML defaults. The user should then edit the config file to set correct paths for their system. 

    Parameters:
    - platform: str, one of the keys in the "data_paths" section of the default config YAML. This determines which set of default paths to populate in the new config file. Valid options are "local", "bp" (for University of Bristol's BluePebble cluster), "oracle" and "isambard-ai".
    """

    # Create empty config file
    config_path = root_dir / "config.yml"

    default_config = _load_default_config()

    # allow both isambard-ai and isambard_ai
    platform_alias = {"isambard-ai":"isambard_ai"}
    platform = platform_aliases.get(platform, platform)

    if platform not in default_config["data_paths"]:
        raise ValueError(f"Unknown platform '{platform}'. Valid options: {list(default_config['data_paths'].keys())}")

    data_paths = default_config["data_paths"].get(platform)

    configs_from_default = {"data_paths": data_paths, "domains": default_config["domains"], "bad_files": default_config["bad_files"]}

    with open(config_path, "w") as f:
        # save as yml not json
        yaml.dump(configs_from_default, f, sort_keys=False)
    
    if platform == "local":
        print(f"Config file created at {config_path}. Please check the paths and update as needed.")
    else:
        print(f"Config file created at {config_path} with default paths for platform '{platform}'. Please check the paths and update as needed.")

class Config():
    """Global class to hold the repository-wide configuration, including data paths and other settings. This class reads from the config file created by the setup() function, and makes the config values available as attributes of the Config object.

    Attributes:
    - root_dir: Path to the root directory of the project.
    - package_dir: Path to the gates package directory.
    - fp_datadir: Path to the footprint data directory, constructed from the base_data_path and fp_datadir values in the config file.
    - met_datadir: Path to the meteorological data directory, constructed from the base_data_path and met_datadir values in the config file.
    - topog_datadir and landcover_datadir: Paths to the topography and landcover data files, constructed from the base_data_path and respective datadir values in the config file.
    - domains: Dictionary of domain definitions, loaded from the config file.
    - bad_files: List of known footprint files that have to be loaded using a workaround in load_fps, loaded from the config file.
    """
    
    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("Config is read-only after initialization.")
        object.__setattr__(self, name, value)
        
    def __init__(self):
        if not (root_dir / "config.yml").exists():
            raise FileNotFoundError(f"Config file not found at {root_dir / 'config.yml'}. Please run `python gates/config.py ` to create a new config file.")

        # after the class is initialised it is "locked" to prevent modifications to a cached config
        object.__setattr__(self, "_locked", False)
        
        self.root_dir = root_dir
        self.package_dir = package_dir

        # Read user config file
        with open(root_dir / "config.yml", "r") as f:
            config_user = yaml.safe_load(f)

        # set all the config values as attributes of the Config object
        for key, value in config_user.items():
            setattr(self, key, value)

        if np.any([not hasattr(self, key) for key in minimum_config_keys]):
            raise ValueError(f"Config file is missing some of the expected keys: {minimum_config_keys}. Please check your config file at {root_dir / 'config.yml'}.")
        
        self.fp_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["fp_datadir"].lstrip("/\\")
        self.met_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["met_datadir"].lstrip("/\\")
        self.topog_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["topog_datadir"].lstrip("/\\")
        self.landcover_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["landcover_datadir"].lstrip("/\\")
        object.__setattr__(self, "_locked", True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create config.yml from default YAML settings."
    )
    parser.add_argument(
        "--platform",
        dest="platform",
        default=None,
        help=(
            "Platform key from data_paths in config_defaults.yml "
            "(for example: local, bp, oracle, isambard-ai)."
        ),
    )

    args = parser.parse_args()
    platform = args.platform or "local"
    setup(platform)
