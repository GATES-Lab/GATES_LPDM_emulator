from pathlib import Path
import numpy as np
import yaml
import argparse

config_cache = None

# Path to the root directories of the project
root_dir = Path(__file__).parent.parent
package_dir = root_dir / "gates"

minimum_config_keys = ["data_paths", "domains", "bad_fp_files", "user_paths"]

def get_config():
    """Get the global Config object, which holds the repository-wide configuration settings.

    Uses a simple caching mechanism to ensure that the Config object is only created
    once, and subsequent calls to ``get_config()`` return the same Config instance.

    Returns:
        Config: The cached (or newly created) global Config instance.
    """
    global config_cache
    if config_cache is None:
        config_cache = Config()
    return config_cache

def _load_default_config():
    """Load the default configuration from a YAML file.

    Returns:
        dict: Parsed contents of ``gates/utils/config_defaults.yml``.

    Raises:
        FileNotFoundError: If the default config file does not exist.
    """
    config_path = package_dir / "utils" / "config_defaults.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Default config file not found at {config_path}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup(platform="local"):
    """Create a config file with default paths, loaded from the YAML defaults.

    The user should then edit the config file to set correct paths for their system.

    Args:
        platform (str, optional): One of the keys in the "data_paths" section of the
            default config YAML. Determines which set of default paths to populate
            in the new config file. Valid options are "local", "bp" (for University
            of Bristol's BluePebble cluster), "oracle" and "isambard-ai".
            Defaults to "local".

    Raises:
        ValueError: If ``platform`` (after alias resolution) is not a recognised key
            in the default config's "data_paths".
    """

    # Create empty config file
    config_path = root_dir / "config.yml"

    default_config = _load_default_config()

    # allow flexibility in platform names
    platform = platform.lower()
    platform_alias = {
        "isambard-ai": "isambard_ai",
        "bluepebble": "bp",
    }
    platform = platform_alias.get(platform, platform)

    available_platforms = list(default_config["data_paths"].keys())

    if platform not in default_config["data_paths"]:
        raise ValueError(f"Unknown platform '{platform}'. Valid options: {available_platforms}")

    data_paths = default_config["data_paths"].get(platform)

    # copy all the default config values, but replace the data_paths with the selected platform's paths
    configs_from_default = default_config.copy()
    configs_from_default["data_paths"] = data_paths

    with open(config_path, "w") as f:
        # save as yml not json
        yaml.dump(configs_from_default, f, sort_keys=False)
    
    if platform == "local":
        print(f"Config file created at {config_path}. Please check the paths and update as needed.\nYou have currently selected the default paths option for running GATES locally, if you are using GATES on another platform make sure you have specified this with the --platform flag.\nCurrently valid arguments are: {available_platforms}")
    else:
        print(f"Config file created at {config_path} with default paths for platform '{platform}'. Please check the paths and update as needed.")

class Config():
    """Global class to hold the repository-wide configuration, including data paths and other settings.

    Reads from the config file created by the ``setup()`` function, and makes the
    config values available as attributes of the Config object. The instance is
    read-only (locked) once ``__init__`` completes.

    Attributes:
        root_dir (Path): Path to the root directory of the project.
        package_dir (Path): Path to the gates package directory.
        fp_datadir (Path): Path to the footprint data directory, constructed from the
            ``base_data_path`` and ``fp_datadir`` values in the config file.
        met_datadir (Path): Path to the meteorological data directory, constructed
            from the ``base_data_path`` and ``met_datadir`` values in the config file.
        topog_datadir (Path): Path to the topography data file, constructed from the
            ``base_data_path`` and ``topog_datadir`` values in the config file.
        landcover_datadir (Path or None): Path to the landcover data file,
            constructed from the ``base_data_path`` and ``landcover_datadir`` values
            in the config file, or None if not specified (optional).
        flux_datadir (Path): Path to the flux data directory, constructed from the
            ``base_data_path`` and ``flux_datadir`` values in the config file.
        domains (dict): Dictionary of domain definitions, loaded from the config file.
        bad_fp_files (list): List of known footprint files that have to be loaded
            using a workaround in ``load_fps``, loaded from the config file.
        save_models_dir (Path): Path to the directory where trained models should be
            saved, loaded from the config file.
        parameter_files_dir (Path): Path to the directory where parameter files for
            training should be saved, loaded from the config file.
    """

    def __setattr__(self, name, value):
        """Set an attribute, raising once the config has been locked (after ``__init__``).

        Args:
            name (str): Attribute name.
            value: Attribute value.

        Raises:
            AttributeError: If the config is already locked (read-only).
        """
        if getattr(self, "_locked", False):
            raise AttributeError("Config is read-only after initialization.")
        object.__setattr__(self, name, value)

    def __init__(self, filename="config.yml"):
        """Load and validate the config file, populating this instance's attributes.

        Args:
            filename (str, optional): Config filename, resolved relative to
                ``root_dir``. Defaults to "config.yml".

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config file is empty/invalid YAML, or is missing any
                of ``minimum_config_keys``.
        """
        if not (root_dir / filename).exists():
            raise FileNotFoundError(f"Config file not found at {root_dir / filename}. Please run `python gates/config.py ` to create a new config file.")

        # after the class is initialised it is "locked" to prevent modifications to a cached config
        object.__setattr__(self, "_locked", False)
        
        self.root_dir = root_dir
        self.package_dir = package_dir

        # Read user config file
        with open(root_dir / "config.yml", "r") as f:
            config_user = yaml.safe_load(f)
        
        if not isinstance(config_user, dict):
             raise ValueError(
                 f"Config file at {root_dir / 'config.yml'} is empty or invalid YAML; expected a mapping at the top level."
             )

        # set all the config values as attributes of the Config object
        for key, value in config_user.items():
            setattr(self, key, value)

        if np.any([not hasattr(self, key) for key in minimum_config_keys]):
            raise ValueError(f"Config file is missing some of the expected keys: {minimum_config_keys}. Please check your config file at {root_dir / 'config.yml'}.")
        
        self.fp_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["fp_datadir"].lstrip("/\\")
        self.met_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["met_datadir"].lstrip("/\\")
        self.topog_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["topog_datadir"].lstrip("/\\")
        # the landcover file is optional
        if "landcover_datadir" in self.data_paths and self.data_paths["landcover_datadir"] is not None:
            self.landcover_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["landcover_datadir"].lstrip("/\\")
        else:
            self.landcover_datadir = None

        self.flux_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["flux_datadir"].lstrip("/\\")
        self.bc_datadir = Path(self.data_paths["base_data_path"]) / self.data_paths["bc_datadir"].lstrip("/\\")

        self.save_models_dir = Path(self.user_paths["save_models_dir"])
        self.parameter_files_dir = Path(self.user_paths["parameter_files_dir"])
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
