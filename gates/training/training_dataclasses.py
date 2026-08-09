from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

@dataclass()
class PathContext:
    """Keep track of necessary paths.

    Attributes:
        model_save_dir (Path): <FILL IN>
        model_name (str): <FILL IN>
        model_path (Path): ``model_save_dir / model_name``.
        training_imgs_path (Path): Directory for training images, set by ``make_dirs()``.
        training_outputs_path (Path): Directory for training outputs (scalers,
            training settings, training logs), set by ``make_dirs()``.
        updates_path (Path): Text file for training update logs, set by ``make_dirs()``.
    """
    model_save_dir: Path
    model_name: str
    model_path: Path ## model_save_dir / model_name

    def make_dirs(self):
        """Create the model directory tree and an empty training-updates log file.

        Creates ``self.model_path`` (and any missing parents), plus
        ``training_imgs`` and ``training_outputs`` subdirectories, and sets
        ``self.training_imgs_path``, ``self.training_outputs_path``, and
        ``self.updates_path`` (creating an empty file at that path).

        Raises:
            FileExistsError: If ``training_imgs`` or ``training_outputs`` already
                exist under ``model_path``, or if ``updates_path`` already exists.
        """
        # make the main model directory
        os.makedirs(self.model_path, exist_ok=True)

        #store the training imgs
        self.training_imgs_path = self.model_path / "training_imgs"
        os.mkdir(self.training_imgs_path)

        # store the training outputs, including scalers, training settings, and training logs
        self.   training_outputs_path = self.model_path / "training_outputs"
        os.mkdir(self.training_outputs_path)

        # write the training updates to a text file in the training outputs directory
        self.updates_path = self.model_path / f"{self.model_name}_updates.txt"

        f = open(self.updates_path, "x")
        f.close()

    def resolve_datapath_args(self, parameters):
        """Resolve the datapath arguments for loading the data passed from the parameter file, which will supersede the config paths.

        Args:
            parameters (dict): Parameter dict, optionally containing a "data_dirs"
                key with any of "fp_datadir", "met_datadir", "topog_datadir",
                "landcover_datadir".

        Returns:
            dict: ``{"fp_datadir": ..., "met_args": {"met_datadir": ...}, "topog_args": {"topog_datadir": ..., "landcover_datadir": ...}}``,
            populated only with the keys present in ``parameters["data_dirs"]``.
        """
        datapath_args = {"met_args":{}, "topog_args":{}}
        if "data_dirs" in parameters:
            if "fp_datadir" in parameters["data_dirs"]:
                datapath_args["fp_datadir"] = parameters["data_dirs"]["fp_datadir"]
            if "met_datadir" in parameters["data_dirs"]:
                datapath_args["met_args"]["met_datadir"] = parameters["data_dirs"]["met_datadir"]
            if "topog_datadir" in parameters["data_dirs"]:
                datapath_args["topog_args"]["topog_datadir"] = parameters["data_dirs"]["topog_datadir"]
            if "landcover_datadir" in parameters["data_dirs"]:
                datapath_args["topog_args"]["landcover_datadir"] = parameters["data_dirs"]["landcover_datadir"]
        #self.datapath_args = datapath_args
        return datapath_args


@dataclass()
class TrainingContext:
    """Keep track of training parameters.

    Attributes:
        parameters (dict): <FILL IN>
        device (str): <FILL IN>
        use_wandb (bool): <FILL IN>
        image_dates (list): <FILL IN>
        image_plots (bool): <FILL IN>
        grid (np.array): <FILL IN>
        fp_labels (list): <FILL IN>
        scalers (dict): <FILL IN>
        n_variables (int): <FILL IN>
        size (int): <FILL IN>
        dynamic_edges_params (dict): <FILL IN>
    """
    parameters: dict
    device: str
    use_wandb: bool
    image_dates: list
    image_plots: bool
    grid: np.array
    fp_labels: list
    scalers: dict
    n_variables: int

    size: int

    dynamic_edges_params:dict


@dataclass()
class BoundaryTrainingContext:
    '''Keep track of training parameters'''
    parameters: dict
    device: str
    use_wandb: bool
    image_dates: list
    image_plots: bool
    grid: np.array
    fp_labels: list
    scalers: dict
    n_variables: int
    size: int
    aux_dim: int = 0

@dataclass()
class ModelContext:
    """Keep track of model parameters and objects.

    Attributes:
        model_name (str): <FILL IN>
        use_wandb (bool): <FILL IN>
        device (str): <FILL IN>
        optimizer (object): <FILL IN>
        criterion (object): <FILL IN>
        criterion_test (object): <FILL IN>
        lr (float): <FILL IN>
        early_stopping (object): <FILL IN>
        epochs_num (int): <FILL IN>
        epochs_visualise (int): <FILL IN>
        epochs_save (int): <FILL IN>
        epochs_patience (int): <FILL IN>
    """
    model_name: str
    use_wandb: bool
    device: str

    optimizer: object
    criterion: object
    criterion_test: object

    lr: float

    early_stopping: object

    epochs_num : int
    epochs_visualise: int
    epochs_save: int
    epochs_patience : int


    #grid : array


@dataclass()
class DualModelContext:
    '''Model context for the dual-head model (footprint + background heads).

    Holds a separate criterion (and display criterion) for each head, plus the
    relative weight applied to the background loss when forming the joint loss
    ``total = fp_loss + bg_loss_weight * bg_loss``.'''
    model_name: str
    use_wandb: bool
    device: str

    optimizer: object

    # footprint head (per-node) criteria
    fp_criterion: object
    fp_criterion_test: object
    # background head (whole-grid) criteria
    bg_criterion: object
    bg_criterion_test: object

    bg_loss_weight: float

    lr: float

    early_stopping: object

    epochs_num: int
    epochs_visualise: int
    epochs_save: int
    epochs_patience: int

    # Background-head schedule (see parameters["bg_head"] in setup_dual_model): freeze the
    # bg head after this many epochs without improvement in its own test loss (None =
    # never freeze). ``bg_frozen`` is flipped by the training loop once the freeze happens.
    bg_freeze_patience: object = None
    bg_freeze_min_delta: float = 0.0
    bg_frozen: bool = False


