from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

@dataclass()
class PathContext:
    '''Keep track of necessary paths'''
    model_save_dir: Path 
    model_name: str
    model_path: Path ## model_save_dir / model_name

    def make_dirs(self):
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
        '''Resolve the datapath arguments for loading the data passed from the parameter file, which will supercede the config paths'''
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
    '''Keep track of model parameters and objects'''
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


