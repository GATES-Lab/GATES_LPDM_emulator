
from datetime import datetime
import json
from pathlib import Path
import random
#import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import random
import xarray as xr
from pathlib import Path
import wandb



def write_to_file(message, file_path):
    """
    Appends a timestamped message to a model-specific log file.

    Args:
        message (str): The message to log.
        file_path (str): The path to the log file.

    Returns:
        None
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Log file not found at path: {file_path}")
    f = open(file_path, "a")
    f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
    f.close()


def load_parameter_file(file_path):
    """
    Loads and parses a JSON file from a given path.

    Args:
        file_name (str): The name of the file to load (including extension).
    Returns:
        dict or list: The parsed JSON contents of the file, or None if the file was not found or an error occurred.
    """
    try:
        with open(file_path, 'r') as file:
            if str(file_path).endswith('.json'):
                data = json.load(file)
            else:
                data = file.read()
                data = json.loads(data)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None
    
def save_wandb_artifact(model_name, name, file_type, description, path):
    artifact = wandb.Artifact(
        name=f"{model_name}-{name.replace('_', '-')}",
        type=file_type,
        description=description
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact)

    return artifact

def save_object(obj, name, folder, model_name, file_type="pickle", description="", use_wandb=True):
    """
    Serialises an object to disk as either a pickle or JSON file, then logs it to Weights & Biases as a versioned artifact.

    Args:
        obj (any): The Python object to save and log.
        name (str): A descriptive name for the artifact, used in both the filename and the W&B artifact name.
        folder (str): The local directory in which to save the file.
        model_name (str): The name of the model, included in the filename and W&B artifact name.
        file_type (str): Serialisation format — either "pickle" (default) or "json".
        description (str): An optional description string attached to the W&B artifact.
        use_wandb (bool): If True, logs the saved file to W&B as a versioned artifact. Defaults to True.

    Returns:
        str: The local file path where the object was saved.
    """
    allowed_file_types = ["pickle", "json"]
    if file_type not in allowed_file_types:
        raise ValueError(f"Invalid file_type '{file_type}'. Allowed values are: {allowed_file_types}")
    
    ext = file_type
    filename = f"{name}_{model_name}.{ext}"
    path = Path(folder) / filename

    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    if file_type == "json":
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    if use_wandb:
        _ = save_wandb_artifact(model_name, name, file_type, description, path)

    return path



def set_reproducibility(seed=None):
    """
    Sets random seeds across Python, NumPy, and PyTorch to ensure reproducible training runs.
    If a GPU is available, CUDA seeds are also set and cuDNN is switched to deterministic mode.

    Args:
        seed (int): The seed value to use for reproducibility. Defaults to 34.

    Returns:
        int: The seed value that was applied.
    """
    if seed is None:
        seed = 34  # Default seed value if none provided

    # Standard CPU-based seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU-specific seeds and settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Forces cuDNN to use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        # Disables the auto-tuner that selects the fastest (but non-deterministic) algorithms
        torch.backends.cudnn.benchmark = False
        print("CUDA seeds set and cuDNN configured to deterministic mode.")
    else:
        print("CUDA not available; skipping GPU seed initialization.")

    return seed



class EarlyStopping:
    """
    Monitors validation loss during training and halts training when no improvement is seen
    for a given number of consecutive epochs. Also saves the best model checkpoint to disk,
    and optionally logs it to Weights & Biases as a versioned artifact.
    """
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', use_wandb=True, model_name="model"):
        """
        Args:
            patience (int): Number of epochs with no improvement to wait before stopping. Defaults to 20.
            verbose (bool): If True, prints a message each time the counter increments or the model is saved. Defaults to False.
            delta (float): Minimum change in validation loss to qualify as an improvement. Defaults to 0.
            path (str): File path at which to save the best model checkpoint. Defaults to 'checkpoint.pt'.
            use_wandb (bool): If True, logs the best model checkpoint to W&B as a versioned artifact each time it improves. Defaults to True.
            model_name (str): The model name used for the W&B artifact name. Defaults to "model".
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.use_wandb = use_wandb
        self.model_name = model_name

    def __call__(self, val_loss, model):
        """
        Evaluates the current validation loss and updates the early-stopping state.

        Args:
            val_loss (float): The validation loss for the current epoch.
            model (torch.nn.Module): The model to checkpoint if validation loss has improved.

        Returns:
            None
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model's state dict to disk when validation loss reaches a new minimum,
        and optionally logs it to W&B as a versioned artifact.

        Args:
            val_loss (float): The new best validation loss.
            model (torch.nn.Module): The model whose weights should be saved.

        Returns:
            None
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        if self.use_wandb:
            _ = save_wandb_artifact(self.model_name, f"checkpoint_epoch_{self.counter}", "model", f"Best model checkpoint at epoch {self.counter} with val_loss: {val_loss:.6f}", self.path)


def save_training_plots(epoch, test_dataset, training_ctx, path, model_name, colorbar=True):
    """
    Generates and saves a 4×4 grid of images comparing model predictions against ground truth
    at a selection of test samples, for visual inspection during training.

    Args:
        epoch (int): The current epoch number, used in the output filename.
        test_dataset: A dataset object exposing fp, fp_untransformed, and predictions attributes.
        training_ctx (TrainingContext): The training context object containing configuration and state information.
            image_plots holds the sample_id values to plot, image_dates the matching
            timestamps used only to label them.
        size (tuple of int): The spatial dimensions (height, width) used to reshape flat arrays into images.
        path (str): The base directory path for saving output images.
        model_name (str): The model name used to locate the output subfolder and name the saved file.
        colorbar (bool): If True, adds a shared colorbar per row aligned across all four columns.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    image_plots = training_ctx.image_plots
    image_dates = training_ctx.image_dates
    size = training_ctx.size

    # select by sample_id: a timestamp no longer identifies a single sample
    subset_fp = test_dataset.sel(sample_id=image_plots).copy()

    rows = [
        ("fp_transformed_pred", "Trans Pred"),
        ("fp_transformed",      "Trans Truth"),
        ("fp_pred",             "Orig Pred"),
        ("fp_original",         "Orig Truth"),
    ]

    fig, ax = plt.subplots(
        4, 5, figsize=(11, 10),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]}
    )

    for row, (var, label) in enumerate(rows):
        row_da = subset_fp[var]

        if colorbar:
            vmin = float(row_da.min())
            vmax = float(row_da.max())
        else:
            vmin, vmax = None, None

        for n in range(4):
            im = ax[row, n].imshow(row_da.isel(sample_id=n).values, origin="lower", vmin=vmin, vmax=vmax)
            ax[row, n].set_xticks([])
            ax[row, n].set_yticks([])

            if row == 1:
                ax[row, n].set_title(f"{label}\n({str(image_dates[n][:10])}, id {image_plots[n]})")
            else:
                ax[row, n].set_title(label)

        if colorbar:
            fig.colorbar(im, cax=ax[row, 4])
        else:
            ax[row, 4].set_visible(False)

    plt.tight_layout()
    save_path = Path(path) / f"training_imgs" / f"{model_name}_{epoch}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path



def plot_receptor_split(fp_xr_dict, path, model_name, verbose=True, use_wandb=False):
    """
    Scatter plot of receptor release points, coloured by split assignment (train/val/
    test), saved into the training_imgs subfolder.

    Schema-agnostic: fp_xr_dict is just whatever {"train": fp_xr, "val": fp_xr,
    "test": fp_xr, ...} came back from load_receptor_data, however it was split
    (random, sequential or box) — this only reads release_lat/release_lon/receptor
    off each split's fp_xr, so it doesn't need to know which schema produced them.

    Args:
        fp_xr_dict (dict): {split_name: xr.Dataset}
            Each Dataset must carry 'receptor', 'release_lat' and 'release_lon' as
            per-sample_id coordinates (as returned by load_receptor_data).
        path (str): The base directory path for saving the output image.
        model_name (str): The model name used to name the saved file.
        verbose (bool): If True, prints the save path once the plot is written. Defaults to True.
        use_wandb (bool): If True, logs the plot to W&B as a versioned artifact. Defaults to False.

    Returns:
        Path: the path the plot was saved to.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines(resolution="10m", color="black", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    all_lon, all_lat = [], []
    for split_name, split_fp in sorted(fp_xr_dict.items(), key=lambda kv: -kv[1].sizes.get("sample_id", 0)):
        if split_fp.sizes.get("sample_id", 0) == 0:
            continue
        _, first_idx = np.unique(split_fp.receptor.values, return_index=True)
        lon = split_fp.release_lon.values[first_idx]
        lat = split_fp.release_lat.values[first_idx]
        all_lon.append(lon)
        all_lat.append(lat)
        ax.scatter(lon, lat, s=10, alpha=0.8,
                   label=f"{split_name} (n={len(first_idx)})",
                   transform=ccrs.PlateCarree())

    if all_lon:
        all_lon, all_lat = np.concatenate(all_lon), np.concatenate(all_lat)
        pad = 0.3
        ax.set_extent(
            [all_lon.min() - pad, all_lon.max() + pad, all_lat.min() - pad, all_lat.max() + pad],
            crs=ccrs.PlateCarree(),
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"Receptor split ({model_name})")
    ax.legend(markerscale=3)

    save_path = Path(path) / "training_imgs" / f"{model_name}_receptor_split.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Receptor split plot saved to: {save_path}")

    if use_wandb:
        wandb.log({"receptor_split_plot": wandb.Image(str(save_path))})

    return save_path


def export_results_to_netcdf(test_fp_dataset, path, model_name, use_wandb=True):
    """
    Reshapes model predictions and ground truth arrays into (time, lat, lon) format, writes them to
    a NetCDF file via Xarray, and logs the file to Weights & Biases as a versioned dataset artifact.

    Args:
        test_out (np.ndarray): Raw model predictions in transformed space, shape (N, flat_spatial).
        transformed_preds (np.ndarray): Predictions mapped back to the original space, shape (N, flat_spatial).
        test_dataset: A dataset object exposing fp (transformed truth) and fp_untransformed attributes.
        test_data: An object exposing met.time.values, providing the time coordinates for the NetCDF file.
        size (tuple of int): The spatial dimensions (height, width) used to reshape flat arrays into (lat, lon) grids.
        path (str): The base directory path for saving the NetCDF file.
        model_name (str): The model name used to locate the output subfolder and tag the artifact.
        use_wandb (bool): If True, logs the NetCDF file to W&B as a versioned artifact. Defaults to True.
    Returns:
        None
    """

    if not isinstance(test_fp_dataset, xr.Dataset):
        raise ValueError("test_fp_dataset must be an xarray Dataset object.")
    # write this properly
    if not all(coord in test_fp_dataset.coords for coord in ["time", "lat", "lon"]):
        print("test_fp_dataset must have 'time', 'lat', and 'lon' coordinates.")

    # append attrs
    attrs={'creation_date': str(datetime.now()), "model_name": model_name}
    test_fp_dataset.attrs.update(attrs)

    netcdf_save_path = f"{path}/sample_predictions_test.nc"
    test_fp_dataset.to_netcdf(netcdf_save_path)
    print("NetCDF file saved at path:", netcdf_save_path)

    if use_wandb:
        save_wandb_artifact(model_name, "predictions", "dataset", f"Sample predictions saved during training for model {model_name}", netcdf_save_path)
