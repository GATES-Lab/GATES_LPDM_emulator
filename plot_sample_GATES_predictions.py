"""
plot_GATES_predictions.py

Quick look at a GATES predictions file: plots 4 randomly sampled footprints (true vs predicted).

Usage
-----
    python plot_sample_GATES_predictions.py \\
        --reference_model my_model_20240115_143022 \\
        --predictions_file predictions_val_best.nc
        [--model_path /path/to/models/] \\
        [--save_path /path/to/outputs/] \\
        [--model_save_name custom_name]

Notes
-----
- ``reference_model`` accepts either a full timestamped directory name
  (e.g. ``my_model_20240115_143022``) or a base name (e.g. ``my_model``),
  in which case the most recently created matching directory is used.
- ``predictions_file`` dictates the particular predictions file to use.
- Worked for both "satellite" and "receptors" sampling modes


"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

import gates.config
from gates.plotting import plot_fp_predictions
from predict_GATES_model import find_model_dir, determine_save_name


def random_idxs(n_samples, n=4):
    """
    Randomly sample k indices; repeats if there are fewer than n samples.
    """
    if n_samples >= n:
        return random.sample(range(n_samples), n)
    return [random.randrange(n_samples) for _ in range(n)]


def get_predictions_path(args) -> Path:
    """
    Identifies which predictions .nc file to plot from a --reference_model.
    """
    cfg = gates.config.get_config()
    model_path = args.model_path or cfg.save_models_dir
    model_dir = find_model_dir(model_path, args.reference_model)

    model_save_name = determine_save_name(args.model_save_name, model_dir)
    save_path = Path(args.save_path) / model_save_name if args.save_path else model_dir
    predictions_dir = save_path / "predictions"

    predictions_path = predictions_dir / args.predictions_file
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    print(f"Using reference model: {model_dir}")
    print(f"Using predictions file: {predictions_path.name}")

    return predictions_path


def main():
    parser = argparse.ArgumentParser(description="Quick plot of GATES predictions.")
    parser.add_argument(
        "--reference_model",
        required=True,
        help=(
            "Trained model name/directory. Pass the full timestamped name "
            "(e.g. my_model_20240115_143022) for an exact match, or the base "
            "name (e.g. my_model) to use the most recently created run."
        ),
    )
    parser.add_argument(
        "--predictions_file",
        required=True,
        help="Name of the predictions file within the model's predictions folder (e.g. predictions_val_best.nc)",
    )
    parser.add_argument(
        "--model_path", 
        default=None,
        help="Root directory to search for trained models (default: from config.yml)"
    )
    parser.add_argument(
        "--save_path", 
        default=None,
        help="Must match the --save_path used at prediction time, if any."
    )
    parser.add_argument(
        "--model_save_name", 
        default=None,
        help="Must match the --model_save_name used at prediction time, if any."
    )
    
    args = parser.parse_args()

    predictions_path = get_predictions_path(args)
    preds = xr.open_dataset(predictions_path)

    sample_dim = "sample_id" if "sample_id" in preds.dims else "time"
    idxs = random_idxs(preds.sizes[sample_dim])
    print(f"Plotting {sample_dim} indices {idxs}")

    plot_fp_predictions(
        preds, idxs_list=idxs, contour=True, thres=1e-4,
        fig_title=f"Sample Footprint Predictions",
        which_dataspace="original",
    )

    save_dir = predictions_path.parent / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / f"{predictions_path.stem}_sample.png"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()