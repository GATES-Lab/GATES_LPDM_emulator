#!/usr/bin/env python
"""Compute monthly and annual footprint metrics for prediction NetCDF files."""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from gates.evaluation.metrics import compute_footprint_metrics
from gates.evaluation.post_processing import threshold_fps


def prediction_directories(root: Path, prediction_dir: Path | None) -> list[Path]:
    """Return one requested prediction directory or all completed directories."""
    if prediction_dir is not None:
        return [prediction_dir]
    return sorted(
        directory
        for directory in root.glob("*/*/predictions_*")
        if directory.is_dir() and any(directory.glob("predictions_2014_*.nc"))
    )


def compute_metrics(prediction_dir: Path, year: int, threshold: float) -> Path:
    """Write per-month and annual metrics for one prediction directory."""
    files = sorted(prediction_dir.glob(f"predictions_{year}_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No prediction files found in {prediction_dir} for {year}."
        )

    all_results = []
    for path in files:
        month = path.stem.rsplit("_", 1)[-1]
        print(f"{prediction_dir}: {year}-{month}", flush=True)
        with xr.open_dataset(path) as predictions:
            predictions = threshold_fps(predictions, threshold=threshold)
            metrics = compute_footprint_metrics(
                predictions["fp_original"],
                predictions["fp_pred_thres"],
                ignore_mask=predictions["fp_nan_mask"],
            )
        metrics["month"] = month
        all_results.append(metrics)
        print(f"  {metrics}", flush=True)

    dataframe = pd.DataFrame(all_results).set_index("month")
    annual_mean = dataframe.mean(numeric_only=True)
    output_path = prediction_dir / "metrics_summary.txt"
    with output_path.open("w") as output_file:
        output_file.write("=== Per-month metrics ===\n")
        output_file.write(dataframe.to_string(float_format=lambda value: f"{value:.12e}"))
        output_file.write("\n\n=== Annual mean ===\n")
        output_file.write(annual_mean.to_string(float_format=lambda value: f"{value:.12e}"))
        output_file.write("\n")

    print(f"Saved to {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics for GATES prediction NetCDF files."
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("model_test_predictions"),
        help="Root containing model output directories.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        help="One prediction directory to process instead of discovering all outputs.",
    )
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--threshold", type=float, default=1.75e-5)
    args = parser.parse_args()

    directories = prediction_directories(args.predictions_root, args.prediction_dir)
    if not directories:
        raise FileNotFoundError(
            f"No completed prediction directories found under {args.predictions_root}."
        )

    for directory in directories:
        compute_metrics(directory, args.year, args.threshold)


if __name__ == "__main__":
    main()