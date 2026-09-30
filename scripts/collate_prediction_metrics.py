#!/usr/bin/env python
"""Collate annual prediction metrics across model and test-region pairs."""

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd


def parse_annual_mean(summary_path: Path) -> dict[str, float]:
    """Read the annual-mean section written by compute_prediction_metrics.py."""
    contents = summary_path.read_text()
    marker = "=== Annual mean ==="
    if marker not in contents:
        raise ValueError(f"Annual mean section not found in {summary_path}")

    annual_mean = pd.read_csv(
        StringIO(contents.split(marker, maxsplit=1)[1].strip()),
        sep=r"\s+",
        header=None,
        names=["metric", "value"],
    )
    return dict(zip(annual_mean["metric"], annual_mean["value"]))


def describe_prediction_dir(summary_path: Path) -> tuple[str, str, str]:
    """Extract model label, reference model, and test region from output layout."""
    prediction_dir = summary_path.parent
    test_region = prediction_dir.name.removeprefix("predictions_")
    reference_model = prediction_dir.parent.name
    model_label = prediction_dir.parent.parent.name.removesuffix("_loro_model")
    return model_label, reference_model, test_region


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate annual metrics from GATES prediction summaries."
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("model_test_predictions"),
        help="Root containing metrics_summary.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for collated output (defaults to predictions root).",
    )
    args = parser.parse_args()

    summary_paths = sorted(args.predictions_root.glob("*/*/predictions_*/metrics_summary.txt"))
    if not summary_paths:
        raise FileNotFoundError(
            f"No metrics_summary.txt files found under {args.predictions_root}"
        )

    rows = []
    for summary_path in summary_paths:
        model_label, reference_model, test_region = describe_prediction_dir(summary_path)
        metrics = parse_annual_mean(summary_path)
        rows.append(
            {
                "model": model_label,
                "reference_model": reference_model,
                "test_region": test_region,
                **metrics,
            }
        )

    table = pd.DataFrame(rows).sort_values(["model", "test_region"])
    output_dir = args.output_dir or args.predictions_root
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "annual_metrics_summary.csv"
    table.to_csv(csv_path, index=False)

    display_table = table.copy()
    if "mse" in display_table:
        display_table["mse"] = display_table["mse"].map(
            lambda value: f"{value * 1e7:.4f}e-7"
        )

    text_path = output_dir / "annual_metrics_summary.txt"
    with text_path.open("w") as output_file:
        output_file.write("=== Annual means by model and test region ===\n")
        output_file.write(display_table.to_string(index=False))
        output_file.write("\n")

        metric_columns = table.columns[3:]
        for metric in metric_columns:
            metric_table = table.pivot(index="test_region", columns="model", values=metric)
            if metric == "mse":
                output_file.write("\n=== mse (x 1e-7) ===\n")
                metric_table = metric_table.map(lambda value: f"{value * 1e7:.4f}")
            else:
                output_file.write(f"\n=== {metric} ===\n")
            output_file.write(metric_table.to_string())
            output_file.write("\n")

    print(f"Collated {len(table)} model/test-region pairs.")
    print(f"CSV table: {csv_path}")
    print(f"Text tables: {text_path}")


if __name__ == "__main__":
    main()