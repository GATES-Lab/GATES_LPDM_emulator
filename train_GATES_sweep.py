#!/usr/bin/env python3
"""
Sweep launcher for GATES training jobs.

Reads a parameter JSON containing a "__sweep__" section, generates all
combinations (Cartesian product), writes one temporary parameter file per
combination into <param_dir>/sweep_configs/, and submits a SLURM job for each.

Sweep section format — add this anywhere inside the parameter JSON:
    "__sweep__": {
        "learning_rate": [1e-4, 5e-5, 1e-5],
        "model_parameters.num_blocks": [2, 4, 6]
    }

Dot-notation addresses nested keys at any depth. Everything else in the JSON
is used as-is for every job. The "__sweep__" key itself is stripped before
writing the per-combination files.

Usage:
    python train_GATES_sweep.py my_config.json
    python train_GATES_sweep.py parameter_files/my_config.json --dry-run
    python train_GATES_sweep.py parameter_files/my_config.json --sbatch-script launch_train_sweep.sh
"""

import argparse
import copy
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

from gates.config import get_config


def set_nested(d, key_path, value):
    """Set a value in a nested dict using a dot-notation path."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def format_value(v):
    """Format a single value compactly for use in file and job names."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.2g}"
    if isinstance(v, (list, dict)):
        return "complex"
    return str(v)


def make_suffix(combo):
    """
    Build a readable suffix string from a {dot_key: value} combo dict.
    Uses the last segment of the key path unless two keys share the same
    last segment, in which case the full path is used.
    """
    short_keys = [kp.split(".")[-1] for kp in combo]
    use_full = len(set(short_keys)) < len(short_keys)
    parts = []
    for key_path, value in combo.items():
        label = key_path.replace(".", "-") if use_full else key_path.split(".")[-1]
        parts.append(f"{label}-{format_value(value)}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Submit one SLURM job per parameter combination found in '__sweep__'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("param_file", help="Path to the parameter JSON with a '__sweep__' section")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting them")
    parser.add_argument(
        "--sbatch-script",
        default="launch_train_sweep.sh",
        help="SLURM batch script to submit (default: launch_train_sweep.sh)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write per-combination JSONs (default: <param_file_dir>/sweep_configs/)",
    )
    args = parser.parse_args()

    param_path = Path(args.param_file)
    if not param_path.exists():
        try:
            cfg = get_config()
            fallback = Path(cfg.parameter_files_dir) / param_path
            if fallback.exists():
                print(f"Note: '{param_path}' not found locally, using config path: {fallback}")
                param_path = fallback
            else:
                sys.exit(
                    f"Error: parameter file not found at '{param_path}' "
                    f"or in config parameter_files_dir ('{fallback}')."
                )
        except Exception as e:
            sys.exit(f"Error: parameter file not found at '{param_path}' and config could not be loaded ({e}).")
    param_path = param_path.resolve()

    with open(param_path) as f:
        params = json.load(f)

    sweep_spec = params.pop("__sweep__", None)
    if not sweep_spec:
        sys.exit("No '__sweep__' section found in the parameter file. Nothing to do.")

    sweep_keys = list(sweep_spec.keys())
    sweep_values = [sweep_spec[k] for k in sweep_keys]
    combos = [dict(zip(sweep_keys, vals)) for vals in product(*sweep_values)]

    n_combos = len(combos)
    print(f"Sweep over {len(sweep_keys)} parameter(s) → {n_combos} combination(s):")
    for k, v in sweep_spec.items():
        print(f"  {k}: {v}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else param_path.parent / "sweep_configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sbatch_script = Path(args.sbatch_script).resolve()
    if not sbatch_script.exists():
        sys.exit(f"Error: sbatch script not found: {sbatch_script}")

    base_name = param_path.stem
    base_model_name = params.get("model_name", base_name)

    submitted = []
    for i, combo in enumerate(combos):
        combo_params = copy.deepcopy(params)
        for key_path, value in combo.items():
            set_nested(combo_params, key_path, value)

        suffix = make_suffix(combo)
        combo_params["model_name"] = f"{base_model_name}_sweep_{suffix}"

        out_file = out_dir / f"{base_name}_sweep_{suffix}.json"
        with open(out_file, "w") as f:
            json.dump(combo_params, f, indent=4)

        # SLURM job name: cap at 80 chars (cluster limit)
        job_name = f"sweep_{base_model_name}_{suffix}"[:80]

        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--output=slurms/slurm-%j_{suffix}.out",
            # NONE prevents inheriting the submitting shell's env; explicit vars are still passed
            f"--export=NONE,SWEEP_PARAM_FILE={out_file},SWEEP_JOB_NAME={job_name}",
            str(sbatch_script),
        ]

        print(f"\n[{i + 1}/{n_combos}] {suffix}")
        print(f"  config  : {out_file}")
        print(f"  job name: {job_name}")
        if args.dry_run:
            print(f"  command : {' '.join(sbatch_cmd)}")
        else:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                cwd=sbatch_script.parent,
            )
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
            else:
                print(f"  submitted: {result.stdout.strip()}")
                submitted.append(result.stdout.strip())

    if args.dry_run:
        print(f"\nDry run — {n_combos} job(s) would be submitted.")
    else:
        print(f"\nDone — {len(submitted)}/{n_combos} job(s) submitted.")


if __name__ == "__main__":
    main()
