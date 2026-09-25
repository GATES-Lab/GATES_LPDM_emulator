"""Re-run a trained dual-head model over the TEST set from a saved checkpoint.

The dual trainers export ``sample_predictions_test.nc`` only at the END of training, so a run
whose footprint head peaked earlier (and then overfitted) has no saved predictions for its best
state. This script regenerates them from any checkpoint:

    python predict_dual_model.py --run_dir <model_runs/run_a> [--run_dir <model_runs/run_b> ...] \
        --checkpoint best_fp [--out sample_predictions_test_best_fp.nc]

Each run's own ``training_outputs/training_settings_*.json`` supplies the parameters, so the data,
scalers and model are rebuilt exactly as that run had them (principle 2). Several runs can be
passed at once: they must share a data-loading configuration, and the (expensive) load and the
transforms are then done once and reused, which also guarantees every run is scored on identical
inputs. The output NetCDF has the same variables as the one written during training.
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy
import numpy as np
import torch

import gates
import gates.training.training_dual as gates_training_dual
from gates.data.load_data import get_grid
from gates.training.training_background import format_aux_data, normalize_boundary_data, denormalize
from gates.training.training_dataclasses import PathContext, BoundaryTrainingContext
from gates.training.training_helperfuns import set_reproducibility, enable_deterministic_algorithms
from train_dual_model import (
    load_dual_data, resolve_background_params, validate_and_predict, build_model_and_context,
)


def restore_scaler_names(parameters, run_dir):
    """Put the scaler CLASS NAMES back into ``parameters`` from the run's saved scaler objects.

    Runs trained before 2026-09-21 have ``training_settings_*.json`` without
    ``input_scaler.scaler`` / ``fp_scaler.scaler``: the setup functions popped those keys from
    the parameter dict in place before the settings were written (fixed in
    ``gates.training.training``). Rebuilding from such a file would silently fit a DIFFERENT
    scaler — for the footprints that changes the target space and makes the fp loss
    meaningless. The scaler objects themselves were saved, so take the names from them.
    """
    files = sorted(Path(run_dir).glob("training_outputs/scalers_*.pickle"))
    if not files:
        raise FileNotFoundError(f"no scalers_*.pickle in {run_dir}/training_outputs")
    with open(files[0], "rb") as f:
        saved = pickle.load(f)
    for key, saved_key in (("input_scaler", "inputs_scaler"), ("fp_scaler", "fp_scaler")):
        block = parameters.setdefault(key, {})
        name = type(saved[saved_key]).__name__
        if block.get("scaler") is None:
            print(f"{Path(run_dir).name}: restoring {key}.scaler = {name} from the saved scaler object")
            block["scaler"] = name
        elif block["scaler"] != name:
            raise SystemExit(f"{run_dir}: {key}.scaler is {block['scaler']} in the settings but the "
                             f"saved scaler object is {name}")
    return saved


def check_refitted_scalers(scalers, saved, run_dir):
    """Confirm the refitted footprint scaler matches the one saved by the run."""
    new, old = scalers["fp_scaler"], saved["fp_scaler"]
    if type(new).__name__ != type(old).__name__:
        raise SystemExit(f"{run_dir}: refitted fp scaler {type(new).__name__} != saved {type(old).__name__}")
    try:
        np, op = new.get_params(), old.get_params()
    except Exception:
        return
    for k in op:
        if not np_allclose(np.get(k), op.get(k)):
            raise SystemExit(f"{run_dir}: refitted fp scaler parameter '{k}' = {np.get(k)} "
                             f"differs from the saved {op.get(k)} — predictions would not match training")
    print(f"refitted fp scaler matches the saved one: {op}")


def np_allclose(a, b):
    if a is None or b is None:
        return a is b
    return bool(numpy.allclose(numpy.asarray(a, dtype=float), numpy.asarray(b, dtype=float)))


def load_run_parameters(run_dir):
    files = sorted(Path(run_dir).glob("training_outputs/training_settings_*.json"))
    if not files:
        raise FileNotFoundError(f"no training_settings_*.json in {run_dir}/training_outputs")
    with open(files[0]) as f:
        p = json.load(f)
    p["use_wandb"] = False
    # the checkpoint carries the trained trunk; a warm-start block would only re-load the
    # pretrained weights before they are overwritten
    p.pop("pretrained_trunk", None)
    p.pop("distributed", None)                      # inference is single-process
    return p


def find_checkpoint(run_dir, name):
    run_dir = Path(run_dir)
    matches = sorted(run_dir.glob(f"*_{name}.pt"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint *_{name}.pt in {run_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"several checkpoints match *_{name}.pt in {run_dir}: {matches}")
    return matches[0]


DATA_KEYS = ("train_load_data", "test_load_data", "variables", "background_setup", "data_dirs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", action="append", required=True, help="repeatable; a trained run's directory")
    ap.add_argument("--checkpoint", default="best_fp", help="checkpoint suffix: best_fp, best_bg, best, or an epoch number")
    ap.add_argument("--out", default=None, help="output file name inside each run dir "
                                                "(default: sample_predictions_test_<checkpoint>.nc)")
    args = ap.parse_args()
    out_name = args.out or f"sample_predictions_test_{args.checkpoint}.nc"

    run_dirs = [Path(r) for r in args.run_dir]
    params = [load_run_parameters(r) for r in run_dirs]
    saved_scalers = [restore_scaler_names(p, r) for p, r in zip(params, run_dirs)]
    for r, p in zip(run_dirs[1:], params[1:]):
        for k in DATA_KEYS:
            if json.dumps(p.get(k), sort_keys=True) != json.dumps(params[0].get(k), sort_keys=True):
                raise SystemExit(f"{r} differs from {run_dirs[0]} in '{k}' — run it separately")
    checkpoints = [find_checkpoint(r, args.checkpoint) for r in run_dirs]
    print("Regenerating test predictions for:")
    for r, c in zip(run_dirs, checkpoints):
        print(f"  {r.name}  <-  {c.name}")

    base = copy.deepcopy(params[0])
    set_reproducibility(base.get("seed") or 34)
    enable_deterministic_algorithms(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    background_params = resolve_background_params(base)
    base["background_setup"] = background_params
    num_classes = base["model_parameters"].get("num_classes", 1)

    print("Loading data (train is needed to refit the same scalers; test is what is scored)...")
    bundle = load_dual_data(base, verbose=base.get("verbose", True))
    (train_fp_data, train_inputs, train_bgs, train_aux, test_fp_data, test_inputs, test_bgs, test_aux) = bundle

    keys = ["summed"] if num_classes == 1 else ["north", "south", "east", "west"]
    train_bgs = train_bgs[keys].to_dataarray().transpose("time", "variable")
    test_bgs = test_bgs[keys].to_dataarray().transpose("time", "variable")
    if background_params["use_auxiliary_bc"]:
        train_aux = format_aux_data(train_aux, time_coord=train_fp_data.time)
        test_aux = format_aux_data(test_aux, time_coord=test_fp_data.time)
    norm_train_bgs, norm_train_aux, norm_vals = normalize_boundary_data(train_bgs, aux_data=train_aux)
    norm_test_bgs, norm_test_aux, norm_vals = normalize_boundary_data(test_bgs, aux_data=test_aux, norm_vals=norm_vals)

    feature_dim = train_inputs.shape[-1]
    aux_dim = train_aux.aux.shape[0] if background_params["use_auxiliary_bc"] else 0
    base["num_features"] = feature_dim + aux_dim

    train_loader, test_loader, fp_labels, test_scaled_fp, scalers = gates_training_dual.setup_dual_dataloaders(
        base, train_inputs, train_fp_data, norm_train_bgs,
        test_inputs, test_fp_data, norm_test_bgs, norm_train_aux, norm_test_aux)
    del train_loader
    check_refitted_scalers(scalers, saved_scalers[0], run_dirs[0])
    grid, _ = get_grid(train_fp_data, base.get("grid_reference_fp"))
    training_ctx = BoundaryTrainingContext(
        base, device, False, [], [], grid, fp_labels, scalers, feature_dim,
        len(train_fp_data.lat.values), aux_dim)

    for run_dir, ckpt, p in zip(run_dirs, checkpoints, params):
        print(f"\n=== {run_dir.name} ({ckpt.name}) ===")
        paths_ctx = PathContext(model_save_dir=str(run_dir.parent), model_name=run_dir.name, model_path=run_dir)
        p_infer = copy.deepcopy(p)
        p_infer["num_features"] = base["num_features"]
        model, model_ctx = build_model_and_context(p_infer, training_ctx, paths_ctx)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        state = state.get("model_state_dict", state) if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(state, strict=True)
        model.to(device)

        avg_total, avg_fp, avg_bg, bg_mae, fp_out, bg_out, bg_true = validate_and_predict(
            model, test_loader, model_ctx, output_norm=norm_vals["outputs"])
        print(f"test losses from this checkpoint: total {avg_total:.4f}, fp {avg_fp:.4f}, "
              f"bg {avg_bg:.4f}, bg MAE {bg_mae:.4g}")

        ds = test_scaled_fp.copy(deep=True)
        original_space = scalers["fp_scaler"].inverse_transform(fp_out)
        ds["fp_transformed_pred"] = (("time", "lat", "lon"), fp_out.reshape(*ds.fp_original.shape))
        ds["fp_pred"] = (("time", "lat", "lon"), original_space.reshape(*ds.fp_original.shape))
        mean, std = norm_vals["outputs"]
        ds["bg_true_ppb"] = (("time",), denormalize(np.asarray(bg_true), mean, std).reshape(-1) * 1e9)
        ds["bg_pred_ppb"] = (("time",), denormalize(np.asarray(bg_out), mean, std).reshape(-1) * 1e9)
        ds.attrs.update({"model_name": run_dir.name, "checkpoint": ckpt.name,
                         "test_fp_loss": float(avg_fp), "test_bg_loss": float(avg_bg)})
        out_path = run_dir / out_name
        ds.to_netcdf(out_path)
        print("wrote", out_path)
        del model, model_ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
