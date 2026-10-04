"""Re-run a trained dual-head model over the TEST set from a saved checkpoint.

The dual trainers export ``sample_predictions_test.nc`` only at the END of training, so a run
whose footprint head peaked earlier (and then overfitted) has no saved predictions for its best
state. This script regenerates them from any checkpoint:

    python predict_dual_model.py --run_dir <model_runs/run_a> [--run_dir <model_runs/run_b> ...] \
        --checkpoint best_fp [--out sample_predictions_test_best_fp.nc]

Each run's own ``training_outputs/training_settings_*.json`` supplies the parameters, so the data,
scalers and model are rebuilt exactly as that run had them (principle 2). Several runs can be
passed at once: they must share a data-loading configuration AND the settings that decide the
fitted scalers (``seed``, ``input_scaler``, ``fp_scaler``: the seed picks the random subsample the
input scaler is fitted on, so a seed replicate has its own scaler and is predicted in its own
call) and the grid (``grid_node_order``, ``grid_reference_fp``). The (expensive) load and the
transforms are then done once and reused, which also guarantees every run is scored on identical
inputs. The output NetCDF has the same variables as the one written during training.

Applying a run to a data configuration it was NOT trained on (window-size study, 2026-09-28)::

    python predict_dual_model.py --run_dir <run> --checkpoint best_fp \
        --override train_load_data.size=100 --override 'train_load_data.years=["2014"]' \
        --override 'train_load_data.months=["01"]' --use-saved-scalers --allow-grid-mismatch \
        --out sample_predictions_test_best_fp_size100.nc

``--override key=value`` (repeatable; dotted keys, JSON values) changes the loaded settings before
the data is loaded, e.g. the window ``size`` of both the training and the test set (``test_load_data``
inherits from ``train_load_data``). ``--use-saved-scalers`` applies the run's SAVED input scaler and
background normalisation instead of refitting them on the (possibly different, possibly tiny)
training set that is loaded — with it, the training years/months only fix the reference footprint
of the grid, so a single month suffices. ``--allow-grid-mismatch`` builds the model for the new grid
and drops the two state entries whose shape is tied to the grid size: ``encoder.h3_nodes`` (an
unused all-zero placeholder) and the background head's final linear layer (whose input is the
flattened window, so the BACKGROUND OUTPUT IS INVALID and written as NaN; the footprint head is
per-node and transfers). Everything else is loaded strictly.
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
import gates.training.training as gates_training
import gates.training.training_dual as gates_training_dual
import gates.training.lean_dual_data as gates_lean
import gates.data.input_domain as gates_input_domain
from gates.data.grid import get_model_grid
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


DATA_KEYS = ("train_load_data", "test_load_data", "variables", "background_setup", "data_dirs", "data_cache")

# Settings that decide the FITTED scalers. Runs predicted in one call are scored on one set of
# transformed inputs (the first run's scalers), and the seed picks the random subsample the input
# scaler is fitted on (input_scaler.fit_on_subsample), so these must agree as well.
SCALER_KEYS = ("seed", "input_scaler", "fp_scaler")

# Settings that decide the grid the model is built on (one grid is built per call).
GRID_KEYS = ("grid_node_order", "grid_reference_fp")


def check_runs_share_inputs(run_dirs, params):
    """Stop unless every run can be scored on the data AND the scalers of the first run.

    Raises ``SystemExit`` naming the first run and key that differ (principle 3: a model is
    applied with the transform parameters it was trained with).
    """
    def value(p, key):
        v = p.get(key)
        if key == "seed" and v is None:
            v = 34  # the trainers' default
        if key == "grid_node_order" and v is None:
            v = "legacy"
        return json.dumps(v, sort_keys=True)

    for r, p in zip(run_dirs[1:], params[1:]):
        for k in DATA_KEYS + SCALER_KEYS + GRID_KEYS:
            if value(p, k) != value(params[0], k):
                raise SystemExit(f"{r} differs from {run_dirs[0]} in '{k}' — run it separately")


# state-dict entries whose SHAPE follows the grid size; only these may be dropped by --allow-grid-mismatch
GRID_TIED_STATE_KEYS = ("encoder.h3_nodes", "bg_decoder.linear_class.weight")


def apply_overrides(parameters, overrides):
    """Apply ``["dotted.key=<json value>", ...]`` to ``parameters`` in place; returns the changes.

    Intermediate dicts are created as needed. Values are parsed as JSON, so strings need quotes
    (``'train_load_data.years=["2014"]'``) while numbers/booleans/null do not.
    """
    changes = []
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--override {key}: value {raw!r} is not valid JSON ({e})")
        node = parameters
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--override {key}: '{part}' is not a dict in the settings")
        changes.append((key, node.get(parts[-1]), value))
        node[parts[-1]] = value
    return changes


class SavedInputScaler:
    """Minimal stand-in for ``InputsDataset`` around an already fitted input scaler object."""

    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, inputs):
        transformed = self.scaler.transform(inputs)
        if transformed.dtype != "float32":
            transformed = transformed.astype("float32", copy=False)
        return transformed


def load_saved_norm_vals(run_dir):
    """The background / auxiliary normalisation (mean, std) the run trained with."""
    files = sorted(Path(run_dir).glob("training_outputs/norm_vals_*.json"))
    if not files:
        raise FileNotFoundError(f"no norm_vals_*.json in {run_dir}/training_outputs")
    with open(files[0]) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


def filter_state_for_model(state, model, allowed=GRID_TIED_STATE_KEYS):
    """Drop the entries of ``state`` whose shape differs from the model's; only ``allowed`` keys may.

    Returns ``(filtered_state, dropped_keys)``. Raises if any other entry mismatches, so an
    architecture that does not match the checkpoint is never loaded partially by accident.
    """
    model_state = model.state_dict()
    dropped, bad = [], []
    for key, tensor in state.items():
        if key in model_state and tuple(model_state[key].shape) != tuple(tensor.shape):
            (dropped if key in allowed else bad).append((key, tuple(tensor.shape), tuple(model_state[key].shape)))
    if bad:
        raise RuntimeError("checkpoint / model shape mismatch beyond the grid-tied entries: "
                           + ", ".join(f"{k} {a} vs {b}" for k, a, b in bad))
    filtered = {k: v for k, v in state.items() if k not in {d[0] for d in dropped}}
    return filtered, [d[0] for d in dropped]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", action="append", required=True, help="repeatable; a trained run's directory")
    ap.add_argument("--checkpoint", default="best_fp", help="checkpoint suffix: best_fp, best_bg, best, or an epoch number")
    ap.add_argument("--out", default=None, help="output file name inside each run dir "
                                                "(default: sample_predictions_test_<checkpoint>.nc)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=JSON",
                    help="repeatable; change a saved setting before loading, e.g. train_load_data.size=100")
    ap.add_argument("--use-saved-scalers", action="store_true",
                    help="apply the run's saved input scaler and background normalisation instead of "
                         "refitting them on the loaded training set")
    ap.add_argument("--allow-grid-mismatch", action="store_true",
                    help="build the model for the loaded window size and drop the grid-tied state entries "
                         "(encoder.h3_nodes, bg head linear); the background output is then invalid (NaN)")
    args = ap.parse_args()
    out_name = args.out or f"sample_predictions_test_{args.checkpoint}.nc"

    run_dirs = [Path(r) for r in args.run_dir]
    params = [load_run_parameters(r) for r in run_dirs]
    saved_scalers = [restore_scaler_names(p, r) for p, r in zip(params, run_dirs)]
    check_runs_share_inputs(run_dirs, params)
    checkpoints = [find_checkpoint(r, args.checkpoint) for r in run_dirs]
    print("Regenerating test predictions for:")
    for r, c in zip(run_dirs, checkpoints):
        print(f"  {r.name}  <-  {c.name}")

    for p in params:
        changes = apply_overrides(p, args.override)
    for key, old, new in changes:
        print(f"override {key}: {old!r} -> {new!r}")
    if args.override and not args.use_saved_scalers:
        print("WARNING: settings overridden but the scalers are REFITTED on the overridden training set")

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
    # Runs trained from the month cache ("data_cache" block): the training inputs are not held in
    # memory, so the input scaler cannot be refitted here; the run's saved one is applied instead.
    from_cache = isinstance(train_inputs, gates_lean.LazyMonthInputs)
    if from_cache and not args.use_saved_scalers:
        print("data_cache run: applying the run's SAVED input scaler and background normalisation "
              "(the training inputs are not loaded)")
        args.use_saved_scalers = True
    if args.use_saved_scalers:
        saved_norm = load_saved_norm_vals(run_dirs[0])
        print(f"using the saved background / auxiliary normalisation of {run_dirs[0].name}: {saved_norm}")
        norm_train_bgs, norm_train_aux, norm_vals = normalize_boundary_data(train_bgs, aux_data=train_aux, norm_vals=saved_norm)
        norm_test_bgs, norm_test_aux, norm_vals = normalize_boundary_data(test_bgs, aux_data=test_aux, norm_vals=saved_norm)
        input_dataset = SavedInputScaler(saved_scalers[0]["inputs_scaler"])
    else:
        norm_train_bgs, norm_train_aux, norm_vals = normalize_boundary_data(train_bgs, aux_data=train_aux)
        norm_test_bgs, norm_test_aux, norm_vals = normalize_boundary_data(test_bgs, aux_data=test_aux, norm_vals=norm_vals)
        input_dataset = None

    feature_dim = train_inputs.shape[-1]
    aux_dim = train_aux.aux.shape[0] if background_params["use_auxiliary_bc"] else 0
    base["num_features"] = feature_dim + aux_dim

    if from_cache:
        fp_dataset = gates_training.setup_fp_dataset(base, train_fp_data)
        test_loader, fp_labels, test_scaled_fp = gates_training_dual.setup_dual_eval_loader(
            base, input_dataset, fp_dataset, test_inputs, test_fp_data, norm_test_bgs, norm_test_aux)
        scalers = {"inputs_scaler": input_dataset.scaler, "fp_scaler": fp_dataset.scaler}
    else:
        train_loader, test_loader, fp_labels, test_scaled_fp, scalers = gates_training_dual.setup_dual_dataloaders(
            base, train_inputs, train_fp_data, norm_train_bgs,
            test_inputs, test_fp_data, norm_test_bgs, norm_train_aux, norm_test_aux,
            input_dataset=input_dataset)
        del train_loader
    check_refitted_scalers(scalers, saved_scalers[0], run_dirs[0])
    # larger input domain (train_load_data.input_domain) in mode "full": the model's grid is the input
    # window, the footprint is scored and written on its own central window, as during training
    test_scaled_fp = gates_input_domain.crop_fp_dataset(
        test_scaled_fp, gates_input_domain.resolve_fp_window(base, train_fp_data))
    # node order of the run(s): "grid_node_order" of the saved settings (absent = "legacy")
    grid, _ = get_model_grid(train_fp_data, base)
    window = len(train_fp_data.lat.values)
    print(f"window {window} x {len(train_fp_data.lon.values)} cells, grid of {len(grid)} nodes, "
          f"{len(test_fp_data.time)} test footprints, scored on {test_scaled_fp.sizes['lat']} x "
          f"{test_scaled_fp.sizes['lon']} cells")
    training_ctx = BoundaryTrainingContext(
        base, device, False, [], [], grid, fp_labels, scalers, feature_dim,
        len(train_fp_data.lat.values), aux_dim)

    for run_dir, ckpt, p in zip(run_dirs, checkpoints, params):
        print(f"\n=== {run_dir.name} ({ckpt.name}) ===")
        paths_ctx = PathContext(model_save_dir=str(run_dir.parent), model_name=run_dir.name, model_path=run_dir)
        p_infer = copy.deepcopy(p)
        p_infer["num_features"] = base["num_features"]
        p_infer["grid_node_order_applied"] = base["grid_node_order_applied"]  # same for all runs (checked)
        if "input_domain_resolved" in base:
            p_infer["input_domain_resolved"] = base["input_domain_resolved"]  # same data for all runs (checked)
        model, model_ctx = build_model_and_context(p_infer, training_ctx, paths_ctx)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        state = state.get("model_state_dict", state) if isinstance(state, dict) and "model_state_dict" in state else state
        dropped = []
        if args.allow_grid_mismatch:
            state, dropped = filter_state_for_model(state, model)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or set(missing) != set(dropped):
                raise RuntimeError(f"unexpected partial load: missing {missing}, unexpected {unexpected}, dropped {dropped}")
            print(f"grid-mismatch load: dropped {dropped} (checkpoint trained on another window size)")
        else:
            model.load_state_dict(state, strict=True)
        bg_valid = "bg_decoder.linear_class.weight" not in dropped
        if not bg_valid:
            print("WARNING: the background head's final layer is untrained for this window -> bg output written as NaN")
        model.to(device)

        avg_total, avg_fp, avg_bg, bg_mae, fp_out, bg_out, bg_true = validate_and_predict(
            model, test_loader, model_ctx, output_norm=norm_vals["outputs"])
        print(f"test losses from this checkpoint: total {avg_total:.4f}, fp {avg_fp:.4f}, "
              f"bg {avg_bg:.4f}, bg MAE {bg_mae:.4g}")

        ds = test_scaled_fp.copy(deep=True)
        fp_out = gates_input_domain.crop_fp_predictions(fp_out, base)  # mode "full" only (no-op otherwise)
        original_space = scalers["fp_scaler"].inverse_transform(fp_out)
        ds["fp_transformed_pred"] = (("time", "lat", "lon"), fp_out.reshape(*ds.fp_original.shape))
        ds["fp_pred"] = (("time", "lat", "lon"), original_space.reshape(*ds.fp_original.shape))
        mean, std = norm_vals["outputs"]
        ds["bg_true_ppb"] = (("time",), denormalize(np.asarray(bg_true), mean, std).reshape(-1) * 1e9)
        bg_pred_ppb = denormalize(np.asarray(bg_out), mean, std).reshape(-1) * 1e9
        ds["bg_pred_ppb"] = (("time",), bg_pred_ppb if bg_valid else np.full_like(bg_pred_ppb, np.nan))
        ds.attrs.update({"model_name": run_dir.name, "checkpoint": ckpt.name,
                         "test_fp_loss": float(avg_fp), "test_bg_loss": float(avg_bg) if bg_valid else float("nan"),
                         "window_size": int(ds.sizes["lat"]), "input_window_size": int(window),
                         "overrides": json.dumps(args.override),
                         "scalers": "saved" if args.use_saved_scalers else "refitted",
                         "dropped_state": json.dumps(dropped)})
        out_path = run_dir / out_name
        ds.to_netcdf(out_path)
        print("wrote", out_path)
        del model, model_ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
