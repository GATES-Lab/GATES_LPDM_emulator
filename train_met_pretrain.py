"""Self-supervised pretraining of the GATES trunk on meteorology (no footprints, no backgrounds).

Pretext tasks (``gates/training/pretraining.py``): ``"masked"`` block-masked reconstruction and
``"next_delta"`` prediction of the time_delta-0 met from the older time deltas. The inputs are
built exactly like the dual model's (same loader, same input scaler class, aux CAMS channels
appended and normalised the same way), so the saved trunk loads straight into
``train_dual_headlr_model.py`` via its ``pretrained_trunk`` block.

The parameter file is a normal dual parameter file plus a ``pretrain`` block::

    "pretrain": {
        "output_dir": "met_pretrain",          # under save_models_dir; one sub-dir per run
        "scaler_fit_years": ["2014", "2015"],  # input scaler + aux normalisation are fit on
                                               # these years ONLY = the fine-tuning train years,
                                               # so pretrain and fine-tune inputs are scaled alike
        "epochs": 100, "batch_size": 20, "learning_rate": 2e-4, "min_lr_fraction": 0.1,
        "block_size": 10, "mask_ratio": 0.5,   # "masked" task only
        "no_cams_years": ["2012"],             # optional: met years WITHOUT CAMS files, loaded with
                                               # the footprint-only loader; their aux channels are 0
        "runs": [
            {"name": "masked_2yr", "task": "masked", "train_years": ["2014", "2015"]},
            {"name": "nextdelta_4yr", "task": "next_delta", "train_years": ["2013","2014","2015","2017"]}
        ]
    }

``train_load_data.years`` must cover every year used by any run; the data is loaded ONCE and the
runs execute sequentially. ``test_load_data`` (the fine-tuning test year) is used ONLY to report
the pretext validation loss and pick ``trunk_best.pt`` — its footprints/backgrounds are never
touched here, and it is never trained on (principle 3).

Outputs per run, in ``<save_models_dir>/<output_dir>/<run name>/``: ``trunk_best.pt`` (lowest
validation pretext loss), ``trunk_last.pt``, ``pretrain_settings.json`` (resolved parameters +
channel groups), ``pretrain_log.json`` (per-epoch losses) and ``updates.txt``. An existing
``trunk_best.pt`` is never overwritten (the run is refused; principle 7).

Usage:  python train_met_pretrain.py parameter_met_pretrain_2014-15_test2016.json [--index i]
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

import gates
import gates.training.training as gates_training
import gates.training.pretraining as pretraining
from gates.data.load_data import get_grid
from gates.training.training_background import format_aux_data, normalize_data
from gates.training.training_helperfuns import (
    load_parameter_file, set_reproducibility, enable_deterministic_algorithms,
)
from train_dual_model import load_dual_data, resolve_background_params
from gates.training.training_dataclasses import PathContext


def load_no_cams_years(parameters, years):
    """Load met inputs (+ footprints, for their times / grid only) for years that have NO CAMS
    boundary files, through the footprint-only yearly loader. Their aux CAMS channels are
    zero-filled later (= the normalised aux mean). Returns (fp_xr, inputs)."""
    base = copy.deepcopy(parameters["train_load_data"])
    for k in ("year", "years", "month", "months"):
        base.pop(k, None)
    months = parameters["train_load_data"].get("months") or [f"{m:02d}" for m in range(1, 13)]
    datapath_args = PathContext(model_save_dir=".", model_name="_dataload", model_path=Path(".")).resolve_datapath_args(parameters)
    client, cluster = gates_training.make_cluster()
    fps, inputs = [], []
    try:
        # One month at a time (like load_GATES_data_with_bg): a whole year of met in one pass
        # needs >100 GB on top of the CAMS-year bundle (job 6660259 was OOM-killed at 300 GB).
        for year in years:
            for month in months:
                data_params = {**base, "year": str(year), "month": str(month)}
                try:
                    fp_m, in_m = gates_training.load_GATES_data_v2(
                        data_params, input_variables=parameters["variables"], datapath_args=datapath_args,
                        verbose=parameters.get("verbose", True), load_into_memory=True)
                except Exception as e:   # same per-month tolerance as the CAMS loader
                    print(f"Error loading no-CAMS data for {year}-{month}: {e}")
                    continue
                fps.append(fp_m.load()); inputs.append(in_m.load())
                print(f"no-CAMS {year}-{month}: {in_m.sizes['fp_time']} samples")
    finally:
        if cluster is not None:
            cluster.close(); client.close()
    import xarray as xr
    fp_xr = xr.concat(fps, dim="time").sortby("time")
    inputs = xr.concat(inputs, dim="fp_time").sortby("fp_time")
    return fp_xr, inputs


def _to_tensor(scaled_inputs, aux_norm):
    """(fp_time, lat, lon, variable_name) [+ (time, aux)] -> float32 tensor [N, H*W, F(+aux)].

    lat-major flattening = ``stack(flat_lat_lon=["lat", "lon"])`` in ``make_inputs_batcher``,
    and the aux channels are appended last and broadcast over the grid exactly like
    ``concat_auxiliary_to_inputs``.
    """
    arr = scaled_inputs.transpose("fp_time", "lat", "lon", "variable_name").values.astype(np.float32, copy=False)
    n, h, w, f = arr.shape
    n_aux = 0 if aux_norm is None else aux_norm.sizes["aux"]
    out = torch.empty((n, h * w, f + n_aux), dtype=torch.float32)
    out[:, :, :f] = torch.from_numpy(arr.reshape(n, h * w, f))
    if n_aux:
        aux = torch.from_numpy(aux_norm.transpose("time", "aux").values.astype(np.float32))
        out[:, :, f:] = aux[:, None, :]
    return out, (h, w)


def prepare_tensors(parameters, bundle, extra=None):
    """Scale inputs / normalise aux (fit on ``pretrain.scaler_fit_years`` only) and return
    in-memory tensors for train and validation plus the channel metadata.

    ``extra`` = optional ``(fp_xr, inputs)`` from :func:`load_no_cams_years`; those samples are
    appended to the training tensor with their aux channels set to 0."""
    pre = parameters["pretrain"]
    (train_fp, train_inputs, _train_bgs, train_aux, test_fp, test_inputs, _test_bgs, test_aux) = bundle
    background_params = resolve_background_params(parameters)
    use_aux = background_params["use_auxiliary_bc"]
    n_no_cams = 0
    if extra is not None:
        extra_fp, extra_inputs = extra
        if [tuple(v) for v in extra_inputs.variable_name.values] != [tuple(v) for v in train_inputs.variable_name.values]:
            raise ValueError("no_cams_years inputs have a different channel layout than the CAMS years")
        n_no_cams = extra_inputs.sizes["fp_time"]

    years = train_inputs.fp_time.dt.year.values          # CAMS years (the bundle) only
    fit_years = [int(y) for y in pre["scaler_fit_years"]]
    fit_sel = np.isin(years, fit_years)                  # indexes the bundle arrays
    all_years = years if extra is None else np.concatenate([years, extra_inputs.fp_time.dt.year.values])
    if not fit_sel.any():
        raise ValueError(f"no training samples in scaler_fit_years {fit_years}")
    print(f"Fitting the input scaler on years {fit_years}: {int(fit_sel.sum())} of {len(years)} samples")

    # setup_input_dataset pops keys from parameters["input_scaler"]: give it a copy.
    input_dataset = gates_training.setup_input_dataset(copy.deepcopy(parameters), train_inputs.isel(fp_time=fit_sel))
    train_scaled = input_dataset.transform(train_inputs)
    test_scaled = input_dataset.transform(test_inputs)
    extra_scaled = input_dataset.transform(extra_inputs) if extra is not None else None
    variable_names = [tuple(v) for v in train_scaled.variable_name.values]

    train_aux_n = test_aux_n = None
    aux_norm_vals = None
    if use_aux:
        train_aux = format_aux_data(train_aux, time_coord=train_fp.time)
        test_aux = format_aux_data(test_aux, time_coord=test_fp.time)
        _, aux_norm_vals = normalize_data(train_aux.isel(time=fit_sel))
        train_aux_n, _ = normalize_data(train_aux, norm_vals=aux_norm_vals)
        test_aux_n, _ = normalize_data(test_aux, norm_vals=aux_norm_vals)

    x_train, (h, w) = _to_tensor(train_scaled, train_aux_n)
    x_val, _ = _to_tensor(test_scaled, test_aux_n)
    if extra is not None:
        x_extra, _ = _to_tensor(extra_scaled, None)
        if use_aux:   # aux channels absent for these years: zero = normalised mean
            x_extra = torch.cat([x_extra, torch.zeros(x_extra.shape[0], x_extra.shape[1], x_train.shape[-1] - x_extra.shape[-1])], dim=-1)
        x_train = torch.cat([x_train, x_extra], dim=0)
        del x_extra
        print(f"appended {n_no_cams} samples from no-CAMS years with zero-filled aux channels")
    for name, t in (("train", x_train), ("val", x_val)):
        n_bad = int((~torch.isfinite(t)).sum())
        if n_bad:
            raise ValueError(f"{n_bad} non-finite values in the {name} pretraining inputs")
    feature_dim = len(variable_names)
    aux_dim = x_train.shape[-1] - feature_dim
    # Build the grid from the scaler_fit_years subset (= the fine-tuning training set), so the
    # reference footprint — and hence the mesh — is the one the fine-tuning run will build.
    grid, _ = get_grid(train_fp.isel(time=fit_sel), parameters.get("grid_reference_fp"))
    info = dict(variable_names=variable_names, years=all_years, feature_dim=feature_dim, aux_dim=aux_dim,
                height=h, width=w, grid=grid, aux_norm_vals=aux_norm_vals, n_no_cams=n_no_cams)
    return x_train, x_val, info


@torch.no_grad()
def evaluate(model, task, x_val, batch_size, device, seed):
    """Validation pretext loss with FIXED masks (seeded), the trivial-baseline loss on the same
    cells, and the per-variable model MSE."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    sums = {"model": 0.0, "baseline": 0.0}
    per_channel = torch.zeros(task.num_targets, device=device, dtype=torch.float64)
    per_channel_base = torch.zeros_like(per_channel)
    n_cells = 0
    for start in range(0, x_val.shape[0], batch_size):
        x = x_val[start:start + batch_size].to(device, non_blocking=True)
        x_c, target, cell_mask = task.corrupt(x, gen)
        pred = pretraining.pretrain_forward(model, x_c)
        base = task.baseline_prediction(x)
        d_model = ((pred - target) ** 2)[cell_mask]
        d_base = ((base - target) ** 2)[cell_mask]
        sums["model"] += float(d_model.sum())
        sums["baseline"] += float(d_base.sum())
        per_channel += d_model.sum(dim=0).double()
        per_channel_base += d_base.sum(dim=0).double()
        n_cells += int(cell_mask.sum())
    denom = n_cells * task.num_targets
    return (sums["model"] / denom, sums["baseline"] / denom, (per_channel / n_cells).cpu().numpy(),
            (per_channel_base / n_cells).cpu().numpy())


def run_pretraining(parameters, run_cfg, x_train_all, x_val, info, out_root):
    pre = parameters["pretrain"]
    name, task_name = run_cfg["name"], run_cfg["task"]
    train_years = [int(y) for y in run_cfg["train_years"]]
    out_dir = out_root / name
    if (out_dir / "trunk_best.pt").exists():
        raise FileExistsError(f"{out_dir / 'trunk_best.pt'} already exists; refusing to overwrite "
                              "(choose a new run name or pretrain.output_dir)")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = parameters.get("seed") or 34
    set_reproducibility(seed)
    enable_deterministic_algorithms(parameters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sel = np.isin(info["years"], train_years)
    missing = [y for y in train_years if y not in set(info["years"].tolist())]
    if missing:
        raise ValueError(f"run {name}: no loaded samples for years {missing}")
    train_idx = torch.from_numpy(np.nonzero(sel)[0])
    print(f"[{name}] task={task_name} years={train_years}: {len(train_idx)} train / {x_val.shape[0]} val samples")

    dynamic_vars = list(parameters["variables"]["met_variables"]) + ["wind_speed", "wind_angle"]
    groups = pretraining.channel_groups(info["variable_names"], dynamic_vars)
    task = pretraining.PretextTask(task_name, groups, info["height"], info["width"],
                                   block_size=pre.get("block_size", 10),
                                   mask_ratio=pre.get("mask_ratio", 0.5)).to(device)
    model = pretraining.build_pretrain_model(parameters, info["grid"], info["feature_dim"],
                                             info["aux_dim"], info["height"], task.num_targets).to(device)

    epochs = int(pre.get("epochs", 100))
    batch_size = int(pre.get("batch_size", 20))
    lr = float(pre.get("learning_rate", 2e-4))
    min_frac = float(pre.get("min_lr_fraction", 0.1))
    # the bg decoder is unused here: keep it out of the optimizer
    params = [p for n, p in model.named_parameters() if not n.startswith("bg_decoder.")]
    optimizer = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = len(train_idx) // batch_size
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    settings = copy.deepcopy(parameters)
    settings["pretrain_run"] = dict(run_cfg, n_train=int(len(train_idx)), n_val=int(x_val.shape[0]),
                                    n_no_cams_samples_loaded=info["n_no_cams"],
                                    num_targets=task.num_targets, feature_dim=info["feature_dim"],
                                    aux_dim=info["aux_dim"], aux_norm_vals=info["aux_norm_vals"],
                                    seed=seed, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    settings["channel_groups"] = groups
    with open(out_dir / "pretrain_settings.json", "w") as f:
        json.dump(settings, f, indent=1, default=str)

    use_wandb = parameters.get("use_wandb", False)
    if use_wandb:
        wcfg = parameters.get("wandb", {})
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        wandb.init(entity=wcfg.get("entity"), project=wcfg.get("project"), group=wcfg.get("group"),
                   tags=list(wcfg.get("tags", [])) + ["met_pretrain", task_name],
                   name=f"{job_id}_pretrain_{name}", job_type="met_pretrain",
                   notes=run_cfg.get("notes"), config=settings)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)
    log, best = [], {"val": float("inf"), "epoch": None}
    target_names = [groups["names"][i] for i in task.target_idx.tolist()]
    updates = open(out_dir / "updates.txt", "a")

    def meta(epoch, val):
        return dict(task=task_name, run_name=name, train_years=train_years, epoch=epoch, val_loss=val,
                    feature_dim=info["feature_dim"], aux_dim=info["aux_dim"],
                    model_parameters=parameters["model_parameters"], seed=seed,
                    scaler_fit_years=pre["scaler_fit_years"], slurm_job_id=os.environ.get("SLURM_JOB_ID"))

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=shuffle_gen)]
        run_loss = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            x = x_train_all[idx].to(device, non_blocking=True)
            x_c, target, cell_mask = task.corrupt(x, gen)
            pred = pretraining.pretrain_forward(model, x_c)
            loss = task.loss(pred, target, cell_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += float(loss)
        train_loss = run_loss / steps_per_epoch
        val_loss, base_loss, per_channel, per_channel_base = evaluate(model, task, x_val, batch_size, device, seed=seed + 1)
        secs = time.perf_counter() - t0
        rec = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_baseline_loss=base_loss,
                   lr=scheduler.get_last_lr()[0], epoch_seconds=secs)
        log.append(rec)
        line = (f"epoch {epoch}: train {train_loss:.5f} | val {val_loss:.5f} | "
                f"baseline {base_loss:.5f} | skill {1 - val_loss / base_loss:.3f} | {secs:.0f}s")
        print(f"[{name}] {line}", flush=True)
        updates.write(line + "\n"); updates.flush()
        if use_wandb:
            wandb.log({"pretrain/train_loss": train_loss, "pretrain/val_loss": val_loss,
                       "pretrain/val_baseline_loss": base_loss,
                       "pretrain/val_skill_vs_baseline": 1 - val_loss / base_loss,
                       "pretrain/lr": rec["lr"], "epoch": epoch})
        if val_loss < best["val"]:
            best = {"val": val_loss, "epoch": epoch}
            pretraining.save_trunk(model, out_dir / "trunk_best.pt", meta(epoch, val_loss))
            best_per_channel = per_channel

    pretraining.save_trunk(model, out_dir / "trunk_last.pt", meta(epochs - 1, log[-1]["val_loss"]))
    per_var = {}
    for n, v in zip(target_names, best_per_channel):
        per_var.setdefault(n.split("|")[0], []).append(float(v))
    per_var = {k: float(np.mean(v)) for k, v in per_var.items()}
    per_var_base = {}
    for n, v in zip(target_names, per_channel_base):  # baseline does not depend on the epoch
        per_var_base.setdefault(n.split("|")[0], []).append(float(v))
    per_var_base = {k: float(np.mean(v)) for k, v in per_var_base.items()}
    with open(out_dir / "pretrain_log.json", "w") as f:
        json.dump({"epochs": log, "best": best, "best_val_mse_per_variable": per_var,
                   "baseline_val_mse_per_variable": per_var_base}, f, indent=1)
    updates.write(f"best val {best['val']:.5f} @ {best['epoch']}; per-variable val MSE at best: {per_var}; baseline: {per_var_base}\n")
    updates.close()
    if use_wandb:
        wandb.run.summary.update({"best/val_loss": best["val"], "best/epoch": best["epoch"],
                                  **{f"best/val_mse/{k}": v for k, v in per_var.items()}})
        wandb.finish()
    print(f"[{name}] done: best val {best['val']:.5f} @ epoch {best['epoch']} -> {out_dir / 'trunk_best.pt'}")
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Self-supervised met pretraining of the GATES trunk.")
    parser.add_argument("file_name")
    parser.add_argument("--file_path", default=None)
    parser.add_argument("--index", type=int, default=None, help="run only pretrain.runs[index]")
    args = parser.parse_args()

    cfg = gates.config.get_config()
    file_path = args.file_path if args.file_path is not None else cfg.parameter_files_dir
    parameters = load_parameter_file(Path(file_path) / args.file_name)
    if parameters is None:
        sys.exit(1)
    pre = parameters["pretrain"]
    runs = pre["runs"] if args.index is None else [pre["runs"][args.index]]
    for r in runs:
        if r["task"] not in pretraining.PRETEXT_TASKS:
            raise ValueError(f"run {r.get('name')}: unknown task {r['task']}")
    no_cams_years = [str(y) for y in pre.get("no_cams_years", [])]
    loaded_years = {str(y) for y in parameters["train_load_data"]["years"]} | set(no_cams_years)
    if set(no_cams_years) & {str(y) for y in parameters["train_load_data"]["years"]}:
        raise ValueError("pretrain.no_cams_years must not repeat train_load_data.years")
    needed = {str(y) for r in runs for y in r["train_years"]} | {str(y) for y in pre["scaler_fit_years"]}
    if not needed <= loaded_years:
        raise ValueError(f"train_load_data.years {sorted(loaded_years)} must include {sorted(needed)}")
    test_years = {str(y) for y in parameters["test_load_data"]["years"]}
    if needed & test_years:
        raise ValueError(f"pretraining years overlap the held-out test years: {sorted(needed & test_years)}")

    save_dir = Path(parameters.get("model_save_dir") or cfg.save_models_dir)
    out_root = save_dir / pre.get("output_dir", "met_pretrain")
    for r in runs:  # fail before the expensive load
        if (out_root / r["name"] / "trunk_best.pt").exists():
            raise FileExistsError(f"{out_root / r['name'] / 'trunk_best.pt'} exists; refusing to overwrite")

    if parameters.get("use_wandb", False):
        wandb.login()
    bundle = load_dual_data(parameters, verbose=parameters.get("verbose", True))
    extra = load_no_cams_years(parameters, no_cams_years) if no_cams_years else None
    x_train, x_val, info = prepare_tensors(parameters, bundle, extra)
    del bundle, extra
    print(f"pretraining tensors: train {tuple(x_train.shape)}, val {tuple(x_val.shape)}")
    if torch.cuda.is_available():
        x_train, x_val = x_train.pin_memory(), x_val.pin_memory()
    for r in runs:
        run_pretraining(parameters, r, x_train, x_val, info, out_root)
    print("All pretraining runs finished.")


if __name__ == "__main__":
    main()
