"""Self-supervised pretraining of the GATES trunk on meteorology (no footprints, no backgrounds).

Pretext tasks (``gates/training/pretraining.py``): ``"masked"`` block-masked reconstruction,
``"next_delta"`` / ``"next_delta_tendency"`` prediction of the time_delta-0 met from the older
time deltas, and the general ``"delta_forecast"`` (any source time deltas -> any target delta,
e.g. the BACKWARD forecast t, t-6 h -> t-12 h that mirrors what a footprint describes). The
inputs are built exactly like the dual model's (same loader, same input scaler class, aux CAMS
channels appended and normalised the same way), so the saved trunk loads straight into
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
        "extra_time_deltas": [24],             # optional: extra met time deltas (hours before the
                                               # footprint time) loaded as TARGET-ONLY channels for
                                               # delta_forecast; never fed to the model, so the trunk's
                                               # input layout stays that of ``variables.time_deltas``
        "runs": [
            {"name": "masked_2yr", "task": "masked", "train_years": ["2014", "2015"]},
            {"name": "nextdelta_4yr", "task": "next_delta", "train_years": ["2013","2014","2015","2017"]},
            {"name": "backward6h_4yr", "task": "delta_forecast", "train_years": ["2013","2014","2015","2017"],
             "source_deltas": [0, 6], "target_delta": 12, "tendency": false}
        ]
    }

``pseudo_footprint`` runs (``gates/training/pseudo_footprint.py``) predict a one-channel kinematic
pseudo-footprint per node computed from the window's boundary-layer winds; their settings come from
``pretrain.pseudo_footprint`` (``hours``, ``step_hours``, ``levels``, ``particles``, ``diffusion_m2s``,
``minimum_oom``, ``seed``), optionally overridden per run by a ``pseudo_footprint`` dict. ``hours`` needs
met up to that many hours before the footprint time, so e.g. ``extra_time_deltas: [18, 24]`` for 24 h.
``pretrain.keep_extra_channels: false`` drops the extra deltas from the tensors once the targets are
computed (saves memory when no ``delta_forecast`` run needs them). ``save_modules`` (file level, or
per run) may add ``"fp_decoder"`` to the saved ``encoder`` + ``processor`` when the run's head is one
channel per node (``pseudo_footprint``), so the footprint head can be warm-started too.

``delta_forecast`` runs need ``source_deltas`` (time deltas the model may see; the other dynamic
channels are zeroed) and ``target_delta`` (must be one of ``variables.time_deltas``, 0, or
``pretrain.extra_time_deltas``); ``tendency`` (default false) predicts the change relative to
the nearest source delta instead. With ``extra_time_deltas``, footprints whose shifted met time
falls outside the loaded met record (the first day of each year) are dropped for ALL channels,
so the sample set is slightly smaller than in a plain load — noted in ``pretrain_settings.json``.

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
import gates.training.pseudo_footprint as pseudo_fp
from gates.data.load_data import get_grid
from gates.training.training_background import format_aux_data, normalize_data
from gates.training.training_helperfuns import (
    load_parameter_file, set_reproducibility, enable_deterministic_algorithms,
)
from train_dual_model import load_dual_data, resolve_background_params
from gates.training.training_dataclasses import PathContext


def dual_time_deltas(parameters):
    """Time deltas of the dual model's input layout: 0 plus ``variables.time_deltas``."""
    return sorted({0} | {int(d) for d in parameters["variables"].get("time_deltas", [])})


def loading_parameters(parameters):
    """Copy of ``parameters`` whose ``variables.time_deltas`` also contains
    ``pretrain.extra_time_deltas`` (the target-only deltas), for the data loaders only."""
    extra = [int(d) for d in parameters["pretrain"].get("extra_time_deltas", [])]
    if not extra:
        return parameters
    overlap = set(extra) & set(dual_time_deltas(parameters))
    if overlap:
        raise ValueError(f"pretrain.extra_time_deltas {sorted(overlap)} already belong to variables.time_deltas")
    out = copy.deepcopy(parameters)
    out["variables"]["time_deltas"] = sorted({int(d) for d in parameters["variables"].get("time_deltas", [])} | set(extra))
    return out


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


def _to_tensor(scaled_inputs, aux_norm, input_mask=None, aux_dim=None):
    """(fp_time, lat, lon, variable_name) [+ (time, aux)] -> float32 tensor
    ``[N, H*W, F_input (+aux) (+extra)]``.

    lat-major flattening = ``stack(flat_lat_lon=["lat", "lon"])`` in ``make_inputs_batcher``,
    and the aux channels are appended after the model's met channels and broadcast over the
    grid exactly like ``concat_auxiliary_to_inputs``. Channels with ``input_mask == False``
    (target-only time deltas) go LAST, after the aux channels, so the leading
    ``F_input + aux`` columns are exactly the dual model's input. ``aux_dim`` zero-fills that
    many aux columns when ``aux_norm`` is None (no-CAMS years).
    """
    arr = scaled_inputs.transpose("fp_time", "lat", "lon", "variable_name").values.astype(np.float32, copy=False)
    n, h, w, f_all = arr.shape
    if input_mask is None:
        input_mask = np.ones(f_all, dtype=bool)
    n_in, n_extra = int(input_mask.sum()), int((~input_mask).sum())
    n_aux = (0 if aux_dim is None else int(aux_dim)) if aux_norm is None else aux_norm.sizes["aux"]
    out = torch.zeros((n, h * w, n_in + n_aux + n_extra), dtype=torch.float32)
    flat = torch.from_numpy(arr.reshape(n, h * w, f_all))
    if n_extra == 0:
        out[:, :, :n_in] = flat
    else:   # column by column: avoids a second full-size temporary from fancy indexing
        for j, col in enumerate(np.nonzero(input_mask)[0]):
            out[:, :, j] = flat[:, :, col]
        for j, col in enumerate(np.nonzero(~input_mask)[0]):
            out[:, :, n_in + n_aux + j] = flat[:, :, col]
    if aux_norm is not None:
        aux = torch.from_numpy(aux_norm.transpose("time", "aux").values.astype(np.float32))
        out[:, :, n_in:n_in + n_aux] = aux[:, None, :]
    return out, (h, w)


def prepare_tensors(parameters, bundle, extra=None, pseudo_fp_configs=()):
    """Scale inputs / normalise aux (fit on ``pretrain.scaler_fit_years`` only) and return
    in-memory tensors for train and validation plus the channel metadata.

    ``extra`` = optional ``(fp_xr, inputs)`` from :func:`load_no_cams_years`; those samples are
    appended to the training tensor with their aux channels set to 0. ``pseudo_fp_configs`` =
    resolved pseudo-footprint configs (one per distinct run setting); their targets are computed
    from the RAW winds here (before scaling) and returned in ``info["pseudo_fp"]``."""
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

    # Split the loaded channels into the dual model's input layout and the target-only extra
    # time deltas (pretrain.extra_time_deltas). The scaler is fit on the input layout ONLY, exactly
    # as the fine-tuning run fits it; the extra channels reuse the scaler of the same (variable,
    # level) (DefaultInputsScaler standardises per variable and level across time deltas anyway).
    layout_deltas = set(dual_time_deltas(parameters))
    all_names = list(train_inputs.variable_name.values)
    if [tuple(v) for v in test_inputs.variable_name.values] != [tuple(v) for v in all_names]:
        raise ValueError("train and test inputs have different channel layouts")
    input_mask = np.array([int(v[2]) in layout_deltas for v in all_names])
    extra_names = [v for v, keep in zip(all_names, input_mask) if not keep]

    # Pseudo-footprint targets from the RAW winds (m/s) and raw window coordinates, in the same
    # sample order as the tensors below (CAMS years, then no-CAMS years).
    pseudo = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cfg in pseudo_fp_configs:
        key = pseudo_fp.config_key(cfg)
        print(f"Computing pseudo-footprint targets for {cfg} ...")
        t0 = time.perf_counter()
        y_tr, ex_tr, st_tr = pseudo_fp.targets_from_inputs(train_inputs, cfg, device=device)
        if extra is not None:
            y_ex, ex_ex, _ = pseudo_fp.targets_from_inputs(extra_inputs, cfg, device=device)
            y_tr, ex_tr = torch.cat([y_tr, y_ex]), torch.cat([ex_tr, ex_ex])
        y_va, ex_va, st_va = pseudo_fp.targets_from_inputs(test_inputs, cfg, device=device)
        pseudo[key] = dict(config=cfg, y_train=y_tr, y_val=y_va, exits_train=ex_tr, exits_val=ex_va,
                           stats_train=st_tr, stats_val=st_va)
        print(f"  done in {time.perf_counter() - t0:.0f}s; train stats {st_tr}; val stats {st_va}")
    # setup_input_dataset pops keys from parameters["input_scaler"]: give it a copy.
    input_dataset = gates_training.setup_input_dataset(
        copy.deepcopy(parameters), train_inputs.isel(fp_time=fit_sel, variable_name=input_mask))
    if extra_names:
        alias_extra_channel_scalers(input_dataset.scaler, extra_names)
    train_scaled = input_dataset.transform(train_inputs)
    test_scaled = input_dataset.transform(test_inputs)
    extra_scaled = input_dataset.transform(extra_inputs) if extra is not None else None
    if extra_names and not pre.get("keep_extra_channels", True):
        print(f"dropping the {len(extra_names)} target-only channels from the tensors (keep_extra_channels: false)")
        train_scaled = train_scaled.isel(variable_name=input_mask)
        test_scaled = test_scaled.isel(variable_name=input_mask)
        extra_scaled = extra_scaled.isel(variable_name=input_mask) if extra_scaled is not None else None
        input_mask = np.ones(int(input_mask.sum()), dtype=bool)
        extra_names = []
    variable_names = [tuple(v) for v, keep in zip(train_scaled.variable_name.values, input_mask) if keep]
    extra_variable_names = [tuple(v) for v, keep in zip(train_scaled.variable_name.values, input_mask) if not keep]

    train_aux_n = test_aux_n = None
    aux_norm_vals = None
    if use_aux:
        train_aux = format_aux_data(train_aux, time_coord=train_fp.time)
        test_aux = format_aux_data(test_aux, time_coord=test_fp.time)
        _, aux_norm_vals = normalize_data(train_aux.isel(time=fit_sel))
        train_aux_n, _ = normalize_data(train_aux, norm_vals=aux_norm_vals)
        test_aux_n, _ = normalize_data(test_aux, norm_vals=aux_norm_vals)

    x_train, (h, w) = _to_tensor(train_scaled, train_aux_n, input_mask)
    x_val, _ = _to_tensor(test_scaled, test_aux_n, input_mask)
    if extra is not None:
        x_extra, _ = _to_tensor(extra_scaled, None, input_mask, aux_dim=x_train.shape[-1] - int(input_mask.sum()) - len(extra_names))
        x_train = torch.cat([x_train, x_extra], dim=0)
        del x_extra
        print(f"appended {n_no_cams} samples from no-CAMS years with zero-filled aux channels")
    for name, t in (("train", x_train), ("val", x_val)):
        n_bad = int((~torch.isfinite(t)).sum())
        if n_bad:
            raise ValueError(f"{n_bad} non-finite values in the {name} pretraining inputs")
    feature_dim = len(variable_names)
    aux_dim = x_train.shape[-1] - feature_dim - len(extra_variable_names)
    model_channels = feature_dim + aux_dim
    # Channel names in TENSOR order: model input (met, then aux placeholders), then target-only.
    tensor_names = variable_names + [("aux", i, 0) for i in range(aux_dim)] + extra_variable_names
    if extra_variable_names:
        print(f"{len(extra_variable_names)} target-only channels (time deltas "
              f"{sorted({int(v[2]) for v in extra_variable_names})}) appended after the {model_channels} model inputs")
    # Build the grid from the scaler_fit_years subset (= the fine-tuning training set), so the
    # reference footprint — and hence the mesh — is the one the fine-tuning run will build.
    grid, _ = get_grid(train_fp.isel(time=fit_sel), parameters.get("grid_reference_fp"))
    info = dict(variable_names=variable_names, extra_variable_names=extra_variable_names,
                tensor_names=tensor_names, model_channels=model_channels, years=all_years,
                feature_dim=feature_dim, aux_dim=aux_dim, height=h, width=w, grid=grid,
                aux_norm_vals=aux_norm_vals, n_no_cams=n_no_cams, pseudo_fp=pseudo)
    return x_train, x_val, info


def alias_extra_channel_scalers(scaler, extra_names):
    """Make ``scaler`` (a fitted ``DefaultInputsScaler``) transform the target-only channels
    ``extra_names`` with the scaler of the same (variable, level) from the input layout."""
    if not hasattr(scaler, "scalers") or not hasattr(scaler, "full_variable_names"):
        raise TypeError(f"extra_time_deltas needs a DefaultInputsScaler-like input scaler, got {type(scaler).__name__}")
    by_var_level = {}
    for name, sc in scaler.scalers.items():
        by_var_level.setdefault((name[0], name[1]), sc)
    for name in extra_names:
        key = (name[0], name[1])
        if key not in by_var_level:
            raise ValueError(f"no fitted scaler for {key} to reuse for the target-only channel {tuple(name)}")
        scaler.scalers[name] = by_var_level[key]
        scaler.full_variable_names.append(name)


@torch.no_grad()
def evaluate(model, task, x_val, batch_size, device, seed, y_val=None):
    """Validation pretext loss with FIXED masks (seeded), the trivial-baseline loss on the same
    cells, and the per-variable model MSE. ``y_val``: external targets ``[N, nodes]`` if the task
    needs them."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    sums = {"model": 0.0, "baseline": 0.0}
    per_channel = torch.zeros(task.num_targets, device=device, dtype=torch.float64)
    per_channel_base = torch.zeros_like(per_channel)
    n_cells = 0
    for start in range(0, x_val.shape[0], batch_size):
        x = x_val[start:start + batch_size].to(device, non_blocking=True)
        y = None if y_val is None else y_val[start:start + batch_size].to(device, non_blocking=True)
        x_c, target, cell_mask = task.corrupt(x, gen, y)
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
    # masked / next_delta are defined on the model's input columns only; delta_forecast may
    # target the extra (target-only) channels that follow them.
    groups = pretraining.channel_groups(info["tensor_names"][:info["model_channels"]], dynamic_vars)
    forecast = None
    if task_name == "delta_forecast":
        forecast = pretraining.forecast_groups(
            info["tensor_names"], dynamic_vars, run_cfg["source_deltas"], run_cfg["target_delta"],
            tendency=run_cfg.get("tendency", False), model_channels=info["model_channels"])
    y_train_all = y_val = baseline_map = pseudo_entry = None
    if task_name == "pseudo_footprint":
        cfg = pseudo_fp.resolve_config(pre.get("pseudo_footprint"), run_cfg.get("pseudo_footprint"))
        pseudo_entry = info["pseudo_fp"][pseudo_fp.config_key(cfg)]
        y_train_all, y_val = pseudo_entry["y_train"], pseudo_entry["y_val"]
        baseline_map = y_train_all[train_idx].mean(dim=0)      # climatology of THIS run's training years
    task = pretraining.PretextTask(task_name, groups, info["height"], info["width"],
                                   block_size=pre.get("block_size", 10),
                                   mask_ratio=pre.get("mask_ratio", 0.5),
                                   forecast=forecast, model_channels=info["model_channels"],
                                   baseline_map=baseline_map).to(device)
    all_names = ["|".join(str(x) for x in v) for v in info["tensor_names"]]
    print(f"[{name}] pretext: {task.describe()}")
    save_modules = list(run_cfg.get("save_modules", pre.get("save_modules", pretraining.TRUNK_MODULES)))
    if "fp_decoder" in save_modules and task.num_targets != 1:
        raise ValueError(f"run {name}: fp_decoder can only be saved when the head is one channel per node "
                         f"(task {task_name} has {task.num_targets} targets)")
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
                                    pretext=task.describe(),
                                    input_variable_names=all_names[:info["model_channels"]],
                                    extra_variable_names=all_names[info["model_channels"]:],
                                    seed=seed, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    settings["channel_groups"] = groups
    settings["forecast_groups"] = forecast
    settings["pretrain_run"]["save_modules"] = save_modules
    if pseudo_entry is not None:
        settings["pretrain_run"]["pseudo_footprint"] = dict(pseudo_entry["config"], stats_train=pseudo_entry["stats_train"],
                                                            stats_val=pseudo_entry["stats_val"])
        # a few validation targets for eyeballing (scripts/plot_pseudo_footprints.py)
        n_ex = min(32, int(y_val.shape[0]))
        np.savez_compressed(out_dir / "pseudo_fp_examples.npz",
                            targets=y_val[:n_ex].numpy().reshape(n_ex, info["height"], info["width"]),
                            exits=pseudo_entry["exits_val"][:n_ex].numpy(), baseline=baseline_map.numpy().reshape(info["height"], info["width"]))
    with open(out_dir / "pretrain_settings.json", "w") as f:
        json.dump(settings, f, indent=1, default=str)

    use_wandb = parameters.get("use_wandb", False)
    if use_wandb:
        wcfg = parameters.get("wandb", {})
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        wandb.init(entity=wcfg.get("entity"), project=wcfg.get("project"), group=wcfg.get("group"),
                   tags=list(wcfg.get("tags", [])) + ["met_pretrain", task_name] + (
                       [f"src{'-'.join(map(str, forecast['source_deltas']))}_tgt{forecast['target_delta']}"] if forecast else []),
                   name=f"{job_id}_pretrain_{name}", job_type="met_pretrain",
                   notes=run_cfg.get("notes"), config=settings)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)
    log, best = [], {"val": float("inf"), "epoch": None}
    target_names = [all_names[i] for i in task.target_idx.tolist()]
    updates = open(out_dir / "updates.txt", "a")
    updates.write(f"pretext: {task.describe()}\n")

    def meta(epoch, val):
        return dict(task=task_name, pretext=task.describe(), run_name=name, train_years=train_years,
                    epoch=epoch, val_loss=val, feature_dim=info["feature_dim"], aux_dim=info["aux_dim"],
                    input_variable_names=all_names[:info["model_channels"]],
                    model_parameters=parameters["model_parameters"], seed=seed,
                    pseudo_footprint=None if pseudo_entry is None else pseudo_entry["config"],
                    scaler_fit_years=pre["scaler_fit_years"], slurm_job_id=os.environ.get("SLURM_JOB_ID"))

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=shuffle_gen)]
        run_loss = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            x = x_train_all[idx].to(device, non_blocking=True)
            y = None if y_train_all is None else y_train_all[idx].to(device, non_blocking=True)
            x_c, target, cell_mask = task.corrupt(x, gen, y)
            pred = pretraining.pretrain_forward(model, x_c)
            loss = task.loss(pred, target, cell_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += float(loss)
        train_loss = run_loss / steps_per_epoch
        val_loss, base_loss, per_channel, per_channel_base = evaluate(model, task, x_val, batch_size, device, seed=seed + 1, y_val=y_val)
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
            pretraining.save_trunk(model, out_dir / "trunk_best.pt", meta(epoch, val_loss), modules=save_modules)
            best_per_channel = per_channel

    pretraining.save_trunk(model, out_dir / "trunk_last.pt", meta(epochs - 1, log[-1]["val_loss"]), modules=save_modules)
    if task.external_target:
        target_names = [task_name]
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


def validate_run(run, input_deltas, loaded_deltas, parameters=None):
    """Fail early on an ill-specified ``pretrain.runs`` entry (before the expensive data load)."""
    name, task = run.get("name"), run.get("task")
    if task not in pretraining.PRETEXT_TASKS:
        raise ValueError(f"run {name}: unknown task {task!r}; allowed {pretraining.PRETEXT_TASKS}")
    modules = run.get("save_modules", (parameters or {}).get("pretrain", {}).get("save_modules", pretraining.TRUNK_MODULES))
    unknown = set(modules) - set(pretraining.SAVEABLE_MODULES)
    if unknown:
        raise ValueError(f"run {name}: save_modules {sorted(unknown)} not allowed; {pretraining.SAVEABLE_MODULES}")
    if "fp_decoder" in modules and task != "pseudo_footprint":
        raise ValueError(f"run {name}: fp_decoder can only be saved by a pseudo_footprint run (one output per node)")
    if task == "pseudo_footprint":
        cfg = pseudo_fp.resolve_config(parameters["pretrain"].get("pseudo_footprint"), run.get("pseudo_footprint"))
        if cfg["hours"] > max(loaded_deltas):
            raise ValueError(f"run {name}: pseudo_footprint.hours={cfg['hours']} needs met up to that delta; loaded {loaded_deltas} "
                             "(add to pretrain.extra_time_deltas)")
        met_levels = {int(l) for l in parameters["variables"]["met_levels"]}
        if not set(cfg["levels"]) <= met_levels:
            raise ValueError(f"run {name}: pseudo_footprint.levels {cfg['levels']} not all in variables.met_levels {sorted(met_levels)}")
        for v in ("x_wind", "y_wind"):
            if v not in parameters["variables"]["met_variables"]:
                raise ValueError(f"run {name}: pseudo_footprint needs {v} in variables.met_variables")
        return cfg
    if task != "delta_forecast":
        return
    for key in ("source_deltas", "target_delta"):
        if key not in run:
            raise ValueError(f"run {name}: delta_forecast needs {key!r}")
    src = sorted({int(d) for d in run["source_deltas"]})
    tgt = int(run["target_delta"])
    if not src or any(d not in input_deltas for d in src):
        raise ValueError(f"run {name}: source_deltas {src} must be a non-empty subset of the model's input deltas {input_deltas}")
    if tgt not in loaded_deltas:
        raise ValueError(f"run {name}: target_delta {tgt} is not loaded ({loaded_deltas}); add it to "
                         "variables.time_deltas or pretrain.extra_time_deltas")
    if tgt in src:
        raise ValueError(f"run {name}: target_delta {tgt} is also a source delta")


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
    load_params = loading_parameters(parameters)          # + extra_time_deltas for the loaders only
    loaded_deltas = sorted({0} | {int(d) for d in load_params["variables"].get("time_deltas", [])})
    pseudo_cfgs = {}
    for r in runs:
        run_pseudo_cfg = validate_run(r, dual_time_deltas(parameters), loaded_deltas, parameters)
        if r["task"] == "pseudo_footprint":
            pseudo_cfgs[pseudo_fp.config_key(run_pseudo_cfg)] = run_pseudo_cfg
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
    bundle = load_dual_data(load_params, verbose=parameters.get("verbose", True))
    extra = load_no_cams_years(load_params, no_cams_years) if no_cams_years else None
    x_train, x_val, info = prepare_tensors(parameters, bundle, extra, pseudo_fp_configs=list(pseudo_cfgs.values()))
    del bundle, extra
    print(f"pretraining tensors: train {tuple(x_train.shape)}, val {tuple(x_val.shape)}")
    if torch.cuda.is_available():
        x_train, x_val = x_train.pin_memory(), x_val.pin_memory()
    for r in runs:
        run_pretraining(parameters, r, x_train, x_val, info, out_root)
    print("All pretraining runs finished.")


if __name__ == "__main__":
    main()
