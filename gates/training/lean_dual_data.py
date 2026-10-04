"""Memory-lean data preparation for the dual-head trainers (large training sets).

The standard pipeline (``load_GATES_data_with_bg`` -> ``setup_dual_dataloaders`` ->
``materialise_batches``) holds about seven copies of the training inputs at its peak (month list +
concatenated array, float32 copy for the scaler fit, fit subsample, scaled copy, copy with the aux
channels, shuffled copy, batches in shared memory): 11.7 GiB per 1000 samples, i.e. ~36,000
training samples on a 460 GB node (job 6919595). This module builds the SAME tensors with one copy:

1. every month of the training and test sets is loaded once and kept on disk
   (``gates/data/month_cache.py``); months already cached are not loaded again;
2. the training inputs are written month by month straight into their final place: a
   shared-memory tensor laid out in batch order, i.e. what ``materialise_batches`` returns;
3. the input scaler is fitted by the scaler's own code, one variable and level at a time (several
   in parallel), on the same random subsample as in the standard pipeline;
4. the inputs are scaled in place, in chunks, by the scaler's own ``transform``;
5. the test set is small and goes through the standard code (``setup_dual_eval_loader``).

The result is identical to the standard pipeline: same samples, same batches in the same order,
same scaler statistics, same values (checked bit for bit on real data,
``smoke_tests_lean/compare_old_vs_lean.py``), and the global NumPy RNG is left in the same state.

Enable it in the parameter file (off by default, the standard pipeline is then untouched)::

    "data_cache": {
        "enabled": true,
        "dir": null,        # default: user_paths.data_cache_dir of config.yml
        "rebuild": false    # true = reload the months from the archive and overwrite the cache
    }

Limits: ``DefaultInputsScaler`` only; the loader must return float32 inputs (it does).
"""

import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import torch
import xarray as xr

import gates
import gates.data.datasets as gates_datasets
from gates.data.month_cache import MonthCache, loading_config, rebuild_inputs
from gates.data.input_domain import input_domain_settings, loader_parameters, apply_input_domain

from .training import build_input_dataset, setup_fp_dataset, make_cluster
from .training_background import (
    MonthLoadError, resolve_month_loading, load_GATES_month_with_bg, month_key_for, concat_months,
    check_unique_times,
)
from .training_dual import setup_dual_eval_loader, _dual_dataloader_settings

GIB = 2 ** 30
_ALLOWED_KEYS = {"enabled", "dir", "rebuild", "threads", "fit_buffer_gib"}
# seed of the (fixed) shuffle of the training samples: the default of make_dual_dataloader
_SHUFFLE_SEED = 42


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

def data_cache_settings(parameters):
    """Validate ``parameters["data_cache"]`` and return it with the defaults filled in."""
    cfg = dict(parameters.get("data_cache", None) or {})
    unknown = set(cfg) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"data_cache: unknown keys {sorted(unknown)}; allowed: {sorted(_ALLOWED_KEYS)}")
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    settings = {
        "enabled": bool(cfg.get("enabled", False)),
        "dir": cfg.get("dir", None),
        "rebuild": bool(cfg.get("rebuild", False)),
        # worker threads for reading / scaling; each holds up to one month of inputs
        "threads": int(cfg.get("threads", None) or max(1, min(8, n_cpus))),
        # working memory of the scaler fit: decides how many variables / levels are fitted at once
        "fit_buffer_gib": float(cfg.get("fit_buffer_gib", None) or 48.0),
    }
    if settings["enabled"] and settings["dir"] is None:
        default_dir = getattr(gates.config.get_config(), "data_cache_dir", None)
        if default_dir is None:
            raise ValueError("data_cache is enabled but no cache directory is configured: set "
                             "user_paths.data_cache_dir in config.yml or data_cache.dir in the parameter file")
        settings["dir"] = str(default_dir)
    return settings


def data_cache_enabled(parameters):
    return bool((parameters.get("data_cache", None) or {}).get("enabled", False))


# ---------------------------------------------------------------
# Months on disk
# ---------------------------------------------------------------

class MonthEntry(NamedTuple):
    cache: MonthCache
    region: str
    year: str
    month: str
    n_samples: int


def list_months(data_parameters, input_variables, datapath_args, background_params, cache_dir):
    """``[(cache, region, year, month), ...]`` of a data split, in loading order, plus the
    resolved loader arguments (``resolve_month_loading``)."""
    resolved = resolve_month_loading(data_parameters, input_variables, datapath_args)
    regions, years, months, base_params, input_vars, far_field_cfg, dp_args = resolved
    data_paths = getattr(gates.config.get_config(), "data_paths", None)
    out = []
    for region in regions:
        config = loading_config(region, base_params, input_vars, far_field_cfg, dp_args,
                                background_params["detrend"], background_params["use_auxiliary_bc"],
                                background_params["auxilary_bc_levels"], data_paths=data_paths)
        cache = MonthCache(cache_dir, region, config)
        out.extend((cache, region, str(year), str(month).zfill(2)) for year in years for month in months)
    return out, resolved


def ensure_months_cached(data_parameters, input_variables, datapath_args, background_params, cache_dir,
                         rebuild=False, verbose=True, shard=None, loading_times_out=None, cluster=None):
    """Make sure every month of a data split is in the cache, loading the missing ones.

    Args:
        data_parameters, input_variables, datapath_args: As for ``load_GATES_data_with_bg``.
        background_params (dict): Resolved ``background_setup``.
        cache_dir (str or Path): Root of the month cache.
        rebuild (bool): Reload and overwrite months that are already cached.
        shard (tuple or None): ``(i, n)`` = only build the missing months whose position in the
            month list is ``i`` modulo ``n`` (for building a cache with ``n`` jobs at once); the
            returned list then holds the months available so far.
        loading_times_out (dict, optional): Filled with the minutes spent LOADING each month that
            was not cached (cached months are not listed).
        cluster (dict, optional): ``{}`` to be filled with a lazily started Dask cluster
            (``{"client", "cluster"}``), started only if a month has to be loaded.

    Returns:
        entries (list[MonthEntry]): The cached months, in loading order.
        failed (list[str]): Months the loader could not build (skipped, as in the standard
            pipeline).
    """
    months, resolved = list_months(data_parameters, input_variables, datapath_args, background_params, cache_dir)
    regions, _, _, base_params, input_vars, far_field_cfg, dp_args = resolved
    # Optional larger input domain ("input_domain" in train_load_data, gates/data/input_domain.py): the
    # month is loaded on the input window and reduced to what the model trains on before it is cached
    # (the option is part of base_params, so such months have a cache key of their own).
    input_domain = input_domain_settings(base_params)
    base_params = loader_parameters(base_params)
    entries, failed = [], []
    for position, (cache, region, year, month) in enumerate(months):
        key = month_key_for(region, year, month, regions)
        mine = shard is None or position % shard[1] == shard[0]
        if (rebuild and mine) or not cache.has(year, month, verbose=verbose):
            if not mine:
                continue
            if cluster is not None and "cluster" not in cluster:
                cluster["client"], cluster["cluster"] = make_cluster()
            start = time.perf_counter()
            if verbose:
                print(f"month cache: loading {region} {year}-{month} from the archive")
            try:
                inputs, fp_xr, background, aux_data, info = load_GATES_month_with_bg(
                    region, year, month, base_params, input_vars, far_field_cfg, dp_args,
                    detrend=background_params["detrend"], verbose=verbose, load_into_memory=True,
                    use_aux_bc=background_params["use_auxiliary_bc"],
                    aux_indeces=background_params["auxilary_bc_levels"])
            except MonthLoadError as e:
                print(f"Error loading data for {year}-{month}: {e}")
                print(f"FLAG: {key} could not be loaded and is SKIPPED (not part of the data set)")
                failed.append(key)
                continue
            if input_domain is not None:
                inputs, fp_xr = apply_input_domain(inputs, fp_xr, input_domain)
            minutes = (time.perf_counter() - start) / 60
            cache.write(year, month, inputs, fp_xr, background, aux_data, info, load_minutes=minutes)
            if loading_times_out is not None:
                loading_times_out[key] = minutes
            del inputs, fp_xr, background, aux_data
            gc.collect()
        entries.append(MonthEntry(cache, region, year, month, cache.read_meta(year, month)["n_samples"]))
    return entries, failed


# ---------------------------------------------------------------
# The training inputs, without the data
# ---------------------------------------------------------------

class LazyMonthInputs:
    """Stand-in for the training ``inputs`` DataArray of a ``DualDataBundle``.

    Knows the shape, the (sorted) sample times and the variable names, and where each month's
    array is on disk, but never holds the data: :func:`setup_lean_dual_data` reads the months
    straight into the training tensor.
    """

    def __init__(self, entries, headers, month_times):
        if not entries:
            raise ValueError("no month could be loaded for the training set")
        self.entries = list(entries)
        # name and attributes of the first month, as xr.concat keeps them in the standard pipeline
        self.header = headers[0]
        for entry, header in zip(entries, headers):
            # (the attributes differ from month to month: they hold the time of loading)
            if header.dims != self.header.dims or not header.equals(self.header):
                raise ValueError(f"{entry.region} {entry.year}-{entry.month}: the inputs have other "
                                 "variables / window coordinates than the first month")
        all_times = np.concatenate(month_times)
        order = np.argsort(all_times, kind="stable")
        self.fp_time = all_times[order]
        check_unique_times(self.fp_time, sorted({e.region for e in entries}))
        position = np.empty(len(all_times), dtype=np.int64)
        position[order] = np.arange(len(all_times))
        # sorted position of each month's samples, in the order they are stored on disk
        self.sorted_positions = np.split(position, np.cumsum([len(t) for t in month_times])[:-1])
        self._prepared = None   # tensors of the last setup_lean_dual_data call (reused across arms)

    @property
    def shape(self):
        return (len(self.fp_time), self.header.sizes["lat"], self.header.sizes["lon"],
                self.header.sizes["variable_name"])

    @property
    def variable_name(self):
        return self.header.variable_name

    def __len__(self):
        return len(self.fp_time)

    def __repr__(self):
        return f"LazyMonthInputs(shape={self.shape}, months={len(self.entries)})"


def load_lean_dual_data(parameters, train_load_data_params, test_load_data_params, input_variables,
                        datapath_args, background_params, verbose=True):
    """Month-cache counterpart of ``train_dual_model.load_dual_data``.

    Returns the same eight objects as the standard loader, except that ``train_inputs`` is a
    :class:`LazyMonthInputs`, and a dict with the loading statistics.
    """
    settings = data_cache_settings(parameters)
    print(f"Month cache: {settings['dir']} (rebuild={settings['rebuild']})")
    cluster = {}
    train_month_mins, test_month_mins = {}, {}
    start = time.perf_counter()
    train_entries, train_failed = ensure_months_cached(
        train_load_data_params, input_variables, datapath_args, background_params, settings["dir"],
        rebuild=settings["rebuild"], verbose=verbose, loading_times_out=train_month_mins, cluster=cluster)
    train_load_mins = (time.perf_counter() - start) / 60
    start = time.perf_counter()
    test_entries, test_failed = ensure_months_cached(
        test_load_data_params, input_variables, datapath_args, background_params, settings["dir"],
        rebuild=settings["rebuild"], verbose=verbose, loading_times_out=test_month_mins, cluster=cluster)
    test_load_mins = (time.perf_counter() - start) / 60
    if cluster.get("cluster") is not None:
        cluster["cluster"].close()
        cluster["client"].close()

    start = time.perf_counter()
    bundle = bundle_from_entries(train_entries, test_entries, threads=settings["threads"])
    train_fp_data, train_inputs, _, _, test_fp_data, test_inputs, _, _ = bundle
    print(f"Month cache: {len(train_entries)} training months = {len(train_inputs)} samples, "
          f"{len(test_entries)} test months = {test_inputs.sizes['fp_time']} samples "
          f"(read in {time.perf_counter() - start:.0f} s)")
    for name, failed in (("training", train_failed), ("test", test_failed)):
        if failed:
            print(f"FLAG: {len(failed)} {name} month(s) could not be loaded and are missing: {failed}")

    nan_months = [f"{e.region} {e.year}-{e.month}" for e in train_entries + test_entries
                  if e.cache.read_meta(e.year, e.month)["inputs_nan_count"]
                  or e.cache.read_meta(e.year, e.month)["inputs_inf_count"]]
    if nan_months:
        print(f"FLAG: the met inputs of these months hold NaN / inf values: {nan_months}")

    parameters["data_cache_resolved"] = {
        "dir": settings["dir"],
        "train_caches": sorted({str(e.cache.dir) for e in train_entries}),
        "test_caches": sorted({str(e.cache.dir) for e in test_entries}),
        "n_train_months": len(train_entries), "n_test_months": len(test_entries),
        "failed_months": train_failed + test_failed,
    }
    summary = {
        "train_load_mins": train_load_mins, "test_load_mins": test_load_mins,
        "total_load_mins": train_load_mins + test_load_mins,
        "n_train_samples": int(len(train_inputs)), "n_test_samples": int(len(test_fp_data.time)),
        "train_month_mins": train_month_mins, "test_month_mins": test_month_mins,
    }
    return bundle, summary


def bundle_from_entries(train_entries, test_entries, threads=1):
    """The eight objects of a ``DualDataBundle`` from cached months.

    Everything is built as the standard loader builds it (months concatenated and sorted by
    time), except ``train_inputs``, which is a :class:`LazyMonthInputs`.
    """
    for entries in (train_entries, test_entries):
        domains = sorted({e.cache.read_meta(e.year, e.month)["domain"] for e in entries})
        if len(domains) > 1:
            raise ValueError(f"regions {sorted({e.region for e in entries})} belong to different domains "
                             f"{domains}: the auxiliary CAMS data is per domain, load them separately")
    train_small = _read_small(train_entries, threads)
    train_fp_data, _, train_bgs, train_aux = concat_months(
        None, [s["fp_xr"] for s in train_small], [s["background"] for s in train_small],
        _aux_once(train_entries, train_small))
    train_inputs = LazyMonthInputs(train_entries, [s["inputs_header"] for s in train_small],
                                   [s["fp_time"] for s in train_small])
    if not np.array_equal(train_fp_data.time.values, train_inputs.fp_time):
        raise ValueError("training footprints and inputs do not have the same times")
    del train_small

    test_small = _read_small(test_entries, threads)
    with ThreadPoolExecutor(threads) as pool:
        arrays = list(pool.map(lambda e: e.cache.read_inputs_array(e.year, e.month), test_entries))
    test_fp_data, test_inputs, test_bgs, test_aux = concat_months(
        [rebuild_inputs(a, s["inputs_header"], s["fp_time"]) for a, s in zip(arrays, test_small)],
        [s["fp_xr"] for s in test_small], [s["background"] for s in test_small],
        _aux_once(test_entries, test_small))
    return (train_fp_data, train_inputs, train_bgs, train_aux,
            test_fp_data, test_inputs, test_bgs, test_aux)


def _read_small(entries, threads):
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda e: e.cache.read_small(e.year, e.month), entries))


def _aux_once(entries, small):
    """The auxiliary CAMS data is per (domain, month): one entry per month, whatever the regions."""
    seen, out = set(), []
    for entry, s in zip(entries, small):
        if (entry.year, entry.month) not in seen:
            seen.add((entry.year, entry.month))
            out.append(s["aux_data"])
    return out


# ---------------------------------------------------------------
# Training tensors
# ---------------------------------------------------------------

def new_shared_tensor(shape, dtype):
    """Uninitialised tensor in shared memory.

    Allocated directly as shared memory (``tensor.share_memory_()`` would first build the tensor
    in private memory and copy it). Pages are taken from the node only when they are written.
    """
    proto = torch.empty(0, dtype=dtype)
    numel = int(np.prod(shape))
    storage = proto._typed_storage()._new_shared(numel)
    return proto.new(storage).resize_(*shape)


def batch_order(times, batch_size, seed=_SHUFFLE_SEED):
    """Where each time-sorted training sample goes in the batch-ordered tensor.

    Reproduces ``_trim_dual_to_batch_size`` + ``make_dual_dataloader(randomize=True)``: the
    trailing ``len(times) % batch_size`` samples are dropped, the rest is shuffled ONCE with
    ``np.random.seed(seed); np.random.permutation(<their times>)`` and cut into consecutive
    batches. Leaves the global NumPy RNG in the state the standard pipeline leaves it in.

    Returns:
        n_used (int): Samples in the batches (a multiple of ``batch_size``).
        sorted_to_slot (np.ndarray): For the time-sorted sample ``i``, its row in the tensor; the
            dropped samples take the rows after ``n_used`` (they can still be part of the
            scaler-fit subsample).
    """
    n_all = len(times)
    n_used = n_all - n_all % batch_size
    if n_used == 0:
        raise ValueError(f"only {n_all} training samples for a batch size of {batch_size}")
    np.random.seed(seed)
    permuted_time = np.random.permutation(times[:n_used])
    slot_to_sorted = np.searchsorted(times[:n_used], permuted_time)
    if not np.array_equal(times[slot_to_sorted], permuted_time):
        raise RuntimeError("could not map the shuffled times back to the samples")
    sorted_to_slot = np.arange(n_all, dtype=np.int64)
    sorted_to_slot[slot_to_sorted] = np.arange(n_used)
    return n_used, sorted_to_slot


def _run_threaded(fn, items, threads):
    items = list(items)
    if threads <= 1 or len(items) <= 1:
        for item in items:
            fn(item)
        return
    with ThreadPoolExecutor(threads) as pool:
        for _ in pool.map(fn, items):  # iterating re-raises a worker's exception
            pass


# working memory of one unit of the scaler fit, in units of its data: the subsample itself, the
# selection the scaler makes of it, and the copy + mask of NumPy's nanmean / nanstd
_FIT_MEMORY_FACTOR = 4


def fit_input_scaler_from_store(input_dataset, store, header, times, sorted_to_slot,
                                fit_buffer_gib=48.0, threads=1):
    """Fit the input scaler on the raw inputs held in ``store`` (rows in tensor order).

    Same subsample (``InputsDataset.select_fit_times``) and same fitting code
    (``DefaultInputsScaler.fit_variable``) as the standard pipeline, so the statistics are
    identical. The fit is fed in units of one variable (min-max / ignored variables) or one level
    of one variable (standardised variables, which the scaler fits per level anyway): it needs
    the memory of a few units (``fit_buffer_gib``), not of the whole subsample, and several units
    are fitted at once.

    Returns the number of samples the scaler was fitted on.
    """
    scaler = input_dataset.scaler
    if not isinstance(scaler, gates_datasets.DefaultInputsScaler):
        raise NotImplementedError(f"data_cache supports DefaultInputsScaler only, got {type(scaler).__name__}")
    selected = input_dataset.select_fit_times(fp_times=times)
    if selected is None:
        selected = times
    slots = sorted_to_slot[np.searchsorted(times, selected)]
    n_sub, n_lat, n_lon = len(slots), header.sizes["lat"], header.sizes["lon"]
    mindex = header.indexes["variable_name"]
    variable_names = header.variable_name.values

    # units, in the order in which DefaultInputsScaler.fit works through them
    units = []
    for v in scaler.start_fit(variable_names):
        channels = [k for k, name in enumerate(variable_names) if name[0] == v]
        if v in scaler.ignore_variables or v in scaler.minmax_variables:
            units.append((v, channels))
        else:
            for level in np.unique([variable_names[k][1] for k in channels]):
                units.append((v, [k for k in channels if variable_names[k][1] == level]))

    largest = max(len(ch) for _, ch in units) * n_sub * n_lat * n_lon * 4
    at_once = max(1, min(threads, int(fit_buffer_gib * GIB / (_FIT_MEMORY_FACTOR * largest))))
    step = max(1, int(0.25 * GIB / (n_lat * n_lon * 4 * max(len(ch) for _, ch in units))))
    verbose = scaler.verbose

    def fit_unit(unit):
        v, channels = unit
        contiguous = channels == list(range(channels[0], channels[-1] + 1))
        columns = slice(channels[0], channels[-1] + 1) if contiguous else None
        data = np.empty((n_sub, n_lat, n_lon, len(channels)), dtype=np.float32)
        for a in range(0, n_sub, step):
            rows = slots[a:a + step]
            block = store[rows, :, columns] if contiguous else store[rows][:, :, channels]
            data[a:a + step] = block.reshape(-1, n_lat, n_lon, len(channels))
        da = xr.DataArray(data, dims=header.dims,
                          coords={"fp_time": selected, "lat": header.coords["lat"].variable,
                                  "lon": header.coords["lon"].variable})
        da = da.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex[channels], "variable_name"))
        # a scaler of its own per unit: the units run in parallel and are merged in order below
        unit_scaler = gates_datasets.DefaultInputsScaler(
            minmax_variables=scaler.minmax_variables, ignore_variables=scaler.ignore_variables,
            verbose=False, compute=scaler.compute)
        unit_scaler.fit_variable(v, da.sel(variable=v), variable_names)
        return unit_scaler.scalers

    for first in range(0, len(units), at_once):
        wave = units[first:first + at_once]
        if verbose:
            print(f"fitting the scalers of {[f'{v} ({len(ch)} channels)' for v, ch in wave]}")
        if len(wave) == 1:
            fitted = [fit_unit(wave[0])]
        else:
            with ThreadPoolExecutor(len(wave)) as pool:
                fitted = list(pool.map(fit_unit, wave))
        for scalers in fitted:          # in unit order: the order of the standard fit
            scaler.scalers.update(scalers)
    missing = [tuple(name) for name in variable_names if tuple(name) not in scaler.scalers]
    if missing:
        raise RuntimeError(f"no scaler was fitted for {missing[:5]}")
    return n_sub


def stack_fp_variables(ds):
    """The footprint variables of ``FootprintDataset.transform`` as one array, as the batcher
    (``make_fps_batcher``) stacks them: variables other than float32 / int32 are cast to float32,
    then all are stacked along a last axis with NumPy's common dtype (float32 + int32 -> float64).

    Returns ``(array (time, lat, lon, variable), labels)``.
    """
    labels = list(ds.data_vars)
    arrays = []
    for var in labels:
        da = ds[var].transpose("time", "lat", "lon")
        if da.dtype != "float32" and da.dtype != "int32":
            da = da.astype("float32", copy=False)
        arrays.append(np.asarray(da.values))
    dtype = np.result_type(*arrays)
    out = np.empty((*arrays[0].shape, len(arrays)), dtype=dtype)
    for k, a in enumerate(arrays):
        out[..., k] = a
    return out, labels


def _prepared_key(parameters, train_bgs, train_aux):
    """What the training tensors depend on, besides the cached months."""
    hasher = hashlib.sha1()
    settings = {k: parameters.get(k) for k in ("input_scaler", "fp_scaler", "seed")}
    settings["batch_size"] = _dual_dataloader_settings(parameters)[0]
    settings["nans_to_zeros"] = parameters.get("dataloader", {}).get("nans_to_zeros", True)
    settings["use_auxiliary_bc"] = parameters.get("background_setup", {}).get("use_auxiliary_bc", False)
    hasher.update(json.dumps(settings, sort_keys=True, default=str).encode())
    hasher.update(np.ascontiguousarray(train_bgs.values).tobytes())
    if train_aux is not None:
        hasher.update(np.ascontiguousarray(train_aux.values).tobytes())
    return hasher.hexdigest()


def setup_lean_dual_data(parameters, train_inputs, train_fps, train_bgs,
                         test_inputs, test_fps, test_bgs,
                         train_auxiliary_cams=None, test_auxiliary_cams=None):
    """Month-cache counterpart of ``setup_dual_dataloaders`` (same arguments, ``train_inputs`` being
    a :class:`LazyMonthInputs`).

    Returns:
        train_tensors (tuple of torch.Tensor): The training batches in shared memory, as
            ``materialise_batches`` returns them: inputs (n_batches, batch, nodes, features),
            footprints (n_batches, batch, nodes, variables), background (n_batches, batch, classes).
        test_loader (DataLoader): yields (inputs, fps, background).
        fp_labels (list): footprint variable label(s).
        test_scaled_fp (xr.Dataset): transformed test footprints.
        scalers (dict): {'inputs_scaler', 'fp_scaler'}.
    """
    settings = data_cache_settings(parameters)
    use_aux = parameters.get("background_setup", {}).get("use_auxiliary_bc", False)
    if not use_aux:
        train_auxiliary_cams = None
    batch_size = _dual_dataloader_settings(parameters)[0]
    times = train_inputs.fp_time
    for name, other in (("footprints", train_fps.time.values), ("backgrounds", train_bgs.time.values)):
        if not np.array_equal(other, times):
            raise ValueError(f"the training {name} do not have the times of the training inputs")

    key = _prepared_key(parameters, train_bgs, train_auxiliary_cams)
    prepared = train_inputs._prepared
    if prepared is not None and prepared["key"] == key:
        print("Month cache: reusing the training tensors of the previous experiment (same data preparation)")
        batch_order(times, batch_size)   # leaves the NumPy RNG as a fresh preparation would
    else:
        train_inputs._prepared = prepared = None
        gc.collect()
        prepared = _build_training_tensors(parameters, settings, train_inputs, train_fps, train_bgs,
                                           train_auxiliary_cams, batch_size)
        prepared["key"] = key
        train_inputs._prepared = prepared

    test_loader, fp_labels_test, test_scaled_fp = setup_dual_eval_loader(
        parameters, prepared["input_dataset"], prepared["fp_dataset"],
        test_inputs, test_fps, test_bgs, test_auxiliary_cams)
    if prepared["fp_labels"] != fp_labels_test:
        raise ValueError("Train and test footprint labels do not match - check the data loading.")
    scalers = {"inputs_scaler": prepared["input_dataset"].scaler, "fp_scaler": prepared["fp_dataset"].scaler}
    return prepared["tensors"], test_loader, prepared["fp_labels"], test_scaled_fp, scalers


def _build_training_tensors(parameters, settings, train_inputs, train_fps, train_bgs, train_aux, batch_size):
    start_all = time.perf_counter()
    threads = settings["threads"]
    times = train_inputs.fp_time
    header = train_inputs.header
    n_all, n_lat, n_lon, n_var = train_inputs.shape
    n_nodes = n_lat * n_lon
    n_aux = 0 if train_aux is None else train_aux.sizes["aux"]
    n_used, sorted_to_slot = batch_order(times, batch_size)
    slot_times = np.empty_like(times)
    slot_times[sorted_to_slot] = times

    # --- 1. raw inputs, month by month, into their final rows ---
    start = time.perf_counter()
    inputs_t = new_shared_tensor((n_all, n_nodes, n_var + n_aux), torch.float32)
    store = inputs_t.numpy()
    print(f"Month cache: building the training inputs in shared memory: {n_all} samples x {n_nodes} nodes x "
          f"{n_var + n_aux} features = {store.nbytes / GIB:.1f} GiB")

    def fill(i):
        entry = train_inputs.entries[i]
        array = entry.cache.read_inputs_array(entry.year, entry.month)
        if array.dtype != np.float32:
            raise NotImplementedError(f"{entry.region} {entry.year}-{entry.month}: inputs are {array.dtype}; "
                                      "data_cache needs the float32 inputs the loader returns")
        if array.shape[1:] != (n_lat, n_lon, n_var):
            raise ValueError(f"{entry.region} {entry.year}-{entry.month}: inputs have shape {array.shape}")
        store[sorted_to_slot[train_inputs.sorted_positions[i]], :, :n_var] = array.reshape(-1, n_nodes, n_var)

    _run_threaded(fill, range(len(train_inputs.entries)), threads)
    print(f"  raw inputs read from {len(train_inputs.entries)} months in {time.perf_counter() - start:.0f} s")

    # --- 2. input scaler: same subsample, same fitting code ---
    start = time.perf_counter()
    stand_in = xr.DataArray(np.zeros(len(times), dtype=np.float32), dims=("fp_time",), coords={"fp_time": times})
    input_dataset = build_input_dataset(parameters, stand_in)
    n_sub = fit_input_scaler_from_store(input_dataset, store, header, times, sorted_to_slot,
                                        fit_buffer_gib=settings["fit_buffer_gib"], threads=threads)
    input_dataset.inputs = None   # the stand-in was only there for the fit
    print(f"  input scaler fitted on {n_sub} samples in {time.perf_counter() - start:.0f} s")

    # --- 3. scale in place with the scaler's own transform ---
    start = time.perf_counter()
    step = max(1, int(0.75 * GIB / (n_nodes * n_var * 4)))

    def scale(a):
        b = min(a + step, n_all)
        raw = store[a:b, :, :n_var].reshape(b - a, n_lat, n_lon, n_var)
        scaled = input_dataset.transform(rebuild_inputs(raw, header, slot_times[a:b]))
        store[a:b, :, :n_var] = scaled.transpose(*header.dims).values.reshape(b - a, n_nodes, n_var)

    _run_threaded(scale, range(0, n_all, step), threads)
    print(f"  inputs scaled in {time.perf_counter() - start:.0f} s")

    # --- 4. auxiliary CAMS features, the same for every node of a sample ---
    if train_aux is not None:
        if not np.array_equal(train_aux.time.values, times):
            raise ValueError("the auxiliary CAMS data do not have the times of the training inputs")
        print("Concatenating inputs and auxiliary cams")
        aux = train_aux.transpose("time", "aux").values.astype(np.float32, copy=False)
        rows = 4096
        for a in range(0, n_all, rows):
            store[sorted_to_slot[a:a + rows], :, n_var:] = aux[a:a + rows, None, :]

    # --- 5. footprints and backgrounds ---
    start = time.perf_counter()
    fp_dataset = setup_fp_dataset(parameters, train_fps)
    scaled_fp = fp_dataset.transform(train_fps, chunk=False)
    rows = 4096
    fps_t = None
    for a in range(0, n_used, rows):
        b = min(a + rows, n_used)
        block, fp_labels = stack_fp_variables(scaled_fp.isel(time=slice(a, b)))
        if fps_t is None:
            fps_t = new_shared_tensor((n_used, n_nodes, block.shape[-1]), torch.from_numpy(block[:0]).dtype)
        fps_t.numpy()[sorted_to_slot[a:b]] = block.reshape(b - a, n_nodes, block.shape[-1])
    del scaled_fp

    bgs = train_bgs
    if bgs.dtype != "float32":
        bgs = bgs.astype("float32", copy=False)
    class_dim = [d for d in bgs.dims if d != "time"]
    if len(class_dim) != 1:
        raise ValueError(f"backgrounds should have dims (time, num_classes), got {bgs.dims}")
    bgs = bgs.transpose("time", class_dim[0]).values
    bgs_t = new_shared_tensor((n_used, bgs.shape[1]), torch.float32)
    bgs_t.numpy()[sorted_to_slot[:n_used]] = bgs[:n_used]
    print(f"  footprints and backgrounds in {time.perf_counter() - start:.0f} s")

    n_batches = n_used // batch_size
    tensors = (inputs_t[:n_used].view(n_batches, batch_size, n_nodes, n_var + n_aux),
               fps_t.view(n_batches, batch_size, n_nodes, fps_t.shape[-1]),
               bgs_t.view(n_batches, batch_size, bgs_t.shape[-1]))
    total = sum(t.numel() * t.element_size() for t in (inputs_t, fps_t, bgs_t)) / GIB
    print(f"Month cache: {n_batches} training batches of {batch_size} in shared memory ({total:.1f} GiB, "
          f"{time.perf_counter() - start_all:.0f} s); {n_all - n_used} trailing sample(s) dropped to fill the batches")
    return {"tensors": tensors, "input_dataset": input_dataset, "fp_dataset": fp_dataset, "fp_labels": fp_labels}
