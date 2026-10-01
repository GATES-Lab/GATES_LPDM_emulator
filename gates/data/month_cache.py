"""On-disk cache of the month-by-month loader outputs (met inputs, footprints, background, aux CAMS).

Why
---
Loading a month of footprints + met is the slow part of every training job (~0.3 min per month
plus ~0.35 min per 1000 footprints), and the standard pipeline then holds several copies of the
concatenated inputs in memory. The cache stores what the loader returns for ONE month of ONE
region, exactly as loaded (raw, before any scaling or cross-month operation):

    <cache_dir>/<REGION>/<config key>/config.json          the loading configuration (see below)
    <cache_dir>/<REGION>/<config key>/<YYYY>-<MM>/
        inputs.npy       met inputs, (n, lat, lon, variable_name), as loaded (float32)
        small.pickle     footprints, background and aux CAMS Datasets + the inputs' coordinates
        meta.json        sample count, NaN counts, footprint files, code fingerprint; written LAST

so a later job reads a month in seconds, and ``gates/training/lean_dual_data.py`` can assemble a
training set month by month without ever building the concatenated array.

What decides whether a cached month is reused
---------------------------------------------
* The ``config key`` is a hash of everything that determines the loader output for a month of a
  region: the ``train_load_data`` / ``test_load_data`` entries other than the years and months
  (freq, size, met_args, ...), the ``variables`` block, the background settings, the data
  directories of ``config.yml`` and any ``data_dirs`` override. A different setting is a different
  directory: a stale month can never be picked up for a changed configuration.
* The footprint files of the month (name, size, modification time) are recorded; a month whose
  files changed is rebuilt.
* The loader SOURCE is fingerprinted but does not invalidate the cache (unrelated edits would
  force a rebuild every time): a changed fingerprint prints a ``FLAG:`` line naming the files,
  and ``"data_cache": {"rebuild": true}`` forces a rebuild.

The cache holds derived data only; it can be deleted at any time.
"""

import datetime
import glob
import hashlib
import json
import os
import pickle
import shutil
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CACHE_FORMAT_VERSION = 1

# Source files that decide what the loader returns for a month (relative to the repository root)
LOADER_SOURCE_FILES = (
    "gates/data/load_data.py",
    "gates/data/datasets.py",
    "gates/data/load_background_data.py",
    "gates/data/far_field.py",
    "gates/training/training_background.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------
# Configuration key and code fingerprint
# ---------------------------------------------------------------

def _jsonable(obj):
    """Canonical JSON-friendly form of a loading configuration (paths as strings, tuples as lists)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def loading_config(region, base_params, input_variables, far_field_cfg, datapath_args,
                   detrend, use_aux_bc, aux_indeces, data_paths=None):
    """Everything that determines the loader output for a month of ``region``.

    The arguments are those of ``load_GATES_month_with_bg`` (outputs of
    ``resolve_month_loading``); ``data_paths`` is the ``data_paths`` block of ``config.yml``.
    """
    return _jsonable({
        "cache_format_version": CACHE_FORMAT_VERSION,
        "region": region,
        "load_data": base_params,
        "variables": input_variables,
        "far_field": far_field_cfg,
        "datapath_args": datapath_args,
        "background": {"detrend": detrend, "use_aux_bc": use_aux_bc,
                       "aux_indeces": list(aux_indeces) if use_aux_bc else None},
        "data_paths": data_paths,
    })


def config_key(config):
    """Short hash of a loading configuration (directory name of its cache)."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def source_fingerprint(root=_REPO_ROOT):
    """``{file: sha1}`` of the loader source files (missing files map to None)."""
    out = {}
    for rel in LOADER_SOURCE_FILES:
        path = Path(root) / rel
        out[rel] = hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None
    return out


def footprint_file_stats(pattern):
    """``[[name, size, mtime_ns], ...]`` of the footprint files matched by the loader's glob."""
    stats = []
    for f in sorted(glob.glob(str(pattern))):
        st = os.stat(f)
        stats.append([os.path.basename(f), int(st.st_size), int(st.st_mtime_ns)])
    return stats


# ---------------------------------------------------------------
# The inputs DataArray <-> (array, header) round trip
# ---------------------------------------------------------------

def split_inputs(inputs):
    """Split the inputs DataArray into its data and a zero-length ``header`` carrying everything
    else (dims, lat / lon / variable_name coordinates, name, attributes) + the ``fp_time`` values."""
    if tuple(inputs.dims) != ("fp_time", "lat", "lon", "variable_name"):
        raise ValueError(f"inputs must have dims (fp_time, lat, lon, variable_name), got {inputs.dims}")
    unexpected = [k for k, v in inputs.coords.items() if "fp_time" in v.dims and k != "fp_time"]
    if unexpected:
        raise ValueError(f"inputs carry per-sample coordinates the month cache does not store: {unexpected}")
    return np.ascontiguousarray(inputs.values), inputs.isel(fp_time=slice(0, 0)), inputs.fp_time.values


def rebuild_inputs(array, header, fp_time):
    """Inverse of :func:`split_inputs` (``array`` may be any array of the right shape)."""
    mindex = header.indexes["variable_name"]
    coords = {"fp_time": ("fp_time", fp_time, header.fp_time.attrs)}
    for name in ("lat", "lon"):
        coords[name] = header.coords[name].variable
    da = xr.DataArray(array, dims=header.dims, coords=coords, name=header.name, attrs=header.attrs)
    return da.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, "variable_name"))


# ---------------------------------------------------------------
# The cache of one region + loading configuration
# ---------------------------------------------------------------

class MonthCache:
    """Months of one region loaded with one loading configuration (see the module docstring)."""

    def __init__(self, cache_dir, region, config):
        self.region = region
        self.config = config
        self.key = config_key(config)
        self.dir = Path(cache_dir) / str(region) / self.key
        self._flagged_sources = False

    # ----- layout -----
    def month_dir(self, year, month):
        return self.dir / f"{year}-{str(month).zfill(2)}"

    def _ensure_dir(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.dir / "config.json"
        if cfg_path.exists():
            stored = json.loads(cfg_path.read_text())
            if stored != self.config:
                raise RuntimeError(f"month cache {self.dir} holds a different loading configuration "
                                   "with the same key (hash collision or edited config.json)")
            return
        tmp = cfg_path.with_name(f"config.json.tmp-{socket.gethostname()}-{os.getpid()}")
        tmp.write_text(json.dumps(self.config, indent=1, sort_keys=True))
        os.replace(tmp, cfg_path)

    # ----- reading -----
    def read_meta(self, year, month):
        path = self.month_dir(year, month) / "meta.json"
        return json.loads(path.read_text()) if path.exists() else None

    def has(self, year, month, verbose=True):
        """Whether the month is cached AND still valid (format, footprint files unchanged)."""
        meta = self.read_meta(year, month)
        if meta is None:
            return False
        tag = f"{self.region} {year}-{str(month).zfill(2)}"
        if meta.get("cache_format_version") != CACHE_FORMAT_VERSION:
            print(f"FLAG: month cache {tag}: format version {meta.get('cache_format_version')} != "
                  f"{CACHE_FORMAT_VERSION} -> rebuilding")
            return False
        if footprint_file_stats(meta["fp_pattern"]) != meta["fp_files"]:
            print(f"FLAG: month cache {tag}: the footprint files changed since it was built -> rebuilding")
            return False
        if meta.get("source_fingerprint") != source_fingerprint() and not self._flagged_sources:
            changed = [f for f, h in source_fingerprint().items()
                       if (meta.get("source_fingerprint") or {}).get(f) != h]
            if verbose:
                print(f"FLAG: month cache {self.dir}: the loader source changed since (some of) it was "
                      f"built on {meta.get('created')}: {changed}. The cached months are REUSED; set "
                      "\"data_cache\": {\"rebuild\": true} if the change affects what is loaded.")
            self._flagged_sources = True
        return True

    def read_small(self, year, month):
        """``{"fp_xr", "background", "aux_data", "inputs_header", "fp_time"}`` of a cached month."""
        path = self.month_dir(year, month) / "small.pickle"
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"cannot read {path} ({type(e).__name__}: {e}). It was probably written with other "
                "library versions: rebuild the cache with \"data_cache\": {\"rebuild\": true}.") from e

    def read_inputs_array(self, year, month, mmap_mode=None):
        return np.load(self.month_dir(year, month) / "inputs.npy", mmap_mode=mmap_mode)

    def read_inputs(self, year, month):
        """The inputs DataArray of a cached month, as the loader returned it."""
        small = self.read_small(year, month)
        return rebuild_inputs(self.read_inputs_array(year, month), small["inputs_header"], small["fp_time"])

    # ----- writing -----
    def write(self, year, month, inputs, fp_xr, background, aux_data, info, load_minutes=None):
        """Store the loader outputs of one month. Atomic: the month directory appears complete or
        not at all (it is written under a temporary name and renamed)."""
        start = time.perf_counter()
        self._ensure_dir()
        final = self.month_dir(year, month)
        tmp = final.with_name(f"{final.name}.tmp-{socket.gethostname()}-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        inputs, fp_xr, background = inputs.load(), fp_xr.load(), background.load()
        if aux_data is not None:
            aux_data = aux_data.load()
        array, header, fp_time = split_inputs(inputs)
        # the split must be lossless: refuse to cache anything the rebuild does not reproduce
        xr.testing.assert_identical(rebuild_inputs(array, header, fp_time), inputs)
        if not np.array_equal(fp_xr.time.values, fp_time):
            raise ValueError(f"{self.region} {year}-{month}: footprint times differ from the input times")

        np.save(tmp / "inputs.npy", array)
        with open(tmp / "small.pickle", "wb") as f:
            pickle.dump({"fp_xr": fp_xr, "background": background, "aux_data": aux_data,
                         "inputs_header": header, "fp_time": fp_time}, f, protocol=4)

        meta = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "region": self.region, "year": str(year), "month": str(month).zfill(2),
            "domain": info["domain"],
            "n_samples": int(array.shape[0]),
            "inputs_shape": [int(v) for v in array.shape],
            "inputs_dtype": str(array.dtype),
            "inputs_nan_count": int(np.isnan(array).sum()),
            "inputs_inf_count": int(np.isinf(array).sum()),
            "time_first": str(fp_time[0]) if len(fp_time) else None,
            "time_last": str(fp_time[-1]) if len(fp_time) else None,
            "times_sorted_unique": bool(np.all(np.diff(fp_time) > np.timedelta64(0))),
            "fp_pattern": str(info["fp_pattern"]),
            "fp_files": footprint_file_stats(info["fp_pattern"]),
            "source_fingerprint": source_fingerprint(),
            "versions": {"numpy": np.__version__, "xarray": xr.__version__, "pandas": pd.__version__},
            "load_minutes": load_minutes,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "created_by_job": os.environ.get("SLURM_JOB_ID"),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))

        if final.exists():  # rebuild, or another job finished the same month first: replace it
            old = final.with_name(f"{final.name}.old-{socket.gethostname()}-{os.getpid()}")
            os.rename(final, old)
            os.rename(tmp, final)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(tmp, final)
        print(f"month cache: wrote {self.region} {year}-{str(month).zfill(2)} ({meta['n_samples']} samples, "
              f"{array.nbytes / 2**30:.2f} GiB, {time.perf_counter() - start:.0f} s) -> {final}")
        return meta
