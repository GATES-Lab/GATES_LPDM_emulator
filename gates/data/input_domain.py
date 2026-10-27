"""A larger INPUT domain than the footprint window (``train_load_data.input_domain``).

The dual model predicts a ``size`` x ``size`` footprint window (50 x 50 cells, ~1,300 x 1,750 km)
from the meteorology of the SAME window. The transport that shapes the far field of a footprint
also happens outside that window. This option feeds the model the meteorology of a LARGER window
around the release point, while the footprint it predicts, the losses, the test metrics and the
exported footprints stay those of the usual window, so a run with the option is compared with the
usual runs one to one. Without the option nothing changes.

Configured inside ``train_load_data`` (``test_load_data`` inherits it, as it inherits ``size``)::

    "train_load_data": {
        "size": 50,                                    # the footprint window, as before
        "input_domain": {"size": 100, "mode": "full"}  # the input window: 100 x 100 cells
    }

``input_domain.size`` must be even and larger than ``size``; both windows are centred on the
release point, so the footprint window is the central square of the input window. Modes:

* ``"full"``: every cell of the input window is a node of the model's graph (10,000 nodes for
  100 x 100). The footprint targets are kept on the input window too, with every cell outside the
  footprint window set to NaN: the ``fp_nan_mask`` that the losses already honour leaves them out
  of the training and test losses. The trainer scores and exports the footprint window only
  (:func:`resolve_fp_window`, :func:`crop_fp_dataset`, :func:`crop_fp_predictions`); the
  background head reads the whole input window (its final layer follows the window size).
* ``"coarse"``: the input window is averaged in ``factor`` x ``factor`` blocks (``factor`` = input
  size / footprint size, i.e. 2 for 100 -> 50) down to a footprint-sized grid that REPLACES the
  usual inputs. The model keeps its usual graph; the node that is ``d`` cells from the release
  point holds the weather of the block ``2 d`` cells away (the half-resolution picture of the
  whole input window, laid over the footprint grid).
* ``"native_and_coarse"``: the usual inputs plus the block-averaged input window as extra channels
  named ``ctx_<variable>`` (one block per node, as in ``"coarse"``). The window-coordinate and
  time-of-year channels are not duplicated (``COARSE_SKIP_VARIABLES``).

  Several coarse grids at once (``coarse_sizes``, mode ``"native_and_coarse"`` only)::

      "input_domain": {"size": 150, "mode": "native_and_coarse", "coarse_sizes": [100, 150]}

  The month is loaded on the largest window (``size`` must equal the last entry) and every listed
  size gives one footprint-sized grid of block means of ITS central window: the 100 x 100 window
  in 2 x 2 blocks as channels ``ctx100_<variable>`` and the 150 x 150 window in 3 x 3 blocks as
  ``ctx150_<variable>`` (the node ``d`` cells from the release point holds the blocks ``2 d`` and
  ``3 d`` cells away). The sizes must be multiples of the footprint size, strictly increasing and
  larger than it. Without ``coarse_sizes`` there is exactly one grid, of the whole input window,
  with the channel names ``ctx_<variable>`` (the runs of summary sections 30 and 31; their cache
  keys and channel names are unchanged). The ``ctx100_`` channels of a ``[100, 150]`` run are the
  ``ctx_`` channels of a ``{"size": 100}`` run, and the native channels those of a plain 50 x 50
  run, bit for bit except the topography (checked on real data, summary section 36): the loader
  interpolates the global topography onto the padded footprint domain, and a window large enough
  to pad the domain in LATITUDE (150 x 150 for BRAZIL; 50 and 100 pad it in longitude only) turns
  that coordinate from float32 into float64, which changes the rounding of the interpolation
  weights by one float32 ulp of the ~4,000 m source heights (2^-11 m). Same cells, same terrain.

The wind direction and speed of a block are recomputed from the block-mean wind components
(``wind_angle = arctan2(-u, -v)``, ``wind_speed = sqrt(u^2 + v^2)``, as the loader defines them):
averaging the angles themselves across the +-pi wrap would be wrong. Everything else is a plain
block mean. Domain edges: the loader pads the meteorology with its edge values only where a window
leaves the MET archive's domain, which for SOUTHAMERICA (lon -105.4..2.0) is much wider than the
LPDM footprint domain (lon -91.3..-24.8): no 100 x 100 window of the BRAZIL data leaves the met
domain (checked 2026-10-04 on the cached months), so the inputs beyond the footprint domain are real
meteorology and real (global-file) topography. Only the FOOTPRINT is padded there, with NaN
(28-32 % of the 100 x 100 windows, up to 27 columns on the east); in mode "full" those cells lie
outside the central footprint window anyway, and the 1.5 % of windows whose central 50 x 50 loses
1-2 eastern columns are masked by ``fp_nan_mask`` exactly as in a 50 x 50 run.

The option is applied by the month-cache pipeline (``"data_cache": {"enabled": true}``,
``gates/training/lean_dual_data.py``): a month is loaded on the input window and reduced to what
the model trains on before it is cached, under its own cache key (the option is part of
``train_load_data``). The standard pipeline refuses it (:func:`require_data_cache`). In the
"coarse" modes the footprint window is the central crop of the loaded window: identical to a
month loaded on the footprint window directly (checked on real data, summary section 30).
"""

import numpy as np
import pandas as pd
import xarray as xr

MODES = ("full", "coarse", "native_and_coarse")
COARSE_PREFIX = "ctx_"
# channels that carry no extra information when block-averaged (window coordinates, time of year)
COARSE_SKIP_VARIABLES = ("x_coords", "y_coords", "lat_coords", "lon_coords", "sin_lat_coords", "sin_lon_coords",
                         "cos_lat_coords", "cos_lon_coords", "xy_distance_centre", "earth_distance_centre",
                         "normalised_time_of_year", "normalised_day_of_year")
_INPUT_DIMS = ("fp_time", "lat", "lon", "variable_name")
_INDEX_NAMES = ["variable", "levels", "time_delta"]


# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------

def input_domain_settings(load_params):
    """Validated ``load_params["input_domain"]`` (a ``train_load_data``-like dict), or None.

    Returns ``{"mode", "size", "fp_size", "offset", "factor"}``: ``offset`` is the index of the
    footprint window's first cell inside the input window, ``factor`` the block size of the coarse
    modes (None for "full"). Mode "native_and_coarse" adds ``"coarse"``: the list of coarse grids
    (``{"size", "factor", "offset", "prefix"}`` each, see :func:`_coarse_grids`), one grid of the
    whole input window with prefix ``ctx_`` unless ``coarse_sizes`` is given.
    """
    cfg = (load_params or {}).get("input_domain", None)
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"input_domain must be a dict with 'size' and 'mode', got {cfg!r}")
    unknown = set(cfg) - {"size", "mode", "coarse_sizes"}
    if unknown:
        raise ValueError(f"input_domain: unknown keys {sorted(unknown)}; allowed: ['coarse_sizes', 'mode', 'size']")
    mode = cfg.get("mode")
    if mode not in MODES:
        raise ValueError(f"input_domain.mode must be one of {list(MODES)}, got {mode!r}")
    size, fp_size = cfg.get("size"), load_params.get("size")
    for name, value in (("input_domain.size", size), ("size", fp_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2:
            raise ValueError(f"{name} must be a positive even integer, got {value!r}")
    if size <= fp_size:
        raise ValueError(f"input_domain.size ({size}) must be larger than the footprint window size ({fp_size})")
    if mode != "full" and size % fp_size:
        raise ValueError(f"input_domain mode {mode!r} needs input_domain.size to be a multiple of size "
                         f"(got {size} and {fp_size})")
    settings = {"mode": mode, "size": size, "fp_size": fp_size, "offset": (size - fp_size) // 2,
                "factor": None if mode == "full" else size // fp_size}
    if "coarse_sizes" in cfg:
        settings["coarse"] = _coarse_grids(cfg["coarse_sizes"], mode, size, fp_size)
    elif mode == "native_and_coarse":
        # one grid of the whole input window, under the channel names of the earlier runs
        settings["coarse"] = [{"size": size, "factor": size // fp_size, "offset": 0, "prefix": COARSE_PREFIX}]
    return settings


def _coarse_grids(coarse_sizes, mode, size, fp_size):
    """Validated ``input_domain.coarse_sizes``: one ``{"size", "factor", "offset", "prefix"}`` per
    listed window (``offset`` = index of the window's first cell inside the loaded input window)."""
    if mode != "native_and_coarse":
        raise ValueError(f"input_domain.coarse_sizes is for mode 'native_and_coarse' only (got mode {mode!r})")
    if not isinstance(coarse_sizes, list) or not coarse_sizes:
        raise ValueError(f"input_domain.coarse_sizes must be a non-empty list of window sizes, got {coarse_sizes!r}")
    grids = []
    for s in coarse_sizes:
        if isinstance(s, bool) or not isinstance(s, int) or s <= fp_size or s % fp_size:
            raise ValueError(f"input_domain.coarse_sizes: {s!r} is not a multiple of the footprint window size "
                             f"({fp_size}) larger than it")
        if grids and s <= grids[-1]["size"]:
            raise ValueError(f"input_domain.coarse_sizes must be strictly increasing, got {coarse_sizes}")
        if s > size:
            raise ValueError(f"input_domain.coarse_sizes: {s} is larger than the input window ({size})")
        grids.append({"size": s, "factor": s // fp_size, "offset": (size - s) // 2, "prefix": f"{COARSE_PREFIX[:-1]}{s}_"})
    if grids[-1]["size"] != size:
        raise ValueError(f"input_domain.size ({size}) must be the largest of coarse_sizes {coarse_sizes}: "
                         "the input window is loaded for the coarse grids only")
    return grids


def loader_parameters(load_params):
    """``load_params`` as the month loader takes them: without ``input_domain`` and, when it is set,
    with ``size`` = the input-domain size. Unchanged (a copy) without the option."""
    settings = input_domain_settings(load_params)
    out = {k: v for k, v in load_params.items() if k != "input_domain"}
    if settings is not None:
        out["size"] = settings["size"]
    return out


def require_data_cache(parameters):
    """The option is applied by the month-cache pipeline only: refuse it on the standard one."""
    if input_domain_settings(parameters.get("train_load_data", {})) is not None \
            and not (parameters.get("data_cache", None) or {}).get("enabled", False):
        raise ValueError("train_load_data.input_domain needs the month cache: set \"data_cache\": {\"enabled\": true}")


def check_compatible(parameters):
    """Validate the option and refuse the combinations that are not handled.

    The east-west mirror resolves and flips channels by name and reflects the footprint grid, which
    the block layout of the coarse modes does not follow exactly, and the mass term of the footprint
    loss builds its regions on the model's grid, which in mode "full" is not the footprint window.
    """
    settings = input_domain_settings(parameters.get("train_load_data", {}))
    if settings is None:
        return None
    require_data_cache(parameters)
    if (parameters.get("augmentation", None) or {}).get("mirror_ew") is not None:
        raise ValueError("train_load_data.input_domain cannot be combined with augmentation.mirror_ew")
    if settings["mode"] == "full" and (parameters.get("loss_functions", {}) or {}).get("fp_mass_loss") is not None:
        raise ValueError("train_load_data.input_domain mode 'full' cannot be combined with loss_functions.fp_mass_loss")
    return settings


# ---------------------------------------------------------------
# The month, as the model trains on it
# ---------------------------------------------------------------

def crop_window(obj, offset, size):
    """The central ``size`` x ``size`` window of a (..., lat, lon, ...) DataArray / Dataset whose
    lat / lon coordinates are the window indices; the result is re-indexed 0..size-1."""
    out = obj.isel(lat=slice(offset, offset + size), lon=slice(offset, offset + size))
    return out.assign_coords(lat=np.arange(size), lon=np.arange(size))


def block_mean(values, factor):
    """Mean of ``factor`` x ``factor`` blocks of a (n, lat, lon, channels) array; float32 result,
    accumulated in float64."""
    n, n_lat, n_lon, c = values.shape
    if n_lat % factor or n_lon % factor:
        raise ValueError(f"a {n_lat} x {n_lon} window cannot be averaged in {factor} x {factor} blocks")
    blocks = values.reshape(n, n_lat // factor, factor, n_lon // factor, factor, c)
    return blocks.mean(axis=(2, 4), dtype=np.float64).astype(np.float32)


def _inputs_like(template, values, names):
    """A loader-style inputs DataArray (dims, fp_time, name, attrs as ``template``) holding
    ``values`` (n, lat, lon, channels) with the channel labels ``names``."""
    index = pd.MultiIndex.from_tuples([tuple(n) for n in names], names=_INDEX_NAMES)
    da = xr.DataArray(values, dims=_INPUT_DIMS,
                      coords={"fp_time": template.coords["fp_time"].variable,
                              "lat": np.arange(values.shape[1]), "lon": np.arange(values.shape[2])},
                      name=template.name, attrs=dict(template.attrs))
    return da.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "variable_name"))


def coarsen_inputs(inputs, factor, prefix="", skip=()):
    """Block-average loader inputs (fp_time, lat, lon, variable_name) in ``factor`` x ``factor``
    blocks; the wind direction and speed are recomputed from the block-mean wind components.

    Args:
        inputs (xr.DataArray): As the loader returns them (raw, unscaled).
        factor (int): Block size.
        prefix (str): Prepended to every variable name of the result (``"ctx_"`` when the result is
            appended to the usual inputs; ``""`` when it replaces them).
        skip (iterable of str): Variables left out of the result.

    Returns:
        xr.DataArray: (fp_time, lat / factor, lon / factor, variable_name), float32.
    """
    if tuple(inputs.dims) != _INPUT_DIMS:
        raise ValueError(f"inputs must have dims {_INPUT_DIMS}, got {inputs.dims}")
    names = [tuple(n) for n in inputs.variable_name.values]
    keep = [k for k, n in enumerate(names) if n[0] not in skip and not n[0].startswith("far_")]
    if not keep:
        raise ValueError("no input channel left to coarsen")
    values = np.ascontiguousarray(inputs.values)
    if values.dtype != np.float32:
        raise ValueError(f"inputs must be float32 (as the loader returns them), got {values.dtype}")
    coarse = block_mean(values if len(keep) == len(names) else values[..., keep], factor)
    kept = [names[k] for k in keep]
    position = {n: k for k, n in enumerate(kept)}
    for name in kept:
        variable, level, delta = name
        if variable not in ("wind_angle", "wind_speed"):
            continue
        u_key, v_key = ("x_wind", level, delta), ("y_wind", level, delta)
        if u_key not in position or v_key not in position:
            raise ValueError(f"{name}: x_wind / y_wind at level {level}, time delta {delta} are needed to "
                             "recompute the wind of the blocks")
        u, v = coarse[..., position[u_key]], coarse[..., position[v_key]]
        if variable == "wind_angle":
            coarse[..., position[name]] = np.arctan2(-u, -v)
        else:
            coarse[..., position[name]] = np.sqrt(u ** 2 + v ** 2)
    return _inputs_like(inputs, coarse, [(prefix + v, level, delta) for v, level, delta in kept])


def apply_input_domain(inputs, fp_xr, settings):
    """Turn a month loaded on the input window into the month the model trains on.

    Args:
        inputs (xr.DataArray): Loader inputs on the input window, (fp_time, lat, lon, variable_name).
        fp_xr (xr.Dataset): Footprints on the input window, (time, lat, lon), with the
            ``lat_coords`` / ``lon_coords`` coordinates of the window.
        settings (dict): From :func:`input_domain_settings`.

    Returns:
        tuple: ``(inputs, fp_xr)`` for the cache. "full": the inputs unchanged, ``fp`` set to NaN
        outside the footprint window. "coarse": the block-averaged inputs, the footprint window.
        "native_and_coarse": the footprint window's own inputs plus the block-averaged channels of
        every coarse grid (``settings["coarse"]``, in order), the footprint window.
    """
    size, fp_size, offset = settings["size"], settings["fp_size"], settings["offset"]
    for name, obj in (("inputs", inputs), ("footprints", fp_xr)):
        if (obj.sizes["lat"], obj.sizes["lon"]) != (size, size):
            raise ValueError(f"{name} were not loaded on the {size} x {size} input window: "
                             f"{obj.sizes['lat']} x {obj.sizes['lon']}")
        for dim in ("lat", "lon"):
            if not np.array_equal(obj[dim].values, np.arange(size)):
                raise ValueError(f"{name}: the {dim} coordinate is not the window index 0..{size - 1}")
    mode = settings["mode"]
    if mode == "full":
        inside = ((fp_xr.lat >= offset) & (fp_xr.lat < offset + fp_size)
                  & (fp_xr.lon >= offset) & (fp_xr.lon < offset + fp_size))
        fp_xr = fp_xr.copy()
        fp_xr["fp"] = fp_xr["fp"].where(inside).astype(fp_xr["fp"].dtype)
        return inputs, fp_xr
    fp_window = crop_window(fp_xr, offset, fp_size)
    if mode == "coarse":
        return coarsen_inputs(inputs, settings["factor"]), fp_window
    native = crop_window(inputs, offset, fp_size)
    parts = [native]
    for grid in settings["coarse"]:
        window = inputs if grid["size"] == size else crop_window(inputs, grid["offset"], grid["size"])
        parts.append(coarsen_inputs(window, grid["factor"], prefix=grid["prefix"], skip=COARSE_SKIP_VARIABLES))
    names = [tuple(n) for part in parts for n in part.variable_name.values]
    values = np.concatenate([np.ascontiguousarray(part.values) for part in parts], axis=-1)
    return _inputs_like(inputs, values, names), fp_window


# ---------------------------------------------------------------
# Trainer side: scoring on the footprint window (mode "full")
# ---------------------------------------------------------------

def resolve_fp_window(parameters, fp_xr):
    """Check the loaded footprints against the option and record it in the parameters.

    Writes ``parameters["input_domain_resolved"]`` (saved with the training settings; see
    :func:`require_resolved`). Returns the footprint window ``{"offset", "size", "n_lat", "n_lon"}``
    to crop the model's outputs with in mode "full", else None.
    """
    settings = input_domain_settings(parameters.get("train_load_data", {}))
    if settings is None:
        parameters.pop("input_domain_resolved", None)
        return None
    n_lat, n_lon = int(fp_xr.sizes["lat"]), int(fp_xr.sizes["lon"])
    expected = settings["size"] if settings["mode"] == "full" else settings["fp_size"]
    if (n_lat, n_lon) != (expected, expected):
        raise ValueError(f"input_domain mode {settings['mode']!r}: the footprints should be on a {expected} x "
                         f"{expected} window, got {n_lat} x {n_lon} (was the data loaded through the month cache?)")
    resolved = {"mode": settings["mode"], "input_size": settings["size"], "fp_size": settings["fp_size"]}
    if settings["mode"] == "full":
        resolved.update({"offset": settings["offset"], "n_lat": n_lat, "n_lon": n_lon})
    if settings.get("coarse"):
        resolved["coarse_sizes"] = [grid["size"] for grid in settings["coarse"]]
    parameters["input_domain_resolved"] = resolved
    if settings["mode"] == "full":
        print(f"Input domain {settings['size']} x {settings['size']} (mode full): the footprint is scored and "
              f"exported on its central {settings['fp_size']} x {settings['fp_size']} window")
        return {"offset": settings["offset"], "size": settings["fp_size"], "n_lat": n_lat, "n_lon": n_lon}
    grids = "" if not settings.get("coarse") else (
        "; coarse grids " + ", ".join(f"{g['size']} x {g['size']} in {g['factor']} x {g['factor']} blocks "
                                      f"as {g['prefix']}*" for g in settings["coarse"]))
    print(f"Input domain {settings['size']} x {settings['size']} (mode {settings['mode']}) on the "
          f"{settings['fp_size']} x {settings['fp_size']} footprint grid{grids}")
    return None


def fp_window(parameters):
    """The footprint window recorded by :func:`resolve_fp_window` in mode "full", else None."""
    resolved = parameters.get("input_domain_resolved", None)
    if not resolved or resolved.get("mode") != "full":
        return None
    return {"offset": resolved["offset"], "size": resolved["fp_size"],
            "n_lat": resolved["n_lat"], "n_lon": resolved["n_lon"]}


def crop_fp_dataset(ds, window):
    """The footprint window of a (time, lat, lon) footprint Dataset (no-op when ``window`` is None)."""
    if window is None:
        return ds
    return crop_window(ds, window["offset"], window["size"])


def crop_fp_predictions(flat, parameters):
    """The footprint window of flattened per-node predictions (n, n_lat * n_lon), latitude-major, as
    the trainers reshape them; returns ``flat`` unchanged unless the run is in mode "full"."""
    window = fp_window(parameters)
    if window is None:
        return flat
    n, o, size = flat.shape[0], window["offset"], window["size"]
    grid = np.asarray(flat).reshape(n, window["n_lat"], window["n_lon"])
    return np.ascontiguousarray(grid[:, o:o + size, o:o + size]).reshape(n, size * size)


def require_resolved(parameters):
    """Fail loudly when mode "full" is requested but the trainer never resolved the footprint window
    (it would score the whole input window, with untrained cells, instead of the footprint)."""
    settings = input_domain_settings(parameters.get("train_load_data", {}))
    if settings is not None and settings["mode"] == "full" \
            and (parameters.get("input_domain_resolved", None) or {}).get("mode") != "full":
        raise ValueError("train_load_data.input_domain mode 'full' is set but this trainer does not score the "
                         "footprint window (train_dual_headlr_model and predict_dual_model do)")
