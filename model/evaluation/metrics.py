"""
Shape- and type-agnostic metric functions for footprint evaluation.

Supports numpy arrays, PyTorch tensors, and xarray DataArrays as inputs.
Supports spatial (H, W), flat (HW,), batched spatial (N, H, W), and
batched flat (N, HW) shapes.

All public functions return numpy scalars or (N,) numpy arrays.
"""

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import xarray as xr
    _XR_AVAILABLE = True
except ImportError:
    _XR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    """Convert tensor / DataArray / array-like to a numpy array."""
    if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if _XR_AVAILABLE and isinstance(arr, xr.DataArray):
        return arr.values
    return np.asarray(arr)


def _spatial_shape_from_xarray(da) -> tuple[int, int]:
    """Infer (H, W) from a DataArray's dimension names."""
    lat_dim = next((d for d in da.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in da.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Cannot infer spatial shape: expected dims containing 'lat'/'lon', "
            f"got {list(da.dims)}"
        )
    return da.sizes[lat_dim], da.sizes[lon_dim]


def _to_batched_spatial(arr: np.ndarray, spatial_shape=None) -> np.ndarray:
    """Reshape any footprint array to canonical (N, H, W).

    Parameters
    ----------
    arr:
        Array of shape (H, W), (HW,), (N, H, W), or (N, HW).
    spatial_shape:
        (H, W) tuple — required when arr is flat (1D or 2D with N > 1).
        If None and arr is 2D, it is assumed to already be (H, W).

    Returns
    -------
    np.ndarray of shape (N, H, W).
    """
    if arr.ndim == 3:
        return arr  # already (N, H, W)

    if arr.ndim == 2:
        if spatial_shape is None:
            # Treat as a single (H, W) footprint
            return arr[np.newaxis]
        else:
            # Treat as (N, HW) batch-flat
            return arr.reshape(-1, *spatial_shape)

    if arr.ndim == 1:
        if spatial_shape is None:
            raise ValueError(
                "spatial_shape=(H, W) is required to reshape a 1D flat footprint."
            )
        return arr.reshape(1, *spatial_shape)

    raise ValueError(f"Expected ndim in {{1, 2, 3}}, got ndim={arr.ndim}.")


def _normalize_ignore_mask(ignore_mask, spatial_shape) -> np.ndarray:
    """Convert an ignore_mask of any type/shape to (N, H, W) boolean numpy.

    Accepts the same types and shapes as footprint arrays. True means ignore.
    """
    if _XR_AVAILABLE and isinstance(ignore_mask, xr.DataArray) and spatial_shape is None:
        spatial_shape = _spatial_shape_from_xarray(ignore_mask)
    return _to_batched_spatial(_to_numpy(ignore_mask).astype(bool), spatial_shape)


def _resolve_inputs(true, pred, spatial_shape):
    """Convert true/pred to (N, H, W) numpy, inferring spatial_shape from
    xarray if not provided."""
    if _XR_AVAILABLE and isinstance(true, xr.DataArray) and spatial_shape is None:
        spatial_shape = _spatial_shape_from_xarray(true)

    true_np = _to_batched_spatial(_to_numpy(true), spatial_shape)
    pred_np = _to_batched_spatial(_to_numpy(pred), spatial_shape)

    if true_np.shape != pred_np.shape:
        raise ValueError(
            f"Shape mismatch after normalization: true={true_np.shape}, pred={pred_np.shape}"
        )
    return true_np, pred_np


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def valid_mask(*arrays: np.ndarray, threshold=None, nonzero=False, ignore_mask=None) -> np.ndarray:
    """Boolean mask that is True where all arrays are finite and optionally above a threshold or non-zero.

    Parameters
    ----------
    *arrays:
        One or more numpy arrays of the same shape.
    threshold:
        If provided, mask requires arr > threshold.
    nonzero:
        If True, mask requires arr != 0 in addition to being finite.
    ignore_mask:
        Boolean numpy array of the same shape as the input arrays.
        True means ignore (exclude from mask).

    Returns
    -------
    Boolean numpy array of the same shape as the inputs.
    """
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr) & ~np.isnan(arr)
        if threshold is not None:
            mask &= arr > threshold
        elif nonzero:
            mask &= arr != 0
    if ignore_mask is not None:
        mask &= ~ignore_mask
    return mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou(true, pred, *, threshold=0, spatial_shape=None, ignore_mask=None, reduce="mean", nonzero=None):
    """Intersection-over-Union for binary footprint masks.

    Parameters
    ----------
    true, pred:
        Footprint arrays. Accepted types: numpy, tensor, xarray DataArray.
        Accepted shapes: (H, W), (HW,), (N, H, W), (N, HW).
    threshold:
        Value above which a cell is considered "active". Default 0.
    spatial_shape:
        (H, W) — required only for flat (HW,) or (N, HW) inputs that are
        not xarray DataArrays.
    ignore_mask:
        Areas to exclude. Same type/shape flexibility as true/pred.
        True means ignore.
    reduce:
        "mean" to return a scalar, None to return per-sample (N,) array.
    nonzero:
        Ignored parameter, used for compatibility with compute_metrics kwargs. To exclude zero cells, set threshold > 0.

    Returns
    -------
    float or np.ndarray of shape (N,).
    """
    true_np, pred_np = _resolve_inputs(true, pred, spatial_shape)

    ignore_np = _normalize_ignore_mask(ignore_mask, spatial_shape) if ignore_mask is not None else None

    true_bin = true_np > threshold
    pred_bin = pred_np > threshold
    if ignore_np is not None:
        true_bin = true_bin & ~ignore_np
        pred_bin = pred_bin & ~ignore_np

    # Sum over spatial dims (H, W), keeping batch dim N
    intersection = np.nansum(true_bin & pred_bin, axis=(1, 2))
    union = np.nansum(true_bin | pred_bin, axis=(1, 2))

    # Avoid division by zero for empty footprints
    scores = np.where(union > 0, intersection / union, np.nan)

    return float(np.nanmean(scores)) if reduce == "mean" else scores


def mse(true, pred, *, log_transform=False, spatial_shape=None, ignore_mask=None,
        nonzero=False, reduce="mean"):
    """Mean Squared Error, computed on valid cells.

    Parameters
    ----------
    true, pred:
        Footprint arrays. Accepted types: numpy, tensor, xarray DataArray.
        Accepted shapes: (H, W), (HW,), (N, H, W), (N, HW).
    log_transform:
        If True, apply log10 before computing MSE. Requires positive valid values.
    spatial_shape:
        (H, W) — required only for flat inputs that are not xarray DataArrays.
    ignore_mask:
        Areas to exclude. Same type/shape flexibility as true/pred. True means ignore.
    nonzero:
        If True, exclude cells where either array is zero.
    reduce:
        "mean" to return a scalar, None to return per-sample (N,) array.

    Returns
    -------
    float or np.ndarray of shape (N,).
    """
    true_np, pred_np = _resolve_inputs(true, pred, spatial_shape)
    ignore_np = _normalize_ignore_mask(ignore_mask, spatial_shape) if ignore_mask is not None else None

    scores = np.full(true_np.shape[0], np.nan)
    for i in range(true_np.shape[0]):
        ig = ignore_np[i] if ignore_np is not None else None
        mask = valid_mask(true_np[i], pred_np[i], nonzero=nonzero, ignore_mask=ig)
        if not np.any(mask):
            continue
        t, p = true_np[i][mask], pred_np[i][mask]
        if log_transform:
            t, p = np.log10(t), np.log10(p)
        scores[i] = np.mean((t - p) ** 2)

    return float(np.nanmean(scores)) if reduce == "mean" else scores


def mae(true, pred, *, spatial_shape=None, ignore_mask=None,
        nonzero=False, reduce="mean"):
    """Mean Absolute Error, computed on valid cells.

    Parameters
    ----------
    true, pred:
        Footprint arrays. Accepted types: numpy, tensor, xarray DataArray.
        Accepted shapes: (H, W), (HW,), (N, H, W), (N, HW).
    spatial_shape:
        (H, W) — required only for flat inputs that are not xarray DataArrays.
    ignore_mask:
        Areas to exclude. Same type/shape flexibility as true/pred. True means ignore.
    nonzero:
        If True, exclude cells where either array is zero.
    reduce:
        "mean" to return a scalar, None to return per-sample (N,) array.

    Returns
    -------
    float or np.ndarray of shape (N,).
    """
    true_np, pred_np = _resolve_inputs(true, pred, spatial_shape)
    ignore_np = _normalize_ignore_mask(ignore_mask, spatial_shape) if ignore_mask is not None else None

    scores = np.full(true_np.shape[0], np.nan)
    for i in range(true_np.shape[0]):
        ig = ignore_np[i] if ignore_np is not None else None
        mask = valid_mask(true_np[i], pred_np[i], nonzero=nonzero, ignore_mask=ig)
        if not np.any(mask):
            continue
        t, p = true_np[i][mask], pred_np[i][mask]
        scores[i] = np.mean(np.abs(t - p))

    return float(np.nanmean(scores)) if reduce == "mean" else scores


def nmae(true, pred, *, spatial_shape=None, ignore_mask=None,
         nonzero=False, reduce="mean"):
    """Normalised Mean Absolute Error: sum(|true - pred|) / sum(true) per sample.

    Normalising by the sum of true values makes this scale-independent and
    interpretable as the fractional total error relative to the true signal.

    Parameters
    ----------
    true, pred:
        Footprint arrays. Accepted types: numpy, tensor, xarray DataArray.
        Accepted shapes: (H, W), (HW,), (N, H, W), (N, HW).
    spatial_shape:
        (H, W) — required only for flat inputs that are not xarray DataArrays.
    ignore_mask:
        Areas to exclude. Same type/shape flexibility as true/pred. True means ignore.
    nonzero:
        If True, exclude cells where either array is zero.
    reduce:
        "mean" to return a scalar, None to return per-sample (N,) array.

    Returns
    -------
    float or np.ndarray of shape (N,).
    """
    true_np, pred_np = _resolve_inputs(true, pred, spatial_shape)
    ignore_np = _normalize_ignore_mask(ignore_mask, spatial_shape) if ignore_mask is not None else None

    scores = np.full(true_np.shape[0], np.nan)
    for i in range(true_np.shape[0]):
        ig = ignore_np[i] if ignore_np is not None else None
        mask = valid_mask(true_np[i], pred_np[i], nonzero=nonzero, ignore_mask=ig)
        if not np.any(mask):
            continue
        t, p = true_np[i][mask], pred_np[i][mask]
        norm = np.sum(t)
        if norm == 0:
            continue
        scores[i] = np.sum(np.abs(t - p)) / norm

    return float(np.nanmean(scores)) if reduce == "mean" else scores


def corrcoef(true, pred, *, log_transform=False, spatial_shape=None, ignore_mask=None,
             nonzero=False, reduce="mean"):
    """Pearson correlation coefficient, computed on valid cells.

    Parameters
    ----------
    true, pred:
        Footprint arrays. Accepted types: numpy, tensor, xarray DataArray.
        Accepted shapes: (H, W), (HW,), (N, H, W), (N, HW).
    log_transform:
        If True, apply log10 before computing the correlation. Requires positive valid values.
    spatial_shape:
        (H, W) — required only for flat inputs that are not xarray DataArrays.
    ignore_mask:
        Areas to exclude. Same type/shape flexibility as true/pred. True means ignore.
    nonzero:
        If True, exclude cells where either array is zero.
    reduce:
        "mean" to return a scalar, None to return per-sample (N,) array.

    Returns
    -------
    float or np.ndarray of shape (N,).
    """
    true_np, pred_np = _resolve_inputs(true, pred, spatial_shape)
    ignore_np = _normalize_ignore_mask(ignore_mask, spatial_shape) if ignore_mask is not None else None

    scores = np.full(true_np.shape[0], np.nan)
    for i in range(true_np.shape[0]):
        ig = ignore_np[i] if ignore_np is not None else None
        mask = valid_mask(true_np[i], pred_np[i], nonzero=nonzero, ignore_mask=ig)
        if np.sum(mask) < 2:  # corrcoef needs at least 2 points
            continue
        t, p = true_np[i][mask], pred_np[i][mask]
        if log_transform:
            t, p = np.log10(t), np.log10(p)
        scores[i] = np.corrcoef(t, p)[0, 1]

    return float(np.nanmean(scores)) if reduce == "mean" else scores


def compute_metrics(true, pred, metrics=["iou", "mse", "mae", "nmae", "corrcoef", "corrcoef_log"], spatial_shape=None, **kwargs):
    """Compute multiple metrics at once and return as a dictionary.
    
    Args:
        true: Ground truth footprint, or footprint dataset (as array, xarray or tensor).
        pred: Predicted footprint array (as above)
        metrics: List of metric names to compute. Supported: "iou", "mse", "mae", "nmae", "corrcoef", "corrcoef_log".
        spatial_shape: (H, W) tuple, required if true/pred are flat arrays that are not xarray DataArrays. Recommended when possible for better error checking, but will be inferred from dims if not provided. 
        **kwargs: Additional keyword arguments to pass to each metric function (e.g. threshold, ignore_mask).

    """
    metric_funcs = {
        "iou": iou,
        "mse": mse,
        "mae": mae,
        "nmae": nmae,
        "corrcoef": corrcoef,
        "corrcoef_log": corrcoef,
    }
    results = {}
    for metric in metrics:
        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric '{metric}'. Supported: {list(metric_funcs.keys())}")
        metric_kwargs = dict(kwargs)
        if metric != "iou":
            metric_kwargs.pop("threshold", None)
        metric_kwargs.pop("log_transform", None)
        if metric == "corrcoef":
            metric_kwargs["log_transform"] = False
        elif metric == "corrcoef_log":
            metric_kwargs["log_transform"] = True
        results[metric] = metric_funcs[metric](true, pred, spatial_shape=spatial_shape, **metric_kwargs)
    return results


##### Flux related metrics 

def calculate_mf(fp, fluxes):
    """
    Calculate the mole fraction (mf) by multiplying the footprint (fp) with the fluxes and summing over the spatial dimensions (lat, lon).

    Inputs:
    - fp: xarray DataArray or Dataset containing the footprint values. Must have dimensions 'time', 'lat', and 'lon'. Variables should contain 'fp' in their name (e.g. 'fp', 'fp_pred') to be included in the calculation.
    - fluxes: xarray DataArray containing the flux values. Must have dimensions 'time', 'lat', and 'lon' that match those of the footprint. Use cut_flux_data() to crop the fluxes to match the footprint dimensions.
    """

    if not fp.time.equals(fluxes.time):
        raise ValueError("Time dimension of footprint and fluxes must match. \n  Crop the fluxes with cut_flux_data() to match the footprint dimensions.")

    if not fp.lat.equals(fluxes.lat) or not fp.lon.equals(fluxes.lon):
        raise ValueError("Spatial dimensions of footprint and fluxes must match. \n  Crop the fluxes with cut_flux_data() to match the footprint dimensions.")

    def _mf_key(name):
        suffix = "_".join(p for p in name.split("_") if p != "fp")
        return f"mf_{suffix}" if suffix else "mf"

    mfs = {}

    if isinstance(fp, xr.DataArray):
        var_name = fp.name or "fp"
        var_name = _mf_key(var_name)
        mfs[var_name] = (fp * fluxes).sum(dim=["lat", "lon"]).flux
    elif isinstance(fp, xr.Dataset):
        for var in fp.data_vars:
            if "fp" in var.split("_"):
                var_name = _mf_key(var)
                mfs[var_name] = (fp[var] * fluxes).sum(dim=["lat", "lon"]).flux
    else:
        raise ValueError("Unsupported type for fp. Expected xarray DataArray or Dataset.")
    
    return xr.Dataset(mfs)
   