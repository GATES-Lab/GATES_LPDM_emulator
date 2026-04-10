import numpy as np
from sklearn.metrics import r2_score

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
# Private helpers supporting mainly evaluation/metrics.py
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