import numpy as np


import torch
import xarray as xr



# ---------------------------------------------------------------------------
# Private helpers supporting mainly evaluation/metrics.py
# ---------------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    """Convert tensor / DataArray / array-like to a numpy array.

    Args:
        arr (torch.Tensor, xr.DataArray, or array-like): Value to convert.

    Returns:
        np.ndarray: The converted array.
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if isinstance(arr, xr.DataArray):
        return arr.values
    return np.asarray(arr)


def _spatial_shape_from_xarray(da) -> tuple[int, int]:
    """Infer (H, W) from a DataArray's dimension names.

    Args:
        da (xr.DataArray): DataArray with dimensions containing "lat" and "lon"
            (case-insensitive, substring match).

    Returns:
        tuple[int, int]: ``(H, W)`` sizes of the lat and lon dimensions.

    Raises:
        ValueError: If no dimension name contains "lat" or none contains "lon".
    """
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

    Args:
        arr (np.ndarray): Array of shape (H, W), (HW,), (N, H, W), or (N, HW).
        spatial_shape (tuple, optional): (H, W) tuple — required when ``arr`` is
            flat (1D or 2D with N > 1). If None and ``arr`` is 2D, it is assumed to
            already be (H, W). Defaults to None.

    Returns:
        np.ndarray: Shape (N, H, W).

    Raises:
        ValueError: If ``arr`` is 1D and ``spatial_shape`` is None, or if ``arr``
            does not have 1, 2, or 3 dimensions.
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

    Args:
        ignore_mask (np.ndarray, torch.Tensor, or xr.DataArray): Mask to normalize.
        spatial_shape (tuple, optional): (H, W) tuple, required for flat inputs that
            are not xarray DataArrays (see ``_to_batched_spatial``).

    Returns:
        np.ndarray: Boolean array of shape (N, H, W).
    """
    if isinstance(ignore_mask, xr.DataArray) and spatial_shape is None:
        spatial_shape = _spatial_shape_from_xarray(ignore_mask)
    return _to_batched_spatial(_to_numpy(ignore_mask).astype(bool), spatial_shape)


def _resolve_inputs(true, pred, spatial_shape):
    """Convert true/pred to (N, H, W) numpy, inferring spatial_shape from xarray if not provided.

    Args:
        true (np.ndarray, torch.Tensor, or xr.DataArray): Ground truth array.
        pred (np.ndarray, torch.Tensor, or xr.DataArray): Predicted array.
        spatial_shape (tuple, optional): (H, W) tuple, required for flat inputs that
            are not xarray DataArrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(true_np, pred_np)``, both shape (N, H, W).

    Raises:
        ValueError: If ``true`` and ``pred`` have mismatched shapes after normalization.
    """
    if isinstance(true, xr.DataArray) and spatial_shape is None:
        spatial_shape = _spatial_shape_from_xarray(true)

    true_np = _to_batched_spatial(_to_numpy(true), spatial_shape)
    pred_np = _to_batched_spatial(_to_numpy(pred), spatial_shape)

    if true_np.shape != pred_np.shape:
        raise ValueError(
            f"Shape mismatch after normalization: true={true_np.shape}, pred={pred_np.shape}"
        )
    return true_np, pred_np

def valid_mask(*arrays: np.ndarray, threshold=None, nonzero=False, ignore_mask=None, first_array_threshold=None) -> np.ndarray:
    """Boolean mask that is True where all arrays are finite and optionally above a threshold or non-zero.

    Args:
        *arrays (np.ndarray): One or more numpy arrays of the same shape.
        threshold (float, optional): If provided, mask requires all arrays >
            threshold. Defaults to None.
        nonzero (bool, optional): If True, mask requires ``arr != 0`` in addition to
            being finite. Defaults to False.
        ignore_mask (np.ndarray, optional): Boolean array of the same shape as the
            input arrays. True means ignore (exclude from mask). Defaults to None.
        first_array_threshold (float, optional): If provided, mask requires only the
            first array > ``first_array_threshold``. Use this, for example, to
            calculate metric scores for each value bin in the true footprint.
            Defaults to None.

    Returns:
        np.ndarray: Boolean array of the same shape as the inputs.
    """
    mask = np.ones(arrays[0].shape, dtype=bool)
    for i, arr in enumerate(arrays):
        mask &= np.isfinite(arr) & ~np.isnan(arr)
        if threshold is not None:
            mask &= arr > threshold
        elif nonzero:
            mask &= arr != 0
        if i == 0 and first_array_threshold is not None:
            mask &= arr > first_array_threshold
    if ignore_mask is not None:
        mask &= ~ignore_mask
    return mask