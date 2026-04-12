"""
Shape- and type-agnostic metric functions for footprint evaluation.

Supports numpy arrays, PyTorch tensors, and xarray DataArrays as inputs.
Supports spatial (H, W), flat (HW,), batched spatial (N, H, W), and
batched flat (N, HW) shapes.

All public functions return numpy scalars or (N,) numpy arrays.
"""

import numpy as np
from sklearn.metrics import r2_score

import torch
import xarray as xr


from gates.utils.shape_utils import _to_numpy, _spatial_shape_from_xarray, _to_batched_spatial, _resolve_inputs, _normalize_ignore_mask, valid_mask


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


def compute_footprint_metrics(true, pred, metrics=["iou", "mse", "mae", "nmae", "corrcoef", "corrcoef_log"], spatial_shape=None, **kwargs):
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


def compute_mfs_metrics(mf_true, mf_pred):
    """Compute summary metrics for two aligned mole-fraction time series.

    Parameters
    ----------
    mf_true, mf_pred:
        xarray DataArrays with a shared time coordinate or two 1D numpy arrays of the same shape.

    Returns
    -------
    dict
        Dictionary with keys:
        - "corrcoef": Pearson correlation coefficient from ``numpy.corrcoef``.
        - "mean_absolute_error": mean of ``|mf_pred - mf_true|``.
        - "mean_bias": mean of ``mf_pred - mf_true``.
        - "true_mean": mean of ``mf_true``.
        - "predicted_mean": mean of ``mf_pred``.
        - "r2_score" from sklearn 
    """
    if isinstance(mf_true, xr.DataArray) and isinstance(mf_pred, xr.DataArray):
        if not mf_true.time.equals(mf_pred.time):
            raise ValueError("mf_true and mf_pred must share the same time coordinates.")

        true_values = np.asarray(mf_true.values)
        pred_values = np.asarray(mf_pred.values)
    
    elif isinstance(mf_true, np.ndarray) and isinstance(mf_pred, np.ndarray):
        if mf_true.shape != mf_pred.shape:
            raise ValueError(f"Shape mismatch: mf_true.shape={mf_true.shape}, mf_pred.shape={mf_pred.shape}")
        ## assert they are 1d arrays
        if mf_true.ndim != 1 or mf_pred.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got mf_true.ndim={mf_true.ndim}, mf_pred.ndim={mf_pred.ndim}")
        
        true_values = mf_true
        pred_values = mf_pred
    else:
        raise NotImplementedError("Currently only xarray DataArrays with time coordinates or 1D numpy arrays are supported for compute_mfs_metrics.")
    
    valid_mask = np.isfinite(true_values) & np.isfinite(pred_values)

    if np.sum(valid_mask) < 2:
        raise ValueError("At least 2 valid points are required!")
    else:
        corrcoef = np.corrcoef(true_values[valid_mask], pred_values[valid_mask])[0, 1]

    mean_absolute_error = np.mean(np.abs(pred_values[valid_mask] - true_values[valid_mask]))
    mean_bias = np.mean(pred_values[valid_mask] - true_values[valid_mask])
    true_mean = np.mean(true_values[valid_mask])
    predicted_mean = np.mean(pred_values[valid_mask])
    
    r2 = r2_score(true_values[valid_mask], pred_values[valid_mask])


    return {
        "corrcoef": float(corrcoef),
        "mean_absolute_error": float(mean_absolute_error),
        "mean_bias": float(mean_bias),
        "true_mean": float(true_mean),
        "predicted_mean": float(predicted_mean),
        "r2_score": float(r2),
    }


##### Flux related metrics 

def calculate_mfs(fp, fluxes, transform_factor=None, spatial_shape=None):
    """
    Calculate the mole fraction (mf) by multiplying the footprint (fp) with the fluxes and summing over the spatial dimensions (lat, lon). 

    Inputs:
    - fp: Footprint array. Can be a numpy array, xarray DataArray, or xarray Dataset. If xarray Dataset, variables containing 'fp' in their name will be included in the calculation (e.g. 'fp', 'fp_pred') and treated separately.
    - fluxes: Flux array. Can be a numpy array, xarray DataArray, or xarray Dataset. If xarray Dataset, it must contain a variable named 'flux' which will be used in the calculation. The spatial dimensions (lat, lon) must match those of the footprint. Use cut_flux_data() to crop the fluxes to match the footprint dimensions.
    - transform_factor: Optional factor to change the units of the resulting mole fraction. If "default", it will be attempted to extract the factor from the fluxes units. If None, no transformation is applied.
    - spatial_shape: (H, W) tuple, required if fp and fluxes are flat numpy arrays. Recommended when possible for better error checking.

    Returns:
    - If inputs are xarray DataArrays or Datasets, returns an xarray Dataset containing the calculated mole fractions for each variable. If inputs are numpy arrays, returns a numpy array of shape (N,) containing the mole fractions for each sample in the batch.
    """
    if isinstance(fp, (xr.DataArray, xr.Dataset)) and isinstance(fluxes, (xr.DataArray, xr.Dataset)):
        print("going here")
        return calculate_mfs_xarray(fp, fluxes, transform_factor)

    else:

        fp = _to_batched_spatial(_to_numpy(fp), spatial_shape)
        fluxes = _to_batched_spatial(_to_numpy(fluxes), spatial_shape)

        if fp.shape != fluxes.shape:
            raise ValueError(f"Shape mismatch after normalization: fp={fp.shape}, fluxes={fluxes.shape}")
        mfs = np.nansum(fp * fluxes, axis=(1, 2))

        if transform_factor is not None:
            if transform_factor == "default":
                raise NotImplementedError("Default transform factor is only implemented for xarray inputs with units attribute.")
            elif not isinstance(transform_factor, (int, float)):
                raise ValueError("transform_factor must be a number or 'default'.")
            mfs = mfs * transform_factor
        
        return mfs
    

def calculate_mfs_xarray(fp, fluxes, transform_factor=None):
    """
    Calculate the mole fraction (mf) by multiplying the footprint (fp) with the fluxes and summing over the spatial dimensions (lat, lon).

    Inputs:
    - fp: xarray DataArray or Dataset containing the footprint values. Must have dimensions 'time', 'lat', and 'lon'. Variables should contain 'fp' in their name (e.g. 'fp', 'fp_pred') to be included in the calculation.
    - fluxes: xarray DataArray containing the flux values. Must have dimensions 'time', 'lat', and 'lon' that match those of the footprint. Use cut_flux_data() to crop the fluxes to match the footprint dimensions.
    - transform_factor: Optional factor to change the units of the resulting mole fraction. If "default", it will be attempted to extract the factor from the fluxes units. If None, no transformation is applied.

    Returns:
    - xarray Dataset containing the calculated mole fractions for each variable in the footprint dataset
    """
    if not isinstance(fp, (xr.DataArray, xr.Dataset)) or not isinstance(fluxes, (xr.DataArray, xr.Dataset)):
        raise ValueError("fp must be an xarray DataArray or Dataset, and fluxes must be an xarray DataArray or Dataset.")

    if isinstance(fluxes, xr.Dataset):
        if "flux" not in fluxes.data_vars:
            raise ValueError("If fluxes is an xarray Dataset, it must contain a variable named 'flux'.")
        fluxes = fluxes.flux

    if not fp.time.equals(fluxes.time):
        raise ValueError("Time dimension of footprint and fluxes must match. \n  Crop the fluxes with cut_flux_data() to match the footprint dimensions.")

    if not fp.lat.equals(fluxes.lat) or not fp.lon.equals(fluxes.lon):
        raise ValueError("Spatial dimensions of footprint and fluxes must match. \n  Crop the fluxes with cut_flux_data() to match the footprint dimensions.")

    def _mf_key(name):
        suffix = "_".join(p for p in name.split("_") if p != "fp")
        return f"mf_{suffix}" if suffix else "mf"

    mfs = {}

    ## create the new variable label by removing "fp" from the original variable name, e.g. "fp_pred" -> "mf_pred", "fp" -> "mf"
    if isinstance(fp, xr.DataArray):
        var_name = fp.name or "fp"
        var_name = _mf_key(var_name)
        mfs[var_name] = (fp * fluxes).sum(dim=["lat", "lon"])
    elif isinstance(fp, xr.Dataset):
        for var in fp.data_vars:
            if "fp" in var.split("_"):
                var_name = _mf_key(var)
                mfs[var_name] = (fp[var] * fluxes).sum(dim=["lat", "lon"])

    
    flux_attrs = fluxes.attrs.copy() if hasattr(fluxes, "attrs") else {}
    
    ## apply the transform factor to the mfs and update the units attribute if it exists. If transform_factor is "default", attempt to infer the factor from the fluxes units (e.g. if units are "mol/m2/s", apply a factor of 1e9 to convert to ppb)
    if transform_factor is not None:
        if transform_factor == "default":
            if hasattr(fluxes, "units"):
                if "units" in flux_attrs:
                    units = flux_attrs["units"]
                    if units=="mol/m2/s":
                        transform_factor = 1e9  # convert to ppb
                    else:
                        print(f"Warning: Unrecognized flux units {units}. No transformation applied.")
            else:
                raise ValueError("transform_factor='default' requires fluxes to have 'units' attribute.")

        elif not isinstance(transform_factor, (int, float)):
            raise ValueError("transform_factor must be a number or 'default'.")
        for key in mfs:
            mfs[key] = mfs[key] * transform_factor
        
        # write the number in scientific notation to the units attribute, e.g. "1e9 * mol/m2/s"
        transform_factor_string = f"{transform_factor:.0e}" if isinstance(transform_factor, (int, float)) else "unknown"

        if "units" in flux_attrs:
            flux_attrs["units"] = f"{transform_factor_string} * {flux_attrs.get('units')} " 
        else:
            flux_attrs["units"] = f"{transform_factor_string} * unknown_units"

        
    mfs_dataset= xr.Dataset(mfs)
    mfs_dataset.attrs.update(flux_attrs)

    return mfs_dataset



def compute_static_mf_metrics(fp_true, fp_pred, spatial_shape=None, flux_patterns=None, transform_factor=None):
    """Compute mole-fraction metrics using static, hand-designed flux patterns.

    For each flux pattern the footprints are multiplied by the flux and summed
    over the spatial domain (via calculate_mfs), and the resulting mole-fraction
    time series are compared with compute_mfs_metrics.

    Parameters
    ----------
    fp : xr.Dataset, xr.DataArray, or np.ndarray
        True (and optionally predicted) footprints.
        - xr.Dataset: must contain variables 'fp' and 'fp_pred'.
        - xr.DataArray: treated as the true footprint; fp_pred must be supplied.
        - np.ndarray: treated as the true footprint; fp_pred must be supplied.
          Shape (H, W), (N, H, W) or flat variants — see spatial_shape.
    fp_pred : xr.DataArray or np.ndarray, optional
        Predicted footprints. Required when fp is not an xr.Dataset that
        already contains fp_pred.
    spatial_shape : tuple (H, W), optional
        Required when fp / fp_pred are flat numpy arrays.
    flux_patterns : dict[str, np.ndarray], optional
        Mapping of label -> 2-D flux array of shape (H, W).
        If None, the following defaults are used:
        - "uniform"     : np.ones((H, W))
        - "checkerboard": 0/1 checkerboard of shape (H, W)
        - "checkerboard_10": 0/1 checkerboard of shape (H, W) with 10x10 blocks
        - "checkerboard_25": 0/1 checkerboard of shape (H, W) with 25x25 blocks
        - "checkerboard_50": 0/1 checkerboard of shape (H, W) with 50x50 blocks (only if H>100 and W>100)
    transform_factor : float or "default", optional
        Passed through to calculate_mfs.

    Returns
    -------
    dict[str, dict]
        {label: metrics_dict} where each metrics_dict is the output of
        compute_mfs_metrics (keys: corrcoef, mean_absolute_error, mean_bias,
        true_mean, predicted_mean).
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve inputs to a (true, pred) pair
    # ------------------------------------------------------------------ #
    if isinstance(fp_true, xr.Dataset) or isinstance(fp_pred, xr.Dataset):
        raise ValueError("xarray Datasets are not supported for fp_true or fp_pred. Use xr.DataArrays instead.")
    elif isinstance(fp_true, xr.DataArray) and isinstance(fp_pred, xr.DataArray):
        spatial_shape = _spatial_shape_from_xarray(fp_true)
        fp = _to_batched_spatial(_to_numpy(fp_true), spatial_shape)
        fp_pred= _to_batched_spatial(_to_numpy(fp_pred), spatial_shape)
        H, W = spatial_shape

    elif isinstance(fp_true, np.ndarray) and isinstance(fp_pred, np.ndarray):
        fp = _to_batched_spatial(_to_numpy(fp_true), spatial_shape)
        fp_pred= _to_batched_spatial(_to_numpy(fp_pred), spatial_shape)
        H, W = fp.shape[1], fp.shape[2]

    else:
        raise ValueError("Unsupported combination of input types for fp_true and fp_pred. Both must be either xarray DataArrays or numpy arrays.")

    # ------------------------------------------------------------------ #
    # 2. Determine spatial shape (H, W) and build default flux patterns
    # ------------------------------------------------------------------ #

    if flux_patterns is None:
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        checkerboard = ((rows + cols) % 2).astype(float)
        checkerboard_10 = ((rows // 10) + (cols // 10)) % 2
        checkerboard_25 = ((rows // 25) + (cols // 25)) % 2

        flux_patterns = {
            "uniform": np.ones((H, W)),
            "checkerboard": checkerboard,
            "checkerboard_10": checkerboard_10.astype(float),
            "checkerboard_25": checkerboard_25.astype(float),
        }

        if H>100 and W>100:
            checkerboard_50 = ((rows // 50) + (cols // 50)) % 2
            flux_patterns["checkerboard_50"] = checkerboard_50.astype(float)

    # ------------------------------------------------------------------ #
    # 3. Compute metrics for each flux pattern
    # ------------------------------------------------------------------ #
    results = {}

    for label, flux_2d in flux_patterns.items():
        flux_2d = np.asarray(flux_2d, dtype=float)
        if flux_2d.shape != (H, W):
            raise ValueError(
                f"Flux pattern '{label}' has shape {flux_2d.shape}, expected ({H}, {W})."
            )
        N = fp_true.shape[0]
        flux_tiled = np.tile(flux_2d, (N, 1, 1))  # (N, H, W)
        mf_true = calculate_mfs(fp_true, flux_tiled, transform_factor)
        mf_pred = calculate_mfs(fp_pred, flux_tiled, transform_factor)

        results[label] = compute_mfs_metrics(mf_true, mf_pred)

    return results


   