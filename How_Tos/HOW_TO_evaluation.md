# How-To: Evaluation

This guide covers evaluation metrics and loss functions for the GATES footprint emulator, using `metrics.py` and `loss_functions.py`.

---

## Footprint Metrics

>Access all the metrics with
`import gates.evaluation.metrics as gates_metrics`

All metric functions compare predicted and true 2D footprint datatasets. They accept numpy arrays, PyTorch tensors, or xarray DataArrays, in spatial `(H, W)`, flat `(HW,)`, batched spatial `(N, H, W)`, or batched flat `(N, HW)` shapes. All return the mean across all footprints by default; pass `reduce=None` to get a per-sample `(N,)` array.

| Metric | Function | Notes |
|--------|----------|-------|
| Intersection-over-Union | `iou` | binary mask overlap; set activation `threshold` |
| Mean Squared Error | `mse` | supports `log_transform=True` |
| Mean Absolute Error | `mae` | |
| Normalised MAE | `nmae` | `sum(abs(pred−true)) / sum(true)` per sample |
| Pearson correlation | `corrcoef` | supports `log_transform=True` |
| Mean bias | `bias` | `mean(pred − true)` |

### Key parameters

**`reduce`** — controls the return shape:
- `reduce="mean"` (default): collapses all samples to a single float
- `reduce=None`: returns `np.ndarray` of shape `(N,)` — useful for time-series diagnostic plots

**`ignore_mask`** — boolean array (`True` = exclude) with the same shape flexibility as the inputs. Pass the `fp_nan_mask` field to exclude out-of-domain NaN padding introduced when cutting footprints to a square, or other pixels to be ignored in the calculation.

**`nonzero`** (`bias`, `mae`, `nmae`, `corrcoef`, `mse`) — if `True`, excludes cells where either array is zero. Useful for focusing metrics on the active footprint region.

**`spatial_shape`** — required as `(H, W)` when inputs are flat (not xarray). Recommended whenever the shape is known, for better error checking.

### Examples

**Single metric:**
```python
import gates.evaluation.metrics as gates_metrics

# scalar score - mean intersection over union across footprints, wherever values are above 1e-4
score = gates_metrics.iou(true_fp, pred_fp, threshold=1e-4, ignore_mask=nan_mask)

# per-footprint array
scores = gates_metrics.iou(true_fp, pred_fp, threshold=1e-4, ignore_mask=nan_mask, reduce=None)
# np.ndarray of shape (N,)
```

**All metrics at once:**
```python
results = gates_metrics.compute_footprint_metrics(
    true_fp, pred_fp,
    spatial_shape=(H, W),
    ignore_mask=nan_mask,
    threshold=1e-4,   # forwarded to iou
    nonzero=False,
)
# {"iou": ..., "mse": ..., "mae": ..., "nmae": ...,
#  "corrcoef": ..., "corrcoef_log": ..., "bias": ...}
```

**Metrics stratified by value range:**

Splits the true footprint into bins and computes all metrics separately within each bin. Useful for diagnosing where the model struggles (e.g., near-zero vs. high-signal pixels).

```python
results = gates_metrics.compute_metrics_by_threshold(
    true_fp, pred_fp,
    thresholds=[0, 1e-2],
    spatial_shape=(H, W),
    ignore_mask=nan_mask,
)
# {"-inf_to_0":    {"iou": ..., "mse": ..., ..., "threshold_range": (-inf, 0)},
#  "0_to_0.01":   {"iou": ..., "mse": ..., ..., "threshold_range": (0, 0.01)},
#  "0.01_to_inf": {"iou": ..., "mse": ..., ..., "threshold_range": (0.01, inf)}}
```

---

## Flux / Mole-Fraction Metrics

Footprints can be evaluated indirectly by computing the implied mole-fraction time series (footprint × gridded fluxes, summed over the domain) and comparing the true vs. predicted series.

Use the `load_flux_data` function to load a monthly dataset of fluxes. The path can be specified, otherwise it will be extracted from `cfg`. You can them crop them to the same shape as the footprints

```python
fluxes = gates.load_flux_data(domain="NORTHAFRICA", year=2016)
# for example, if you have a LoadSquareSatelliteData object loaded as data
cropped_fluxes = gates.cut_flux_data(fluxes, data.fp_data_full, size=data.size)
```

### Step 1 — Compute mole fractions: `calculate_mfs`
`calculate_mfs` takes in `fp` and `fluxes`, which should have the same dimensions and be either xarrays or numpy arrays. The `transform_factor` is a scaler multiplier to the fluxes. If `"default"`, it checks the units in the attributes of the fluxes. The only valid default units are `"mol/m2/s"`, which lead to a transform factor of `1e9`.

If  `fps` is a dataset with multiple variables, `calculate_mfs` will return a dataset. If `fps` is an array (of shape `(B,H,W)` or similar) it will return a single 1D-timeseries as a numpy array.

```python
# xarray path — fp and fluxes share time/lat/lon coordinates
mf_ds = gates_metrics.calculate_mfs(data.fp_xr, cropped_fluxes, transform_factor="default")
# returns xr.Dataset; variable names mirror the footprint variables:
# "fp" → "mf", "fp_pred" → "mf_pred", etc.
# transform_factor="default" reads units from flux_da.attrs["units"]

# numpy path
mf = calculate_mfs(fp_np, flux_np, spatial_shape=(H, W))
# returns np.ndarray of shape (N,)
```

### Step 2 — Evaluate the time series: `compute_mfs_metrics`

Accepts xarray DataArrays (must share the same `time` coordinate) or 1D numpy arrays.

```python
from gates.evaluation.metrics import compute_mfs_metrics

results = compute_mfs_metrics(mf_true, mf_pred)
# {"corrcoef": ..., "mean_absolute_error": ..., "mean_bias": ...,
#  "true_mean": ..., "predicted_mean": ..., "r2_score": ...}
```

### Quick diagnostics without flux data: `compute_static_mf_metrics`

Evaluates mole-fraction metrics using synthetic flux patterns (uniform, checkerboard at 1-, 10-, 25-, and 50-cell scales) — no real flux data required. Useful for fast model comparison. Returns `{pattern_label: metrics_dict}`. Signature: `compute_static_mf_metrics(fp_true, fp_pred, spatial_shape=None, flux_patterns=None, transform_factor=None)`.

---

## Loss Functions

All loss functions are `torch.nn.Module` subclasses. 
> You can import it as `import gates.evaluation.loss_functions as gates_losses`. 

The forward signature is always:

```python
loss_fn = gates_losses.MSELoss()
loss = criterion(pred, target, fp_batch=None, nan_mask=None)
```


The `make_dataloader` [function](../data/datasets.py) outputs the footprints as `fp_batch`, a tensor of shape `(B,H,W,V)` where the variables along `V` are labelled with the `fp_labels`. 
These variables are the target footprint together with auxiliary variables used in the loss functions. An example of a list might look like  like `["fp_transformed", "fp_original", "fp_nan_mask", "flux"]`

You can pass a mask of `nans`, indicating pixels to be ignored by the model when calculating the mtrics. They can be passed when calculating the loss as `nan_mask` (with the same shape as the preds), or you can pass the nan_mask_label when initialising the loss, so that the function automatically selects the index with `fp_batch[...,nan_mask_label]`
```python
loss_fn = gates_losses.MSELoss(fp_labels=fp_labels, nan_mask_label="fp_nan_mask")
loss = criterion(pred, fp_batch[...,0], fp_batch=fp_batch)
```


- `nan_mask_label` — if set, the NaN mask is extracted from `fp_batch` automatically rather than passed in `forward`; `nan_mask` convention is 1 = invalid, 0 = valid
- All losses use `torch.nanmean` / `torch.nansum`, so NaNs already in the data are silently ignored

| Class | Use case |
|-------|----------|
| `MSELoss` | Standard baseline |
| `ThresholdedMSELoss` | Calculates the MSE for different bins of the data separately and adds|
| `PixelWeightedMSELoss` | Penalise errors more in high-value pixels |
| `SumWeightedMSELoss` | Penalise errors more in footprints with large total mass |
| `MSEPlusSumLoss` | MSE + penalty on error in the spatial sum |

**Helper transform/normalisation functions** (passed as `transform_fn` or `normalize_fn`):

`scale(s, w)`, `scale_and_shift(s, w, a)`, `power(s, p)`, `normalize_batch(s)`, `normalize_by_mean(mean)`

### Examples

```python
from gates.evaluation.loss_functions import (
    MSELoss, ThresholdedMSELoss, PixelWeightedMSELoss,
    SumWeightedMSELoss, MSEPlusSumLoss,
    scale, scale_and_shift, normalize_by_mean,
)

fp_labels = ["fp_transformed", "fp_original", "fp_nan_mask", "flux"]

# Standard MSE — nan_mask extracted from fp_batch automatically
criterion = MSELoss(fp_labels, nan_mask_label="fp_nan_mask")
loss = criterion(pred, target, fp_batch)

# ThresholdedMSE — computes MSE separately for pixels above/below threshold,
# then sums with alpha weighting. Avoids the MSE being dominated by near-zero pixels.
criterion = ThresholdedMSELoss(
    fp_labels, weight_label="fp_original",
    threshold=0, alpha=2.0,          # above-threshold pixels weighted 2×
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)

# Multiple thresholds and per-bin alphas:
criterion = ThresholdedMSELoss(
    fp_labels, weight_label="fp_original",
    threshold=[0, 1e-3], alpha=[1.0, 2.0, 5.0],
    nan_mask_label="fp_nan_mask",
)

# PixelWeightedMSE — multiplies squared error by a per-pixel weight derived from
# a chosen fp_batch variable. transform_fn shapes the weight field.
criterion = PixelWeightedMSELoss(
    fp_labels, weight_label="fp_original",
    transform_fn=scale_and_shift, w=1000, a=1.0,
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)

# SumWeightedMSE — weights each sample's MSE by the spatial sum of a chosen variable,
# making the loss sensitive to footprint mass. normalize_fn makes weights scale-independent.
criterion = SumWeightedMSELoss(
    fp_labels, weight_label="fp_original",
    normalize_fn="batch",            # "batch" or normalize_by_mean(dataset_mean)
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)

# MSEPlusSumLoss — pixel-level MSE plus a penalty on the error in the spatial integral
# (footprint × weight field, summed over domain). With weight_label="flux" this penalises
# errors in emulated mole fractions; with weight_label="ones" it penalises errors in
# total footprint mass. alpha controls the relative contribution of the two terms.
criterion = MSEPlusSumLoss(
    fp_labels, weight_label="flux",
    alpha=1.0, normalize_fn="batch",
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)
```
