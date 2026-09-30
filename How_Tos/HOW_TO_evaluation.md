# How to Evaluate Predictions

This guide covers the metrics used to evaluate GATES footprint predictions, from `gates/evaluation/metrics.py`. For the loss functions used in training, see the [loss functions guide](HOW_TO_LOSSES.md).

> Import the module with `import gates.evaluation.metrics as gates_metrics`.

There are two kinds of metric:

- **Footprint metrics** compare the predicted and true footprints pixel by pixel.
- **Flux / mole-fraction metrics** multiply each footprint by a flux map and sum over the domain, then compare the resulting mole-fraction time series. This measures what matters for the inversion.

The inputs are usually the variables of a predictions file (`sample_predictions_test.nc` from training, or `predictions_*.nc` from `scripts/predict_GATES_model.py`): `fp_original` is the true footprint, `fp_pred` the prediction, and `fp_nan_mask` marks the padding outside the original domain. See the [prediction guide](HOW_TO_PREDICT.md) for the file contents.

---

## Footprint metrics

All footprint metric functions compare predicted and true 2D footprints. They accept numpy arrays, PyTorch tensors or xarray DataArrays, with shape `(H, W)`, `(HW,)`, `(N, H, W)` or `(N, HW)`. Each metric is computed per footprint; by default the function returns the mean over footprints, and `reduce=None` returns the per-footprint `(N,)` array instead.

| Metric | Function | Notes |
|--------|----------|-------|
| Intersection-over-Union | `iou` | Overlap of the areas above `threshold` (default 0). |
| IoU at quantiles | `iou_at_quantiles` | IoU with the threshold set at quantiles of the true footprint. Returns a dict (see below). |
| Mean squared error | `mse` | `log_transform=True` computes it on `log10` values (zeros are excluded). |
| Mean absolute error | `mae` | |
| Normalised MAE | `nmae` | `sum(abs(pred − true)) / sum(true)` per footprint. |
| Pearson correlation | `corrcoef` | `log_transform=True` computes it on `log10` values (zeros are excluded). |
| Mean bias | `bias` | `mean(pred − true)`. |

### Common arguments

- **`reduce`**: `"mean"` (default) returns a single float, the mean over footprints. `None` returns an `(N,)` array, e.g. for a time-series plot.
- **`ignore_mask`**: pixels to exclude (`True` = exclude), in any of the input shapes. Pass `fp_nan_mask` to exclude the padding outside the original domain.
- **`nonzero`** (`bias`, `mae`, `nmae`, `corrcoef`, `mse`): if `True`, excludes pixels where either footprint is zero, so the metric only covers the footprint itself.
- **`spatial_shape`**: `(H, W)`. Required for flat inputs that aren't xarray; recommended whenever you know it, for better error checking.

### Examples

**A single metric:**

```python
import gates.evaluation.metrics as gates_metrics

# mean IoU over all footprints, counting pixels above 1e-4 as part of the footprint
score = gates_metrics.iou(true_fp, pred_fp, threshold=1e-4, ignore_mask=nan_mask)

# one IoU per footprint, as an np.ndarray of shape (N,)
scores = gates_metrics.iou(true_fp, pred_fp, threshold=1e-4, ignore_mask=nan_mask, reduce=None)
```

**All metrics at once:**

```python
results = gates_metrics.compute_footprint_metrics(
    true_fp, pred_fp,
    spatial_shape=(H, W),
    ignore_mask=nan_mask,
    threshold=1e-4,   # used by iou only
    nonzero=False,
)
# {"iou": ..., "mse": ..., "mae": ..., "nmae": ...,
#  "corrcoef": ..., "corrcoef_log": ..., "bias": ...}
```

Pass `metrics=[...]` to compute only some of them. Other keyword arguments (`reduce`, `nonzero`) are passed to every metric.

**IoU at quantiles:**

A single low threshold gives an IoU that is similar for every model, because it is dominated by the broad extent of the footprint. `iou_at_quantiles` sets the threshold at quantiles of the *true* footprint instead, and applies it to both footprints, so the high quantiles score the core of the footprint, where models differ.

```python
results = gates_metrics.iou_at_quantiles(true_fp, pred_fp, ignore_mask=nan_mask)
# {"iou_q25": ..., "iou_q50": ..., "iou_q90": ..., "iou_q99": ...}
```

- `quantiles` sets the quantiles (default `(0.25, 0.5, 0.9, 0.99)`).
- `per_sample=True` (default) takes the quantile of each footprint separately. `False` uses one quantile over the whole dataset.
- A footprint whose quantile is exactly zero (common for the low quantiles of sparse footprints) gets NaN and is left out of the mean.

**Metrics by value range:**

`compute_metrics_by_threshold` splits the pixels into bins by the value of the true footprint, and computes the metrics separately for the pixels in each bin. This shows where the model struggles, e.g. near-zero vs. high-value pixels.

```python
results = gates_metrics.compute_metrics_by_threshold(
    true_fp, pred_fp,
    thresholds=[0, 1e-2],
    spatial_shape=(H, W),
    ignore_mask=nan_mask,
)
# {"-inf_to_0":   {"mse": ..., "mae": ..., ..., "iou": ..., "threshold_range": (-inf, 0)},
#  "0_to_0.01":   {"mse": ..., "mae": ..., ..., "iou": ..., "threshold_range": (0, 0.01)},
#  "0.01_to_inf": {"mse": ..., "mae": ..., ..., "iou": ..., "threshold_range": (0.01, inf)}}
```

- `thresholds` must be a number or a list. A numpy array gives a single `nan_to_nan` bin with every metric NaN (a known bug), so convert it with `list(...)` first.
- The IoU is the exception: it is not computed within the bin, but over the whole domain with the bin's lower edge as the threshold. The IoU of the lowest bin (`-inf_to_...`) is therefore always 1.
- A bin with no pixels prints a warning and gets NaN metrics.

---

## Flux / mole-fraction metrics

The footprint tells you how sensitive a measurement is to emissions in each grid cell. Multiplying it by a flux map and summing over the domain gives the mole fraction that the measurement would see. Comparing the true and predicted mole-fraction time series measures the error in the quantity the inversion uses.

### Loading and cropping the flux

`load_flux_data` loads a year of monthly flux maps for a domain, as a DataArray `(time, lat, lon)`. The file is found from the `flux_datadir` in `config.yml`, or you can pass `flux_path`. See the [data guide](HOW_TO_DATA.md) for the file pattern.

`cut_flux_data` crops the flux to the same `size × size` square as each footprint, using the flux map of the matching month. It needs the full-domain footprints, `data.fp_data_full`, which have the release coordinates. It returns two things: a Dataset with the cropped `flux` (and `lat_coords` / `lon_coords`), and the footprint times that had no flux within 32 days (their flux is NaN).

```python
import gates
from gates import LoadSquareSatelliteData

data = LoadSquareSatelliteData(
    year=2018, region="BRAZIL", month="01", size=10, crop_met=False,
    met_args={"met_variables": []},   # no met needed for evaluation
)

fluxes = gates.load_flux_data(domain="BRAZIL", year=2018, search_others=True)
cropped, missing_times = gates.cut_flux_data(fluxes, data.fp_data_full, size=data.size)
```

- `domain` can be the region name from `config.yml` (`"BRAZIL"`) or its domain name (`"SOUTHAMERICA"`).
- `search_others=True` accepts a file with a different suffix if the expected one doesn't exist. For example, the 2018 South America flux is `..._copy-from-2016.nc`.
- The cropped `lat` / `lon` are `0 … size-1`, like the cut footprints, so the two line up.

### Step 1: compute the mole fractions with `calculate_mfs`

`calculate_mfs(fp, fluxes, transform_factor=None, spatial_shape=None)` multiplies the footprints by the fluxes and sums over the domain. `fp` and `fluxes` must both be xarray or both numpy, with the same shape.

- **xarray:** `fp` can be a DataArray or a Dataset, and `fluxes` a DataArray or a Dataset with a `flux` variable. The times and the `lat` / `lon` coordinates must match, including any extra coordinates on `time`. Predictions files have an `idx` coordinate on `time`, so the check fails even when the times are the same; drop it with `preds.drop_vars("idx")` first. It returns a Dataset with one mole fraction per footprint variable: `fp` → `mf`, `fp_pred` → `mf_pred`. Variables without `fp` in their name are skipped.
- **numpy:** returns a `(N,)` array. Flat inputs need `spatial_shape`.
- **`transform_factor`:** a number multiplies the result. `"default"` (xarray only) reads the flux units: `"mol/m2/s"` gives a factor of `1e9`, i.e. the mole fraction in ppb. Other units print a warning and use a factor of 1. The units attribute of the result is set to e.g. `"1e+09 * mol/m2/s"`.

```python
# xarray: data.fp_xr has the true footprints in the variable "fp"
mf_ds = gates_metrics.calculate_mfs(data.fp_xr, cropped, transform_factor="default")
# xr.Dataset with the variable "mf", one value per footprint

# a predictions file: drop the idx coordinate first (see below)
preds = preds.drop_vars("idx")
mf_ds = gates_metrics.calculate_mfs(
    preds[["fp_original", "fp_pred"]], cropped.flux, transform_factor="default")
# "mf_original" and "mf_pred"

# numpy, flat
mf = gates_metrics.calculate_mfs(fp_np, flux_np, spatial_shape=(H, W))
# np.ndarray of shape (N,)
```

### Step 2: compare the time series with `compute_mfs_metrics`

`compute_mfs_metrics(mf_true, mf_pred)` takes two DataArrays with the same `time` coordinate, or two 1D numpy arrays. Non-finite values are skipped; at least two valid points are needed.

```python
results = gates_metrics.compute_mfs_metrics(mf_ds["mf_original"], mf_ds["mf_pred"])
# {"corrcoef": ..., "mae": ..., "mean_bias": ...,
#  "true_mean": ..., "predicted_mean": ..., "r2_score": ...}
```

### Both steps against a real flux: `compute_flux_metrics`

`compute_flux_metrics(fp_true, fp_pred, flux, spatial_shape=None, transform_factor=None, ignore_mask=None)` runs `calculate_mfs` on the true and predicted footprints and compares them with `compute_mfs_metrics`. It returns the same keys.

```python
results = gates_metrics.compute_flux_metrics(
    preds.fp_original, preds.fp_pred, cropped.flux,
    ignore_mask=preds.fp_nan_mask,
)
```

- `fp_true`, `fp_pred` and `flux` must **all** be xarray DataArrays or **all** numpy arrays. An `xr.Dataset` is rejected. Numpy inputs need `spatial_shape=(H, W)`.
- `ignore_mask` (`True` = exclude) sets the flux to zero in the masked pixels.
- `transform_factor` is passed to `calculate_mfs`. Here it must be a number: `"default"` needs xarray, and the arrays are converted to numpy first.

### Without flux data: `compute_static_mf_metrics`

`compute_static_mf_metrics(fp_true, fp_pred, spatial_shape=None, flux_patterns=None, transform_factor=None, ignore_mask=None)` does the same with synthetic flux maps, so it needs no flux file. It is a quick way to compare models. It returns `{pattern: metrics_dict}`.

The default patterns are:

| Pattern | Flux |
|---------|------|
| `uniform` | 1 everywhere, so the mole fraction is the footprint total |
| `checkerboard` | alternating 0 / 1 cells |
| `checkerboard_10`, `checkerboard_25` | alternating 0 / 1 blocks of 10 × 10 and 25 × 25 cells |
| `checkerboard_50` | 50 × 50 blocks; only added when H and W are both over 100 |

A checkerboard is only meaningful when the domain is larger than its block size. When H and W are both 10 or less, `checkerboard_10` is all zeros, and the same for `checkerboard_25` up to 25: the mole fractions are all zero, `corrcoef` is NaN and `r2_score` is 1.

Pass your own patterns as `flux_patterns={"name": array_of_shape_(H, W), ...}`.

### During training

Each evaluation epoch, `calculate_losses` (`gates/training/training.py`) runs the footprint metrics, `iou_at_quantiles`, `compute_static_mf_metrics`, and `compute_flux_metrics` when the test data has a `flux` variable. The flux is added by the [`flux` block of the parameter file](HOW_TO_PARAMETER_FILE.md#flux). No `transform_factor` is passed, so the flux metrics are in the units of the loaded flux (multiplied by `convert_units_args.unit_multiplier` if set). The [W&B guide](HOW_TO_WandB.md) lists every metric that is logged, and its name.

---

## See also

- [Loss functions guide](HOW_TO_LOSSES.md): the losses used in training, and how to set them in the parameter file.
- [W&B guide](HOW_TO_WandB.md): the metrics logged during training.
- [Data guide](HOW_TO_DATA.md): loading footprints, and the flux file pattern.
