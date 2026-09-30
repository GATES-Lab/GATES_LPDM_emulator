# How to Choose a Loss Function

This guide covers the loss functions in `gates/evaluation/loss_functions.py`: how the training script builds them from the parameter file, how to call them directly, and what each one does.

For the metrics used to score predictions after training, see the [evaluation guide](HOW_TO_evaluation.md). The notebook `notebooks/loss_function_examples.ipynb` plots what each loss looks at, on real test predictions. For the rest of the parameter file, see the [parameter file guide](HOW_TO_PARAMETER_FILE.md#loss_functions).

> Import the module with `import gates.evaluation.loss_functions as gates_losses`.

---

## Setting the loss in the parameter file

The `loss_functions` block names two losses:

```json
"loss_functions": {
    "criterion": "gates_losses.PixelWeightedMSELoss",
    "criterion_params": {
        "weight_label": "fp_original",
        "transform_fn": "scale_and_shift",
        "w": 1000,
        "a": 1.0
    },
    "criterion_test": "gates_losses.MSELoss",
    "criterion_test_params": {}
}
```

- **`criterion`** is the loss that training minimises.
- **`criterion_test`** is the loss reported as the test loss, and as the "display" loss printed for the training batches. It doesn't affect training, so keep it simple (usually `MSELoss`) so that runs with different `criterion`s can be compared.

`setup_GATES_model` (`gates/training/training.py`) builds each one like this:

1. The string is run through `eval`, so it must be written as `gates_losses.<ClassName>`.
2. The class is called with `fp_labels` and `nan_mask_label` filled in by the training script, plus everything in `criterion_params` (or `criterion_test_params`) as keyword arguments.
   - `fp_labels` is the list of variables in the footprint batch, as returned by the dataloader.
   - `nan_mask_label` is `"fp_nan_mask"` when `dataloader.nans_to_zeros` is `true` (the default), and `None` otherwise. With `nans_to_zeros`, the NaNs outside the domain are set to zero in the footprints, and `fp_nan_mask` marks where they were, so the loss can ignore those pixels.  
   > Note: add link here to the sections in the data how to

Because the script sets `fp_labels` and `nan_mask_label` itself, **don't put them in `criterion_params`**: passing them twice raises a `TypeError`.

Functions such as `transform_fn` can be given as strings (e.g. `"scale_and_shift"`). They are run through `eval` inside `loss_functions.py`, so use the bare function name, without the `gates_losses.` prefix. The exceptions are listed under [Known issues](#known-issues).

---

## Calling a loss directly

All losses are `torch.nn.Module` subclasses with the same forward signature:

```python
loss = criterion(pred, target, fp_batch=None, nan_mask=None)
```

- `pred` and `target` are the model output and the true (transformed) footprint. In training they are flat, with shape `(B, H*W, 1)`; `(B, H, W)` and `(B, H*W)` also work.
- `fp_batch` is the footprint batch from the dataloader, with shape `(B, H*W, V)` (or `(B, H, W, V)`). The `V` variables along the last axis are named by `fp_labels`, for example `["fp_transformed", "fp_original", "fp_nan_mask", "flux"]`. The first one is the target; the others are used by the losses for weighting and masking. See the [data guide](HOW_TO_DATA.md) for how the dataloader builds it.
- Every loss accepts `fp_batch`, even the ones that don't use it, so the training loop can call them all the same way.

### Ignoring pixels

There are two ways to tell a loss which pixels to ignore. The mask convention is `1` (or `True`) = ignore, `0` = keep.

- Pass the mask to `forward` as `nan_mask`, with the same shape as `pred`.
- Or pass `nan_mask_label` when you create the loss, and the loss takes the mask from `fp_batch[..., fp_labels.index(nan_mask_label)]`. This is what training does.

An explicit `nan_mask` takes priority over `nan_mask_label`. All losses also use `torch.nanmean` / `torch.nansum`, so any NaNs already in the data are ignored.

```python
import gates.evaluation.loss_functions as gates_losses

fp_labels = ["fp_transformed", "fp_original", "fp_nan_mask", "flux"]
target = fp_batch[..., 0:1]          # fp_transformed, shape (B, H*W, 1)

criterion = gates_losses.MSELoss(fp_labels=fp_labels, nan_mask_label="fp_nan_mask")
loss = criterion(pred, target, fp_batch)
```

---

## Loss functions

| Class | What it does | Uses `fp_batch` for |
|-------|--------------|---------------------|
| `MSELoss` | Plain MSE. The standard baseline, and the usual `criterion_test`. | the mask only |
| `ThresholdedMSELoss` | Splits the pixels into bins by the value of a chosen variable, computes the MSE in each bin separately, and adds them with a weight per bin. Stops the loss being dominated by the many near-zero pixels. | the binning variable (`weight_label`) |
| `PixelWeightedMSELoss` | Multiplies each pixel's squared error by a weight taken from a chosen variable. Penalises errors more in high-value pixels. | the weight field (`weight_label`) |
| `SumWeightedMSELoss` | Weights each footprint's MSE by the spatial sum of a chosen variable. Penalises errors more in footprints with a large total. | the field that is summed (`weight_label`) |
| `MSEPlusSumLoss` | MSE plus a penalty on the error in the spatial sum of the footprint (optionally multiplied by a field such as the flux). | the weight field, unless `weight_label="ones"` |
| `GradientMSELoss` | MSE plus a penalty on the error in the spatial gradient. Discourages blurred predictions. | the mask, and optionally a pixel weight |
| `StructuralLoss` | MSE plus a penalty for low spatial correlation between prediction and truth. Rewards getting the pattern right, whatever the scale. | the mask, and optionally a pixel weight |

### Helper functions

These shape a weight field or a per-footprint sum. Pass them as `transform_fn`, `weight_transform_fn` or `normalize_fn`, with their parameters as extra keyword arguments.

| Function | Returns | Typical use |
|----------|---------|-------------|
| `scale(s, w=1.0)` | `w * s` | `transform_fn` |
| `scale_and_shift(s, w=1.0, a=0.0)` | `w * s + a` | `transform_fn`; `a` keeps a minimum weight for zero pixels |
| `power(s, p=2.0)` | `s ** p` | `transform_fn` |
| `normalize_batch(s)` | `s / mean(s)` over the batch | `normalize_fn="batch"` |
| `normalize_by_mean(mean)` | a function `s -> s / mean` | `normalize_fn`, with a mean computed once over the training set |

`normalize_batch` makes the weights independent of the footprint scale, but the weight given to a sample then depends on the rest of its batch. `normalize_by_mean` gives the same weight to a sample whatever batch it is in.

---

## Examples

The examples below create each loss directly. To use one in training, put the class in `criterion` and the other arguments (everything except `fp_labels` and `nan_mask_label`) in `criterion_params`.

```python
import gates.evaluation.loss_functions as gates_losses

fp_labels = ["fp_transformed", "fp_original", "fp_nan_mask", "flux"]
```

### `ThresholdedMSELoss`

Pixels are binned by the value of `weight_label`. With `threshold=0` there are two bins, `<= 0` and `> 0`. A single `alpha` weights every bin above the first threshold; the first bin always gets 1.

```python
# the MSE over pixels with fp_original > 0 counts twice as much as over pixels with fp_original <= 0
criterion = gates_losses.ThresholdedMSELoss(
    fp_labels, weight_label="fp_original",
    threshold=0, alpha=2.0,
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)

# several thresholds: alpha needs one entry per bin (len(threshold) + 1)
criterion = gates_losses.ThresholdedMSELoss(
    fp_labels, weight_label="fp_original",
    threshold=[0, 1e-3], alpha=[1.0, 2.0, 5.0],
    nan_mask_label="fp_nan_mask",
)
```

### `PixelWeightedMSELoss`

The squared error in each pixel is multiplied by `transform_fn(fp_batch[..., weight_label], **kwargs)`. With the settings below, a pixel's weight is `1000 * fp_original + 1`. This is the loss in the parameter file templates.

```python
criterion = gates_losses.PixelWeightedMSELoss(
    fp_labels, weight_label="fp_original",
    transform_fn=gates_losses.scale_and_shift, w=1000, a=1.0,
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)
```

### `SumWeightedMSELoss`

Each footprint's MSE is multiplied by the spatial sum of `weight_label`, after `normalize_fn` and then `transform_fn`. With `weight_label="fp_original"` this weights footprints by their total; with `weight_label="flux"` it weights them by their flux-weighted total, so footprints that matter more for the mole fraction count more.

```python
criterion = gates_losses.SumWeightedMSELoss(
    fp_labels, weight_label="fp_original",
    normalize_fn="batch",            # or gates_losses.normalize_by_mean(dataset_mean)
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)
```

### `MSEPlusSumLoss`

The loss is `MSE + alpha * MSE(sum_pred, sum_true)`, where each sum is the footprint multiplied by the weight field and summed over the domain.

- `weight_label="flux"` penalises errors in the emulated mole fraction.
- `weight_label="ones"` (the default) penalises errors in the total footprint; `fp_batch` isn't needed.
- `normalize_fn` is applied to the true and predicted sums together, so their ratio is kept.
- `alpha=0` gives plain MSE.
- In training, `pred` and `target` are the *transformed* footprints (after the `fp_scaler`), so the sum is of the transformed footprint times the flux, not the real mole fraction. The transformed prediction is also above zero almost everywhere, so its sum is larger than it would be for the original footprint.

```python
criterion = gates_losses.MSEPlusSumLoss(
    fp_labels, weight_label="flux",
    alpha=1.0, normalize_fn="batch",
    nan_mask_label="fp_nan_mask",
)
loss = criterion(pred, target, fp_batch)
```

### `GradientMSELoss`

MSE rewards predicting the average footprint, which is smoother than any real one. This loss adds `beta` times the error in the spatial gradient (the differences between neighbouring pixels), which a blurred prediction can't match (the gradient difference loss of Mathieu et al., 2016).

- `beta=0` gives plain MSE.
- `core="charbonnier"` replaces the squared error with `sqrt(e**2 + eps**2)`, which is less dominated by large errors.
- Flat inputs are reshaped to a square `(H, W)`. For a non-square domain, pass `spatial_shape=(H, W)`.
- `weight_label` / `weight_transform_fn` / `weight_transform_kwargs` weight the MSE term by pixel, as in `PixelWeightedMSELoss`. The gradient term is not weighted. Unlike `PixelWeightedMSELoss`, the transform's parameters go in the `weight_transform_kwargs` dict, not as separate keyword arguments.

```python
criterion = gates_losses.GradientMSELoss(
    fp_labels, nan_mask_label="fp_nan_mask",
    beta=0.5, core="charbonnier",
    weight_label="fp_original",
    weight_transform_fn="scale_and_shift",
    weight_transform_kwargs={"w": 1000, "a": 1.0},
)
loss = criterion(pred, target, fp_batch)
```

### `StructuralLoss`

Adds `gamma * mean(1 - r)`, where `r` is the spatial Pearson correlation between each predicted and true footprint. This scores whether the pattern is right, independent of an overall offset or scale.

- `gamma=0` gives plain MSE.
- Footprints whose target is constant (e.g. all zero) are left out of the correlation term.
- `mode="ms_ssim"` is accepted but not implemented; it raises `NotImplementedError` on the first call.
- `core`, `eps` and the `weight_*` arguments work as in `GradientMSELoss`.

```python
criterion = gates_losses.StructuralLoss(
    fp_labels, nan_mask_label="fp_nan_mask",
    gamma=0.3,
)
loss = criterion(pred, target, fp_batch)
```

---

## Known issues

- **`normalize_fn` strings:** only `"batch"` works as a string. Any other string, e.g. `"normalize_by_mean(0.5)"`, raises a `ValueError`, even though the docstring says it is evaluated. From a parameter file, use `"batch"` or leave `normalize_fn` out.
- **`MSEPlusSumLoss.transform_fn`** is not run through `eval`, so it can't be set from a parameter file (a string raises `TypeError: 'str' object is not callable`). It only works when you pass a function in Python.
