# Brief A — New loss functions (plan + build)

## Model context

**GATES** is a graph-neural-network emulator of a Lagrangian Particle Dispersion
Model. It predicts atmospheric transport **footprints** (2D lat×lon sensitivity
fields) from meteorological inputs, replacing an expensive physics simulation.
Architecture is **encode–process–decode** (`gates/model/forecast.py`): a
grid→hexagonal-mesh encoder (h3, res 4), `num_blocks` rounds of mesh
message-passing (processor), and a mesh→grid decoder. Inputs are
`(time, lat, lon, variable)`; output is one footprint value per grid node.
Training is driven by a JSON parameter file via `train_GATES_model.py`. Known
problem being worked: predictions are **blurry**, and a batch of
capacity/temporal experiments (`EXPERIMENTS.md`) has plateaued — so we're now
attacking the decoder, the loss, and the training recipe.

## Task

Add two new loss modules to `gates/evaluation/loss_functions.py`. Existing losses
(`MSELoss`, `PixelWeightedMSELoss`, `SumWeightedMSELoss`, `MSEPlusSumLoss`,
`ThresholdedMSELoss`) are all essentially per-pixel MSE variants — and plain MSE
rewards the conditional mean, which **is** the blur.

Follow the existing conventions exactly:
- subclass `nn.Module`, signature `forward(self, pred, target, fp_batch=None, nan_mask=None)`
- use the module's `_resolve_nan_mask` / `_mask` / `_spatial_dims` helpers
- construct with `fp_labels` / `nan_mask_label`
- losses are selected in config via `loss_functions.criterion` (eval'd by name in
  `gates/training/training.py:749`) with `criterion_params` for kwargs.

### 1. `GradientMSELoss` (priority)

`L = MSE + β·MSE(∇pred, ∇target)`, where ∇ is a Sobel / finite-difference spatial
gradient over the (H,W) dims. Penalising gradient error directly punishes
over-smoothing (a blurred prediction has correct values but collapsed gradients).
Handle both `(B,H,W)` and flattened `(B,H*W)` pred shapes (the training loop
flattens — see `flatten=True` in the dataloader). Expose `beta` as a
`criterion_params` kwarg.

### 2. `StructuralLoss`

Either `1 − MS-SSIM` or a per-sample `1 − spatial Pearson correlation` term,
combinable with MSE (`L = MSE + γ·structural`). Rewards local structure/shape
independent of scale.

### Notes

- Both are **additive** to MSE, not replacements; tune the weights.
- **Bonus (one-liner):** offer a Charbonnier core `√(e²+ε²)` as a robust drop-in
  for the squared-error term.
- Deliver: the modules + a short docstring each (matching house style) + example
  `criterion_params` config snippets. No training run needed — just plan and
  build, then hand back for A/B.

---

## Literature: why these two losses

### Why plain MSE blurs

Pixelwise MSE (and its weighted variants) is minimised by the **conditional mean**
of the target given the inputs. When the input→output map is even slightly
uncertain — as it is for turbulent atmospheric transport — the value that
minimises expected squared error at each pixel is the *average* over all plausible
footprints, which is smoother than any individual one. The network is thus rewarded
for hedging, and its optimum is a blurred field. In forecast verification this is
the **"double penalty" problem**: a sharp prediction that is slightly displaced is
penalised twice (a miss *and* a false alarm), so MSE-type scores actively prefer
smeared, low-amplitude fields (Gilleland et al., 2009, *Wea. Forecasting*). The
same effect is documented across ML weather/climate work: MSE-trained precipitation
nowcasters and emulators produce over-smoothed, blurry outputs that lose
small-scale structure (Ravuri et al., 2021, *Nature*, DGMR; Rasp & Thuerey, 2021,
*JAMES*). Both losses below score **spatial structure** instead of per-pixel
amplitude.

### 1. Gradient / edge losses (→ `GradientMSELoss`)

**Idea.** A blurred field has roughly the right values but *collapsed slopes*:
edges and ridges are flattened. Penalising the error in the **spatial gradient** ∇
directly punishes that flattening. Because
`MSE(∇pred, ∇target) == MSE(∇(pred−target))`, the gradient term is just the
smoothness of the *error* field — only over-smoothing reduces it.

**Literature.** This is the **Gradient Difference Loss (GDL)** of Mathieu, Couprie
& LeCun (2016, *ICLR*, "Deep multi-scale video prediction beyond mean square
error"), introduced for exactly this symptom: L2-trained video prediction was
blurry, and a gradient-difference term restored sharp edges. The spatial gradient
is typically computed with **Sobel** finite differences (Sobel & Feldman, 1968).
Gradient/high-frequency penalties recur throughout climate ML — super-resolution /
downscaling of wind, solar and precipitation add gradient or spectral terms to
recover texture MSE destroys (Stengel et al., 2020, *PNAS*; Vandal et al., 2017,
DeepSD). Cheap, differentiable, and — unlike GANs — no training instability, which
is why it's the priority.

### 2. Structural-similarity losses (→ `StructuralLoss`)

**Idea.** Compare *local patterns* rather than pixels one-by-one — does a patch of
the prediction have the same shape/contrast/texture as the same patch of truth,
regardless of absolute level? Two candidates:

- **(MS-)SSIM** compares local luminance (mean), contrast (variance) and structure
  (covariance) over a sliding window; multi-scale does this across resolutions.
  As a loss: `1 − MS-SSIM`.
- **Spatial Pearson correlation** — per-sample correlation between predicted and
  true fields over space; pure pattern agreement, invariant to per-footprint offset
  or rescaling. As a loss: `1 − corr`.

**Literature.** SSIM is Wang et al. (2004, *IEEE TIP*), multi-scale Wang et al.
(2003, *Asilomar*) — designed because MSE correlates poorly with perceived
structural fidelity. Using **MS-SSIM (+ L1) as a training loss** was established by
Zhao et al. (2017, *IEEE TCI*, "Loss Functions for Image Restoration with Neural
Networks"), the same `MSE + γ·structural` recipe used here. In geoscience SSIM is a
common evaluation metric for precipitation downscaling/nowcasting generative models
(Harris et al., 2022, *JAMES*; Ayzel et al., 2020, RainNet). The
**spatial-correlation** variant is essentially the **Anomaly / Pattern Correlation
Coefficient (ACC)**, a decades-old spatial-forecast skill score and a headline
metric in ML weather benchmarks (Rasp et al., 2020/2023, WeatherBench). It is the
cheapest structural term (no windowing, trivially differentiable) — the sensible
first thing to A/B before MS-SSIM.

**Caveat for our data.** Footprints are sparse and heavy-tailed (mostly ≈0,
log-shift transformed). SSIM's default constants assume roughly `[0,1]`-scaled
images, and correlation over mostly-zero fields is dominated by the few active
pixels. Both terms are best applied in the model's **transformed (log) space** and
kept **additive** to MSE, never standalone.

---

## What was implemented (`gates/evaluation/loss_functions.py`)

Follows house conventions exactly: `nn.Module`,
`forward(pred, target, fp_batch=None, nan_mask=None)`, the shared
`_resolve_nan_mask` / `_mask` / `_spatial_dims` / `_label_index` helpers, and the
`(fp_labels=None, nan_mask_label=None, **params)` constructor contract that
`training.py:757` depends on.

**Shared internal helpers**

- `_to_2d(t, spatial_shape=None)` — un-flattens `(B, N)` / `(B, N, 1)` → `(B, H, W)`
  (the training loop delivers preds as `(B, N, 1)` with `N = size²`). Square by
  default; pass `spatial_shape=(H, W)` for non-square domains.
- `_apply_core(e, core, eps)` — `"squared"` (`e²`) or the robust Charbonnier
  `√(e²+ε²)` (the bonus one-liner). Reused by both losses, so squared↔charbonnier
  is one kwarg.

**`GradientMSELoss`** — `L = core(pred−target) + β·core(∇(pred−target))`. Forward
finite-difference gradient over both spatial dims; gradient entries adjacent to a
masked/NaN pixel are dropped. Kwargs: `beta`, `core`, `eps`, `spatial_shape`.

**`StructuralLoss`** — `L = core(pred−target) + γ·structural`.
`mode="correlation"` (default) = per-sample spatial Pearson correlation over valid
pixels, `structural = mean(1 − corr)`. `mode="ms_ssim"` is stubbed and raises
`NotImplementedError` (deferred). Kwargs: `gamma`, `mode`, `core`, `eps`.

**Verified** on synthetic data: sharp vs. blurred preds (blur penalised more by all
variants), 2D and flattened `(B, N, 1)` paths agree, NaN masking works both
explicitly and via `nan_mask_label`, gradients flow, `ms_ssim` raises. No training
run (per brief).

### Config snippets

```json
"loss_functions": {
  "criterion": "gates_losses.GradientMSELoss",
  "criterion_params": {"beta": 0.5, "core": "charbonnier"},
  "criterion_test": "gates_losses.MSELoss",
  "criterion_test_params": {}
}
```
```json
"loss_functions": {
  "criterion": "gates_losses.StructuralLoss",
  "criterion_params": {"mode": "correlation", "gamma": 0.3},
  "criterion_test": "gates_losses.MSELoss",
  "criterion_test_params": {}
}
```

`beta`/`gamma` are the knobs to sweep — start so the added term is ~10–50 % of the
base MSE at convergence. Ready-to-run sweeps live in
`parameter_files/experiments_loss/` (see `EXPERIMENTS_LOSS.md`).
