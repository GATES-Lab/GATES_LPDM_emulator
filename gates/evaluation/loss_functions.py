import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


# ---------------------------------------------------------------------------
# Helper functions
#
# All helpers share the same calling convention: fn(s, **kwargs) -> Tensor,
# where s is the tensor to transform.  They can be used interchangeably as
# the `transform_fn` argument in PixelWeightedMSE / SumWeightedMSE, or the
# `normalize_fn` argument.
# ---------------------------------------------------------------------------

def scale(s, w=1.0):
    """Multiply by a scalar weight w."""
    return w * s


def scale_and_shift(s, w=1.0, a=0.0):
    """Multiply by w and add constant a."""
    return w * s + a


def power(s, p=2.0):
    """Raise to the power p.  Amplifies differences between large and small values."""
    return s ** p


def normalize_batch(s):
    """Divide by the batch mean (per-batch normalisation).

    Weights become scale-independent but vary with batch composition: the same
    sample may receive a different weight depending on what else is in the batch.
    """
    return s / (s.mean() + 1e-8)


def normalize_by_mean(mean):
    """Return a normalisation function that divides by a fixed dataset-level mean.

    Call once at setup time with the mean computed over the training set,
    then pass the returned function as normalize_fn::

        fp_sum_mean = ...  # computed once over training set
        criterion = SumWeightedMSE(fp_labels,
                                    normalize_fn=normalize_by_mean(fp_sum_mean),
                                    transform_fn=scale, w=1.0)

    Weights are scale-independent and batch-consistent.
    """
    def _normalize(s):
        return s / (mean + 1e-8)
    return _normalize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spatial_dims(t):
    """All dimension indices except the batch dimension (0)."""
    return tuple(range(1, t.dim()))


def _mask(t, nan_mask):
    """Return t with nan_mask positions set to NaN (no-op if nan_mask is None).

    nan_mask convention: 1 = invalid/NaN, 0 = valid.
    """
    if nan_mask is None:
        return t
    return t.masked_fill(nan_mask.bool(), float('nan'))


def _get_normalize_fn(normalize_fn):
    """Resolve normalize_fn to a callable (or None).

    Pass:
    - None           -> no normalisation
    - "batch"        -> normalize_batch
    - callable       -> used as-is (e.g. normalize_by_mean(mean))
    """
    if normalize_fn is None or callable(normalize_fn):
        return normalize_fn
    if normalize_fn == "batch":
        return normalize_batch
    elif isinstance(normalize_fn, str):
        try:
            normalize_fn = eval(normalize_fn)
        except Exception as e:
            print(f"Error occurred while evaluating normalize_fn: {e}")
    raise ValueError(
        f"Unknown normalize_fn string '{normalize_fn}'. "
        "Pass None, 'batch', normalize_by_mean(mean), or 'normalize_by_mean(mean)' as a string."
    )

def _resolve_dims(t, dims):
    """Return dims resolved to a tuple of dimension indices.
    """
    if dims is not None and len(t.shape) < len(dims):
        t = t.unsqueeze(-1).expand(-1,-1,dims[-1])
    return t

def _resolve_nan_mask(nan_mask_idx, nan_mask, fp_batch, dims=None):
    """Return the nan_mask to use, resolving from fp_batch if needed.

    Priority: explicit nan_mask argument > nan_mask_label from fp_batch > None.

    nan_mask convention: 1 = invalid/NaN, 0 = valid.
    """
    # make sure that nan_mask has the same number of dimensions as pred,
    if nan_mask is not None:
        nan_mask = _resolve_dims(nan_mask, dims)
        return nan_mask
    if nan_mask_idx is not None and fp_batch is not None:
        nan_mask = fp_batch[..., nan_mask_idx].bool()
        nan_mask = _resolve_dims(nan_mask, dims)
        return nan_mask       
    return None


def _label_index(fp_labels, label, arg_name='weight_label'):
    """Return the index of label in fp_labels.  Raises ValueError if absent."""
    if fp_labels is None or label not in fp_labels:
        raise ValueError(
            f"{arg_name} '{label}' not found in fp_labels {fp_labels}"
        )
    return fp_labels.index(label)


def _filter_kwargs(func, kwargs):
    """Filter kwargs to only include parameters that func accepts.
    
    Uses inspect.signature to determine which parameters the function accepts,
    then returns only the kwargs that match those parameters.
    Handles None gracefully by returning an empty dict.
    
    Args:
        func: callable object to inspect
        kwargs: dict of keyword arguments to filter
    
    Returns:
        dict: filtered kwargs containing only valid parameters for func
    """
    if func is None or not callable(func):
        return {}
    
    try:
        sig = inspect.signature(func)
        # Get parameter names, excluding 'self' and the first positional param (s)
        valid_params = set(sig.parameters.keys())
        # Remove the first parameter (usually 's' for transform functions)
        param_list = list(sig.parameters.keys())
        if param_list:
            valid_params.discard(param_list[0])
        
        # Filter kwargs to only include valid parameters
        return {k: v for k, v in kwargs.items() if k in valid_params}
    except Exception as e:
        print(f"Warning: Could not inspect function signature: {e}")
        # If inspection fails, return all kwargs (original behavior)
        return kwargs


# ---------------------------------------------------------------------------
# Loss functions
#
# All forward methods use torch.nanmean / torch.nansum throughout, so any NaNs
# already present in the data are also ignored without special handling.
#
# Shared conventions:
#   fp_batch=None        — forward always accepts fp_batch even if unused,
#                          so a single training loop can call every loss identically
#   nan_mask_label       — optional constructor arg; if set, the nan_mask is
#                          extracted from fp_batch[..., nan_mask_label_idx] rather
#                          than passed separately in forward
#   fp_label='ones'      — sentinel: use a tensor of ones; fp_batch not required
#   transform_fn            — maps the weight field to a per-pixel multiplier
#                          (e.g. scale, power); **weight_kwargs are forwarded
#   normalize_fn         — makes a per-sample scalar scale-independent before it is
#                          used as a weight or compared as an integral
#                          (e.g. normalize_batch, normalize_by_mean(mean))
# ---------------------------------------------------------------------------

class MSELoss(nn.Module):
    """Standard MSE between predicted and target footprints.

    Usage::
        criterion = MSELoss()
        loss = criterion(pred, target)
        loss = criterion(pred, target, fp_batch, nan_mask=nan_mask)

        # Extract nan_mask automatically from fp_batch:
        criterion = MSELoss(fp_labels, nan_mask_label='fp_nan_mask')
        loss = criterion(pred, target, fp_batch)

    Args:
        fp_labels      (list[str] | None): variable names along the last dim of
                       fp_batch; only required when nan_mask_label is set
        nan_mask_label (str | None): extract nan_mask from this fp_batch variable
                       instead of passing it as a forward argument (default None)

        pred     (Tensor): model output,           shape (B, H, W) or (B, H*W)
        target   (Tensor): ground truth footprint,  same shape as pred
        fp_batch (Tensor | None): supporting data,  shape (B, H, W, V) or (B, H*W, V)
        nan_mask (Tensor | None): 1=invalid pixel,  shape matching pred (default None)
    """

    def __init__(self, fp_labels=None, nan_mask_label=None):
        super().__init__()
        if nan_mask_label is not None:
            if fp_labels is None:
                raise ValueError("fp_labels must be provided if nan_mask_label is set")
            self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') 
        else:
            self.nan_mask_idx = None

    def forward(self, pred, target, fp_batch=None, nan_mask=None):
        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)

        se = _mask((pred - target) ** 2, nan_mask)
        return torch.nanmean(se)

class ThresholdedMSELoss(nn.Module):
    """MSE loss calculated separately for pixels in value ranges defined by thresholds on a chosen fp_batch variable, then summed with optional weighting.
    For example, threshold=0 calculates the MSE separately for pixels where fp_batch[..., weight_label] <= 0 and > 0, then sums them. THis avoids the dampening of MSE scale that occurs when calculating MSE over many near-zero pixels. Alpha controls the relative weight of the MSE in different pixel subsets.  

    formula:
    L = sum_i alpha_i * mse((pred - target) where (fp_batch[..., weight_label] in bin_i))
    where the bins are defined by the thresholds as:
    bin_0: (-inf, threshold_0]
    bin_1: (threshold_0, threshold_1]
    ...
    bin_n: (threshold_{n-1}, inf)

    = mse((pred - target) where (fp_batch[..., weight_label] > threshold)) + alpha * mse((pred - target) where (fp_batch[..., weight_label] <= threshold))

    Usage::
        criterion = ThresholdedMSELoss(fp_labels, weight_label='fp', threshold=0, alpha=1.0, nan_mask_label='fp_nan_mask')
        loss = criterion(pred, target, fp_batch)
        loss = criterion(pred, target, fp_batch, nan_mask=nan_mask)

        Multiple thresholds and alphas:
        criterion = ThresholdedMSELoss(fp_labels, weight_label='fp', threshold=[0, 1e-3], alpha=[1.0, 2.0], nan_mask_label='fp_nan_mask')
        loss = criterion(pred, target, fp_batch)

    Args:
        fp_labels      (list[str]): variable names along the last dim of fp_batch
        weight_label   (str): which fp_labels entry to use for thresholding
        threshold      (float or list[float]): thresholds to define pixel subsets; can be a single number or a list for multiple thresholds (default 0)
        alpha          (float or list[float]): relative weight(s) for the MSE in each pixel subset; can be a single number or a list of the same length as threshold + 1 (default 1.0). If a single number is provided, the first subset (pixels <= threshold) is weighted by one, and all the subsequent subsets (pixels > threshold) are weighted by alpha. If a list is provided, the first entry is the weight for the first subset (pixels <= threshold_0), the second entry is the weight for the second subset (threshold_0 < pixels <= threshold_1), and so on, with the last entry being the weight for the final subset (pixels > threshold_{n-1}).
        nan_mask_label (str | None): extract nan_mask from this fp_batch variable instead of passing it as a forward argument (default None)

    """
    def __init__(self, fp_labels, weight_label, threshold=0, alpha=1.0, nan_mask_label=None):
        super().__init__()
        if isinstance(threshold, (int, float)):
            threshold = [threshold]
        if not isinstance(threshold, list):
            raise ValueError("threshold must be a list or a single number")
        if isinstance(alpha, (int, float)):
            alpha = [1.0] + [alpha] * len(threshold)

        if not isinstance(alpha, list)  or len(alpha) != len(threshold) + 1:
            raise ValueError("alpha must be a list of the same length as threshold +1, or a single number")


        self.weight_idx = _label_index(fp_labels, weight_label)
        self.threshold = threshold
        self.bins = [-float('inf')] + self.threshold + [float('inf')]
        self.alpha = alpha
        self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') if nan_mask_label else None
    
    def forward(self, pred, target, fp_batch, nan_mask=None):
        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)
        weight_field = fp_batch[..., self.weight_idx]

        weight_field = _resolve_dims(weight_field, pred.shape)

        # add min and max in batch to threshold to make bins        
        added_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for bin_start, bin_end, a in zip(self.bins[:-1], self.bins[1:], self.alpha):
            mask = (weight_field > bin_start) & (weight_field <= bin_end) 
            if nan_mask is not None:
                mask = mask & ~nan_mask
            if mask.sum() != 0:
                mse = torch.nanmean((pred[mask] - target[mask]) ** 2)
                added_loss += a * mse

        return added_loss


class PixelWeightedMSELoss(nn.Module):
    """Per-pixel MSE weighted element-wise by a chosen fp_batch variable.

    The per-pixel squared error is multiplied by transform_fn(fp_batch[..., weight_label]),
    then averaged across all valid pixels and samples.

    formula:
    L = nanmean((pred - target)^2 * transform_fn(fp_batch[..., weight_label]))

    Penalise errors more in pixels where fp_batch[..., weight_label] is large (after transformation).

    # This function is equivalent to WeightedByTruth in the original codebase, with weight_by_label = "fp_original", transform_fn = scale_and_shift, w=1000, a=1.0.

    Any nans in the original data can be ignored from the calculation using nan_mask. Either pass nan_mask explicitly to forward, or set nan_mask_label to extract it from fp_batch.

    Usage::
        criterion = PixelWeightedMSELoss(fp_labels, weight_label='fp_original',
                                      transform_fn=scale, w=500)
        loss = criterion(pred, target, fp_batch)
        loss = criterion(pred, target, fp_batch, nan_mask=nan_mask)

    Args:
        fp_labels      (list[str]): variable names along the last dim of fp_batch
        weight_label   (str): which fp_labels entry to use as the per-pixel weight
        transform_fn   (callable | None): transform function applied to the weight tensor before
                       multiplying; e.g. scale, scale_and_shift, power;
                       extra kwargs are forwarded (default None = identity)
        nan_mask_label (str | None): extract nan_mask from this fp_batch variable
                       instead of passing it as a forward argument (default None)
        **transform_kwargs: keyword arguments forwarded to transform_fn

        pred     (Tensor): model output,           shape (B, H, W) or (B, H*W)
        target   (Tensor): ground truth footprint,  same shape as pred
        fp_batch (Tensor): supporting data,         shape (B, H, W, V) or (B, H*W, V)
        nan_mask (Tensor | None): 1=invalid pixel,  shape matching pred (default None)
    """

    def __init__(self, fp_labels, weight_label='fp_original',
                 transform_fn=None, nan_mask_label=None, **transform_kwargs):
        super().__init__()
        self.fp_weight_idx = _label_index(fp_labels, weight_label)
        if type(transform_fn) == str:
            transform_fn = eval(transform_fn)
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs
        self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') \
            if nan_mask_label else None

    def forward(self, pred, target, fp_batch, nan_mask=None):
        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)
        weights = fp_batch[..., self.fp_weight_idx]

        weights = _resolve_dims(weights, pred.shape)

        if self.transform_fn is not None:
            # Filter kwargs to only include parameters that transform_fn accepts
            filtered_kwargs = _filter_kwargs(self.transform_fn, self.transform_kwargs)
            weights = self.transform_fn(weights, **filtered_kwargs)
        loss = _mask((pred - target) ** 2 * weights, nan_mask)
        return torch.nanmean(loss)




class SumWeightedMSELoss(nn.Module):
    """Per-footprint MSE weighted by a function of the spatial sum of a chosen variable.

    Each footprint's MSE is multiplied by a per-sample weight derived from the spatial
    sum of fp_batch[..., weight_label] (applying normalize_fn and/or transform_fn as specified) 
    
    Formula:
    L = nanmean(per_sample_mse * weight)

    Penalises errors more in footprints where the spatial sum of fp_batch[..., weight_label] is large. For example, if weight_label='fp_original', the sum is the total original footprint mass, so this loss penalises errors more in footprints with larger total mass. If weight_label='flux', the sum is the total flux-weighted footprint mass, so this loss penalises errors more in footprints that contribute more to the tracer concentration in the inversion.
    
    normalize_fn controls whether and how the sums are normalised before weighting.
    Pass None (default) for no normalisation, "batch" for per-batch normalisation,
    or normalize_by_mean(dataset_mean) for dataset-level normalisation. transform_fn controls applies a transformation to the sums before weighting.

    Usage::
        # multiplicative, no normalisation (default)
        criterion = SumWeightedMSELoss(fp_labels, weight_label='flux', transform_fn=scale, w=1e-9)
        # with per-batch normalisation
        criterion = SumWeightedMSELoss(fp_labels, weight_label='fp_original',  normalize_fn='batch')
        # with dataset-level normalisation
        criterion = SumWeightedMSELoss(fp_labels, weight_label='fp_original',  normalize_fn=normalize_by_mean(fp_sum_mean))

        loss = criterion(pred, target, fp_batch)
        loss = criterion(pred, target, fp_batch, nan_mask=nan_mask)

    Args:
        fp_labels      (list[str]): variable names along the last dim of fp_batch
        weight_label   (str): which variable's spatial sum to derive weights from
        normalize_fn   (callable | "batch" | None): applied to fp_sum before
                       transform_fn; "batch" selects normalize_batch (default None)
        transform_fn   (callable): maps fp_sum (B,) to weights (B,); default scale
        nan_mask_label (str | None): extract nan_mask from this fp_batch variable
                       instead of passing it as a forward argument (default None)
        **transform_kwargs: forwarded to transform_fn

        pred     (Tensor): model output,           shape (B, H, W) or (B, H*W)
        target   (Tensor): ground truth footprint,  same shape as pred
        fp_batch (Tensor): supporting data,         shape (B, H, W, V) or (B, H*W, V)
        nan_mask (Tensor | None): 1=invalid pixel,  shape matching pred (default None)
    """

    def __init__(self, fp_labels, weight_label='fp_original',
                 normalize_fn=None, transform_fn=None,
                 nan_mask_label=None, **transform_kwargs):
        super().__init__()
        self.weight_idx = _label_index(fp_labels, weight_label)
        self.normalize_fn = _get_normalize_fn(normalize_fn)
        if type(transform_fn) == str:
            transform_fn = eval(transform_fn)
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs
        self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') \
            if nan_mask_label else None

    def forward(self, pred, target, fp_batch, nan_mask=None):
        nan_mask_preds = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)
        spatial = _spatial_dims(pred)
        # take MSE per sample
        se = _mask((pred - target) ** 2, nan_mask_preds)
        per_sample_mse = torch.nanmean(se, dim=spatial)   # (B,)

        # sum the weight field over space
        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch)
        weights = _mask(fp_batch[..., self.weight_idx], nan_mask)

        weights_sum = torch.nansum(weights, dim=_spatial_dims(weights))  # (B,)

        if self.normalize_fn is not None:
            weights_sum = self.normalize_fn(weights_sum)

        if self.transform_fn is not None:
            # Filter kwargs to only include parameters that transform_fn accepts
            filtered_kwargs = _filter_kwargs(self.transform_fn, self.transform_kwargs)
            weights_sum = self.transform_fn(weights_sum, **filtered_kwargs)   # (B,)

        return torch.nanmean(per_sample_mse * weights_sum)


class MSEPlusSumLoss(nn.Module):
    """Loss function with two terms: pixel-level MSE plus the error in the spatial sum of the footprint, multiplied by a field.

    Two-term loss::

        L = nanmean((pred - target)^2)
          + alpha * nanmean((sum_pred - sum_target)^2)

    where the per-sample sum is::

        sum_i = nansum(pred_i * w_i, dim=spatial)

    and w is taken from fp_batch[..., weight_label], optionally transformed by transform_fn,
    or set to ones when weight_label='ones' (fp_batch not required in that case).

    If weight_label="flux", the sum is the flux-weighted footprint mass, so the second term penalises errors in emulated mole fractions. If weight_label="ones", the sum is the total footprint mass, so the second term penalises errors in the total mass of the footprint.

    normalize_fn is applied to the integral scalars before computing their squared
    error, making the penalty scale-independent. the normalisation can be per-batch ("batch") or dataset-level (normalize_by_mean(mean)).

    alpha controls the relative weight of the two terms. Setting alpha=0 recovers the standard MSELoss.

    Usage::
        # Penalise error in total footprint mass (fp_label='ones' is the default):
        criterion = MSEPlusSumLoss(alpha=1.0)
        loss = criterion(pred, target)

        # Penalise error in flux-weighted integral:
        criterion = MSEPlusSumLoss(fp_labels, weight_label='flux', alpha=1.0)
        loss = criterion(pred, target, fp_batch)

        # With dataset-level normalisation of the integral:
        criterion = MSEPlusSumLoss(fp_labels, weight_label='flux',  normalize_fn=normalize_by_mean(integral_mean),
         alpha=1.0)
        loss = criterion(pred, target, fp_batch)

    Args:
        fp_labels      (list[str] | None): variable names along the last dim of
                       fp_batch; not required when weight_label='ones'
        weight_label   (str): fp_batch variable to use as weight field w, or 'ones'
                       to weight by ones (default 'ones')
        alpha    (float): scalar weight on the integral-error penalty (default 1.0)
        normalize_fn   (callable | "batch" | None): applied to the per-sample integral
                       scalars before squaring their error (default None)
        transform_fn   (callable | None): applied to w before multiplying with
                       pred / target (default None = identity)
        nan_mask_label (str | None): extract nan_mask from this fp_batch variable
                       instead of passing it as a forward argument (default None)
        **transform_kwargs: forwarded to transform_fn

        pred     (Tensor): model output,           shape (B, H, W) or (B, H*W)
        target   (Tensor): ground truth footprint,  same shape as pred
        fp_batch (Tensor | None): supporting data,  shape (B, H, W, V) or (B, H*W, V); not required when weight_label='ones' and nan_mask_label is None
        nan_mask (Tensor | None): 1=invalid pixel,  shape matching pred (default None)
    """

    def __init__(self, fp_labels=None, weight_label='ones',
                 alpha=1.0, normalize_fn=None,
                 transform_fn=None, nan_mask_label=None, **transform_kwargs):
        super().__init__()
        self.weight_idx = "ones" if weight_label == 'ones' else _label_index(fp_labels, weight_label)
        self.alpha = alpha
        self.normalize_fn = _get_normalize_fn(normalize_fn)
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs
        self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') \
            if nan_mask_label else None

    def forward(self, pred, target, fp_batch=None, nan_mask=None):
        nan_mask_preds = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)
        spatial = _spatial_dims(pred)

        # calculate the sample-level MSE 
        se     = _mask((pred - target) ** 2, nan_mask_preds)
        mse = torch.nanmean(se)

        pred_m = _mask(pred, nan_mask_preds)
        tgt_m  = _mask(target, nan_mask_preds)


        if self.weight_idx == "ones":
            w = torch.ones_like(pred_m)
        else:
            #w = _resolve_dims(fp_batch[..., self.weight_idx], pred.shape)
            nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch)
            w = fp_batch[..., self.weight_idx]
            w = _mask(w, nan_mask)

        if self.transform_fn is not None:
            # Filter kwargs to only include parameters that transform_fn accepts
            filtered_kwargs = _filter_kwargs(self.transform_fn, self.transform_kwargs)
            w = self.transform_fn(w, **filtered_kwargs)
        
        w = _resolve_dims(w, pred.shape)

        integral_pred   = torch.nansum(pred_m * w, dim=spatial)    # (B,)
        integral_target = torch.nansum(tgt_m  * w, dim=spatial)    # (B,)

        if self.normalize_fn is not None:
            # normalize both integrals together instead of separately, to preserve their relative scale
            integral_stack = torch.stack([integral_pred, integral_target], dim=0)  # (2, B)
            integral_stack = self.normalize_fn(integral_stack)
            integral_pred   = integral_stack[0]
            integral_target = integral_stack[1]

        integral_mse = torch.nanmean((integral_pred - integral_target) ** 2)

        return mse + self.alpha * integral_mse
