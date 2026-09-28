import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper functions
#
# All helpers share the same calling convention: fn(s, **kwargs) -> Tensor,
# where s is the tensor to transform.  They can be used interchangeably as
# the `transform_fn` argument in PixelWeightedMSE / SumWeightedMSE, or the
# `normalize_fn` argument.
# ---------------------------------------------------------------------------

def scale(s, w=1.0):
    """Multiply by a scalar weight w.

    Args:
        s (torch.Tensor): Tensor to transform.
        w (float, optional): Scalar weight. Defaults to 1.0.

    Returns:
        torch.Tensor: ``w * s``.
    """
    return w * s


def scale_and_shift(s, w=1.0, a=0.0):
    """Multiply by w and add constant a.

    Args:
        s (torch.Tensor): Tensor to transform.
        w (float, optional): Scalar weight. Defaults to 1.0.
        a (float, optional): Constant to add. Defaults to 0.0.

    Returns:
        torch.Tensor: ``w * s + a``.
    """
    return w * s + a


def power(s, p=2.0):
    """Raise to the power p. Amplifies differences between large and small values.

    Args:
        s (torch.Tensor): Tensor to transform.
        p (float, optional): Exponent. Defaults to 2.0.

    Returns:
        torch.Tensor: ``s ** p``.
    """
    return s ** p


def normalize_batch(s):
    """Divide by the batch mean (per-batch normalisation).

    Weights become scale-independent but vary with batch composition: the same
    sample may receive a different weight depending on what else is in the batch.

    Args:
        s (torch.Tensor): Tensor to normalise.

    Returns:
        torch.Tensor: ``s / (s.mean() + 1e-8)``.
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

    Args:
        mean (float or torch.Tensor): Dataset-level mean to normalise by.

    Returns:
        callable: Function ``s -> s / (mean + 1e-8)``.
    """
    def _normalize(s):
        return s / (mean + 1e-8)
    return _normalize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spatial_dims(t):
    """All dimension indices except the batch dimension (0).

    Args:
        t (torch.Tensor): Tensor whose spatial dims to compute.

    Returns:
        tuple[int]: Dimension indices ``1, ..., t.dim() - 1``.
    """
    return tuple(range(1, t.dim()))


def _mask(t, nan_mask):
    """Return t with nan_mask positions set to NaN (no-op if nan_mask is None).

    nan_mask convention: 1 = invalid/NaN, 0 = valid.

    Args:
        t (torch.Tensor): Tensor to mask.
        nan_mask (torch.Tensor or None): Positions to set to NaN, or None for no-op.

    Returns:
        torch.Tensor: ``t`` with ``nan_mask`` positions set to NaN.
    """
    if nan_mask is None:
        return t
    return t.masked_fill(nan_mask.bool(), float('nan'))


def _get_normalize_fn(normalize_fn):
    """Resolve normalize_fn to a callable (or None).

    Args:
        normalize_fn (callable, str, or None): One of:

            - None: no normalisation.
            - "batch": resolves to ``normalize_batch``.
            - callable: used as-is (e.g. ``normalize_by_mean(mean)``).
            - other str: evaluated with ``eval()`` (e.g. ``"normalize_by_mean(mean)"``).

    Returns:
        callable or None: The resolved normalisation function, or None.

    Raises:
        ValueError: If ``normalize_fn`` is a string that cannot be resolved.
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
    """Expand ``t`` with a trailing broadcast dimension if it has fewer dims than ``dims``.

    Args:
        t (torch.Tensor): Tensor to (possibly) expand.
        dims (tuple or torch.Size): Target shape to match the number of dimensions of.

    Returns:
        torch.Tensor: ``t``, expanded with a trailing dimension if needed.
    """
    if dims is not None and len(t.shape) < len(dims):
        t = t.unsqueeze(-1).expand(-1,-1,dims[-1])
    return t

def _resolve_nan_mask(nan_mask_idx, nan_mask, fp_batch, dims=None):
    """Return the nan_mask to use, resolving from fp_batch if needed.

    Priority: explicit nan_mask argument > nan_mask_label from fp_batch > None.

    nan_mask convention: 1 = invalid/NaN, 0 = valid.

    Args:
        nan_mask_idx (int or None): Index of the nan-mask variable in ``fp_batch``'s
            last dimension, or None if not using this source.
        nan_mask (torch.Tensor or None): Explicit nan mask, takes priority if given.
        fp_batch (torch.Tensor or None): Supporting data tensor to extract the nan
            mask from when ``nan_mask`` is None and ``nan_mask_idx`` is set.
        dims (tuple, optional): Target shape to resolve the mask's dimensions
            against (see ``_resolve_dims``). Defaults to None.

    Returns:
        torch.Tensor or None: The resolved nan mask, or None if neither source is available.
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
    """Return the index of label in fp_labels.

    Args:
        fp_labels (list[str]): Variable names to search.
        label (str): Label to find the index of.
        arg_name (str, optional): Name to use in the error message if not found.
            Defaults to 'weight_label'.

    Returns:
        int: Index of ``label`` in ``fp_labels``.

    Raises:
        ValueError: If ``fp_labels`` is None or ``label`` is not found in it.
    """
    if fp_labels is None or label not in fp_labels:
        raise ValueError(
            f"{arg_name} '{label}' not found in fp_labels {fp_labels}"
        )
    return fp_labels.index(label)


def _to_2d(t, spatial_shape=None):
    """Reshape a flattened footprint tensor to ``(B, H, W)`` for spatial ops.

    The training loop flattens footprints, so a prediction arrives as ``(B, N)``
    or ``(B, N, 1)`` with ``N = H * W``. Spatial losses (gradient, structure) need
    the 2D layout back. Tensors already shaped ``(B, H, W)`` are returned unchanged.

    nan_mask convention: 1 = invalid/NaN, 0 = valid (preserved by the reshape).

    Args:
        t (torch.Tensor or None): Tensor shaped ``(B, H, W)``, ``(B, N)`` or
            ``(B, N, 1)``. None is passed through as None.
        spatial_shape (tuple[int, int] or None, optional): Explicit ``(H, W)`` for
            non-square domains. If None, a square ``H = W = sqrt(N)`` is assumed.
            Defaults to None.

    Returns:
        torch.Tensor or None: ``t`` reshaped to ``(B, H, W)``, or None.

    Raises:
        ValueError: If ``t`` has an unsupported rank, or ``N`` cannot be reshaped
            to the requested / inferred ``(H, W)``.
    """
    if t is None:
        return None
    # drop a trailing singleton channel dim: (B, N, 1) -> (B, N)
    if t.dim() == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.dim() == 3:
        return t  # already (B, H, W)
    if t.dim() != 2:
        raise ValueError(
            f"Expected (B, H, W), (B, N) or (B, N, 1); got shape {tuple(t.shape)}"
        )
    b, n = t.shape
    if spatial_shape is not None:
        h, w = spatial_shape
    else:
        h = int(round(n ** 0.5))
        w = h
    if h * w != n:
        raise ValueError(
            f"Cannot reshape flattened length {n} to ({h}, {w}); "
            "pass spatial_shape=(H, W) for non-square domains."
        )
    return t.reshape(b, h, w)


def _apply_core(e, core="squared", eps=1e-3):
    """Map an error tensor to a per-element penalty.

    ``"squared"`` gives the usual ``e**2``; ``"charbonnier"`` gives the robust
    ``sqrt(e**2 + eps**2)``, a smooth approximation to ``|e|`` that is less
    dominated by large residuals (Charbonnier et al., 1994).

    Args:
        e (torch.Tensor): Error tensor (e.g. ``pred - target``).
        core (str, optional): ``"squared"`` or ``"charbonnier"``. Defaults to
            ``"squared"``.
        eps (float, optional): Charbonnier smoothing constant. Defaults to 1e-3.

    Returns:
        torch.Tensor: Per-element penalty, same shape as ``e``.

    Raises:
        ValueError: If ``core`` is not ``"squared"`` or ``"charbonnier"``.
    """
    if core == "squared":
        return e ** 2
    if core == "charbonnier":
        return torch.sqrt(e ** 2 + eps ** 2)
    raise ValueError(f"Unknown core '{core}'. Use 'squared' or 'charbonnier'.")


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
    """

    def __init__(self, fp_labels=None, nan_mask_label=None):
        """Initialize the loss.

        Args:
            fp_labels (list[str], optional): Variable names along the last dim of
                ``fp_batch``; only required when ``nan_mask_label`` is set.
                Defaults to None.
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.

        Raises:
            ValueError: If ``nan_mask_label`` is set but ``fp_labels`` is None.
        """
        super().__init__()
        if nan_mask_label is not None:
            if fp_labels is None:
                raise ValueError("fp_labels must be provided if nan_mask_label is set")
            self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label')
        else:
            self.nan_mask_idx = None

    def forward(self, pred, target, fp_batch=None, nan_mask=None):
        """Compute the MSE loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W) or (B, H*W).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor, optional): Supporting data, shape (B, H, W, V)
                or (B, H*W, V); used to extract the nan mask if ``nan_mask_label``
                was set. Defaults to None.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar MSE loss.
        """
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
    """
    def __init__(self, fp_labels, weight_label, threshold=0, alpha=1.0, nan_mask_label=None):
        """Initialize the loss.

        Args:
            fp_labels (list[str]): Variable names along the last dim of ``fp_batch``.
            weight_label (str): Which ``fp_labels`` entry to use for thresholding.
            threshold (float or list[float], optional): Thresholds to define pixel
                subsets; can be a single number or a list for multiple thresholds.
                Defaults to 0.
            alpha (float or list[float], optional): Relative weight(s) for the MSE in
                each pixel subset; can be a single number or a list of the same
                length as ``threshold`` + 1. If a single number is provided, the
                first subset (pixels <= threshold) is weighted by one, and all the
                subsequent subsets (pixels > threshold) are weighted by ``alpha``. If
                a list is provided, the first entry is the weight for the first
                subset (pixels <= threshold_0), the second entry is the weight for
                the second subset (threshold_0 < pixels <= threshold_1), and so on,
                with the last entry being the weight for the final subset (pixels >
                threshold_{n-1}). Defaults to 1.0.
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.

        Raises:
            ValueError: If ``threshold`` is not a number or list, or if ``alpha`` is
                not a list of length ``len(threshold) + 1`` (after normalising a
                single-number ``alpha``).
        """
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
        """Compute the thresholded, summed MSE loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W) or (B, H*W).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor): Supporting data, shape (B, H, W, V) or
                (B, H*W, V); provides the thresholding field and, optionally, the
                nan mask.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar loss, the alpha-weighted sum of per-bin MSEs.
        """
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
    """

    def __init__(self, fp_labels, weight_label='fp_original',
                 transform_fn=None, nan_mask_label=None, **transform_kwargs):
        """Initialize the loss.

        Args:
            fp_labels (list[str]): Variable names along the last dim of ``fp_batch``.
            weight_label (str, optional): Which ``fp_labels`` entry to use as the
                per-pixel weight. Defaults to 'fp_original'.
            transform_fn (callable or str, optional): Transform function applied to
                the weight tensor before multiplying, e.g. ``scale``,
                ``scale_and_shift``, ``power``; extra kwargs are forwarded. A string
                is evaluated with ``eval()``. Defaults to None (identity).
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.
            **transform_kwargs: Keyword arguments forwarded to ``transform_fn``.
        """
        super().__init__()
        self.fp_weight_idx = _label_index(fp_labels, weight_label)
        if type(transform_fn) == str:
            transform_fn = eval(transform_fn)
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs
        self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') \
            if nan_mask_label else None

    def forward(self, pred, target, fp_batch, nan_mask=None):
        """Compute the pixel-weighted MSE loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W) or (B, H*W).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor): Supporting data, shape (B, H, W, V) or
                (B, H*W, V); provides the per-pixel weight field and, optionally,
                the nan mask.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar, weighted MSE loss.
        """
        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)
        weights = fp_batch[..., self.fp_weight_idx]

        weights = _resolve_dims(weights, pred.shape)

        if self.transform_fn is not None:
            weights = self.transform_fn(weights, **self.transform_kwargs)
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
    """

    def __init__(self, fp_labels, weight_label='fp_original',
                 normalize_fn=None, transform_fn=None,
                 nan_mask_label=None, **transform_kwargs):
        """Initialize the loss.

        Args:
            fp_labels (list[str]): Variable names along the last dim of ``fp_batch``.
            weight_label (str, optional): Which variable's spatial sum to derive
                weights from. Defaults to 'fp_original'.
            normalize_fn (callable, "batch", or None, optional): Applied to fp_sum
                before ``transform_fn``; "batch" selects ``normalize_batch``.
                Defaults to None.
            transform_fn (callable or str, optional): Maps fp_sum (B,) to weights
                (B,). A string is evaluated with ``eval()``. Defaults to None
                (identity — no transform).
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.
            **transform_kwargs: Forwarded to ``transform_fn``.
        """
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
        """Compute the sum-weighted MSE loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W) or (B, H*W).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor): Supporting data, shape (B, H, W, V) or
                (B, H*W, V); provides the field to spatially sum for weighting and,
                optionally, the nan mask.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar, sum-weighted MSE loss.
        """
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
            weights_sum = self.transform_fn(weights_sum, **self.transform_kwargs)   # (B,)

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
    """

    def __init__(self, fp_labels=None, weight_label='ones',
                 alpha=1.0, normalize_fn=None,
                 transform_fn=None, nan_mask_label=None, **transform_kwargs):
        """Initialize the loss.

        Args:
            fp_labels (list[str], optional): Variable names along the last dim of
                ``fp_batch``; not required when ``weight_label='ones'``.
                Defaults to None.
            weight_label (str, optional): ``fp_batch`` variable to use as weight
                field w, or "ones" to weight by ones. Defaults to 'ones'.
            alpha (float, optional): Scalar weight on the integral-error penalty.
                Defaults to 1.0.
            normalize_fn (callable, "batch", or None, optional): Applied to the
                per-sample integral scalars before squaring their error.
                Defaults to None.
            transform_fn (callable, optional): Applied to w before multiplying with
                pred / target. Defaults to None (identity).
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.
            **transform_kwargs: Forwarded to ``transform_fn``.
        """
        super().__init__()
        self.weight_idx = "ones" if weight_label == 'ones' else _label_index(fp_labels, weight_label)
        self.alpha = alpha
        self.normalize_fn = _get_normalize_fn(normalize_fn)
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs
        self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label') \
            if nan_mask_label else None

    def forward(self, pred, target, fp_batch=None, nan_mask=None):
        """Compute the two-term MSE-plus-integral-error loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W) or (B, H*W).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor, optional): Supporting data, shape (B, H, W, V)
                or (B, H*W, V); provides the weight field w and, optionally, the nan
                mask. Not required when ``weight_label='ones'`` and
                ``nan_mask_label`` is None. Defaults to None.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar loss, ``mse + alpha * integral_mse``.
        """
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
            w = self.transform_fn(w, **self.transform_kwargs)
        
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


class GradientMSELoss(nn.Module):
    """MSE plus a penalty on the spatial-gradient error (anti-blur loss).

    Plain MSE rewards the conditional mean, which is smoother than any single
    footprint: a blurred prediction keeps roughly correct values but collapses
    the field's slopes. This loss adds a term on the spatial gradient, which a
    smoothed prediction cannot satisfy. Because
    ``MSE(grad(pred), grad(target)) == MSE(grad(pred - target))``, the gradient
    term is simply the smoothness of the *error* field — only over-smoothing
    reduces it. This is the Gradient Difference Loss of Mathieu et al. (2016).

    Formula::

        L = core(pred - target)
          + beta * core(grad(pred - target))

    where ``core`` is squared error (default) or the robust Charbonnier
    ``sqrt(e**2 + eps**2)``, and ``grad`` is the forward finite difference along
    both spatial dims. Gradient entries touching a NaN/masked pixel are dropped.

    Handles both ``(B, H, W)`` and flattened ``(B, N)`` / ``(B, N, 1)`` preds; a
    flattened footprint is reshaped to a square ``(H, W)`` unless ``spatial_shape``
    is given (see ``_to_2d``).

    Usage::
        criterion = GradientMSELoss(beta=0.5)
        loss = criterion(pred, target)

        # Charbonnier core + automatic nan mask from fp_batch:
        criterion = GradientMSELoss(fp_labels, nan_mask_label='fp_nan_mask',
                                    beta=0.5, core='charbonnier')
        loss = criterion(pred, target, fp_batch)
    """

    def __init__(self, fp_labels=None, nan_mask_label=None, beta=1.0,
                 spatial_shape=None, core='squared', eps=1e-3):
        """Initialize the loss.

        Args:
            fp_labels (list[str], optional): Variable names along the last dim of
                ``fp_batch``; only required when ``nan_mask_label`` is set.
                Defaults to None.
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.
            beta (float, optional): Weight on the gradient-error term. beta=0
                recovers plain (core) MSE. Defaults to 1.0.
            spatial_shape (tuple[int, int], optional): ``(H, W)`` used to un-flatten
                predictions; None assumes a square domain. Defaults to None.
            core (str, optional): ``"squared"`` or ``"charbonnier"``. Defaults to
                ``"squared"``.
            eps (float, optional): Charbonnier smoothing constant. Defaults to 1e-3.

        Raises:
            ValueError: If ``nan_mask_label`` is set but ``fp_labels`` is None.
        """
        super().__init__()
        if nan_mask_label is not None:
            if fp_labels is None:
                raise ValueError("fp_labels must be provided if nan_mask_label is set")
            self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label')
        else:
            self.nan_mask_idx = None
        self.beta = beta
        self.spatial_shape = spatial_shape
        self.core = core
        self.eps = eps

    def forward(self, pred, target, fp_batch=None, nan_mask=None):
        """Compute the gradient-augmented MSE loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W), (B, N) or (B, N, 1).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor, optional): Supporting data used to extract the
                nan mask if ``nan_mask_label`` was set. Defaults to None.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar loss, ``base + beta * gradient_error``.
        """
        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)

        pred2 = _to_2d(pred, self.spatial_shape)
        tgt2 = _to_2d(target, self.spatial_shape)
        mask2 = _to_2d(nan_mask, self.spatial_shape)

        e = pred2 - tgt2

        # base per-pixel term
        base = torch.nanmean(_mask(_apply_core(e, self.core, self.eps), mask2))

        # spatial gradient of the error field (== grad(pred) - grad(target))
        e_filled = e if mask2 is None else e.masked_fill(mask2.bool(), 0.0)
        dH = e_filled[:, 1:, :] - e_filled[:, :-1, :]
        dW = e_filled[:, :, 1:] - e_filled[:, :, :-1]

        if mask2 is not None:
            valid = ~mask2.bool()
            # a difference is only valid if both pixels it spans are valid
            valid_H = valid[:, 1:, :] & valid[:, :-1, :]
            valid_W = valid[:, :, 1:] & valid[:, :, :-1]
            dH = dH.masked_fill(~valid_H, float('nan'))
            dW = dW.masked_fill(~valid_W, float('nan'))

        grad = torch.nanmean(torch.cat([
            _apply_core(dH, self.core, self.eps).reshape(-1),
            _apply_core(dW, self.core, self.eps).reshape(-1),
        ]))

        return base + self.beta * grad


class StructuralLoss(nn.Module):
    """MSE plus a structural term rewarding pattern/shape agreement.

    Per-pixel MSE is scale-driven and blur-prone. A structural term instead
    scores whether the prediction has the same spatial *pattern* as the truth,
    largely independent of per-footprint offset or scale.

    Formula::

        L = core(pred - target) + gamma * structural

    Structural terms (``mode``):

    - ``"correlation"`` (default): per-sample spatial Pearson correlation between
      predicted and true fields over valid pixels; ``structural = mean(1 - corr)``.
      This is the Anomaly/Pattern Correlation Coefficient used as a standard
      spatial-forecast skill score. Cheap, no windowing, and applied over the
      flattened spatial dims directly.
    - ``"ms_ssim"``: multi-scale structural similarity (Wang et al., 2003), used
      as ``1 - MS-SSIM``. Not yet implemented (raises ``NotImplementedError``).

    Both terms are best applied in the model's transformed (log) space and kept
    additive to MSE — footprints are sparse/heavy-tailed, so a standalone
    structural term is ill-conditioned.

    Usage::
        criterion = StructuralLoss(gamma=0.3)  # mode='correlation'
        loss = criterion(pred, target)

        criterion = StructuralLoss(fp_labels, nan_mask_label='fp_nan_mask',
                                   gamma=0.3, mode='correlation')
        loss = criterion(pred, target, fp_batch)
    """

    def __init__(self, fp_labels=None, nan_mask_label=None, gamma=1.0,
                 mode='correlation', core='squared', eps=1e-3):
        """Initialize the loss.

        Args:
            fp_labels (list[str], optional): Variable names along the last dim of
                ``fp_batch``; only required when ``nan_mask_label`` is set.
                Defaults to None.
            nan_mask_label (str, optional): Extract nan_mask from this ``fp_batch``
                variable instead of passing it as a forward argument. Defaults to None.
            gamma (float, optional): Weight on the structural term. gamma=0 recovers
                plain (core) MSE. Defaults to 1.0.
            mode (str, optional): ``"correlation"`` (spatial Pearson) or
                ``"ms_ssim"`` (not yet implemented). Defaults to ``"correlation"``.
            core (str, optional): ``"squared"`` or ``"charbonnier"`` for the base
                MSE term. Defaults to ``"squared"``.
            eps (float, optional): Charbonnier smoothing constant. Defaults to 1e-3.

        Raises:
            ValueError: If ``nan_mask_label`` is set but ``fp_labels`` is None, or
                if ``mode`` is not a recognised option.
        """
        super().__init__()
        if nan_mask_label is not None:
            if fp_labels is None:
                raise ValueError("fp_labels must be provided if nan_mask_label is set")
            self.nan_mask_idx = _label_index(fp_labels, nan_mask_label, 'nan_mask_label')
        else:
            self.nan_mask_idx = None
        if mode not in ('correlation', 'ms_ssim'):
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'correlation' or 'ms_ssim'."
            )
        self.gamma = gamma
        self.mode = mode
        self.core = core
        self.eps = eps

    def forward(self, pred, target, fp_batch=None, nan_mask=None):
        """Compute the structure-augmented MSE loss.

        Args:
            pred (torch.Tensor): Model output, shape (B, H, W), (B, N) or (B, N, 1).
            target (torch.Tensor): Ground truth footprint, same shape as ``pred``.
            fp_batch (torch.Tensor, optional): Supporting data used to extract the
                nan mask if ``nan_mask_label`` was set. Defaults to None.
            nan_mask (torch.Tensor, optional): 1=invalid pixel, shape matching
                ``pred``. Defaults to None.

        Returns:
            torch.Tensor: Scalar loss, ``base + gamma * structural``.

        Raises:
            NotImplementedError: If ``mode='ms_ssim'`` (not yet implemented).
        """
        if self.mode == 'ms_ssim':
            raise NotImplementedError(
                "StructuralLoss mode='ms_ssim' is not implemented yet; "
                "use mode='correlation'."
            )

        nan_mask = _resolve_nan_mask(self.nan_mask_idx, nan_mask, fp_batch, dims=pred.shape)
        spatial = _spatial_dims(pred)

        # base per-pixel term
        base = torch.nanmean(_mask(_apply_core(pred - target, self.core, self.eps), nan_mask))

        # per-sample spatial Pearson correlation over valid pixels
        p = _mask(pred, nan_mask)
        t = _mask(target, nan_mask)
        p_centred = p - torch.nanmean(p, dim=spatial, keepdim=True)
        t_centred = t - torch.nanmean(t, dim=spatial, keepdim=True)

        cov = torch.nansum(p_centred * t_centred, dim=spatial)
        var_p = torch.nansum(p_centred ** 2, dim=spatial)
        var_t = torch.nansum(t_centred ** 2, dim=spatial)
        corr = cov / (torch.sqrt(var_p * var_t) + 1e-8)   # (B,)

        structural = torch.nanmean(1.0 - corr)

        return base + self.gamma * structural
