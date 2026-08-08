"""ANALYSIS-ONLY diagnostics for the dual-head model (footprint + background heads).

Nothing in this module affects optimization: gradients are measured with
``torch.autograd.grad`` (which does not write into ``param.grad``), the training graph is
left intact for the usual ``loss.backward()``, and every output goes to logging only.
Removing all calls into this module leaves training numerically unchanged (verified by
``tests/test_dual_analysis.py``).

What it measures, per training epoch (logged under the ``analysis/`` W&B prefix and
appended to the run's ``*_updates.txt``):

- **Loss-scale diagnostics** (free — computed from the already-averaged epoch losses):
  the raw ratio ``fp_loss / bg_loss`` for train and test, and the *contribution* ratio
  ``(1-w)·fp_loss / (w·bg_loss)`` with ``w = bg_loss_weight``. The two heads' criteria
  are in different units, so ``w`` silently absorbs a scale factor; these make it visible.
- **Trunk-gradient diagnostics** (cost: two extra partial backward passes on each
  measured batch): the L2 norm each head's loss alone induces on the SHARED trunk
  (encoder + processor), the weighted-norm ratio ``(1-w)·|g_fp| / (w·|g_bg|)`` (which
  head actually dominates the shared parameters' updates), and the cosine similarity
  between the two trunk gradients. Cosine < 0 means the heads pull the trunk in
  conflicting directions (the regime where schemes like PCGrad help); ~0 = orthogonal;
  > 0 = aligned (positive transfer).

Enable via the parameter file (off by default; leave off for sweeps unless you are
studying head interaction — the extra backward passes slow training)::

    "dual_analysis": {
        "enabled": true,
        "grad_norm_every": 50    // measure trunk grads every N training batches
    }

Once the bg head is frozen (``bg_head.freeze_patience``), its trunk-gradient norm is
reported as 0 and the cosine as NaN: the frozen head genuinely contributes no gradient.
"""

import math

import torch


def trunk_parameters(model):
    """Parameters of the shared trunk (encoder + processor) that require grad."""
    return [p for module in (model.encoder, model.processor)
            for p in module.parameters() if p.requires_grad]


def _global_norm(grads):
    """Global L2 norm over a list of per-parameter gradient tensors (None entries skipped)."""
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.pow(2).sum().item())
    return math.sqrt(total)


def _cosine(grads_a, grads_b):
    dot = 0.0
    for a, b in zip(grads_a, grads_b):
        if a is not None and b is not None:
            dot += float((a * b).sum().item())
    denom = _global_norm(grads_a) * _global_norm(grads_b)
    return dot / denom if denom > 0 else float("nan")


def _safe_ratio(num, den):
    return num / den if den > 0 else float("nan")


def loss_ratio_metrics(avg_train_fp, avg_train_bg, avg_test_fp, avg_test_bg, bg_loss_weight):
    """Loss-scale diagnostics from the epoch's already-computed average losses.

    Returns a dict of ``analysis/*`` metrics: the raw fp/bg loss ratios (train and test)
    and the train-time contribution ratio ``(1-w)·fp / (w·bg)`` — i.e. how the joint loss
    actually splits between the heads. NaN where a denominator is zero (e.g. ``w`` = 0).
    """
    w = bg_loss_weight
    return {
        "analysis/train_loss_ratio_fp_over_bg": _safe_ratio(avg_train_fp, avg_train_bg),
        "analysis/test_loss_ratio_fp_over_bg": _safe_ratio(avg_test_fp, avg_test_bg),
        "analysis/train_loss_contribution_fp_over_bg": _safe_ratio((1 - w) * avg_train_fp,
                                                                   w * avg_train_bg),
    }


class TrunkGradAnalyser:
    """Accumulates per-head trunk-gradient diagnostics over one epoch. ANALYSIS ONLY.

    Usage in the training loop, on each measured batch and BEFORE ``loss.backward()``
    (the losses' graph is needed; it is retained, so the training backward is
    unaffected and ``param.grad`` is never touched)::

        if analyser.should_measure(batch_idx):
            analyser.measure(fp_loss, bg_loss, bg_loss_weight, bg_frozen=...)

    and once per epoch::

        metrics = analyser.epoch_means()   # {} if nothing was measured
        analyser.reset()
    """

    def __init__(self, model, every=50):
        """
        Args:
            model (GraphSatelliteDualForecaster): The dual model (for its trunk params).
            every (int): Measure on every ``every``-th training batch (batch 0 included,
                so at least one measurement per epoch).
        """
        self.trunk_params = trunk_parameters(model)
        self.every = max(1, int(every))
        self.reset()

    def reset(self):
        self._sum_fp = 0.0
        self._sum_bg = 0.0
        self._sum_ratio = 0.0
        self._sum_cos = 0.0
        self._n = 0
        self._n_ratio = 0
        self._n_cos = 0

    def should_measure(self, batch_idx):
        return batch_idx % self.every == 0

    def measure(self, fp_loss, bg_loss, bg_loss_weight, bg_frozen=False):
        """Measure each head's gradient norm on the shared trunk for the current batch.

        With ``bg_frozen=True`` the bg loss carries no graph (its predictions are
        detached in the training loop), so its trunk contribution is recorded as 0 and
        no cosine is accumulated.
        """
        grads_fp = torch.autograd.grad(fp_loss, self.trunk_params,
                                       retain_graph=True, allow_unused=True)
        fp_norm = _global_norm(grads_fp)
        if bg_frozen:
            bg_norm = 0.0
        else:
            grads_bg = torch.autograd.grad(bg_loss, self.trunk_params,
                                           retain_graph=True, allow_unused=True)
            bg_norm = _global_norm(grads_bg)
            cos = _cosine(grads_fp, grads_bg)
            if math.isfinite(cos):
                self._sum_cos += cos
                self._n_cos += 1

        self._sum_fp += fp_norm
        self._sum_bg += bg_norm
        self._n += 1
        ratio = _safe_ratio((1 - bg_loss_weight) * fp_norm, bg_loss_weight * bg_norm)
        if math.isfinite(ratio):
            self._sum_ratio += ratio
            self._n_ratio += 1

    def epoch_means(self):
        """Mean diagnostics over the epoch's measured batches ({} if none)."""
        if self._n == 0:
            return {}
        return {
            "analysis/trunk_grad_norm_fp": self._sum_fp / self._n,
            "analysis/trunk_grad_norm_bg": self._sum_bg / self._n,
            "analysis/trunk_grad_contribution_fp_over_bg":
                (self._sum_ratio / self._n_ratio) if self._n_ratio else float("nan"),
            "analysis/trunk_grad_cosine":
                (self._sum_cos / self._n_cos) if self._n_cos else float("nan"),
        }
