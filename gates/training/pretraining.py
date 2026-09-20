"""Self-supervised pretraining of the GATES trunk (encoder + processor) on meteorology alone.

Two pretext tasks, both defined on exactly the input tensors the dual model sees
(``[B, num_nodes, feature_dim + aux_dim]``, scaled like the dual pipeline), so the
pretrained trunk can be loaded into ``GraphSatelliteDualForecaster`` unchanged:

- ``"masked"``     masked reconstruction: square blocks of the lat/lon window have ALL their
                   dynamic met channels (every variable, level and time delta) set to 0
                   (= the scaled mean) and the model reconstructs them; the loss is the MSE
                   over the masked cells only.
- ``"next_delta"`` next-time-delta prediction: the most recent met (``time_delta == 0``) is
                   zeroed everywhere and predicted from the older time deltas (t-6 h, t-12 h)
                   — a 6 h forecast of the window; the loss is the MSE over all cells.
- ``"next_delta_tendency"`` same inputs, but the target is the CHANGE ``x(t0) - x(t-6h)`` per
                   channel, so persistence is the zero prediction and the trunk must learn
                   tendencies instead of re-learning to copy (the plain task was worse than
                   persistence for the smooth fields: pressure, temperature). Its loss is
                   directly comparable with the plain task's persistence baseline.

Static channels (coordinates, topography) and the auxiliary CAMS channels are never masked
and never targets. ``wind_angle`` is masked with the other dynamic channels but is NOT a
target: it is circular, so an MSE on it is ill-defined at the wrap-around.

The pretraining network is a ``GraphSatelliteDualForecaster`` whose footprint decoder is
widened to one output per target channel (its bg decoder is built but unused), which
guarantees the trunk is constructed by the same code path as the dual model. Only
``encoder`` + ``processor`` weights are saved / transferred (see :func:`save_trunk`,
:func:`load_pretrained_trunk`).
"""

import copy
from pathlib import Path

import numpy as np
import torch

TRUNK_MODULES = ("encoder", "processor")
PRETEXT_TASKS = ("masked", "next_delta", "next_delta_tendency")
# Variables that are never masked / never targets (everything else with a met name is dynamic).
NON_TARGET_DYNAMIC = ("wind_angle",)
# Mesh-size-dependent, never-trained tensors (see load_pretrained_trunk).
MESH_PLACEHOLDER_KEYS = ("h3_nodes",)


def channel_groups(variable_names, dynamic_variables):
    """Split the input channels into the index groups the pretext tasks need.

    Args:
        variable_names (sequence of tuple): ``(variable, levels, time_delta)`` per input
            channel, in channel order (the ``variable_name`` MultiIndex of the scaled inputs,
            BEFORE the aux CAMS channels are appended — aux channels are simply not listed
            in any group).
        dynamic_variables (sequence of str): names of the time-varying met variables
            (including derived ones such as ``wind_speed`` / ``wind_angle``).

    Returns:
        dict with integer index lists:
            ``dynamic``         every dynamic met channel (what "masked" hides),
            ``dynamic_targets`` dynamic channels used as reconstruction targets,
            ``delta0``          dynamic channels at time_delta 0 (what "next_delta" hides),
            ``delta0_targets``  delta-0 channels used as targets,
            ``delta0_persistence_source``  for each entry of ``delta0_targets``, the index of
                                the same (variable, level) at the smallest non-zero time delta
                                (the persistence baseline),
        plus ``names`` (list of "var|level|delta" strings for the record).
    """
    dynamic_variables = set(dynamic_variables)
    names = [tuple(v) for v in variable_names]
    dyn = [i for i, v in enumerate(names) if v[0] in dynamic_variables]
    if not dyn:
        raise ValueError("no dynamic met channels found in variable_names")
    dyn_t = [i for i in dyn if names[i][0] not in NON_TARGET_DYNAMIC]
    d0 = [i for i in dyn if int(names[i][2]) == 0]
    d0_t = [i for i in d0 if names[i][0] not in NON_TARGET_DYNAMIC]
    older = sorted({int(names[i][2]) for i in dyn if int(names[i][2]) > 0})
    if not older:
        raise ValueError("next_delta pretraining needs at least one time_delta > 0")
    lookup = {(names[i][0], names[i][1], int(names[i][2])): i for i in dyn}
    persistence = [lookup[(names[i][0], names[i][1], older[0])] for i in d0_t]
    return {
        "dynamic": dyn, "dynamic_targets": dyn_t, "delta0": d0, "delta0_targets": d0_t,
        "delta0_persistence_source": persistence,
        "names": ["|".join(str(x) for x in v) for v in names],
    }


def block_mask(batch_size, height, width, block_size, mask_ratio, generator, device):
    """Boolean ``[B, H*W]`` mask (True = masked) made of ``block_size`` x ``block_size`` blocks,
    each masked independently with probability ``mask_ratio``; every sample is guaranteed at
    least one masked block (so the loss is never empty)."""
    nby = -(-height // block_size)
    nbx = -(-width // block_size)
    blocks = torch.rand(batch_size, nby, nbx, generator=generator, device=device) < mask_ratio
    empty = ~blocks.flatten(1).any(dim=1)
    if empty.any():
        idx = torch.randint(0, nby * nbx, (int(empty.sum()),), generator=generator, device=device)
        flat = blocks.flatten(1)
        flat[empty.nonzero(as_tuple=True)[0], idx] = True
        blocks = flat.view(batch_size, nby, nbx)
    cells = blocks.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)
    return cells[:, :height, :width].reshape(batch_size, height * width)


class PretextTask:
    """Builds (corrupted inputs, targets, loss mask) for one pretext task from a clean batch."""

    def __init__(self, task, groups, height, width, block_size=10, mask_ratio=0.5):
        if task not in PRETEXT_TASKS:
            raise ValueError(f"unknown pretext task {task!r}; allowed: {PRETEXT_TASKS}")
        self.task = task
        self.height, self.width = height, width
        self.block_size, self.mask_ratio = block_size, mask_ratio
        self.tendency = task == "next_delta_tendency"
        if task == "masked":
            self.hide_idx = torch.tensor(groups["dynamic"])
            self.target_idx = torch.tensor(groups["dynamic_targets"])
        else:
            self.hide_idx = torch.tensor(groups["delta0"])
            self.target_idx = torch.tensor(groups["delta0_targets"])
        self.persistence_idx = torch.tensor(groups["delta0_persistence_source"])
        self.num_targets = len(self.target_idx)

    def to(self, device):
        self.hide_idx = self.hide_idx.to(device)
        self.target_idx = self.target_idx.to(device)
        self.persistence_idx = self.persistence_idx.to(device)
        return self

    def corrupt(self, x, generator):
        """x: clean ``[B, N, F]``. Returns ``(x_corrupt, target [B,N,T], cell_mask [B,N] bool)``."""
        target = x[:, :, self.target_idx]
        if self.tendency:
            target = target - x[:, :, self.persistence_idx]
        if self.task == "masked":
            cell_mask = block_mask(x.shape[0], self.height, self.width, self.block_size,
                                   self.mask_ratio, generator, x.device)
        else:
            cell_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x_corrupt = x.clone()
        hidden = x_corrupt[:, :, self.hide_idx]
        hidden[cell_mask] = 0.0
        x_corrupt[:, :, self.hide_idx] = hidden
        return x_corrupt, target, cell_mask

    @staticmethod
    def loss(pred, target, cell_mask):
        """MSE over the masked cells (all target channels)."""
        diff2 = (pred - target) ** 2
        return diff2[cell_mask].mean()

    def baseline_prediction(self, x):
        """Trivial reference prediction from the CLEAN batch: persistence of the youngest older
        time delta for ``next_delta``; the scaled mean (0) for ``masked``."""
        if self.task == "next_delta":
            return x[:, :, self.persistence_idx]
        return torch.zeros_like(x[:, :, self.target_idx])   # masked: mean; tendency: persistence


def build_pretrain_model(parameters, grid, feature_dim, aux_dim, size, num_targets):
    """A ``GraphSatelliteDualForecaster`` built exactly like ``setup_dual_model`` builds it,
    except that the footprint decoder outputs ``num_targets`` channels per node."""
    from model.forecast import GraphSatelliteDualForecaster

    model_parameters = copy.deepcopy(parameters["model_parameters"])
    decoder = model_parameters.pop("decoder", "conv")
    num_classes = model_parameters.pop("num_classes", 1)
    model_parameters["fp_output_dim"] = num_targets
    return GraphSatelliteDualForecaster(
        grid, whole_world=False, feature_dim=feature_dim, aux_dim=aux_dim,
        num_classes=num_classes, input_height=size, input_width=size,
        decoder_type=decoder, **model_parameters,
    )


def pretrain_forward(model, x):
    """Trunk + (widened) footprint decoder only; the bg decoder is not run."""
    h, edge_idx, edge_attr = model.encoder(x)
    h = model.processor(h, edge_idx, edge_attr, batch=model.encoder.batch_size)
    return model.fp_decoder(h, x)


def save_trunk(model, path, meta):
    """Save the trunk (encoder + processor) state dicts plus a ``meta`` record."""
    payload = {name: getattr(model, name).state_dict() for name in TRUNK_MODULES}
    payload["meta"] = meta
    torch.save(payload, path)


def load_pretrained_trunk(model, parameters):
    """Initialise ``model``'s trunk from a :func:`save_trunk` file if the parameter file asks
    for it::

        "pretrained_trunk": {"path": "/abs/or/model_runs-relative/trunk_best.pt",
                             "modules": ["encoder", "processor"]}   # default: both

    Loading is strict (every tensor of each listed module must match in name and shape), so
    a trunk pretrained with different inputs / model_parameters fails loudly. The only
    exception is the all-zero mesh placeholder ``h3_nodes`` (see ``MESH_PLACEHOLDER_KEYS``). No block (or a
    null path) leaves the model untouched. Returns the checkpoint's ``meta`` dict, or None.
    """
    cfg = parameters.get("pretrained_trunk") or {}
    path = cfg.get("path")
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"pretrained_trunk.path does not exist: {path}")
    modules = cfg.get("modules", list(TRUNK_MODULES))
    unknown = set(modules) - set(TRUNK_MODULES)
    if unknown:
        raise ValueError(f"pretrained_trunk.modules: unknown {sorted(unknown)}; allowed {TRUNK_MODULES}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name in modules:
        state = dict(payload[name])
        module = getattr(model, name)
        # ``h3_nodes`` is the encoder's all-zero mesh-node placeholder (one row per mesh node; it
        # is re-created every forward and never trained). Its row count depends on the reference
        # footprint the grid was built from, so it is the one tensor allowed to differ: keep the
        # model's own, after checking the checkpoint's really is the untrained placeholder.
        for key in MESH_PLACEHOLDER_KEYS:
            if key in state and state[key].shape != module.state_dict()[key].shape:
                if float(state[key].abs().max()) != 0.0:
                    raise RuntimeError(f"{name}.{key} differs in shape and is not all-zero in {path}")
                print(f"pretrained_trunk: keeping the model's own {name}.{key} "
                      f"{tuple(module.state_dict()[key].shape)} (checkpoint mesh: {tuple(state[key].shape)})")
                state[key] = module.state_dict()[key]
        module.load_state_dict(state, strict=True)
    meta = payload.get("meta", {})
    print(f"Loaded pretrained trunk modules {modules} from {path} "
          f"(task={meta.get('task')}, years={meta.get('train_years')}, "
          f"epoch={meta.get('epoch')}, val_loss={meta.get('val_loss')})")
    return meta
