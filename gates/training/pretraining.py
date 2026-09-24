"""Self-supervised pretraining of the GATES trunk (encoder + processor) on meteorology alone.

Pretext tasks, all defined on exactly the input tensors the dual model sees
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
- ``"delta_forecast"`` the general time-shift task, of which ``next_delta`` is the special case
                   ``source_deltas=[6, 12], target_delta=0``: every dynamic channel whose
                   time delta is NOT in ``source_deltas`` is zeroed in the input and the channels
                   at ``target_delta`` are predicted. Time deltas are hours BEFORE the footprint
                   time, so a *larger* target delta than the sources is a BACKWARD forecast
                   (where did this air come from?) — the direction a footprint describes.
                   Recipes: ``[0, 6] -> 12`` backward 6 h, ``[0] -> 12`` backward 12 h,
                   ``[12] -> 0`` forward 12 h, ``[0, 6, 12] -> 24`` backward 24 h. A target
                   delta outside the dual model's input layout (e.g. 24 h) is loaded as extra
                   TARGET-ONLY channels appended after the model's input columns
                   (``pretrain.extra_time_deltas`` in ``train_met_pretrain.py``): the trunk
                   still sees exactly the dual model's inputs. ``tendency: true`` predicts the
                   change relative to the nearest source delta instead (zero = persistence).
- ``"pseudo_footprint"`` the footprint-shaped task: nothing is hidden, the model sees exactly the
                   dual input and predicts ONE channel per node — a kinematic pseudo-footprint
                   (backward residence-time map of the air arriving in the release cell, built from
                   the window's boundary-layer winds by ``gates/training/pseudo_footprint.py`` and
                   scaled like the real footprints). The target is external to the input tensor
                   (``corrupt(x, generator, y)``); the trivial baseline is the climatological mean
                   map of the training samples. Because the head is a one-channel per-node map,
                   the footprint decoder can be saved and transferred along with the trunk
                   (``save_modules``).

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
# Modules a pretraining run may save; the fp decoder only when its output is one channel per node.
SAVEABLE_MODULES = TRUNK_MODULES + ("fp_decoder",)
PRETEXT_TASKS = ("masked", "next_delta", "next_delta_tendency", "delta_forecast", "pseudo_footprint")
EXTERNAL_TARGET_TASKS = ("pseudo_footprint",)
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
    deltas = sorted({int(names[i][2]) for i in dyn})
    older = [d for d in deltas if d > 0]
    if not older:
        raise ValueError("next_delta pretraining needs at least one time_delta > 0")
    # next_delta == the forecast [older deltas] -> 0 (persistence from the youngest older delta)
    fc = forecast_groups(names, dynamic_variables, source_deltas=older, target_delta=0)
    return {
        "dynamic": dyn, "dynamic_targets": dyn_t, "delta0": fc["hide"], "delta0_targets": fc["targets"],
        "delta0_persistence_source": fc["persistence_source"], "deltas": deltas,
        "names": ["|".join(str(x) for x in v) for v in names],
    }


def forecast_groups(variable_names, dynamic_variables, source_deltas, target_delta,
                    tendency=False, model_channels=None):
    """Index groups for the general ``delta_forecast`` task.

    Args:
        variable_names: as in :func:`channel_groups`, in TENSOR channel order. Target-only
            channels (time deltas outside the model's input layout) may be listed after the
            model's input columns; use ``model_channels`` to say where the model input ends.
        dynamic_variables: names of the time-varying met variables.
        source_deltas (sequence of int): time deltas (hours before the footprint time) the
            model may see; every other dynamic channel inside the model input is zeroed.
        target_delta (int): time delta of the channels to predict.
        tendency (bool): predict ``x(target) - x(persistence source)`` instead of ``x(target)``.
        model_channels (int or None): number of leading tensor columns the model receives;
            None = all columns (no target-only channels).

    Returns:
        dict: ``hide`` (dynamic channels zeroed in the model input), ``targets`` (channels at
        ``target_delta``, minus ``NON_TARGET_DYNAMIC``), ``persistence_source`` (for each
        target, the same (variable, level) at the source delta closest in time to the target
        = the persistence baseline), ``persistence_delta``, and the resolved
        ``source_deltas`` / ``target_delta`` / ``tendency`` for the record.
    """
    dynamic_variables = set(dynamic_variables)
    names = [tuple(v) for v in variable_names]
    dyn = [i for i, v in enumerate(names) if v[0] in dynamic_variables]
    present = sorted({int(names[i][2]) for i in dyn})
    n_model = len(names) if model_channels is None else int(model_channels)
    input_deltas = sorted({int(names[i][2]) for i in dyn if i < n_model})
    source_deltas = sorted({int(d) for d in source_deltas})
    target_delta = int(target_delta)
    if not source_deltas:
        raise ValueError("delta_forecast needs at least one source time delta")
    if any(d not in input_deltas for d in source_deltas):
        raise ValueError(f"source_deltas {source_deltas} must be among the model's input time deltas {input_deltas}")
    if target_delta not in present:
        raise ValueError(f"target_delta {target_delta} is not among the loaded time deltas {present}")
    if target_delta in source_deltas:
        raise ValueError(f"target_delta {target_delta} must not be one of source_deltas {source_deltas}")
    hide = [i for i in dyn if i < n_model and int(names[i][2]) not in source_deltas]
    targets = [i for i in dyn if int(names[i][2]) == target_delta and names[i][0] not in NON_TARGET_DYNAMIC]
    if not targets:
        raise ValueError(f"no target channels at time delta {target_delta}")
    persistence_delta = min(source_deltas, key=lambda d: (abs(d - target_delta), d))
    lookup = {(names[i][0], names[i][1], int(names[i][2])): i for i in dyn}
    persistence = [lookup[(names[i][0], names[i][1], persistence_delta)] for i in targets]
    return {"hide": hide, "targets": targets, "persistence_source": persistence,
            "persistence_delta": persistence_delta, "source_deltas": source_deltas,
            "target_delta": target_delta, "tendency": bool(tendency)}


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
    """Builds (corrupted inputs, targets, loss mask) for one pretext task from a clean batch.

    ``model_channels`` is the number of leading tensor columns the model receives (met + aux);
    columns after it are target-only (see :func:`forecast_groups`) and are stripped from the
    corrupted input. ``forecast`` (from :func:`forecast_groups`) is required for
    ``"delta_forecast"`` and ignored otherwise.
    """

    def __init__(self, task, groups, height, width, block_size=10, mask_ratio=0.5,
                 forecast=None, model_channels=None, baseline_map=None):
        if task not in PRETEXT_TASKS:
            raise ValueError(f"unknown pretext task {task!r}; allowed: {PRETEXT_TASKS}")
        self.task = task
        self.height, self.width = height, width
        self.block_size, self.mask_ratio = block_size, mask_ratio
        self.model_channels = model_channels
        self.tendency = task == "next_delta_tendency"
        self.persistence_baseline = task == "next_delta"
        self.forecast = None
        self.external_target = task in EXTERNAL_TARGET_TASKS
        self.baseline_map = None
        if self.external_target:
            hide, targets, persistence = [], [], []
            if baseline_map is None:
                raise ValueError(f"{task} needs baseline_map (the mean target map over the training samples)")
            self.baseline_map = torch.as_tensor(baseline_map, dtype=torch.float32).reshape(-1)
        elif task == "masked":
            hide, targets = groups["dynamic"], groups["dynamic_targets"]
            persistence = groups["delta0_persistence_source"]
        elif task == "delta_forecast":
            if forecast is None:
                raise ValueError("delta_forecast needs the groups from forecast_groups()")
            hide, targets, persistence = forecast["hide"], forecast["targets"], forecast["persistence_source"]
            self.tendency = bool(forecast.get("tendency", False))
            self.persistence_baseline = not self.tendency
            self.forecast = forecast
        else:
            hide, targets = groups["delta0"], groups["delta0_targets"]
            persistence = groups["delta0_persistence_source"]
        if model_channels is not None and hide and max(hide) >= model_channels:
            raise ValueError("hidden channels must lie inside the model's input columns")
        self.hide_idx = torch.tensor(hide, dtype=torch.long)        # may be empty (nothing hidden)
        self.target_idx = torch.tensor(targets, dtype=torch.long)
        self.persistence_idx = torch.tensor(persistence, dtype=torch.long)
        self.num_targets = 1 if self.external_target else len(self.target_idx)

    def describe(self):
        """JSON-able summary of what the task hides / predicts (for settings and trunk meta)."""
        d = {"task": self.task, "tendency": self.tendency, "num_targets": self.num_targets,
             "num_hidden_channels": int(self.hide_idx.numel()), "external_target": self.external_target}
        if self.forecast is not None:
            d.update({k: self.forecast[k] for k in ("source_deltas", "target_delta", "persistence_delta")})
        return d

    def to(self, device):
        self.hide_idx = self.hide_idx.to(device)
        self.target_idx = self.target_idx.to(device)
        self.persistence_idx = self.persistence_idx.to(device)
        if self.baseline_map is not None:
            self.baseline_map = self.baseline_map.to(device)
        return self

    def corrupt(self, x, generator, y=None):
        """x: clean ``[B, N, F]`` (all tensor columns); ``y``: external target ``[B, N]`` (only for
        :data:`EXTERNAL_TARGET_TASKS`). Returns ``(x_corrupt [B,N,model_channels], target [B,N,T],
        cell_mask [B,N] bool)``."""
        if self.external_target:
            if y is None:
                raise ValueError(f"{self.task} needs the external target y")
            target = y.reshape(x.shape[0], x.shape[1], 1)
        else:
            target = x[:, :, self.target_idx]
        if self.tendency:
            target = target - x[:, :, self.persistence_idx]
        if self.task == "masked":
            cell_mask = block_mask(x.shape[0], self.height, self.width, self.block_size,
                                   self.mask_ratio, generator, x.device)
        else:
            cell_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x_corrupt = x.clone() if self.model_channels is None else x[:, :, :self.model_channels].clone()
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
        """Trivial reference prediction from the CLEAN batch: persistence of the source time
        delta nearest to the target for the plain forecast tasks; the scaled mean (0) for
        ``masked`` and for the tendency variants (where 0 IS persistence)."""
        if self.external_target:                      # climatological mean map
            return self.baseline_map.view(1, -1, 1).expand(x.shape[0], -1, 1)
        if self.persistence_baseline:
            return x[:, :, self.persistence_idx]
        return torch.zeros_like(x[:, :, self.target_idx])


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


def save_trunk(model, path, meta, modules=TRUNK_MODULES):
    """Save the state dicts of ``modules`` (default: the trunk = encoder + processor; the fp decoder
    may be added when its output is one channel per node) plus a ``meta`` record."""
    unknown = set(modules) - set(SAVEABLE_MODULES)
    if unknown:
        raise ValueError(f"save_trunk: cannot save {sorted(unknown)}; allowed {SAVEABLE_MODULES}")
    payload = {name: getattr(model, name).state_dict() for name in modules}
    payload["meta"] = dict(meta, saved_modules=list(modules))
    torch.save(payload, path)


def load_pretrained_trunk(model, parameters):
    """Initialise ``model``'s trunk from a :func:`save_trunk` file if the parameter file asks
    for it::

        "pretrained_trunk": {"path": "/abs/or/model_runs-relative/trunk_best.pt",
                             "modules": ["encoder", "processor"]}   # default: the trunk;
                             # add "fp_decoder" for a checkpoint that saved it (pseudo_footprint)

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
    unknown = set(modules) - set(SAVEABLE_MODULES)
    if unknown:
        raise ValueError(f"pretrained_trunk.modules: unknown {sorted(unknown)}; allowed {SAVEABLE_MODULES}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [m for m in modules if m not in payload]
    if missing:
        raise KeyError(f"pretrained_trunk.modules {missing} were not saved in {path} "
                       f"(saved: {[k for k in payload if k != 'meta']})")
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
          f"(task={meta.get('task')}, pretext={meta.get('pretext')}, years={meta.get('train_years')}, "
          f"epoch={meta.get('epoch')}, val_loss={meta.get('val_loss')})")
    return meta
