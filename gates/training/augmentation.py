"""Training-time data augmentation for the dual-head trainers: the EAST-WEST MIRROR.

What it does
------------
A training sample is a window of meteorology centred on the release point, the footprint of that
release on the same window, and the background mole fraction. The mirror shows the model the same
physical situation reflected in the meridian through the release point: what was east of the
release is now west of it. Inputs AND targets are reflected together, so the pair stays a valid
(meteorology -> footprint) example:

* every map (met and static inputs, every footprint variable) is reflected along longitude;
* the components that point along the reflected axis change sign in PHYSICAL units: ``x_wind``
  (u -> -u) and ``wind_angle`` (``arctan2(-u, -v)`` -> minus itself). The inputs are already
  scaled when they reach the trainer, so the sign change is applied as ``offset - value`` with the
  offset taken from the fitted scaler of that channel (standard scaling: ``-2 mean / std``);
* ``x_coords`` / ``y_coords`` are left as they are: they describe the grid cell (its offset from
  the release point), not the world, and the mirrored world is shown on the same grid;
* the auxiliary boundary channels of the east and west boundary are exchanged; with four
  background classes the east and west targets are exchanged too. The summed background is
  unchanged: it is the same air reaching the same boundary values.

Geometry. The release point sits at index ``n_lon // 2`` of an even-sized window, so the window
has one more column to the west of it than to the east. The reflection maps column ``j`` to
``n_lon - j``: the release column stays in place, columns ``1 .. n_lon - 1`` swap in pairs and
column 0 has no partner inside the window. In a mirrored sample that column is filled with the
values of its neighbour (as the loader pads windows that leave the domain) and flagged in
``fp_nan_mask``, so the losses ignore it.

Assumptions (principle 1 - say what is uncertain)
-------------------------------------------------
* The reflection is exact for what the LPDM does with a given wind field on a regular
  longitude grid. It is NOT a symmetry of the atmosphere: the turning of the wind with height,
  the sense of rotation of weather systems and the position of mountains and coasts have a
  handedness. The model is given only three snapshots of the meteorology; whatever it infers
  about the unseen part from such regularities is contradicted by the mirrored samples.
  Whether the extra data outweighs that is an empirical question (experiment summary section 22).
* Absolute longitude has no meaning in a mirrored window: inputs that contain ``lon_coords`` or
  its sine / cosine are refused.
* The auxiliary boundary channels are the values at the MIDPOINT of each boundary. Exchanging
  east and west is exact; the north and south midpoints are kept although the reflected curtain
  would be sampled at the reflected midpoint.

Use
---
Parameter file (all trainers that do not apply it fail loudly, see ``require_resolved``)::

    "augmentation": {"mirror_ew": {"probability": 0.5}}

Every training sample is mirrored with this probability, drawn afresh every epoch from
``(seed, rank, epoch, batch index)`` alone, so a run is reproducible and the draws do not depend
on how many GPUs share the work of a rank. Test data are never mirrored. With probability 0.5
the model sees both orientations of every footprint over the epochs, at the memory, the epoch
time and the number of optimizer steps of the run without augmentation.

Everything the transform needs is resolved ONCE (``resolve_mirror_augmentation``) and written to
``parameters["augmentation_resolved"]``, so it is saved with the training settings and the
multi-GPU workers rebuild the identical transform.
"""

import numpy as np
import torch

MIRROR_KEY = "mirror_ew"

# variables whose physical value changes sign under the east-west reflection
SIGN_FLIP_VARIABLES = ("x_wind", "wind_angle")
# variables that describe the grid cell, not the world: never reflected
GRID_COORDINATE_VARIABLES = ("x_coords", "y_coords")
# absolute longitude: meaningless in a mirrored window
LONGITUDE_POSITION_VARIABLES = ("lon_coords", "sin_lon_coords", "cos_lon_coords")

NAN_MASK_LABEL = "fp_nan_mask"
_SEED_TAG = 0xEA57  # separates these draws from any other use of the run's seed


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

def mirror_config(parameters):
    """Validated ``augmentation.mirror_ew`` block, or None when the mirror is off.

    Raises ``ValueError`` on unknown keys or a probability outside [0, 1].
    """
    block = parameters.get("augmentation", None) or {}
    unknown = set(block) - {MIRROR_KEY}
    if unknown:
        raise ValueError(f"augmentation: unknown keys {sorted(unknown)}; allowed: ['{MIRROR_KEY}']")
    cfg = block.get(MIRROR_KEY, None)
    if not cfg:
        return None
    unknown = set(cfg) - {"probability"}
    if unknown:
        raise ValueError(f"augmentation.{MIRROR_KEY}: unknown keys {sorted(unknown)}; allowed: ['probability']")
    probability = cfg.get("probability", 0.5)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"augmentation.{MIRROR_KEY}.probability must be a number in [0, 1], got {probability!r}")
    if probability == 0:
        return None
    return {"probability": float(probability)}


def require_resolved(parameters):
    """Fail loudly when the mirror is requested but was never resolved.

    Called where the model is built (``setup_dual_model``), so a trainer that does not apply the
    augmentation cannot silently train without it.
    """
    if mirror_config(parameters) is not None and not (parameters.get("augmentation_resolved") or {}).get(MIRROR_KEY):
        raise ValueError(f"augmentation.{MIRROR_KEY} is set but parameters['augmentation_resolved'] is missing: "
                         "this trainer does not apply the augmentation (train_dual_headlr_model does)")


def aux_direction_layout(aux_dataset):
    """Order of the auxiliary boundary channels, as ``format_aux_data`` stacks them.

    Args:
        aux_dataset (xr.Dataset): auxiliary CAMS data BEFORE ``format_aux_data``: one variable per
            boundary, coordinate ``height_index``.

    Returns:
        dict: ``{"directions": [...], "n_heights": int}``; channel ``d * n_heights + h`` is
        boundary ``directions[d]`` at the ``h``-th height.
    """
    return {"directions": [str(v) for v in aux_dataset.data_vars],
            "n_heights": int(aux_dataset.sizes["height_index"])}


def _name(entry):
    """``(variable, level, time_delta)`` of an input channel with plain python types."""
    def plain(x):
        try:
            if float(x) == int(float(x)):
                return int(float(x))
            return float(x)
        except (TypeError, ValueError):
            return str(x)
    return (str(entry[0]),) + tuple(plain(x) for x in tuple(entry)[1:])


def _sign_flip_offset(scaler, channel):
    """``offset`` such that ``offset - scaled(x) == scaled(-x)`` for a fitted per-channel scaler."""
    kind = getattr(scaler, "scaler_type", None)
    if kind == "standard":
        # XarrayScaler.transform: (x - mean) / (std + 1e-8)
        return -2.0 * float(scaler.mean) / (float(scaler.std) + 1e-8)
    if kind == "minmax":
        # XarrayMinMaxScaler.transform: lo + (hi - lo) * (x - min) / (max - min + 1e-8)
        lo, hi = (float(v) for v in scaler.feature_range)
        return 2.0 * lo - 2.0 * (hi - lo) * float(scaler.min) / (float(scaler.max) - float(scaler.min) + 1e-8)
    if getattr(scaler, "scaler_name", None) == "GhostScaler":
        return 0.0
    raise ValueError(f"augmentation.{MIRROR_KEY}: cannot mirror channel {channel}: unsupported scaler "
                     f"{type(scaler).__name__}")


def resolve_mirror_augmentation(parameters, input_names, input_scaler, fp_labels, n_lat, n_lon,
                                aux_layout=None, num_classes=1):
    """Resolve the mirror into channel positions and constants; record them in ``parameters``.

    Args:
        parameters (dict): full parameter dict; ``parameters["augmentation_resolved"]`` is written
            (or removed when the mirror is off).
        input_names (iterable): ``(variable, level, time_delta)`` of the met / static input
            channels, in input order. The auxiliary boundary channels follow them in the model input.
        input_scaler: the FITTED ``DefaultInputsScaler`` of the run (``.scalers`` by channel).
        fp_labels (list[str]): variables along the last dimension of the footprint batches.
        n_lat, n_lon (int): window size. The batches are flattened latitude-major
            (node = ``lat_index * n_lon + lon_index``).
        aux_layout (dict or None): :func:`aux_direction_layout`, or None without auxiliary channels.
        num_classes (int): background classes (1 = summed; 4 = north, south, east, west).

    Returns:
        dict or None: the resolved block, or None when the mirror is off.
    """
    cfg = mirror_config(parameters)
    if cfg is None:
        if "augmentation_resolved" in parameters:
            parameters["augmentation_resolved"].pop(MIRROR_KEY, None)
            if not parameters["augmentation_resolved"]:
                parameters.pop("augmentation_resolved")
        return None

    n_lat, n_lon = int(n_lat), int(n_lon)
    if n_lon % 2 != 0:
        raise ValueError(f"augmentation.{MIRROR_KEY}: the window must have an even number of columns "
                         f"(release point at n_lon // 2), got {n_lon}")
    names = [_name(n) for n in input_names]
    variables = [n[0] for n in names]
    refused = sorted({v for v in variables if v in LONGITUDE_POSITION_VARIABLES})
    if refused:
        raise ValueError(f"augmentation.{MIRROR_KEY}: the inputs contain absolute longitude channels {refused}, "
                         "which have no meaning in a mirrored window; use inputs without the absolute "
                         "position channels (e.g. parameter_dual_*nolatlon*.json)")
    if NAN_MASK_LABEL not in list(fp_labels):
        raise ValueError(f"augmentation.{MIRROR_KEY} needs the '{NAN_MASK_LABEL}' footprint variable to flag the "
                         "column without a mirror partner (set dataloader.nans_to_zeros: true)")

    scalers = getattr(input_scaler, "scalers", None)
    if not isinstance(scalers, dict):
        raise ValueError(f"augmentation.{MIRROR_KEY} needs a fitted DefaultInputsScaler (per-channel scalers), "
                         f"got {type(input_scaler).__name__}")
    by_name = {_name(k): v for k, v in scalers.items()}

    negate, offsets = [], []
    for position, name in enumerate(names):
        if name[0] in SIGN_FLIP_VARIABLES:
            if name not in by_name:
                raise ValueError(f"augmentation.{MIRROR_KEY}: no fitted scaler for input channel {name}")
            negate.append(position)
            offsets.append(_sign_flip_offset(by_name[name], name))
    if "x_wind" in variables and "y_wind" not in variables:
        print(f"FLAG: augmentation.{MIRROR_KEY}: x_wind is an input but y_wind is not")
    keep = [p for p, n in enumerate(names) if n[0] in GRID_COORDINATE_VARIABLES]

    # expected scaled value of x_coords in the release column (x = 0), for the layout check
    release_x = None
    x_positions = [p for p, n in enumerate(names) if n[0] == "x_coords"]
    if x_positions and names[x_positions[0]] in by_name:
        sc = by_name[names[x_positions[0]]]
        if getattr(sc, "scaler_type", None) == "minmax":
            lo, hi = (float(v) for v in sc.feature_range)
            release_x = lo + (hi - lo) * (0.0 - float(sc.min)) / (float(sc.max) - float(sc.min) + 1e-8)

    n_inputs = len(names)
    aux_swap = []
    if aux_layout is not None:
        directions, n_heights = list(aux_layout["directions"]), int(aux_layout["n_heights"])
        if "east" not in directions or "west" not in directions:
            raise ValueError(f"augmentation.{MIRROR_KEY}: auxiliary boundary channels {directions} have no "
                             "'east' / 'west' pair to exchange")
        east, west = directions.index("east"), directions.index("west")
        aux_swap = [[n_inputs + east * n_heights + h, n_inputs + west * n_heights + h] for h in range(n_heights)]
        n_features = n_inputs + len(directions) * n_heights
    else:
        n_features = n_inputs

    if num_classes == 1:
        bg_swap = []
    elif num_classes == 4:
        bg_swap = [2, 3]  # classes are north, south, east, west
    else:
        raise ValueError(f"augmentation.{MIRROR_KEY}: num_classes must be 1 or 4, got {num_classes}")

    resolved = {
        "probability": cfg["probability"],
        "seed": int(parameters.get("seed", 34)),
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_features": n_features,
        "release_column": n_lon // 2,
        # mirrored column j takes the values of source column lon_source[j]
        "lon_source": [n_lon - 1] + [n_lon - j for j in range(1, n_lon)],
        "unpaired_columns": [0],
        "negate_channels": negate,
        "negate_offsets": offsets,
        "negate_names": [list(names[p]) for p in negate],
        "unmirrored_channels": keep,
        "x_coords_channel": x_positions[0] if x_positions else None,
        "x_coords_release_value": release_x,
        "aux_swap": aux_swap,
        "bg_swap": bg_swap,
        "nan_mask_index": list(fp_labels).index(NAN_MASK_LABEL),
        "n_fp_variables": len(list(fp_labels)),
    }
    parameters.setdefault("augmentation_resolved", {})[MIRROR_KEY] = resolved
    print(f"East-west mirror: probability {resolved['probability']:g}; {len(negate)} channels change sign "
          f"({sorted({n[0] for n in names if n[0] in SIGN_FLIP_VARIABLES})}), {len(keep)} grid-coordinate "
          f"channels kept, {len(aux_swap)} east/west auxiliary pairs exchanged, column "
          f"{resolved['unpaired_columns']} of mirrored samples masked in the loss")
    return resolved


# ---------------------------------------------------------------
# The transform
# ---------------------------------------------------------------

class EastWestMirror:
    """Applies the resolved east-west mirror to training batches (see the module docstring).

    ``mirror(features, fps, bgs, epoch, batch_index)`` returns NEW tensors; the inputs are never
    modified (the training batches live in shared memory and are reused every epoch).
    """

    def __init__(self, resolved, rank=0):
        self.cfg = dict(resolved)
        self.rank = int(rank)
        self.probability = float(resolved["probability"])
        self.seed = int(resolved["seed"])
        self.n_lat, self.n_lon = int(resolved["n_lat"]), int(resolved["n_lon"])
        n_features = int(resolved["n_features"])

        self._lon_source = torch.tensor(resolved["lon_source"], dtype=torch.long)
        sign = torch.ones(n_features, dtype=torch.float32)
        offset = torch.zeros(n_features, dtype=torch.float32)
        for channel, value in zip(resolved["negate_channels"], resolved["negate_offsets"]):
            sign[channel] = -1.0
            offset[channel] = value
        keep = torch.zeros(n_features, dtype=torch.bool)
        for channel in resolved["unmirrored_channels"]:
            keep[channel] = True
        order = torch.arange(n_features)
        for east, west in resolved["aux_swap"]:
            order[east], order[west] = west, east
        self._sign, self._offset, self._keep, self._channel_order = sign, offset, keep, order

        n_fp = int(resolved["n_fp_variables"])
        unpaired = torch.zeros(self.n_lon, dtype=torch.bool)
        for column in resolved["unpaired_columns"]:
            unpaired[column] = True
        is_mask = torch.zeros(n_fp, dtype=torch.bool)
        is_mask[resolved["nan_mask_index"]] = True
        self._flag = unpaired[None, None, :, None] & is_mask[None, None, None, :]
        self._bg_swap = list(resolved["bg_swap"])

        self._device = None
        self._checked = False
        self.n_seen = 0
        self.n_mirrored = 0

    # -- bookkeeping ------------------------------------------------------------------------

    def start_epoch(self):
        self.n_seen = 0
        self.n_mirrored = 0

    @property
    def fraction(self):
        """Share of this rank's samples mirrored since ``start_epoch``."""
        return self.n_mirrored / self.n_seen if self.n_seen else 0.0

    def _to(self, device):
        if self._device != device:
            for name in ("_lon_source", "_sign", "_offset", "_keep", "_channel_order", "_flag"):
                setattr(self, name, getattr(self, name).to(device))
            self._device = device

    # -- which samples ----------------------------------------------------------------------

    def sample_mask(self, epoch, batch_index, batch_size):
        """Which samples of a batch are mirrored: a function of (seed, rank, epoch, batch) only."""
        sequence = np.random.SeedSequence([self.seed, _SEED_TAG, self.rank, int(epoch), int(batch_index)])
        return np.random.default_rng(sequence).random(int(batch_size)) < self.probability

    # -- checks -----------------------------------------------------------------------------

    def check_layout(self, features):
        """Verify on real inputs that longitude is the axis being reflected.

        ``x_coords`` (the offset from the release point along longitude) must be the same in every
        row, increase along the columns and be zero in the release column.
        """
        channel = self.cfg.get("x_coords_channel")
        if channel is None:
            print(f"FLAG: augmentation.{MIRROR_KEY}: no x_coords input, the orientation of the batches "
                  "cannot be checked")
            return
        x = features[0].reshape(self.n_lat, self.n_lon, -1)[..., channel].detach().float().cpu()
        if not bool(torch.all(x == x[:1])) or not bool(torch.all(x[:, 1:] > x[:, :-1])):
            raise ValueError(f"augmentation.{MIRROR_KEY}: x_coords does not increase along the second axis of the "
                             "window: the batches are not flattened latitude-major, the mirror would "
                             "reflect the wrong axis")
        expected = self.cfg.get("x_coords_release_value")
        column = int(self.cfg["release_column"])
        if expected is not None and abs(float(x[0, column]) - float(expected)) > 1e-4:
            raise ValueError(f"augmentation.{MIRROR_KEY}: the release point is not in column {column} "
                             f"(scaled x_coords there is {float(x[0, column]):.5f}, expected {float(expected):.5f})")

    # -- the reflection ---------------------------------------------------------------------

    def mirror_inputs(self, features):
        """Reflect every sample of ``features`` (B, nodes, features)."""
        b, n, f = features.shape
        x = features.reshape(b, self.n_lat, self.n_lon, f)
        m = x.index_select(2, self._lon_source).index_select(3, self._channel_order)
        m = m * self._sign.to(m.dtype) + self._offset.to(m.dtype)
        m = torch.where(self._keep, x, m)
        return m.reshape(b, n, f)

    def mirror_footprints(self, fps):
        """Reflect every sample of ``fps`` (B, nodes, variables) and flag the unpaired column."""
        b, n, v = fps.shape
        m = fps.reshape(b, self.n_lat, self.n_lon, v).index_select(2, self._lon_source)
        m = torch.where(self._flag, torch.ones_like(m), m)
        return m.reshape(b, n, v)

    def mirror_backgrounds(self, bgs):
        if not self._bg_swap:
            return bgs
        order = torch.arange(bgs.shape[-1], device=bgs.device)
        east, west = self._bg_swap
        order[east], order[west] = west, east
        return bgs.index_select(-1, order)

    def mirror(self, features, fps, bgs, epoch, batch_index):
        """Mirror a random part of the batch. Returns ``(features, fps, bgs, n_mirrored)``."""
        self._to(features.device)
        if not self._checked:
            self.check_layout(features)
            self._checked = True
        chosen = self.sample_mask(epoch, batch_index, features.shape[0])
        self.n_seen += len(chosen)
        n_mirrored = int(chosen.sum())
        self.n_mirrored += n_mirrored
        if n_mirrored == 0:
            return features, fps, bgs, 0
        chosen = torch.from_numpy(chosen).to(features.device)
        features = torch.where(chosen[:, None, None], self.mirror_inputs(features), features)
        fps = torch.where(chosen[:, None, None], self.mirror_footprints(fps), fps)
        if self._bg_swap:
            bgs = torch.where(chosen[:, None], self.mirror_backgrounds(bgs), bgs)
        return features, fps, bgs, n_mirrored


def build_mirror(parameters, rank=0):
    """The :class:`EastWestMirror` of a run, or None when the mirror is off."""
    if mirror_config(parameters) is None:
        return None
    require_resolved(parameters)
    return EastWestMirror(parameters["augmentation_resolved"][MIRROR_KEY], rank=rank)
