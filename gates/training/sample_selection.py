"""Which footprints the model trains on: selection of the TRAINING samples from the loaded pool.

Why
---
Every sample is one GOSAT sounding. Consecutive soundings are seconds and ~140 km apart (the same
synoptic scene shifted by a few window cells), so a year of ~17,000 BRAZIL soundings holds only
~650 overpasses, and the regular ``freq`` thinning keeps every overpass whatever its value
(experiment summary section 33: freq 1 = no lower floor). Clear-sky sampling makes the wet season
two to three times rarer than the dry season, and the models' error sits in the rare regimes (wet
season, westerly transport, broad footprints; section 41). This module picks a subset of a given
size from the loaded pool, with these rules:

* ``"random"``   a seeded uniform draw: the null hypothesis ("which samples" does not matter);
* ``"stride"``   evenly spaced samples of the time-sorted pool: the regular thinning at a budget;
* ``"month"``    the calendar months equally represented (experiment summary section 54): the budget
  is split over the twelve calendar months in proportion to ``n_month ** temper`` (default 0 = the
  same number from every month, 1 = the pool's own month shares) and, inside a month, the samples
  (of all years) are taken evenly spaced in time. A month with fewer samples than its share is
  taken whole and the rest is shared out among the others;
* ``"direction"`` the transport directions equally represented (experiment summary section 56):
  like ``"month"``, with nine groups instead of twelve - the compass sector (N, NE, ... NW) in
  which the centre of mass of the footprint lies as seen from the release point, i.e. where the
  air came from, and "near release" for a centre of mass within two cells. This is the grouping
  of ``scripts/worst_case_error.py``, in which the persistently worst footprints of sections 53
  and 54 are over-represented (SW / W / NW);
* ``"coverage"`` tempered inverse-density coverage ("purposeful" sampling). Every sample is
  described by region-free numbers (footprint: log10 mass, far-field share, centroid offset,
  extent; season: sin / cos of the day of year; position: release latitude / longitude), the
  descriptors are standardised and clustered (k-means), and the budget is allocated to the
  clusters in proportion to ``n_cluster ** temper`` (1 = proportional = the pool's own mix,
  0 = the same number from every cluster, 0.5 = the default compromise). Inside a cluster the
  samples are taken evenly spaced in time, so the near-duplicates of one overpass fill the budget
  last. Rare regimes get more than their share of the budget, common ones less;
* ``"spacing"``  a minimum DISTANCE between the soundings of one satellite overpass, with a spacing
  of its own for releases over land and over the sea (experiment summary sections 58 and 59).
  Over the sea GOSAT follows the sun-glint track and samples densely (nearest sounding of the
  same overpass: 29 km, against 96 km on land), and the footprints of neighbours inside an
  overpass are near-duplicates (copying the neighbour beats the trained models out to ~150 km
  over the sea, ~75 km on land). The regular ``freq`` thinning cannot remove them: it counts
  soundings, not kilometres. Here every overpass (soundings separated by at most
  ``overpass_gap_minutes``) is walked in time order, land and sea releases separately, and a
  sounding is kept if it is at least ``land_km`` / ``sea_km`` from every sounding already kept
  in that overpass: the first sounding of every overpass is always kept, so no overpass is lost.
  A budget (``n_samples`` / ``fraction``) is optional: without one every survivor is used, with
  one the survivors are thinned evenly in time to the budget. Land or sea = the land mask
  (``land_binary_mask`` of config.yml's ``landcover_datadir``, or ``land_mask_file``) at the
  centre of the release cell of the footprint window.

The selection uses only the SMALL cached data (footprints, times, release points) and runs before
any input array is read, so the training tensor is allocated for the selected samples only. The
test set is never touched. Everything is deterministic given the pool and ``seed`` (own NumPy
Generator: the global RNG, which decides the batch shuffle and the scaler-fit subsample, is not
used), and the selection is recorded in ``parameters["sample_selection_resolved"]`` (settings
file) and in ``sample_selection_<model>.csv`` next to the training settings.

Parameter file (TOP level, not inside ``train_load_data``, so that the month cache of the pool is
reused and the arms of one job can differ in their selection)::

    "sample_selection": {"mode": "coverage", "n_samples": 2048, "seed": 1,
                         "clusters": 64, "temper": 0.5,
                         "descriptors": ["footprint", "season", "position"], "far_radius": 15}

    "sample_selection": {"mode": "month", "n_samples": 8192}                  (equal months)
    "sample_selection": {"mode": "stride", "n_samples": 8192, "offset": 0.33}  (another draw)
    "sample_selection": {"mode": "spacing", "land_km": 100, "sea_km": 300, "n_samples": 50048}

``n_samples`` (or ``fraction`` of the pool) should be a multiple of the batch size: the batches
drop the trailing samples. ``offset`` (0 <= offset < 1, default 0; stride / month / coverage)
moves the evenly spaced picks by that fraction of their spacing: the same rule on other, (almost)
disjoint, samples = a replicate of a deterministic rule. Month cache only
(``data_cache.enabled``); the standard pipeline refuses the block
(``lean_dual_data.require_data_cache_for_sources``).
"""

import hashlib

import numpy as np
import pandas as pd

MODES = ("random", "stride", "coverage", "month", "direction", "spacing")
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DIRECTION_GROUPS = SECTORS + ("near release",)     # group index 0..8 of the direction mode
NEAR_RELEASE_CELLS = 2.0                           # centre of mass within this distance = "near release"
DESCRIPTORS = ("footprint", "season", "position")
# keys of the spacing mode only (refused for the other modes, which would ignore them silently)
_SPACING_KEYS = {"land_km", "sea_km", "overpass_gap_minutes", "land_mask_file"}
_ALLOWED_KEYS = {"mode", "n_samples", "fraction", "seed", "clusters", "temper", "descriptors",
                 "far_radius", "kmeans_iterations", "offset"} | _SPACING_KEYS
SURFACE_GROUPS = ("land release", "sea release")   # group index 0 / 1 of the spacing mode
EARTH_RADIUS_KM = 6371.0
LAND_MASK_VARIABLE = "land_binary_mask"            # variable of the land-cover file (1 = land, 0 = sea)
FOOTPRINT_NAMES = ["log10_mass", "far_share", "centroid_dx", "centroid_dy", "extent"]
_BLOCK = 4096   # samples per block of the footprint statistics (bounds the temporaries)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

def selection_config(parameters):
    """Validated ``parameters["sample_selection"]`` with the defaults filled in, or None when absent.

    Raises ``ValueError`` on unknown keys, an unknown mode, a missing or double budget, or values
    outside their range.
    """
    block = parameters.get("sample_selection", None)
    if not block:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"sample_selection must be a dict, got {type(block).__name__}")
    unknown = set(block) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"sample_selection: unknown keys {sorted(unknown)}; allowed: {sorted(_ALLOWED_KEYS)}")
    mode = block.get("mode", None)
    if mode not in MODES:
        raise ValueError(f"sample_selection.mode must be one of {MODES}, got {mode!r}")
    if mode == "spacing":
        # the budget is optional: without one, every sounding that survives the thinning is used
        if "n_samples" in block and "fraction" in block:
            raise ValueError("sample_selection: give at most one of n_samples (int) or fraction (0 < f <= 1)")
    elif ("n_samples" in block) == ("fraction" in block):
        raise ValueError("sample_selection needs exactly one of n_samples (int) or fraction (0 < f <= 1)")
    stray = sorted(set(block) & _SPACING_KEYS) if mode != "spacing" else []
    if stray:
        raise ValueError(f"sample_selection: {stray} belong to the spacing mode, not to {mode!r}")
    n_samples = block.get("n_samples", None)
    if n_samples is not None and (isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1):
        raise ValueError(f"sample_selection.n_samples must be a positive int, got {n_samples!r}")
    fraction = block.get("fraction", None)
    if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                                 or not 0.0 < fraction <= 1.0):
        raise ValueError(f"sample_selection.fraction must be in (0, 1], got {fraction!r}")
    seed = block.get("seed", parameters.get("seed", 34))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"sample_selection.seed must be an int, got {seed!r}")
    clusters = block.get("clusters", 64)
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 1:
        raise ValueError(f"sample_selection.clusters must be a positive int, got {clusters!r}")
    temper = block.get("temper", 0.0 if mode in ("month", "direction") else 0.5)   # equal groups unless stated
    if isinstance(temper, bool) or not isinstance(temper, (int, float)) or not 0.0 <= temper <= 1.0:
        raise ValueError(f"sample_selection.temper must be in [0, 1], got {temper!r}")
    offset = block.get("offset", 0.0)
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not 0.0 <= offset < 1.0:
        raise ValueError(f"sample_selection.offset must be in [0, 1), got {offset!r}")
    if mode == "random" and offset != 0:
        raise ValueError("sample_selection.offset has no meaning for the random mode (use another seed)")
    descriptors = list(block.get("descriptors", list(DESCRIPTORS)))
    bad = [d for d in descriptors if d not in DESCRIPTORS]
    if bad or not descriptors:
        raise ValueError(f"sample_selection.descriptors must be a non-empty subset of {DESCRIPTORS}, got {descriptors}")
    far_radius = block.get("far_radius", 15.0)
    if isinstance(far_radius, bool) or not isinstance(far_radius, (int, float)) or far_radius <= 0:
        raise ValueError(f"sample_selection.far_radius must be > 0, got {far_radius!r}")
    iterations = block.get("kmeans_iterations", 100)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"sample_selection.kmeans_iterations must be a positive int, got {iterations!r}")
    config = {"mode": mode, "n_samples": n_samples, "fraction": fraction, "seed": int(seed),
              "clusters": int(clusters), "temper": float(temper), "descriptors": descriptors,
              "far_radius": float(far_radius), "kmeans_iterations": int(iterations), "offset": float(offset)}
    if mode == "spacing":
        for key in ("land_km", "sea_km"):
            value = block.get(key, None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"sample_selection.{key} (minimum distance in km between the soundings of one "
                                 f"overpass) must be a number >= 0 in the spacing mode, got {value!r}")
            config[key] = float(value)
        gap = block.get("overpass_gap_minutes", 30.0)
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or gap <= 0:
            raise ValueError(f"sample_selection.overpass_gap_minutes must be > 0, got {gap!r}")
        config["overpass_gap_minutes"] = float(gap)
        mask_file = block.get("land_mask_file", None)
        if mask_file is not None and not isinstance(mask_file, str):
            raise ValueError(f"sample_selection.land_mask_file must be a path, got {mask_file!r}")
        config["land_mask_file"] = mask_file
    return config


def budget_for(config, n_pool):
    """Number of samples to select from a pool of ``n_pool`` (the spacing mode without a budget is
    resolved by :func:`select_samples`: every survivor of the thinning)."""
    budget = config["n_samples"] if config["n_samples"] is not None else int(round(config["fraction"] * n_pool))
    if budget < 1 or budget > n_pool:
        raise ValueError(f"sample_selection: {budget} samples requested from a pool of {n_pool}")
    return int(budget)


# ---------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------

def footprint_descriptors(fp, far_radius=15.0):
    """Per-footprint statistics of a window footprint array.

    Args:
        fp (np.ndarray): Footprints ``(n, lat, lon)``; NaN cells (window outside the domain) carry
            no mass. The release point is the cell ``(lat // 2, lon // 2)`` as in
            ``gates.evaluation.loss_functions.ring_regions``.
        far_radius (float): Cells beyond this distance from the release point are the far field
            (15 = the far ring of the near / mid / far split used in ``scripts/compare_footprints.py``).

    Returns:
        np.ndarray: ``(n, 5)``: log10 of the total mass, far-field mass share, mass-weighted
        centroid offset (dx = columns east, dy = rows north) in cells, and the extent (root mean
        square distance of the mass from the release point, in cells). A footprint without mass
        gets the smallest log-mass of the others and zeros.
    """
    n, height, width = fp.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dy = (yy - height // 2).astype(np.float64)
    dx = (xx - width // 2).astype(np.float64)
    r2 = dx ** 2 + dy ** 2
    far = (np.sqrt(r2) > far_radius).astype(np.float64)
    out = np.zeros((n, 5), dtype=np.float64)
    for a in range(0, n, _BLOCK):
        f = np.nan_to_num(np.asarray(fp[a:a + _BLOCK], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        f[f < 0] = 0.0
        mass = f.sum(axis=(1, 2))
        safe = np.where(mass > 0, mass, 1.0)
        out[a:a + _BLOCK, 1] = np.tensordot(f, far, axes=([1, 2], [0, 1])) / safe
        out[a:a + _BLOCK, 2] = np.tensordot(f, dx, axes=([1, 2], [0, 1])) / safe
        out[a:a + _BLOCK, 3] = np.tensordot(f, dy, axes=([1, 2], [0, 1])) / safe
        out[a:a + _BLOCK, 4] = np.sqrt(np.tensordot(f, r2, axes=([1, 2], [0, 1])) / safe)
        out[a:a + _BLOCK, 0] = np.where(mass > 0, np.log10(safe), np.nan)
    empty = np.isnan(out[:, 0])
    if empty.any():
        out[empty, 0] = np.nanmin(out[:, 0]) if (~empty).any() else 0.0
    return out


def window_orientation(fps):
    """Whether the row index of the footprint window runs northwards and the column index eastwards.

    Read from the per-sample window coordinates ``lat_coords`` ``(time, lat)`` / ``lon_coords``
    ``(time, lon)`` the loader stores with the footprints.

    Returns:
        tuple: ``(sign_lat, sign_lon, flags)``: +1.0 / -1.0 each, and the names of the coordinates
        that were missing (then +1 is assumed: rows northwards, columns eastwards).
    """
    signs, flags = [], []
    for name, dim in (("lat_coords", "lat"), ("lon_coords", "lon")):
        if name in fps and dim in fps[name].dims:
            values = np.asarray(fps[name].transpose(..., dim).values, dtype=np.float64)
            step = np.nanmean(values[..., -1] - values[..., 0])
            signs.append(-1.0 if step < 0 else 1.0)
        else:
            signs.append(1.0)
            flags.append(f"{name} missing: the window is assumed to run {'northwards' if dim == 'lat' else 'eastwards'} with its index")
    return signs[0], signs[1], flags


def transport_direction(fps, near_cells=NEAR_RELEASE_CELLS):
    """Group of every footprint of the pool by where its mass lies as seen from the release point.

    The compass sector (45 degrees wide, centred on N, NE, ... NW) of the mass-weighted centroid of
    the footprint, or "near release" when the centroid is within ``near_cells`` cells of the release
    cell (also for a footprint without mass). The same definition as the "transport direction"
    grouping of ``scripts/worst_case_error.py``.

    Returns:
        tuple: ``(labels (n,) int64 indices into DIRECTION_GROUPS, flags (list[str]))``.
    """
    fp_var = "fp" if "fp" in fps else list(fps.data_vars)[0]
    desc = footprint_descriptors(fps[fp_var].transpose("time", "lat", "lon").values)
    sign_lat, sign_lon, flags = window_orientation(fps)
    east, north = sign_lon * desc[:, 2], sign_lat * desc[:, 3]
    sector = ((np.degrees(np.arctan2(east, north)) % 360.0 + 22.5) // 45.0).astype(np.int64) % 8
    labels = np.where(np.hypot(east, north) <= near_cells, len(SECTORS), sector).astype(np.int64)
    return labels, flags


def season_descriptors(times):
    """``(n, 2)``: sine and cosine of the day of the year (with the hour), so that December and
    January are neighbours."""
    t = pd.DatetimeIndex(np.asarray(times))
    angle = 2.0 * np.pi * (t.dayofyear.values + t.hour.values / 24.0) / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)])


def descriptor_matrix(fps, times, config):
    """Standardised descriptors of every sample of the pool.

    Args:
        fps (xr.Dataset): Footprints of the pool, variable ``fp`` ``(time, lat, lon)`` and, if the
            loader kept them, ``release_lat`` / ``release_lon`` ``(time,)``.
        times (np.ndarray): Sample times (datetime64), in the order of ``fps``.
        config (dict): Output of :func:`selection_config`.

    Returns:
        tuple: ``(X (n, d) float64, names (list[str]), flags (list[str]))``; a column with no
        spread is left centred (not scaled). ``flags`` names requested descriptors that were not
        available.
    """
    columns, names, flags = [], [], []
    fp_var = "fp" if "fp" in fps else list(fps.data_vars)[0]
    if "footprint" in config["descriptors"]:
        columns.append(footprint_descriptors(fps[fp_var].transpose("time", "lat", "lon").values, config["far_radius"]))
        names += FOOTPRINT_NAMES
    if "season" in config["descriptors"]:
        columns.append(season_descriptors(times))
        names += ["sin_doy", "cos_doy"]
    if "position" in config["descriptors"]:
        if "release_lat" in fps and "release_lon" in fps:
            columns.append(np.column_stack([np.asarray(fps["release_lat"].values, dtype=np.float64),
                                            np.asarray(fps["release_lon"].values, dtype=np.float64)]))
            names += ["release_lat", "release_lon"]
        else:
            flags.append("position descriptors requested but the footprints carry no release_lat / release_lon")
    if not columns:
        raise ValueError("sample_selection: no descriptor is available for the coverage mode")
    X = np.concatenate(columns, axis=1)
    if not np.isfinite(X).all():
        raise ValueError("sample_selection: non-finite descriptor values")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (X - mean) / std, names, flags


# ---------------------------------------------------------------
# k-means and the allocation of the budget
# ---------------------------------------------------------------

def _sq_distances(X, centres, block=8192):
    """Squared Euclidean distances ``(n, k)``, computed by broadcasting in blocks (no BLAS: the
    result does not depend on the thread count)."""
    out = np.empty((len(X), len(centres)), dtype=np.float64)
    for a in range(0, len(X), block):
        diff = X[a:a + block, None, :] - centres[None, :, :]
        out[a:a + block] = np.einsum("ijk,ijk->ij", diff, diff)
    return out


def kmeans(X, k, seed, iterations=100):
    """Lloyd's k-means with k-means++ seeding, deterministic for a given ``seed``.

    Returns ``(labels (n,), centres (k, d))``; an emptied cluster is re-seeded on the point
    farthest from its centre. ``k`` is reduced to the number of points if larger.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    k = int(min(k, n))
    rng = np.random.default_rng(seed)
    centres = np.empty((k, X.shape[1]), dtype=np.float64)
    centres[0] = X[rng.integers(n)]
    d2 = ((X - centres[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = d2.sum()
        p = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centres[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - centres[j]) ** 2).sum(axis=1))
    labels = None
    for _ in range(iterations):
        dist = _sq_distances(X, centres)
        new = dist.argmin(axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        nearest = dist[np.arange(n), labels]
        for j in range(k):
            members = labels == j
            if members.any():
                centres[j] = X[members].mean(axis=0)
            else:
                centres[j] = X[nearest.argmax()]
    return labels, centres


def tempered_allocation(counts, budget, temper):
    """Split ``budget`` over clusters of sizes ``counts`` in proportion to ``counts ** temper``.

    No cluster gets more than its size (the excess is shared out again among the others, largest
    remainders first), and the allocations sum to ``budget``. ``temper`` 1 reproduces the pool's
    proportions, 0 gives every cluster the same number.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if budget > counts.sum():
        raise ValueError(f"budget {budget} exceeds the {counts.sum()} samples of the clusters")
    alloc = np.zeros(len(counts), dtype=np.int64)
    capped = counts <= 0
    remaining = int(budget)
    while remaining > 0:
        free = np.nonzero(~capped)[0]
        if len(free) == 0:
            raise RuntimeError("tempered_allocation: no cluster left to allocate to")
        weights = counts[free].astype(np.float64) ** float(temper)
        share = remaining * weights / weights.sum()
        target = np.floor(share).astype(np.int64)
        leftover = remaining - int(target.sum())
        order = np.argsort(-(share - target), kind="stable")
        target[order[:leftover]] += 1
        room = counts[free] - alloc[free]
        over = target > room
        if not over.any():
            alloc[free] += target
            remaining = 0
        else:
            alloc[free[over]] += room[over]
            remaining -= int(room[over].sum())
            capped[free[over]] = True
    return alloc


def evenly_spaced(n, a, offset=0.0):
    """``a`` distinct indices of ``range(n)``, as evenly spaced as possible (all of them if
    ``a >= n``).

    ``offset`` 0 (the default) takes the first and the last index. Any other value in (0, 1) moves
    every pick by ``round(offset * n / a)`` positions (that fraction of the spacing; a pick moved
    past the end re-enters at the start): the SAME spacing pattern on other samples, so that two
    offsets whose shifts differ by less than the spacing share (almost) no sample.
    """
    if a <= 0:
        return np.zeros(0, dtype=np.int64)
    if a >= n:
        return np.arange(n, dtype=np.int64)
    picks = np.unique(np.round(np.linspace(0, n - 1, a)).astype(np.int64))
    if len(picks) < a:   # cannot happen for a <= n (spacing >= 1), kept as a safety net
        rest = np.setdiff1d(np.arange(n, dtype=np.int64), picks, assume_unique=True)
        picks = np.sort(np.concatenate([picks, rest[:a - len(picks)]]))
    if offset:
        picks = np.sort((picks + int(round(float(offset) * n / a))) % n)
    return picks


# ---------------------------------------------------------------
# Spacing inside the overpass, land and sea
# ---------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (degrees in, broadcasting)."""
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def overpass_ids(times, gap_minutes=30.0):
    """Overpass number of every sounding: a gap of more than ``gap_minutes`` between consecutive
    soundings starts a new overpass. ``times`` must be sorted ascending."""
    times = np.asarray(times).astype("datetime64[ns]")
    if len(times) == 0:
        return np.zeros(0, dtype=np.int64)
    gap = np.diff(times).astype("timedelta64[ns]").astype(np.int64)
    if (gap < 0).any():
        raise ValueError("overpass_ids: the times are not sorted")
    return np.concatenate([[0], np.cumsum(gap > int(round(gap_minutes * 60 * 1e9)))]).astype(np.int64)


def thin_min_spacing(lat, lon, overpass, d_km):
    """Greedy thinning inside every overpass, in time order: a sounding is kept if it is at least
    ``d_km`` from every sounding already kept in its overpass (so the first one always is).

    Args:
        lat, lon (np.ndarray): Release points in degrees, in time order.
        overpass (np.ndarray): Overpass number of every sounding (non-decreasing).
        d_km (float): Minimum distance between kept soundings of one overpass; 0 keeps everything.

    Returns:
        np.ndarray: Boolean mask of the kept soundings.
    """
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    overpass = np.asarray(overpass)
    keep = np.zeros(len(lat), dtype=bool)
    if len(lat) == 0:
        return keep
    if d_km <= 0:
        keep[:] = True
        return keep
    start = np.concatenate([[0], np.nonzero(overpass[1:] != overpass[:-1])[0] + 1, [len(overpass)]])
    for a, b in zip(start[:-1], start[1:]):
        kept_lat, kept_lon = [], []
        for i in range(a, b):
            if not kept_lat or haversine_km(lat[i], lon[i], np.array(kept_lat), np.array(kept_lon)).min() >= d_km:
                keep[i] = True
                kept_lat.append(lat[i])
                kept_lon.append(lon[i])
    return keep


def release_positions(fps):
    """Release points and release-cell centres of the pool.

    Returns:
        tuple: ``(lat, lon, cell_lat, cell_lon, flags)``: the release points (``release_lat`` /
        ``release_lon``, used for the distances) and the centres of the window cell that holds the
        release point (cell ``(lat // 2, lon // 2)`` of ``lat_coords`` / ``lon_coords``, used for
        the land mask: the surface the model sees under the release). Without window coordinates
        the release points stand in for the cell centres, with a flag.
    """
    if "release_lat" not in fps or "release_lon" not in fps:
        raise ValueError("sample_selection: the spacing mode needs release_lat / release_lon in the footprints")
    lat = np.asarray(fps["release_lat"].values, dtype=np.float64)
    lon = np.asarray(fps["release_lon"].values, dtype=np.float64)
    flags = []
    if ("lat_coords" in fps and "lon_coords" in fps and "lat" in fps["lat_coords"].dims
            and "lon" in fps["lon_coords"].dims):
        cell_lat = np.asarray(fps["lat_coords"].transpose(..., "lat").values, dtype=np.float64)[..., fps.sizes["lat"] // 2]
        cell_lon = np.asarray(fps["lon_coords"].transpose(..., "lon").values, dtype=np.float64)[..., fps.sizes["lon"] // 2]
    else:
        cell_lat, cell_lon = lat, lon
        flags.append("lat_coords / lon_coords missing: the land mask is read at the release point, not at the "
                     "centre of the release cell")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all() and np.isfinite(cell_lat).all() and np.isfinite(cell_lon).all()):
        raise ValueError("sample_selection: non-finite release positions")
    return lat, lon, cell_lat, cell_lon, flags


def _nearest_index(axis, values, period=None):
    """Index of the axis value nearest to each of ``values`` (``axis`` ascending; ``period`` = the
    axis wraps around, e.g. 360 for a global longitude)."""
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(axis)
    if period is not None:
        values = axis[0] + np.mod(values - axis[0], period)
        axis = np.concatenate([axis, [axis[0] + period]])
    right = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    left = right - 1
    index = np.where(values - axis[left] <= axis[right] - values, left, right)
    return index % n if period is not None else index


def land_mask_at(lat, lon, path, variable=LAND_MASK_VARIABLE):
    """Whether the points lie on LAND according to a global land-mask file (nearest cell).

    Args:
        lat, lon (np.ndarray): Points in degrees; the longitude convention (0..360 or -180..180)
            does not matter.
        path (str or Path): netCDF file with ``variable`` on (lat, lon) or (latitude, longitude),
            > 0.5 = land. The longitude axis must be global (it is treated as periodic).

    Returns:
        np.ndarray: Boolean, True = land.
    """
    import xarray as xr   # (the rest of the module needs NumPy and pandas only)
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"sample_selection: no variable {variable!r} in the land-mask file {path}")
        mask = ds[variable]
        lat_name = next((d for d in ("lat", "latitude") if d in mask.dims), None)
        lon_name = next((d for d in ("lon", "longitude") if d in mask.dims), None)
        if lat_name is None or lon_name is None or mask.ndim != 2:
            raise ValueError(f"sample_selection: {variable} of {path} must be on (lat, lon), got {mask.dims}")
        mask = mask.sortby(lat_name).sortby(lon_name).transpose(lat_name, lon_name)
        values = np.asarray(mask.values, dtype=np.float64)
        axis_lat = np.asarray(mask[lat_name].values, dtype=np.float64)
        axis_lon = np.asarray(mask[lon_name].values, dtype=np.float64)
    step = float(np.median(np.diff(axis_lon)))
    if abs((axis_lon[-1] - axis_lon[0] + step) - 360.0) > 1.5 * step:
        raise ValueError(f"sample_selection: the longitude axis of {path} is not global "
                         f"({axis_lon[0]:.3f}..{axis_lon[-1]:.3f})")
    picked = values[_nearest_index(axis_lat, lat), _nearest_index(axis_lon, lon, period=360.0)]
    if not np.isfinite(picked).all():
        raise ValueError(f"sample_selection: the land mask {path} is undefined at {int((~np.isfinite(picked)).sum())} release cells")
    return picked > 0.5


def _land_mask_path(config):
    """The land-mask file of a spacing selection: its ``land_mask_file`` or config.yml's
    ``data_paths.landcover_datadir``."""
    if config.get("land_mask_file"):
        return str(config["land_mask_file"])
    import gates.config
    path = getattr(gates.config.get_config(), "landcover_datadir", None)
    if path is None:
        raise ValueError("sample_selection: the spacing mode needs a land mask: set data_paths.landcover_datadir "
                         "in config.yml or sample_selection.land_mask_file")
    return str(path)


def spacing_survivors(fps, times, config):
    """The soundings that survive the distance thinning of the spacing mode.

    Returns:
        tuple: ``(survivors (ascending pool indices), labels (n_pool,) int64: 0 = land release,
        1 = sea release, extra (dict for the report))``.
    """
    lat, lon, cell_lat, cell_lon, flags = release_positions(fps)
    mask_path = _land_mask_path(config)
    sea = ~land_mask_at(cell_lat, cell_lon, mask_path)
    overpass = overpass_ids(times, config["overpass_gap_minutes"])
    kept = np.zeros(len(overpass), dtype=bool)
    for is_sea, d_km in ((False, config["land_km"]), (True, config["sea_km"])):
        index = np.nonzero(sea == is_sea)[0]            # ascending = in time order
        kept[index[thin_min_spacing(lat[index], lon[index], overpass[index], d_km)]] = True
    survivors = np.nonzero(kept)[0]
    labels = sea.astype(np.int64)
    extra = {"groups": "surface under the release point", "group_names": list(SURFACE_GROUPS),
             "land_km": config["land_km"], "sea_km": config["sea_km"],
             "overpass_gap_minutes": config["overpass_gap_minutes"], "land_mask_file": mask_path,
             "n_overpasses_pool": int(overpass[-1]) + 1 if len(overpass) else 0,
             "n_overpasses_thinned": int(len(np.unique(overpass[survivors]))),
             "group_sizes": np.bincount(labels, minlength=2).tolist(),
             "group_thinned": np.bincount(labels[survivors], minlength=2).tolist(), "flags": flags}
    return survivors, labels, extra


# ---------------------------------------------------------------
# The selection
# ---------------------------------------------------------------

def fingerprint(times):
    """Short hash of a set of sample times (to recognise a selection across rebuilds)."""
    t = np.asarray(times).astype("datetime64[ns]").astype(np.int64)
    return hashlib.sha1(np.ascontiguousarray(t).tobytes()).hexdigest()[:16]


def _month_shares(times, keep):
    t = pd.DatetimeIndex(np.asarray(times))
    pool = np.bincount(t.month.values, minlength=13)[1:]
    chosen = np.bincount(t.month.values[keep], minlength=13)[1:]
    return (pool / max(1, pool.sum())).round(4).tolist(), (chosen / max(1, chosen.sum())).round(4).tolist()


def _pick_in_groups(labels, alloc, offset=0.0):
    """``alloc[j]`` samples of every group ``j`` (``labels == j``), evenly spaced in time inside
    the group; ascending indices into the pool."""
    parts = []
    for j in range(len(alloc)):
        members = np.nonzero(labels == j)[0]          # ascending = in time order
        parts.append(members[evenly_spaced(len(members), int(alloc[j]), offset)])
    return np.sort(np.concatenate(parts))


def select_samples(fps, times, config):
    """Choose the training samples of the pool.

    Args:
        fps (xr.Dataset): Footprints of the pool (``fp`` ``(time, lat, lon)``, optional release
            coordinates), sorted by time.
        times (np.ndarray): The pool's sample times (datetime64, sorted ascending).
        config (dict): Output of :func:`selection_config`.

    Returns:
        tuple: ``keep`` (np.ndarray of ascending indices into the pool), ``report`` (dict, JSON
        friendly, for ``parameters["sample_selection_resolved"]``), ``labels`` (group label of
        every pool sample: the k-means cluster in coverage mode, the calendar month minus one in
        month mode, the index into ``DIRECTION_GROUPS`` in direction mode, 0 = land / 1 = sea
        release (``SURFACE_GROUPS``) in spacing mode, else None).
    """
    times = np.asarray(times)
    n_pool = len(times)
    if n_pool != fps.sizes["time"]:
        raise ValueError(f"sample_selection: {n_pool} times but {fps.sizes['time']} footprints")
    mode = config["mode"]
    labels = None
    if mode == "spacing":
        survivors, labels, extra = spacing_survivors(fps, times, config)
        no_budget = config["n_samples"] is None and config["fraction"] is None
        budget = len(survivors) if no_budget else budget_for(config, n_pool)
        if budget > len(survivors):
            raise ValueError(f"sample_selection: the spacing rule (land >= {config['land_km']:g} km, sea >= "
                             f"{config['sea_km']:g} km) leaves {len(survivors)} of the {n_pool} samples, fewer than "
                             f"the {budget} requested: lower the budget or the spacings")
        keep = survivors[evenly_spaced(len(survivors), budget, config["offset"])]
    else:
        budget = budget_for(config, n_pool)
    report = {"mode": mode, "n_pool": int(n_pool), "n_selected": int(budget), "seed": config["seed"]}
    if mode == "spacing":
        extra["group_allocation"] = np.bincount(labels[keep], minlength=2).tolist()
        extra["n_overpasses_selected"] = int(len(np.unique(overpass_ids(times, config["overpass_gap_minutes"])[keep])))
        report.update(extra)
    elif mode == "random":
        rng = np.random.default_rng(config["seed"])
        keep = np.sort(rng.choice(n_pool, budget, replace=False))
    elif mode == "stride":
        keep = evenly_spaced(n_pool, budget, config["offset"])
    elif mode == "month":
        labels = pd.DatetimeIndex(times).month.values.astype(np.int64) - 1     # 0 = January
        counts = np.bincount(labels, minlength=12)
        alloc = tempered_allocation(counts, budget, config["temper"])
        keep = _pick_in_groups(labels, alloc, config["offset"])
        report.update({"groups": "calendar month", "temper": config["temper"],
                       "group_sizes": counts.tolist(), "group_allocation": alloc.tolist()})
    elif mode == "direction":
        labels, flags = transport_direction(fps)
        counts = np.bincount(labels, minlength=len(DIRECTION_GROUPS))
        alloc = tempered_allocation(counts, budget, config["temper"])
        keep = _pick_in_groups(labels, alloc, config["offset"])
        report.update({"groups": "transport direction", "group_names": list(DIRECTION_GROUPS), "temper": config["temper"],
                       "near_release_cells": NEAR_RELEASE_CELLS, "group_sizes": counts.tolist(),
                       "group_allocation": alloc.tolist(), "flags": flags})
    else:
        X, names, flags = descriptor_matrix(fps, times, config)
        labels, _ = kmeans(X, config["clusters"], config["seed"], config["kmeans_iterations"])
        k = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=k)
        alloc = tempered_allocation(counts, budget, config["temper"])
        keep = _pick_in_groups(labels, alloc, config["offset"])
        report.update({"descriptors": names, "clusters": k, "temper": config["temper"],
                       "far_radius": config["far_radius"], "cluster_sizes": counts.tolist(),
                       "cluster_allocation": alloc.tolist(), "flags": flags})
    if config["offset"]:
        report["offset"] = config["offset"]
    if len(keep) != budget or len(np.unique(keep)) != budget:
        raise RuntimeError(f"sample_selection: selected {len(keep)} samples ({len(np.unique(keep))} distinct) "
                           f"instead of {budget}")
    pool_share, chosen_share = _month_shares(times, keep)
    report.update({"fingerprint": fingerprint(times[keep]),
                   "time_first": str(times[keep][0]), "time_last": str(times[keep][-1]),
                   "month_share_pool": pool_share, "month_share_selected": chosen_share})
    return keep, report, labels


def describe(report):
    """One-paragraph summary of a selection report for the logs."""
    text = (f"sample_selection: {report['mode']} picked {report['n_selected']} of {report['n_pool']} samples "
            f"(seed {report['seed']}, fingerprint {report['fingerprint']})")
    if report["mode"] == "coverage":
        sizes, alloc = np.asarray(report["cluster_sizes"]), np.asarray(report["cluster_allocation"])
        text += (f"; {report['clusters']} clusters on {report['descriptors']} with temper {report['temper']}: "
                 f"cluster sizes {sizes.min()}-{int(np.median(sizes))}-{sizes.max()} (min-median-max), "
                 f"allocation {alloc.min()}-{int(np.median(alloc))}-{alloc.max()}, "
                 f"{int((alloc == sizes).sum())} cluster(s) taken whole")
        for flag in report.get("flags", []):
            text += f"\nFLAG: {flag}"
    if report["mode"] == "month":
        sizes, alloc = np.asarray(report["group_sizes"]), np.asarray(report["group_allocation"])
        text += (f"; calendar months with temper {report['temper']}: month sizes {sizes.min()}-{sizes.max()} "
                 f"(min-max), allocation {alloc.min()}-{alloc.max()}, {int((alloc == sizes).sum())} month(s) taken whole")
    if report["mode"] == "direction":
        sizes, alloc = np.asarray(report["group_sizes"]), np.asarray(report["group_allocation"])
        text += (f"; transport directions with temper {report['temper']} ("
                 + ", ".join(f"{name} {a} of {s}" for name, a, s in zip(report["group_names"], alloc, sizes))
                 + f"), {int(((alloc == sizes) & (sizes > 0)).sum())} direction(s) taken whole")
        for flag in report.get("flags", []):
            text += f"\nFLAG: {flag}"
    if report["mode"] == "spacing":
        sizes, thinned, alloc = report["group_sizes"], report["group_thinned"], report["group_allocation"]
        text += (f"; soundings of one overpass at least {report['land_km']:g} km apart over land and "
                 f"{report['sea_km']:g} km over the sea (overpass = gaps of at most {report['overpass_gap_minutes']:g} min): "
                 f"land releases {sizes[0]} -> {thinned[0]} -> {alloc[0]}, sea releases {sizes[1]} -> {thinned[1]} -> "
                 f"{alloc[1]} (pool -> thinned -> selected); overpasses {report['n_overpasses_pool']} -> "
                 f"{report['n_overpasses_selected']}")
        for flag in report.get("flags", []):
            text += f"\nFLAG: {flag}"
    if report.get("offset"):
        text += f"; offset {report['offset']}"
    pool, chosen = np.asarray(report["month_share_pool"]), np.asarray(report["month_share_selected"])
    djf = [11, 0, 1]
    jja = [5, 6, 7]
    text += (f"; month shares DJF {pool[djf].sum():.3f} -> {chosen[djf].sum():.3f}, "
             f"JJA {pool[jja].sum():.3f} -> {chosen[jja].sum():.3f} (pool -> selected)")
    return text
