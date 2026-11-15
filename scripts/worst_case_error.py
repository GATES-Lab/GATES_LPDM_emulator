"""Worst-case footprint error of dual-head runs: the TAIL of the per-footprint error, not only its mean.

    python scripts/worst_case_error.py snapshot <out_prefix> <labelA>=<run_dir>::<file.nc> <labelB>=<run_dir>::<file.nc> \
        [--yardstick <label>=<run_dir>::<file.nc> <label>=<run_dir>::<file.nc>] [--title "..."] [--bootstrap 2000]
    python scripts/worst_case_error.py track <out_prefix> <labelA>=<run_dir> <labelB>=<run_dir> [--title "..."] [--from-epoch 50]

The runs must have been predicted on the same test footprints (``predict_dual_model.py``: same times, same truth).

Per footprint: the training-space MSE (the footprint loss in its own units; its mean over the footprints is the
logged test loss), the absolute log mass ratio |ln(mass pred / mass true)| (0.69 = a factor of 2) and the absolute
error of the mole-fraction enhancement of a uniform flux (footprint mass x flux x 1e9 = ppb; default flux
1e-8 mol m-2 s-1, the convention of experiment summary section 41).

``snapshot`` (one prediction file per run): mean, median and the tail of each quantity - 90th / 95th / 99th
percentile, maximum, mean of the worst 10 % / 5 % / 1 % of the footprints, share of the summed error held by the worst
10 % -; the worst GROUP when the footprints are grouped by month, season, true extent, true far-field share, transport
direction and release latitude; whether the two runs fail on the same footprints (rank correlation, overlap of the
worst deciles, each run's error on the other's worst decile); and a paired bootstrap over the test footprints of the
A - B difference of every statistic. The bootstrap interval is the uncertainty from the finite TEST SET only: it says
nothing about the spread between two trainings, for which ``--yardstick`` takes two runs that differ in their seed
alone and prints how far apart those are on the same statistics. Writes ``<out_prefix>_per_sample.csv`` and
``<out_prefix>.png``.

``track``: the same statistics at every epoch for which both runs hold ``sample_predictions_test_<epoch>.nc``
(predictions at the periodic checkpoints), as tables, ``<out_prefix>.csv`` and ``<out_prefix>.png``: how the mean and
the worst case move through training, and at how many checkpoints each run is ahead. A single checkpoint catches
each run at an arbitrary phase of its epoch-to-epoch swings, so the track ends with the CHECKPOINT-AVERAGED view:
every footprint's loss averaged over the checkpoints from ``--from-epoch``, then the same tail table, bootstrap,
same-footprints and worst-group analysis as ``snapshot``, plus the PERSISTENT worst footprints (in a run's worst
5 % at two thirds of those checkpoints or more): how many, how many are shared by the two runs, and what they are.
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

THRESH = 1e-4                    # "extent" threshold, as scripts/compare_footprints.py
FAR_RADIUS = 15                  # far field = beyond 15 cells from the release point
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASON_OF_MONTH = ["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"]
STATS = ["mean", "median", "p90", "p95", "p99", "max", "worst10", "worst5", "worst1", "share10"]
STAT_NAMES = {"mean": "mean", "median": "median", "p90": "90th percentile", "p95": "95th percentile",
              "p99": "99th percentile", "max": "maximum", "worst10": "mean of the worst 10 %",
              "worst5": "mean of the worst 5 %", "worst1": "mean of the worst 1 %",
              "share10": "share of the total in the worst 10 %"}
QUANTITIES = {"mse": "footprint loss per footprint (training-space MSE; its mean is the test loss)",
              "mass": "absolute log mass ratio |ln(mass pred / mass true)| (0.69 = a factor of 2)",
              "enh": "absolute error of the uniform-flux enhancement (ppb)"}
# categorical slots 1 and 2 of the validated palette (light surface), text and grid tokens
COL_A, COL_B = "#2a78d6", "#eb6834"
COL_TEXT, COL_MUTED, COL_GRID, COL_SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ---------------------------------------------------------------
# Per-footprint quantities, tail statistics, groups
# ---------------------------------------------------------------

def load(arg):
    """``<label>=<run_dir>::<file.nc>`` -> (label, loaded Dataset)."""
    label, rest = arg.split("=", 1)
    run_dir, fname = rest.split("::")
    d = xr.open_dataset(Path(run_dir) / fname).load()
    print(f"{label}: {Path(run_dir).name}/{fname} ({d.attrs.get('checkpoint', '?')}, {d.sizes['time']} footprints, "
          f"fp loss {d.attrs.get('test_fp_loss', float('nan')):.4f})")
    return label, d


def check_same_test_set(ref, other, label):
    if not np.array_equal(ref.time.values, other.time.values):
        raise SystemExit(f"{label}: different test times - not comparable")
    if not np.allclose(ref.fp_original.values, other.fp_original.values, equal_nan=True):
        raise SystemExit(f"{label}: different ground truth - not comparable")


def per_footprint(d, flux=1e-8):
    """Per-footprint errors of a prediction Dataset: ``{"mse", "mass", "enh"}``, each of shape (n,)."""
    valid = ~d.fp_nan_mask.values.astype(bool)
    true_lin = np.where(valid, d.fp_original.values.astype("float64"), 0.0)
    pred_lin = np.where(valid, d.fp_pred.values.astype("float64"), 0.0)
    true_log = d.fp_transformed.values.astype("float64")
    pred_log = d.fp_transformed_pred.values.astype("float64")
    mse = (((pred_log - true_log) ** 2) * valid).sum((1, 2)) / valid.sum((1, 2))
    mass_true, mass_pred = true_lin.sum((1, 2)), pred_lin.sum((1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mass = np.abs(np.log(mass_pred / mass_true))
    mass[~np.isfinite(mass)] = np.nan
    return {"mse": mse, "mass": mass, "enh": np.abs(mass_pred - mass_true) * flux * 1e9}


def tail(x):
    """Mean, median and tail statistics of a per-footprint error (non-finite entries ignored)."""
    s = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    n = len(s)
    k = lambda share: max(1, int(round(share * n)))   # noqa: E731
    return {"mean": s.mean(), "median": np.median(s), "p90": np.percentile(s, 90), "p95": np.percentile(s, 95),
            "p99": np.percentile(s, 99), "max": s[-1], "worst10": s[-k(0.10):].mean(), "worst5": s[-k(0.05):].mean(),
            "worst1": s[-k(0.01):].mean(), "share10": s[-k(0.10):].sum() / s.sum()}


def bootstrap_difference(xa, xb, n_boot=2000, seed=0):
    """Paired bootstrap over the footprints of ``tail(A) - tail(B)``: ``{stat: (n_boot,) differences}``."""
    keep = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = np.asarray(xa, float)[keep], np.asarray(xb, float)[keep]
    n = len(xa)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    k = lambda share: max(1, int(round(share * n)))   # noqa: E731

    def stats(sample):
        s = np.sort(sample, axis=1)
        return {"mean": s.mean(1), "median": np.median(s, axis=1), "p90": np.percentile(s, 90, axis=1),
                "p95": np.percentile(s, 95, axis=1), "p99": np.percentile(s, 99, axis=1), "max": s[:, -1],
                "worst10": s[:, -k(0.10):].mean(1), "worst5": s[:, -k(0.05):].mean(1), "worst1": s[:, -k(0.01):].mean(1),
                "share10": s[:, -k(0.10):].sum(1) / s.sum(1)}

    sa, sb = stats(xa[idx]), stats(xb[idx])
    return {key: sa[key] - sb[key] for key in STATS}


def quartile_labels(values, name, fmt="{:.3g}"):
    edges = np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0])
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, 3)
    return np.array([f"{name} q{i + 1} ({fmt.format(edges[i])}..{fmt.format(edges[i + 1])})" for i in idx], dtype=object)


def groupings(d):
    """Group label of every test footprint under six groupings (from the truth and the window coordinates)."""
    valid = ~d.fp_nan_mask.values.astype(bool)
    true_lin = np.where(valid, d.fp_original.values.astype("float64"), 0.0)
    n, h, w = true_lin.shape
    month = pd.DatetimeIndex(d.time.values).month.values
    lat, lon = d.lat_coords.values.astype(float), d.lon_coords.values.astype(float)
    lat_inc = bool(np.nanmean(lat[:, -1] - lat[:, 0]) > 0)
    lon_inc = bool(np.nanmean(lon[:, -1] - lon[:, 0]) > 0)
    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - h // 2) * (1 if lat_inc else -1)    # north positive
    dx = (xx - w // 2) * (1 if lon_inc else -1)    # east positive
    rad = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    mass = true_lin.sum((1, 2))
    safe = np.where(mass > 0, mass, np.nan)
    cy, cx = (true_lin * dy).sum((1, 2)) / safe, (true_lin * dx).sum((1, 2)) / safe
    sector = ((np.degrees(np.arctan2(cx, cy)) % 360 + 22.5) // 45).astype(int) % 8
    direction = np.where(np.hypot(cy, cx) <= 2, "near release", np.array(SECTORS, dtype=object)[sector])
    return {
        "month": np.array([f"month {m:02d}" for m in month], dtype=object),
        "season": np.array(SEASON_OF_MONTH, dtype=object)[month - 1],
        "true extent": quartile_labels((true_lin > THRESH).sum((1, 2)), "extent", "{:.0f}"),
        "true far-field share": quartile_labels((true_lin * (rad > FAR_RADIUS)).sum((1, 2)) / safe, "far share", "{:.2f}"),
        "transport direction": direction,
        "release latitude": quartile_labels(lat[:, h // 2], "lat", "{:.1f}"),
    }


def group_means(x, labels):
    """``[(group, n, mean)]`` sorted by the group name."""
    return [(g, int((labels == g).sum()), float(np.nanmean(x[labels == g]))) for g in sorted(set(labels))]


# ---------------------------------------------------------------
# Printed blocks shared by the snapshot and the checkpoint-averaged view of the track
# ---------------------------------------------------------------

def print_tail_table(xa, xb, title, n_boot, fmt="{:.4f}", yard=None):
    ta, tb = tail(xa), tail(xb)
    boot = bootstrap_difference(xa, xb, n_boot)
    print(f"\n### {title}")
    head = f"{'statistic':>38} | {'A':>8} | {'B':>8} | {'A - B':>8} | 95 % interval of A - B | A lower in"
    print(head + (" | run-to-run |diff|" if yard else ""))
    for s in STATS:
        lo, hi = np.percentile(boot[s], [2.5, 97.5])
        line = (f"{STAT_NAMES[s]:>38} | {fmt.format(ta[s]):>8} | {fmt.format(tb[s]):>8} | {ta[s] - tb[s]:>+8.4f} | "
                f"[{lo:+.4f}, {hi:+.4f}]".ljust(102) + f" | {100 * (boot[s] < 0).mean():5.1f} %")
        print(line + (f" | {yard[s]:.4f}" if yard else ""))
    bad = int((~np.isfinite(xa)).sum() + (~np.isfinite(xb)).sum())
    if bad:
        print(f"  ({bad} non-finite value(s) left out)")
    return ta, tb


def print_same_footprints(xa, xb):
    """Do the two runs fail on the same footprints?"""
    rank = lambda v: np.argsort(np.argsort(v))   # noqa: E731
    wa, wb = xa >= np.percentile(xa, 90), xb >= np.percentile(xb, 90)
    print("\n### the same footprints? (footprint loss)")
    print(f"  rank correlation of the two runs' per-footprint losses: {np.corrcoef(rank(xa), rank(xb))[0, 1]:.3f}")
    print(f"  A lower on {100 * (xa < xb).mean():.1f} % of all footprints; median of A - B {np.median(xa - xb):+.4f}")
    print(f"  overlap of the worst deciles: {100 * (wa & wb).sum() / wa.sum():.1f} % of A's worst 10 % are also in B's "
          f"(10 % by chance)")
    print(f"  on A's worst decile: A {xa[wa].mean():.4f}, B {xb[wa].mean():.4f}; on B's worst decile: A {xa[wb].mean():.4f}, "
          f"B {xb[wb].mean():.4f}")
    union = wa | wb
    print(f"  on the {int(union.sum())} footprints in either worst decile: A {xa[union].mean():.4f}, B {xb[union].mean():.4f}, "
          f"A lower on {100 * (xa[union] < xb[union]).mean():.1f} %")


def print_worst_groups(xa, xb, groups, full=True):
    """Group means of the footprint loss under every grouping, and each run's worst group."""
    if full:
        print("\n### worst GROUP (mean footprint loss of the group; the worst group of each run is marked)")
    worst = {}
    for name, labels in groups.items():
        ga, gb = group_means(xa, labels), group_means(xb, labels)
        ia, ib = int(np.argmax([g[2] for g in ga])), int(np.argmax([g[2] for g in gb]))
        worst[name] = (ga[ia], gb[ib])
        if full:
            print(f"  by {name}: n | A | B | A - B")
            for i, ((g, cnt, ma), (_, _, mb)) in enumerate(zip(ga, gb)):
                mark = ("  <- worst of A" if i == ia else "") + ("  <- worst of B" if i == ib else "")
                print(f"    {g:>34} | {cnt:5d} | {ma:.4f} | {mb:.4f} | {ma - mb:+.4f}{mark}")
    print("\n### worst-group summary (footprint loss)")
    print(f"  {'grouping':>22} | worst group of A (value) | worst group of B (value) | A - B")
    for name, (wga, wgb) in worst.items():
        print(f"  {name:>22} | {wga[0]} ({wga[2]:.4f}) | {wgb[0]} ({wgb[2]:.4f}) | {wga[2] - wgb[2]:+.4f}")
    return worst


# ---------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------

def snapshot(args):
    (la, da), (lb, db) = load(args.runs[0]), load(args.runs[1])
    check_same_test_set(da, db, lb)
    qa, qb = per_footprint(da, args.flux), per_footprint(db, args.flux)
    n = len(qa["mse"])
    print(f"\n{n} test footprints; A = {la}, B = {lb}; differences are A - B (negative = A lower = A better); the 95 % "
          f"intervals are a paired bootstrap over the test footprints ({args.bootstrap} resamples): test-set uncertainty only")

    yard = None
    if args.yardstick:
        (ly1, dy1), (ly2, dy2) = load(args.yardstick[0]), load(args.yardstick[1])
        check_same_test_set(da, dy1, ly1)
        check_same_test_set(da, dy2, ly2)
        y1, y2 = per_footprint(dy1, args.flux), per_footprint(dy2, args.flux)
        yard = {key: {s: abs(tail(y1[key])[s] - tail(y2[key])[s]) for s in STATS} for key in QUANTITIES}
        print(f"run-to-run yardstick: |{ly1} - {ly2}| on every statistic (two trainings that differ in their seed alone)")

    for key, title in QUANTITIES.items():
        print_tail_table(qa[key], qb[key], title, args.bootstrap, fmt="{:.3f}" if key == "enh" else "{:.4f}",
                         yard=yard[key] if yard else None)
    xa, xb = qa["mse"], qb["mse"]
    print_same_footprints(xa, xb)
    groups = groupings(da)
    print_worst_groups(xa, xb, groups)

    frame = pd.DataFrame({"time": da.time.values, "mse_A": xa, "mse_B": xb, "mass_A": qa["mass"], "mass_B": qb["mass"],
                          "enh_ppb_A": qa["enh"], "enh_ppb_B": qb["enh"], **{f"group_{k.replace(' ', '_')}": v for k, v in groups.items()}})
    frame.to_csv(f"{args.out_prefix}_per_sample.csv", index=False)
    figure_snapshot(args, la, lb, xa, xb, groups)
    print(f"\nsaved {args.out_prefix}_per_sample.csv and {args.out_prefix}.png")


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(COL_SURFACE)
    ax.set_title(title, fontsize=10.5, color=COL_TEXT, loc="left")
    ax.set_xlabel(xlabel, fontsize=9.5, color=COL_MUTED)
    ax.set_ylabel(ylabel, fontsize=9.5, color=COL_MUTED)
    ax.grid(True, color=COL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COL_MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _figure_legend(fig, la, lb):
    """One legend for the figure (every panel shows the same two runs), in the title band at the right."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, linewidth=2, marker="o", markersize=7, markeredgecolor=COL_SURFACE, markeredgewidth=2)
               for c in (COL_A, COL_B)]
    fig.legend(handles, [la, lb], loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, frameon=False, fontsize=10,
               labelcolor=COL_TEXT, handlelength=2.2, columnspacing=1.8)


def _direct_labels(ax, x, ya, yb, la, lb, side="right"):
    """Name the two lines at one end, in text ink, when they are far enough apart there (else the legend carries it)."""
    lo, hi = ax.get_ylim()
    if abs(ya - yb) < 0.05 * (hi - lo):
        return
    dx = 8 if side == "right" else -8
    for y, lab in ((ya, la), (yb, lb)):
        ax.annotate(lab, (x, y), xytext=(dx, 0), textcoords="offset points", fontsize=9, color=COL_MUTED,
                    ha="left" if side == "right" else "right", va="center", annotation_clip=False)


def figure_snapshot(args, la, lb, xa, xb, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), facecolor=COL_SURFACE, gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    # (a) the upper half of the per-footprint loss distribution
    ax = axes[0]
    pct = np.linspace(50, 100, 501)
    for x, col, lab in ((xa, COL_A, la), (xb, COL_B, lb)):
        ax.plot(pct, np.percentile(x, pct), color=col, linewidth=2, solid_capstyle="round", label=lab)
    _style(ax, "Loss of the worst footprints (upper half of the distribution)", "percentile of the test footprints",
           "footprint loss per footprint")
    ax.set_xlim(50, 100.5)
    ax.set_xticks([50, 60, 70, 80, 90, 95, 99])
    # (b) by month
    ax = axes[1]
    months = sorted(set(groups["month"]))
    for x, col, lab in ((xa, COL_A, la), (xb, COL_B, lb)):
        ax.plot(range(1, len(months) + 1), [np.mean(x[groups["month"] == m]) for m in months], color=col, linewidth=2,
                marker="o", markersize=7, markeredgecolor=COL_SURFACE, markeredgewidth=2, label=lab)
    _style(ax, "Mean loss by month of the test year", "month", "mean footprint loss")
    ax.set_xticks(range(1, len(months) + 1))
    ax.set_xlim(0.5, len(months) + 0.5)
    last = months[-1]
    _direct_labels(ax, len(months), np.mean(xa[groups["month"] == last]), np.mean(xb[groups["month"] == last]), la, lb)
    # (c) by transport direction: one row per direction, the two runs joined by a connector
    ax = axes[2]
    order = [g for g in ["near release"] + SECTORS if g in set(groups["transport direction"])]
    for row, g in enumerate(order):
        sel = groups["transport direction"] == g
        va, vb = np.mean(xa[sel]), np.mean(xb[sel])
        ax.plot([va, vb], [row, row], color=COL_GRID, linewidth=2.5, solid_capstyle="round", zorder=1)
        ax.scatter([va], [row], s=70, color=COL_A, edgecolor=COL_SURFACE, linewidth=2, zorder=3)
        ax.scatter([vb], [row], s=70, color=COL_B, edgecolor=COL_SURFACE, linewidth=2, zorder=3)
    _style(ax, "Mean loss by transport direction (truth)", "mean footprint loss", "")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{g} (n={int((groups['transport direction'] == g).sum())})" for g in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    _figure_legend(fig, la, lb)
    fig.suptitle(args.title or f"Worst-case footprint error: {la} vs {lb}", fontsize=12, color=COL_TEXT, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 0.985, 0.93))
    fig.savefig(f"{args.out_prefix}.png", dpi=150, facecolor=COL_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------
# track
# ---------------------------------------------------------------

def epoch_files(run_dir):
    """``{epoch: file name}`` of the predictions at numbered checkpoints of a run."""
    out = {}
    for path in glob.glob(str(Path(run_dir) / "sample_predictions_test_*.nc")):
        m = re.fullmatch(r"sample_predictions_test_(\d+)\.nc", Path(path).name)
        if m:
            out[int(m.group(1))] = Path(path).name
    return out


def track(args):
    (la, ra), (lb, rb) = (r.split("=", 1) for r in args.runs)
    fa, fb = epoch_files(ra), epoch_files(rb)
    epochs = sorted(set(fa) & set(fb))
    if not epochs:
        raise SystemExit("no epoch for which both runs hold sample_predictions_test_<epoch>.nc")
    print(f"A = {la} ({Path(ra).name}), B = {lb} ({Path(rb).name}); {len(epochs)} common checkpoints: {epochs}")
    rows, month_labels = [], None
    per_a, per_b, times = {}, {}, None
    for ep in epochs:
        da = xr.open_dataset(Path(ra) / fa[ep]).load()
        db = xr.open_dataset(Path(rb) / fb[ep]).load()
        check_same_test_set(da, db, f"{lb} @ {ep}")
        if month_labels is None:
            month_labels = groupings(da)
        qa, qb = per_footprint(da, args.flux), per_footprint(db, args.flux)
        per_a[ep], per_b[ep], times = qa["mse"], qb["mse"], da.time.values
        row = {"epoch": ep}
        for tag, q in (("A", qa), ("B", qb)):
            t = tail(q["mse"])
            row.update({f"{s}_{tag}": t[s] for s in ("mean", "p90", "p95", "p99", "max", "worst10", "worst5", "worst1", "share10")})
            row[f"argmax_{tag}"] = int(np.argmax(q["mse"]))
            for name in ("month", "transport direction"):
                gm = group_means(q["mse"], month_labels[name])
                worst = max(gm, key=lambda g: g[2])
                row[f"worst_{name.split()[-1]}_{tag}"] = worst[2]
                row[f"worst_{name.split()[-1]}_name_{tag}"] = worst[0]
            row[f"mass_p95_{tag}"] = tail(q["mass"])["p95"]
            row[f"enh_p95_{tag}"] = tail(q["enh"])["p95"]
            row[f"enh_max_{tag}"] = tail(q["enh"])["max"]
        rows.append(row)
        da.close()
        db.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(f"{args.out_prefix}.csv", index=False)

    shown = [("mean", "mean loss"), ("p95", "95th percentile"), ("worst5", "mean of the worst 5 %"), ("worst1", "mean of the worst 1 %"),
             ("max", "maximum"), ("share10", "share of total in worst 10 %"), ("worst_month", "worst month (mean)"),
             ("worst_direction", "worst direction (mean)"),
             ("mass_p95", "|ln mass ratio|, 95th pct"), ("enh_p95", "enhancement error, 95th pct (ppb)")]
    print("\n### per checkpoint: A / B (A - B)")
    print(f"{'epoch':>5} | " + " | ".join(f"{name:>30}" for _, name in shown))
    for _, r in frame.iterrows():
        print(f"{int(r['epoch']):>5} | " + " | ".join(
            f"{r[k + '_A']:8.4f} / {r[k + '_B']:8.4f} ({r[k + '_A'] - r[k + '_B']:+.4f})".rjust(30) for k, _ in shown))
    late = frame[frame["epoch"] >= args.from_epoch]
    print(f"\n### summary over the {len(late)} checkpoints from epoch {args.from_epoch} (medians over the checkpoints; "
          f"'A lower' = number of checkpoints at which A is below B)")
    print(f"{'statistic':>36} | median A | median B | A - B | A lower | min A | min B")
    for k, name in shown:
        a, b = late[k + "_A"].values, late[k + "_B"].values
        print(f"{name:>36} | {np.median(a):8.4f} | {np.median(b):8.4f} | {np.median(a) - np.median(b):+.4f} | "
              f"{int((a < b).sum()):2d} of {len(a):2d} | {a.min():.4f} | {b.min():.4f}")
    for tag, lab in (("A", la), ("B", lb)):
        names = late[f"worst_month_name_{tag}"].value_counts()
        dirs = late[f"worst_direction_name_{tag}"].value_counts()
        print(f"  {lab}: worst month most often {names.index[0]} ({names.iloc[0]} of {len(late)}), worst direction most often "
              f"{dirs.index[0]} ({dirs.iloc[0]} of {len(late)})")
    averaged_view(args, la, lb, late["epoch"].astype(int).tolist(), per_a, per_b, month_labels, times, frame)
    figure_track(args, la, lb, frame)
    print(f"\nsaved {args.out_prefix}.csv, {args.out_prefix}_checkpoint_average_per_sample.csv and {args.out_prefix}.png")


def averaged_view(args, la, lb, late_epochs, per_a, per_b, groups, times, frame):
    """Every footprint's loss averaged over the late checkpoints: the comparison without the checkpoint-to-checkpoint
    swings, and the footprints that are persistently among the worst."""
    A = np.stack([per_a[e] for e in late_epochs])      # (checkpoints, footprints)
    B = np.stack([per_b[e] for e in late_epochs])
    xa, xb = A.mean(0), B.mean(0)
    n_ck, n = A.shape
    print(f"\n########## CHECKPOINT-AVERAGED view: each footprint's loss averaged over the {n_ck} checkpoints {late_epochs}")
    print(f"  (A = {la}, B = {lb}; the 95 % intervals are a paired bootstrap over the {n} test footprints: test-set uncertainty only)")
    print_tail_table(xa, xb, "footprint loss per footprint, averaged over the checkpoints", args.bootstrap)
    print_same_footprints(xa, xb)
    print_worst_groups(xa, xb, groups, full=True)

    # persistent worst footprints: in the worst 5 % of a run at two thirds of the checkpoints or more
    need = int(np.ceil(2 * n_ck / 3))
    in_a = (A >= np.percentile(A, 95, axis=1, keepdims=True)).sum(0)
    in_b = (B >= np.percentile(B, 95, axis=1, keepdims=True)).sum(0)
    pa, pb = in_a >= need, in_b >= need
    print(f"\n### persistent worst footprints (in the run's worst 5 % at {need} of the {n_ck} checkpoints or more)")
    print(f"  A: {int(pa.sum())} footprints, B: {int(pb.sum())}, in both: {int((pa & pb).sum())}; never in the worst 5 % of A: "
          f"{int((in_a == 0).sum())} of {n}, of B: {int((in_b == 0).sum())}")
    print(f"  on A's persistent worst: A {xa[pa].mean() if pa.any() else float('nan'):.4f}, B {xb[pa].mean() if pa.any() else float('nan'):.4f}; "
          f"on B's persistent worst: A {xa[pb].mean() if pb.any() else float('nan'):.4f}, B {xb[pb].mean() if pb.any() else float('nan'):.4f}")
    union = pa | pb
    if union.any():
        print(f"  what the {int(union.sum())} footprints persistent in either run are (share of them / share of the whole test set):")
        for name in ("season", "true extent", "true far-field share", "transport direction", "release latitude"):
            labels = groups[name]
            cells = []
            for g in sorted(set(labels)):
                share_p, share_all = (labels[union] == g).mean(), (labels == g).mean()
                if share_p >= 1.5 * share_all and (labels[union] == g).sum() >= 3:
                    cells.append(f"{g} {share_p:.2f} / {share_all:.2f}")
            print(f"    {name}: " + ("; ".join(cells) if cells else "no group over-represented by a factor of 1.5"))
    # the single worst footprint of every checkpoint
    arg_a = [int(frame.loc[frame["epoch"] == e, "argmax_A"].iloc[0]) for e in late_epochs]
    arg_b = [int(frame.loc[frame["epoch"] == e, "argmax_B"].iloc[0]) for e in late_epochs]
    stamp = lambda i: str(times[i])[:16]   # noqa: E731
    print(f"  the single worst footprint per checkpoint: A has {len(set(arg_a))} distinct footprints over {n_ck} checkpoints, "
          f"B {len(set(arg_b))}; the same footprint in both runs at {sum(a == b for a, b in zip(arg_a, arg_b))} checkpoint(s)")
    for tag, args_ in (("A", arg_a), ("B", arg_b)):
        counts = pd.Series(args_).value_counts()
        i = int(counts.index[0])
        print(f"    most frequent worst footprint of {tag}: {stamp(i)} ({counts.iloc[0]} of {n_ck} checkpoints; "
              f"{groups['season'][i]}, {groups['true extent'][i]}, transport {groups['transport direction'][i]}; "
              f"checkpoint-averaged loss A {xa[i]:.3f}, B {xb[i]:.3f})")
    pd.DataFrame({"time": times, "mse_avg_A": xa, "mse_avg_B": xb, "times_in_worst5_A": in_a, "times_in_worst5_B": in_b,
                  **{f"group_{k.replace(' ', '_')}": v for k, v in groups.items()}}
                 ).to_csv(f"{args.out_prefix}_checkpoint_average_per_sample.csv", index=False)


def figure_track(args, la, lb, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("mean", "Mean loss"), ("p95", "95th percentile"), ("worst5", "Mean of the worst 5 %"), ("max", "Worst single footprint")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.4), facecolor=COL_SURFACE, sharex=True)
    shown = frame[frame["epoch"] >= args.plot_from]
    for ax, (k, title) in zip(axes, panels):
        for tag, col, lab in (("A", COL_A, la), ("B", COL_B, lb)):
            ax.plot(shown["epoch"], shown[f"{k}_{tag}"], color=col, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=COL_SURFACE, markeredgewidth=2, solid_capstyle="round", label=lab)
        _style(ax, title, "epoch of the checkpoint", "footprint loss per footprint" if ax is axes[0] else "")
        ax.set_xlim(shown["epoch"].min() - 8, shown["epoch"].max() + 8)
    _figure_legend(fig, la, lb)
    fig.suptitle(args.title or f"Mean and worst-case footprint error through training: {la} vs {lb}", fontsize=12,
                 color=COL_TEXT, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 0.985, 0.90))
    fig.savefig(f"{args.out_prefix}.png", dpi=150, facecolor=COL_SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("snapshot", help="tail statistics of one prediction file per run")
    s.add_argument("out_prefix")
    s.add_argument("runs", nargs=2, help="<label>=<run_dir>::<file.nc>, twice (A then B)")
    s.add_argument("--yardstick", nargs=2, default=None, help="two runs that differ in their seed alone")
    s.add_argument("--bootstrap", type=int, default=2000)
    t = sub.add_parser("track", help="the statistics at every common numbered checkpoint")
    t.add_argument("out_prefix")
    t.add_argument("runs", nargs=2, help="<label>=<run_dir>, twice (A then B)")
    t.add_argument("--from-epoch", type=int, default=50, help="first checkpoint of the summary (default 50)")
    t.add_argument("--plot-from", type=int, default=25, help="first checkpoint drawn (default 25: epoch 0 is off the scale)")
    t.add_argument("--bootstrap", type=int, default=2000)
    for p in (s, t):
        p.add_argument("--title", default="")
        p.add_argument("--flux", type=float, default=1e-8, help="uniform flux of the enhancement, mol m-2 s-1")
    args = ap.parse_args()
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    (snapshot if args.mode == "snapshot" else track)(args)


if __name__ == "__main__":
    sys.exit(main())
