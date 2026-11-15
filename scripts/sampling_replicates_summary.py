"""Two sampling rules, several replicate runs each: is one better on the mean or on the WORST-CASE error?

    python scripts/sampling_replicates_summary.py <out_prefix> --label-a "month-balanced" --label-b "evenly spaced" \
        --pair <track_prefix>::<run_dir_A>::<run_dir_B> [--pair ...] [--from-epoch 50] [--title "..."] \
        [--pool-directions <pool_directions.csv>]

Every ``--pair`` is one replicate of the two rules (A = first rule, B = second rule):

* ``<track_prefix>`` is the output prefix of ``scripts/worst_case_error.py track`` for the pair: ``<prefix>.csv`` holds the
  statistics of both runs at every common checkpoint, ``<prefix>_checkpoint_average_per_sample.csv`` each test footprint's
  loss averaged over the late checkpoints, with its groups (month, season, extent, far share, direction, latitude);
* the two run directories give the per-epoch test losses of the training logs (``*_updates.txt``) and the selected training
  samples (``training_outputs/sample_selection_*.csv``).

Printed (and written to ``<out_prefix>_runs.csv`` / ``<out_prefix>_months.csv`` / ``<out_prefix>.png``):

1. every run's statistics: the footprint loss over the epochs, the tail of the checkpoint-averaged per-footprint loss, the
   worst groups, the per-checkpoint tail, the background error;
2. the two rules side by side: the replicate values, their means, the difference, the run-to-run standard deviation of each
   rule and the SEPARATION verdict - "A lower in all" when every run of A is below every run of B (for three runs against
   three the chance of that under exchangeable runs is 1 in 20), "B lower in all", else "overlap";
3. the calendar-month profile: training samples per month under each rule, and the test loss per month;
4. with ``--pool-directions`` (the table ``smoke_tests_zarr/check_sampling_selection.py`` writes: the transport direction of
   every pool footprint) the same profile per TRANSPORT DIRECTION (``<out_prefix>_directions.csv``), which then also fills
   the lower row of the figure.

Reading: with a handful of runs the separation verdict is the only test used; a difference whose replicates overlap is not
a finding, whatever the size of the mean difference. Needs numpy / pandas / matplotlib: run inside the container (light).
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worst_case_error import COL_A, COL_B, COL_GRID, COL_MUTED, COL_SURFACE, COL_TEXT, _figure_legend, _style, tail  # noqa: E402

LOG_LINE = re.compile(r"Epoch (\d+) \(\w+\), total [\d.eE+-]+/[\d.eE+-]+, fp ([\d.eE+-]+)/([\d.eE+-]+), "
                      r"bg ([\d.eE+-]+)/([\d.eE+-]+)(?:, bg MAE\(denorm\) ([\d.eE+-]+))?")
MONTHS = [f"month {m:02d}" for m in range(1, 13)]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "near release"]   # the groups of worst_case_error.py, in compass order
# (key, label, format); every statistic is "lower is better"
ROWS = [
    ("loss_mean", "test loss, mean over the epochs", "{:.4f}"),
    ("loss_median", "test loss, median over the epochs", "{:.4f}"),
    ("loss_floor25", "test loss, lowest 25-epoch mean (floor)", "{:.4f}"),
    ("loss_best", "test loss, best single epoch", "{:.4f}"),
    ("loss_high10", "test loss, highest 10-epoch mean", "{:.4f}"),
    ("loss_sd", "test loss, s.d. over the epochs", "{:.4f}"),
    ("mean", "per-footprint loss: mean", "{:.4f}"),
    ("macro_month", "per-footprint loss: mean of the 12 monthly means", "{:.4f}"),
    ("p95", "per-footprint loss: 95th percentile", "{:.4f}"),
    ("worst10", "per-footprint loss: mean of the worst 10 %", "{:.4f}"),
    ("worst5", "per-footprint loss: mean of the worst 5 %", "{:.4f}"),
    ("worst1", "per-footprint loss: mean of the worst 1 %", "{:.4f}"),
    ("max", "per-footprint loss: single worst footprint", "{:.4f}"),
    ("share10", "share of the total in the worst 10 %", "{:.4f}"),
    ("worst_month", "worst calendar month (mean loss)", "{:.4f}"),
    ("worst_season", "worst season (mean loss)", "{:.4f}"),
    ("worst_direction", "worst transport direction (mean loss)", "{:.4f}"),
    ("worst_extent", "worst extent quartile (mean loss)", "{:.4f}"),
    ("DJF", "DJF (mean loss)", "{:.4f}"),
    ("JJA", "JJA (mean loss)", "{:.4f}"),
    ("ck_worst5", "per checkpoint: mean of the worst 5 %", "{:.4f}"),
    ("ck_max", "per checkpoint: single worst footprint", "{:.3f}"),
    ("ck_share10", "per checkpoint: share in the worst 10 %", "{:.4f}"),
    ("ck_mass_p95", "per checkpoint: |ln mass ratio|, 95th pct", "{:.3f}"),
    ("ck_enh_p95", "per checkpoint: enhancement error, 95th pct (ppb)", "{:.2f}"),
    ("bg_median_ppb", "background error, median over the epochs (ppb)", "{:.2f}"),
    ("bg_best_ppb", "background error, best epoch (ppb)", "{:.2f}"),
]


# ---------------------------------------------------------------
# Reading
# ---------------------------------------------------------------

def read_log(run_dir):
    """Per-epoch losses of a run from its ``*_updates.txt``: DataFrame (epoch, train_fp, test_fp, test_bg, bg_ppb)."""
    paths = glob.glob(str(Path(run_dir) / "*_updates.txt"))
    if len(paths) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one *_updates.txt, found {len(paths)}")
    rows = []
    with open(paths[0]) as handle:
        for line in handle:
            m = LOG_LINE.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(5)),
                             float(m.group(6)) * 1e9 if m.group(6) else np.nan))
    if not rows:
        raise ValueError(f"{paths[0]}: no epoch line")
    return pd.DataFrame(rows, columns=["epoch", "train_fp", "test_fp", "test_bg", "bg_ppb"]).drop_duplicates("epoch").sort_values("epoch")


def running_mean_extreme(x, width, fn):
    """Lowest (``fn=min``) or highest (``fn=max``) mean of ``width`` consecutive values; NaN if the series is shorter."""
    x = np.asarray(x, dtype=float)
    if len(x) < width:
        return np.nan
    means = np.convolve(x, np.ones(width) / width, mode="valid")
    return float(fn(means))


def loss_statistics(log, from_epoch):
    """Statistics of the per-epoch test footprint loss (and the background error) of one run."""
    late = log[log["epoch"] >= from_epoch]
    if late.empty:
        raise ValueError(f"no epoch from {from_epoch} in the log")
    out = {"epochs": int(log["epoch"].max()),
           "loss_mean": late["test_fp"].mean(), "loss_median": late["test_fp"].median(), "loss_sd": late["test_fp"].std(),
           "loss_floor25": running_mean_extreme(log["test_fp"], 25, min), "loss_best": log["test_fp"].min(),
           "loss_best_epoch": int(log.loc[log["test_fp"].idxmin(), "epoch"]),
           "loss_low10": running_mean_extreme(late["test_fp"], 10, min),
           "loss_high10": running_mean_extreme(late["test_fp"], 10, max),
           "train_fp_end": float(log["train_fp"].iloc[-1]),
           "bg_median_ppb": late["bg_ppb"].median(), "bg_best_ppb": log["bg_ppb"].min()}
    return out


def tail_statistics(sample, tag):
    """Tail and group statistics of one run from the checkpoint-averaged per-footprint losses (column ``mse_avg_<tag>``)."""
    x = sample[f"mse_avg_{tag}"].to_numpy(dtype=float)
    out = dict(tail(x))
    by_month = sample.groupby("group_month")[f"mse_avg_{tag}"].mean()
    for name in MONTHS:
        out[name] = float(by_month.get(name, np.nan))
    present = by_month.reindex([m for m in MONTHS if m in by_month.index])
    out["macro_month"] = float(present.mean())
    out["worst_month"] = float(present.max())
    out["worst_month_name"] = str(present.idxmax())
    by_season = sample.groupby("group_season")[f"mse_avg_{tag}"].mean()
    for name in SEASONS:
        out[name] = float(by_season.get(name, np.nan))
    out["worst_season"] = float(by_season.max())
    out["worst_season_name"] = str(by_season.idxmax())
    for key, column in (("worst_direction", "group_transport_direction"), ("worst_extent", "group_true_extent")):
        groups = sample.groupby(column)[f"mse_avg_{tag}"].mean()
        out[key] = float(groups.max())
        out[key + "_name"] = str(groups.idxmax())
    by_direction = sample.groupby("group_transport_direction")[f"mse_avg_{tag}"].mean()
    for name in DIRECTIONS:
        out["direction " + name] = float(by_direction.get(name, np.nan))
    return out


def checkpoint_statistics(frame, tag, from_epoch):
    """Means over the late checkpoints of the per-checkpoint tail statistics of one run."""
    late = frame[frame["epoch"] >= from_epoch]
    out = {"checkpoints": int(len(late))}
    for key in ("worst5", "max", "share10", "mass_p95", "enh_p95"):
        out["ck_" + key] = float(late[f"{key}_{tag}"].mean())
    return out


def training_months(run_dir):
    """Selected training samples per calendar month (12 counts), or None if the run wrote no selection file."""
    paths = glob.glob(str(Path(run_dir) / "training_outputs" / "sample_selection_*.csv"))
    if not paths:
        return None
    month = pd.DatetimeIndex(pd.read_csv(paths[0], usecols=["time"])["time"]).month.values
    return np.bincount(month, minlength=13)[1:]


def training_directions(run_dir, pool_directions):
    """Selected training samples per transport direction (counts in the order of DIRECTIONS), from the run's selection file
    and the direction of every pool footprint (Series indexed by the position in the pool); None without a selection file."""
    paths = glob.glob(str(Path(run_dir) / "training_outputs" / "sample_selection_*.csv"))
    if not paths or pool_directions is None:
        return None
    picked = pool_directions.reindex(pd.read_csv(paths[0], usecols=["pool_index"])["pool_index"].to_numpy())
    if picked.isna().any():
        raise ValueError(f"{paths[0]}: {int(picked.isna().sum())} selected samples are not in the pool-direction table")
    counts = picked.value_counts()
    return np.array([int(counts.get(name, 0)) for name in DIRECTIONS])


def run_statistics(track_prefix, run_dir, tag, from_epoch):
    """All statistics of one run of a pair (``tag`` = "A" or "B")."""
    frame = pd.read_csv(f"{track_prefix}.csv")
    sample = pd.read_csv(f"{track_prefix}_checkpoint_average_per_sample.csv")
    out = {"run": Path(run_dir).name}
    out.update(loss_statistics(read_log(run_dir), from_epoch))
    out.update(tail_statistics(sample, tag))
    out.update(checkpoint_statistics(frame, tag, from_epoch))
    return out


# ---------------------------------------------------------------
# The two rules side by side
# ---------------------------------------------------------------

def separation(a, b):
    """"A lower in all" if every value of ``a`` is below every value of ``b``, "B lower in all" for the reverse, else
    "overlap". For n against m exchangeable runs the chance of a given complete separation is 1 / C(n + m, n)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return "n/a"
    if a.max() < b.min():
        return "A lower in all"
    if b.max() < a.min():
        return "B lower in all"
    return "overlap"


def compare(runs_a, runs_b, key):
    """Replicate values of one statistic under the two rules, with the mean difference and the verdict."""
    a = np.array([r[key] for r in runs_a], dtype=float)
    b = np.array([r[key] for r in runs_b], dtype=float)
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan   # noqa: E731
    paired = a - b if len(a) == len(b) else np.array([])
    verdict = separation(a, b)
    if len(a) == 1 and len(b) == 1:      # one run per rule: which run is lower, NOT a test
        verdict = {"A lower in all": "A lower (one run)", "B lower in all": "B lower (one run)"}.get(verdict, "equal (one run)")
    return {"a": a, "b": b, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(a.mean() - b.mean()),
            "sd_a": sd(a), "sd_b": sd(b), "a_lower_pairs": int((paired < 0).sum()), "pairs": int(len(paired)),
            "verdict": verdict}


def print_runs(runs_a, runs_b, la, lb):
    print("\n### every run (A = %s, B = %s)" % (la, lb))
    head = f"{'statistic':>52} | " + " | ".join(f"{'A r' + str(i + 1):>9}" for i in range(len(runs_a))) + " | " \
        + " | ".join(f"{'B r' + str(i + 1):>9}" for i in range(len(runs_b)))
    print(head)
    for key, label, fmt in ROWS:
        cells = [fmt.format(r[key]) for r in runs_a] + [fmt.format(r[key]) for r in runs_b]
        print(f"{label:>52} | " + " | ".join(f"{c:>9}" for c in cells))
    for key, label in (("loss_best_epoch", "epoch of the best single epoch"), ("worst_month_name", "worst month"),
                       ("worst_season_name", "worst season"), ("worst_direction_name", "worst direction"),
                       ("worst_extent_name", "worst extent quartile")):
        cells = [str(r[key]).replace("month ", "").split(" (")[0] for r in runs_a + runs_b]
        print(f"{label:>52} | " + " | ".join(f"{c:>9}" for c in cells))
    print("  runs A: " + "; ".join(f"r{i + 1} = {r['run']}" for i, r in enumerate(runs_a)))
    print("  runs B: " + "; ".join(f"r{i + 1} = {r['run']}" for i, r in enumerate(runs_b)))


def print_comparison(runs_a, runs_b, la, lb):
    n, m = len(runs_a), len(runs_b)
    from math import comb
    if n == 1 and m == 1:
        print(f"\n### the two rules (A = {la}, B = {lb}; ONE run per rule: the last column only says which run is lower, it is NOT a "
              "test - a difference has to be read against the spread between two runs of one rule; A - B < 0 = A better)")
    else:
        print(f"\n### the two rules (A = {la}, B = {lb}; {n} against {m} runs; 'lower in all' = complete separation of the replicates, "
              f"chance 1 in {comb(n + m, n)} for exchangeable runs; A - B < 0 = A better)")
    print(f"{'statistic':>52} | {'mean A':>8} | {'mean B':>8} | {'A - B':>8} | {'s.d. A':>7} | {'s.d. B':>7} | {'A lower':>8} | verdict")
    rows = []
    for key, label, fmt in ROWS:
        c = compare(runs_a, runs_b, key)
        print(f"{label:>52} | {fmt.format(c['mean_a']):>8} | {fmt.format(c['mean_b']):>8} | {c['diff']:>+8.4f} | "
              f"{c['sd_a']:>7.4f} | {c['sd_b']:>7.4f} | {c['a_lower_pairs']:>3} of {c['pairs']:<2} | {c['verdict']}")
        rows.append({"statistic": key, "label": label, **{k: v for k, v in c.items() if k not in ("a", "b")},
                     **{f"A_r{i + 1}": v for i, v in enumerate(c["a"])}, **{f"B_r{i + 1}": v for i, v in enumerate(c["b"])}})
    return pd.DataFrame(rows)


def group_profile(runs_a, runs_b, train_a, train_b, keys, labels, test_counts, column):
    """Per group (calendar month or transport direction): training samples under each rule (mean of the replicates), test
    footprints, test loss of the replicates of each rule, the difference and in how many replicate pairs A is lower.

    ``keys`` name the group in the run statistics and in ``test_counts``, ``labels`` in the output column ``column``;
    ``train_a`` / ``train_b`` hold one count array per run (or None)."""
    rows = []
    for j, (key, label) in enumerate(zip(keys, labels)):
        a = np.array([r[key] for r in runs_a], dtype=float)
        b = np.array([r[key] for r in runs_b], dtype=float)
        paired = a - b if len(a) == len(b) else np.array([])
        verdict = separation(a, b)
        if len(a) == 1 and len(b) == 1:      # one run per rule: which run is lower, NOT a test
            verdict = {"A lower in all": "A lower (one run)", "B lower in all": "B lower (one run)"}.get(verdict, "equal (one run)")
        rows.append({column: label, "n_test": int(test_counts.get(key, 0)),
                     "train_A": np.mean([t[j] for t in train_a if t is not None]) if any(t is not None for t in train_a) else np.nan,
                     "train_B": np.mean([t[j] for t in train_b if t is not None]) if any(t is not None for t in train_b) else np.nan,
                     "loss_A": a.mean(), "loss_A_min": a.min(), "loss_A_max": a.max(),
                     "loss_B": b.mean(), "loss_B_min": b.min(), "loss_B_max": b.max(),
                     "diff": a.mean() - b.mean(), "a_lower_pairs": int((paired < 0).sum()), "verdict": verdict})
    return pd.DataFrame(rows)


def month_profile(runs_a, runs_b, dirs_a, dirs_b, test_counts):
    """The profile over the twelve calendar months (see :func:`group_profile`)."""
    return group_profile(runs_a, runs_b, [training_months(d) for d in dirs_a], [training_months(d) for d in dirs_b],
                         MONTHS, list(range(1, 13)), test_counts, "month")


def direction_profile(runs_a, runs_b, dirs_a, dirs_b, test_counts, pool_directions):
    """The profile over the nine transport directions (see :func:`group_profile`); ``test_counts`` by direction name."""
    counts = {"direction " + name: int(test_counts.get(name, 0)) for name in DIRECTIONS}
    return group_profile(runs_a, runs_b, [training_directions(d, pool_directions) for d in dirs_a],
                         [training_directions(d, pool_directions) for d in dirs_b],
                         ["direction " + name for name in DIRECTIONS], DIRECTIONS, counts, "direction")


def print_profile(profile, la, lb, pairs, column="month", title="calendar months"):
    what = column
    width = max(5, max(len(str(v)) for v in profile[column]))
    print(f"\n### {title} (A = {la}, B = {lb}; training samples per {what} = mean of the replicates; test loss = each "
          f"footprint's loss averaged over the late checkpoints, then the {what} mean, then the mean [min..max] of the replicates)")
    print(f"{what:>{width}} | {'test n':>6} | {'train A':>7} | {'train B':>7} | {'A / B':>6} | {'loss A [min..max]':>26} | "
          f"{'loss B [min..max]':>26} | {'A - B':>8} | A lower | verdict")
    for _, r in profile.iterrows():
        ratio = r["train_A"] / r["train_B"] if np.isfinite(r["train_B"]) and r["train_B"] > 0 else np.nan
        print(f"{str(r[column]):>{width}} | {r['n_test']:>6} | {r['train_A']:>7.0f} | {r['train_B']:>7.0f} | {ratio:>6.2f} | "
              f"{r['loss_A']:>8.4f} [{r['loss_A_min']:.4f}..{r['loss_A_max']:.4f}] | {r['loss_B']:>8.4f} [{r['loss_B_min']:.4f}..{r['loss_B_max']:.4f}] | "
              f"{r['diff']:>+8.4f} | {r['a_lower_pairs']} of {pairs} | {r['verdict']}")
    gained = profile[profile["train_A"] > profile["train_B"] * 1.05]
    lost = profile[profile["train_A"] < profile["train_B"] * 0.95]
    for name, part in ((f"{what}s with MORE training samples under A", gained), (f"{what}s with FEWER training samples under A", lost)):
        part = part[(part["n_test"] > 0) & part["diff"].notna()]     # a group without test footprints has no loss
        if len(part):
            weights = part["n_test"].to_numpy(dtype=float)
            print(f"  {name} ({', '.join(str(m) for m in part[column])}): test-weighted A - B = "
                  f"{np.average(part['diff'], weights=weights):+.4f}; A lower in {int((part['diff'] < 0).sum())} of {len(part)} {what}s")
    usable = profile[(profile["train_A"] > 0) & (profile["train_B"] > 0) & profile["diff"].notna()]
    if len(usable) >= 3:
        change = np.log(usable["train_A"] / usable["train_B"])
        if change.std() > 0:
            print(f"  correlation over the {len(usable)} {what}s between ln(train A / train B) and the loss difference A - B: "
                  f"{np.corrcoef(change, usable['diff'])[0, 1]:+.2f} (negative = the {what}s that gained samples improved)")
            slope = np.polyfit(change, usable["diff"], 1)[0]
            print(f"  least-squares slope: {slope:+.4f} loss per unit of ln(sample ratio), i.e. {slope * np.log(2):+.4f} per doubling of a "
                  f"{what}'s training samples")


# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------

def figure(out_prefix, title, la, lb, runs_a, runs_b, profile, column="month"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = "calendar month" if column == "month" else "transport direction"
    panels = [("loss_mean", "Mean test loss over the epochs"), ("worst5", "Mean of the worst 5 % of footprints"),
              ("worst_month", "Worst calendar month") if column == "month" else ("worst_direction", "Worst transport direction"),
              ("max", "Single worst footprint")]
    fig = plt.figure(figsize=(16.5, 8.6), facecolor=COL_SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05])
    marker = dict(marker="o", markersize=8, markeredgecolor=COL_SURFACE, markeredgewidth=2, linestyle="none")
    for j, (key, name) in enumerate(panels):
        ax = fig.add_subplot(grid[0, j])
        a = np.array([r[key] for r in runs_a], dtype=float)
        b = np.array([r[key] for r in runs_b], dtype=float)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        pairs = min(len(a), len(b))
        for i in range(pairs):                                     # the replicate pairs, joined by a hairline
            ax.plot([0, 1], [a[i], b[i]], color=COL_GRID, linewidth=1.2, zorder=1)
        nudge, last = {}, None                                     # replicate tags; two tags closer than 7 % of the range are spread
        for i in sorted(range(pairs), key=lambda i: b[i]):
            nudge[i] = 0 if last is None or b[i] - b[last] > 0.07 * (hi - lo) else nudge[last] + 10
            last = i
        single = len(a) == 1 and len(b) == 1                       # one run per rule: the dot IS the value, no tag, no mean
        for i in range(pairs if not single else 0):
            ax.annotate(f"r{i + 1}", (1, b[i]), xytext=(9, nudge[i]), textcoords="offset points", fontsize=8.5,
                        color=COL_MUTED, va="center")
        ax.plot(np.zeros(len(a)), a, color=COL_A, zorder=3, **marker)
        ax.plot(np.ones(len(b)), b, color=COL_B, zorder=3, **marker)
        for x, v, col in (() if single else ((0, a.mean(), COL_A), (1, b.mean(), COL_B))):   # the mean of the replicates
            ax.plot([x - 0.17, x + 0.17], [v, v], color=col, linewidth=2, solid_capstyle="round", zorder=2)
        _style(ax, name, "", "footprint loss" if j == 0 else "")
        ax.set_xlim(-0.55, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([la, lb], color=COL_TEXT, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        pad = 0.18 * (hi - lo) if hi > lo else 0.05 * max(abs(hi), 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    x = np.arange(len(profile), dtype=float)
    ticks = [str(v) for v in profile[column]]
    ax = fig.add_subplot(grid[1, :2])
    for name, col in (("train_A", COL_A), ("train_B", COL_B)):
        ax.plot(x, profile[name], color=col, linewidth=2, marker="o", markersize=7, markeredgecolor=COL_SURFACE,
                markeredgewidth=2, solid_capstyle="round")
    _style(ax, f"Training samples per {group}", group, "samples in the training set")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, None)
    ax = fig.add_subplot(grid[1, 2:])
    for key, col, shift in (("loss_A", COL_A, -0.07), ("loss_B", COL_B, 0.07)):
        ax.vlines(x + shift, profile[key + "_min"], profile[key + "_max"], color=col, linewidth=1.4, alpha=0.55)
        ax.plot(x + shift, profile[key], color=col, linewidth=2, marker="o", markersize=7, markeredgecolor=COL_SURFACE,
                markeredgewidth=2, solid_capstyle="round")
    one_run = len(runs_a) == 1 and len(runs_b) == 1
    _style(ax, f"Test loss per {group}" + ("" if one_run else " (dot = mean of the replicates, bar = their range)"),
           group, "footprint loss")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    _figure_legend(fig, la, lb)
    fig.suptitle(title or f"{la} vs {lb}: replicate runs", fontsize=12, color=COL_TEXT, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 0.99, 0.93), h_pad=2.2)
    fig.savefig(f"{out_prefix}.png", dpi=150, facecolor=COL_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_prefix")
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--pair", action="append", required=True, help="<track_prefix>::<run_dir_A>::<run_dir_B> (one per replicate)")
    ap.add_argument("--from-epoch", type=int, default=50, help="first epoch / checkpoint of the summaries (default 50); use the "
                    "value the tracks were made with")
    ap.add_argument("--title", default="")
    ap.add_argument("--pool-directions", default=None, help="CSV with the columns pool_index and direction (the table "
                    "check_sampling_selection.py writes): adds the profile per transport direction")
    args = ap.parse_args(argv)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    runs_a, runs_b, dirs_a, dirs_b, test_counts, direction_counts = [], [], [], [], None, None
    for spec in args.pair:
        parts = spec.split("::")
        if len(parts) != 3:
            raise SystemExit(f"--pair needs <track_prefix>::<run_dir_A>::<run_dir_B>, got {spec!r}")
        prefix, dir_a, dir_b = parts
        runs_a.append(run_statistics(prefix, dir_a, "A", args.from_epoch))
        runs_b.append(run_statistics(prefix, dir_b, "B", args.from_epoch))
        dirs_a.append(dir_a)
        dirs_b.append(dir_b)
        counts = pd.read_csv(f"{prefix}_checkpoint_average_per_sample.csv", usecols=["group_month"])["group_month"].value_counts()
        if test_counts is not None and not counts.sort_index().equals(test_counts.sort_index()):
            raise SystemExit(f"{prefix}: another test set than the first pair")
        test_counts = counts
        direction_counts = pd.read_csv(f"{prefix}_checkpoint_average_per_sample.csv",
                                       usecols=["group_transport_direction"])["group_transport_direction"].value_counts()
    la, lb = args.label_a, args.label_b
    print(f"########## {la} (A) against {lb} (B): {len(runs_a)} replicate pair(s), epochs and checkpoints from {args.from_epoch}; "
          f"{int(test_counts.sum())} test footprints; checkpoints per run: {sorted({r['checkpoints'] for r in runs_a + runs_b})}")
    print_runs(runs_a, runs_b, la, lb)
    table = print_comparison(runs_a, runs_b, la, lb)
    profile = month_profile(runs_a, runs_b, dirs_a, dirs_b, test_counts)
    print_profile(profile, la, lb, len(runs_a))
    directions = None
    if args.pool_directions:
        pool = pd.read_csv(args.pool_directions, usecols=["pool_index", "direction"]).set_index("pool_index")["direction"]
        directions = direction_profile(runs_a, runs_b, dirs_a, dirs_b, direction_counts, pool)
        print_profile(directions, la, lb, len(runs_a), "direction",
                      "transport directions (the compass sector in which the centre of mass of the TRUE footprint lies as seen "
                      "from the release point; pool share of the directions: "
                      + ", ".join(f"{n} {100 * float((pool == n).mean()):.1f} %" for n in DIRECTIONS) + ")")
    frame = pd.DataFrame([{"rule": "A", "label": la, "replicate": i + 1, **r} for i, r in enumerate(runs_a)]
                         + [{"rule": "B", "label": lb, "replicate": i + 1, **r} for i, r in enumerate(runs_b)])
    frame.to_csv(f"{args.out_prefix}_runs.csv", index=False)
    table.to_csv(f"{args.out_prefix}_comparison.csv", index=False)
    profile.to_csv(f"{args.out_prefix}_months.csv", index=False)
    if directions is not None:
        directions.to_csv(f"{args.out_prefix}_directions.csv", index=False)
        figure(args.out_prefix, args.title, la, lb, runs_a, runs_b, directions, "direction")
    else:
        figure(args.out_prefix, args.title, la, lb, runs_a, runs_b, profile)
    print(f"\nsaved {args.out_prefix}_runs.csv, {args.out_prefix}_comparison.csv, {args.out_prefix}_months.csv and {args.out_prefix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
