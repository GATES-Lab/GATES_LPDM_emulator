"""Compare a multi-GPU dual-head run with its single-GPU reference from their *_updates.txt logs.

Prints the per-head best test losses / epochs / wall time and saves a 2x2 figure
(test fp and bg loss, against epoch and against wall-clock hours).

    python scripts/compare_multigpu_run.py <ref_run_dir> <multigpu_run_dir> <out.png> [ref_label] [new_label]
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EPOCH_RE = re.compile(r"^(\S+ \S+) Epoch (\d+) \((\w+)\), total ([\d.]+)/([\d.]+), "
                      r"fp ([\d.]+)/([\d.]+), bg ([\d.]+)/([\d.]+)")


def read_run(run_dir):
    path = next(Path(run_dir).glob("*_updates.txt"))
    lines = path.read_text().splitlines()
    start = datetime.strptime(" ".join(lines[0].split()[:2]), "%d/%m/%y %H:%M:%S")
    rows = []
    for line in lines:
        m = EPOCH_RE.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%d/%m/%y %H:%M:%S")
            rows.append([int(m.group(2)), (t - start).total_seconds() / 3600]
                        + [float(m.group(i)) for i in range(4, 10)])
    a = np.array(rows)
    return {"epoch": a[:, 0], "hours": a[:, 1], "train_fp": a[:, 4], "test_fp": a[:, 5],
            "train_bg": a[:, 6], "test_bg": a[:, 7]}


def summarise(label, r):
    i_fp, i_bg = r["test_fp"].argmin(), r["test_bg"].argmin()
    sec = np.diff(r["hours"]).mean() * 3600
    print(f"{label:>22}: epochs {len(r['epoch'])}, {sec:6.1f} s/epoch, total {r['hours'][-1]:.2f} h | "
          f"fp best {r['test_fp'][i_fp]:.4f}@{int(r['epoch'][i_fp])} (final {r['test_fp'][-1]:.4f}, "
          f"last-10 mean {r['test_fp'][-10:].mean():.4f}) | bg best {r['test_bg'][i_bg]:.4f}@"
          f"{int(r['epoch'][i_bg])} (final {r['test_bg'][-1]:.4f}, train-test gap at end "
          f"{r['test_bg'][-1] - r['train_bg'][-1]:+.4f})")


if __name__ == "__main__":
    ref_dir, new_dir, out = sys.argv[1:4]
    labels = (sys.argv[4:6] + ["1 GPU (reference)", "multi-GPU"])[:2] if len(sys.argv) >= 6 else ["1 GPU (reference)", "multi-GPU"]
    runs = [read_run(ref_dir), read_run(new_dir)]
    for label, r in zip(labels, runs):
        summarise(label, r)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834"]  # categorical slots 1, 2 (fixed order: reference, new)
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#fcfcfb", sharey="row")
    for row, (key, name) in enumerate([("test_fp", "test footprint loss"), ("test_bg", "test background loss")]):
        for col, (xkey, xname) in enumerate([("epoch", "epoch"), ("hours", "wall-clock hours since run start")]):
            ax = axes[row, col]
            ax.set_facecolor("#fcfcfb")
            for k, (label, r, c) in enumerate(zip(labels, runs, colors)):
                ax.plot(r[xkey], r[key], color=c, lw=2, label=label)
                i = r[key].argmin()
                ax.plot(r[xkey][i], r[key][i], "o", ms=8, color=c, mec="#fcfcfb", mew=2)
                ax.annotate(f"{r[key][i]:.3f}", (r[xkey][i], r[key][i]), textcoords="offset points",
                            xytext=(0, 9 if k == 0 else -14), ha="center", fontsize=9, color=ink)
            lo = min(r[key].min() for r in runs)
            ax.set_ylim(lo - 0.02, lo + (0.25 if key == "test_fp" else 0.2))
            ax.set_xlabel(xname, color=muted)
            if col == 0:
                ax.set_ylabel(name, color=muted)
            ax.grid(color=grid, lw=0.8)
            ax.tick_params(colors=muted)
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].legend(frameon=False, labelcolor=ink)
    fig.suptitle("Multi-GPU vs single-GPU: same parameters, effective batch 20 vs 5 (dots = best epoch)",
                 color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("saved", out)
