"""Summarise the SSL met-pretraining fine-tune study from *_updates.txt logs (+ pretrain logs).

    python scripts/compare_ssl_finetune.py <out.png> <label>=<run_dir> [<label>=<run_dir> ...]

The first run is the reference. Prints per-head best / final / last-10-mean test losses, the
epoch at which each run first reaches the reference's best fp (sample efficiency), and the
early-epoch fp losses; saves test fp / bg curves.
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
from compare_multigpu_run import read_run

out = sys.argv[1]
runs = [(a.split("=", 1)[0], read_run(a.split("=", 1)[1])) for a in sys.argv[2:]]
ref_best = runs[0][1]["test_fp"].min()
print(f"{'run':>18} | fp best@ep | fp last10 | fp@5  fp@10 fp@25 fp@50 | ep to ref-best fp | bg best@ep | bg last10 | bg gap end")
for label, r in runs:
    fp, bg = r["test_fp"], r["test_bg"]
    hit = np.nonzero(fp <= ref_best)[0]
    early = " ".join(f"{fp[e]:.3f}" if e < len(fp) else "  -  " for e in (5, 10, 25, 50))
    print(f"{label:>18} | {fp.min():.4f}@{int(fp.argmin()):3d} | {fp[-10:].mean():.4f}    | {early} | "
          f"{(int(hit[0]) if len(hit) else 'never'):>10} | {bg.min():.4f}@{int(bg.argmin()):3d} | "
          f"{bg[-10:].mean():.4f}    | {bg[-1] - r['train_bg'][-1]:+.4f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
colors = ["#52514e", "#9a9994", "#2a78d6", "#eb6834", "#1f9d6b", "#a04fc9", "#c9a227"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#fcfcfb")
for ax, key, name in zip(axes, ("test_fp", "test_bg"), ("test footprint loss (MSE, log-scaled fp)", "test background loss (normalised MSE)")):
    ax.set_facecolor("#fcfcfb")
    for (label, r), c in zip(runs, colors):
        y = r[key]
        ax.plot(r["epoch"], y, color=c, lw=0.8, alpha=0.35)
        k = 5
        ax.plot(r["epoch"][k - 1:], np.convolve(y, np.ones(k) / k, "valid"), color=c, lw=2,
                label=f"{label} (best {y.min():.4f})")
    lo = min(r[key].min() for _, r in runs)
    ax.set_ylim(lo - 0.01, lo + (0.12 if key == "test_fp" else 0.1))
    ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.grid(color="#e6e5e1", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Trunk initialisation: random vs self-supervised met pretraining (thin = raw, thick = 5-epoch mean)", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("saved", out)
