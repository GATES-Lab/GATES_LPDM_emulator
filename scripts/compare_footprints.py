"""Compare the predicted TEST footprints of two dual-head runs against the same truth.

    python scripts/compare_footprints.py <out_prefix> <labelA>=<runA_dir> <labelB>=<runB_dir> \
        [--file sample_predictions_test_best_fp.nc]

Each run directory must hold the ``sample_predictions_test.nc`` written at the end of training
(variables fp_original / fp_pred in linear units, fp_transformed / fp_transformed_pred in the
log-scaled training space, fp_nan_mask with 1 = invalid).

Reports, for each run: the training-space MSE, linear-space footprint metrics
(``gates.evaluation.metrics``), mass conservation, the near-field / far-field split of the
footprint (the release point sits at the centre of the window), bias by true-magnitude decile,
and mole-fraction metrics for static flux patterns. Saves two figures: example footprints and
the diagnostic curves.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from gates.evaluation.metrics import compute_footprint_metrics, compute_static_mf_metrics

CENTRE = None  # set from the grid size


def load(run_dir, fname="sample_predictions_test.nc"):
    d = xr.open_dataset(Path(run_dir) / fname).load()
    print(f"{Path(run_dir).name}/{fname}: {d.attrs.get('checkpoint', 'final epoch')}"
          + (f", fp loss {d.attrs['test_fp_loss']:.4f}" if "test_fp_loss" in d.attrs else ""))
    return d


def radial_index(h, w):
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def main():
    argv = sys.argv[1:]
    fname = "sample_predictions_test.nc"
    if "--file" in argv:
        i = argv.index("--file")
        fname = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    out_prefix = argv[0]
    runs = [(a.split("=", 1)[0], load(a.split("=", 1)[1], fname)) for a in argv[1:]]
    ref = runs[0][1]
    for label, d in runs[1:]:
        if not np.array_equal(ref.time.values, d.time.values):
            raise SystemExit(f"{label}: different test times — not comparable")
        if not np.allclose(ref.fp_original.values, d.fp_original.values, equal_nan=True):
            raise SystemExit(f"{label}: different ground truth — not comparable")

    true_lin = ref.fp_original.values.astype("float64")
    true_log = ref.fp_transformed.values.astype("float64")
    invalid = ref.fp_nan_mask.values.astype(bool)
    n, h, w = true_lin.shape
    rad = radial_index(h, w)
    near = rad <= 5
    mid = (rad > 5) & (rad <= 15)
    far = rad > 15
    valid = ~invalid
    print(f"test set: {n} samples, {h}x{w} grid, {invalid.mean()*100:.3f}% invalid pixels\n")

    # magnitude deciles from the true field (positive values only)
    pos = true_lin[valid & (true_lin > 0)]
    qs = np.quantile(pos, np.linspace(0, 1, 11))

    rows, curves = [], {}
    for label, d in runs:
        pred_lin = d.fp_pred.values.astype("float64")
        pred_log = d.fp_transformed_pred.values.astype("float64")
        mse_log = float(np.mean((pred_log[valid] - true_log[valid]) ** 2))

        m = compute_footprint_metrics(
            np.where(valid, true_lin, 0.0), np.where(valid, pred_lin, 0.0),
            metrics=["mse", "mae", "nmae", "corrcoef", "corrcoef_log", "bias", "iou"],
            spatial_shape=(h, w), ignore_mask=invalid, threshold=1e-4)

        # mass conservation: per-sample sum of the predicted vs true footprint
        s_true = np.where(valid, true_lin, 0).sum(axis=(1, 2))
        s_pred = np.where(valid, pred_lin, 0).sum(axis=(1, 2))
        mass_ratio = s_pred / s_true

        # where the mass sits
        share = {name: (np.where(valid & msk, pred_lin, 0).sum() / s_pred.sum())
                 for name, msk in (("near", near), ("mid", mid), ("far", far))}
        share_true = {name: (np.where(valid & msk, true_lin, 0).sum() / s_true.sum())
                      for name, msk in (("near", near), ("mid", mid), ("far", far))}

        # regional errors (relative MAE against the regional mean of the truth)
        reg = {}
        for name, msk in (("near", near), ("mid", mid), ("far", far)):
            sel = valid & msk
            reg[name] = float(np.abs(pred_lin[sel] - true_lin[sel]).mean() / true_lin[sel].mean())

        # radial profile of the mean footprint, and relative MAE per ring
        r_edges = np.arange(0, 36, 2)
        prof_t, prof_p, prof_e = [], [], []
        for lo, hi in zip(r_edges[:-1], r_edges[1:]):
            msk = valid & (rad >= lo) & (rad < hi)
            prof_t.append(true_lin[msk].mean())
            prof_p.append(pred_lin[msk].mean())
            prof_e.append(np.abs(pred_lin[msk] - true_lin[msk]).mean() / true_lin[msk].mean())

        # relative bias per true-magnitude decile
        dec = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            msk = valid & (true_lin >= lo) & (true_lin < hi)
            dec.append((pred_lin[msk].mean() - true_lin[msk].mean()) / true_lin[msk].mean())

        mf = compute_static_mf_metrics(np.where(valid, true_lin, 0.0), np.where(valid, pred_lin, 0.0),
                                       spatial_shape=(h, w), ignore_mask=invalid)

        rows.append((label, mse_log, m, mass_ratio, share, share_true, reg, mf))
        curves[label] = dict(prof_t=np.array(prof_t), prof_p=np.array(prof_p),
                             prof_e=np.array(prof_e), dec=np.array(dec),
                             mass_ratio=mass_ratio, pred=pred_lin,
                             per_sample_mse=((pred_log - true_log) ** 2 * valid).sum((1, 2)) / valid.sum((1, 2)))

    print(f"{'run':>16} | MSE(log) |  corr  corr_log |  NMAE  |  bias(lin)  | IoU>1e-4 | mass pred/true (med, IQR)")
    for label, mse_log, m, mass, *_ in rows:
        q1, q3 = np.percentile(mass, [25, 75])
        print(f"{label:>16} | {mse_log:.4f}   | {m['corrcoef']:.4f} {m['corrcoef_log']:.4f}   | "
              f"{m['nmae']:.4f} | {m['bias']:+.3e} | {m['iou']:.4f}   | {np.median(mass):.3f}  [{q1:.3f}, {q3:.3f}]")

    print(f"\nWhere the predicted footprint mass sits (truth: near {rows[0][5]['near']:.3f}, "
          f"mid {rows[0][5]['mid']:.3f}, far {rows[0][5]['far']:.3f}) and relative MAE by region")
    print(f"{'run':>16} | mass share near / mid / far | rel. MAE near / mid / far")
    for label, _, _, _, share, _, reg, _ in rows:
        print(f"{label:>16} | {share['near']:.3f} {share['mid']:.3f} {share['far']:.3f}        | "
              f"{reg['near']:.3f} {reg['mid']:.3f} {reg['far']:.3f}")

    print(f"\nMole fractions from static flux patterns (correlation / relative mean bias)")
    print(f"{'run':>16} | " + " | ".join(f"{k:>22}" for k in rows[0][7]))
    for label, *_ , mf in rows:
        cells = []
        for k, v in mf.items():
            cells.append(f"r={v['corrcoef']:.4f} b={v['mean_bias']/v['true_mean']:+.3f}")
        print(f"{label:>16} | " + " | ".join(f"{c:>22}" for c in cells))

    # per-sample head-to-head
    if len(rows) == 2:
        a, b = rows[0][0], rows[1][0]
        da, db = curves[a]["per_sample_mse"], curves[b]["per_sample_mse"]
        print(f"\nper-sample training-space MSE: {a} better on {(da < db).mean()*100:.1f}% of the "
              f"{len(da)} test footprints; median difference {np.median(da - db):+.4f}")

    np.save(f"{out_prefix}_curves.npy", {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                                         for k, v in curves.items()}, allow_pickle=True)

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    labels = [r[0] for r in rows]
    # pick samples: median difficulty, and the one where the two runs differ most
    diff = np.abs(curves[labels[0]]["per_sample_mse"] - curves[labels[-1]]["per_sample_mse"])
    mean_mse = np.mean([curves[l]["per_sample_mse"] for l in labels], axis=0)
    picks = [int(np.argsort(mean_mse)[len(mean_mse) // 2]), int(np.argmax(diff)),
             int(np.argsort(mean_mse)[int(0.9 * len(mean_mse))])]
    names = ["typical (median error)", "largest disagreement", "hard case (90th pct error)"]

    ncol = 1 + 2 * len(labels)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.1 * ncol, 3.2 * len(picks)), facecolor="#fcfcfb")
    vmin, vmax = 1e-6, float(np.nanmax(true_lin))
    for r, (idx, nm) in enumerate(zip(picks, names)):
        t = np.where(valid[idx], true_lin[idx], np.nan)
        axes[r, 0].imshow(t, norm=LogNorm(vmin, vmax), cmap="magma_r", origin="lower")
        axes[r, 0].set_ylabel(f"{nm}\n{str(ref.time.values[idx])[:16]}", fontsize=8)
        axes[r, 0].set_title("LPDM truth" if r == 0 else "", fontsize=10)
        for k, l in enumerate(labels):
            p = np.where(valid[idx], curves[l]["pred"][idx], np.nan)
            axes[r, 1 + k].imshow(p, norm=LogNorm(vmin, vmax), cmap="magma_r", origin="lower")
            if r == 0:
                axes[r, 1 + k].set_title(l, fontsize=10)
            d = np.log10(np.clip(p, 1e-12, None)) - np.log10(np.clip(t, 1e-12, None))
            axes[r, 1 + len(labels) + k].imshow(d, norm=TwoSlopeNorm(0, -2, 2), cmap="RdBu_r", origin="lower")
            if r == 0:
                axes[r, 1 + len(labels) + k].set_title(f"log10 ratio\n{l}", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.plot(w // 2, h // 2, "x", color="#1f9d6b", ms=7, mew=2)
    fig.suptitle("Test footprints: LPDM truth vs predictions (log colour scale; x = release point)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_examples.png", dpi=140)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    r_mid = np.arange(1, 35, 2)
    colors = ["#52514e", "#2a78d6", "#eb6834", "#1f9d6b"]
    axes[0].plot(r_mid, curves[labels[0]]["prof_t"], color="#0b0b0b", lw=2.5, label="LPDM truth")
    for l, c in zip(labels, colors[1:]):
        axes[0].plot(r_mid, curves[l]["prof_p"], color=c, lw=2, ls="--", label=l)
    axes[0].set_yscale("log"); axes[0].set_xlabel("distance from release point (grid cells)")
    axes[0].set_ylabel("mean footprint"); axes[0].set_title("Radial profile of the mean footprint")
    axes[0].legend(frameon=False, fontsize=9)
    for l, c in zip(labels, colors[1:]):
        axes[1].plot(r_mid, curves[l]["prof_e"], color=c, lw=2, label=l)
    axes[1].set_xlabel("distance from release point (grid cells)"); axes[1].set_ylabel("MAE / regional mean")
    axes[1].set_title("Relative error vs distance"); axes[1].legend(frameon=False, fontsize=9)
    for l, c in zip(labels, colors[1:]):
        axes[2].plot(np.arange(1, 11), curves[l]["dec"], "o-", color=c, lw=2, label=l)
    axes[2].axhline(0, color="#0b0b0b", lw=1)
    axes[2].set_xlabel("decile of the true footprint value"); axes[2].set_ylabel("relative bias")
    axes[2].set_title("Bias by footprint magnitude"); axes[2].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(color="#e6e5e1", lw=0.8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_diagnostics.png", dpi=140)
    print("saved", f"{out_prefix}_examples.png", f"{out_prefix}_diagnostics.png")


if __name__ == "__main__":
    main()
