#!/usr/bin/env python

import os
import glob
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========================
# User configuration
# ========================
DATA_DIR = "data/met_archive/SOUTHAMERICA/"
OUT_DIR = "check_met_plots/"

# Variables to check (edit to match your data)
VARS_TO_PLOT = [
'x_wind', 'y_wind', 'air_temperature'
]

# ========================
# Setup
# ========================
os.makedirs(OUT_DIR, exist_ok=True)

nc_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))

print(f"Found {len(nc_files)} NetCDF files")


# ========================
# Main loop
# ========================
for nc in nc_files:
    fname = os.path.basename(nc)
    stem = fname.replace(".nc", "")
    print(f"Processing {fname}")

    try:
        ds = xr.open_dataset(nc)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(VARS_TO_PLOT),
            figsize=(5 * len(VARS_TO_PLOT), 5),
            constrained_layout=True,
        )

        # Ensure axes is iterable even if len == 1
        if len(VARS_TO_PLOT) == 1:
            axes = [axes]

        plotted_any = False

        for ax, var in zip(axes, VARS_TO_PLOT):
            if var not in ds:
                ax.set_title(f"{var}\n(not found)")
                ax.axis("off")
                continue

            da = ds[var]

            # Reduce dimensions for plotting
            if "time" in da.dims:
                da = da.isel(time=0)
            if "model_level_number" in da.dims:
                da = da.isel(model_level_number=0)

            da.plot(ax=ax, add_colorbar=True)
            ax.set_title(var)
            plotted_any = True

        if plotted_any:
            fig.suptitle(fname, fontsize=14)

            out_png = os.path.join(OUT_DIR, f"{stem}.png")
            plt.savefig(out_png, dpi=150)
            print(f"  → saved {out_png}")

        plt.close(fig)
        ds.close()

    except Exception as e:
        print(f"ERROR processing {fname}: {e}")

print("Done.")