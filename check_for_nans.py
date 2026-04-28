import xarray as xr
import numpy as np
import glob

# ── Configure these paths ───────────────────────────────────────
met_dir = "data/met_archive/SOUTHAMERICA"
fp_dir  = "data/fp_archive/SOUTHAMERICA"
year    = "2013"
region  = "BRAZIL"
# ────────────────────────────────────────────────────────────────

met_vars = [
    "x_wind", "y_wind", "upward_air_velocity",
    "atmosphere_boundary_layer_thickness", "surface_air_pressure",
    "air_temperature", "air_pressure"
]

print("=" * 60)
print(f"FOOTPRINT NaN check — {year} {region}")
print("=" * 60)
for month in [f"{m:02d}" for m in range(1, 13)]:
    files = glob.glob(f"{fp_dir}/*{region}*{year}{month}*.nc")
    if not files:
        print(f"  {year}-{month}: NO FILE FOUND"); continue
    ds = xr.open_dataset(files[0], engine="h5netcdf", chunks={})
    for var in ds.data_vars:
        n = int(ds[var].isnull().sum().values)
        if n > 0:
            print(f"  {year}-{month} | {var}: {n} NaNs  shape={ds[var].shape}")
        #else:
    print(f"  {year}-{month}: clean for NaNs")
    ds.close()

print()
print("=" * 60)
print(f"MET NaN check — {year} {region}")
print("=" * 60)
for month in [f"{m:02d}" for m in range(1, 13)]:
    print(f"Checking {year}-{month} met...")
    files = glob.glob(f"{met_dir}/*{year}{month}*.nc")
    if not files:
        print(f"  {year}-{month}: NO MET FILE FOUND"); continue
    ds = xr.open_dataset(files[0], chunks={})
    flagged = []
    for var in met_vars:
        if var not in ds: continue
        has_nan = bool(ds[var].isnull().any().values)
        if has_nan:
            time_nan_mask = ds[var].isnull().any(dim=[d for d in ds[var].dims if d != "time"])
            nan_times = ds.time.values[time_nan_mask.values]
            flagged.append(f"{var} ({len(nan_times)} time steps, first={np.datetime_as_string(nan_times[0], unit='h')})")
    if flagged:
        print(f"  {year}-{month}: NaNs in —")
        for f in flagged:
            print(f"    {f}")
    else:
        print(f"  {year}-{month}: clean")
    ds.close()