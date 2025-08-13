import os
import glob
from netCDF4 import Dataset

directory = "/projects/b5s/data/fp_archive/INDIA"
nc_files = glob.glob(os.path.join(directory, "*.nc"))

print(f"Found {len(nc_files)} NetCDF files in {directory}\n")

for path in nc_files:
    try:
        with Dataset(path, "r") as ds:
            vars_list = list(ds.variables.keys())
        print(f"[OK]   {os.path.basename(path)} — {len(vars_list)} variables")
    except Exception as e:
        print(f"[FAIL] {os.path.basename(path)} — {e}")
