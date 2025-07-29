import os
import xarray as xr
import numpy as np

input_dir = "/projects/b5s/data/met_archive/UM/INDIA/"

output_dir = "/projects/b5s/data/met_archive_32/UM/INDIA/"

# Make sure output folder exists
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".nc"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        ds = xr.open_dataset(input_path)

        # Convert float64 data variables to float32, leave others unchanged
        new_vars = {}
        for var in ds.data_vars:
            if ds[var].dtype == np.float64:
                new_vars[var] = ds[var].astype(np.float32)
            else:
                new_vars[var] = ds[var]

        # Reconstruct the dataset with original coords and attrs
        ds_float32 = xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)

        # Save as NetCDF
        ds_float32.to_netcdf(output_path)

        # Compare file sizes
        orig_size = os.path.getsize(input_path)
        new_size = os.path.getsize(output_path)
        print(f"{filename}: Saved {(orig_size - new_size) / 1e6:.2f} MB")

    except Exception as e:
        print(f"Error processing {filename}: {e}")