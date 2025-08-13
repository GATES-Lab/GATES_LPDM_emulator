import xarray as xr
import os

# Directory containing the original files
input_dir = "/projects/b5s/data/met_archive/UM/INDIA/"
# Directory to save cleaned files
output_dir = "/home/b5s/jeffc.b5s/cleaned_met_files/"
os.makedirs(output_dir, exist_ok=True)

# Files you want to fix
problem_dates = ["201203", "201307", "201402", "201410", "201411"]

for date in problem_dates:
    filename = f"INDIA_Met_{date}.nc"
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        with xr.open_dataset(input_path) as ds:
            print(f"🧹 Fixing: {filename}")
            ds_clean = ds.drop_vars(["forecast_period", "forecast_reference_time", "newtime"], errors="ignore")
            ds_clean.to_netcdf(output_path, format="NETCDF4")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
