import xarray as xr
import glob
import os


input_dir = "/projects/b5s/data/met_archive/UM/INDIA/"
output_dir = "/home/b5s/jeffc.b5s/cleaned_met_files/UM/INDIA/"
os.makedirs(output_dir, exist_ok=True)

paths = sorted(glob.glob(f"{input_dir}/INDIA_Met_201[3]*.nc"))


'''
for path in paths:
    with xr.open_dataset(path) as ds:
        coord_vars = set(ds.coords)
        problem_vars = coord_vars.intersection({"sigma", "level_height"})
        if problem_vars:
            print(f"Still a coordinate in {os.path.basename(path)}: {problem_vars}")


for path in paths:
    filename = os.path.basename(path)
    print(f"Processing: {filename}")
    
    try:
        ds = xr.open_dataset(path)

        # Force reset of coords
        for var in ["sigma", "level_height"]:
            if var in ds.coords:
                print(f"  Forcing reset of coord: {var}")
                ds = ds.reset_coords(var, drop=True)

        output_path = os.path.join(output_dir, filename)
        ds.to_netcdf(output_path, format="NETCDF4")
        print(f" Saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

'''



import xarray as xr
import glob
import os

#/projects/b5s/data/met_archive/UM/INDIA/
paths = sorted(glob.glob("/home/b5s/jeffc.b5s/cleaned_met_files/UM/INDIA/INDIA_Met_2013*.nc"))

for path in paths:
    try:
        with xr.open_dataset(path) as ds:
            time_vals = ds.time.values if "time" in ds.coords or "time" in ds.dims else "No 'time' found"
            print(f"{os.path.basename(path)}: {time_vals}")
    except Exception as e:
        print(f"Failed to open {os.path.basename(path)}: {e}")
