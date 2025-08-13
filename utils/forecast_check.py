import xarray as xr
import glob

paths = sorted(glob.glob("/projects/b5s/data/met_archive/UM/INDIA/INDIA_Met_201[2-4]*.nc"))
#paths = sorted(glob.glob("/home/b5s/jeffc.b5s/cleaned_met_files/INDIA_Met_201*.nc"))

for path in paths:
    try:
        with xr.open_dataset(path) as ds:
            if "forecast_period" not in ds.coords:
                print("Missing forecast_period:", path)
            elif "forecast_period" in ds.coords:
                print("*** Found forecast_period:", path)
    except Exception as e:
        print("Failed to open:", path, e)
