import xarray as xr
import os


def clean_netcdf_file(input_path, output_path, vars_to_drop=None):
    """
    Open a NetCDF file, drop selected variables, and write a cleaned version.

    Parameters
    ----------
    input_path : str
        Path to the original NetCDF file.
    output_path : str
        Path where the cleaned NetCDF file will be saved.
    vars_to_drop : list of str, optional
        Variables to remove from the dataset. Missing variables
        are ignored.
    """
    if vars_to_drop is None:
        vars_to_drop = [
            "forecast_period",
            "forecast_reference_time",
            "newtime",
        ]

    with xr.open_dataset(input_path) as ds:
        ds_clean = ds.drop_vars(vars_to_drop, errors="ignore")
        ds_clean.to_netcdf(output_path, format="NETCDF4")


def clean_problem_files(input_dir, output_dir, problem_dates):
    """
    Clean a set of known problematic NetCDF files.

    Parameters
    ----------
    input_dir : str
        Directory containing the original NetCDF files.
    output_dir : str
        Directory where cleaned files will be written.
    problem_dates : list of str
        Year–month strings (YYYYMM) identifying files to clean.
    """
    os.makedirs(output_dir, exist_ok=True)

    for date in problem_dates:
        filename = f"INDIA_Met_{date}.nc"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            print(f"Fixing: {filename}")
            clean_netcdf_file(input_path, output_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    input_dir = "/projects/b5s/data/met_archive/UM/INDIA/"
    output_dir = "/home/b5s/jeffc.b5s/cleaned_met_files/"
    problem_dates = ["201203", "201307", "201402", "201410", "201411"]

    clean_problem_files(input_dir, output_dir, problem_dates)