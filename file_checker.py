"""
file_checker.py — Diagnose footprint (and met) files for time coordinate issues.

Usage:
    python file_checker.py                        # checks CHINA 2012 fp files (default)
    python file_checker.py --pattern "data/fp_archive/CHINA/*EASTASIA_2012*.nc"
    python file_checker.py --pattern "data/fp_archive/INDIA/*SOUTHASIA_201[3-5]*.nc"
    python file_checker.py --pattern "data/met_archive/NORTHAFRICA/NORTHAFRICA_Met_201[3-4]*.nc"

For each file it reports:
    - time coordinate range and length
    - whether time is monotonically increasing
    - any duplicate time values
    - whether xarray can open it at all (catches corrupt HDF5)

Files with problems are summarised at the end.
"""

import argparse
import glob
import numpy as np
import xarray as xr


DEFAULT_PATTERN = "data/fp_archive/CHINA/*EASTASIA_2012*.nc"


def check_file(path: str) -> dict:
    """Open a single netCDF file and check its time coordinate."""
    result = {"path": path, "ok": True, "issues": []}
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Cannot open: {e}")
        return result

    if "time" not in ds.coords and "time" not in ds.dims:
        result["issues"].append("No 'time' coordinate found")
        ds.close()
        return result

    try:
        time_vals = ds["time"].values
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Cannot read time values: {e}")
        ds.close()
        return result

    result["n_times"] = len(time_vals)
    result["time_start"] = str(time_vals[0]) if len(time_vals) > 0 else "N/A"
    result["time_end"] = str(time_vals[-1]) if len(time_vals) > 0 else "N/A"

    if len(time_vals) > 1:
        diffs = np.diff(time_vals.astype("datetime64[ns]").astype(np.int64))
        is_increasing = bool(np.all(diffs > 0))
        is_decreasing = bool(np.all(diffs < 0))
        n_duplicates = int(np.sum(diffs == 0))
        n_negative = int(np.sum(diffs < 0))

        result["monotonic_increasing"] = is_increasing
        result["n_duplicates"] = n_duplicates

        if not is_increasing and not is_decreasing:
            result["ok"] = False
            msg = "Time is NOT monotonically increasing or decreasing"
            if n_duplicates:
                msg += f" ({n_duplicates} duplicate timestamps)"
            if n_negative:
                msg += f" ({n_negative} backwards steps)"
            result["issues"].append(msg)
            # Show where the problems occur
            bad_indices = np.where(diffs <= 0)[0]
            for idx in bad_indices[:5]:  # show first 5
                result["issues"].append(
                    f"  Problem at index {idx}->{idx+1}: {time_vals[idx]} -> {time_vals[idx+1]}"
                )
            if len(bad_indices) > 5:
                result["issues"].append(f"  ... and {len(bad_indices) - 5} more")
    else:
        result["monotonic_increasing"] = True
        result["n_duplicates"] = 0

    ds.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Check fp/met files for time coord issues")
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for files to check (quote it!)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching: {args.pattern}")
        return

    print(f"Checking {len(files)} files matching: {args.pattern}\n")

    bad_files = []
    for path in files:
        fname = path.split("/")[-1]
        result = check_file(path)
        status = "OK" if result["ok"] else "PROBLEM"
        n = result.get("n_times", "?")
        t0 = result.get("time_start", "?")
        t1 = result.get("time_end", "?")
        mono = "✓" if result.get("monotonic_increasing", False) else "✗"
        dups = result.get("n_duplicates", "?")
        print(f"[{status:7s}] {fname}  |  n={n}  mono={mono}  dups={dups}  [{t0} → {t1}]")
        for issue in result.get("issues", []):
            print(f"           {issue}")
        if not result["ok"]:
            bad_files.append(path)

    print(f"\n{'='*60}")
    if bad_files:
        print(f"SUMMARY: {len(bad_files)} problematic file(s):")
        for f in bad_files:
            print(f"  {f}")
    else:
        print(f"SUMMARY: All {len(files)} files look OK.")


if __name__ == "__main__":
    main()
