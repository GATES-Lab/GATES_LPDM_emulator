import sys
import json
from netCDF4 import Dataset

def check_netcdf(path):
    """
    Check whether a NetCDF file can be opened and return basic metadata.

    Parameters
    ----------
    path : str
        Path to the NetCDF file.

    Returns
    -------
    dict
        Dictionary containing status and file information.
        On success:
            {
                "status": "ok",
                "vars": int,
                "dims": int,
                "format": str
            }
        On failure:
            {
                "status": "fail",
                "error": str
            }
    """
    try:
        with Dataset(path, "r") as ds:
            return {
                "status": "ok",
                "vars": len(ds.variables),
                "dims": len(ds.dimensions),
                "format": ds.file_format,
            }
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e),
        }


def main():
    if len(sys.argv) != 2:
        print(
            json.dumps(
                {"status": "fail", "error": "usage: worker_check_nc.py <path.nc>"}
            )
        )
        sys.exit(2)

    path = sys.argv[1]
    result = check_netcdf(path)

    print(json.dumps(result))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()