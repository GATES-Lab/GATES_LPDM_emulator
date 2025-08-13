import sys, json
from netCDF4 import Dataset

if len(sys.argv) != 2:
    print(json.dumps({"status": "fail", "error": "usage: worker_check_nc.py <path.nc>"}))
    sys.exit(2)

path = sys.argv[1]

try:
    with Dataset(path, "r") as ds:
        info = {
            "status": "ok",
            "vars": len(ds.variables),
            "dims": len(ds.dimensions),
            "format": ds.file_format,  # e.g. NETCDF4, NETCDF4_CLASSIC
        }
    print(json.dumps(info))
    sys.exit(0)
except Exception as e:
    print(json.dumps({"status": "fail", "error": str(e)}))
    sys.exit(1)
