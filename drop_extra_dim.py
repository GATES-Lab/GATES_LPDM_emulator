"""
Drop the leftover ``dim_0`` coordinate from the 2015 and 2016 yearly stores.

Those two stores were written before ``join._DROP_VARIABLES`` started dropping
``dim_0``, so they carry a bare 0..N index along time that every data variable
lists in its ``coordinates`` attribute. This is metadata surgery in place: the
dim_0 chunks are deleted and every reference to it removed. No data variable is
read or rewritten.
"""

import json
import shutil
from pathlib import Path

ZARR_DIR = Path("/projects/b5bn/data/met_archive_zarr/SOUTHAMERICA")

STORES = [
    ZARR_DIR / "SOUTHAMERICA_Met_2015.zarr",
    ZARR_DIR / "SOUTHAMERICA_Met_2016.zarr",
]

DROP = "dim_0"


def strip_from_coordinates(attrs):
    """Remove DROP from a ``coordinates`` attribute. True if it was there."""
    coordinates = attrs.get("coordinates", "").split()
    if DROP not in coordinates:
        return False
    attrs["coordinates"] = " ".join(name for name in coordinates if name != DROP)
    return True


def fix_store(store):
    # Per-variable .zattrs on disk.
    for zattrs_path in sorted(store.glob("*/.zattrs")):
        attrs = json.loads(zattrs_path.read_text())
        if strip_from_coordinates(attrs):
            zattrs_path.write_text(json.dumps(attrs, indent=4))
            print(f"  {zattrs_path.parent.name}: dropped {DROP} from coordinates")

    # The consolidated copy xarray actually reads.
    consolidated_path = store / ".zmetadata"
    consolidated = json.loads(consolidated_path.read_text())
    metadata = consolidated["metadata"]

    for key in [key for key in metadata if key.startswith(f"{DROP}/")]:
        del metadata[key]
        print(f"  .zmetadata: removed {key}")

    for key, attrs in metadata.items():
        if key.endswith(".zattrs"):
            strip_from_coordinates(attrs)

    consolidated_path.write_text(json.dumps(consolidated, indent=4))

    shutil.rmtree(store / DROP)
    print(f"  deleted {DROP}/")


for store in STORES:
    print(store.name)
    fix_store(store)
 