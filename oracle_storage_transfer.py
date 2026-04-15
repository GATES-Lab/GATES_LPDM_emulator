import os 
import shutil
import glob
import argparse
from model.data.load_data import *


'''
Use this for staging data from cold storage to hot storage

This script supports two main operations:
- populate: copy required train/test data from a source directory  (e.g. cold storage or network mount) into a local working directory.
- cleanup: remove staged data from the destination directory.


Typical usage:
- Preview what would happen    python oracle_storage_transfer.py --mode populate --which both --dry-run
- Copy train and test data     python oracle_storage_transfer.py --mode populate --which both
- Remove staged data           python oracle_storage_transfer.py --mode cleanup

The script relies on a parameter JSON file to determine region and time periods for the data to be staged.
'''

DOMAINS = {
    "BRAZIL": "SOUTHAMERICA",
    "SOUTHAMERICA": "SOUTHAMERICA",
    "SAHARA": "NORTHAFRICA",
    "INDIA": "INDIA",
}

def parse_args():
    """
    Parse command-line arguments for the storage transfer script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following attributes:
        - param_file : str
            Path to the JSON parameter file describing data loading options.
        - source_dir : str
            Base directory containing the source (cold storage) data.
        - dest_dir : str
            Destination directory where data will be staged.
        - which : {"train", "test", "both"}
            Specifies whether to operate on training data, test data,
            or both.
        - mode : {"populate", "cleanup"}
            Operation mode: stage data into dest_dir or remove it.
        - dry_run : bool
            If True, print planned actions without copying or deleting files.
    """
    p = argparse.ArgumentParser(description="Stage data from cold storage to hot storage, and optionally clean up.")
    p.add_argument("--param-file", default="./parameter_files/parameter_file_paper.json")
    p.add_argument("--source-dir", default="/mnt/data/")
    p.add_argument("--dest-dir", default="data/")
    p.add_argument("--which", choices=["train", "test", "both"], default="both")
    p.add_argument("--mode", choices=["populate", "cleanup"], required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    """
    Entry point for staging or cleaning up data.

    This function:
    1. Parses command-line arguments.
    2. In cleanup mode:
       - Performs safety checks on the destination directory.
       - Either reports or deletes all staged data.
    3. In populate mode:
       - Loads the parameter file.
       - Determines the domain/region mapping.
       - Identifies train and/or test periods to stage.
       - Creates required directory structure.
       - Copies data from the source directory into the destination directory.

    Safety Notes
    ------------
    - Cleanup mode includes a guard to prevent deletion of dangerous paths
      such as '/', '/mnt', or '/mnt/data'.
    - When --dry-run is specified, no filesystem modifications are made.

    Raises
    ------
    KeyError
        If required fields (e.g. train_load_data.year or test_load_data.year)
        are missing from the parameter file.
    ValueError
        If cleanup is requested on a suspicious destination directory.
    """
    args = parse_args()

    if args.mode == "cleanup":
        # Recommended guard: refuse to delete suspicious paths
        dest_abs = os.path.abspath(args.dest_dir)
        if dest_abs in ("/", "/mnt", "/mnt/data"):
            raise ValueError(f"Refusing to delete dangerous dest_dir: {dest_abs}")

        if args.dry_run:
            print(f"[DRY RUN] Would empty folder: {args.dest_dir}")
        else:
            empty_folder(args.dest_dir)
        return

    # populate mode
    parameters = load_file("", args.param_file)

    domain = parameters["train_load_data"]["region"]
    region = DOMAINS[domain]

    create_data_directories(region)

    # collect periods
    train_period = parameters.get("train_load_data", {}).get("year")
    test_period  = parameters.get("test_load_data", {}).get("year")

    periods = []
    if args.which in ("train", "both"):
        if not train_period:
            raise KeyError("train_load_data.year missing in parameter file")
        periods.append(("train", train_period))

    if args.which in ("test", "both"):
        if not test_period:
            raise KeyError("test_load_data.year missing in parameter file")
        periods.append(("test", test_period))

    for label, period in periods:
        print(f"\n=== Populating {label} data for period '{period}' (region={region}) ===")
        populate_data_directories(
            region=region,
            period=period,
            base_dir=args.source_dir,
            dest_dir=args.dest_dir,
            dry_run=args.dry_run,
        )

if __name__ == "__main__":
    main()