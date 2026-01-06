import os 
import shutil
import glob
import argparse
from model.data.load_data import *

## Use this for staging data from cold storage to hot storage
# Eg to see what it would do:
#python oracle_storage_dev.py --mode populate --which both --dry-run
#python oracle_storage_dev.py --mode cleanup --dry-run

# Then to copy data python oracle_storage_dev.py --mode populate --which both
# Then to delete data python oracle_storage_dev.py --mode cleanup


DOMAINS = {
    "BRAZIL": "SOUTHAMERICA",
    "SOUTHAMERICA": "SOUTHAMERICA",
    "SAHARA": "NORTHAFRICA",
    "INDIA": "INDIA",
}

def parse_args():
    p = argparse.ArgumentParser(description="Stage data from cold storage to hot storage, and optionally clean up.")
    p.add_argument("--param-file", default="./parameter_files/parameter_file_paper.json")
    p.add_argument("--source-dir", default="/mnt/data/")
    p.add_argument("--dest-dir", default="data/")
    p.add_argument("--which", choices=["train", "test", "both"], default="both")
    p.add_argument("--mode", choices=["populate", "cleanup"], required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
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
