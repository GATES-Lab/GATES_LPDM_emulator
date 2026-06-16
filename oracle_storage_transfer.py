import os 
import shutil
import glob
import argparse
import json
import yaml


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


def load_file(file_name, file_path):
    """Load JSON parameters from an explicit path or from file_name under file_path."""
    if not file_path:
        file_path = "/user/work/ef17148/GCN/graphnet/graph_weather/train_satellite_files/"
    file_path = f"{file_path}{file_name}"
    try:
        with open(file_path, "r") as file:
            if file_path.endswith(".json"):
                data = json.load(file)
            else:
                data = file.read()
                data = json.loads(data)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None


def create_data_directories(region):
    """Create standard staged-data folder structure if it doesn't exist."""
    base_dir = "data"
    directories = [
        os.path.join(base_dir, "fp_archive", region),
        os.path.join(base_dir, "met_archive", region),
        os.path.join(base_dir, "LPDM", "topog_NAME"),
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)


def populate_data_directories(region, period, base_dir, dest_dir, dry_run=False, fp_domain=None):
    """
    Copy train/test files matching a region+period pattern into dest_dir.

    Parameters
    ----------
    fp_domain : str, optional
        If provided, used as filename prefix filter for fp_archive files.
    """
    os.makedirs(dest_dir, exist_ok=True)

    source_subdirs = ["fp_archive", "met_archive"]
    for subdir in source_subdirs:
        print("Now processing:", subdir)
        source_path = os.path.join(base_dir, subdir, region)
        prefix = fp_domain if (subdir == "fp_archive" and fp_domain) else region
        pattern = os.path.join(source_path, f"*{prefix}*{period}*.nc")
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"No files found for pattern: {pattern}")
            continue

        for file_path in matching_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, subdir, region, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            if os.path.exists(dest_path):
                print(f"Skipping {filename} — already exists in destination.")
                continue

            if dry_run:
                print(f"[DRY RUN] Would copy {file_path} -> {dest_path}")
            else:
                shutil.copy2(file_path, dest_path)
                print(f"Copied {filename} to {dest_path}")

    print("Now processing: topo and landuse files")
    lpdm_source = os.path.join(base_dir, "LPDM", "topog_NAME")
    lpdm_dest = os.path.join(dest_dir, "LPDM", "topog_NAME")
    os.makedirs(lpdm_dest, exist_ok=True)

    for file_path in glob.glob(os.path.join(lpdm_source, "*")):
        if not os.path.isfile(file_path):
            continue

        filename = os.path.basename(file_path)
        dest_path = os.path.join(lpdm_dest, filename)

        if os.path.exists(dest_path):
            print(f"Skipping one-off file {filename} — already exists in {lpdm_dest}")
            continue

        if dry_run:
            print(f"[DRY RUN] Would copy {file_path} -> {dest_path}")
        else:
            shutil.copy2(file_path, dest_path)
            print(f"Copied one-off file {filename} to {dest_path}")


def empty_folder(folder_path):
    """Delete all files/subfolders inside folder_path."""
    if not os.path.exists(folder_path):
        print(f"Folder does not exist, nothing to delete: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print("Deleted:", file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


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
    p.add_argument("--param-file", default="./parameter_files/NEW_parameter_template_gpu_new3.json")
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
    #Load config and domains
    parameters = load_file("", args.param_file)

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    domain = parameters["train_load_data"]["region"]
    domain_config = config["domains"][domain]
    region = domain_config["domain_name"]
    fp_domain = domain_config.get("fp_domain")

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
            fp_domain=fp_domain,
        )

if __name__ == "__main__":
    main()