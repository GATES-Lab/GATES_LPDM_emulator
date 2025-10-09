import os 
import shutil
import glob
# Create file structure for adding data

def create_data_directories(region):
    """
    For use on Oracle (and possibly cloud systems)
    Create the standard data folder structure if it doesn't already exist.
    """

    base_dir = "data"

    directories = [
        os.path.join(base_dir, "fp_archive", region),
        os.path.join(base_dir, "met_archive", region),
        os.path.join(base_dir, "LPDM", "topog_NAME"),
    ]

    for dir in directories:
        os.makedirs(dir, exist_ok=True)

def populate_data_directories(region, period, base_dir, dest_dir):
    """
    Copy files matching a region and period pattern from multiple archive folders
    (e.g. fp_archive, met_archive) into dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Define which subfolders to look in
    source_subdirs = ["fp_archive", "met_archive"]

    # Loop through each archive subdirectory
    for subdir in source_subdirs:
        print("Now processing:", subdir)
        source_path = os.path.join(base_dir, subdir, region)

        # Build the file pattern (e.g. /base/fp_archive/NORTHAFRICA/NORTHAFRICA_Met_20160[1-3].nc)
        pattern = os.path.join(source_path, f"*{region}*{period}.nc")

        # Find matching files
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"No files found for pattern: {pattern}")
            continue

        for file_path in matching_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, filename)

            if os.path.exists(dest_path):
                print(f"Skipping {filename} — already exists in destination.")
                continue

            shutil.copy(file_path, dest_path)
            print(f"Copied {os.path.basename(file_path)} from {subdir} to {dest_dir}")

    # Also copy topo and landuse files
    print("Now processing: topo and landuse files")
    LPDM_source = "/mnt/data/LPDM/topog_NAME"
    LPDM_dest = os.path.join(dest_dir, "LPDM", "topog_NAME")
    os.makedirs(LPDM_dest, exist_ok=True)

    for file_path in glob.glob(os.path.join(LPDM_source, "*")):
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(LPDM_dest, filename)

            if os.path.exists(dest_path):
                print(f"Skipping one-off file {filename} — already exists in {LPDM_dest}")
                continue

            shutil.copy(file_path, dest_path)
            print(f"Copied one-off file {filename} to {LPDM_dest}")


def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove subdirectory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# 1. Select region to process
region = "NORTHAFRICA"
period = "20160[1-2]"
source_dir = "/mnt/data/"
dest_dir = "data/"

# 2. Create data directories if they don't already exist
create_data_directories(region)

# 3. Copy content from cold to hot storage

populate_data_directories(
    region,
    period,
    source_dir,
    dest_dir,
)


# 4. Run script(s)


# 5. Empty data directory to cut costs
#empty_folder("data/")