import os 
import shutil
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

def populate_data_directories(region):
    print("populate")

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

# 2. Create data directories if they don't already exist
create_data_directories(region)

# 3. Copy content from cold to hot storage

# 4. Run script(s)


# 5. Empty data directory to cut costs
empty_folder("data/")