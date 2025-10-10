import os 
import shutil
import glob
from model.data.load_data import *

# 1. Select region and time period to process, using parameter file
source_dir = "/mnt/data/"
dest_dir = "data/"

file_name = "./parameter_files/parameter_file_paper.json"
parameters = load_file("", file_name) 

domain = parameters["train_load_data"]["region"]
domains = {"BRAZIL":"SOUTHAMERICA", "SOUTHAMERICA":"SOUTHAMERICA", "SAHARA":"NORTHAFRICA", "INDIA":"INDIA"} 

region = domains[domain]   
period = parameters["train_load_data"]["year"]

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
# Do so in the oracle_slurm.sh script

# 5. Empty data directory to cut costs
#empty_folder("data/")