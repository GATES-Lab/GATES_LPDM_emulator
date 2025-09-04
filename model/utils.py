import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from datetime import date

def parse_years(year_str):
    # Case 1: Range like '201[4-5]'
    match = re.fullmatch(r'201\[(\d)-(\d)\]', year_str)
    if match:
        start, end = map(int, match.groups())
        return [2010 + i for i in range(start, end + 1)]
    
    # Case 2: Exact year like '2014' or '2015'
    if re.fullmatch(r'20\d{2}', year_str):
        return [int(year_str)]
    
    # Fallback
    raise ValueError(f"Invalid year format: {year_str}")


def baseline_mol_updated(desired_data,months,years):
    '''
    Function used to get the value for the input and output
    '''
    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]                                                                      
    # Load the CSV file
    #df = pd.read_csv('Analysis/CH4_Semihemispheric_modelled_mole_fractions.csv')
    df = pd.read_csv('/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/CH4_Semihemispheric_modelled_mole_fractions.csv')
    baseline_list = np.zeros((total_data_points,4))

    '''
    # Specify the year and month you're interested in
    # Filter the data for the specific year and month
    filtered_data = df[(df['Year'] == specific_year) & (df['Month'] == specific_month)]

    # Extract the 4th, 5th, 6th, and 7th columns
    selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values

    # Display the results
    print(selected_columns)
    '''

    # Iterate through the different numbers 
    #months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    #year = 2016
    datetime_array = np.array(desired_data.fp_data_full.particle_locations_n.time)
    specific_years = np.array([np.datetime64(date, 'Y').astype(int) + 1970 for date in datetime_array])
    specific_months = np.array([np.datetime64(date, 'M').astype(int) % 12 + 1 for date in datetime_array])

    total_data_points = desired_data.fp_data_full.particle_locations_n.time.shape[-1]
    north_list = np.zeros(total_data_points) # Creates a list of size N with None values
    south_list = np.zeros(total_data_points)
    east_list = np.zeros(total_data_points)
    west_list = np.zeros(total_data_points)
    for year in years:
        print('year',year)
        for month in months:
            # Cams field for a particular month
            if year > 2017:
                # Change made due to the naming of the data
                cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion_climatology.nc")
            else:
                cams = xr.open_dataset(f"/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year}{month}_CAMS-inversion.nc")    
            # Extract year and month
            #import ipdb; ipdb.set_trace()
            

            # Desired year and month
            #desired_year = 2016
            desired_year = year
            desired_month = month
            print('desired month',desired_month)
            
            '''
            coarse_cams  = cams.coarsen(lon=desired_data.coarsening_factor, lat=desired_data.coarsening_factor, boundary="pad").mean()
            '''
            # Find the index of the first occurrence
            indices = np.where((specific_years == desired_year) & (specific_months == int(desired_month)))[0]
            print(indices)
            if len(indices)>0:
                # Get the inputs
                filtered_data = df[(df['Year'] == desired_year) & (df['Month'] == int(desired_month))]
                # Extract the 4th, 5th, 6th, and 7th columns
                selected_columns = filtered_data.iloc[:, [3, 4, 5, 6]].values
                baseline_list[indices] = selected_columns/1000 # Convert from parts per trillion to parts per million

                print(indices[0])
                # Multply the first value with all the other values of the array
                print(cams.vmr_n.shape)
                # CAMS field should be stationary over the period of a month
                #import ipdb; ipdb.set_trace()
                '''
                north_mol = np.sum(cams.vmr_n * desired_data.locs.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(cams.vmr_s * desired_data.locs.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(cams.vmr_e * desired_data.locs.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(cams.vmr_w * desired_data.locs.particle_locations_w[:,:,indices], axis=(0,1))
                '''
                '''
                north_mol = np.sum(coarse_cams.vmr_n * desired_data.locs.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(coarse_cams.vmr_s * desired_data.locs.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(coarse_cams.vmr_e * desired_data.locs.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(coarse_cams.vmr_w * desired_data.locs.particle_locations_w[:,:,indices], axis=(0,1))
                '''
                
                north_mol = np.sum(cams.vmr_n * desired_data.fp_data_full.particle_locations_n[:,:,indices], axis=(0,1))
                south_mol = np.sum(cams.vmr_s * desired_data.fp_data_full.particle_locations_s[:,:,indices], axis=(0,1))
                east_mol = np.sum(cams.vmr_e * desired_data.fp_data_full.particle_locations_e[:,:,indices], axis=(0,1))
                west_mol = np.sum(cams.vmr_w * desired_data.fp_data_full.particle_locations_w[:,:,indices], axis=(0,1))
                
                #import ipdb; ipdb.set_trace()
                north_list[indices] = north_mol
                south_list[indices] = south_mol
                east_list[indices] = east_mol
                west_list[indices] = west_mol


        #print(baseline_list)
        #cams = xr.open_dataset("/group/chemistry/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_201611_CAMS-inversion.nc")
        # Making the assumption that the values in the month are not different, get the first value
        # Multiple the different values
    
    return baseline_list, north_list, south_list, east_list, west_list


def write_to_file(path, model_name,message):
    f = open(f"{path}{model_name}/{model_name}_updates.txt", "a")
    f.write(datetime.now().strftime("%d/%m/%y %H:%M:%S") + " " + message + "\n")
    f.close()
