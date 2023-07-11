# graphnet_LPDM_emulator

## Loading data
Use the `LoadSatelliteData` object to load data for a date period.
``` 
data = LoadSatelliteData(year=2016, region="BRAZIL", freq=2, metsize=50, size =50, topog="default", verbose=True, cut_met = False, met_datadir="/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_")
```

This function works fine but needs some cleaning up (there are some inconsistencies in formats, some parameters are redundant and chunks should be split into separate functions). Some notes:

- All of the necessary data is in the ACRG folder (/group/chemistry/acrg/), the paths all default to this unless specifed
- Only valid `region`s are BRAZIL, SAHARA, INDIA (ie there are default domains and some footprint and met data)
- `year` can be an int (eg 2016) or a str (eg "201[4-5]"). There is a parameter `month` to load a specific month of data (so `year=2016, month="01"`). The month parameter isn't bulletproof, I have been using `year` directly, eg `year=201601` or `year="201601"` instead
- footprints:
  - The function takes footprint data and cuts it to a square of size *size* around the measurement point. The original footprint data is conserved in  `data.fp_data_full` and the cut footprint data is stored flattened in a np array of shape (time, size*size) in `data.fp_data`
  - The latitudes and longitudes of the cut square for each footprint is stored in `data.fp_lats` and `data.fp_lons` (each of these has size (time, size))
  - Size should be even 
- Met:
  - Passing `cut_met=True` (and no `met_datadir`) does the same for the meteorology files cutting to `metsize`, outputting the met centered around the release point for each footprint where the where the coordinates are not lat-lon but 0,1,2,...size where int(size/2) is the release point.
  - However, this is very memory intensive so I have been running the above monthly, save that cut met file in `/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big`, and then load the pre-cut meteorology as above using `met_datadir`. You can pass any cut met file as long as the size is bigger than metsize though best to past the exact if it exists (eg `Met_cut_v2_50_` will be properly resized for `metsize=50` and below, but not for bigger sizes.
  - The met data is stored in `data.met`.
  - See below for running LoadSatelliteData to generate the cut meteorology files.
  - Currently `size` should be equal to  `metsize`.
- `freq` reduces the time frequency of the data before loading to reduce computational expense (ie `freq=3` will only load one in every three footprints and corresponding data).
- The function removes any datapoints where there are Nans in the met or fp data, be it because of a data problem or because the footprint is partially out of the domain.
- The topography for each footprint is stored at `data.topog` of size (time, size, size) 

### Preparing met files
See file `generate_sat_met.py` (and send to the cluster using `launch_cpu_job.sh`). Meteorology files are in `/group/chemistry/acrg/met_archive/UM/{domain}/{domain}_Met_`, at the same lat-lon resolution as the footprints and with hourly time resolution. Relevant pararameters:

- Here `metsize` needs to be the desired cutting size but `size` can be anything as it isn't used
- `met_levels` and `met_variables` define which levels and variables will be saved
- By default the meteorology is linearly interpolated to the timestamp of each footprint
- `met_jump` can be an int or a list. If a list, for each int `jump` in `met_jump`, the met is interpolated to `T - jump` where `T` is the timestamp of each footprint, and saved to the corresponding path in `savemetpath`.




