import sys


import numpy as np

from graphnet_LPDM_emulator.model.data.load_data import *
from datetime import datetime

print("starting")
for year in ["2014", "2015", "2016"]:
    for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
        try:
            print(year+month, datetime.now())
            data = LoadSatelliteData(year+month, region="BRAZIL", met_jump=[0,6,12], metsize=100, size = 10, met_levels=[3,9,15,21,30,42,51], met_variables=["air_pressure", "air_temperature", "atmosphere_boundary_layer_thickness", "surface_air_pressure", "upward_air_velocity", "x_wind", "y_wind"], verbose=True, cut_met = True, met_datadir="/group/chemistry/acrg/met_archive/UM/SOUTHAMERICA/SOUTHAMERICA_Met_", savemetpath=[f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_onlyvalid_100_{year}{month}.nc", f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_onlyvalid_100_6h_{year}{month}.nc", f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_onlyvalid_100_12h_{year}{month}.nc"], savemet=True)

        except Exception as e:
            print(e)
            continue

#            data = LoadSatelliteData(year+month, region="BRAZIL",  met_jump=[12,24], metsize=50, size = 10, met_levels=[3,9,15,21,30,42,51], met_variables=["air_pressure", "air_temperature", "atmosphere_boundary_layer_thickness", "surface_air_pressure", "upward_air_velocity", "x_wind", "y_wind"], verbose=True, cut_met = True, met_datadir="/group/chemistry/acrg/met_archive/UM/SOUTHAMERICA/SOUTHAMERICA_Met_", savemetpath=[f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_12h_{year}{month}.nc", f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_24h_{year}{month}.nc"], savemet=True)

#data = LoadSatelliteData(year+month, region="BRAZIL",  met_jump=[6,12,24], metsize=50, size = 10, met_levels=[3,9,15,21,30,42,51], met_variables=["air_pressure", "air_temperature", "atmosphere_boundary_layer_thickness", "surface_air_pressure", "upward_air_velocity", "x_wind", "y_wind"], verbose=True, cut_met = True, met_datadir="/group/chemistry/acrg/met_archive/UM/SOUTHAMERICA/SOUTHAMERICA_Met_", savemetpath=[f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_6h_{year}{month}.nc", f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_12h_{year}{month}.nc", f"/group/chemistry/acrg/met_archive/UM/cut_SOUTHAMERICA_big/Met_cut_v2_50_24h_{year}{month}.nc"], savemet=True)