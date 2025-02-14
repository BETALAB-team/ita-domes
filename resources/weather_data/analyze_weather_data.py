# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:54:17 2024

@author: vivijac14771
"""

from funcs.io_functions import weather_template_import
import pickle
import glob
import pandas as pd
import os

def select_data_from_locations(wdata, variable, year, locations):    
    selected_data = pd.DataFrame(columns=locations)
    for loc in locations:
        selected_data[loc] = wdata[year][loc][variable]
    return selected_data

def average_data_from_locations(wdata, locations_north, locations_centr, locations_south):
    for key in wdata:
        wdata[key]['north'] = wdata[key][locations_north].mean(axis = 1)
        wdata[key]['centr'] = wdata[key][locations_centr].mean(axis = 1)
        wdata[key]['south'] = wdata[key][locations_south].mean(axis = 1)
    return wdata

def calc_degree_days(tdata, baseline_temp, zones):
    days_start = 104
    days_end = 76
    hours_start = days_start*24
    hours_end = days_end*24 
    dd = dict()
    dd = pd.DataFrame(columns=tdata.keys())
    for key in tdata:        
        for zone in zones:
            dd_start = baseline_temp - tdata[key][zone][0:hours_start]
            dd_end = baseline_temp - tdata[key][zone][8760-hours_end:8760]
            dd.loc[zone,key] = (sum(dd_start.values) + sum(dd_end.values))/24
    return dd

weather_files_path = os.getcwd() + '/*.pickle'
weather_files = glob.glob(weather_files_path)

wdata = dict()

for wf in weather_files:
    label = wf.split('\\')[-1].split('.')[0].split('_')[-1]
    wdata[label] = weather_template_import(wf)

#%%    

years = ['CTI','2013', '2014', '2015','2016', '2017','2018','2019','2020','2021']
locs_north = ['TO', 'MI', 'PD', 'BO']
locs_centr = ['FI', 'PG', 'AN', 'RM']
locs_south = ['BA', 'NA', 'RC', 'PA']
locations = locs_north + locs_centr + locs_south

temp_data = dict()
for year in years:
    temp_data[year] = select_data_from_locations(wdata, 'air_temp', year, locations)

#%%
temp_data = average_data_from_locations(temp_data, 
                                        locs_north, locs_centr, locs_south)

degree_days = calc_degree_days(temp_data, 20.0, ['north', 'centr', 'south'])
