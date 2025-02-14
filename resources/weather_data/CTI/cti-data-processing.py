# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:15:30 2024

@author: vivijac14771
"""

import os
import io
import glob
import blosc
import pandas as pd
import numpy as np
import pickle
import pvlib

import pythermalcomfort.utilities
from eureca_building.weather import WeatherFile


#%%
cti_file_paths = glob.glob('*.xlsx')
weather = dict()

for file in cti_file_paths:
    region_name = file.split('.')[0]
    weather[region_name] = pd.read_excel(file,
                                          sheet_name=None, 
                                          header=0, index_col=0)

    weather[region_name]['Stazioni'].fillna('NA', inplace = True)

    del weather[region_name]['Copyright']
    num_province = len(weather[region_name].keys()) - 1
    info_stazioni = weather[region_name]['Stazioni'].iloc[4:4+num_province,:]
    info_stazioni = info_stazioni.rename(columns={'Unnamed: 1': 'Prov', "Unnamed: 2": 'Localita'})
    info_stazioni = info_stazioni.set_index('Prov')

    info_stazioni["Long mod"] = ((info_stazioni["Unnamed: 5"]/60) + info_stazioni["Unnamed: 4"])/60 + info_stazioni["Unnamed: 3"]
    info_stazioni["Lat mod"] = ((info_stazioni["Unnamed: 8"]/60) + info_stazioni["Unnamed: 7"])/60 + info_stazioni["Unnamed: 6"]
    info_stazioni["Quota"] = info_stazioni["Unnamed: 9"]

    weather[region_name]['Stazioni'] = info_stazioni

#%%
regions = list(weather.keys())
w = dict()
w_eureca = dict()
for reg in regions:
    print(reg)
    provs = list(weather[reg].keys())
    w_eureca_reg = dict()
    for prov in provs:
        if not(prov == 'Stazioni'):
            
            info_staz = weather[reg]["Stazioni"].loc[prov].to_dict()
            
            lat = info_staz["Lat mod"]
            long = info_staz["Long mod"]
            altit = info_staz["Quota"]
            
            pv_lib_loc = pvlib.location.Location(lat, long, tz="CET", altitude = altit)
            pv_lib_loc = pv_lib_loc.get_solarposition(times=pd.date_range("2014/12/31 23:00", freq = "1h", periods = 8760))
            
            pv_lib_loc['cos zenith'] = np.cos(np.deg2rad(pv_lib_loc['apparent_zenith']))
            # pv_lib_loc['cos zenith'][pv_lib_loc['apparent_zenith'] > 88] = np.nan
            pv_lib_loc.loc[pv_lib_loc['apparent_zenith'] > 88, 'cos zenith'] = np.nan

            dir_norm = np.nan_to_num(weather[reg][prov]['  RDIR'].values / pv_lib_loc['cos zenith'].values)
            

            w[prov] = pd.DataFrame(columns=['hour','hour_day','month','air_temp','rel_hum',
                                            'sol_ghi','sol_dir_nor', 'sol_dir_hor','sol_dif','wind_vel'])
            
            w[prov]['air_temp'] = weather[reg][prov]['  TEMP']
            w[prov]['rel_hum'] = weather[reg][prov]['  UREL']
            w[prov]['sol_ghi'] = weather[reg][prov]['  RADG']
            w[prov]['sol_dir_hor'] = weather[reg][prov]['  RDIR']
            w[prov]['sol_dir_nor'] = dir_norm
            w[prov]['sol_dif'] = weather[reg][prov]['  RDIF']
            w[prov]['wind_vel'] = weather[reg][prov]['  VELV']
            w[prov]['hour_day'] = weather[reg][prov][' HH']
            w[prov]['month'] = weather[reg][prov].index
            w[prov]['hour'] = range(0,8760,1)
            w[prov] = w[prov].set_index('hour')
            

            header = f"""LOCATION,{prov},-,ITA,IGDG,161050,{lat},{long},1.0,{altit}
    DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,-4,-2.8,-11,1.5,0.1,-8.8,1.8,0.6,13.2,2.6,10.9,2.8,1.8,50,Cooling,7,8.8,31.1,23.5,29.9,23,28.8,22.3,25.3,28.9,24.3,28.1,23.4,27.3,2.8,160,24.1,19,27.4,23.1,17.8,26.8,22,16.7,26.1,77.2,28.9,73.1,27.9,69.6,27.2,878,Extremes,9.5,7.6,6.2,29.4,-6.5,33.2,1.9,1.8,-7.8,34.5,-8.9,35.6,-9.9,36.6,-11.3,37.9
    TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature For Period,Extreme,7/13,7/19,Summer - Week Nearest Average Temperature For Period,Typical,8/17,8/23,Winter - Week Nearest Min Temperature For Period,Extreme,2/10,2/16,Winter - Week Nearest Average Temperature For Period,Typical,1/13,1/19,Autumn - Week Nearest Average Temperature For Period,Typical,10/13,10/19,Spring - Week Nearest Average Temperature For Period,Typical,4/26,5/ 2
    GROUND TEMPERATURES,3,.5,,,,4.39,3.54,5.05,7.43,13.65,18.45,21.68,22.67,20.99,17.27,12.29,7.67,2,,,,7.54,6.02,6.33,7.59,11.75,15.51,18.52,20.14,19.79,17.68,14.25,10.61,4,,,,10.15,8.59,8.24,8.72,11.07,13.61,15.93,17.55,17.93,17.01,15.00,12.53
    HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0
    COMMENTS 1,Custom/User Format -- WMO#161050; Italian Climate Data Set Gianni de Giorgio; Period of record 1951-1970
    COMMENTS 2, -- Ground temps produced with a standard soil diffusivity of 2.3225760E-03 [m**2/day]
    DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31
    """
            
            air_temp = w[prov]["air_temp"].values
            rel_hum = w[prov]["rel_hum"].values
            t_dp = pythermalcomfort.utilities.dew_point_tmp(air_temp,rel_hum)
            sol_ghi = w[prov]["sol_ghi"].values
            sol_dir_nor = w[prov]["sol_dir_nor"].values
            sol_dif = w[prov]["sol_dif"].values
            wind_vel = w[prov]["wind_vel"].values
            
            data = pd.DataFrame(index = pd.date_range("2005/01/01 00:00", periods=8760, freq="1h"))
            data["year"] = 2007
            data["month"] = data.index.month
            data["day"] = data.index.day
            data["hour"] = data.index.hour + 1
            data["minutes"] = 0
            data["source"] = "__"
            data["temp_air"] = air_temp
            data["temp_dew"] = t_dp
            data["relative_humidity"] = rel_hum
            data["atmospheric_pressure"] = 101205
            data["etr"] = 9999
            data["etrn"] = 9999
            data["ghi_infrared"] = 9999
            data["ghi"] = sol_ghi
            data["dni"] = sol_dir_nor
            data["dhi"] = sol_dif
            data["global_hor_ill"] = 999900
            data["direct_normal_ill"] = 999900
            data["diffuse_normal_ill"] = 999900
            data["zenith_luminance"] = 9999
            data["wind_direction"] = 999
            data["wind_speed"] = wind_vel
            data["total_sky_cover"] = 99
            data["opaque_sky_cover"] = 1
            data["visibility"] = 9999
            data["ceiling_height"] = 99999
            
            
            data["present_weather_observation"] = 0
            data["present_weather_codes"] = 0
            data["precipitable_water"] = 0
            data["aerosol_optical_depth"] = 0
            data["snow_depth"] = 0
            data["days_since_last_snowfall"] = 0
            data["albedo"] = 0
            data["liquid_precipitation_depth"] = 0
            data["liquid_precipitation_quantity"] = 0
            
            np_data = data.to_csv(index = False, header = False)
            
            epw_string = header + np_data
            
            epw = io.StringIO(epw_string)
            
            w_eureca[prov] = WeatherFile(epw)
            w_eureca_reg[prov] = w_eureca[prov]
    cp_file = blosc.compress(pickle.dumps(w_eureca_reg))
    with open(f'WeatherData_CTI_eureca_classes_{reg}.dat', 'wb') as handle:
        handle.write(cp_file)

            
#%%
# with open('WeatherData_CTI.pickle', 'wb') as handle:
#     pickle.dump(w, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # with open('WeatherData_CTI_eureca_classes.pickle', 'wb') as handle:
# #     pickle.dump(w_eureca, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# cp_file = blosc.compress(pickle.dumps(w_eureca))
# with open('WeatherData_CTI_eureca_classes.dat', 'wb') as handle:
#     handle.write(cp_file)
