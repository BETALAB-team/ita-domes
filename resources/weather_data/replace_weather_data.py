# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:08:09 2024

@author: vivijac14771
"""

from funcs.io_functions import weather_template_import
import pickle
import glob
import pandas as pd
# import blosc

template_weather_file = 'WeatherData_CTI.pickle'
wd = weather_template_import(template_weather_file)
# template_weather_file = 'WeatherData_CTI_eureca_classes.dat'
# with open(template_weather_file, 'rb') as handle:
#     w_eureca_cp = handle.read()
# wd_eureca = pickle.loads(blosc.decompress(w_eureca_cp))

new_weather_file_folder = 'C:/Projects/meteo-data-processing/output-data-files'

anno = 2016

new_weather_files_path = new_weather_file_folder + '/DatiMeteo' + str(anno) + '/*.csv'
new_weather_files = glob.glob(new_weather_files_path)

new_wd = wd.copy()

for wf in new_weather_files:
    prov = wf.split('/')[-1].split('_CSV')[0][-2:]
    template_weather_data = wd[prov]
    
    new_weather_data = pd.read_csv(wf)
    
    template_weather_data['air_temp'] = new_weather_data['  TEMP']
    template_weather_data['rel_hum'] = new_weather_data['  UREL']
    template_weather_data['sol_ghi'] = new_weather_data['  RADG']
    template_weather_data['sol_dir'] = new_weather_data['  RDIR']
    template_weather_data['sol_dif'] = new_weather_data['  RDIF']
    template_weather_data['wind_vel'] = new_weather_data['  VELV']
    
    new_wd[prov] = template_weather_data
    print(prov)
    

new_weather_filename = 'WeatherData_' + str(anno) + '.pickle'     
with open(new_weather_filename, 'wb') as handle:
    pickle.dump(new_wd, handle, protocol=pickle.HIGHEST_PROTOCOL)