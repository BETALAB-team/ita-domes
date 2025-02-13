# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:34:51 2024

@author: vivijac14771
"""

import os
import io
import blosc
import pandas as pd
import numpy as np
import random
import pickle
import pvlib
# import pythermalcomfort.psychrometrics
from eureca_building.weather import WeatherFile
import time
import sys

from importlib import resources as impresources
from resources import istat_data as istat_data_file
from resources import HVAC_data as HVAC_data_file
from resources import user_data as user_data_file
from resources import appliances_data as appliances_data_file
from resources import conversion_factors as conversion_factors_file

def read_istat_data(year = 2013, selected_regions = None, number_of_buildings = None):
    
    tic = time.process_time()
    
    print("\nImporting ISTAT dataset..")
    istat_data = dict()
    
    # Import database ISTAT 2013
    istat_path = impresources.files(istat_data_file) / ('istat_microdata_' + str(year) + '.csv')
    
    istat_tot = pd.read_csv(istat_path, delimiter=';')
    istat_tot.index = istat_tot['id']  # edifici da 1 a 20000
    
    if not(selected_regions == None):
        ed_list = istat_tot.loc[istat_tot['reg'].isin(selected_regions)].index  #4,5,6
        istat_1 = istat_tot.loc[ed_list]    
        if not(number_of_buildings == None):
            ed_list_1 = select_subset(istat_1, number_of_buildings = number_of_buildings)
            istat = istat_1.loc[ed_list_1]
        else:
            ed_list_1 = 'all'
            istat = istat_1
    else:
        istat_1 = istat_tot
        if not(number_of_buildings == None):
            ed_list_1 = select_subset(istat_1, number_of_buildings = number_of_buildings)
            istat = istat_1.loc[ed_list_1]
        else:
            ed_list_1 = 'all'
            istat = istat_1
    # try:    
    #     istat = istat_1.drop('sample', axis=1)
    # except:
    #     istat = istat_1
        # print('No sample column to be removed')
        
    istat_data = {'year': year,
                  'istat_tot': istat_tot,
                  'istat': istat, 
                  'selected_buildings': ed_list_1
                  }
    
    comp_time = time.process_time() - tic
    print("[Import concluded in {:.2f} s]".format(comp_time))

    return istat_data


def select_subset(istat_df, number_of_buildings = None):
    ed_list = np.sort(random.sample(list(istat_df.id), number_of_buildings))
    return ed_list


def read_appliances_data(year = 2013):
    
    tic = time.process_time()
    
    print("\nImporting electrical appliances dataset..")
    
    # Import household appliances GfK
    appliances_file = impresources.files(appliances_data_file) / ('Database_elettrodomestici_' + str(year) + '.xlsx')
    
    # Lights
    db_lights = pd.read_excel(appliances_file,
                              sheet_name="illuminazioneMOIRAE", 
                              header=0,index_col=0)
    # Small household appliances
    db_little_appliances = pd.read_excel(appliances_file,
                                         sheet_name="piccoli_elettrodomestici_MOIRAE", 
                                         header=0,index_col=0)
    # Fridges and freezers
    db_refrigerators = pd.read_excel(appliances_file,
                                     sheet_name="frigoriferi", 
                                     header=0,index_col=0)
    # Big household appliances (washing machine, tumble dryers etc)
    db_appliances = pd.read_excel(appliances_file,
                                  sheet_name="grandi_elettrodomestici", 
                                  header=0,index_col=0)
    # TVs
    db_TVs = pd.read_excel(appliances_file,
                           sheet_name="schermi_MOIRAE", 
                           header=0,index_col=0)
    # Cooking devices 
    db_cookings = pd.read_excel(appliances_file,
                                sheet_name="cottura_cibi", 
                                header=0,index_col=0)
    # Cooling systems
    db_cool = pd.read_excel(appliances_file,
                            sheet_name="Raffrescamento", 
                            header=0, index_col=[0,1,2], usecols = "B,E:G")
    db_cool.sort_index(inplace = True)
    db_cool_cor = pd.read_excel(appliances_file,
                                sheet_name="CorrezioneRaffrescamento", 
                                header=0, index_col=[0,1], usecols = "A,C,E")
    db_cool_cor.sort_index(inplace = True)
    
    # Standby 
    db_standby = pd.read_excel(appliances_file,
                               sheet_name="Standby", 
                               header=0)
    # DHW 
    db_dhw = pd.read_excel(appliances_file,
                           sheet_name="ACS", 
                           header=0,index_col=0)
    
    appliances_data = {'year': year,
                       'db_lights': db_lights, 
                       'db_little_appliances': db_little_appliances,
                       'db_refrigerators': db_refrigerators,
                       'db_appliances': db_appliances,
                       'db_TVs': db_TVs,
                       'db_cookings': db_cookings,
                       'db_cool': db_cool,
                       'db_cool_cor': db_cool_cor,
                       'db_standby': db_standby,
                       'db_dhw': db_dhw
                       }  
    
    comp_time = time.process_time() - tic
    print("[Import concluded in {:.2f} s]".format(comp_time))
    
    return appliances_data



def read_weather_data(year = 'CTI', replace_data = True, month_resample = True, **kwargs):
    
    # irradiances_calculation = kwargs['irradiances_calculation']
    
    tic = time.process_time()
    
    print("\nImporting weather data..")
    
    if replace_data == True:
        weather_file = 'WeatherData_' + str(year) + '.pickle'
    else:
        weather_file = 'WeatherData_CTI.pickle'
        
    weather_path = os.path.join(main_wd, 'resources', 'weather_data', 
                                weather_file)
    # read weather data
    w = pd.read_pickle(weather_path)
    
    # ----------------------------------------------------------------------
    #                         to be replaced 
    
    print('Reading weather file classes from CTI reference year')
    weather_file_eureca = 'WeatherData_CTI_eureca_classes.dat'
    weather_path_eureca = os.path.join(main_wd, 'resources', 'weather_data', 
                                weather_file_eureca)
    with open(weather_path_eureca, 'rb') as handle:
        w_eureca_cp = handle.read()
    w_eureca = pickle.loads(blosc.decompress(w_eureca_cp))
    # ----------------------------------------------------------------------
    #                            new code
    if replace_data == True:
        # irradiances_calculation = True
        w_eureca = replace_eureca_weather_classes(w, 
                                                  w_eureca, 
                                                  irradiances_calculation = True)
    
    # ----------------------------------------------------------------------
        
    print('Reading template weather file from CTI reference year')
    weather_data = dict()
    weather_data['hourly'] = w
    weather_data['eureca_class'] = w_eureca
    if month_resample == True:
        w_resampled = resample_weather_data(w)
        weather_data['monthly'] = w_resampled
        
    comp_time = time.process_time() - tic
    print("[Import concluded in {:.2f} s]".format(comp_time))
         
    return weather_data


def resample_weather_data(weather_df):
    provs = list(weather_df.keys())
    w = dict()
    w_vars = ['air_temp','rel_hum','sol_ghi','wind_vel']
    for prov in provs: 
        w[prov] = pd.DataFrame(columns=w_vars)
        for var in w_vars:
            w[prov][var] = weather_df[prov].groupby('month')[var].mean()
    return w
  
      
def read_envelopes_data(selected_buildings = None):
    
    tic = time.process_time()
    
    print("\nImporting building envelopes information (already processed)..")
    # Import building envelopes
    envelope_file = 'envelopes.pickle'
    envelope_path = os.path.join(main_wd, 'resources', 'envelope_data', 
                                 envelope_file)
    selected_envelopes = dict()
    with open(envelope_path, 'rb') as handle:
        envelopes = pickle.load(handle) 
    if selected_buildings is None:
        selected_envelopes = envelopes
    else:
        # selected_envelopes = [envelopes[s] for s in selected_buildings]
        for s in selected_buildings:
            selected_envelopes.update({s: envelopes[s]})
            
    comp_time = time.process_time() - tic
    print("[Import concluded in {:.2f} s]".format(comp_time))
    
    return selected_envelopes


def save_output(res):
    tic = time.process_time()   
    print("\nSaving simulation results into output file..")
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    output_file = 'output_' + time_string + '.pickle'
    output_path = os.path.join(main_wd, 'output', output_file)
    output = dict()
    # output['consumption_appliances'] = consumption_appliances
    # output['consumption_hvac'] = consumption_hvac
    # output['buildings_data'] = buildings_data
    output['res'] = res
    with open(output_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    comp_time = time.process_time() - tic
    print("[Results successfully saved in {:.2f} s]".format(comp_time))
    return

def import_output(filename):
    main_wd = sys.path[0]
    results_path = os.path.join(main_wd, 'output', filename)
    with open(results_path, 'rb') as handle:
        output = pickle.load(handle)
    # consumption_appliances = output['consumption_appliances']
    # consumption_hvac = output['consumption_hvac']
    # buildings_data = output['buildings_data']
    res = output['res']
    return res #consumption_appliances, consumption_hvac, buildings_data

def weather_template_import(filename):
    # weather_path = os.path.join(main_wd, 'resources', 'weather_data', filename)
    # with open(filename, 'rb') as handle:
    #     w = pickle.load(handle)
    w = pd.read_pickle(filename)
    
    return w

def replace_eureca_weather_classes(w, w_eureca, irradiances_calculation = True):
    
    # path to template epw file (to be changed according to region)
    # epw_template_file = os.path.join(main_wd, 'resources', 'weather_data', 
    #                                  'epw_template_files', 'ITA_Rome.162420_IWEC.epw')
    w_eureca_new = w_eureca
    provs = list(w.keys())
    # w_vars = ['air_temp','rel_hum','sol_ghi','wind_vel']
    for prov in provs:  
        # epw_template_file = w_eureca[prov]
        # w_eureca_new[prov] = WeatherFile(epw = epw_template_file, 
        #                              year=None, 
        #                              time_steps= 1, 
        #                              irradiances_calculation= True, 
        #                              azimuth_subdivisions = 8, 
        #                              height_subdivisions = 3, 
        #                              urban_shading_tol=[80., 100., 80.], 
        #                              new_weather_data = w[prov])
        # w_class = w_eureca[prov]
        w_eureca_new[prov].replace_weather_data(irradiances_calculation = irradiances_calculation, 
                                                new_weather_data = w[prov])
    
    return w_eureca_new



