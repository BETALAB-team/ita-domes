# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:05:03 2024

@author: vivijac14771
"""

import sys
sys.path.insert(0, r"C:\Projects\GRINS-Tool")

from funcs.io_functions import read_istat_data, read_appliances_data, read_weather_data, read_envelopes_data
from funcs.main_functions import process_building_data, process_users_data, process_HVAC_data, process_results 
from funcs.main_functions import simulate_appliances, simulate_hvac
import random
import logging
random.seed(10)
logging.getLogger().setLevel(logging.CRITICAL)


#%% (1) Lettura e importazione dei dati di input

# lettura file istat 
istat_data = read_istat_data(year = 2013, 
                             selected_regions = None,
                             number_of_buildings = 10) #None

# lettura file elettrodomestici
appliances_data = read_appliances_data(year = 2013)

# lettura file climatici (CTI, ARPA)
weather_data = read_weather_data(year = 2013,   
                                 replace_data = True,
                                 month_resample = True)

envelopes_data = read_envelopes_data(istat_data['selected_buildings'])


#%% (2) Simulazione elettrodomestici e luci
consumption_appliances = simulate_appliances(appliances_data, istat_data)


#%% (3) Pre-processing dati 

# lettura file involucri edilizi e utenti (informazioni già processate da file istat)
users_data = process_users_data(istat_data, consumption_appliances)

# associazione edificio, impianti e elettrodomestici in base alle risposte istat
buildings_data = process_building_data(istat_data, envelopes_data, consumption_appliances, users_data)

# associazione edificio, impianti e elettrodomestici in base alle risposte istat
buildings_data, _ = process_HVAC_data(istat_data, buildings_data, consumption_appliances)
# info_impianti = _


#%% (4) Simulazione edifici

# simulazione heating and cooling
consumption_hvac = simulate_hvac(buildings_data, weather_data, output_folder = None, model = '1C')


#%% (5) Processing dei risultati

# summary risultati per ogni utente
output_path = "."
res = process_results(output_path, consumption_appliances, consumption_hvac, buildings_data,
                      save_results = True)


#%%



