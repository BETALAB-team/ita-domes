# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:32:07 2024

@author: vivijac14771
"""

import os

import pickle
import numpy as np
# from classes.Envelope import loadEnvelopes
# from classes.Archetype import Archetypes
from funcs.io_functions import read_istat_data
from classes.EndUse import initializeSchedules
# from funcs.aux_functions import fascia_climatica # regioni

# cwd = os.getcwd()
# envelope_file = 'Buildings_Envelope'
# env_path = os.path.join(cwd,envelope_file)

#%%

def create_standard_users():
    ''' Schedule loading'''
    # Si carica un foglio di schedule e si creaano gli oggetti Archetype
    cwd = os.getcwd()
    standard_schedules_file = 'schedules_EN16798_Residential.xlsx'
    schedpath = os.path.join(cwd, standard_schedules_file)                   # Annual schedules
    # schedpath = os.path.join('resources','user_data','schedules_EN16798_Residential.xlsx')     
    time      = np.arange(8760)
    standard_users  = initializeSchedules(schedpath,time,1)
    return standard_users


#%% 

standard_users = create_standard_users()


with open('standard_users.pickle', 'wb') as handle:
    pickle.dump(standard_users, handle, protocol=pickle.HIGHEST_PROTOCOL)

# users = {}


#%%

# os.chdir('..\..')

# # lettura file istat 
# istat_data = read_istat_data(year = 2013, 
#                              selected_regions = None, # None
#                              number_of_buildings = None)   #None

# # lettura file istat 
# istat_data = read_istat_data(year = year, 
#                              selected_regions = [4,5,6], # None
#                              number_of_buildings = 10)   #None


    
#%%

    
    
    
    
    
