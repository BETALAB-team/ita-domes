# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:27:49 2024

@author: vivijac14771
"""


import sys
sys.path.insert(0, r"C:\Projects\GRINS-Tool")
from funcs.io_functions import import_output
from classes.Results import PostProcess

#%%
# res = import_output('output_2024-11-25_19-56-49.pickle')  #  #
# results = res.all
# results['info'] = {'model': '1C', 'um': 'GWh'}
# # results = res


#%%
res = import_output('output_weather2013_v3.pickle')  #  #output_2024-11-25_19-56-49.pickle
# results = res.all
# results['info'] = {'model': '1C', 'um': 'GWh'}
results = res.resume



#%%
FinalEnergy = PostProcess(results, 
                          kpi = 'final_energy',    
                          units = 'GWh')

results = FinalEnergy.update_results()

# PrimaryEnergy = PostProcess(results, 
#                             kpi = 'primary_energy',    
#                             units = 'ktep')

# Emissions = PostProcess(results, 
#                         kpi = 'CO2_emissions',    
#                         units = 'ktep')
print(FinalEnergy.df_carrier_region.sum()/11.63)

#%% visualizzazione anno con suddivisione geografica (mappa regioni)
# FinalEnergy.heatmaps(group_by = 'use', 
#                      specify = 'pro_capite' # 'specific'
#                      ) # 0: electricity, 1: gas , 2: gpl , 3: biomass


# visualizz
