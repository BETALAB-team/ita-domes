# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:08:26 2024

@author: vivijac14771
"""
import os
import pickle
from classes.Envelope import loadEnvelopes
from classes.Archetype import Archetypes
from funcs.io_functions import read_istat_data
from funcs.aux_functions import fascia_climatica # regioni

cwd = os.getcwd()
envelope_file = 'Buildings_Envelope'
env_path = os.path.join(cwd,envelope_file)

# os.chdir('..\..')

# lettura file istat 
istat_data = read_istat_data(year = 2013, 
                             selected_regions = None,
                             number_of_buildings = 20000)   # to be replaced with number_of_buildings

istat = istat_data['istat']
istat['Fascia_clima'] = istat['reg'].apply(fascia_climatica)

total_envelopes = {}

slots = {
    1: [0,4000],
    2: [4000,8000],
    3: [8000,12000],
    4: [12000,16000],
    5: [16000,20000],
}

total_env = {}

for s, s_lim in slots.items():
    total_envelopes = {}

    for a in range(s_lim[0],s_lim[1]):

        i = a + 1

        _ = print(i) if i%100 == 0 else None

        # Building archetype class based on ISTAT survey
        archetype = Archetypes(i,istat)


        envelopes = loadEnvelopes(env_path,
                                  archetype.archID,
                                  archetype.insulator_layer,
                                  retrofits = {
                                      "Wins" : False,
                                      "Walls" : False,
                                      },
                                  zone = istat.loc[i]['Fascia_clima'])
        total_envelopes[i] = [elem for elem in envelopes.values()][0]

    total_env.update(total_envelopes)

    with open(f'envelopes_{s}.pickle', 'wb') as handle:
        pickle.dump(total_envelopes, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'envelopes.pickle', 'wb') as handle:
    pickle.dump(total_env, handle, protocol=pickle.HIGHEST_PROTOCOL)