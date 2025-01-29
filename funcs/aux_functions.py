# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:14:44 2024

@author: vivijac14771
"""

import pandas as pd
import os
import numpy as np
import sys
global main_wd 
main_wd = sys.path[0]


#%%

regioni = {1: 'Piemonte',
         2: 'ValledAosta',
         3: 'Lombardia',
         4: 'TrentinoAltoAdige',
         5:  'Veneto',
         6:  'FriuliVeneziaGiulia',
         7:  'Liguria',
         8:  'EmiliaRomagna',
         9:  'Toscana',
         10:  'Umbria',
         11:  'Marche',
         12:  'Lazio',
         13:  'Abruzzo',
         14:  'Molise',
         15:  'Campania',
         16:  'Puglia',
         17:  'Basilicata',
         18:  'Calabria',
         19:  'Sicilia',
         20:  'Sardegna'}


#%%

def wrn(message):
    '''
    This function takes a string and print it in the log file
    
    Parameters
        ----------
        message : string
            message to be printed in the log file
    '''
    
    if not isinstance(message, str):
        try:
            message = str(message)
        except:
            raise TypeError(f'wrnn, message must be a string: message {message}')
    
    if not os.path.isdir(os.path.join('.','Resources')):
         os.mkdir('Resources')    
    output_path = os.path.join('.','Resources')
    
    if not os.path.isfile(os.path.join(output_path,'warnings.txt')):
         f = open(os.path.join(output_path,'warnings.txt'),"w")
         f.close()
         
    with open(os.path.join(output_path,'warnings.txt'),"a") as f:
        f.write(message + '\n')
    # warn(message)
    
#%%

def scegli_provincia_distr(reg, rand, sort_by = 'Numero_edifici'):
    
    census = pd.read_csv(os.path.join(main_wd,'resources','istat_data','residential_buildings_census.csv'), 
                         keep_default_na=False, index_col=0)
    # napoli = pd.DataFrame({'prov':'NA','Regione':'Campania','Numero_edifici':292920,'Persone_edificio':10.37}, index=[0])
    # census = pd.concat([census,napoli],ignore_index = True)
    # census = census.set_index('prov')
    
    census_reg = census[census['Regione']==reg]

    sorted_census = census_reg.sort_values(by='Numero_edifici', ascending=False)
    sorted_census['Ratio'] = sorted_census['Numero_edifici']/sum(sorted_census['Numero_edifici'])
    
    if sort_by == 'Numero_persone':
        sorted_census['Numero_persone'] = np.multiply(sorted_census['Numero_edifici'].values, 
                                                      sorted_census['Persone_edificio'].values)
        sorted_census['Ratio'] = sorted_census['Numero_persone']/sum(sorted_census['Numero_persone'])
    
    sorted_census['Prob'] = np.cumsum(sorted_census['Ratio'])

    sorted_census['lower'] = sorted_census['Prob']<rand

    prov = sorted_census.loc[sorted_census['lower'] == False]['Prob'].idxmin()
    # print(reg)
    # print(prov)
    # print('Selected province for region ' + reg + ' is ' + prov)
    
    return prov

#%%

def simplify_geom(S_floor):
    H = 3.
    if S_floor < 80:
        L1 = 6.
    else:
        L1 = 10.     
    L2 =  S_floor/L1
    A_tot = (L1 + L2)*H
    WWR = 0.125
    A_op  = (1 - WWR)*A_tot
    A_fin = A_tot - A_op
    V = S_floor*H   
    return A_op, A_fin, V


#%%

def fascia_climatica(cod_reg):
    # Naturalmente semplificato, non abbiamo il comune
        climatica ={
                'Regione': 'Regione',
                1: 'E', # Piemonte    
                2: 'F', #  Valle d'Aosta    
                3: 'E', #  Lombardia    
                4: 'F', #  Trentino-Alto Adige    
                5: 'E', #  Veneto    
                6: 'E', #  Friuli-Venezia Giulia    
                7: 'D', #  Liguria    
                8: 'E', #  Emilia-Romagna    
                9: 'D', #  Toscana    
                10: 'E', #  Umbria    
                11: 'D', #  Marche    
                12: 'D', #  Lazio    
                13: 'E', #  Abruzzo    
                14: 'D', #  Molise    
                15: 'D',#  Campania    
                16: 'C',#  Puglia    
                17: 'E',#  Basilicata    
                18: 'D',#  Calabria    
                19: 'C', #  Sicilia    
                20: 'C'#  Sardegna 
                }
        return climatica[cod_reg]

#%% from auxiliary_functions()

def weightCoeff(istat):
    weight = istat
    weight.rename(columns = {"coef_red":"Weights"}, 
             inplace = True)  
    return weight['Weights']


'''Assign Region (str) name to each ISTAT building'''
def decodificRegion(Regione):
                                 
    region={}
    for k in Regione.index:
        r = Regione[k]
        if r == 1: reg='Piemonte'               
        elif r == 2: reg = 'ValledAosta'
        elif r == 3: reg = 'Lombardia'
        elif r == 4: reg = 'TrentinoAltoAdige'
        elif r == 5: reg = 'Veneto'
        elif r == 6: reg = 'FriuliVeneziaGiulia'
        elif r == 7: reg = 'Liguria'
        elif r == 8: reg = 'EmiliaRomagna'
        elif r == 9: reg = 'Toscana'
        elif r == 10: reg = 'Umbria'
        elif r == 11: reg = 'Marche'
        elif r == 12: reg = 'Lazio'
        elif r == 13: reg = 'Abruzzo'
        elif r == 14: reg = 'Molise'
        elif r == 15: reg = 'Campania'
        elif r == 16: reg = 'Puglia'
        elif r == 17: reg = 'Basilicata'
        elif r == 18: reg = 'Calabria'
        elif r == 19: reg = 'Sicilia'
        elif r == 20: reg = 'Sardegna'
        region[k] = reg
    return region

def decodificRegion2(cod_reg):
                                 
    return {1: 'Piemonte',
         2: 'ValledAosta',
         3: 'Lombardia',
         4: 'TrentinoAltoAdige',
         5:  'Veneto',
         6:  'FriuliVeneziaGiulia',
         7:  'Liguria',
         8:  'EmiliaRomagna',
         9:  'Toscana',
         10:  'Umbria',
         11:  'Marche',
         12:  'Lazio',
         13:  'Abruzzo',
         14:  'Molise',
         15:  'Campania',
         16:  'Puglia',
         17:  'Basilicata',
         18:  'Calabria',
         19:  'Sicilia',
         20:  'Sardegna'
            }[cod_reg]