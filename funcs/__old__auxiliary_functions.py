''' IMPORTING MODULES'''

import os
from warnings import warn
import numpy as np
import pandas as pd


#%% ---------------------------------------------------------------------------------------------------
#%% Useful functions

'''
Some auxiliary functions for the tool
'''

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
    
def file_climatico(zona):
    # Naturalmente semplificato, non abbiamo il comune
        file ={
                'Regione': 'Regione',
                'A': 'Palermo',
                'B': 'Palermo',
                'C': 'Palermo',
                'D': 'Rome',
                'E': 'Milano',
                'F': 'Bolzano',  
                }
        return file[zona]
    
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
     
def scegli_provincia(reg):
    return {'Piemonte': 'TO',
         'ValledAosta': 'AO',
         'Lombardia': 'MI',
         'TrentinoAltoAdige': 'TN',
         'Veneto': 'VE',
         'FriuliVeneziaGiulia': 'UD',
         'Liguria': 'GE',
         'EmiliaRomagna': 'BO',
         'Toscana': 'FI',
         'Umbria': 'PG',
         'Marche': 'AN',
         'Lazio': 'RM',
         'Abruzzo': 'AQ',
         'Molise': 'CB',
         'Campania': 'NA',
         'Puglia': 'BA',
         'Basilicata': 'PZ',
         'Calabria': 'RC',
         'Sicilia': 'PA',
         'Sardegna': 'CA'
            }[reg]


    
list_file_climatico= {'Venezia':'ITA_Venezia-Tessera.161050_IGDG.epw',
                      'Bolzano':'ITA_Bolzano.160200_IGDG.epw',
                      'Palermo':'ITA_Palermo.164050_IWEC.epw',
                      'Rome':'ITA_Rome.162420_IWEC.epw',
                      'Milano':'ITA_Milano-Malpensa.160660_IGDG.epw'}

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
         20:  'Sardegna'
            }

list_2_file_climatico = {1: 'ITA_Torino-Caselle.160590_IGDG.epw',
          2: 'ITA_Torino-Caselle.160590_IGDG.epw',
          3: 'ITA_Milano-Malpensa.160660_IGDG.epw',
          4: 'ITA_Bolzano.160200_IGDG.epw',
          5:  'ITA_Venezia-Tessera.161050_IGDG.epw',
          6:  'ITA_Udine-Campoformido.160440_IGDG.epw',
          7:  'ITA_Genova-Sestri.161200_IGDG.epw',
          8:  'ITA_Bologna-Borgo.Panigale.161400_IGDG.epw',
          9:  'ITA_Firenze-Peretola.161700_IGDG.epw',
          10:  'ITA_Perugia.161810_IGDG.epw',
          11:  'ITA_Ancona.161910_IGDG.epw',
          12:  'ITA_Roma-Fiumicino.162420_IGDG.epw',
          13:  'ITA_Pescara.162300_IGDG.epw',
          14:  'ITA_Campobasso.162520_IGDG.epw',
          15:  'ITA_Napoli-Capodichino.162890_IGDG.epw',
          16:  'ITA_Bari-Palese.Macchie.162700_IGDG.epw',
          17:  'ITA_Potenza.163000_IGDG.epw',
          18:  'ITA_Crotone.163500_IGDG.epw',
          19:  'ITA_Palermo-Punta.Raisi.164050_IGDG.epw',
          20:  'ITA_Cagliari-Elmas.165600_IGDG.epw'
            }

regioni_rev = {'Piemonte': 1,
         'ValledAosta': 2,
         'Lombardia': 3,
         'TrentinoAltoAdige': 4,
         'Veneto': 5,
         'FriuliVeneziaGiulia': 6,
         'Liguria': 7,
         'EmiliaRomagna': 8,
         'Toscana': 9,
         'Umbria': 10,
         'Marche': 11,
         'Lazio': 12,
         'Abruzzo': 13,
         'Molise': 14,
         'Campania': 15,
         'Puglia': 16,
         'Basilicata': 17,
         'Calabria': 18,
         'Sicilia': 19,
         'Sardegna': 20
            }
