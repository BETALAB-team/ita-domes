 # -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:45:36 2021

@author: fabix
"""

import pandas as pd
from pandas import DataFrame
import os
import numpy as np
from funcs.aux_functions import weightCoeff, decodificRegion
import copy

# def weightCoeff(istat):
#     weight = istat
#     weight.rename(columns = {"coef_red":"Weights"}, 
#              inplace = True)  
#     return weight['Weights']

# '''Assign Region (str) name to each ISTAT building'''
# def decodificRegion(Regione):
                                 
#     region={}
#     for k in range(len(Regione)):
#         r = Regione[k]
#         if r == 1: reg='Piemonte'               
#         elif r == 2: reg = 'ValledAosta'
#         elif r == 3: reg = 'Lombardia'
#         elif r == 4: reg = 'TrentinoAltoAdige'
#         elif r == 5: reg = 'Veneto'
#         elif r == 6: reg = 'FriuliVeneziaGiulia'
#         elif r == 7: reg = 'Liguria'
#         elif r == 8: reg = 'EmiliaRomagna'
#         elif r == 9: reg = 'Toscana'
#         elif r == 10: reg = 'Umbria'
#         elif r == 11: reg = 'Marche'
#         elif r == 12: reg = 'Lazio'
#         elif r == 13: reg = 'Abruzzo'
#         elif r == 14: reg = 'Molise'
#         elif r == 15: reg = 'Campania'
#         elif r == 16: reg = 'Puglia'
#         elif r == 17: reg = 'Basilicata'
#         elif r == 18: reg = 'Calabria'
#         elif r == 19: reg = 'Sicilia'
#         elif r == 20: reg = 'Sardegna'
#         region[k] = reg
#     return region


def Surf(istat):
        
    isurf=istat['q_2_7_class']
    Sfloor={}
    for i in isurf.index:
        if isurf.loc[i]==1: s = 15 # [mq]
        elif isurf.loc[i]==2: s = 30 # [mq]
        elif isurf.loc[i]==3: s = 50 # [mq]
        elif isurf.loc[i]==4: s = 75 # [mq]
        elif isurf.loc[i]==5: s = 105 # [mq]
        elif isurf.loc[i]==6: s = 135 # [mq]
        elif isurf.loc[i]==7: s = 165 # [mq]
        else: s = 100 # [mq]
        Sfloor[i]=s
    Sfloor = pd.Series(Sfloor, index = isurf.index)
    return Sfloor



''' Domestic Hot Water '''
def loadDHW2(istat,db_dhw):
    
    iDHW = istat#.iloc[:n_data,[0,89,138,139,140,141,142,143,144,145,146,147,199,
                #               165,200,201,202,204,205,206,207,208,368,91,88]]
    iDHW.rename(columns = {"id":"sample",
                           "q_1_1_sq1":"ComponentiFamiglia",                    # No. componenti nucleo familiare
                           "q_2_39":"dotazione_ACS",
                           "q_2_40A":"imp_Centralizzato",
                           "q_2_40B":"imp_Autonomo",
                           "q_2_40C":"imp_Singolo",
                           "q_2_41A":"Singolo_Elettrico",
                           "q_2_41B":"Singolo_GasNaturale",
                           "q_2_41C":"Singolo_Gasolio",
                           "q_2_41D":"Singolo_GPL",
                           "q_2_41E":"Singolo_Biomassa",
                           "q_2_42":"imp_Prevalente",                           # per chi ha più sistemi di produzione ACS 
                           "q_4_1":"imp_ACS/RiscAmb",                           # imp. ACS coincide con imp. riscaldamento? 1:SI. 2:NO.
                           "q_3_1":"CX_fuel_RiscAmb",                                       ## 1.fuel imp. riscald. (centr./aut.)
                           "q_4_2":"DX_fuel_ACS",                                           ## 2.fuel imp. ACS (centr./aut.))
                           "q_4_3":"dotazione_PDC",                             # dotazione pompa di calore pe ACS 1:SI. 2/3:NO.
                           "q_4_4":"tipologia_PDC",                             # tipo: 1:ARIA. 2:ACQFALDA. 3:ACQSUPERF. 4:GEO. 5:None
                           "q_4_6":"Ist/Accum",                                 # (sing)
                           "q_4_7":"CapacitàScaldabagno",                       # capacità scaldabagno (sing)
                           "q_4_8":"uso_SolareTermico",                         # 1:soloACS 2:ACS+amb (sing)
                           "q_4_9":"SuperficeCollettori",                       # superficie collettore solare termico
                           "q_4_10":"CapacitàAccumuloSolare",                   # accumulo serbatoio solare termico
                           "q_4_13_ric":"ClasseEpocaImpiantoACS",               # Età impianto ACS
                           "q_2_1":"TipologiaAbitazione",                      # 1:SFH. 2:MFH. 3:AB<10 4:AB10-27 5:AB>28
                           "reg":"Region"}, 
             inplace = True)
    
    'ISTAT User Data: iDHW'
    codice_ID = iDHW.index
    regione = iDHW["Region"]
    regione = decodificRegion(regione)
    Cw = weightCoeff(istat)
    floorSurf= Surf(istat)  # mq
    
    px = iDHW["ComponentiFamiglia"]
    arch = iDHW["TipologiaAbitazione"]
  
    imp_prev = iDHW['imp_Prevalente']
    
    PlantInfo = {}
    Plant = {}
    PlantAge = {}
    FuelDHW = {}

    risposte = pd.DataFrame(index = codice_ID, columns = ['dotazione_ACS',
                                                          'ImpiantoCentralizzato',
                                                          'ImpiantoAutonomo',
                                                          'ImpiantoSingoli',
                                                          'ImpiantoPrevalente',
                                                          'AccumuloIstantaneo',
                                                          'Anno',
                                                          "Singolo_Elettrico",
                                                          "Singolo_GasNaturale",
                                                          "Singolo_Gasolio",
                                                          "Singolo_GPL",
                                                          "Singolo_Biomassa",
                                                          "UgualeRisc",
                                                          "Impianto",
                                                          "Fuel",
                                                          "StringaInfo",
                                                          "StringaPlant"
                                                          "PDC"])
    
    for i in istat.index:
        if iDHW["dotazione_ACS"].loc[i] == 1:
            
            risposte.loc[i,'dotazione_ACS'] = 'SI'

            if iDHW['imp_Prevalente'].loc[i] == 0:
                plantinfo = 'DHW System: unique '
                risposte.loc[i,'ImpiantoPrevalente'] = 'Un solo impianto'
            elif iDHW['imp_Prevalente'].loc[i] == 1:
                 
                plantinfo = 'DHW System: centralised-shared (2+) '
                risposte.loc[i,'ImpiantoPrevalente'] = 'Centralizzato'
            elif iDHW['imp_Prevalente'].loc[i] == 2:
                plantinfo = 'DHW System: autonomous (2+) '
                risposte.loc[i,'ImpiantoPrevalente'] = 'Autonomo'
            elif iDHW['imp_Prevalente'].loc[i] == 3:
                plantinfo = '(2+) Single DHW System '
                risposte.loc[i,'ImpiantoPrevalente'] = 'AppSingoli'
            else: plantinfo = 'NoAnsw '
            
            age = iDHW['ClasseEpocaImpiantoACS'].loc[i]
            if age==1: y='1 year'                
            elif age==2: y='1-2 years'
            elif age==3: y='2-3 years'
            elif age==4: y='3-6 years'    
            elif age==5: y='6-10 years'    
            elif age==6: y='10-20 years'       
            elif age==7: y='20 years'  
            else: y='None'  
            PlantAge[i]=y
            
            risposte.loc[i,'Anno'] = y
            
            c = iDHW["imp_Centralizzato"].loc[i]
            a = iDHW["imp_Autonomo"].loc[i]
            s = iDHW["imp_Singolo"].loc[i] 
            
            risposte.loc[i,'ImpiantoCentralizzato'] = 'SI' if c == 1 else 'NO'
            risposte.loc[i,'ImpiantoAutonomo'] = 'SI' if a == 1 else 'NO'
            risposte.loc[i,'ImpiantoSingoli'] = 'SI' if s == 1 else 'NO'
            
            fsg_1 = iDHW["Singolo_Elettrico"].loc[i]
            fsg_2 = iDHW["Singolo_GasNaturale"].loc[i]
            fsg_3 = iDHW["Singolo_Gasolio"].loc[i]
            fsg_4 = iDHW["Singolo_GPL"].loc[i]                
            fsg_5 = iDHW["Singolo_Biomassa"].loc[i]  
            
            risposte.loc[i,'Singolo_Elettrico'] = 'SI' if fsg_1 == 1 else 'NO'
            risposte.loc[i,'Singolo_GasNaturale'] = 'SI' if fsg_2 == 1 else 'NO'
            risposte.loc[i,'Singolo_Gasolio'] = 'SI' if fsg_3 == 1 else 'NO'
            risposte.loc[i,'Singolo_GPL'] = 'SI' if fsg_4 == 1 else 'NO'
            risposte.loc[i,'Singolo_Biomassa'] = 'SI' if fsg_5 == 1 else 'NO'
                                      
            
            cx = iDHW["imp_ACS/RiscAmb"].loc[i] 
            
            risposte.loc[i,'UgualeRisc'] = 'SI' if cx == 1 else 'NO'
            
            if (s == 1 and a != 1 and c != 1) or iDHW['imp_Prevalente'].loc[i] == 3:
                # Caso apparecchi singoli prevalenti
                         
                plantinfo += '- Single Plant'  
                plant = 'Single'
                
                if iDHW["Ist/Accum"].loc[i]==1:  plantinfo += ' with Instantaneous Gen.'; risposte.loc[i,'AccumuloIstantaneo'] = 'Istantaneo'
                elif iDHW["Ist/Accum"].loc[i]==2: plantinfo += ' with Buffer Tank'; risposte.loc[i,'AccumuloIstantaneo'] = 'Accumulo'
                
                acs_prevalente = iDHW['q_4_5'].loc[i]
                
                if acs_prevalente == 1: fuel = 'Electric'                
                elif acs_prevalente == 2: fuel = 'NaturalGas'  
                elif acs_prevalente == 3: fuel = 'Gasoline'
                elif acs_prevalente == 4: fuel = 'LPG'
                elif acs_prevalente == 5: fuel = 'Biomass'
                else:
                    if fsg_1 == 1: fuel = 'Electric'                
                    elif fsg_2 == 1: fuel = 'NaturalGas'  
                    elif fsg_3 == 1: fuel = 'Gasoline'
                    elif fsg_4 == 1: fuel = 'LPG'
                    elif fsg_5 == 1: fuel = 'Biomass'
                    
                FuelDHW[i] = fuel
                risposte.loc[i,'Impianto'] = 'Singoli'
            
            elif iDHW['imp_Prevalente'].loc[i] != 3:
                
                if c == 1: 
                    plantinfo += '- Centralised Plant'
                    plant = 'Centralised'
                    risposte.loc[i,'Impianto'] = 'Centralizzato'
                elif a == 1:
                    plantinfo += '- Autonomous Plant'
                    plant = 'Autonomous'                    
                    risposte.loc[i,'Impianto'] = 'Autonomo'
                    
                # Coincidenza tra impianto ACS e di Riscaldamento
                if cx == 1 :
                    
                    plantinfo += ' (for SpaceHeating and DHW)'
                    
                    fcx = iDHW["CX_fuel_RiscAmb"].loc[i]
                    if fcx == 1: fuel = 'NaturalGas'
                    elif fcx == 2: fuel = 'Gasoline'
                    elif fcx == 3: fuel = 'LPG'
                    elif fcx == 4: fuel = 'Electric'
                    elif fcx == 5: fuel = 'Oil'
                    elif fcx == 6: fuel = 'Biomass'
                    elif fcx == 7: fuel = 'Coke'
                    elif fcx == 9: fuel = 'NoAnsw'
                    else: fuel = 'NotAssigned'
                    FuelDHW[i] = fuel
                    
                # Differente impianto per ACS e per il Riscaldamento    
                elif cx == 0 or cx == 2:
                    
                    plantinfo += ' (DHW only)'
                    
                    fdx = iDHW["DX_fuel_ACS"].loc[i]
                    if fdx == 1: fuel = 'NaturalGas'
                    elif fdx == 2: fuel = 'Gasoline'
                    elif fdx == 3: fuel = 'LPG'
                    elif fdx == 4: fuel = 'Electric'
                    elif fdx == 5: fuel = 'Oil'
                    elif fdx == 6: fuel = 'Biomass'
                    elif fdx == 7: fuel = 'Coke'
                    elif fdx == 8: fuel = 'Solar'
                    elif fdx == 9: fuel = 'NoAnsw'  
                    else: fuel = 'NotAssigned'
                    FuelDHW[i] = fuel
                    
                else:  FuelDHW[i] = 'Warning: error'  

            
            
            else:  FuelDHW[i] = 'Warning: error'
            
            risposte.loc[i,'Fuel'] = FuelDHW[i]
            
            PlantInfo[i] = plantinfo
            Plant[i] = plant
            
            risposte.loc[i,'StringaInfo'] = plantinfo
            risposte.loc[i,'StringaPlant'] = plant
            
            if iDHW.loc[i,"dotazione_PDC"] == 1:
                risposte.loc[i,'PDC'] = {
                    1 : 'PDC aria',
                    2 : 'PDC acqua falda',
                    3 : 'PDC acqua superficiale',
                    4 : 'PDC terreno',
                    9 : 'PDC aria'
                    }[iDHW.loc[i]["tipologia_PDC"]]
            else:
                risposte.loc[i,'PDC'] = None
            
        # in case no DHW plant is used
        else: 
            FuelDHW[i] = 'no_DHW'
            risposte.loc[i,'dotazione_ACS'] = 'NO'
            
            risposte.loc[i,'StringaInfo'] = 'None'
            risposte.loc[i,'StringaPlant'] = 'None'
    
    # Da vaillant esempi di alcuni serbatoi: circa 0.8 kWh/24h
  
    
    # Va inserita pdc
    pars = px,arch,floorSurf
    Qnd = DHWdemand(codice_ID,regione,pars)
    
    
    return Qnd, risposte

''' Domestic Hot Water '''
def loadDHW(istat,db_dhw):
    
    iDHW = istat#.iloc[:n_data,[0,89,138,139,140,141,142,143,144,145,146,147,199,
                #               165,200,201,202,204,205,206,207,208,368,91,88]]
    iDHW.rename(columns = {"id":"sample",
                           "q_1_1_sq1":"ComponentiFamiglia",                    # No. componenti nucleo familiare
                           "q_2_39":"dotazione_ACS",
                           "q_2_40A":"imp_Centralizzato",
                           "q_2_40B":"imp_Autonomo",
                           "q_2_40C":"imp_Singolo",
                           "q_2_41A":"Singolo_Elettrico",
                           "q_2_41B":"Singolo_GasNaturale",
                           "q_2_41C":"Singolo_Gasolio",
                           "q_2_41D":"Singolo_GPL",
                           "q_2_41E":"Singolo_Biomassa",
                           "q_2_42":"imp_Prevalente",                           # per chi ha più sistemi di produzione ACS 
                           "q_4_1":"imp_ACS/RiscAmb",                           # imp. ACS coincide con imp. riscaldamento? 1:SI. 2:NO.
                           "q_3_1":"CX_fuel_RiscAmb",                                       ## 1.fuel imp. riscald. (centr./aut.)
                           "q_4_2":"DX_fuel_ACS",                                           ## 2.fuel imp. ACS (centr./aut.))
                           "q_4_3":"dotazione_PDC",                             # dotazione pompa di calore pe ACS 1:SI. 2/3:NO.
                           "q_4_4":"tipologia_PDC",                             # tipo: 1:ARIA. 2:ACQFALDA. 3:ACQSUPERF. 4:GEO. 5:None
                           "q_4_6":"Ist/Accum",                                 # (sing)
                           "q_4_7":"CapacitàScaldabagno",                       # capacità scaldabagno (sing)
                           "q_4_8":"uso_SolareTermico",                         # 1:soloACS 2:ACS+amb (sing)
                           "q_4_9":"SuperficeCollettori",                       # superficie collettore solare termico
                           "q_4_10":"CapacitàAccumuloSolare",                   # accumulo serbatoio solare termico
                           "q_4_13_ric":"ClasseEpocaImpiantoACS",               # Età impianto ACS
                           "q_2_1":"TipologiaAbitazione",                      # 1:SFH. 2:MFH. 3:AB<10 4:AB10-27 5:AB>28
                           "reg":"Region"}, 
             inplace = True)
    
    'ISTAT User Data: iDHW'
    codice_ID = iDHW.index
    regione = iDHW["Region"]
    regione = decodificRegion(regione)
    Cw = weightCoeff(istat)
    floorSurf= Surf(istat)  # mq
    
    px = iDHW["ComponentiFamiglia"]
    arch = iDHW["TipologiaAbitazione"]
  
    imp_prev = iDHW['imp_Prevalente']
    
    PlantInfo = {}
    Plant = {}
    PlantAge = {}
    FuelDHW = {}

    
    for i in istat.index:
          
        if iDHW["dotazione_ACS"].loc[i] == 1:

            if iDHW['imp_Prevalente'].loc[i] == 0:
                plantinfo = 'DHW System: unique '
            elif iDHW['imp_Prevalente'].loc[i] == 1:
                 plantinfo = 'DHW System: centralised-shared (2+) '
            elif iDHW['imp_Prevalente'].loc[i] == 2:
                plantinfo = 'DHW System: autonomous (2+) '
            elif iDHW['imp_Prevalente'].loc[i] == 3:
                plantinfo = '(2+) Single DHW System '
            else: plantinfo = 'NoAnsw '
            
            age = iDHW['ClasseEpocaImpiantoACS'].loc[i]
            if age==1: y='1 year'                
            elif age==2: y='1-2 years'
            elif age==3: y='2-3 years'
            elif age==4: y='3-6 years'    
            elif age==5: y='6-10 years'    
            elif age==6: y='10-20 years'       
            elif age==7: y='20 years'  
            else: y='None'  
            PlantAge[i]=y
            
            c = iDHW["imp_Centralizzato"].loc[i]
            a = iDHW["imp_Autonomo"].loc[i]
            s = iDHW["imp_Singolo"].loc[i] 
            
            fsg_1 = iDHW["Singolo_Elettrico"].loc[i]
            fsg_2 = iDHW["Singolo_GasNaturale"].loc[i]
            fsg_3 = iDHW["Singolo_Gasolio"].loc[i]
            fsg_4 = iDHW["Singolo_GPL"].loc[i]                
            fsg_5 = iDHW["Singolo_Biomassa"].loc[i]   
            
            cx = iDHW["imp_ACS/RiscAmb"].loc[i] 
            
            
            if (c == 1 or a == 1) and s != 1:
                
                if c == 1: 
                    plantinfo += '- Centralised Plant'
                    plant = 'Centralised'
                elif a == 1:
                    plantinfo += '- Autonomous Plant'
                    plant = 'Autonomous'
                    
                # Coincidenza tra impianto ACS e di Riscaldamento
                if cx == 1 :
                    
                    plantinfo += ' (for SpaceHeating and DHW)'
                    
                    fcx = iDHW["CX_fuel_RiscAmb"].loc[i]
                    if fcx == 1: fuel = 'NaturalGas'
                    elif fcx == 2: fuel = 'Gasoline'
                    elif fcx == 3: fuel = 'LPG'
                    elif fcx == 4: fuel = 'Electric'
                    elif fcx == 5: fuel = 'Oil'
                    elif fcx == 6: fuel = 'Biomass'
                    elif fcx == 7: fuel = 'Coke'
                    elif fcx == 9: fuel = 'NoAnsw'
                    else: fuel = 'NotAssigned'
                    FuelDHW[i] = fuel
                    
                # Differente impianto per ACS e per il Riscaldamento    
                elif cx == 0 or cx == 2:
                    
                    plantinfo += ' (DHW only)'
                    
                    fdx = iDHW["DX_fuel_ACS"].loc[i]
                    if fdx == 1: fuel = 'NaturalGas'
                    elif fdx == 2: fuel = 'Gasoline'
                    elif fdx == 3: fuel = 'LPG'
                    elif fdx == 4: fuel = 'Electric'
                    elif fdx == 5: fuel = 'Oil'
                    elif fdx == 6: fuel = 'Biomass'
                    elif fdx == 7: fuel = 'Coke'
                    elif fdx == 8: fuel = 'Solar'
                    elif fdx == 9: fuel = 'NoAnsw'  
                    else: fuel = 'NotAssigned'
                    FuelDHW[i] = fuel
                    
                else:  FuelDHW[i] = 'Warning: error'  

            elif s == 1:
                         
                plantinfo += '- Single Plant'  
                plant = 'Single'
                
                if iDHW["Ist/Accum"].loc[i]==1:  plantinfo += ' with Instantaneous Gen.'
                elif iDHW["Ist/Accum"].loc[i]==2: plantinfo += ' with Buffer Tank'
                
                if fsg_1 == 1: fuel = 'Electric'                
                elif fsg_2 == 1: fuel = 'NaturalGas'  
                elif fsg_3 == 1: fuel = 'Gasoline'
                elif fsg_4 == 1: fuel = 'LPG'
                elif fsg_5 == 1: fuel = 'Biomass'
                FuelDHW[i] = fuel
            
            else:  FuelDHW[i] = 'Warning: error'
            
            PlantInfo[i] = plantinfo
            Plant[i] = plant
            
        # in case no DHW plant is used
        else: FuelDHW[i] = 'no_DHW'
    
        
    
    # Efficiencies DHW plants from database DB (UNI/TS 11300-2)
    DB_NetwEff=[]
    for x in range(48,72):
        eff_ds = db_dhw.iloc[x,2]  # [-]
        eff_em = db_dhw.iloc[x,3]  # [-]
        eff_rg = db_dhw.iloc[x,4]  # [-]
        eff_netw = (eff_ds * eff_em * eff_rg)   
        DB_NetwEff.append(eff_netw)
    
    DB_plants=[]
    for x in range(48,72):
        plant = db_dhw.iloc[x,6]  
        DB_plants.append(plant)    
                
    DB_fuels=[]
    for x in range(0,40):
        fuel_type = db_dhw.iloc[x,0]  
        DB_fuels.append(fuel_type)
    
    DB_ageplant=[]
    for x in range(0,40):
        ages = db_dhw.iloc[x,1]  
        DB_ageplant.append(ages)
    
    DB_eff_gn=[]
    for x in range(0,40):
        gn = db_dhw.iloc[x,5]  
        DB_eff_gn.append(gn)
    
    df1 = pd.DataFrame([FuelDHW, PlantAge]).transpose()
    df2 = pd.DataFrame([DB_fuels,DB_ageplant,DB_eff_gn]).transpose()
    
    # queste linee sono solo per farlo funzionare!!!!!!!!!!!
    
    df1.replace('Solar','Electric', inplace = True)
    df1.replace('no_DHW','Electric', inplace = True)
    df1.replace('NoAnswNone','Electric', inplace = True)
    df1.replace('NoAnsw','Electric', inplace = True)
    df1.replace('Oil','Electric', inplace = True)
    df1.replace(np.nan,'None', inplace = True)
    
    ########################################################
    
    df2.index = df2[0] + df2[1]
    df = (df1[0]+df1[1]).apply((lambda x: df2.loc[x]))
    eff_gn = df[2]
    
    # VA FATTO---- Commentato solo per far funzionare
    # df1 = pd.DataFrame([Plant, PlantAge]).transpose()
    # df2 = pd.DataFrame([DB_plants,DB_ageplant,DB_NetwEff]).transpose()
    # df = pd.merge(df1,df2, how='left')
    # eff_nw = df[2]
    eff_nw = 0.9
    
    eff_gl = (eff_gn * eff_nw)
    
    pars = px,arch,floorSurf#,FuelDHW,PlantInfo,Plant,PlantAge,iDHW,DB_fuels,DB_eff_gn,DB_ageplant,DB_NetwEff,eff_gl

    Qnd = DHWdemand(codice_ID,regione,pars)
    
    return Qnd





class DHWdemand:
    
    '''
    
    '''
    def __init__(self,ID,regione,pars):
        
        self.ID = ID
        self.region = regione
        
        
        # DHW needs evalution with UNI 11300-2:2014 and UNI 9182:2014
        self.Qw_UNI11300 = pd.Series(index = self.ID, dtype='float64')   # [kWh/year]
        self.Qw_UNI9182 = pd.Series(index = self.ID, dtype='float64')    # [kWh/year]
        
        self.qw_UNI11300 = pd.Series(index = self.ID, dtype='float64')   # [kWh/(mq year)]
        self.qw_UNI9182 = pd.Series(index = self.ID, dtype='float64')    # [kWh/(mq year)]
        
        self.px = pars[0]           # n° people per flat (from ISTAT survey)
        self.archetype = {}         # building archetype (from ISTAT survey)
        self.floor = pars[2]        # surface area per flat (from ISTAT survey)
                
        # self.fuel = pars[3]         # fuel typology
        
        
        ''' DHW needs calc '''
        self.Kn = {}         # coeff. based on N° of flats per building [UNI 9182:2010]
        self.Cw = 1.162     # [Wh/(kg K)]
        self.Vw = {}        # UNI 11300: daily need [lt/day] IT_aver = 157 lt/day
        self.Vpx = {}       # UNI 9182: daily need [lt/day/px]
        self.DTw = 25       # [°C]
        self.days = 330     # [days/year] 
        self.rho = 994.1    # [kg/m^3]
        
        # pre-processing data for DHW needs based on UNI 9182:2014
        for i in self.ID:
            px = self.px.loc[i]             
            Af = self.floor.loc[i]  
            
            arch = pars[1].loc[i]
            if arch== 1: 
                ArchType= "SFH" 
                self.Kn = 1.15
                self.Vpx = 75                  # medium [lt/(day px)] 
            elif arch== 2: 
                ArchType= "MFH" 
                self.Kn = 0.86
                self.Vpx = 75                  # medium [lt/(day px)] 
            elif arch== 3: 
                ArchType= "AB_LowDensity"      # < 10 dwellings
                self.Kn = 0.58                 
                self.Vpx = 70                  # medium [lt/(day px)] 
            elif arch== 4: 
                ArchType= "AB_MediumDensity"   # 11-27 dwellings
                self.Kn = 0.42
                self.Vpx = 50                  # popular [lt/(day px)] 
            elif arch== 5: 
                ArchType= "AB_HighDensity"     # > 28 dwellings
                self.Kn = 0.30
                self.Vpx = 50                  # popular [lt/(day px)] 
            self.archetype[i] = ArchType
            
            
            # --- UNI 11300-2:2014 --- 
            ##
            ###
            # Vw [lt/day] = a [lt/(m2 day)] * Af [m2] + b [lt/day]       
            if Af<=35:               a=0.0; b=50.0
            elif 35<Af<=50:          a=2.667; b=-43.33
            elif 50<Af<=200:         a=1.067; b=36.67               
            elif Af>200:             a=0.0; b=250.0
            
            # Water Need [m3/day]
            self.Vw[i] = (a*Af + b)/1000 
            #self.Vw = pd.Series(self.Vw, index = self.ID)
            
            Q_UNI11300 = 1000*(self.Cw/1000)*self.Vw[i]*self.DTw*self.days      # [kWh/year]
            Q_UNI11300 = round(Q_UNI11300,1)
            q_UNI11300 = Q_UNI11300/Af                                          # [kWh/(mq year)]
            
            #  --- UNI 9182:2014 ---            
            ##
            ###
            
            # num. vani nei microdati MPR: da inserire (K)
            
            # Qw,nd [kWh/year] = N° people * Kn * density * Cw * (T_draw - Tsource) 
            Q_UNI9182 = (px*self.Kn*self.rho*self.Cw*(self.Vpx/1000)*self.DTw*self.days)/1000   # [kWh/year]
            Q_UNI9182 = round(Q_UNI9182,1)
            q_UNI9182 = Q_UNI9182/Af
            
            #
            # Overall Need
            self.Qw_UNI11300.loc[i] = Q_UNI11300
            self.Qw_UNI9182.loc[i] = Q_UNI9182
            # DHW per square meters
            self.qw_UNI11300.loc[i] = q_UNI11300
            self.qw_UNI9182.loc[i] = q_UNI9182
        
        
if __name__ == '__main__':
    
    ##### Import database ISTAT 2013
    try:
        istat
    except:
        ''' DATA IMPORT '''
        istat = pd.read_csv(os.path.join('..','Input','istat_microdata_csv.csv'), delimiter=';')
        ## id = (istat['id']-1).tolist()
        
    n_data = 20000
    GG = 330        # (giorni/anno presenza utenti)
    SE = 47.15         # (settimane/anno presenza utenti)
    
    ##### Import database household appliances 
    db_elettrodomestici_path = os.path.join('..','Input','Database_elettrodomestici.xlsx')
    
    db_dhw = pd.read_excel(db_elettrodomestici_path,
                                  sheet_name="ACS", header=0,index_col=0)
    DHW_demand = loadDHW(copy.deepcopy(istat),db_dhw)
    
    risposte = loadDHW2(copy.deepcopy(istat),db_dhw)
    
    #risposte.to_excel('..\\Output_\\Acs_info.xlsx')
    
    # consumptions = mainEE(db_elettrodomestici_path, istat, n_data = n_data)
    
    # EEC_Outputs, EEC_Outputs_df = storeOutputs(consumptions)
    # EEC_Outputs_df.to_csv(os.path.join('.','Output_','consumption.csv'))
    
    # Sum_EEC = round(weightedEEC(consumptions,istat, n_data = n_data),5) 
        