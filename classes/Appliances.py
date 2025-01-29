 # -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:45:36 2021

@author: fabix
"""

import pandas as pd
from pandas import DataFrame
import os
import numpy as np
from funcs.aux_functions import weightCoeff, decodificRegion, fascia_climatica
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

names = ['tipo','dati']
# def weightCoeff(istat):
#     weight = istat.iloc[:,[384]]
#     weight.rename(columns = {"coef_red":"Weights"}, 
#              inplace = True)  
#     return weight

# '''Assign Region (str) name to each ISTAT building'''
# def decodificRegion(Regione):
                                 
#     region={}
#     for k in Regione.index:
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

''' ILLUMINAZIONE '''
def loadLights(istat,db_lights):
    
    lights_idata = istat#.iloc[:,[0,282,283,284,286,287,288,88]]
    lights_idata.rename(columns = {"id":"sample",
                        "q_7_2A":"ill_risp_A",
                        "q_7_2B":"ill_risp_B",
                        "q_7_2C":"ill_risp_C",
                        "q_7_4A":"ill_trad_A",
                        "q_7_4B":"ill_trad_B",
                        "q_7_4C":"ill_trad_C",
                        "reg":"Region"}, 
             inplace = True)
    '_idata'
    codice_ID = lights_idata['sample']
    regione = lights_idata["Region"]
    regione = decodificRegion(regione)
    
    ES_lights_A = lights_idata["ill_risp_A"]
    ES_lights_B = lights_idata["ill_risp_B"]
    ES_lights_C = lights_idata["ill_risp_C"]
    TR_lights_A = lights_idata["ill_trad_A"]
    TR_lights_B = lights_idata["ill_trad_B"]
    TR_lights_C = lights_idata["ill_trad_C"] 
    
    P_ES_lights = db_lights.iloc[0,0]  # kW
    P_TR_lights = db_lights.iloc[1,0]  # kW
    
    param_lights = codice_ID, ES_lights_A, ES_lights_B, ES_lights_C, TR_lights_A, TR_lights_B, TR_lights_C, P_ES_lights, P_TR_lights

    EE_demand_lights = Lights(codice_ID,regione,param_lights)
    
    
    luci_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
    luci_info['luci','N_lampadine_alta_effi_3h'] = ES_lights_A
    luci_info['luci','N_lampadine_alta_effi_7h'] = ES_lights_B
    luci_info['luci','N_lampadine_alta_effi_14h'] = ES_lights_C
    luci_info['luci','N_lampadine_bassa_effi_3h'] = TR_lights_A
    luci_info['luci','N_lampadine_bassa_effi_7h'] = TR_lights_B
    luci_info['luci','N_lampadine_bass_effi_14h'] = TR_lights_C
    return EE_demand_lights, luci_info


class Lights:
    
    '''
    
    Lampadine Alta Efficienza (ES)
    Lampadine Tradizionali (TR)
    
    uso inferiore a 4 ore/giorno (A)
    uso compreso tra le 4-12 ore/giorno (B)
    uso superiore alle 12 ore/giorno (C)
    
    '''
    def __init__(self,ID,regione,lights_par):
        
        self.ID = ID
        self.region = regione
        
        self.lights_demand = {}
        
        gganno = 330  #giorni/anno
        
        '''fonte: ENEA'''
        A = 3   # ore/giorno
        B = 7   # ore/giorno
        C = 14  # ore/giorno 
        
        self.Num_Lamp_AltaEff_Ut_Basso      = lights_par[1]
        self.Num_Lamp_AltaEff_Ut_Medio      = lights_par[2]
        self.Num_Lamp_AltaEff_Ut_Intenso    = lights_par[3]
        self.Num_Lamp_Trad_Ut_Basso         = lights_par[4]
        self.Num_Lamp_Trad_Ut_Medio         = lights_par[5]
        self.Num_Lamp_Trad_Ut_Intenso       = lights_par[6]
        
        self.Potenza_Lamp_AltaEff           = lights_par[7]
        self.Potenza_Lamp_Trad              = lights_par[8]
        
        self.lights_demand = gganno*(self.Potenza_Lamp_AltaEff*(A*self.Num_Lamp_AltaEff_Ut_Basso
                                                                +B*self.Num_Lamp_AltaEff_Ut_Medio+
                                                                C*self.Num_Lamp_AltaEff_Ut_Intenso)
                                     + self.Potenza_Lamp_Trad*(A*self.Num_Lamp_Trad_Ut_Basso
                                                               +B*self.Num_Lamp_Trad_Ut_Medio
                                                              +C*self.Num_Lamp_Trad_Ut_Intenso))

        '''mean consumption per household'''
        n_l=len(self.lights_demand[self.lights_demand>0])
        overall_LightsDemand=self.lights_demand.sum()
        self.Mean_LightsDemand=round(overall_LightsDemand/n_l,1) 

    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
    
    def extractBdData(self,BdID):
        bdData = {'AnnualConsumption_EE: Lights' : self.lights_demand[BdID]}
        return bdData

''' PICCOLI_ELETTRODOMESTICI ''' 
def loadLittleAppliances(istat,db_little_appliances):
    
    
    little_appliances_idata = istat#.iloc[:,[0,89,320,321,322,323,331,332,333,334,335,336,337,338,339,340,341,88]]
    little_appliances_idata.rename(columns = {"id":"sample",
                        "q_1_1_sq1":"ComponentiFamiglia",                      
                        "q_8_28A":"Aspirapolvere",
                        "q_8_28B":"Vaporella",
                        "q_8_28C":"RobotPavimento",
                        "q_8_28D":"FerroDaStiro",                                              
                        "q_8_38A":"RobotCucina",
                        "q_8_38B":"Frullatore",
                        "q_8_38C":"MacinaCaffe",
                        "q_8_38D":"Affettatrice",
                        "q_8_38E":"MacchinaPane",
                        "q_8_38F":"ScaldatoreBevande",
                        "q_8_38G":"Tostapane",
                        "q_8_38H":"FornoMicroonde",
                        "q_8_38I":"FornoBiomassa",  # no elett
                        "q_8_38L":"Barbecue",   # no elett
                        "q_8_38M":"GrigliaElettrica",
                        "reg":"Region"}, 
             inplace = True)
    '_idata' 'LAp1 = little_appliance_no1'
    codice_ID = little_appliances_idata.index
    regione = little_appliances_idata["Region"]
    regione = decodificRegion(regione)
    inhab = little_appliances_idata["ComponentiFamiglia"]
    
    Aspirapolvere = little_appliances_idata["Aspirapolvere"]
    Vaporella = little_appliances_idata["Vaporella"]    
    RobotPavimento = little_appliances_idata["RobotPavimento"]
    FerroDaStiro = little_appliances_idata["FerroDaStiro"]
    RobotCucina = little_appliances_idata["RobotCucina"]
    Frullatore = little_appliances_idata["Frullatore"]
    MacinaCaffe = little_appliances_idata["MacinaCaffe"]
    Affettatrice = little_appliances_idata["Affettatrice"]
    MacchinaPane = little_appliances_idata["MacchinaPane"]
    ScaldatoreBevande = little_appliances_idata["ScaldatoreBevande"]
    Tostapane = little_appliances_idata["Tostapane"]    
    FornoMicroonde = little_appliances_idata["FornoMicroonde"]
    GrigliaElettrica = little_appliances_idata["GrigliaElettrica"]
    
    'db_'
    EE_Aspirapolvere = db_little_appliances.iloc[0,0]  # kWh/y
    EE_Vaporella = db_little_appliances.iloc[1,0]  # kWh/y
    EE_RobotPavimento = db_little_appliances.iloc[2,0]  # kWh/y
    EE_FerroDaStiro = db_little_appliances.iloc[3,0]  # kWh/y    
    EE_RobotCucina = db_little_appliances.iloc[4,0]  # kWh/y
    EE_Frullatore = db_little_appliances.iloc[5,0]  # kWh/y
    EE_MacinaCaffe = db_little_appliances.iloc[6,0]  # kWh/y   
    EE_Affettatrice = db_little_appliances.iloc[7,0]  # kWh/y
    EE_MacchinaPane = db_little_appliances.iloc[8,0]  # kWh/y   
    EE_ScaldatoreBevande = db_little_appliances.iloc[9,0]  # kWh/y
    EE_Tostapane = db_little_appliances.iloc[10,0]  # kWh/y
    EE_FornoMicroonde = db_little_appliances.iloc[11,0]  # kWh/y
    EE_GrigliaElettrica = db_little_appliances.iloc[12,0]  # kWh/y 

    fixedAppliances = db_little_appliances.iloc[13,0]  # kWh/year/px   
    
    param_istat = Aspirapolvere,Vaporella,RobotPavimento,FerroDaStiro,RobotCucina,Frullatore,MacinaCaffe,Affettatrice,MacchinaPane,ScaldatoreBevande,Tostapane,FornoMicroonde,GrigliaElettrica,inhab
    param_db = EE_Aspirapolvere,EE_Vaporella,EE_RobotPavimento,EE_FerroDaStiro,EE_RobotCucina,EE_Frullatore,EE_MacinaCaffe,EE_Affettatrice,EE_MacchinaPane,EE_ScaldatoreBevande,EE_Tostapane,EE_FornoMicroonde,EE_GrigliaElettrica,fixedAppliances                            
    
    EE_demand_little_appliances = LittleAppliances(codice_ID,regione,param_istat,param_db)
    
    little_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
    little_info['piccoli_elettro','aspirapolvere'] = Aspirapolvere
    little_info['piccoli_elettro','lavapavimenti'] = Vaporella
    little_info['piccoli_elettro','robot_aspirapolvere'] = RobotPavimento
    little_info['piccoli_elettro','ferro_stiro'] = FerroDaStiro
    little_info['piccoli_elettro','RobotCucina'] = RobotCucina
    little_info['piccoli_elettro','Frullatore'] = Frullatore
    little_info['piccoli_elettro','MacinaCaffe'] = MacinaCaffe
    little_info['piccoli_elettro','Affettatrice'] = Affettatrice
    little_info['piccoli_elettro','MacchinaPane'] = MacchinaPane
    little_info['piccoli_elettro','ScaldatoreBevande'] = ScaldatoreBevande
    little_info['piccoli_elettro','Tostapane'] = Tostapane
    little_info['piccoli_elettro','FornoMicroonde'] = FornoMicroonde
    little_info['piccoli_elettro','GrigliaElettrica'] = GrigliaElettrica
    
    little_info.replace({1: 'Si', 2: ''}, inplace = True)
    
    return EE_demand_little_appliances,little_info

class LittleAppliances:
  
    def __init__(self,ID,regione,i_par,db_par):
        
        self.ID = ID
        self.region = regione
        
        self.little_appliances_demand = {}
        
        sett = 47  #settimane/anno
        
        self.Aspirapolvere = i_par[0].replace(2,0)
        self.Vaporella = i_par[1].replace(2,0)
        self.RobotPavimento = i_par[2].replace(2,0)
        self.FerroDaStiro = i_par[3].replace(2,0)                
        self.RobotCucina = i_par[4].replace(2,0)
        self.Frullatore = i_par[5].replace(2,0)
        self.MacinaCaffe = i_par[6].replace(2,0)
        self.Affettatrice = i_par[7].replace(2,0)
        self.MacchinaPane = i_par[8].replace(2,0)
        self.ScaldatoreBevande = i_par[9].replace(2,0)
        self.Tostapane = i_par[10].replace(2,0)
        self.FornoMicroonde = i_par[11].replace(2,0)
        self.GrigliaElettrica = i_par[12].replace(2,0)

        self.EE_Aspirapolvere = db_par[0]
        self.EE_Vaporella = db_par[1]
        self.EE_RobotPavimento = db_par[2]
        self.EE_FerroDaStiro = db_par[3]                
        self.EE_RobotCucina = db_par[4]
        self.EE_Frullatore = db_par[5]
        self.EE_MacinaCaffe = db_par[6]
        self.EE_Affettatrice = db_par[7]
        self.EE_MacchinaPane = db_par[8]
        self.EE_ScaldatoreBevande = db_par[9]
        self.EE_Tostapane = db_par[10]
        self.EE_FornoMicroonde = db_par[11]
        self.EE_GrigliaElettrica = db_par[12]

        self.n_inhabitants = i_par[13]
        self.OtherAppliancesDemand = db_par[13]
        
        self.little_appliances_demand = (self.Aspirapolvere*self.EE_Aspirapolvere
                                               + self.Vaporella*self.EE_Vaporella
                                               + self.RobotPavimento*self.EE_RobotPavimento
                                               + self.FerroDaStiro*self.EE_FerroDaStiro
                                               + self.RobotCucina*self.EE_RobotCucina 
                                               + self.Frullatore*self.EE_Frullatore 
                                               + self.MacinaCaffe*self.EE_MacinaCaffe
                                               + self.Affettatrice*self.EE_Affettatrice
                                               + self.MacchinaPane*self.EE_MacchinaPane
                                               + self.ScaldatoreBevande*self.EE_ScaldatoreBevande 
                                               + self.Tostapane*self.EE_Tostapane 
                                               + self.FornoMicroonde*self.EE_FornoMicroonde
                                               + self.GrigliaElettrica*self.EE_GrigliaElettrica)
                                               # + self.n_inhabitants*self.OtherAppliancesDemand)

        self.little_appliances_demand=round(self.little_appliances_demand,1)
        
        '''mean consumption per household'''
        n_la=len(self.little_appliances_demand[self.little_appliances_demand>0])
        overall_LittleAppliancesDemand=self.little_appliances_demand.sum()
        self.Mean_LittleAppliancesDemand=round(overall_LittleAppliancesDemand/n_la,1) 
        
    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
       
    def extractBdData(self,BdID):
        bdData = {'AnnualConsumption_EE: LittleAppliances' : self.little_appliances_demand[BdID]}
        return bdData

        
''' FRIGORIFERIE E CONGELATORI (database)'''
def loadRefrigerators_moirae(istat,db_refrigerators):

    refrigerators_idata = istat#.iloc[:,[0,298,299,300,301,374,375,88]]
    refrigerators_idata.rename(columns = {"id":"sample",
                        "q_8_2":"Num. Frigoriferi",
                        "q_8_5":"Capacità Frigorifero",
                        "q_8_6":"Dotazione Vano Congelatore",
                        "q_8_7":"Dotazione Freezer",
                        "q_8_3_ric":"Età Frigorifero",
                        "q_8_8_ric":"Età Freezer",
                        "reg":"Region"}, 
             inplace = True)
    
    '_idata  ----------------------------------------------------------------' 
    codice_ID = refrigerators_idata.index
    regione = refrigerators_idata["Region"]
    regione = decodificRegion(regione)
    
    n_Refrigerators = refrigerators_idata["Num. Frigoriferi"]
    n_IntFreezer = refrigerators_idata["Dotazione Vano Congelatore"]
    n_Freezers = refrigerators_idata["Dotazione Freezer"]

    Cap_Refrigerators = refrigerators_idata["Capacità Frigorifero"]
    Refr_Capacity={}
    for i in refrigerators_idata.index:
        C = Cap_Refrigerators.loc[i]
        if C==1: C='piccolo'                   
        elif C==2: C='medio'                             
        elif C==3: C='grande'                                 
        elif C==4: C='molto_grande'                              
        else: C='non_specificato'
        Refr_Capacity[i]=C
        
    Age_Refrigerators = refrigerators_idata["Età Frigorifero"]
    Age_Freezers = refrigerators_idata["Età Freezer"]
    Refr_Age={}
    Freez_Age={}
    for i in refrigerators_idata.index:
        y = Age_Refrigerators.loc[i]
        if y==1: y='1 year'                
        elif y==2: y='1-2 years'
        elif y==3: y='2-3 years'
        elif y==4: y='3-6 years'    
        elif y==5: y='6-10 years'    
        elif y==6: y='10-20 years'       
        elif y==7: y='20 years'  
        else: y='None'  
        Refr_Age[i]=y
        
        yf = Age_Freezers.loc[i]
        if yf==1: yf='1 year'                
        elif yf==2: yf='1-2 years'
        elif yf==3: yf='2-3 years'
        elif yf==4: yf='3-6 years'    
        elif yf==5: yf='6-10 years'    
        elif yf==6: yf='10-20 years'       
        elif yf==7: yf='20 years'  
        else: yf='None'  
        Freez_Age[i]=yf
    
    'db_  -------------------------------------------------------------------'
    db_sizes=[]
    db_ee_classes=[]
    db_demand=[] 
    db_ee_classes_f=[]
    db_demand_f=[]
    '''refrigerators'''
    for x in range(0,35):
        tb = db_refrigerators.iloc[x,0]  # kWh/year
        db_sizes.append(tb)
    for x in range(0,35):
        tb = db_refrigerators.iloc[x,1]  # kWh/year
        db_ee_classes.append(tb)
    for x in range(0,35):
        tb = db_refrigerators.iloc[x,2]  # kWh/year
        db_demand.append(tb)
    '''freezers'''
    for x in range(35,42):
        tb = db_refrigerators.iloc[x,1]  # kWh/year
        db_ee_classes_f.append(tb)
    for x in range(35,42):
        tb = db_refrigerators.iloc[x,2]  # kWh/year
        db_demand_f.append(tb)
    
    
    param_istat = codice_ID, n_Refrigerators,Refr_Capacity,Refr_Age,n_IntFreezer,n_Freezers,Freez_Age
    param_db = db_sizes,db_ee_classes,db_demand,db_ee_classes_f,db_demand_f
    EE_demand_refrigerators = Refrigerators_moirae(codice_ID,regione,param_istat,param_db)
    
    
    
    return EE_demand_refrigerators

class Refrigerators_moirae:
  
    def __init__(self,ID,regione,i_par,db_par):
        
        self.ID=ID
        self.region=regione
        
        self.refrigerators_demand = {}
        
        self.Num = i_par[1]               
        self.Size = i_par[2]
        self.Age = i_par[3]
 
        self.intFreezer = i_par[4].replace(2,0)    # Vano Congelatore interno
        self.Freezers = i_par[5].replace(2,0)      # Feezer 
        self.FreezerAge = i_par[6]  
                         
        self.EE_class={}
        for k in self.ID:
            age=self.Age[k]
            if age=='1 year':         EE='A+++'               
            elif age=='1-2 years':    EE='A++'                  
            elif age=='2-3 years':    EE='A+'                  
            elif age=='3-6 years':    EE='A'                    
            elif age=='6-10 years':   EE='B'                     
            elif age=='10-20 years':  EE='C'                
            elif age=='20 years':     EE='D'                         
            else:                     EE='A'     
            self.EE_class[k] = EE
            
            
        self.Freezer_EE_class={}
        for k in self.ID:
            age=self.FreezerAge[k]
            if age=='1 year':         EE='A+++'               
            elif age=='1-2 years':    EE='A++'                  
            elif age=='2-3 years':    EE='A+'                  
            elif age=='3-6 years':    EE='A'                    
            elif age=='6-10 years':   EE='B'                     
            elif age=='10-20 years':  EE='C'                
            elif age=='20 years':     EE='D'                         
            else:                     EE='A'     
            self.Freezer_EE_class[k] = EE
            
 
        '''Refrigerators'''   
        df1 = pd.DataFrame([self.Size, self.EE_class]).transpose()
        df2 = pd.DataFrame([db_par[0],db_par[1],db_par[2]]).transpose()
        df2.index = df2[0] + df2[1]
        one_app_demand = (df1[0]+df1[1]).apply((lambda x: df2.loc[x]))
        self.refrigerators_demand = (self.Num)*(one_app_demand[2])
        '''Freezers'''
        df1_f = pd.DataFrame([self.Freezer_EE_class]).transpose()
        df2_f = pd.DataFrame([db_par[3],db_par[4]]).transpose()
        df2_f.index = df2_f[0]
        one_app_demand_f = df1_f[0].apply((lambda x: df2_f.loc[x]))
        self.freezers_demand = (self.Freezers)*(one_app_demand_f[1])
        
        self.overall_refrigerators_demand = self.refrigerators_demand+self.freezers_demand
        
        '''mean consumption per household'''
        n_ref=len(self.overall_refrigerators_demand[self.overall_refrigerators_demand>0])
        overall_RefrigeratorDemand=self.overall_refrigerators_demand.sum()
        self.Mean_RefrigeratorDemand=round(overall_RefrigeratorDemand/n_ref,1) 

''' FRIGORIFERIE E CONGELATORI (norma UE)'''
def loadRefrigerators_CE(istat,db_refrigerators):

    refrigerators_idata = istat#.iloc[:,[0,298,299,300,301,374,375,88]]
    refrigerators_idata.rename(columns = {"id":"sample",
                        "q_8_2":"Num. Frigoriferi",
                        "q_8_5":"Capacità Frigorifero",
                        "q_8_6":"Dotazione Vano Congelatore",
                        "q_8_7":"Dotazione Freezer",
                        "q_8_3_ric":"Età Frigorifero",
                        "q_8_8_ric":"Età Freezer",
                        "reg":"Region"}, 
             inplace = True)
    
   
    codice_ID = refrigerators_idata.index
    regione = refrigerators_idata["Region"]
    regione = decodificRegion(regione)
    
    n_Refrigerators = refrigerators_idata["Num. Frigoriferi"]
    n_BoxFreezer = refrigerators_idata["Dotazione Vano Congelatore"]
    n_Freezers = refrigerators_idata["Dotazione Freezer"]
    Age_Refrigerators = refrigerators_idata["Età Frigorifero"]
    Age_Freezers = refrigerators_idata["Età Freezer"]
    
    '''
    Metodo basato su: 
        {Regolamento delegato (UE) n. 1060/2010 della Commissione del 28 settembre 2010 
         che integra la direttiva 2010/30/UE del Parlamento europeo e del Consiglio 
         per quanto riguarda l’etichettatura indicante il consumo d’energia degli 
         apparecchi di refrigerazione per uso domestico, 
         Gazzetta ufficiale dell’Unione europea del 30 novembre 2010}
     
    Il volume del frigorifero (V_refr), Tnom=5°C, si ottiene sottraendo 
    al volume totale (V) il volume del congelatore (V_free) o dello scomparto (V_box)
    Volume di riferimento del congelatore è 220 L in assunza di altri dati.
    
    VOLUME [L]
    
    '''
    Capacity = refrigerators_idata["Capacità Frigorifero"]
    capacity={}
    volume={}
    volume_box={}
    Tbox={}
    M={}
    N={}
    
    for i in codice_ID:
        C = Capacity.loc[i]
        
        if   C==1: C='piccolo';         V=150;   V_box=19;    T_box=5;    m=0.233; n=245                  
        elif C==2: C='medio';           V=250;   V_box=50;   T_box=0;    m=0.233; n=245               
        elif C==3: C='grande';          V=400;   V_box=100;   T_box=-6;   m=0.643; n=191                                
        elif C==4: C='molto_grande';    V=600;   V_box=200;   T_box=-12;  m=0.450; n=245                           
        else:      C='non_specificato'; V=0;     V_box=0;    T_box=0;    m=0;     n=0 
             
        capacity[i]=C
        volume[i]=V
        volume_box[i]=V_box
        Tbox[i]=T_box
        M[i]=m
        N[i]=n
    
    Tfreez=-18 # [°C]
    Mfreez=0.472
    Nfreez=286
    
    Veq_par = [Tbox, M, N, Tfreez, Mfreez, Nfreez]
    
    age_refrigerator={}
    age_freezer={}
    for i in codice_ID:
        y = Age_Refrigerators.loc[i]
        if y==1: y='1 year'                
        elif y==2: y='1-2 years'
        elif y==3: y='2-3 years'
        elif y==4: y='3-6 years'    
        elif y==5: y='6-10 years'    
        elif y==6: y='10-20 years'       
        elif y==7: y='20 years'  
        else: y='None'  
        age_refrigerator[i]=y
        
        yf = Age_Freezers.loc[i]
        if yf==1: yf='1 year'                
        elif yf==2: yf='1-2 years'
        elif yf==3: yf='2-3 years'
        elif yf==4: yf='3-6 years'    
        elif yf==5: yf='6-10 years'    
        elif yf==6: yf='10-20 years'       
        elif yf==7: yf='20 years'  
        else: yf='None'  
        age_freezer[i]=yf
      
    par = codice_ID,n_Refrigerators,n_BoxFreezer,n_Freezers,volume,volume_box,age_refrigerator,age_freezer,capacity,Veq_par
    
    EE_demand_refrigerators = Refrigerators_CE(codice_ID,regione,par)
    
    frighi_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
    frighi_info['frighi','N.frighi'] = n_Refrigerators
    frighi_info['frighi','Grandezza'] = Capacity
    frighi_info['frighi','CongelatoreFrigo'] = n_BoxFreezer.replace({1: 'Si', 2: ''})
    frighi_info['frighi','CongelatoreSeparato'] = n_Freezers.replace({1: 'Si', 2: ''})
    frighi_info['frighi','EtaFrigo'] = Age_Refrigerators
    frighi_info['frighi','EtaCongelatore'] = Age_Freezers
    
    
    return EE_demand_refrigerators, frighi_info

class Refrigerators_CE:
    
    '''
    metodo basato su:
    {Regolamento delegato (UE) n. 1060/2010, CE}
    '''
    
    def __init__(self,ID,regione,par):
        
        self.ID = ID
        self.region = regione
        
        self.refrigerators_demand = {}
        
        self.n_Refrigerators = par[1]
        self.n_BoxFreezer = par[2].replace(2,0)    
        self.n_Freezers = par[3].replace(2,0)    
        
        self.V = pd.Series(par[4], index = self.ID)
        self.V_freezer = 220 # [L] in mancanca di altri dati
        self.V_box = pd.Series(par[5], index = self.ID)
        
        self.V_refrigerator = self.V - (self.n_BoxFreezer*self.V_box) #!!!- (self.n_Freezers*self.V_freezer)
        
        self.RefrigeratorAge = par[6]    
        self.FreezerAge = par[7]  
        
        self.Capacity = par[8]
        self.FFc = 1  #fattore correttivo per sistema antibrina
        Tnom = pd.Series(par[9][0], index = self.ID) #Temp. nominale scompartimento
        M = pd.Series(par[9][1], index = self.ID) #coeff. per calcolo metodo SAEc
        N = pd.Series(par[9][2], index = self.ID) #coeff. per calcolo metodo SAEc
        Tfreez = par[9][3] #Temp. nominale congelatore
        Mfreez = par[9][4] #coeff. per calcolo metodo SAEc
        Nfreez = par[9][5] #coeff. per calcolo metodo SAEc
        Ga = 330 # giorni anno presenza continuativa utenti
        CH = 50 # [kWh/anno] consumo extra in presenza di frigorifero dotato di scoparto congelatore

        # Volume equivalente
        self.Veq = ( self.V_refrigerator*((25-Tnom)/20) + 
                        (self.n_BoxFreezer)*self.V_box*((25-Tfreez)/20) )*self.FFc
        self.Veq_freez = ( (self.n_Freezers)*self.V_freezer*((25-Tfreez)/20) )*self.FFc
        '''
        Classe di Efficienza Energetica
        EEI = Indice di Efficienza Energetica
        '''
        self.Refrigerator_classe={}
        self.Refrigerator_EEI={}
        for k in self.ID:
            age=self.RefrigeratorAge[k]
            if age=='1 year':         classe='A+++'; EEI=20               
            elif age=='1-2 years':    classe='A++';  EEI=(22+33)/2                     
            elif age=='2-3 years':    classe='A+';   EEI=(33+44)/2              
            elif age=='3-6 years':    classe='A';    EEI=(44+55)/2                
            elif age=='6-10 years':   classe='B';    EEI=(55+75)/2                 
            elif age=='10-20 years':  classe='C';    EEI=(75+95)/2            
            elif age=='20 years':     classe='D';    EEI=(95+110)/2                     
            else:                     classe='A';    EEI=65
            self.Refrigerator_classe[k] = classe
            self.Refrigerator_EEI[k] = EEI; self.Refrigerator_EEI = pd.Series(self.Refrigerator_EEI, index = self.ID)
                
        self.Freezer_classe={}
        self.Freezer_EEI={}
        for k in self.ID:
            age=self.FreezerAge[k]
            if age=='1 year':         classe='A+++'; EEI=20               
            elif age=='1-2 years':    classe='A++';  EEI=(22+33)/2                     
            elif age=='2-3 years':    classe='A+';   EEI=(33+44)/2              
            elif age=='3-6 years':    classe='A';    EEI=(44+55)/2                
            elif age=='6-10 years':   classe='B';    EEI=(55+75)/2                 
            elif age=='10-20 years':  classe='C';    EEI=(75+95)/2            
            elif age=='20 years':     classe='D';    EEI=(95+110)/2                     
            else:                     classe='A';    EEI=65
            self.Freezer_classe[k] = classe
            self.Freezer_EEI[k] = EEI; self.Freezer_EEI = pd.Series(self.Freezer_EEI, index = self.ID)
        
        # Consumo annuo standard di energia: SECc (kWh/anno)
        self.SEC = self.Veq*M + N + CH*self.n_BoxFreezer
        self.SEC_freez = self.Veq_freez*Mfreez + Nfreez
        
        # fattore di correzione basato sul numero di dispositivi in quanto 
        # vengono forniti i dati solo del dispostivo principale
        f={}
        for k in self.ID:
            nref = self.n_Refrigerators[k]     
            if nref == 1:     fact = 1
            elif nref == 2:   fact = 1.3
            elif nref > 2:    fact = 1.5
            else:             fact = 0
            f[k] = fact; f = pd.Series(f, index = self.ID)
        ff = pd.Series(self.n_Freezers, index = self.ID)
        
        # Consumo annuo (kWh/anno)
        self.AC = f*self.SEC*(self.Refrigerator_EEI/100)  
        self.AC_freez = ff*self.SEC_freez*(self.Freezer_EEI/100)
        
        self.refrigerators_demand = (Ga/365)*(self.AC + self.AC_freez) 
        self.refrigerators_demand=round(self.refrigerators_demand, 1)
        
        '''mean consumption per household'''
        n_ref=len(self.refrigerators_demand[self.refrigerators_demand>0])
        overall_RefrigeratorDemand=self.refrigerators_demand.sum()
        self.Mean_RefrigeratorDemand=round(overall_RefrigeratorDemand/n_ref,1) 
        
    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
        
    def extractBdData(self,BdID):
        bdData = {'AnnualConsumption_EE: Refrigerators' : self.refrigerators_demand[BdID]}
        return bdData

''' GRANDI ELETTRODOMESTICI (Lavatrici, Asciugatrici e Lavastoviglie) (database)'''
def loadAppliances_CE(istat,db_appliances):
    
    appliances_idata = istat#.iloc[:n_data,[0,303,304,305,306,307,311,313,314,316,376,377,378,88]]
    appliances_idata.rename(columns = {"id":"sample",
                        "q_8_13":"Capacità Lavatrice",   # se 0 --> non è presente
                        "q_8_14":"NUm. Cicli Settimanali Lavatrice",
                        "q_8_15A":"Num. Cicli Settimanali a Bassa Temperatura",
                        "q_8_15B":"Num. Cicli a Media Temperatura",
                        "q_8_15C":"Num. Cicli ad Alta Temperatura",                                                
                        "q_8_17":"Funzione Asciugatrice",  # 1-0    ....)*fct_dryer.....
                        "q_8_21":"Capacità Asciugatrice",  # se 0 --> non è presente o è incorporata con lavatrice (v. 8_17)
                        "q_8_22":"Num. Cicli Settimanali Asciugatrice",
                        "q_8_26":"Num. Cicli Settimanali Lavastoviglie",  # se 0 --> non è presente
                        "q_8_11_ric":"Età Lavatrice",
                        "q_8_19_ric":"Età Asciugatrice",
                        "q_8_24_ric":"Età Lavastoviglie",
                        "reg":"Region",
                        }, 
             inplace = True)
    
    '_idata  ----------------------------------------------------------------' 
    codice_ID = appliances_idata.index
    regione = appliances_idata["Region"]
    regione = decodificRegion(regione)
    
    n_Wash= appliances_idata["NUm. Cicli Settimanali Lavatrice"]
    Washing = appliances_idata["Capacità Lavatrice"]
    Washing_Dryer = appliances_idata["Funzione Asciugatrice"]
    Dryer = appliances_idata["Capacità Asciugatrice"]
    
    washing={}
    dryer={} 
    Washing_Capacity={}
    Dryer_Capacity={}
    
    for i in codice_ID:
        C = Washing.loc[i]
        D = Dryer.loc[i]
        
        if C>0:
            if C==1: C='piccolo'       
            elif C==2: C='medio'                             
            elif C==3: C='grande'       
            washing[i]=1                                                    
            Washing_Capacity[i]=C
        else: washing[i]=0; Washing_Capacity[i]='none'
         
        if D>0:        
            if D==1: D='piccolo'       
            elif D==2: D='medio'                             
            elif D==3: D='grande'
            dryer[i]=1              
            Dryer_Capacity[i]=D
        else: dryer[i]=0; Dryer_Capacity[i]='none'

        
    Washing_Low_T_cycles = appliances_idata["Num. Cicli Settimanali a Bassa Temperatura"]
    Washing_Med_T_cycles = appliances_idata["Num. Cicli a Media Temperatura"]
    Washing_High_T_cycles = appliances_idata["Num. Cicli ad Alta Temperatura"]
    
    Dryer_cycles = appliances_idata["Num. Cicli Settimanali Asciugatrice"]
    Dishwasher_cycles = appliances_idata["Num. Cicli Settimanali Lavastoviglie"]
   
    Age_Washing = appliances_idata["Età Lavatrice"]
    Age_Dryer = appliances_idata["Età Asciugatrice"]
    Age_Dishwasher = appliances_idata["Età Lavastoviglie"]
    
    washing_age={}
    dryer_age={}
    dishwasher_age={}

    for i in codice_ID:
        yw = Age_Washing.loc[i]
        if yw==1: yw='1 year'                
        elif yw==2: yw='1-2 years'
        elif yw==3: yw='2-3 years'
        elif yw==4: yw='3-6 years'    
        elif yw==5: yw='6-10 years'    
        elif yw==6: yw='10-20 years'       
        elif yw==7: yw='20 years'  
        else: yw='None'  
        washing_age[i]=yw

        yd = Age_Dryer.loc[i]
        if yd==1: yd='1 year'                
        elif yd==2: yd='1-2 years'
        elif yd==3: yd='2-3 years'
        elif yd==4: yd='3-6 years'    
        elif yd==5: yd='6-10 years'    
        elif yd==6: yd='10-20 years'       
        elif yd==7: yd='20 years'  
        else: yd='None'  
        dryer_age[i]=yd

        ydw = Age_Dishwasher.loc[i]
        if ydw==1: ydw='1 year'                
        elif ydw==2: ydw='1-2 years'
        elif ydw==3: ydw='2-3 years'
        elif ydw==4: ydw='3-6 years'    
        elif ydw==5: ydw='6-10 years'    
        elif ydw==6: ydw='10-20 years'       
        elif ydw==7: ydw='20 years'  
        else: ydw='None'  
        dishwasher_age[i]=ydw

    
    'db_  -------------------------------------------------------------------'
    db_washing_capacity=[]
    db_washing_classes=[]
    db_washing_demand=[] 
    db_dryer_capacity=[]
    db_dryer_classes=[]
    db_dryer_demand=[]
    db_dishwasher_classes=[]
    db_dishwasher_demand=[]
    
    '''WashingMachines'''
    for x in range(0,28):
        tb = db_appliances.iloc[x,0]  
        db_washing_capacity.append(tb)
    for x in range(0,28):
        tb = db_appliances.iloc[x,1]  
        db_washing_classes.append(tb)
    for x in range(0,28):
        tb = db_appliances.iloc[x,2]  # kWh/cycle
        db_washing_demand.append(tb)
        
    '''DryerMachine'''
    for x in range(28,56):
        tb = db_appliances.iloc[x,0]  
        db_dryer_capacity.append(tb)
    for x in range(28,56):
        tb = db_appliances.iloc[x,1]  
        db_dryer_classes.append(tb)
    for x in range(28,56):
        tb = db_appliances.iloc[x,2]  # kWh/cycle
        db_dryer_demand.append(tb)
    
    '''Dishwashers'''
    for x in range(56,63):
        tb = db_appliances.iloc[x,1]  
        db_dishwasher_classes.append(tb)
    for x in range(56,63):
        tb = db_appliances.iloc[x,2]  # kWh/cycle
        db_dishwasher_demand.append(tb)
    
    
    param_istat = codice_ID,washing,dryer,Washing_Capacity,Dryer_Capacity,Washing_Low_T_cycles,Washing_Med_T_cycles,Washing_High_T_cycles,Dryer_cycles,Dishwasher_cycles,washing_age,dryer_age,dishwasher_age,n_Wash            
    param_db = db_washing_capacity,db_washing_classes,db_washing_demand,db_dryer_capacity,db_dryer_classes,db_dryer_demand,db_dishwasher_classes,db_dishwasher_demand
    
    EE_demand_appliances_2 = Appliances__CE(codice_ID,regione,param_istat,param_db)
    
    big_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
    big_info['grandi_elettro','Lavatrice'] = pd.Series(Washing_Capacity,index = codice_ID).replace({'none',''})
    big_info['grandi_elettro','LavaggiSettBassaT'] = Washing_Low_T_cycles
    big_info['grandi_elettro','LavaggiSettMediaT'] = Washing_Med_T_cycles
    big_info['grandi_elettro','LavaggiSettAltaT'] = Washing_High_T_cycles
    big_info['grandi_elettro','Asciugatrice'] = pd.Series(Dryer_Capacity,index = codice_ID).replace({'none',''})
    big_info['grandi_elettro','N_CicliAsciugatrice'] = Dryer_cycles
    big_info['grandi_elettro','N_CicliLavastoviglie'] = pd.Series(Dishwasher_cycles,index = codice_ID).replace({0,''})
    big_info['grandi_elettro','EtaLavatrice'] = pd.Series(washing_age,index = codice_ID).replace('None','')
    big_info['grandi_elettro','EtaAsciugatrice'] = pd.Series(dryer_age,index = codice_ID).replace('None','')
    big_info['grandi_elettro','EtaLavastoviglie'] = pd.Series(dishwasher_age,index = codice_ID).replace('None','')
    
    
    return EE_demand_appliances_2, big_info

class Appliances__CE:
    
      
    def __init__(self,ID,regione,i_par,db_par):
        
        self.ID = ID
        self.region = regione

        self.big_appliances_demand = {}
        
        self.WashingMachine=i_par[1]               
        self.DryerWashingMachine=pd.Series(i_par[2], index = self.ID)
        self.WashingCapacity=i_par[3]
        self.DryerCapacity=i_par[4]
        self.n_LowTCycles=i_par[5]
        self.n_MedTCycles=i_par[6]
        self.n_HighTCycles=i_par[7]
        self.n_DryerCycles=i_par[8]
        self.n_DishWashCycle=i_par[9]
        self.WashingMachineAge=i_par[10]
        self.DryerAge=i_par[11]
        self.DishwasherAge=i_par[12]
        
        self.n_WashCycles = i_par[13]
        self.n_WashCycles=self.n_WashCycles.replace(98,1)
        
        weeks=47
        # Fattori di correzione per uso lavtrice a diverse temperature (fonte MICENE:
        # A. D. Franco Di Andrea, "Curve di carico dei principali elettrodomestici 
        # e degli apparecchi di illuminazione," MICENE - MIsure dei Consumi di 
        # ENergia Elettrica in 110 abitazioni Italiane2004, Available: www.eerg.it
        LT=0.3
        MT=1
        HT=2.56
        

        '''WashingMachines'''   

        cycle_demand = 1.8 #kWh/cycle
        self.WashingMachine_demand = (LT*self.n_LowTCycles+MT*self.n_MedTCycles+HT*self.n_HighTCycles)*(cycle_demand)*weeks
        self.WashingMachine_demand += 1.8*(self.n_WashCycles-(self.n_LowTCycles+self.n_MedTCycles+self.n_HighTCycles))*weeks/4
        self.WashingMachine_demand += self.DryerWashingMachine*1   # devo tenere conto del consumo dell'asciugatrice integrata nella lavatrice
        
        
        '''DryerMachines'''   

        cycle_demand = 1 #kWh/cycle
        self.Dryer_demand = (self.n_DryerCycles)*(cycle_demand)*weeks
        
        '''Dishwashers'''   

        cycle_demand = 1.5 # kWh/cycle
        self.Dishwasher_demand = (self.n_DishWashCycle)*(cycle_demand)*weeks       
        
        
        self.big_appliances_demand=(self.WashingMachine_demand+self.Dryer_demand+self.Dishwasher_demand)
        
        
        '''Mean Consumption per Household'''
        n_wm=len(self.WashingMachine_demand[self.WashingMachine_demand>0])
        overall_WashingMachineDemand=self.WashingMachine_demand.sum()
        self.Mean_WashingMachineDemand=round(overall_WashingMachineDemand/n_wm,1)
         
        n_dm=len(self.Dryer_demand[self.Dryer_demand>0])
        overall_DryerDemand=self.Dryer_demand.sum()
        if n_dm>0:
            self.Mean_DryerDemand=round(overall_DryerDemand/n_dm,1)

        n_d=len(self.Dishwasher_demand[self.Dishwasher_demand>0])
        overall_DishwasherDemand=self.Dishwasher_demand.sum()
        if n_d > 0:
            self.Mean_DishwasherDemand=round(overall_DishwasherDemand/n_d,1)        

    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
    
    
    def extractBdData(self,BdID):
        bdData = {'AnnualConsumption_EE: Washing' : self.WashingMachine_demand[BdID],
                  'AnnualConsumption_EE: Dryers' : self.Dryer_demand[BdID],
                  'AnnualConsumption_EE: Dishwashers': self.Dishwasher_demand[BdID]
                  }
        return bdData

''' GRANDI ELETTRODOMESTICI (Lavatrici, Asciugatrici e Lavastoviglie) (database)'''
def loadAppliances_moirae(istat,db_appliances):
    
    appliances_idata = istat#.iloc[,[0,303,304,305,306,307,311,313,314,316,376,377,378,88]]
    appliances_idata.rename(columns = {"id":"sample",
                        "q_8_13":"Capacità Lavatrice",   # se 0 --> non è presente
                        "q_8_14":"NUm. Cicli Settimanali Lavatrice",
                        "q_8_15A":"Num. Cicli Settimanali a Bassa Temperatura",
                        "q_8_15B":"Num. Cicli a Media Temperatura",
                        "q_8_15C":"Num. Cicli ad Alta Temperatura",                                                
                        "q_8_17":"Funzione Asciugatrice",  # 1-0    ....)*fct_dryer.....
                        "q_8_21":"Capacità Asciugatrice",  # se 0 --> non è presente o è incorporata con lavatrice (v. 8_17)
                        "q_8_22":"Num. Cicli Settimanali Asciugatrice",
                        "q_8_26":"Num. Cicli Settimanali Lavastoviglie",  # se 0 --> non è presente
                        "q_8_11_ric":"Età Lavatrice",
                        "q_8_19_ric":"Età Asciugatrice",
                        "q_8_24_ric":"Età Lavastoviglie",
                        "reg":"Region",
                        }, 
             inplace = True)
    
    '_idata  ----------------------------------------------------------------' 
    codice_ID = appliances_idata.index
    regione = appliances_idata["Region"]
    regione = decodificRegion(regione)
    
    n_Wash= appliances_idata["NUm. Cicli Settimanali Lavatrice"]
    Washing = appliances_idata["Capacità Lavatrice"]
    Washing_Dryer = appliances_idata["Funzione Asciugatrice"]
    Dryer = appliances_idata["Capacità Asciugatrice"]
    
    washing={}
    dryer={} 
    Washing_Capacity={}
    Dryer_Capacity={}
    
    for i in codice_ID:
        C = Washing[i]
        D = Dryer[i]
        
        if C>0:
            if C==1: C='piccolo'       
            elif C==2: C='medio'                             
            elif C==3: C='grande'       
            washing[i]=1                                                    
            Washing_Capacity[i]=C
        else: washing[i]=0; Washing_Capacity[i]='none'
         
        if D>0:        
            if D==1: D='piccolo'       
            elif D==2: D='medio'                             
            elif D==3: D='grande'
            dryer[i]=1              
            Dryer_Capacity[i]=D
        else: dryer[i]=0; Dryer_Capacity[i]='none'

        
    Washing_Low_T_cycles = appliances_idata["Num. Cicli Settimanali a Bassa Temperatura"]
    Washing_Med_T_cycles = appliances_idata["Num. Cicli a Media Temperatura"]
    Washing_High_T_cycles = appliances_idata["Num. Cicli ad Alta Temperatura"]
    
    Dryer_cycles = appliances_idata["Num. Cicli Settimanali Asciugatrice"]
    Dishwasher_cycles = appliances_idata["Num. Cicli Settimanali Lavastoviglie"]
   
    Age_Washing = appliances_idata["Età Lavatrice"]
    Age_Dryer = appliances_idata["Età Asciugatrice"]
    Age_Dishwasher = appliances_idata["Età Lavastoviglie"]
    
    washing_age={}
    dryer_age={}
    dishwasher_age={}

    for i in codice_ID:
        yw = Age_Washing.loc[i]
        if yw==1: yw='1 year'                
        elif yw==2: yw='1-2 years'
        elif yw==3: yw='2-3 years'
        elif yw==4: yw='3-6 years'    
        elif yw==5: yw='6-10 years'    
        elif yw==6: yw='10-20 years'       
        elif yw==7: yw='20 years'  
        else: yw='None'  
        washing_age[i]=yw

        yd = Age_Dryer.loc[i]
        if yd==1: yd='1 year'                
        elif yd==2: yd='1-2 years'
        elif yd==3: yd='2-3 years'
        elif yd==4: yd='3-6 years'    
        elif yd==5: yd='6-10 years'    
        elif yd==6: yd='10-20 years'       
        elif yd==7: yd='20 years'  
        else: yd='None'  
        dryer_age[i]=yd

        ydw = Age_Dishwasher.loc[i]
        if ydw==1: ydw='1 year'                
        elif ydw==2: ydw='1-2 years'
        elif ydw==3: ydw='2-3 years'
        elif ydw==4: ydw='3-6 years'    
        elif ydw==5: ydw='6-10 years'    
        elif ydw==6: ydw='10-20 years'       
        elif ydw==7: ydw='20 years'  
        else: ydw='None'  
        dishwasher_age[i]=ydw

    
    'db_  -------------------------------------------------------------------'
    db_washing_capacity=[]
    db_washing_classes=[]
    db_washing_demand=[] 
    db_dryer_capacity=[]
    db_dryer_classes=[]
    db_dryer_demand=[]
    db_dishwasher_classes=[]
    db_dishwasher_demand=[]
    
    '''WashingMachines'''
    for x in range(0,28):
        tb = db_appliances.iloc[x,0]  
        db_washing_capacity.append(tb)
    for x in range(0,28):
        tb = db_appliances.iloc[x,1]  
        db_washing_classes.append(tb)
    for x in range(0,28):
        tb = db_appliances.iloc[x,2]  # kWh/cycle
        db_washing_demand.append(tb)
        
    '''DryerMachine'''
    for x in range(28,56):
        tb = db_appliances.iloc[x,0]  
        db_dryer_capacity.append(tb)
    for x in range(28,56):
        tb = db_appliances.iloc[x,1]  
        db_dryer_classes.append(tb)
    for x in range(28,56):
        tb = db_appliances.iloc[x,2]  # kWh/cycle
        db_dryer_demand.append(tb)
    
    '''Dishwashers'''
    for x in range(56,63):
        tb = db_appliances.iloc[x,1]  
        db_dishwasher_classes.append(tb)
    for x in range(56,63):
        tb = db_appliances.iloc[x,2]  # kWh/cycle
        db_dishwasher_demand.append(tb)
    
    
    param_istat = codice_ID,washing,dryer,Washing_Capacity,Dryer_Capacity,Washing_Low_T_cycles,Washing_Med_T_cycles,Washing_High_T_cycles,Dryer_cycles,Dishwasher_cycles,washing_age,dryer_age,dishwasher_age,n_Wash            
    param_db = db_washing_capacity,db_washing_classes,db_washing_demand,db_dryer_capacity,db_dryer_classes,db_dryer_demand,db_dishwasher_classes,db_dishwasher_demand
    
    EE_demand_appliances = Appliances_moirae(codice_ID,regione,param_istat,param_db)
    return EE_demand_appliances

class Appliances_moirae:
    # Based on MOIRAE
      
    def __init__(self,ID,regione,i_par,db_par):
        
        self.ID = ID
        self.region = regione

        self.big_appliances_demand = {}
        
        self.WashingMachine=i_par[1]               
        self.DryerWashingMachine=pd.Series(i_par[2])
        self.WashingCapacity=i_par[3]
        self.DryerCapacity=i_par[4]
        self.n_LT=i_par[5]
        self.n_MT=i_par[6]
        self.n_HT=i_par[7]
        self.n_DryerCycles=i_par[8]
        self.n_DishWashCycle=i_par[9]
        self.WashingMachineAge=i_par[10]
        self.DryerAge=i_par[11]
        self.DishwasherAge=i_par[12]
        
        self.n_WashingCycles = i_par[13]
        self.n_WashingCycles=self.n_WashingCycles.replace(98,1)
        self.nWM_T0 = ( self.n_LT + self.n_MT + self.n_HT ) 
        weeks=47
        # Fattori di correzione per uso lavtrice a diverse temperature (fonte MICENE:
        # A. D. Franco Di Andrea, "Curve di carico dei principali elettrodomestici 
        # e degli apparecchi di illuminazione," MICENE - MIsure dei Consumi di 
        # ENergia Elettrica in 110 abitazioni Italiane2004, Available: www.eerg.it
        LT=0.299799
        MT=1.000000
        HT=2.559058
        
        self.WashingMachine_class={}
        for k in self.ID:
            age=self.WashingMachineAge[k]
            if age=='1 year':         EE='A+++'               
            elif age=='1-2 years':    EE='A++'                  
            elif age=='2-3 years':    EE='A+'                  
            elif age=='3-6 years':    EE='A'                    
            elif age=='6-10 years':   EE='B'                     
            elif age=='10-20 years':  EE='C'                
            elif age=='20 years':     EE='D'                         
            else:                     EE='D'     
            self.WashingMachine_class[k] = EE
            
        self.DryerMachine_class={}
        for k in self.ID:
            age=self.DryerAge[k]
            if age=='1 year':         EE='A+++'               
            elif age=='1-2 years':    EE='A++'                  
            elif age=='2-3 years':    EE='A+'                  
            elif age=='3-6 years':    EE='A'                    
            elif age=='6-10 years':   EE='B'                     
            elif age=='10-20 years':  EE='C'                
            elif age=='20 years':     EE='D'                         
            else:                     EE='D'     
            self.DryerMachine_class[k] = EE
            
        self.Dishwasher_class={}
        for k in self.ID:
            age=self.DishwasherAge[k]
            if age=='1 year':         EE='A+++'               
            elif age=='1-2 years':    EE='A++'                  
            elif age=='2-3 years':    EE='A+'                  
            elif age=='3-6 years':    EE='A'                    
            elif age=='6-10 years':   EE='B'                     
            elif age=='10-20 years':  EE='C'                
            elif age=='20 years':     EE='D'                         
            else:                     EE='D'     
            self.Dishwasher_class[k] = EE


        '''WashingMachines'''   
        self.nWM_0 = self.n_WashingCycles - self.nWM_T0 # num. cycle performed at unknown temperature

        df1 = pd.DataFrame([self.WashingCapacity, self.WashingMachine_class]).transpose()
        df2 = pd.DataFrame([db_par[0],db_par[1],db_par[2]]).transpose()
        df2.index = df2[0] + df2[1]
        yearly_demand = (df1[0]+df1[1]).apply((lambda x: df2.loc[x]))
        yearly_demand_mean = yearly_demand[2].mean()
        
        self.corr_coeff={}
        for j in self.ID:
            if self.n_WashingCycles[j] >= 1:
                c = ( LT*self.n_LT[j] + MT*self.n_MT[j] + HT*self.n_HT[j] + self.nWM_0[j])/( self.nWM_T0[j] + self.nWM_0[j] )
            elif self.n_WashingCycles[j] == 0: 
                c = 0.
            self.corr_coeff[j]=c
        self.corr_coeff=pd.Series(self.corr_coeff)
                                         
        self.WashingMachine_demand = ( self.corr_coeff )*( yearly_demand[2] )    # kWh/year       
        # self.WashingMachine_demand += ( self.nWM_0 )*( yearly_demand_mean )      # kWh/year       
        self.WashingMachine_demand = pd.Series(self.WashingMachine_demand)                           

        
        '''DryerMachines'''   
        df1 = pd.DataFrame([self.DryerCapacity, self.DryerMachine_class]).transpose()
        df2 = pd.DataFrame([db_par[3],db_par[4],db_par[5]]).transpose()
        df2.index = df2[0] + df2[1]
        yearly_demand = (df1[0]+df1[1]).apply((lambda x: df2.loc[x]))
        n0={}
        for j in self.ID:
            if self.n_DryerCycles[j] >= 1:
                c = 1
            elif self.n_DryerCycles[j] == 0: 
                c = 0.
            n0[j]=c
        n0=pd.Series(n0)
        self.Dryer_demand = n0*( yearly_demand[2] )          # kWh/year
        
        '''Dishwashers'''   
        df1 = pd.DataFrame([self.Dishwasher_class]).transpose()
        df2 = pd.DataFrame([db_par[6],db_par[7]]).transpose()
        df2.index = df2[0]
        yearly_demand = df1[0].apply((lambda x: df2.loc[x]))
        n0={}
        for j in self.ID:
            if self.n_DishWashCycle[j] >= 1:
                c = 1
            elif self.n_DishWashCycle[j] == 0: 
                c = 0.
            n0[j]=c
        n0=pd.Series(n0)
        self.Dishwasher_demand = n0*(yearly_demand[1])      # kWh/year
        
        
        self.big_appliances_demand=(self.WashingMachine_demand 
                                    +self.Dryer_demand
                                    +self.Dishwasher_demand)
        
        
        '''Mean Consumption per Household'''
        n_wm=len(self.WashingMachine_demand[self.WashingMachine_demand>0])
        overall_WashingMachineDemand=self.WashingMachine_demand.sum()
        if n_wm > 0:
            self.Mean_WashingMachineDemand=round(overall_WashingMachineDemand/n_wm,1)
         
        n_dm=len(self.Dryer_demand[self.Dryer_demand>0])
        overall_DryerDemand=self.Dryer_demand.sum()
        if n_dm > 0:
            self.Mean_DryerDemand=round(overall_DryerDemand/n_dm,1)

        n_d=len(self.Dishwasher_demand[self.Dishwasher_demand>0])
        overall_DishwasherDemand=self.Dishwasher_demand.sum()
        if n_d > 0:
            self.Mean_DishwasherDemand=round(overall_DishwasherDemand/n_d,1)        

    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
    
    
    def extractBdData(self,BdID):
        bdData = {'AnnualConsumption_EE: Washing' : self.WashingMachine_demand[BdID],
                  'AnnualConsumption_EE: Dryers' : self.Dryer_demand[BdID],
                  'AnnualConsumption_EE: Dishwashers': self.Dishwasher_demand[BdID]
                  }
        return bdData        
        

''' TELEVISORI '''
def loadScreens(istat,db_screens):

    screens_idata = istat#.iloc[:n_data,[0,343,344,345,346,347,348,351,352,353,354,88]]
    screens_idata.rename(columns = {"id":"sample",
                        "q_8_40":"n_TVs",
                        "q_8_41A":"n_CathRayTube",
                        "q_8_41B":"n_Plasma",
                        "q_8_41C":"n_LCD",
                        "q_8_41D":"n_LED",
                        "q_8_42":"TV hours",
                        "q_8_45A":"PC desktop",
                        "q_8_45B":"PC laptop",
                        "q_8_46":"PC hours",
                        "q_8_47":"PC utilizzo",
                        "reg":"Region"}, 
             inplace = True)
    '_idata'
    codice_ID = screens_idata.index
    regione = screens_idata["Region"]
    regione = decodificRegion(regione)
    
    nTVs = screens_idata["n_TVs"]
    tv_CRT = screens_idata["n_CathRayTube"]
    tv_PLS = screens_idata["n_Plasma"]
    tv_LCD = screens_idata["n_LCD"]
    tv_LED = screens_idata["n_LED"]
    
    TV_oph = screens_idata["TV hours"]
    
    pc_desk = screens_idata["PC desktop"]
    pc_lapt = screens_idata["PC laptop"]
    
    PC_oph = screens_idata["PC hours"]
    pc_use = screens_idata["PC utilizzo"]
    
    'db_'
    P_CRT = db_screens.iloc[0,0]  # kW
    P_PLS = db_screens.iloc[1,0]  # kW
    P_LCD = db_screens.iloc[2,0]  # kW
    P_LED = db_screens.iloc[3,0]  # kW
    
    P_CRT_sb = db_screens.iloc[0,1]  # kW
    P_PLS_sb = db_screens.iloc[1,1]  # kW
    P_LCD_sb = db_screens.iloc[2,1]  # kW
    P_LED_sb = db_screens.iloc[3,1]  # kW
    
    EE_desk = db_screens.iloc[5,0]  # kWh
    EE_lapt = db_screens.iloc[6,0]  # kWh 
    
    param_tv = codice_ID,tv_CRT,tv_PLS,tv_LCD,tv_LED,TV_oph,P_CRT,P_PLS,P_LCD,P_LED, P_CRT_sb,P_PLS_sb ,P_LCD_sb, P_LED_sb , nTVs
    param_pc = pc_desk,pc_lapt,PC_oph,EE_desk,EE_lapt
    EE_demand_TVs = Screens(codice_ID,regione,param_tv,param_pc)
    
    screens_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
    screens_info['schermi','N_TV_catodico'] = tv_CRT
    screens_info['schermi','N_TV_plasma'] = tv_PLS
    screens_info['schermi','N_TV_LCD'] = tv_LCD
    screens_info['schermi','N_TV_LED'] = tv_LED
    screens_info['schermi','OreAccansioneTV'] = EE_demand_TVs.TVhours
    screens_info['schermi','N_PC_fissi'] = pc_desk
    screens_info['schermi','N_PC_portatili'] = pc_lapt
    screens_info['schermi','EtaLavatrice'] = EE_demand_TVs.PChours
    
    
    
    return EE_demand_TVs, screens_info

class Screens:
    
    def __init__(self,ID,regione,TV_par,PC_par):
        
        self.ID = ID
        self.region= regione
        
        self.TVs_demand = {}
        self.PCs_demand = {}
        
        gganno = 330  #giorni/anno
        
        self.TVhours_class = pd.Series(TV_par[5], index = self.ID)
        self.PChours_class = pd.Series(PC_par[2], index = self.ID)
        '''
        In media, quante ore al giorno è acceso il Suo televisore/PC? 
        Se utilizza più di un televisore sommi le ore in cui ciascun apparecchio resta acceso.
        - Meno di due ………………………………….1
        - Da 2 a meno di 4 ore ………………………2
        - Da 4 a meno di 6 ore ………………………3
        - da 6 a meno di 12 ore …..…………………4
        - 12 ore o più..……………..…………………5
        '''
        self.TVhours={}
        self.PChours={}
        for k in self.ID:
            h = self.TVhours_class[k]
            hh= self.PChours_class[k]
            if h==1: h=1.5               
            elif h==2: h=3                 
            elif h==3: h=5                 
            elif h==4: h=8           
            elif h==5: h=12                                     
            else: h=0    
            self.TVhours[k] = h   
            
            if hh==1: hh=1.5               
            elif hh==2: hh=3                 
            elif hh==3: hh=5                 
            elif hh==4: hh=8           
            elif hh==5: hh=12                                     
            else: hh=0  
            self.PChours[k] = hh
        self.TVhours=pd.Series(self.TVhours, index = self.ID) # ore totali/giorno
        self.PChours=pd.Series(self.PChours, index = self.ID) # ore totali/giorno
        
        self.Num_TV_CRT = TV_par[1]
        self.Num_TV_PLS = TV_par[2]
        self.Num_TV_LCD = TV_par[3]
        self.Num_TV_LED = TV_par[4]
        self.P_CRT = TV_par[6]
        self.P_PLS = TV_par[7]
        self.P_LCD = TV_par[8]
        self.P_LED = TV_par[9]
        self.P_CRT_sb = TV_par[10]
        self.P_PLS_sb = TV_par[11]
        self.P_LCD_sb = TV_par[12]
        self.P_LED_sb = TV_par[13]
        self.P_av = (self.P_CRT+self.P_PLS+self.P_LCD+self.P_LED)/4 
        self.P_av_sb = (self.P_CRT_sb+self.P_PLS_sb+self.P_LCD_sb+self.P_LED_sb)/4 
        self.n_TVs = TV_par[14]
        self.Num_TV_CRT = self.Num_TV_CRT.replace(99,0)
        self.Num_TV_PLS = self.Num_TV_PLS.replace(99,0)
        self.Num_TV_LCD = self.Num_TV_LCD.replace(99,0)
        self.Num_TV_LED = self.Num_TV_LED.replace(99,0)    
        # number of undefined types of TVs 
        self.Num_TV_gn = self.n_TVs - (self.Num_TV_CRT + self.Num_TV_PLS + self.Num_TV_LCD + self.Num_TV_LED)
       
        self.Num_PC_desk = PC_par[0]
        self.Num_PC_lapt = PC_par[1]
        self.n_PCs = self.Num_PC_desk+self.Num_PC_lapt
        self.EE_desk = PC_par[3]
        self.EE_lapt = PC_par[4]
                    

        self.TVs_demand = gganno*self.TVhours*(self.Num_TV_CRT*self.P_CRT 
                                              + self.Num_TV_PLS*self.P_PLS
                                              + self.Num_TV_LCD*self.P_LCD
                                              + self.Num_TV_LED*self.P_LED
                                              + self.Num_TV_gn*self.P_av)/self.n_TVs + \
                          gganno*(24. - self.TVhours)*(self.Num_TV_CRT*self.P_CRT_sb 
                                              + self.Num_TV_PLS*self.P_PLS_sb
                                              + self.Num_TV_LCD*self.P_LCD_sb
                                              + self.Num_TV_LED*self.P_LED_sb
                                              + self.Num_TV_gn*self.P_av_sb)/self.n_TVs
        
        self.TVs_demand=self.TVs_demand.fillna(0)
        self.TVs_demand=round(self.TVs_demand, 1)
        
        # self.PCs_demand = gganno*self.PChours*(self.Num_PC_desk*self.P_desk
        #                                              + self.Num_PC_lapt*self.P_lapt)/(self.n_PCs)
        self.PCs_demand = (self.Num_PC_desk*self.EE_desk + self.Num_PC_lapt*self.EE_lapt) # kWh
        
        self.PCs_demand=self.PCs_demand.fillna(0)
        self.PCs_demand=round(self.PCs_demand, 1)
        
        self.screen_demand=self.TVs_demand+self.PCs_demand
        
        '''mean consumption per household'''
        n_tv=len(self.TVs_demand[self.TVs_demand>0])
        overall_TVsDemand=self.TVs_demand.sum()
        self.Mean_TVsDemand=round(overall_TVsDemand/n_tv,1) 
        
        n_pc=len(self.PCs_demand[self.PCs_demand>0])
        overall_PCsDemand=self.PCs_demand.sum()
        self.Mean_PCsDemand=round(overall_PCsDemand/n_pc,1) 

    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
        
    def extractBdData(self,ID):
        bdData = {'AnnualConsumption_EE: TVs' : self.TVs_demand[ID],
                  'AnnualConsumption_EE: PCs' : self.PCs_demand[ID]}
        return bdData


def loadCookings(istat,db_cookings):

    cookings_idata = istat#.iloc[:n_data,[0,324,325,326,327,328,329,330,379,88]]
    cookings_idata.rename(columns = {"id":"sample",
                        "q_8_29":"pres_Hobs",########
                        "q_8_30":"fuel_Hobs",
                        "q_8_31":"freq_Hobs",
                        "q_8_32":"pres_Oven",#######
                        "q_8_33":"fuel_Oven",
                        "q_8_34":"freq_Oven",
                        "q_8_37":"size_Oven",
                        "q_8_35_ric":"age_Oven",
                        "reg":"Region"}, 
             inplace = True)
    '_idata'
    codice_ID = cookings_idata.index
    regione = cookings_idata["Region"]
    regione = decodificRegion(regione)
    
    pres_Hobs = cookings_idata["pres_Hobs"]
    fuel_Hobs = cookings_idata["fuel_Hobs"]  # SE 1 natural gas
    freq_Hobs = cookings_idata["freq_Hobs"]
    
    pres_Oven = cookings_idata["pres_Oven"]
    fuel_Oven = cookings_idata["fuel_Oven"]  # SE 1 natural gas
    freq_Oven = cookings_idata["freq_Oven"]
    size_Oven = cookings_idata["size_Oven"]
    age_Oven = cookings_idata["age_Oven"]
    
    hobs_fuel={}
    oven_fuel={}
    oven_age={}
    weekly_use_hobs={}
    weekly_use_oven={}
    oven_vol={}
    
    for i in codice_ID:
         
        # fuel type
        ftype = fuel_Hobs.loc[i]
        if ftype==1: ftype='NaturalGas'
        elif ftype==2: ftype='Electric'
        elif ftype==3: ftype='LPG'
        elif ftype==4: ftype='Biomass'
        else: ftype='None'
        hobs_fuel[i]=ftype

        ftype = fuel_Oven.loc[i]
        if ftype==1: ftype='NaturalGas'
        elif ftype==2: ftype='Electric'
        elif ftype==3: ftype='LPG'
        elif ftype==4: ftype='Biomass'
        else: ftype='None'
        oven_fuel[i]=ftype       
        
        y = age_Oven.loc[i]
        if y==1: y='1 year'                
        elif y==2: y='1-2 years'
        elif y==3: y='2-3 years'
        elif y==4: y='3-6 years'    
        elif y==5: y='6-10 years'    
        elif y==6: y='10-20 years'       
        elif y==7: y='20 years'  
        # else: y='None'  
        oven_age[i]=y

        # in hours/week (MOIRAE)
        frq = freq_Hobs.loc[i]
        if frq==1: frq=9
        elif frq==2: ftype=5
        elif frq==3: frq=2.5
        elif frq==4: frq=1
        elif frq==5: frq=0.5
        else: frq=0
        weekly_use_hobs[i]=frq
        
        frq = freq_Oven.loc[i]
        if frq==1: frq=9
        elif frq==2: ftype=5
        elif frq==3: frq=2.5
        elif frq==4: frq=1
        elif frq==5: frq=0.5
        else: frq=0
        weekly_use_oven[i]=frq
        
        # Int. Oven Volume (lt) (fonte: GFK)
        sz = size_Oven.loc[i]
        if sz==1: sz=40 
        elif sz==2: sz=54
        elif sz==3: sz=65
        else: sz=0.
        oven_vol[i]=sz   
        
    'db_'
    Pel_hob = db_cookings.iloc[0,0]  # kW
    ee_hobs = db_cookings.iloc[0,1] # [-]
    Pel_oven = db_cookings.iloc[1,0]  # kW
    
    Pth_hob = db_cookings.iloc[2,0]  # kW
    te_gas_hobs = db_cookings.iloc[2,1] # [-]
    te_lpg_hobs = db_cookings.iloc[3,1] # [-]
    te_biom_hobs = db_cookings.iloc[4,1] # [-]    
    Pth_oven = db_cookings.iloc[5,0]  # kW    
    
    param_hobs = pres_Hobs,hobs_fuel,weekly_use_hobs,Pel_hob,ee_hobs,Pth_hob,te_gas_hobs,te_lpg_hobs,te_biom_hobs
    param_ovens = pres_Oven,oven_fuel,weekly_use_oven,oven_vol,oven_age,Pel_oven,Pth_oven
    EE_demand_cookings = Cookings(codice_ID,regione,param_hobs,param_ovens)
    
    
    cook_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
    cook_info['cottura','Fornello'] = pd.Series(hobs_fuel, index = codice_ID).replace({'None',''})
    cook_info['cottura','UsoFornello'] = EE_demand_cookings.hobs_weekly_use.replace({9,''})
    cook_info['cottura','Forno'] = pd.Series(oven_fuel, index = codice_ID).replace({'None',''})
    cook_info['cottura','UsoForno'] = EE_demand_cookings.ovens_weekly_use
    cook_info['cottura','EtaForno'] =  pd.Series(oven_age, index = codice_ID)
    cook_info['cottura','GrandezzaForno'] =  pd.Series(oven_vol   , index = codice_ID)
    
    
    return EE_demand_cookings, cook_info           

class Cookings:
    
    def __init__(self,ID,regione,hobs_par,ovens_par):
        
        self.ID = ID
        self.region= regione
        
        self.Hobs_EEdemand={}
        self.Ovens_EEdemand={}
        self.Ovens_EEdemand2={}
        self.Hobs_TEdemand={}
        self.Ovens_TEdemand={}
        
        weeks=47
        
        self.hobs=hobs_par[0].replace(2,0)   
        self.hobs_fuel=hobs_par[1]
        self.hobs_weekly_use=pd.Series(hobs_par[2], index = self.ID)
        self.Pel_hobs=hobs_par[3]        
        self.ee_hobs=hobs_par[4]  
        self.Pth_hobs=hobs_par[5]
        te_gas_hobs=hobs_par[6]
        te_lpg_hobs=hobs_par[7]
        te_biom_hobs=hobs_par[8]
        
        self.ovens=ovens_par[0].replace(2,0)   
        self.ovens_fuel=ovens_par[1]
        self.ovens_weekly_use=pd.Series(ovens_par[2], index = self.ID)
        self.Volume_ovens=pd.Series(ovens_par[3])
        self.age_class=ovens_par[4]
        self.Pel_ovens=ovens_par[5]     
        self.Pth_ovens=ovens_par[6]
        
        self.hobs_e = pd.DataFrame(np.zeros([len(self.ID),4],dtype = float), index = self.ID, columns = ['Electric','NaturalGas','LPG','Biomass'])
        
        self.te_hobs={}
        for j in self.ID:
            if self.hobs_fuel[j]=='Electric':
                dee=self.hobs.loc[j]*weeks*self.hobs_weekly_use.loc[j]*self.Pel_hobs
                dte=0
                self.hobs_e.loc[j,'Electric'] = dee
                                
            elif self.hobs_fuel[j] == 'NaturalGas':
                    te=te_gas_hobs
                    self.te_hobs[j]=te
                    dee=0
                    dte=self.hobs.loc[j]*weeks*self.hobs_weekly_use.loc[j]*self.Pth_hobs
                    self.hobs_e.loc[j,'NaturalGas'] = dte
                    
            elif self.hobs_fuel[j] == 'LPG':
                    te=te_lpg_hobs 
                    self.te_hobs[j]=te
                    dee=0
                    dte=self.hobs.loc[j]*weeks*self.hobs_weekly_use.loc[j]*self.Pth_hobs
                    self.hobs_e.loc[j,'LPG'] = dte
                    
            elif self.hobs_fuel[j] == 'Biomass':
                    te=te_biom_hobs
                    self.te_hobs[j]=te
                    dee=0
                    dte=self.hobs.loc[j]*weeks*self.hobs_weekly_use.loc[j]*self.Pth_hobs
                    self.hobs_e.loc[j,'Biomass'] = dte
            
            else:
                dee=0.0000001
                dte=0.0000001
                # self.hobs_e.loc[j,'Biomass'] = 0.0000001
                
            self.Hobs_EEdemand[j]=dee
            self.Hobs_TEdemand[j]=dte
            
        '''HOBS''' 
        # based on average P per cycle (MOIRAE)
        self.Hobs_EEdemand=pd.Series(self.Hobs_EEdemand, index = self.ID)
        self.Hobs_EEdemand=round(self.Hobs_EEdemand,1)
        self.Hobs_TEdemand=pd.Series(self.Hobs_TEdemand, index = self.ID)
        self.Hobs_TEdemand=round(self.Hobs_TEdemand,1)
         
        '''OVENS'''
        # based on CE No. 65/2014 (ANNEX I)
        self.Oven_classe={}
        self.Oven_EEI={}
        for k in self.ID:
            if self.ovens[k]==1:
                age=self.age_class[k]
                if age=='1 year':         classe='A+++'; EEI=45               
                elif age=='1-2 years':    classe='A++';  EEI=(45+62)/2                     
                elif age=='2-3 years':    classe='A+';   EEI=(62+82)/2              
                elif age=='3-6 years':    classe='A';    EEI=(82+107)/2                
                elif age=='6-10 years':   classe='B';    EEI=(107+132)/2                 
                elif age=='10-20 years':  classe='C';    EEI=(132+159)/2            
                elif age=='20 years':     classe='D';    EEI=159                     
                else:                     classe='NotAvailable(set=B)';    EEI=(107+132)/2
                self.Oven_classe[k] = classe
                self.Oven_EEI[k] = EEI; self.Oven_EEI = pd.Series(self.Oven_EEI)
            else: 
                self.Oven_classe[k]=0
                self.Oven_EEI[k] =0


        self.ovens_e = pd.DataFrame(np.zeros([len(self.ID),4],dtype = float), index = self.ID, columns = ['Electric','NaturalGas','LPG','Biomass'])
        
        for j in self.ID:
            if self.ovens_fuel[j]=='Electric':
                SEC_oven = 0.0042*self.Volume_ovens[j]+0.55  #kWh
                EC_oven= SEC_oven*self.Oven_EEI[j]/100
                
                dee1=self.ovens[j]*self.Pel_ovens*self.ovens_weekly_use[j]*weeks
                dee2=self.ovens[j]*(EC_oven)*weeks*self.ovens_weekly_use[j]
                dte=0.
                
                self.ovens_e.loc[j,'Electric'] = dee1
                
            elif self.ovens_fuel[j] != 'Electric': 
                dee1=0.
                dee2=0.

                SEC_oven = (0.044*self.Volume_ovens[j]+3.53)/3.6  #kWh
                EC_oven= SEC_oven*self.Oven_EEI[j]/100
                
                # dte=self.ovens[j]*(EC_oven)*weeks*self.ovens_weekly_use[j]
                dte=self.ovens[j]*self.Pth_ovens*self.ovens_weekly_use[j]*weeks
                
                self.ovens_e.loc[j,self.ovens_fuel[j]] = dte
               
            self.Ovens_EEdemand[j]=dee1                
            self.Ovens_EEdemand2[j]=dee2
            self.Ovens_TEdemand[j]=dte

   
        # based on average P per cycle (MOIRAE)
        self.Ovens_EEdemand=pd.Series(self.Ovens_EEdemand, index = self.ID)
        self.Ovens_EEdemand=round(self.Ovens_EEdemand,1)
        
        # based on EU standard (EE)
        self.Ovens_EEdemand2=pd.Series(self.Ovens_EEdemand2, index = self.ID)
        self.Ovens_EEdemand2=round(self.Ovens_EEdemand2,1)

        # based on MOIRAE (TE)
        self.Ovens_TEdemand=pd.Series(self.Ovens_TEdemand, index = self.ID)
        self.Ovens_TEdemand=round(self.Ovens_TEdemand,1)
        
        self.EEcookings_demand=self.Hobs_EEdemand+self.Ovens_EEdemand
        self.TEcookings_demand=self.Hobs_TEdemand+self.Ovens_TEdemand
        
        
        '''mean consumption per household'''
        n=len(self.Hobs_EEdemand[self.Hobs_EEdemand>0])
        overall_HobsEEDemand=self.Hobs_EEdemand.sum()
        if n > 0:
            self.Mean_HobsEEDemand=round(overall_HobsEEDemand/n,1) 

        n=len(self.Hobs_TEdemand[self.Hobs_TEdemand>0])
        overall_HobsTEDemand=self.Hobs_TEdemand.sum()
        if n > 0:
            self.Mean_HobsTEDemand=round(overall_HobsTEDemand/n,1)

        n=len(self.Ovens_EEdemand2[self.Ovens_EEdemand2>0])
        overall_OvensEEDemand=self.Ovens_EEdemand2.sum()
        if n > 0:
            self.Mean_OvensEEDemand=round(overall_OvensEEDemand/n,1) 

        n=len(self.Ovens_TEdemand[self.Ovens_TEdemand>0])
        overall_OvensTEDemand=self.Ovens_TEdemand.sum()
        if n > 0:
            self.Mean_OvensTEDemand=round(overall_OvensTEDemand/n,1)

    def setRegion(self,Regione):
        Region={'Region' : self.region[Regione]}
        return Region
        
    def extractBdData(self,ID):
        bdData = {'Fuel: Hobs' : self.hobs_fuel[ID],
                  'Fuel: Ovens' : self.ovens_fuel[ID],
                  'AnnualConsumption_EE: Hobs' : self.Hobs_EEdemand[ID],
                  'AnnualConsumption_TE: Hobs' : self.Hobs_TEdemand[ID],
                  'AnnualConsumption_EE: Ovens' : self.Ovens_EEdemand2[ID],
                  'AnnualConsumption_TE: Ovens' : self.Ovens_TEdemand[ID]
                  }
        return bdData    
        
        
''' RAFFRESCAMENTO '''
def loadCooling(istat,db_cool,db_cool_cor):
    
    cool_idata = istat#.iloc[:,[0,282,283,284,286,287,288,88]]
    cool_idata.rename(columns = {"id":"sample",
                        "q_2_1":'TipologiaAbitazione',
                        "q_5_0":"TipoRaffr",
                        "q_5_2":"SePDC",
                        "q_5_3":"TipoPDC",
                        "q_5_4_ric":"ClasseEta",
                        "q_5_7":"UsoClassi",
                        "q_5_8A":"OreMatt",
                        "q_5_8B":"OrePom",
                        "q_5_8C":"OreNot",
                        "q_5_9":"TutteStanze",
                        "q_5_10":"QuanteStanze",
                        "reg":"Region"}, 
             inplace = True)
    '_idata'
    cool_idata['Fascia Clima'] = cool_idata['Region'].apply(fascia_climatica)

    EE_demand_cool = CoolingSystems(cool_idata,db_cool,db_cool_cor)
     
    return EE_demand_cool


class CoolingSystems:
    
    '''
    
    Cooling system from thesis
    Source Assoclima
    
    '''
    def __init__(self,cool_idata,db_cool,db_cool_cor):
        
        self.ID = cool_idata.index
        self.Clima = cool_idata['Fascia Clima']
        
        TipoRaff = cool_idata['TipoRaffr'].replace(to_replace = {1: 'Centr',
                                                                  2: 'multi-split',
                                                                  3: 'mono-split',
                                                                  4: 'portable',
                                                                  5: 'heat pump',
                                                                  6: 'portable hp',
                                                                  0: 'No'})
        TipoPDC = cool_idata['TipoPDC'].replace(to_replace =    {0: 'No',
                                                                  1: 'air',
                                                                  2: 'water',
                                                                  3: 'water',
                                                                  4: 'water',
                                                                  9: 'air'})
        
        ClasseEta = cool_idata['ClasseEta'].replace(to_replace = {0: 2013,
                                                                  1: 2013,
                                                                  2: 2012,
                                                                  3: 2011,
                                                                  4: 2009,
                                                                  5: 2005,
                                                                  6: 2001,
                                                                  7: 2001,
                                                                  9: 2011
                                                                  })
        
        
        EtaEd = cool_idata['q_2_4_ric'].replace(to_replace = { 1: 'after 2000',
                                                                2: '1990-1999',
                                                                3: '1980-1989',
                                                                4: '1970-1979',
                                                                5: '1960-1969',
                                                                6: '1950-1959',
                                                                7: '1900-1950',
                                                                8: 'before 1900',
                                                                9: 'media'})
        
        # ZonaClim = cool_idata['Fascia Clima'].reaplace(to_replace = { 'C': 'after 2000',
        #                                                         'D': '1990-1999',
        #                                                         'E': '1980-1989',
        #                                                         'F': '1970-1979'})
        
        TipoCasa = cool_idata['TipologiaAbitazione'].replace(to_replace = { 1: 'unifamiliare',
                                                                        2: 'plurifamiliare o schiera',
                                                                        3: 'appartamento <10',
                                                                        4: 'appartamento 10-27',
                                                                        5: 'appartamento >28'})
        
        Freq = cool_idata['UsoClassi'].replace(to_replace = { 1: 'always',
                                                                2: 'some days a week',
                                                                3: 'once per week',
                                                                4: 'less than 4 times month',
                                                                5: 'occasionally',
                                                                0: 'occasionally'})
        
        Dict_freq = { 1: 6,
                        2: 4,
                        3: 1,
                        4: 0.75,
                        5: 0.5,
                        0: 0.}

        settimane_estive = { 'C': 138//7,
                            'D': 138//7,
                            'E': 107//7,
                            'F': 62//7}
        
                     
        EE_demand_cool = pd.Series(index = cool_idata.index, dtype='float64')
        
        for idx in self.ID:
            
            if TipoRaff.loc[idx] in ['Centr','multi-split','mono-split','portable','portable hp']:
                TipoPDC.loc[idx] = 'No'
            power = db_cool.loc[TipoRaff.loc[idx],TipoPDC.loc[idx],ClasseEta.loc[idx]]['power'].iloc[0]
                      
            corr_eta = db_cool_cor.loc['building year',EtaEd.loc[idx]]['coefficient']
            corr_zona = db_cool_cor.loc['climate zone',cool_idata.loc[idx]['Fascia Clima']]['coefficient']
            corr_tipo_casa = db_cool_cor.loc['building type',TipoCasa.loc[idx]]['coefficient']
            corr_freq = db_cool_cor.loc['frequency',Freq.loc[idx]]['coefficient']
            
            n_ore = cool_idata.loc[idx]['OreMatt'] + cool_idata.loc[idx]['OrePom'] + cool_idata.loc[idx]['OreNot']
            
            corr_n_ore = 0.
            if n_ore  < 2:
                corr_n_ore = db_cool_cor.loc['hours','0-1']['coefficient']
            elif n_ore  < 4:
                corr_n_ore = db_cool_cor.loc['hours','2-3']['coefficient']
            elif n_ore  < 7:
                corr_n_ore = db_cool_cor.loc['hours','4-6']['coefficient']
            else:
                corr_n_ore = db_cool_cor.loc['hours','>7']['coefficient']
                
            corr_tot = corr_eta*corr_zona*corr_tipo_casa*corr_freq*corr_n_ore
            
            n_settimane = settimane_estive[cool_idata.loc[idx]['Fascia Clima']]
            frequenza = Dict_freq[cool_idata.loc[idx]['UsoClassi']]
            
            EE_demand_cool.loc[idx] = power * n_ore * frequenza * n_settimane * corr_tot
        
        self.EE_demand_cool = EE_demand_cool
        
        cool_info = pd.DataFrame(index = cool_idata.index, columns = pd.MultiIndex.from_arrays([[],[]],names = names ))
        cool_info['Cooling','TipoRaff'] = TipoRaff
        cool_info['Cooling','TipoPDCRaff'] = TipoPDC.replace({'No':''})
        cool_info['Cooling','EtaRaff'] = ClasseEta
        cool_info['Cooling','ClassiUsoRaff'] = Freq
        cool_info['Cooling','OreMattRaff'] =  cool_idata['OreMatt']
        cool_info['Cooling','OrePomRaff'] =  cool_idata['OrePom']
        cool_info['Cooling','OreNotRaff'] =  cool_idata['OreNot']
        cool_info['Cooling','TutteLeStanzeRaff'] =  cool_idata['TutteStanze'].replace({1:'Si',2:''})
        self.cool_info = cool_info
        
''' Piccolissimi (standby e carica batterie) '''
def loadStandby(istat,db_standby):
    
    lights_idata = istat#.iloc[:,[0,282,283,284,286,287,288,88]]

    codice_ID = lights_idata['sample']
    componenti = lights_idata["ComponentiFamiglia"]
    carico_medio = db_standby.iloc[0,0]

    EE_demand_standby = Standby(codice_ID,componenti,carico_medio)
    return EE_demand_standby


class Standby:
    
    '''
    
    carica batterie e altri apparecchi piccolissimi
    
    '''
    def __init__(self,codice_ID,componenti,carico_medio):
      
        self.standby_demand = carico_medio * componenti
        
        
##### Saving Outputs [kWh/year per household]
###      
def storeOutputs(Consumptions):
    df = pd.DataFrame(columns = [
                        'Cooling Systems (kWh/y)',
                        'Standby appliances (kWh/y)',
                        'Lights Demand (kWh/y)' ,
                        'Little Appliances Demand (kWh/y)',
                        'Refrigerators Demand CE (kWh/y)',
                        'Refrigerators Demand MOIRAE (kWh/y)',
                        'Big Appliances Demand MOIRAE (kWh/y)',
                        'Big Appliances Demand CE (kWh/y)',
                        'TVs & PCs Demand (kWh/y)' ,
                        'Electric Cookings Demand (kWh/y)',
                        'Thermal Cookings Demand (kWh/y)',
                        'Electric Cookings Consumption (kWh/y)',
                        'Gas Cookings Consumption (kWh/y)',
                        'GPL Cookings Consumption (kWh/y)',
                        'Biomass Cookings Consumption (kWh/y)',
                        'Electric Ovens Consumption (kWh/y)',
                        'Gas Ovens Consumption (kWh/y)',
                        'GPL Ovens Consumption (kWh/y)',
                        'Biomass Ovens Consumption (kWh/y)'
                        ]
        )
    
    
    
    store_demand_from_cool = pd.DataFrame(Consumptions['CoolingSystems'].EE_demand_cool)
    df['Cooling Systems (kWh/y)'] = Consumptions['CoolingSystems'].EE_demand_cool
    
    store_demand_from_standby = pd.DataFrame(Consumptions['Standby'].standby_demand)
    df['Standby appliances (kWh/y)'] = Consumptions['Standby'].standby_demand
    
    store_demand_from_lights = pd.DataFrame(Consumptions['Lights'].lights_demand)     
    df['Lights Demand (kWh/y)'] = Consumptions['Lights'].lights_demand
    
    store_demand_from_little_appliances = pd.DataFrame(Consumptions['Little_appliances'].little_appliances_demand)   
    df['Little Appliances Demand (kWh/y)'] = Consumptions['Little_appliances'].little_appliances_demand
    
    store_demand_from_refrigerators = pd.DataFrame(Consumptions['Refrigerators CE'].refrigerators_demand)     
    df['Refrigerators Demand CE (kWh/y)'] = Consumptions['Refrigerators CE'].refrigerators_demand
    
    store_demand_from_refrigerators_moirae = pd.DataFrame(Consumptions['Refrigerators MOIRAE'].refrigerators_demand)     
    df['Refrigerators Demand MOIRAE (kWh/y)'] = Consumptions['Refrigerators MOIRAE'].refrigerators_demand
    
    store_demand_from_big_appliances = pd.DataFrame(Consumptions['Big_appliances MOIRAE'].big_appliances_demand)     
    df['Big Appliances Demand MOIRAE (kWh/y)'] = Consumptions['Big_appliances MOIRAE'].big_appliances_demand
    
    store_demand_from_big_appliances_ce = pd.DataFrame(Consumptions['Big_appliances CE'].big_appliances_demand)
    df['Big Appliances Demand CE (kWh/y)'] = Consumptions['Big_appliances CE'].big_appliances_demand
    
    store_demand_from_screens = pd.DataFrame(Consumptions['Screens'].screen_demand)    
    df['TVs & PCs Demand (kWh/y)'] = Consumptions['Screens'].screen_demand
    
    store_demand_from_EEcookers = pd.DataFrame(Consumptions['Cookings'].EEcookings_demand)     
    df['Electric Cookings Demand (kWh/y)'] = Consumptions['Cookings'].EEcookings_demand
    
    store_demand_from_DWH_11300 = pd.DataFrame(Consumptions['DHW'].Qw_UNI11300)   
    df['DHW 11300 (kWh/y)'] = Consumptions['DHW'].Qw_UNI11300
    
    store_demand_from_DWH_9182 = pd.DataFrame(Consumptions['DHW'].Qw_UNI9182)   
    df['DHW 9182 (kWh/y)'] = Consumptions['DHW'].Qw_UNI9182
    
    store_demand_from_TEcookers = pd.DataFrame(Consumptions['Cookings'].TEcookings_demand)     
    df['Thermal Cookings Demand (kWh/y)'] = Consumptions['Cookings'].TEcookings_demand
    
    
    
    store_demand_from_cookers_el = pd.DataFrame(Consumptions['Cookings'].hobs_e['Electric']) 
    df['Electric Cookings Consumption (kWh/y)'] = Consumptions['Cookings'].hobs_e['Electric']
    
    store_demand_from_cookers_gas = pd.DataFrame(Consumptions['Cookings'].hobs_e['NaturalGas'])   
    df['Gas Cookings Consumption (kWh/y)'] = Consumptions['Cookings'].hobs_e['NaturalGas']
    
    store_demand_from_cookers_gpl = pd.DataFrame(Consumptions['Cookings'].hobs_e['LPG'])
    df['GPL Cookings Consumption (kWh/y)'] = Consumptions['Cookings'].hobs_e['LPG']
    
    store_demand_from_cookers_bio = pd.DataFrame(Consumptions['Cookings'].hobs_e['Biomass'])    
    df['Biomass Cookings Consumption (kWh/y)'] = Consumptions['Cookings'].hobs_e['Biomass']
    
    
    store_demand_from_ovens_el = pd.DataFrame(Consumptions['Cookings'].ovens_e['Electric']) 
    df['Electric Ovens Consumption (kWh/y)'] = Consumptions['Cookings'].ovens_e['Electric']
    
    store_demand_from_ovens_gas = pd.DataFrame(Consumptions['Cookings'].ovens_e['NaturalGas'])   
    df['Gas Ovens Consumption (kWh/y)'] = Consumptions['Cookings'].ovens_e['NaturalGas']
    
    store_demand_from_ovens_gpl = pd.DataFrame(Consumptions['Cookings'].ovens_e['LPG'])
    df['GPL Ovens Consumption (kWh/y)'] = Consumptions['Cookings'].ovens_e['LPG']
    
    store_demand_from_ovens_bio = pd.DataFrame(Consumptions['Cookings'].ovens_e['Biomass'])     
    df['Biomass Ovens Consumption (kWh/y)'] = Consumptions['Cookings'].ovens_e['Biomass']  
    
       
    # [kWh/year]
    Storage =(store_demand_from_cool,
              store_demand_from_standby,
              store_demand_from_lights, 
              store_demand_from_little_appliances, 
              store_demand_from_refrigerators,
              store_demand_from_big_appliances_ce,
              store_demand_from_screens,
              store_demand_from_EEcookers,
              store_demand_from_DWH_11300,
              store_demand_from_DWH_9182,
              store_demand_from_TEcookers)
    return Storage, df
