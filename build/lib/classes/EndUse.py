'''IMPORTING MODULES'''

import sys
import pandas as pd
import os
import numpy as np
from funcs.aux_functions import wrn
import random

#%% ---------------------------------------------------------------------------------------------------
#%% Useful functions to create the schedule archetypes

def initializeSchedules(path,timeIndex,ts):
    '''
    Archetype loading in case you use loadSchedComp
    This function takes the path of the excel file constining the schedules and loads it
    Works for the yearly schedule (see file ScheduleComp.xlsx in /Input/
    
    Parameters
    ----------
    path : string
        Path containing the string of the file_schedule.xlsx
    timeIndex : np array of int
        This is the array containing the index of the simulation time steps .
    ts : int
        Number of time steps per hour.

    Returns
    -------
    archetypes: dictionary with archetype_key/Archetype(object) data
    '''
    
    # Check input data type  

    if not isinstance(path, str):
        raise TypeError(f'ERROR input path is not a string: path {path}') 
    if not isinstance(timeIndex, np.ndarray):
        raise TypeError(f'ERROR input timeIndex is not a np.array: timeIndex {timeIndex}') 
    if not isinstance(ts, int):
        raise TypeError(f'ERROR input ts is not an integer: ts {ts}')         
    
    # Control input data quality
    
    if ts > 4:
        wrn(f"WARNING loadSimpleArchetype function, input ts is higher than 4, this means more than 4 time steps per hours were set: ts {ts}")
         
    try:
        standard_sched = pd.read_excel(path,sheet_name="Schedule",header=2,index_col=[0]).set_index(timeIndex)
        standard_user = pd.read_excel(path,sheet_name="Archetype",header=1,index_col=[0])
    except FileNotFoundError:
        raise FileNotFoundError(f'ERROR Failed to open the schedule xlsx file {path}... Insert a proper path')

    # Reset some indexes set the right index
    
    standard_sched = standard_sched.reset_index().drop(columns=["Time"])
    standard_sched.index=timeIndex
    
    # Creation of the archetypes' dictionary
        
    standard_users = dict()
    for i in standard_user.index:
        if i != 'Archetype':
            standard_users[i] = User(i)
            standard_users[i].loadSchedComp(standard_user.loc[i],standard_sched)
            standard_users[i].rescale_df(ts)
            standard_users[i].create_np()
    return standard_users

#%%

def loadSimpleArchetype(path,timeIndex,first_day = 1,ts = 1, PlantDays = [2520,3984,6192,6912]):
    '''
    Archetype loading in case you use loadSchedSemp
    This function takes the path of the excel file constining the schedules and loads it
    Works for the daily schedule (see file ScheduleSemp.xlsx in /Input/
    
    Parameters
    ----------
    path : string
        Path containing the string of the file_schedule.xlsx
    timeIndex : np array of int
        This is the array containing the index of the simulation time steps .
    first_day : int
        First day of the week (1 monday, 2 tuesday, ..... 7 sunday)
    ts : int
        Number of time steps per hour.
    PlantDays : list
        List of integers that sets the 4 time steps of heating and cooling season start and stops
        [last heating timestep,
         first cooling timestep,
         last cooling timstep,
         fist heating time step]

    Returns
    -------
    archetypes: dictionary with archetype_key/Archetype(object) data
    '''
    
    # Check input data type  
    
    if not isinstance(path, str):
        raise TypeError(f'ERROR input path is not a string: path {path}') 
    if not isinstance(timeIndex, np.ndarray):
        raise TypeError(f'ERROR input timeIndex is not a np.array: timeIndex {timeIndex}') 
    if not isinstance(first_day, int):
        raise TypeError(f'ERROR input first_day is not an integer: first_day {first_day}')   
    if not isinstance(ts, int):
        raise TypeError(f'ERROR input ts is not an integer: ts {ts}')
    if not isinstance(PlantDays, list):
        raise TypeError(f'ERROR input PlantDays is not a list: PlantDays {PlantDays}\nThis parameter must be a list of integers (default [2520,3984,6192,6912])')
    if not isinstance(PlantDays[0], int) or not isinstance(PlantDays[1], int) or not isinstance(PlantDays[2], int) or not isinstance(PlantDays[3], int):
        raise TypeError(f'ERROR input PlantDays in not a list of integers: PlantDays {PlantDays}')   
    
    # Control input data quality
    
    if first_day > 7 or first_day < 1:
            wrn(f"WARNING loadSimpleArchetype function, input fisrt_day should be in the range [0,7]: first_day {first_day}")
    if ts > 4:
            wrn(f"WARNING loadSimpleArchetype function, input ts is higher than 4, this means more than 4 time steps per hours were set: ts {ts}")
        
    try:
        ex = pd.ExcelFile(path)
    except FileNotFoundError:
        raise FileNotFoundError(f'ERROR Failed to open the schedule xlsx file {path}... Insert a proper path')
    
    # Creation of the archetypes' dictionary
    
    archetypes = dict()
    last_H_day = PlantDays[0]
    first_C_day = PlantDays[1]
    last_C_day = PlantDays[2]
    first_H_day = PlantDays[3]
    
    # read archetype names from the excel sheet
    archetype_list = pd.read_excel(path,sheet_name='GeneralData',header=[0],index_col=[0], skiprows = 1)
    names = archetype_list.index
    
    for use in names:
        archetypes[use] = Archetype(use)
        year = pd.DataFrame()
        schedule = pd.read_excel(path,sheet_name=use,header=[0,1,2],index_col=[0])
        schedule = schedule.iloc[1:25]*schedule.loc['Valore nominale']
        week = pd.concat([schedule['Weekday']]*5+[schedule['Weekend']]*2)
        
        # The heating cooling availabilty is create considering the PlantDays variable
        
        first_week = week.iloc[(first_day-1)*24 :]
        year = pd.concat([first_week]+[week]*53).iloc[:8760].set_index(timeIndex)
        year.loc[last_H_day:first_C_day,('Plant Availability','[-]')]=year.loc[last_H_day:first_C_day,('Plant Availability','[-]')]*0
        year.loc[first_C_day:last_C_day,('Plant Availability','[-]')]=year.loc[first_C_day:last_C_day,('Plant Availability','[-]')]*-1
        year.loc[last_C_day:first_H_day,('Plant Availability','[-]')]=year.loc[last_C_day:first_H_day,('Plant Availability','[-]')]*0
        archetypes[use].loadSchedSemp(year, archetype_list.loc[use])
        archetypes[use].rescale_df(ts)
        archetypes[use].create_np()

    return archetypes

#%%--------------------------------------------------------------------------------------------------- 
#%% User class

class User:
    '''
    This class manages the end use with its schedules

    init method: sets only the name and creats a Dataframe:
        name: a string with the name
        
    loadSchedComp: loads the complex (annual) method for schedules:
        arch: list containing the name of the schedules of the archetype
        sched: dictionary with all the annual schedules
        
    loadSchedSemp: loads the simple method:
        takes a yearly dataframe with the archetype schedules
        
    rescale_df: rescales the dataframe with respect to the ts parameter (time steps per hour)
    
    create_np: convert every schedule in numpy arrays
        
    Methods:
        init
        loadSchedComp
        loadSchedSemp
        rescale_df
        create_np
    ''' 
    
    periodi_risc = {'A' : (74,335), # 1 dic 15 marzo
                    'B' : (90,335), # 1 dic 31 marzo
                    'C' : (90,319), # 15 nov 31 marzo
                    'D' : (105,319), # 1 Nov 15 Apr
                    'E' : (105,288), # 15 Ott 15 Apr
                    'F' : (120,274)  # No lim, metto ottobre maggio                    
        }
    
    periodi_raff = {'A' : (121,273), # 1 Maggio 30 sett
                    'B' : (121,273), # 1 Maggio 30 sett
                    'C' : (135,273), # 15 Maggio 30 sett
                    'D' : (135,273), # 15 Maggio 30 sett
                    'E' : (152,258), # 1 Giu 15 Sett
                    'F' : (182,243)  # 1 Luglio, 31 Agosto                    
        }
    
    
    def __init__(self,name):
    
        '''
        Initializes the vectors of the AHU
        
        Parameters
            ----------
            name : string
                name of the archetype
                
        Returns
        -------
        None.
        
        '''    
        
        # Check input data type
        
        if not isinstance(name, str):
            raise TypeError(f'ERROR Archetype initialization, name must be a string: name {name}')
        
        # Inizialization
        
        self.name = name
        self.sched_df = pd.DataFrame()
        self.scalar_data = {}
        '''
        Take care of units!!!
        All formula referes to the complex schedule file units, 
        Simple schedule file has different units which are converted in the loadSchedSemp method
        (vapuor flow rates)
        '''
    
    def loadSchedComp(self,arch,sched):
    
        '''
        Used for ScheduleComp.xlsx Excel file
        
        Parameters
            ----------
            arch : pandas dataframe
                This must include all the schedules' keys
            
            sched : pandas dataframe
                This dataframe includes all the yearly schedules
                
        Returns
        -------
        None.
        
        '''
            
        # Each schedule is set to a different attribute of the class
        
        try:
            self.sched_df['appliances'] = sched[arch['Appliances']]
            self.sched_df['lighting'] = sched[arch['Lighting']]
            self.sched_df['people'] = sched[arch['People (Sensible)']]
            self.sched_df['vapour'] = sched[arch['Vapour']]
            self.sched_df['heatingTSP'] = sched[arch['HeatTSP']]
            self.sched_df['coolingTSP'] = sched[arch['CoolTSP']]
            self.sched_df['HeatingRHSP'] = sched[arch['HeatRHSP']]
            self.sched_df['CoolingRHSP'] = sched[arch['CoolRHSP']]
            self.sched_df['ventFlowRate'] = sched[arch['VentFlowRate']]
            self.sched_df['infFlowRate'] = sched[arch['InfFlowRate']]
            self.sched_df['plantOnOffSens'] = sched[arch['PlantONOFFSens']]
            self.sched_df['plantOnOffLat'] = sched[arch['PlantONOFFLat']]
            self.sched_df['AHUOnOff'] = sched[arch['AHUONOFF']]
            self.sched_df['AHUHUM'] = sched[arch['AHUHUM']]
            self.sched_df['AHUTSupp'] = sched[arch['AHUTSupp']]
            self.sched_df['AHUxSupp'] = sched[arch['AHUxSupp']]
            
            self.scalar_data['conFrac'] = float(arch['ConvFrac'])
            self.scalar_data['AHUHUM'] = bool(arch['AHUHum'])
            self.scalar_data['sensRec'] = float(arch['SensRec'])
            self.scalar_data['latRec'] = float(arch['LatRec'])
            self.scalar_data['outdoorAirRatio'] = float(arch['OutAirRatio'])
        except KeyError:
            raise KeyError(f'ERROR Archetype object {self.name}: can not find all schedules')
        
    def loadSchedSemp(self,year_df,supplementary_data):
        '''
        Used for ScheduleSemp.xlsx Excel file
        
        Parameters
            ----------
            arch : pandas dataframe
                This includes the yearly schedule of a single archetype
            supplementary_data : pd.series
                This series includes some additional data about the archetype (Sensible and 
                                                                               Latent AHU recovery, 
                                                                               Convective fraction of internal gains)
            
        Returns
        -------
        None.
        
        '''    
    
        # Each schedule is set to a different attribute of the class
    
        try:
            self.sched_df['appliances'] = year_df['Appliances','[W/m²]']
            self.sched_df['lighting'] = year_df['Lighting','[W/m²]']
            self.sched_df['people'] = year_df['Occupancy (Sensible)','[W/m²]']
            self.sched_df['vapour'] = year_df['Vapour FlowRate','[g/(m² s)]']/1000 #--> kg conversion (a)
            self.sched_df['heatingTSP'] = year_df['HeatSP','[°C]']
            self.sched_df['coolingTSP'] = year_df['CoolSP','[°C]']
            self.sched_df['HeatingRHSP'] = year_df['HumSP','[%]']/100
            self.sched_df['CoolingRHSP'] = year_df['DehumSP','[%]']/100
            self.sched_df['ventFlowRate'] = year_df['Ventilation FlowRate','[m³/(s m²)]']
            self.sched_df['infFlowRate'] = year_df['Infiltration FlowRate','[Vol/h]']
            self.sched_df['plantOnOffSens'] = year_df['Plant Availability','[-]']
            self.sched_df['plantOnOffLat'] = year_df['Plant Availability','[-]']
            self.sched_df['AHUOnOff'] = year_df['Plant Availability','[-]']
        except KeyError:
            raise KeyError(f'ERROR Archetype object {self.name}: can not find all schedules')
        
        # Following parameters are not set in the Excel file
        
        self.sched_df['AHUTSupp'] = pd.Series([22.]*8760,index=self.sched_df['appliances'].index)
        self.sched_df['AHUxSupp'] = pd.Series([0.005]*8760,index=self.sched_df['appliances'].index)
        self.sched_df['AHUxSupp'].iloc[3984:6192] = 0.007
        
        try:
            self.scalar_data['conFrac'] = float(supplementary_data.loc['ConvFrac'])
            self.scalar_data['AHUHUM'] = bool(supplementary_data.loc['AHUHum'])
            self.scalar_data['sensRec'] = float(supplementary_data.loc['SensRec'])
            self.scalar_data['latRec']= float(supplementary_data.loc['LatRec'])
            self.scalar_data['outdoorAirRatio'] = float(supplementary_data.loc['OutAirRatio'])
        except KeyError:
            raise KeyError(f"ERROR Loading end use {self.name}. GeneralData does not have the correct columns names: ConvFrac, AHUHum, SensRec, LatRec, OutAirRatio")
        except ValueError:
            raise ValueError(f"""ERROR 
                             Loading end use {self.name}. GeneralData
                             I'm not able to parse the General data. 
                                 ConvFrac should be a float {supplementary_data['ConvFrac']}
                                 AHUHum should be a boolean {supplementary_data['AHUHum']}
                                 SensRec should be a float {supplementary_data['SensRec']}
                                 LatRec  should be a float {supplementary_data['LatRec']}
                                 OutAirRatio   should be a float {supplementary_data['OutAirRatio']}
                             """)
        
        # Check the quality of input data
        if not 0. <= self.scalar_data['conFrac'] <= 1.:
            wrn(f"WARNING Loading end use {self.name}. Convective fraction of the heat gain outside boundary condition [0-1]: ConvFrac {self.scalar_data['conFrac']}")
        if not 0. <= self.scalar_data['sensRec'] <= 1.:
            wrn(f"WARNING Loading end use {self.name}. Sensible recovery of the AHU outside boundary condition [0-1]: sensRec {self.scalar_data['sensRec']}")
        if not 0. <= self.scalar_data['latRec'] <= 1.:
            wrn(f"WARNING Loading end use {self.name}. Latent recovery of the AHU outside boundary condition [0-1]: sensRec {self.scalar_data['latRec']}")
        if not 0. <= self.scalar_data['outdoorAirRatio'] <= 1.:
            wrn(f"WARNING Loading end use {self.name}. Outdoor air ratio of the AHU outside boundary condition [0-1]: outdoorAirRatio {self.scalar_data['outdoorAirRatio']}")
        
    def rescale_df(self,ts):
        '''
        rescale the archetype dataframe with respect to the number of time steps per hour
        
        Parameters
            ----------
            ts : int
                Number of time steps per hour
                
        Returns
        -------
        None.        
        '''   
        
        # Check input data type
        
        if not isinstance(ts, int):
            raise TypeError(f'ERROR input ts is not an integer: ts {ts}')         
    
        # Rescale 
        
        m = str(60/ts) + 'min'
        time = pd.date_range('2019-01-01', periods=8760, freq='1h')
        self.sched_df.set_index(time, inplace=True) 
        self.sched_df = self.sched_df.resample(m).pad()                               # Steps interpolation Resampling
        #Boundary = Boundary_0.resample(str(ts)+'S').interpolate(method='linear')        # Linear interpolation Resampling 
        # There are several upsample methods: pad(), bfill(), mean(), interpolate(), apply custum function...
        # For info look:
        #   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        #   https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
        
    def create_np(self):
    
        '''
        Creates the np.array attributes from the self.sched_df dataframe
        
        Parameters
            ----------
                
        Returns
        -------
        None.        
        '''      
    
        self.appliances = self.sched_df['appliances'].to_numpy(dtype = np.float_)
        self.lighting = self.sched_df['lighting'].to_numpy(dtype = np.float_)
        self.people = self.sched_df['people'].to_numpy(dtype = np.float_)
        self.vapour = self.sched_df['vapour'].to_numpy(dtype = np.float_)
        self.heatingTSP = self.sched_df['heatingTSP'].to_numpy(dtype = np.float_)
        self.coolingTSP = self.sched_df['coolingTSP'].to_numpy(dtype = np.float_)
        self.HeatingRHSP = self.sched_df['HeatingRHSP'].to_numpy(dtype = np.float_)
        self.CoolingRHSP = self.sched_df['CoolingRHSP'].to_numpy(dtype = np.float_)
        self.ventFlowRate = self.sched_df['ventFlowRate'].to_numpy(dtype = np.float_)
        self.infFlowRate = self.sched_df['infFlowRate'].to_numpy(dtype = np.float_)
        self.plantOnOffSens = self.sched_df['plantOnOffSens'].to_numpy(dtype = np.int_)
        self.plantOnOffLat = self.sched_df['plantOnOffLat'].to_numpy(dtype = np.int_)
        self.AHUOnOff = self.sched_df['AHUOnOff'].to_numpy(dtype = np.int_)
        #self.AHUHUM = self.sched_df['AHUHUM'].to_numpy(dtype = np.bool_)
        self.AHUTSupp = self.sched_df['AHUTSupp'].to_numpy(dtype = np.float_)
        self.AHUxSupp = self.sched_df['AHUxSupp'].to_numpy(dtype = np.float_)
        #self.conFrac = self.sched_df['conFrac'].to_numpy(dtype = np.float_)
        #self.AHUHumidistat = self.sched_df['AHUHum']
        #self.sensRec = self.sched_df['sensRec'].to_numpy(dtype = np.float_)
        #self.latRec = self.sched_df['latRec'].to_numpy(dtype = np.float_)
        #self.outdoorAirRatio = self.sched_df['outdoorAirRatio'].to_numpy(dtype = np.float_)

    
    def substitute_istat_sched(self,istatdata, el_cons, primo_giorno = 7):  #retrofits
        
        # Variabili utili per impianti
        # setpoint
        # umidità
        # ventilazione
        # infiltrazione
        # plant on_off
        
        # parte 1
        
        # anno = istatdata['anno']
        # rip_geo = istatdata['anno']	# Ripartizione geografica
        # regione	 = istatdata['reg']
        # Fascia_clima = istatdata['Fascia_clima']
        # n_px = istatdata['q_1_1_sq1'] #umero persone che vivono nell'abitazione
        
        # parte 3
        
        # ore_acc_matt = istatdata['q_3_11A'] # 'n_h_acc_matt',   #Acc.Impianto: 5-13am
        # ore_acc_pome = istatdata['q_3_11B'] # 'n_h_acc_pom',   #Acc.Impianto: 13-21am
        # ore_acc_nott = istatdata['q_3_11C'] # 'n_h_acc_not',   #Acc.Impianto: 21-5am
        # ore_acc_tot = istatdata['q_3_11tot'] # 'n_h_acc_tot', #Acc.ImpiantoTot
        # regolazione = istatdata['q_3_12'] # termostato 1 si  2 no
        # regolazione_termostato = istatdata['q_3_13_1'] # termostato 1 si  0 no
        # regolazione_cronotermostato = istatdata['q_3_13_2'] # termostato 1 si  0 no
        # regolazione_valvole = istatdata['q_3_13_3'] # termostato 1 si  0 no
        # isolamento = istatdata['q_3_17'] # Isolamento termico della casa: se 4 spifferi
        
        # Retrofit_regolazione = istatdata['q_3_19_6']	# 1 Si 0 No
        
        # parte 5 condizionamento
        
        # giorni_raff = istatdata['q_5_7']	#Frequenza di utilizzo dell'impianto di co
        # ore_raff_matt = istatdata['q_5_8A']	#Numero di ore di accensione per fascia or
        # ore_raff_pom = istatdata['q_5_8B']	#Numero di ore di accensione per fascia or
        # ore_raff_nott = istatdata['q_5_8C']	#Numero di ore di accensione per fascia or
        # ore_raff_nott = istatdata['q_5_8tot']	#Numero di ore di accensione per fascia or
        # stanze_raff = istatdata['q_5_9']	#Sistema di condizionamento in tutte/alcun 1 tutte 2 solo alcune
        # n_stanze_raff = istatdata['q_5_10']	#Numero di stanze con il condizionatore ac

        # parte 7 luci 
        
        # ci sarebbe il numero di ore di accensione delle luci, al momento non serve
        
        # parte 8 elettrodomenstici
        
        # numero di lavaggi lavatrice e fascia oraria
        # stessa cosa asciugatrice
        # lavastoviglie
        # e altri
        
        self.substitute_istat_coolingTSP(istatdata)
        self.substitute_istat_heatingTSP(istatdata)
        # self.substitute_istat_inf_vent(istatdata, retrofits)
        # il prossimo va fatto dopo quelli di h e c setpoint
        self.substitute_istat_plant_availability(istatdata)
        self.substitute_istat_appliance(el_cons)
        self.substitute_istat_occupancy(istatdata,primo_giorno = primo_giorno)        

    def substitute_istat_occupancy(self,istatdata,primo_giorno = 7):
        # schedule norma
        # Primo giorno intero da 1 a 7
        Occ_w = np.array([1, 1,1,1,1,1,0.5,0.5,0.5,0.1,0.1,0.1, 0.1, 
                0.2,0.2,0.2,0.5,  0.5,  0.5,  0.8, 0.8, 0.8,1, 1])
        Occ_we = np.array([1, 1,1,1,1,1,0.8,0.8,0.8,0.8,0.8,0.8, 0.8, 
                0.8,0.8,0.8,0.8,  0.8,  0.8,  0.8, 0.8, 0.8,1, 1])
        
        week = np.hstack(   [np.hstack([Occ_w]*5), 
                            np.hstack([Occ_we]*2)])
        first_week = week[(primo_giorno-1)*24:]
        year = np.hstack(   [first_week, 
                            np.hstack([week]*53)])[:8760]
        
        n_px = istatdata['q_1_1_sq1'] #umero persone che vivono nell'abitazione
        
        self.n_px = n_px
        self.people = year #* n_px #* 65./0.5462  # 65 W/px di carico totale (entra questo nelle formule poi)
        # va in kg/s
        self.vapour = year #* n_px #* 90/3600./1000  # 90g/h di carico latente (entra questo nelle formule poi)    
    
    def substitute_istat_appliance(self,el_cons):
        
        lights = el_cons['Lights Demand (kWh/y)']  # Tutto carico termico
        little_app = el_cons['Little Appliances Demand (kWh/y)'] # Tutto carico termico
        ref = el_cons['Refrigerators Demand MOIRAE (kWh/y)'] # Tutto carico termico
        big_app = el_cons['Big Appliances Demand MOIRAE (kWh/y)'] * 0.1 # Ipotesi 10% carico termico
        tv_pc = el_cons['TVs & PCs Demand (kWh/y)'] # Tutto carico termico
        el_cooking = el_cons['Electric Cookings Demand (kWh/y)'] * 0.5 # 50 % 
        
        Totale_app = (little_app + ref +
                big_app + tv_pc + el_cooking) * 1000 # Wh/y
        Totale_lights = (lights) * 1000 # Wh/y
        
        # Schedule EN 16798 residenziale
        # Occupancy residenziale weekend e weekday
        # Appliances = media Appliance e lights
                
        App_w = np.array([0.25,0.25,0.25,0.25,0.25,0.25, 0.325,0.425,0.425, 0.325,0.275,0.325,
                0.325,0.325,0.325, 0.275,0.35, 0.45,0.45, 0.5, 0.5,0.5,  0.375,0.375])
        # App_we = App_w

        yearly = np.hstack([App_w]*365)    
        self.appliances = yearly/yearly.sum() * Totale_app # schedule in W
        self.lighting = yearly/yearly.sum() * Totale_lights # schedule in W
        
    def substitute_istat_coolingTSP(self,istatdata):
        giorni_raff = istatdata['q_5_7']	#Frequenza di utilizzo dell'impianto di co
        ore_raff_matt = istatdata['q_5_8A']	#Numero di ore di accensione per fascia or
        ore_raff_pom = istatdata['q_5_8B']	#Numero di ore di accensione per fascia or
        ore_raff_nott = istatdata['q_5_8C']	#Numero di ore di accensione per fascia or
        stanze_raff = istatdata['q_5_9']	#Sistema di condizionamento in tutte/alcun 1 tutte 2 solo alcune
        # n_stanze_raff = istatdata['q_5_10']	#Numero di stanze con il condizionatore ac
        

        prof = self.create_prof(ore_raff_matt,ore_raff_pom,ore_raff_nott,istatdata['id']) # vettore di 0 e 1
        if  stanze_raff == 1:
            #se ho in tutte le stanze il clima uso 25 °C
            T_c = 40 - prof*15.
        else:
            # altrimenti 26 °C
            T_c = 40 - prof*14.       
        
        T_c_off = np.ones(24)*40.
        
        # np.random.seed(istatdata['id'])
        base = np.random.randint(1,8, size=7)
        if giorni_raff == 1:
            n_giorni = 6          
        elif giorni_raff == 2:
            n_giorni = 3
        elif giorni_raff == 3:    
            n_giorni = 1
        elif giorni_raff > 3 or giorni_raff == 0:    
            n_giorni = 0
            
        base = 1-np.greater(base,n_giorni)
        weekly = np.array([])
        for d in base:
            if d == 0:
                weekly = np.hstack([weekly,T_c_off])
            else:
                weekly = np.hstack([weekly,T_c])
        
        
        # metto da giugno ad agosto
        # season = np.hstack([weekly]*14)[0:(30+31+31)*24]
        # tot = np.hstack([40.*np.ones((31+28+31+30+31)*24),
        #                   season,
        #                   40.*np.ones((30+31+30+31)*24)])
        weekly_off = 40*np.ones(24*7)
        
        giorni_lim = self.periodi_raff[istatdata['Fascia_clima']]
        
        
        inizio_raff = int(giorni_lim[0]/7)
        fine_raff = int(giorni_lim[1]/7)
        
       
        tot = np.hstack([
            np.hstack([weekly_off]*inizio_raff), 
            np.hstack([weekly]*(fine_raff - inizio_raff)), 
            np.hstack([weekly_off] * (53 - fine_raff))
            ])
        
        
        self.coolingTSP = tot[0:8760]
        self.CoolingRHSP = np.greater(tot,30)*.45 + .55        
        
    def substitute_istat_heatingTSP(self,istatdata):
        ore_acc_matt = istatdata['q_3_11A'] # 'n_h_acc_matt',   #Acc.Impianto: 5-13am
        ore_acc_pome = istatdata['q_3_11B'] # 'n_h_acc_pom',   #Acc.Impianto: 13-21am
        ore_acc_nott = istatdata['q_3_11C'] # 'n_h_acc_not',   #Acc.Impianto: 21-5am
        prof = self.create_prof(ore_acc_matt,ore_acc_pome,ore_acc_nott,istatdata['id']) # vettore di 0 e 1
        #se ha un cronotermostato (13-2) o valvole termostatiche (13-3) o 
        if  istatdata['q_3_13_2'] == 1 or istatdata['q_3_13_3'] == 1 or istatdata['q_3_19_6'] == 1:
            
            # con setback
            T_h = prof*3 + 18.
        else:
            # altrimenti senza setback
            T_h = prof*11 + 10.
        
        # T_h += 1    # commented
        
        if istatdata['q_3_0'] in [1,2]:
            # presenza di stufe a legna (32c) o caminetti (32d)
            if istatdata['q_2_32C'] == 1 or istatdata['q_2_32D'] == 1:
                T_h += 2
                
        # print(T_h)
        
        giorni_lim = self.periodi_risc[istatdata['Fascia_clima']]
        off = np.zeros(24)
       
        tot = np.hstack([
            np.hstack([T_h]*giorni_lim[0]), 
            np.hstack([off]*(giorni_lim[1] - giorni_lim[0])), 
            np.hstack([T_h]*(365 - giorni_lim[1]))
            ])
        
        self.heatingTSP = tot
        self.HeatingRHSP = np.zeros(8760, dtype = float)

    def substitute_istat_inf_vent(self,istatdata, retrofits):
        self.ventFlowRate = np.zeros(8760, dtype = float)
        if istatdata['q_3_17'] == 4:
            self.infFlowRate = np.ones(8760) * 0.4
        else:
            self.infFlowRate = np.ones(8760) * 0.3
            
        if retrofits['Wins'] or retrofits['Walls']:
            print('Infiltration reduction')
            self.infFlowRate = np.ones(8760) * 0.3
            
    def substitute_istat_plant_availability(self,istatdata):
        self.plantOnOffSens = np.zeros(8760, dtype = int) + 1*(np.greater(self.heatingTSP,5) - 1*(1 - np.greater(self.coolingTSP,30)))
        self.plantOnOffLat = np.zeros(8760, dtype = int) - 1*(1 - np.greater(self.coolingTSP,30))
            
    def create_random_prof(self,n,seed):
        n = int(n)
        if n > 8:
            n = 8
        random.seed(int(seed))
        r = random.randint(0,8-n)
        a = np.concatenate([np.array([0.]*r),np.array([1.]*n),np.array([0.]*(8-r-n))])
        return a 
    
    def create_prof(self,h_mt,h_pm,h_nt,seed):
        mt = self.create_random_prof(h_mt,seed)
        pm = self.create_random_prof(h_pm,seed)
        nt = self.create_random_prof(h_nt,seed)
        return np.concatenate([nt[-5:],mt,pm,nt[:3]])


#%%--------------------------------------------------------------------------------------------------- 
#%%
def CalcSpesa(sdata, i):
    # ispesa =  sdata.loc[:,["spesa_legna","spesa_pellet",
    #                        "spesa_elettrica","spesa_metano",
    #                        "spesa_gasolio","spesa_gpl"]]
    sp_legna = sdata["q_6_9_ric"] # spesa legna
    sp_pellet = sdata["q_6_15_ric"] # spesa pellet
    sp_elettrica = sdata["q_9_1_ric"] # spesa elettrica
    sp_metano = sdata["q_9_4_ric"] # spesa metano
    sp_gasolio = sdata["q_9_5_ric"] # spesa gasolio
    sp_gpl = sdata["q_9_6_ric"] # spesa gpl

    '''spesa annua energia termica ed elettrica'''
    ''' (€/anno) '''
    
    sp_legna_i = sp_legna[i]
    sp_pellet_i = sp_pellet[i]
    sp_elettrica_i = sp_elettrica[i]
    sp_metano_i = sp_metano[i]
    sp_gasolio_i = sp_gasolio[i]
    sp_gpl_i = sp_gpl[i]

    
    '''Legname'''
    if   sp_legna_i==1: spesa_legna_fam = 100;         # [€/anno]
    elif sp_legna_i==2: spesa_legna_fam = 250;         # [€/anno]
    elif sp_legna_i==3: spesa_legna_fam = 350;         # [€/anno]
    elif sp_legna_i==4: spesa_legna_fam = 450;         # [€/anno]
    elif sp_legna_i==5: spesa_legna_fam = 550;         # [€/anno]
    elif sp_legna_i==6: spesa_legna_fam = 650;         # [€/anno]
    elif sp_legna_i==7: spesa_legna_fam = 800;         # [€/anno]
    else: spesa_legna_fam = 0;                         # [€/anno]
                 
    
    '''Pellet'''
    if   sp_pellet_i==1: spesa_pellet_fam = 100;       # [€/anno]
    elif sp_pellet_i==2: spesa_pellet_fam = 250;       # [€/anno]
    elif sp_pellet_i==3: spesa_pellet_fam = 350;       # [€/anno]
    elif sp_pellet_i==4: spesa_pellet_fam = 450;       # [€/anno]
    elif sp_pellet_i==5: spesa_pellet_fam = 550;       # [€/anno]
    elif sp_pellet_i==6: spesa_pellet_fam = 650;       # [€/anno]
    elif sp_pellet_i==7: spesa_pellet_fam = 800;       # [€/anno]
    else: spesa_pellet_fam = 0;                        # [€/anno]
                                                                                   

    '''Elettrica'''
    if   sp_elettrica_i==1: spesa_elettrica_fam = 200;       # [€/anno]
    elif sp_elettrica_i==2: spesa_elettrica_fam = 350;       # [€/anno]
    elif sp_elettrica_i==3: spesa_elettrica_fam = 450;       # [€/anno]
    elif sp_elettrica_i==4: spesa_elettrica_fam = 550;       # [€/anno]
    elif sp_elettrica_i==5: spesa_elettrica_fam = 650;       # [€/anno]
    elif sp_elettrica_i==6: spesa_elettrica_fam = 750;       # [€/anno]
    elif sp_elettrica_i==7: spesa_elettrica_fam = 850;       # [€/anno]
    elif sp_elettrica_i==8: spesa_elettrica_fam = 950;       # [€/anno]
    elif sp_elettrica_i==9: spesa_elettrica_fam = 1100;      # [€/anno]
    else: spesa_elettrica_fam = 0;                           # [€/anno]
    
    '''Metano'''
    if   sp_metano_i==1: spesa_metano_fam = 200;       # [€/anno]
    elif sp_metano_i==2: spesa_metano_fam = 350;       # [€/anno]
    elif sp_metano_i==3: spesa_metano_fam = 450;       # [€/anno]
    elif sp_metano_i==4: spesa_metano_fam = 550;       # [€/anno]
    elif sp_metano_i==5: spesa_metano_fam = 650;       # [€/anno]
    elif sp_metano_i==6: spesa_metano_fam = 750;       # [€/anno]
    elif sp_metano_i==7: spesa_metano_fam = 850;       # [€/anno]
    elif sp_metano_i==8: spesa_metano_fam = 950;       # [€/anno]
    elif sp_metano_i==9: spesa_metano_fam = 1100;      # [€/anno]
    elif sp_metano_i==10: spesa_metano_fam = 1300;     # [€/anno]
    elif sp_metano_i==11: spesa_metano_fam = 1500;     # [€/anno]
    elif sp_metano_i==12: spesa_metano_fam = 1700;     # [€/anno]
    else: spesa_metano_fam = 0;                        # [€/anno]
    
    '''Gasolio'''
    if   sp_gasolio_i==1: spesa_gasolio_fam = 500;       # [€/anno]
    elif sp_gasolio_i==2: spesa_gasolio_fam = 700;       # [€/anno]
    elif sp_gasolio_i==3: spesa_gasolio_fam = 900;       # [€/anno]
    elif sp_gasolio_i==4: spesa_gasolio_fam = 1100;      # [€/anno]
    elif sp_gasolio_i==5: spesa_gasolio_fam = 1300;      # [€/anno]
    elif sp_gasolio_i==6: spesa_gasolio_fam = 1500;      # [€/anno]
    elif sp_gasolio_i==7: spesa_gasolio_fam = 1700;      # [€/anno]
    elif sp_gasolio_i==8: spesa_gasolio_fam = 1900;      # [€/anno]
    elif sp_gasolio_i==9: spesa_gasolio_fam = 2100;      # [€/anno]
    elif sp_gasolio_i==10: spesa_gasolio_fam = 2300;     # [€/anno]
    else: spesa_gasolio_fam = 0;                         # [€/anno]
    
    '''GPL'''
    if   sp_gpl_i==1: spesa_gpl_fam = 50;        # [€/anno]
    elif sp_gpl_i==2: spesa_gpl_fam = 150;       # [€/anno]
    elif sp_gpl_i==3: spesa_gpl_fam = 250;       # [€/anno]
    elif sp_gpl_i==4: spesa_gpl_fam = 350;       # [€/anno]
    elif sp_gpl_i==5: spesa_gpl_fam = 500;       # [€/anno]
    elif sp_gpl_i==6: spesa_gpl_fam = 700;       # [€/anno]
    elif sp_gpl_i==7: spesa_gpl_fam = 900;       # [€/anno]
    else: spesa_gpl_fam = 0;                     # [€/anno]
    
    ''' SPESA ANNUA COMPLESSIVA (€/anno) '''
    spesa_et = spesa_legna_fam + spesa_pellet_fam + spesa_metano_fam + spesa_gasolio_fam + spesa_gpl_fam
    spesa_ee = spesa_elettrica_fam
    return spesa_et, spesa_ee   
    

def load_appliances_schedules(resources_path, primo_giorno = 7):
    schedules_std = pd.read_excel(os.path.join(resources_path, 'ElettrodomesticiSchedule', 'ScheduleApp16798.xlsx'), index_col = 0, header = [0,1])
    week_day = schedules_std['Weekday'][['Appliances','Lights']].values
    week_end = schedules_std['Weekday'][['Appliances','Lights']].values
    
    week = np.vstack(   [np.vstack([week_day]*5), 
                        np.vstack([week_end]*2)])
    first_week = week[(primo_giorno-1)*24:,:]
    year = np.vstack(   [first_week, 
                        np.vstack([week]*53)])[:8760]
    appliances = year[:,0]
    lights = year[:,1]
    return appliances, lights
    
    
#%%
'''
TEST METHOD
'''

if __name__=='__main__':
    from funcs.auxiliary_functions import fascia_climatica
    import matplotlib.pyplot as plt
    schedulepath = os.path.join('..','Input','ScheduleComp.xlsx')
    time=np.arange(8760)
    archetypes = loadArchetype(schedulepath,time,1)
    A = archetypes['Residenziale']
    istat = pd.read_csv(os.path.join('..','istat_microdata_csv.csv'), delimiter=';')
    istat['Fascia_clima'] = istat['reg'].apply(fascia_climatica)
    b = []
    el_cons = pd.read_csv('..\consumption.csv', header = 0, index_col = 0)
    for i in range(10):
        r = random.randint(0,20000)
        
        A.substitute_istat_sched(istat.iloc[r])
        A.substitute_istat_occupancy(istat.iloc[r], primo_giorno=1)
        A.substitute_istat_appliance(el_cons.iloc[r])
