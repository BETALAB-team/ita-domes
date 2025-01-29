# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:45:09 2024

@author: vivijac14771
"""

import pandas as pd
import os
import copy
import pickle
import numpy as np
from funcs.aux_functions import regioni, scegli_provincia_distr, fascia_climatica #simplify_geom, 
from classes.Appliances import *
from classes.DHW import loadDHW2
from classes.Archetype import Archetypes
# from classes.EndUse import User
from classes.Impianti11300_v2 import crea_impianti
from classes.Results import Process #, Visual
# from classes.Building import BuildingModel
# from classes.thermalZone import Building
from progressbar import progressbar as pg
import time
from concurrent.futures import ProcessPoolExecutor

#########################################################
# Config loading
# Loads a global config object
from eureca_building.config import load_config
config_dict = {
  "DEFAULT": {},
  "model": {
    "name": "model"
  },
  "simulation settings": {
    "time steps per hour": "1",
    "simulation reference year" : "2023",
    "start date": "01-01 00:00",
    "final date": "12-31 23:00",
    "heating season start": "10-01 23:00",
    "heating season end": "05-31 23:00",
    "cooling season start": "06-01 23:00",
    "cooling season end": "09-30 23:00"
  },
  "solar radiation settings": {
    "do solar radiation calculation": "True",
    "height subdivisions": "4",
    "azimuth subdivisions": "8",
    "urban shading tolerances": "80.,100.,80."
  }
}
load_config(config_dict)

# from eureca_building.config import CONFIG

#########################################################

# from eureca_building.weather import WeatherFile
from eureca_building.window import SimpleWindow
from eureca_building.surface import Surface, SurfaceInternalMass
from eureca_building.internal_load import People, ElectricLoad  #Lights, 
from eureca_building.ventilation import Infiltration, MechanicalVentilation
from eureca_building.thermal_zone import ThermalZone
from eureca_building.air_handling_unit import AirHandlingUnit
from eureca_building.schedule import Schedule
from eureca_building.domestic_hot_water import DomesticHotWater
# from eureca_building.construction_dataset import ConstructionDataset
from eureca_building.construction import Construction
from eureca_building.setpoints import SetpointDualBand
from eureca_building.building import Building
# from eureca_building.fluids_properties import air_properties
from classes.HVAC_Systems import _Heating_GRINS, _Cooling_GRINS

# from building_energy_sim.datasets import av_electric_cons_per_country_sr, loads_df, sp_df



def process_building_data(istat_data, envelopes_data, consumption_appliances, user_data):
    
    tic = time.process_time()
    
    print("\nPreparing input data for building simulations..")
     
    istat = istat_data['istat']
    istat['rand'] = np.random.random(size=len(istat.index))
    istat['Fascia_clima'] = istat['reg'].apply(fascia_climatica)
       
    info_building = ['Geometry',
                     'Construction',
                     # 'Building_info',
                     'Surfaces',
                     'Use_info',
                     'Building_services',
                     'Location',
                     'General_info'
                     ]
    
    ######Geometry

    # building_data = pd.DataFrame(index = istat.index, columns = info_building, dtype = str)  # p 
    building_data = dict.fromkeys(list(istat.index))
    
    for i in istat.index:
        
        building_data[i] = dict.fromkeys(info_building)
        
        # Building archetype class based on ISTAT survey
        archetype = Archetypes(i,istat) # geometria, condizioni al contorno
        envelope  = envelopes_data[i]   # stratigrafie
        user      = user_data[i]        # utente
        # dhw       = consumption_appliances['DHW_info'].loc[i]
        
        ######################### GENERAL INFO ###############################
        # initialize general info
        building_data[i]['General_info'] = dict.fromkeys(['Abitanti','CoeffRiportoGlobale','Regione',
                                                          'type','num_dwellings','age','Piano','Orientazione'])
        # get info from istat dataset
        building_data[i]['General_info']['Abitanti']            = istat.loc[i]['q_1_1_sq1']
        building_data[i]['General_info']['CoeffRiportoGlobale'] = istat.loc[i]['coef_red']   
        
        ########################### BUILDING  ############################
        building_data[i]['General_info']['type']          = archetype.building_typology_class
        building_data[i]['General_info']['num_dwellings'] = 1
        building_data[i]['General_info']['age']           = archetype.building_age_class
        building_data[i]['General_info']['Piano']         = archetype.apartment_floor_typology
        building_data[i]['General_info']['Orientazione']  = archetype.orientation_txt
        
        
        ############################# GEOMETRY ##############################
        # initialize geometry
        building_data[i]['Geometry'] = dict.fromkeys(['volume','num_floors','footprint_area',
                                                      'floor_height','floor_area'])
        # get info from istat dataset
        building_data[i]['Geometry']['volume']         = archetype.volume
        building_data[i]['Geometry']['num_floors']     = archetype.nFloors
        building_data[i]['Geometry']['footprint_area'] = archetype.floorArea
        building_data[i]['Geometry']['floor_height']   = archetype.dwelling_height
        building_data[i]['Geometry']['floor_area']     = archetype.floorArea
        
        ############################# SURFACES ###############################
        building_data[i]['Surfaces']  = archetype.surfaces   
                
        
        ############################ CONSTRUCTION ############################
        # initialize construction
        building_data[i]['Construction'] = dict.fromkeys(['U_walls','U_roof','U_floor',
                                                          'U_glass','g_glass',
                                                          'ACR_infiltration'])
        # get info from istat dataset
        building_data[i]['Construction']['U_walls']          = envelope.ExtWall.U
        building_data[i]['Construction']['U_roof']           = envelope.Roof.U
        building_data[i]['Construction']['U_floor']          = envelope.GroundFloor.U
        # building_data[i]['Construction']['U_thermalbridge']  = 0.1
        
        # building_data[i]['Construction']['U_glass']          = envelope.Window.U  # questa è già la U_window?
        # building_data[i]['Construction']['U_frame']          = envelope.Window.U
        # building_data[i]['Construction']['g_glass']          = envelope.Window.SHGC
        # building_data[i]['Construction']['f_window']         = envelope.Window.U

        building_data[i]['Construction']['U_win']          = envelope.Window.U  # questa è già la U_window?
        building_data[i]['Construction']['g_win']          = envelope.Window.SHGC
        building_data[i]['Construction']['frame_factor']   = envelope.Window.F_f
        
        building_data[i]['Construction']['ACR_infiltration'] = 0.3

        # building_data[i]['Construction']['C_envelope']       = envelope.GroundFloor.U
        # building_data[i]['Construction']['C_internal']       = envelope.GroundFloor.U
        
        ############################## USE INFO ##############################
        # initialize user info
        building_data[i]['Use_info'] = dict.fromkeys(['n_px', 'people', 'appliances',
                                                      'schedule_set_heating',
                                                      'schedule_set_cooling'])
        building_data[i]['Use_info']['n_px'] = user.n_px
        building_data[i]['Use_info']['people'] = user.people # da enduse numero di persone (istat) * schedule normalizzata da norma
        building_data[i]['Use_info']['appliances'] = user.appliances
        building_data[i]['Use_info']['schedule_set_heating'] = user.heatingTSP # T_h (24 valori invece che 8760) da enduse come lista
        building_data[i]['Use_info']['schedule_set_cooling'] = user.coolingTSP # T_c (?) da enduse come lista
        
        
        ######################### BUILDING SERVICES ##########################
        building_data[i]['Building_services'] = dict.fromkeys(['DHW_demand_kWh'])
        
        building_data[i]['Building_services']['DHW_demand_kWh'] = consumption_appliances["DHW"].Qw_UNI11300.loc[i]
        # all the other info about HVAC systems are assigned inside function process_HVAC_data
        
        ############################ LOCATION ################################
        building_data[i]['Location'] = dict.fromkeys(['Regione', 'Provincia'])
        # building_data[i]['Location']['Latitude'] = 
        # building_data[i]['Location']['Longitude'] = 
        
        building_data[i]['Location']['Regione']   = regioni[istat['reg'].loc[i]]       
        building_data[i]['Location']['Provincia'] = scegli_provincia_distr(building_data[i]['Location']['Regione'], 
                                                                    istat.loc[i]['rand'], 
                                                                    sort_by = 'Numero_edifici')


    comp_time = time.process_time() - tic
    print("[Building data prepared in {:.2f} s]".format(comp_time))
       
    return building_data 

#%%

def process_HVAC_data(dataset, buildings_data, consumption_appliances):
    
    file_matrici = os.path.join('resources','HVAC_data','Impianti.xlsx')
    
    info_acs = consumption_appliances["DHW_info"]
    
    info_impianti = crea_impianti(dataset, file_matrici, info_acs) # P_desing risc
    
    info_impianti["PDC"] = info_impianti["PDC"].fillna("No")
    info_impianti["PDC acs"] = info_impianti["PDC acs"].fillna("No")
    info_impianti["Fuel acs"] = info_impianti["Fuel acs"].fillna("No")
    
    for bd_k, bd_data in buildings_data.items():
        
        bd_data['Building_services']['eff_emission'] = info_impianti.loc[bd_k]['emissione']
        bd_data['Building_services']['eff_distribution'] = info_impianti.loc[bd_k]['distribuzione']
        bd_data['Building_services']['eff_regulation'] = info_impianti.loc[bd_k]['regolazione']
        bd_data['Building_services']['eff_generation'] = info_impianti.loc[bd_k]['generazione']
            
        bd_data['Building_services']['dhw_eff_emission'] = info_impianti.loc[bd_k]['emissione']
        bd_data['Building_services']['dhw_eff_distribution'] = info_impianti.loc[bd_k]['distribuzione']
        bd_data['Building_services']['dhw_eff_regulation'] = info_impianti.loc[bd_k]['regolazione']
        bd_data['Building_services']['dhw_eff_generation'] = info_impianti.loc[bd_k]['generazione']
        
        bd_data['Building_services']['heat_emission_temp'] = {
                                                            "Radiante":40.,
                                                            "Radiatori":75.,
                                                            "Fancoil":60.,
                                                            }[info_impianti.loc[bd_k]['TipoEmettitore']]
        bd_data['Building_services']['heat_emission_conv_frac'] = {
                                                            "Radiante":.35,
                                                            "Radiatori":.35,
                                                            "Fancoil":.99,
                                                            }[info_impianti.loc[bd_k]['TipoEmettitore']]
        
        bd_data['Building_services']['heating_fuel'] = info_impianti.loc[bd_k]['Fuel']
        bd_data['Building_services']['biomass'] = info_impianti.loc[bd_k]["Biomass"]
        bd_data['Building_services']['dhw_heating_fuel'] = info_impianti.loc[bd_k]['Fuel acs']
        
        bd_data['Building_services']['PDC'] = info_impianti.loc[bd_k]['PDC']
        bd_data['Building_services']['dhw_PDC'] = info_impianti.loc[bd_k]['PDC acs']
        
        

        ################################################################################################
        ################################### DA CAMBIARE PARTE COOLING ##################################
        ################################################################################################
        bd_data['Building_services']['cooling system'] = info_impianti.loc[bd_k]['PDC']
        bd_data['Building_services']['cool_emission_temp'] = {
                                                            "Radiante":14,
                                                            "Radiatori":20,
                                                            "Fancoil":7,
                                                            }[info_impianti.loc[bd_k]['TipoEmettitore']]
    
    return buildings_data,  info_impianti


#%%
def process_users_data(istat_data, consumption_appliances):
    
    tic = time.process_time()
    
    print("\nOverwriting real users on standard users data..")
    # Import building envelopes
    standard_users_path = os.path.join('resources', 'user_data', 'standard_users.pickle')
    
    with open(standard_users_path, 'rb') as handle:
        standard_users = pickle.load(handle) 
        
    
    # selected_users = dict()    
    # if selected_buildings is None:
    #     selected_users = users
    # else:
    #     # selected_envelopes = [envelopes[s] for s in selected_buildings]
    #     for s in selected_buildings:
    #         selected_users.update({s: users[s]})
    
    istat = istat_data['istat']
    istat['Fascia_clima'] = istat['reg'].apply(fascia_climatica)
    el_cons = consumption_appliances['el_cons']
    users = dict()

    for i in istat.index:
        user = copy.deepcopy(standard_users['Residenziale'])  
        # Replace standard user with "real" user based on ISTAT survey
        user.substitute_istat_sched(istat.loc[i],
                                    el_cons.loc[i],
                                    # retrofits = None, 
                                    primo_giorno = 7)
        users[i] = user
    
    comp_time = time.process_time() - tic
    print("[Appliances simulations concluded in {:.2f} s]".format(comp_time))
    
    return users



#%%

def simulate_appliances(appliances_data, istat_data):
    
    tic = time.process_time()
    
    ##### Electric Devices Consumption Calculation [kWh/year per household]  ###
    #
    # Ogni appliance è calcolata con una funzione e una classe, contenute in 
    # classes.Elettrodomestici_MOIRAE
    # Per ogni informazione e per modificare i modelli andare nelle funzioni
    print("\nElectrical appliances simulation started:")
    
    # istat = istat_data['istat']
    istat = copy.deepcopy(istat_data['istat'])
    
    # Creazione di alcune variabili contenitore
    # names = ['tipologia','dati']
    # famiglia = ['N_abitanti']    
    elettro_info = pd.DataFrame(index = istat.index, columns = pd.MultiIndex.from_arrays([[],[]], names = ['tipo','dati'])   )
    consumption_appliances = {}
       
    #####################  LIGHTs #############################################
    EEC_lights, luci_info = loadLights(istat, appliances_data['db_lights'])
    consumption_appliances['Lights'] = EEC_lights
    elettro_info = pd.concat([elettro_info, luci_info], axis = 1)
    print("Lights")
    
    
    ##################### LITTLE APPLIANCEs ###################################
    EEC_little_appliances, little_info = loadLittleAppliances(istat, 
                                                              appliances_data['db_little_appliances'])
    consumption_appliances['Little_appliances'] = EEC_little_appliances
    elettro_info = pd.concat([elettro_info, little_info], axis = 1)
    print("Little appliances")
    
    
    ##################### REFRIGERATORs #######################################
    EEC_refrigerators = loadRefrigerators_moirae(istat,
                                                 appliances_data['db_refrigerators'])               # metodo 1 (MOIRAE)
    EEC_refrigerators_2, frighi_info = loadRefrigerators_CE(istat,
                                                            appliances_data['db_refrigerators'])    # metodo 2 (calcolo su CE n.1060/2010)
    consumption_appliances['Refrigerators MOIRAE'] = EEC_refrigerators
    consumption_appliances['Refrigerators CE'] = EEC_refrigerators_2
    elettro_info = pd.concat([elettro_info, frighi_info], axis = 1)    
    print("Refrigerators")
    
    
    ##################### APPLIANCEs ##########################################
    EEC_appliances = loadAppliances_moirae(istat,
                                           appliances_data['db_appliances'])                  # metodo 1 (MOIRAE)
    EEC_appliances_2, big_info = loadAppliances_CE(istat,
                                                   appliances_data['db_appliances'])          # metodo 2 (calcolo su CE n.1060/2010)    
    consumption_appliances['Big_appliances MOIRAE'] = EEC_appliances
    consumption_appliances['Big_appliances CE'] = EEC_appliances_2
    elettro_info = pd.concat([elettro_info, big_info], axis = 1)    
    print("Big appliances")
    
    
    #####################  TELEVISIONs & PCs ##################################
    EEC_screens, screens_info = loadScreens(istat,
                                            appliances_data['db_TVs'])
    consumption_appliances['Screens'] = EEC_screens
    elettro_info = pd.concat([elettro_info, screens_info], axis = 1)    
    print("TVs and screens")
    
    
    #####################  COOKINGS (hobs & oves) #############################
    EEC_cookings, cook_info = loadCookings(istat,
                                           appliances_data['db_cookings'])
    consumption_appliances['Cookings'] = EEC_cookings
    elettro_info = pd.concat([elettro_info, cook_info], axis = 1)
    print("Cookings")
    
    
    
    ########### Cooling demand [kWh/y] ########################################
    
    # Calcolo dei condizionatori
    EEC_cooling = loadCooling(istat, 
                              appliances_data['db_cool'], 
                              appliances_data['db_cool_cor'])
    consumption_appliances['CoolingSystems'] = EEC_cooling
    elettro_info = pd.concat([elettro_info, EEC_cooling.cool_info], axis = 1)  
    print("Cooling systems")
    

    ################ Standby demand [kWh/y]  ##################################

    EEC_standby = loadStandby(istat, 
                              appliances_data['db_standby'])
    consumption_appliances['Standby'] = EEC_standby
    print("Standby appliances")
        
    
    # Salvataggio dati consumi in un file excel
    consumption_appliances['appliances_info'] = elettro_info
    # elettro_info.to_excel(os.path.join(output_path,'elettro_info.xlsx'))
    

    ##################### DHW demand [kWh/year per household] #################

    DHW_demand, acs_info = loadDHW2(istat,
                                    appliances_data['db_dhw'])
    
    consumption_appliances['DHW'] = DHW_demand
    consumption_appliances['DHW_info'] = acs_info
    
    # acs_info.to_excel(os.path.join(output_path,'Acs_info.xlsx'))
    print("DHW")
    
    
    ############################ Store outputs ################################
    
    consumption_appliances['el_cons'] = pd.DataFrame() 
    _, db_cons = storeOutputs(consumption_appliances)
    consumption_appliances['el_cons'] = db_cons
    
    comp_time = time.process_time() - tic
    print("[Appliances simulations concluded in {:.2f} s]".format(comp_time))
    
    return consumption_appliances

#%%

def building_simulation_task(bd_k, bd_dict_info, weather_data):
    
    # location = as_float(bd_dict_info["Location"])
    general_info = bd_dict_info["General_info"]
    geometry = bd_dict_info["Geometry"]
    construction = bd_dict_info["Construction"]
    usage = bd_dict_info["Use_info"]
    services = bd_dict_info["Building_services"]
    location = bd_dict_info["Location"]
    
    surfaces_data = []
    for s in bd_dict_info["Surfaces"]:
        surfaces_data.append(s)
    
    prov = location['Provincia']         
    weather_file = weather_data["eureca_class"][prov]
    
    # print(bd_k)
    
    #########################################################
    # Constructions
    ext_wall = Construction.from_U_value("ExtWall",
                                         construction["U_walls"],
                                         weight_class="Heavy", #Light Heavy Medium
                                         construction_type="ExtWall")
    roof = Construction.from_U_value("Roof",
                                     construction["U_roof"],  
                                     weight_class="Heavy", #Light Heavy Medium
                                     construction_type="Roof")
    floor = Construction.from_U_value("GroundFloor",
                                      construction["U_floor"],
                                      weight_class="Heavy", #Light Heavy Medium
                                      construction_type="GroundFloor")

    internal_wall = Construction.from_U_value("IntWall",
                                            1.,
                                            weight_class="Heavy", #Light Heavy Medium
                                            construction_type="IntWall")
    ceiling = Construction.from_U_value("IntCeiling",
                                            1.,
                                            weight_class="Heavy", #Light Heavy Medium
                                            construction_type="IntWall")

    window = SimpleWindow(
                    name="Window",
                    u_value= construction["U_win"],
                    solar_heat_gain_coef=construction["g_win"],
                    frame_factor=construction["frame_factor"], # If no frame this should be zero
                    shading_coef_int=0.99,
                    shading_coef_ext=0.99, # The shading is set at the surface level
    )

    #########################################################
    # Surfaces
    surfaces_obj = []
    total_glazed_area = 0
    trespa_external_wall_opaque_area = 0.
    roof_opaque_area = 0.
    for s in surfaces_data:
        area = s["area"]
        if area > 1e-4:
            
            h = 90 - s["inclination"] # 0 Horizontal, 90 vertical
            a = s["orientation"] - 180 # 0 south, 90 west

            surf_obj = Surface.from_area_azimuth_height(
                f"Surface {s['name']}",
                area=area,
                height = h,
                azimuth = a,
                wwr=s["glazing_ratio"],
                surface_type={
                    "roof":"Roof",
                    "ext_wall":"ExtWall",
                    "wall":"ExtWall",
                    "floor":"GroundFloor",
                }[s["type"]],
                construction={
                    "roof":roof,
                    "ext_wall":ext_wall,
                    "wall":ext_wall,
                    "floor":floor,
                }[s["type"]],
                window=window,
            )
            surf_obj.reduce_area(s["area_adjacent"])
            surf_obj.shading_coefficient = s["shading"] if (type(s["shading"]) == float or type(s["shading"]) == int) else 1.
            surfaces_obj.append(surf_obj)
            total_glazed_area += surf_obj._glazed_area
            if s["type"].lower() in ["wall", "extwall","ext_wall"]:
                trespa_external_wall_opaque_area += surf_obj._opaque_area
            if s["type"].lower() in ["roof"]:
                roof_opaque_area += surf_obj._opaque_area

    intwall = SurfaceInternalMass(
        "IntWall",
        area=geometry["floor_area"]*2.5*2,
        surface_type="IntWall",
        construction=internal_wall,
    )
    intceiling = SurfaceInternalMass(
        "IntCeiling",
        area=geometry["floor_area"]*2,
        surface_type="IntCeiling",
        construction=ceiling,
    )
    surfaces_obj.append(intwall)
    surfaces_obj.append(intceiling)

    #########################################################
    # Internal Loads
    
    people_schedule_obj = Schedule(
        name='People',
        schedule_type='dimensionless',
        schedule = usage['people'],
    )
    
    app_schedule_obj = Schedule(
        name='Appliances',
        schedule_type='dimensionless',
        schedule = usage['appliances'],
    )

    people = People(
        name='occupancy_tz',
        unit='px',
        nominal_value=usage['n_px'],  #n_px
        schedule=people_schedule_obj,
        fraction_latent=0.0,
        fraction_radiant=0.3,
        fraction_convective=0.7,
        metabolic_rate=110*0.55,   # 
    )
    appliances = ElectricLoad(
        name='appliances',
        unit='W',
        nominal_value= 1., 
        schedule=app_schedule_obj,
        fraction_radiant=0.2,
        fraction_convective=0.8,
        fraction_to_zone=0.99, # la quota di carico termico è già calcolata in EndUse.substitute_istat_appliance
    )

    #########################################################
    # Setpoints

    heat_temp = Schedule(
        name = "t_heat",
        schedule_type="temperature",
        schedule=usage['schedule_set_heating']
    )

    cool_temp = Schedule(
        name = "t_cool",
        schedule_type="temperature",
        schedule=usage['schedule_set_cooling']
    )

    heat_hum = Schedule.from_constant_value(
        name="h_heat",
        schedule_type="dimensionless",
        value=0.
    )

    cool_hum = Schedule.from_constant_value(
        name="h_cool",
        schedule_type="dimensionless",
        value=1.
    )

    t_sp = SetpointDualBand(
        "t_sp",
        "temperature",
        schedule_lower=heat_temp,
        schedule_upper=cool_temp,
    )
    h_sp = SetpointDualBand(
        "h_sp",
        "relative_humidity",
        schedule_lower=heat_hum,
        schedule_upper=cool_hum,
    )

    #########################################################
    # Ventilation
    infiltration_sched = Schedule(
        "inf_sched",
        "dimensionless",
        np.array(([0.3] * 24) * 365),
    )

    # To account for window opening
    filt = (weather_file.hourly_data["out_air_db_temperature"] > 24) & (cool_temp.schedule > 30.)
    infiltration_sched.schedule[filt] = 5.

    inf_obj = Infiltration(
        name='inf_obj',
        unit='Vol/h',
        nominal_value=1.,
        schedule=infiltration_sched,
    )


    # Mechanical ventilation
    vent_sched = Schedule.from_constant_value(
        name="vent_sched",
        schedule_type="dimensionless",            
        value=0.
    )

    vent_obj = MechanicalVentilation(
        name='vent_obj',
        unit='Vol/h',
        nominal_value= 0.,
        schedule=vent_sched,
    )

    T_supply_sched = Schedule(
        "T_supply_sched",
        "temperature",
        np.array(([23.] * 8 + [23.] * 2 + [23.] * 4 + [23.] * 10) * 365),
    )

    x_supply_sched = Schedule(
        "x_supply_sched",
        "dimensionless",
        np.array(([0.0101] * 8 + [0.0101] * 2  + [0.0101] * 4 + [0.0101] * 10 ) * 365)*0.7,
    )

    availability_sched = np.array(([0] * 8 + [0] * 2  + [0] * 4 + [0] * 10 ) * 365)
    availability_sched[120*24:273*24] = -1*availability_sched[120*24:273*24]
    ahu_availability_sched = Schedule(
        "ahu_availability_sched",
        "availability",
        availability_sched,
    )

    
    #########################################################
    # Create zone
    tz1 = ThermalZone(
        name=f"Zone {bd_k}",
        surface_list=surfaces_obj,
        net_floor_area=geometry["floor_area"],
        volume=geometry["volume"],
        number_of_units = 1,
    )
    # Add everything
    
    if model == '1C':
        # 1C model
        tz1._ISO13790_params()
    elif model == '2C':
        # 2C model
        tz1._VDI6007_params()
    
    tz1.add_internal_load(people)
    tz1.add_internal_load(appliances)

    tz_loads = tz1.extract_convective_radiative_latent_electric_load()
    
    if model == '1C':
        # 1C model
        tz1.calculate_zone_loads_ISO13790(weather_file)
    elif model == '2C':
        # 2C model
        tz1.calculate_zone_loads_VDI6007(weather_file)
               

    tz1.add_temperature_setpoint(t_sp)
    tz1.add_humidity_setpoint(h_sp)

    tz1.add_infiltration(inf_obj)
    tz_inf = tz1.calc_infiltration(weather_file)

    ahu = AirHandlingUnit(
    "ahu",
    vent_obj,
    T_supply_sched,
    x_supply_sched,
    ahu_availability_sched,
    True,
    0.0,
    0.0,
    0.99,
    weather_file,
    tz1,
)


    dhw_1 = DomesticHotWater(
        "dhw_1",
        calculation_method="Number of occupants",
        n_of_occupants = usage['n_px'],
    )
    
    
    tz1.add_domestic_hot_water(weather_file, dhw_1)
    
    if model == 'MeanMonthly':
        bd = Building(f"Bd {bd_k}", thermal_zones_list=[tz1], model = '1C')
        bd.heating_system = _Heating_GRINS(heating_system_info = services)
        bd.cooling_system = _Cooling_GRINS(cooling_system_info = services)
        tz1.heating_sigma = bd.heating_system.sigma
        tz1.cooling_sigma = bd.cooling_system.sigma
        
        bd.set_hvac_system_capacity(weather_file)  
        
        df_res =  bd.simulate_quasi_steady_state(weather_file)
        
        df_res_res = df_res.copy()
    
    else:
        
        tz1.design_sensible_cooling_load(weather_file, model = model)
        tz1.design_heating_load(weather_file.hourly_data["out_air_db_temperature"].min() - 15.)

        bd = Building(f"Bd {bd_k}", thermal_zones_list=[tz1], model = model)

        bd.heating_system = _Heating_GRINS(heating_system_info = services)
        bd.cooling_system = _Cooling_GRINS(cooling_system_info = services)
        tz1.heating_sigma = bd.heating_system.sigma
        tz1.cooling_sigma = bd.cooling_system.sigma
        
        bd.set_hvac_system_capacity(weather_file)  
        
        df_res = bd.simulate(weather_file, output_folder=output_folder)
        
        df_res = df_res.droplevel(1, axis = 1)
        
        df_res_res = df_res.resample('M').sum()
        df_res_res.iloc[:,16:20] = df_res.iloc[:,16:20].resample('M').mean()
            

    # results_dict[bd_k] = df_res_res
   
    return df_res_res

def simulate_hvac(building_data, weather_data, **kwargs):
    
    output_folder=kwargs['output_folder']
    model=kwargs['model']
    
    
    print("\nBuilding simulation started:")
    tic = time.process_time()

    results_dict = {}
    
    # _check_results = {}
    # _hourly_results_dict = {}
    num_processes = 8
    args = ((bd_k, bd_dict_info, weather_data) for (bd_k, bd_dict_info) in building_data.items())
    with ProcessPoolExecutor(num_processes) as executor:        
        for df_res_res in executor.map(building_simulation_task, args):
            print(str(bd_k))
            results_dict[bd_k] = df_res_res

    # with ProcessPoolExecutor(8) as exe:
    #     for bd_k, bd_dict_info in zip(building_data.items(), exe.map(building_simulation_task, building_data.items())):
    #         results_dict[bd_k] = df_res_res
    

    comp_time = time.process_time() - tic    
    print("[Building simulations concluded in {:.2f} s]".format(comp_time))
        

    return results_dict


def process_results(consumption_appliances, consumption_hvac, buildings_data, **kwargs):
    
    um = kwargs['um']
    
    print("\nResults processing started:")
    tic = time.process_time()
    
    res = Process(consumption_appliances, consumption_hvac, buildings_data)
    
    res.read_conversion_factors()        
    
    res.save_simulation_output()
    res.generate_weighted_df()
    
    res.generate_carrier_df(um = um)
    res.store_all_results_into_dict()
    
    comp_time = time.process_time() - tic    
    print("[Results processing concluded in {:.2f} s]".format(comp_time))
    
    return res


# def visualize_results(res, heatmap = True, variable = 'Gas'):
    
#     print("\nResults visualizing started:")
#     tic = time.process_time()
    
#     if heatmap == True:
#         res.energy_map(variable)
    
#     comp_time = time.process_time() - tic    
#     print("[Results visualizing concluded in {:.2f} s]".format(comp_time))
    
#     return 