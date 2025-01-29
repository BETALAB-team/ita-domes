

import numpy as np
import pandas as pd
import os
import math
from eureca_building.config import CONFIG
from eureca_building.systems import System
from eureca_building.fluids_properties import fuels_pci


class _Heating_GRINS(System):

    gas_consumption = 0
    electric_consumption = 0
    wood_consumption = 0
    oil_consumption = 0
    coal_consumption = 0
    DH_consumption = 0
    pellet_consumption = 0

    def __init__(self, *args, **kwargs):

        # kwargs = {
        #     "generation_type":...,
        #
        #     "eff_emission": ...,
        #     "eff_distribution": ...,
        #     "eff_regulation": ...,
        #     "eff_generation": ...,
        #
        #     "dhw_eff_emission": ...,
        #     "dhw_eff_distribution": ...,
        #     "dhw_eff_regulation": ...,
        #     "dhw_eff_generation": ...,
        #
        #     "heating_fuel": ...,
        #     "dhw_heating_fuel": ...,
        #
        #     "heat_emission_temp": ...,
        #     "heat_emission_conv_frac": ...,
        # }

        self.system_info = kwargs["heating_system_info"]

        self.emission_control_efficiency = self.system_info["eff_emission"] * self.system_info["eff_regulation"]
        self.emission_temp = self.system_info["heat_emission_temp"]
        self.distribution_efficiency = self.system_info["eff_distribution"]
        self.convective_fraction = self.system_info["heat_emission_conv_frac"]
        self.generation_efficiency = self.system_info["eff_generation"]

        self.sigma = {
            "1C" : (1-self.convective_fraction, self.convective_fraction),
            "2C" : (
            (1-self.convective_fraction)/2, # Radiative IW
            (1-self.convective_fraction)/2, # Radiative AW
            self.convective_fraction)       # Convective
        }

        self.dhw_emission_control_efficiency = self.system_info["dhw_eff_emission"] * self.system_info["dhw_eff_regulation"]
        self.dhw_distribution_efficiency = self.system_info["dhw_eff_distribution"]
        self.dhw_generation_efficiency = self.system_info["dhw_eff_generation"]

        self.fuel_type = self.system_info["heating_fuel"]
        self.biomass_type = self.system_info["biomass"]
        self.dhw_fuel_type = self.system_info["dhw_heating_fuel"]

        # In case of HP needs to be changed
        self.total_efficiency = self.emission_control_efficiency * self.distribution_efficiency * self.generation_efficiency
        self.dhw_total_efficiency = self.dhw_emission_control_efficiency * self.dhw_distribution_efficiency * self.dhw_generation_efficiency

        # Input Data
        # self.PCI_natural_gas = fuels_pci["Natural Gas"]  # [Wh/Nm3]

        self.charging_mode = np.nan
        self.dhw_tank_current_charge_perc = np.nan
        self.dhw_capacity_to_tank = np.nan

    def set_system_capacity(self, design_power, weather):
        ''''Choice of system size based on estimated nominal Power

        Parameters
        ----------
        design_power : float
            Design Heating power  [W]
        Weather : eureca_building.weather.WeatherFile
            WeatherFile object
        '''

        if "PDC aria" in self.system_info["PDC"]:
            self.COP = 3.2
        elif "PDC acqua falda" in self.system_info["PDC"]:
            self.COP = 3.2
        elif  "PDC terreno" in self.system_info["PDC"]:
            self.COP = 3.2
        else:
            self.COP = 1


        if "PDC aria" in self.system_info["dhw_PDC"]:
            self.dhw_COP = 3.2
        elif  "PDC acqua falda" in self.system_info["dhw_PDC"]:
            self.dhw_COP = 3.2
        elif  "PDC terreno" in self.system_info["dhw_PDC"]:
            self.dhw_COP = 3.2
        else:
            self.dhw_COP = 1


    def solve_system(self, heat_flow, dhw_flow, weather, t, T_int, RH_int):
        '''This method allows to calculate the system power for each time step

        Parameters
        ----------
        heat_flow : float
            required power  [W]
        Weather : eureca_building.weather.WeatherFile
            WeatherFile object
        t : int
            Simulation time step
        T_int : float
            Zone temperature [°]
        RH_int : float
            Zone relative humidity [%]
        '''
        # Corrected efficiency and losses at nominal power

        self.oil_consumption = 0
        self.coal_consumption = 0
        self.DH_consumption = 0
        self.wood_consumption = 0
        self.pellet_consumption = 0
        self.electric_consumption = 0
        self.gas_consumption = 0
        self.gasoline_consumption = 0
        self.lpg_consumption = 0

        total_energy = heat_flow / self.total_efficiency

        if "Oil" in self.fuel_type:
            self.oil_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["Oil"]
        elif "Gasoline" in self.fuel_type:
            self.gasoline_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["Gasoline"]
        elif "Coal" in self.fuel_type:
            self.coal_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["Coal"]
        elif "LPG" in self.fuel_type:
            self.lpg_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["LPG"]
        elif "NaturalGas" in self.fuel_type:
            self.gas_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["Natural Gas"]
        elif "Biomass" in self.fuel_type:
            if "Wood" in self.biomass_type:
                self.wood_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["Wood"]
            else:
                self.pellet_consumption = total_energy / CONFIG.ts_per_hour / fuels_pci["Pellets"]
        elif "Electric" in self.fuel_type:
            self.electric_consumption = total_energy / CONFIG.ts_per_hour / self.COP
        elif "DH" in self.fuel_type:
            self.DH_consumption = total_energy / CONFIG.ts_per_hour

        dhw_total_energy = dhw_flow / self.dhw_total_efficiency

        if "Oil" in self.dhw_fuel_type:
            self.oil_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["Oil"]
            
        elif "Gasoline" in self.dhw_fuel_type:
            self.gasoline_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["Gasoline"]
            
        elif "Coal" in self.dhw_fuel_type:
            self.coal_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["Coal"]
            
        elif "LPG" in self.dhw_fuel_type:
            self.lpg_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["LPG"]
            
        elif "NaturalGas" in self.dhw_fuel_type:
            self.gas_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["Natural Gas"]
            
        elif "Biomass" in self.dhw_fuel_type:
            if "Wood" in self.biomass_type:
                self.wood_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["Wood"]
            else:
                self.pellet_consumption += dhw_total_energy / CONFIG.ts_per_hour / fuels_pci["Pellets"]
        elif "Electric" in self.dhw_fuel_type:
            self.electric_consumption += dhw_total_energy / CONFIG.ts_per_hour / self.dhw_COP
        elif "DH" in self.dhw_fuel_type:
            self.DH_consumption += dhw_total_energy / CONFIG.ts_per_hour


    def solve_quasi_steady_state(self, heat_flow, dhw_flow):
        '''This method allows to calculate the system power for each month

        Parameters
        ----------
        heat_flow : float
            required power  [Wh]
        dhw_flow : float
            required power  [Wh]
        '''
        # Corrected efficiency and losses at nominal power

        self.oil_consumption = 0
        self.coal_consumption = 0
        self.DH_consumption = 0
        self.wood_consumption = 0
        self.pellet_consumption = 0
        self.electric_consumption = 0
        self.gas_consumption = 0
        self.gasoline_consumption = 0
        self.lpg_consumption = 0

        total_energy = heat_flow / self.total_efficiency # Wh

        if "Oil" in self.fuel_type:
            self.oil_consumption = total_energy / fuels_pci["Oil"]
        elif "Gasoline" in self.fuel_type:
            self.gasoline_consumption = total_energy / fuels_pci["Gasoline"]
        elif "Coal" in self.fuel_type:
            self.coal_consumption = total_energy / fuels_pci["Coal"]
        elif "LPG" in self.fuel_type:
            self.lpg_consumption = total_energy / fuels_pci["LPG"]
        elif "NaturalGas" in self.fuel_type:
            self.gas_consumption = total_energy / fuels_pci["Natural Gas"]
        elif "Biomass" in self.fuel_type:
            if "Wood" in self.biomass_type:
                self.wood_consumption = total_energy / fuels_pci["Wood"]
            else:
                self.pellet_consumption = total_energy / fuels_pci["Pellets"]
        elif "Electric" in self.fuel_type:
            self.electric_consumption = total_energy / self.COP
        elif "DH" in self.fuel_type:
            self.DH_consumption = total_energy / CONFIG.ts_per_hour

        dhw_total_energy = dhw_flow / self.dhw_total_efficiency

        if "Oil" in self.dhw_fuel_type:
            self.oil_consumption += dhw_total_energy / fuels_pci["Oil"]
        elif "Gasoline" in self.dhw_fuel_type:
            self.gasoline_consumption += dhw_total_energy / fuels_pci["Gasoline"]
        elif "Coal" in self.dhw_fuel_type:
            self.coal_consumption += dhw_total_energy / fuels_pci["Coal"]
        elif "LPG" in self.dhw_fuel_type:
            self.lpg_consumption += dhw_total_energy / fuels_pci["LPG"]
        elif "NaturalGas" in self.dhw_fuel_type:
            self.gas_consumption += dhw_total_energy / fuels_pci["Natural Gas"]
        elif "Biomass" in self.dhw_fuel_type:
            if "Wood" in self.biomass_type:
                self.wood_consumption += dhw_total_energy / fuels_pci["Wood"]
            else:
                self.pellet_consumption += dhw_total_energy / fuels_pci["Pellets"]
        elif "Electric" in self.dhw_fuel_type:
            self.electric_consumption += dhw_total_energy / self.dhw_COP
        elif "DH" in self.dhw_fuel_type:
            self.DH_consumption += dhw_total_energy / CONFIG.ts_per_hour

class _Cooling_GRINS(System):

    gas_consumption = 0
    electric_consumption = 0
    wood_consumption = 0
    oil_consumption = 0
    coal_consumption = 0
    DH_consumption = 0
    pellet_consumption = 0

    def __init__(self, *args, **kwargs):

        # kwargs = {
        #     "generation_type":...,
        #
        #     "eff_emission": ...,
        #     "eff_distribution": ...,
        #     "eff_regulation": ...,
        #     "eff_generation": ...,
        #
        #     "dhw_eff_emission": ...,
        #     "dhw_eff_distribution": ...,
        #     "dhw_eff_regulation": ...,
        #     "dhw_eff_generation": ...,
        #
        #     "heating_fuel": ...,
        #     "dhw_heating_fuel": ...,
        #
        #     "heat_emission_temp": ...,
        #     "heat_emission_conv_frac": ...,
        # }

        self.system_info = kwargs["cooling_system_info"]

        self.emission_temp = self.system_info["cool_emission_temp"]

        self.convective_fraction = 0.99

        self.sigma = {
            "1C" : (1-self.convective_fraction, self.convective_fraction),
            "2C" : (
            (1-self.convective_fraction)/2, # Radiative IW
            (1-self.convective_fraction)/2, # Radiative AW
            self.convective_fraction)       # Convective
        }

        self.fuel_type = "Electric"

    def set_system_capacity(self, design_power, weather):
        ''''Choice of system size based on estimated nominal Power

        Parameters
        ----------
        design_power : float
            Design Heating power  [W]
        Weather : eureca_building.weather.WeatherFile
            WeatherFile object
        '''

        if "PDC aria" in self.system_info["cooling system"]:
            self.EER = 2.2
        elif "PDC acqua falda" in self.system_info["cooling system"]:
            self.EER = 2.2
        elif  "PDC terreno" in self.system_info["cooling system"]:
            self.EER = 2.2
        else:
            self.EER = np.nan

    def solve_system(self, heat_flow, weather, t, T_int, RH_int):
        '''This method allows to calculate the system power for each time step

        Parameters
        ----------
        heat_flow : float
            required power  [W]
        Weather : eureca_building.weather.WeatherFile
            WeatherFile object
        t : int
            Simulation time step
        T_int : float
            Zone temperature [°]
        RH_int : float
            Zone relative humidity [%]
        '''
        # Corrected efficiency and losses at nominal power

        total_energy = heat_flow

        self.electric_consumption = -1 * total_energy / CONFIG.ts_per_hour / self.EER

    def solve_quasi_steady_state(self, heat_flow):
        '''This method allows to calculate the system power for each month

        Parameters
        ----------
        heat_flow : float
            required power  [Wh]
        '''
        # Corrected efficiency and losses at nominal power

        total_energy = heat_flow

        self.electric_consumption = -1 * total_energy / self.EER
        
        
#%% Inherited from MODENA

