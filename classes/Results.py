# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:50:14 2024

@author: vivijac14771
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from funcs.aux_functions import decodificRegion2
import os
import sys
# global main_wd
from pathlib import Path
import json
from importlib import resources as impresources
from resources import conversion_factors as conversion_factors_file
main_wd = sys.path[0]

class Process(object):
    
    def __init__(self,consumption_appliances, consumption_hvac, buildings_data):
                
        ''' 
        Processing dei risultati 
        '''

        self.appl_df = consumption_appliances['el_cons']
        
        self.hvac_df = pd.DataFrame()
        dd = dict()
        for bd_key, bd_df in consumption_hvac.items():
            if not(bd_key=='model'):
                dd[bd_key] = consumption_hvac[bd_key].sum()
            
        _hvac_df = pd.DataFrame.from_dict(dd)
        self.hvac_df = _hvac_df.T
        
        self.info = {'model': consumption_hvac['model'], 
                     'um': 'GWh'}
        
        self.df  = pd.DataFrame(index = self.appl_df.index, 
                                columns = ['appliances_el_kWh', 
                                   'cooking_el_kWh',
                                   'cooking_gpl_kWh',
                                   'cooking_biomass_kWh',
                                   'heating_el_kWh',
                                   'heating_gas_Nm3',
                                   'heating_gpl_kWh',
                                   'heating_oil_l',
                                   'heating_dh_kWh',
                                   'heating_wood_kg',
                                   'heating_pellet_kg',
                                   'heating_biomass_kWh']
                                )
        
        info_df = pd.DataFrame(buildings_data.items(),columns=['id','data'])
        info_df['reg'] = 0.*info_df['id']
        info_df['coeff'] = 0.*info_df['id']
        info_df = info_df.set_index('id')
        
        for i in info_df.index:
            info_df.loc[i, 'reg'] = info_df['data'].loc[i]['Location']['Regione']
            info_df.loc[i, 'coeff'] = info_df['data'].loc[i]['General_info']['CoeffRiportoGlobale']
            
        self.df['reg'] = info_df['reg']
        self.df['coeff'] = info_df['coeff']
        
        
        self.read_fuel_properties()
        self.simulation_results()
        self.store_results_into_dict()
        
        
    def simulation_results(self):
        
        self.df['appliances_el_kWh'] = self.appl_df[['Standby appliances (kWh/y)' , 
                                                     'Lights Demand (kWh/y)' , 
                                                     'Little Appliances Demand (kWh/y)' , 
                                                     'Refrigerators Demand CE (kWh/y)' , 
                                                     'Big Appliances Demand CE (kWh/y)' , 
                                                     'TVs & PCs Demand (kWh/y)' ]].sum(axis=1)
        self.df['cooking_el_kWh'] = self.appl_df[['Electric Cookings Consumption (kWh/y)' , 
                                                       'Electric Ovens Consumption (kWh/y)']].sum(axis=1)
        # CHANGED --------------------------------------------------------------------------------------------
        self.df['cooking_gas_kWh'] = self.appl_df[['Gas Cookings Consumption (kWh/y)' , 
                                                   'Gas Ovens Consumption (kWh/y)']].sum(axis=1)
        self.df['cooking_gas_Nm3'] = self.df['cooking_gas_kWh']/10.6
        self.df['cooking_gas_Sm3'] = 1.05*self.df['cooking_gas_Nm3']
        # ----------------------------------------------------------------------------------------------------
        self.df['cooking_gpl_kWh'] = self.appl_df[['GPL Cookings Consumption (kWh/y)' , 
                                                   'GPL Ovens Consumption (kWh/y)']].sum(axis=1)
        
        self.df['cooking_biomass_kWh'] = self.appl_df[['Biomass Cookings Consumption (kWh/y)' , 
                                                            'Biomass Ovens Consumption (kWh/y)']].sum(axis=1)
        # ----------------------------------------------------------------------------------------------------
        self.df['heating_gpl_kg'] = self.hvac_df.get('Heating system LPG consumption [kg]')
        self.df['heating_gpl_MJ'] = self.df.get('heating_gpl_kg')*self.fuel_properties['LHV'].loc['LPG']
        self.df['heating_gpl_kWh'] = self.df.get('heating_gpl_MJ')/3.6
        # ----------------------------------------------------------------------------------------------------
        self.df['heating_gas_Nm3'] = self.hvac_df.get('Heating system gas consumption [Nm3]')
        self.df['heating_gas_Sm3'] = 1.05*self.hvac_df['Heating system gas consumption [Nm3]']
        self.df['heating_gas_kg'] = self.df['heating_gas_Sm3']*self.fuel_properties['Density'].loc['NaturalGas']
        self.df['heating_gas_MJ'] = self.df['heating_gas_kg']*self.fuel_properties['LHV'].loc['NaturalGas']
        self.df['heating_gas_kWh'] = self.df['heating_gas_MJ']/3.6
        # ----------------------------------------------------------------------------------------------------
        self.df['heating_gasoline_l'] =  self.hvac_df.get('Heating system gasoline consumption [L]')
        self.df['heating_gasoline_kg'] = self.df['heating_gasoline_l']/self.fuel_properties['Density'].loc['Gasoline']
        self.df['heating_gasoline_MJ'] = self.df['heating_gasoline_kg']*self.fuel_properties['LHV'].loc['Gasoline']
        self.df['heating_gasoline_kWh'] = self.df['heating_gasoline_MJ']/3.6
        # ----------------------------------------------------------------------------------------------------
        self.df['heating_wood_kg'] = self.hvac_df.get('Heating system wood consumption [kg]')  
        self.df['heating_wood_MJ'] = self.df['heating_wood_kg']*self.fuel_properties['LHV'].loc['Wood']
        self.df['heating_wood_kWh'] = self.df['heating_wood_MJ']/3.6
        
        self.df['heating_pellet_kg'] = self.hvac_df.get('Heating system wood consumption [kg]')
        self.df['heating_pellet_MJ'] = self.df['heating_pellet_kg']*self.fuel_properties['LHV'].loc['Pellets']
        self.df['heating_pellet_kWh'] = self.df['heating_pellet_MJ']/3.6
        
        self.df['heating_biomass_kWh'] = self.df['heating_wood_kWh'] + self.df['heating_pellet_kWh']
        
        # ----------------------------------------------------------------------------------------------------
        self.df['heating_el_kWh'] = self.hvac_df.get('Heating system electric consumption [Wh]')/1000
        self.df['heating_dh_kWh'] = self.hvac_df.get('Heating system DH consumption [Wh]')/1000
        # ----------------------------------------------------------------------------------------------------
        # df_res['Electric Consumption [Wh]'] = df_res['Heating system electric consumption [Wh]']+\
        #                                         df_res['Cooling system electric consumption [Wh]']+\
        #                                         df_res['AHU electric consumption [Wh]']+\
        #                                         df_res['Appliances electric consumption [Wh]']
        # df['cooling_el_kWh'] = hvac_df['Cooling system electric consumption [Wh]']/1000
        # ----------------------------------------------------------------------------------------------------
        self.df['cooling_el_kWh'] = self.appl_df['Cooling Systems (kWh/y)']
        self.df['ahu_el_kWh'] = self.hvac_df['AHU electric consumption [Wh]']/1000
        # ----------------------------------------------------------------------------------------------------

        self.df['heating_coal_kg'] = self.hvac_df['Heating system coal consumption [kg]'].sum()
 
        return
    

    def read_fuel_properties(self):

        # conv_filepath = os.path.join(main_wd, 'resources', 'conversion_factors', conv_filename)

        conv_filepath = impresources.files(conversion_factors_file) / 'Fuels.xlsx'
        self.fuel_properties = pd.read_excel(conv_filepath, sheet_name='Properties', index_col=0)

        # self.pe_conversion_factors = pd.read_excel(conv_filepath, sheet_name='PrimaryEnergy', index_col=0)
              
        return
    
    def store_results_into_dict(self):
        
        self.resume = dict()
        self.resume['df'] = self.df
        self.resume['info'] = self.info
        
        return
    
    
#%%

class PostProcess(object):
    
    def __init__(self, results, **kwargs):
                
        ''' 
        Postprocessing and visualization of results
        '''
        self.df                = results['df']
        # self.df_weighted       = results['df_weighted']
        # self.df_carrier        = results['df_carrier']
        # self.df_carrier_region = results['df_carrier_region']
        self.info              = results['info']
         
        self.info['units']    = kwargs['units']
        # self.info['units']    = kwargs['units']
        
        self.map_df     = pd.DataFrame()
        self.merged_df  = pd.DataFrame()
        
        self.read_conversion_factors()
        
        if not(self.info['units'] == 'GWh'):            
            self.convert_results()
            
        self.generate_weighted_df()
            
        self.generate_carrier_df()
        self.generate_use_df()
        
        self.df_carrier_region = self.generate_region_df(self.df_carrier)
        self.df_use_region = self.generate_region_df(self.df_use)
        
        # self.update_results() #called from outside
    
        
    def generate_weighted_df(self):
        
        self.df_weighted = self.df.drop('reg', axis=1)
        self.df_weighted = self.df_weighted.drop('coeff', axis=1)
        self.df_weighted = self.df_weighted.multiply(self.df['coeff'], axis="index")
        self.df_weighted['reg'] = self.df['reg']
        return
    
    
    def generate_region_df(self, df_input):
        
        # tot_df = pd.concat([appl_df, hvac_df, reg_df], axis = 1)
        df_region = pd.DataFrame()
        df_region = df_input.groupby(['reg']).sum()
        
        return df_region
    
    
    def generate_carrier_df(self):
        
        units = self.info['units'] 
        
        print(f"\n Resuming consumption by energy carrier in {units}..")
        
        self.df_carrier  = pd.DataFrame(index = self.df.index, 
                                        columns = ['Electricity',
                                                   'NaturalGas',
                                                   'LPG',
                                                   'Biomass',
                                                   'Gasoline',
                                                   'DistrictHeat'])
        
        columns_el = ['appliances_el_kWh', 'cooking_el_kWh','heating_el_kWh', 'cooling_el_kWh','ahu_el_kWh']
        columns_gas = ['heating_gas_kWh', 'cooking_gas_kWh']
        columns_lpg = ['cooking_gpl_kWh', 'heating_gpl_kWh']
        columns_bio = ['cooking_biomass_kWh', 'heating_biomass_kWh']
        columns_other = ['heating_gasoline_kWh', 'heating_dh_kWh']
        
        all_columns = columns_el + columns_gas + columns_lpg + columns_bio + columns_other
        
        self.df_weighted = self.df_weighted.reindex(columns = all_columns)
        
        Electricity_GWh = self.df_weighted[columns_el].sum(axis = 1)/1e6
        
        NaturalGas_GWh = self.df_weighted[columns_gas].sum(axis = 1)/1e6
        
        LPG_GWh = self.df_weighted[columns_lpg].sum(axis = 1)/1e6
        
        Biomass_GWh = self.df_weighted[columns_bio].sum(axis = 1)/1e6
        
        Gasoline_GWh = self.df_weighted[['heating_gasoline_kWh']].sum(axis = 1)/1e6
        
        DistrictHeat_GWh = self.df_weighted[['heating_dh_kWh']].sum(axis = 1)/1e6
        
        self.df_carrier['Electricity'] = Electricity_GWh
        self.df_carrier['NaturalGas']  = NaturalGas_GWh
        self.df_carrier['LPG']         = LPG_GWh
        self.df_carrier['Biomass']     = Biomass_GWh
        self.df_carrier['Gasoline']    = Gasoline_GWh
        self.df_carrier['DistrictHeat'] = DistrictHeat_GWh
 
        self.df_carrier['reg'] = self.df['reg']
        
        return
    
    def generate_use_df(self):
        
        units = self.info['units'] 
        
        print(f"\n Resuming consumption by end use in {units}..")
        
        self.df_use  = pd.DataFrame(index = self.df.index, 
                                    columns = ['Appliances',
                                               'Cooking',
                                               'SpaceHeating',
                                               'SpaceCooling',
                                               'AHU'])
        
        columns_appl = ['appliances_el_kWh']
        columns_cook = ['cooking_el_kWh', 'cooking_gas_kWh','cooking_gpl_kWh','cooking_biomass_kWh']
        columns_heat = ['heating_el_kWh', 'heating_gas_kWh', 'heating_gpl_kWh','heating_biomass_kWh','heating_gasoline_kWh', 'heating_dh_kWh']
        columns_cool = ['cooling_el_kWh' ]
        columns_ahu  = ['ahu_el_kWh']
        
        all_columns = columns_appl + columns_cook + columns_heat + columns_cool + columns_ahu
        
        self.df_weighted = self.df_weighted.reindex(columns = all_columns)
        
        Appliances_GWh = self.df_weighted[columns_appl].sum(axis = 1)/1e6
        
        Cooking_GWh = self.df_weighted[columns_cook].sum(axis = 1)/1e6
        
        Heating_GWh = self.df_weighted[columns_heat].sum(axis = 1)/1e6
        
        Cooling_GWh = self.df_weighted[columns_cool].sum(axis = 1)/1e6
        
        AHU_GWh = self.df_weighted[columns_ahu].sum(axis = 1)/1e6
        
        
        self.df_use['Appliances']   = Appliances_GWh
        self.df_use['Cooking']      = Cooking_GWh
        self.df_use['SpaceHeating'] = Heating_GWh
        self.df_use['SpaceCooling'] = Cooling_GWh
        self.df_use['AHU']          = AHU_GWh

        self.df_use['reg'] = self.df['reg']
        
        return
    
    
    def update_results(self):
        
        self.all = dict()
        self.all['df'] = self.df
        self.all['df_weighted'] = self.df_weighted
        self.all['df_carrier'] = self.df_carrier
        self.all['df_use'] = self.df_use
        self.df_carrier_region = self.generate_region_df(self.df_carrier)
        self.df_use_region = self.generate_region_df(self.df_use)
        self.all['df_carrier_region'] = self.df_carrier_region
        self.all['df_use_region'] = self.df_use_region
        self.all['info'] = self.info
        
        return self.all
        
    def convert_results(self):
        
        conversion_factor_ktep_to_GWh = 11.63
        units = self.info['units']
        
        if units == 'ktep':
            conversion_factor = 1./conversion_factor_ktep_to_GWh
        elif units == 'GWh':
            conversion_factor = 1.
        
        for col in self.df.columns:
            try:
                self.df[col] = self.df[col].multiply(conversion_factor)
            except:
                print(f"\Could not convert {col} because of TypeError")
                
        print(f"\nResults converted from GWh to {units}..")

        return
    
    def create_geodata(self):
        
        shp_filepath = os.path.join(main_wd, 'resources', 'istat_data', 'Reg01012024_g', 'Reg01012024_g_WGS84.shp')
        # shp_path = './resources/istat_data/Reg01012024_g/Reg01012024_g_WGS84.shp'
        map_df = gpd.read_file(shp_filepath)
        map_df['reg'] = map_df['COD_REG'].apply(decodificRegion2)
        
        return map_df

    def read_census_data(self):
        # path to template epw file (to be changed according to region)
        census_file = os.path.join(main_wd, 'resources', 'istat_data', 
                                         'residential_buildings_census.csv')
        
        self.census_df = pd.read_csv(census_file, keep_default_na=False, index_col=0)
        
        return     
    
    def heatmaps(self, group_by = 'carrier', **kwargs):
        
        specify = kwargs['specify']
        # save_figures = kwargs['save_figures']
        
        if group_by == 'carrier':
            df = self.df_carrier_region
        elif group_by == 'use':
            df = self.df_use_region
            
        # if specify == 'pro_capite':
        #     df = df/num_people  
        # elif specify == 'pro_surface':
        #     df = df/surf_area
        
        map_df = self.create_geodata()
        
        for carrier in df.columns[0:5]:
        
            # carrier = df.columns[column]
            
            unit = '[' + self.info['units'] + ']'
            
            merged_df = pd.merge(map_df, df[carrier], how = 'outer',  on='reg')
            
            # vmin, vmax = 0, max(self.results[variable].values)# create figure and axes for Matplotlib
            fig, ax = plt.subplots(1, figsize=(8, 6))
            
            # orig_map = plt.get_cmap('OrRd')  # RdYlGn
            # reversed_map = orig_map.reversed()
            
            merged_df.plot(column=carrier,
                                # title = title,
                                legend = True,
                                # legend_kwds= {'labels': [unit]},
                                cmap='OrRd',   #orig_map, 
                                # linewidth=0.5, 
                                ax=ax, 
                                edgecolor='0.8')
            
            # a.set_label(unit, 'center')
    
            ax.axis('off')    # remove the axis
            ax.set_title(f'{carrier} consumption {unit}', fontdict={'fontsize': '12', 'fontweight' : '3'})# create an annotation for the data source
        
        plt.show()
            
            # if save_figures is True:
            #     plt.savefig(args, kwargs)
            
        return
    
    def read_conversion_factors(self):
        
        conv_filename = 'Fuels.xlsx'
        conv_filepath = os.path.join(main_wd, 'resources', 'conversion_factors', conv_filename)
        
        self.fuel_properties = pd.read_excel(conv_filepath, sheet_name='Properties', index_col=0)
        self.pe_conversion_factors = pd.read_excel(conv_filepath, sheet_name='PrimaryEnergy', index_col=0)
        
        
    def print_csv_files(self, output_folder = None):
        for key,value in self.all.items():
            
            if isinstance(value, pd.DataFrame):
                file_name = key + '.csv'
                output_file = os.path.join(main_wd, 'output', output_folder, file_name)
                output_filepath = Path(output_file)
                output_filepath.parent.mkdir(parents=True, exist_ok=True)
                value.to_csv(output_filepath)
            elif isinstance(value, dict):
                file_name = key + '.json'
                output_file = os.path.join(main_wd, 'output', output_folder, file_name)  
                output_filepath = Path(output_file)
                output_filepath.parent.mkdir(parents=True, exist_ok=True)
                json_obj = json.dumps(value)
                json_file =  open(output_filepath, "w")
                json_file.write(json_obj)
                json_file.close()
            else:
                print(f"\Could not print {key} because is neither a Dataframe nor a dictionary")

            
            
        
        

   
#%%
    

    
    
    # def degree_days(T_ext, threshold = 16):
    #     temp_hours = T_ext.reshape(365,24)
    #     temp_days = np.mean(temp_hours, axis = 1)
    #     dd = np.sum(threshold - temp_days[temp_days < threshold])
    #     return dd

    # def weather_data(s, weather_dict):
    #     s['DD'] = np.zeros(len(s.index))
    #     for key, w in weather_dict.items():
    #         s['DD'].loc[s.index == key] = degree_days(w.Text, threshold = 16)  
    #     return s


    # def validation_gas(in_path, p, res, cons, year = 2013):
        
    #     # Consumi Gas del MISE (dati SNAM)
    #     filename = 'Gas_Distribuito_Province_' + str(year) + '.xls'

    #     consumi_gas = pd.read_excel(os.path.join(in_path,'DatiMISE',filename), 
    #                                 header = 4, index_col = 1, keep_default_na=False, usecols= 'A:F',)

    #     consumi_gas = consumi_gas[['REGIONE','RETI DI DISTRIBUZIONE ']] #'INDUSTRIALE','TERMOELETTRICO',

    #     consumi_gas.rename(columns = {'RETI DI DISTRIBUZIONE ':'GasCons_MISE',
    #                                   'REGIONE': 'Prov',
    #                               # 'INDUSTRIALE': 'Industria'
    #                               }, inplace = True)

    #     cons_df = consumi_gas.drop(['TOTALE','TOTALE '])
    #     cons_df = cons_df.loc[cons_df['Prov'].str.len()==2]
    #     cons_df = cons_df.set_index(['Prov'])
        
    #     # Consumi Gas calcolati dal modello
    #     cons_calc = cons/(10.69*1e6) # kWh --> Mm3
    #     res['GasCons_Calc'] = np.multiply(cons_calc['NaturalGas'],p['RiportGlobale'])
    #     cons_df['GasCons_Calc'] = np.zeros(len(cons_df.index))
    #     for prov in cons_df.index:
    #         r = res['GasCons_Calc'].loc[p['Provincia']==prov]
    #         cons_df['GasCons_Calc'].loc[cons_df.index==prov] = sum(r.values)
    #         # s['GasCons_Meas'].loc[s['prov']==prov] = cons_df['GasCons_MISE']                 
        
    #     return cons_df
    
    

        
        