# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:40:32 2023

@author: vivijac14771
"""

import pandas as pd
import os


def read_results(year = None):
    
    cwd = os.getcwd()
    
    filename = 'SummaryRisultati.xlsx'
    if year == None:
        folder = 'OutputGlobaleCTI'
    else:
        folder = 'OutputGlobaleCTI-' + str(year)
    filepath = os.path.join(cwd, folder, filename)
    
    data = pd.read_excel(filepath,
                         sheet_name='consumi_per_vettore', 
                         # skiprows=4, 
                         usecols='A:I')
    
    info = pd.read_excel(filepath,
                         sheet_name='info_edificio', 
                         # skiprows=4, 
                         usecols='A:O')
    
    coeff = info[['id','RiportGlobale']]
    
    data = data.set_index('id')
    coeff = coeff.set_index('id')
    
    data2 = data.copy()    
    for col in data.columns:
        data2[col] = data[col] * coeff.RiportGlobale
    
    data3 = data2.sum(axis = 0)
    res = pd.DataFrame(columns=['kWh','MWh','GWh','vettori'])
    res['kWh'] = data3.values
    res['MWh'] = res['kWh']/1e3
    res['GWh'] = res['kWh']/1e6
    res['vettori'] = data.columns #.str.replace(' (kWh/y)','')
    res['vettori'] = res['vettori'].str.replace(" (kWh/y)","")
    res['vettori'] = res['vettori'].str.replace(r"\(.*\)","")
    # res['vettori'] = res['vettori'].str.strip()
    res = res.set_index('vettori')
    res.index = res.index.str.strip()
    
    pres = primary_energy_conversion(res)
    
    return pres


def primary_energy_conversion(res):
    
    pres = res.copy() 
    pres['Sm3'] = 0*res['kWh']
    pres['tep'] = 0*res['kWh']
    pres['kg']  = 0*res['kWh']  
    
    # Fonte dati:
    #
    # (1) Circolare MISE del 18 dicembre 2014
    # (2) Circolare MICA del 2 marzo 1992, N. 219/F
    # 
    # prese da https://fire-italia.org/wp-content/uploads/2023/04/2023-04-Guida-contabilita.pdf
    #
    # fattori_conv_pe = {'NaturalGas': 0.882, # tep/1000 Nm3, oppure 0.836 tep/1000 Sm3
    #                    'Gasoline': 1.017, # tep/t
    #                    'LPG': 1.099, # tep/t
    #                    'Wood': 0.20, # tep/t (cippato)
    #                    'Pellets': 0.40, #tep/t
    #                    'Electric': 0.187, #tep/MWh (da rete elettrica nazionale)
    #                    'Oil': 1.2 # tep/t,
    #                    'Coke': 0.74 #tep/t (carbon fossile)
    #                    }
    
    # Conversione gas naturale
    pres['Sm3'].loc['NaturalGas'] = res.loc['NaturalGas']['kWh']/10.69  # 1 Sm3 = 10.69 kWh 
    pres['tep'].loc['NaturalGas'] = pres.loc['NaturalGas']['Sm3']/1000*0.836
    
    # Conversione gasolio
    PCI_gasolio = 11.8 # kWh/kg
    pres['kg'].loc['Gasoline'] = pres.loc['Gasoline']['kWh']/PCI_gasolio
    pres['t']  = pres['kg']/1e3
    pres['tep'].loc['Gasoline'] = pres.loc['Gasoline']['t']*1.017
    
    # Conversione LPG
    PCI_gpl = 12.8 # kWh/kg
    pres['kg'].loc['LPG'] = pres.loc['LPG']['kWh']/PCI_gpl
    pres['t']  = pres['kg']/1e3
    pres['tep'].loc['LPG'] = pres.loc['LPG']['t']*1.099
    
    # Conversione legna e pellet
    PCI_legna = 3.5 # kWh/kg
    pres['kg'].loc['Wood'] = pres.loc['Wood']['kWh']/PCI_legna
    pres['t']  = pres['kg']/1e3
    pres['tep'].loc['Wood'] = pres.loc['Wood']['t']*0.2
    
    pres['kg'].loc['Pellets'] = pres.loc['Pellets']['kWh']/PCI_legna
    pres['t']  = pres['kg']/1e3
    pres['tep'].loc['Pellets'] = pres.loc['Pellets']['t']*0.4
    
    # Conversione energia elettrica
    pres['tep'].loc['Electric'] = pres.loc['Electric']['MWh']*0.187
    
    # Conversione oil products
    PCI_benzine = 12.2 
    pres['kg'].loc['Oil'] = pres.loc['Oil']['kWh']/PCI_benzine
    pres['t']  = pres['kg']/1e3
    pres['tep'].loc['Oil'] = pres.loc['Oil']['t']*1.2
    
    # Conversione coke
    PCI_coke = 9.6
    pres['kg'].loc['Coke'] = pres.loc['Coke']['kWh']/PCI_coke
    pres['t']  = pres['kg']/1e3
    pres['tep'].loc['Coke'] = pres.loc['Coke']['t']*0.74
    
    # pres['t']  = pres['kg']/1e3
    pres['ktep'] = pres['tep']/1e3
    
    return pres


def read_ben():
    
    cwd = os.getcwd()
    
    
    # natural_gas_MSm3_mase = gas_data.values[0].sum(axis = 0)
    # natural_gas_ktep_mase = PCI*(1e6*natural_gas_MSm3_mase)/11630/1000  # Sm3 --> KWh --> tep --> ktep

    df = pd.DataFrame(columns = ['Year', 'Natural gas', 'Electric energy', 'Biomass', 'LPG', 'Gasoline'])

    settore = ['Households']  #['Households','Commercial & public services']

    df2 = df.copy()
    
    filename = 'BEN - Italia Metodologia Eurostat 1990 - 2020.xlsx'
    filepath = os.path.join(cwd, 'Resources', 'validation-data', filename)
    
    
    for year in range(2013,2021):
        
        data = pd.read_excel(filepath,
                              sheet_name=str(year), skiprows=4, usecols='C,I:CB')

        data.rename(columns = {'Unnamed: 2':'Sector'}, inplace = True)
        data = data.set_index('Sector')

        data = data.loc[settore]
        
        # data = data.loc[:, (data != 0).any(axis=0)]
        data = data.loc[:, (data != 'Z').any(axis=0)]
        
        df['Natural gas'] = data['Natural gas']   # ktep
        df['Electric energy'] = float(data['Electricity'])  # ktep
        df['Biomass'] = float(data['Primary solid biofuels'] + data['Charcoal']) # ktep
        df['LPG'] = float(data['Liquefied petroleum gases'])  # ktep
        df['Gasoline'] = float(data['Gas oil and diesel oil (excluding biofuel portion)']+data['Motor gasoline (excluding biofuel portion)']) #ktep
        df['Other oil products'] = float(data['Oil and petroleum products'] - data['Liquefied petroleum gases'])   # ktep
        df['Renewable heat'] = float(data['Solar thermal'] + data['Geothermal'] + data['Ambient heat (heat pumps)'] + data['Biogases'])
        df['District heating'] = float(data['Heat'])
        df['Year'] = year
        # error_ben = data['Total'] - df.sum(axis = 1)
        
        df2 = pd.concat([df2,df], axis = 0) #df2.append(df)

    df2 = df2.set_index('Year')
        
    return df2


#%%
ben = read_ben()
res = read_results()

#%%





