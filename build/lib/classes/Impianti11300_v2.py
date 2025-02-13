import pandas as pd
import os
import math



def rendimenti(riga_utenza, matrice_emis, matrice_rego, matrice_distri, matrice_gene):
    
    retrofits={
        "PDC":False,
        "GasBoil":False,
        }
    
    ore_acc_matt = riga_utenza['q_3_11A']
    ore_acc_pome = riga_utenza['q_3_11B']
    ore_acc_nott = riga_utenza['q_3_11C']
    # EMISSIONE
    if riga_utenza['q_3_4_3'] == 1:   # Presenza pavimento radiante
        tipo_emettitore = 'Radiante'
        eta_em= matrice_emis.loc['Radiant floor']['Medium Heat Load']
        
    elif riga_utenza['q_3_4_2'] ==1:  # Presenza fan coil
        tipo_emettitore = 'Fancoil'
        eta_em= matrice_emis.loc['Fan Coil']['Medium Heat Load']
        
    elif riga_utenza['q_3_4_1'] ==1:  # Presenza radiatori/termi
        tipo_emettitore = 'Radiatori'
        eta_em= matrice_emis.loc['Radiators EXT']['Medium Heat Load'] 
         
    else:  #caso particolare in cui non si ha nessuno dei 3
        tipo_emettitore = 'Radiatori'
        eta_em= matrice_emis.loc['Radiators INT']['Medium Heat Load'] 
           
         
    Pn = riga_utenza['P_nom_H'] if not math.isnan(riga_utenza['P_nom_H']) else 10. #kW 
    
    # DISTRIBUZIONE
    
    n_em = riga_utenza['q_3_4_1'] + riga_utenza['q_3_4_2'] + riga_utenza['q_3_4_3']
    if riga_utenza['q_2_4_ric'] > 3:
        
        # PRIMA ANNI 90
        
        
        
        if riga_utenza['q_3_4_1'] ==1  or n_em == 0:
            # terminale radiatori
            if riga_utenza['q_2_1'] == 1:
                # villetta unifamiliare
                eta_distri_1= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'detached house  ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiator ']['C_coefficient']
                eta_distri=1-(1-eta_distri_1)*C
                
            elif riga_utenza['q_2_1'] == 2:
                # villetta plurifamiliare
                eta_distri_2= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiator ']['C_coefficient']
                eta_distri=1-(1-eta_distri_2)*C
            
            elif riga_utenza['q_2_1'] in [3,4,5]:
                # condominio
                if riga_utenza['q_3_0'] == 1:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'centralised',  'condominium']['D/E']
                else:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiator ']['C_coefficient']
                eta_distri=1-(1-eta_distri_4)*C
        
        
        # terminale fan coil
        elif riga_utenza['q_3_4_2'] ==1:
            if riga_utenza['q_2_1'] == 1 :
                # villetta unifamiliare
                eta_distri_1= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'detached house  ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Fan coil']['C_coefficient']
                eta_distri=1-(1-eta_distri_1)*C
            
            elif riga_utenza['q_2_1'] == 2:
                # villetta plurifamiliare
                eta_distri_2= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Fan coil']['C_coefficient']
                eta_distri=1-(1-eta_distri_2)*C
            
            elif riga_utenza['q_2_1'] in [3,4,5]:
                # condominio
                if riga_utenza['q_3_0'] == 1:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'centralised',  'condominium']['D/E']
                else:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Fan coil']['C_coefficient']
                eta_distri=1-(1-eta_distri_4)*C
        
        
        # terminale impianto radiante
        elif riga_utenza['q_3_4_3'] ==1:
            if riga_utenza['q_2_1'] == 1:
                # villetta unifamiliare
                eta_distri_1= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'detached house  ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiant system']['C_coefficient']
                eta_distri=1-(1-eta_distri_1)*C
            
            elif riga_utenza['q_2_1'] == 2:
                # villetta plurifamiliare
                eta_distri_2= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiant system']['C_coefficient']
                eta_distri=1-(1-eta_distri_2)*C
            
            elif riga_utenza['q_2_1'] in [3,4,5]:
                # condominio
                if riga_utenza['q_3_0'] == 1:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'centralised',  'condominium']['D/E']
                else:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiant system']['C_coefficient']
                eta_distri=1-(1-eta_distri_4)*C
    
    
    
    
    elif riga_utenza['q_2_4_ric'] <= 3:
        # DOPO ANNI 90
        
        # SE non si dichiarano modalità di diffusione del calore si mette termosifoni
        
        if riga_utenza['q_3_4_1'] ==1  or n_em == 0:
            # terminale radiatori
            
            if riga_utenza['q_2_1'] == 1:
                # villetta unifamiliare
                eta_distri_1= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'detached house  ']['A'] 
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiator ']['C_coefficient']
                eta_distri=1-(1-eta_distri_1)*C
            
            elif riga_utenza['q_2_1'] == 2:
                # villetta plurifamiliare
                eta_distri_2= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['A'] 
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiator ']['C_coefficient']
                eta_distri=1-(1-eta_distri_2)*C
            
            elif riga_utenza['q_2_1'] in [3,4,5]:
                # condominio
                if riga_utenza['q_3_0'] == 1:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'centralised',  'condominium']['A']
                else:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiator ']['C_coefficient']
                eta_distri=1-(1-eta_distri_4)*C
                
        
         # terminale fan coil
        elif riga_utenza['q_3_4_2'] ==1:
            if riga_utenza['q_2_1'] == 1:
                # villetta unifamiliare
                eta_distri_1= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'detached house  ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Fan coil']['C_coefficient']
                eta_distri=1-(1-eta_distri_1)*C
            
            elif riga_utenza['q_2_1'] == 2:
                # villetta plurifamiliare
                eta_distri_2= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Fan coil']['C_coefficient']
                eta_distri=1-(1-eta_distri_2)*C
            
            elif riga_utenza['q_2_1'] in [3,4,5]:
                # condominio
                if riga_utenza['q_3_0'] == 1:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'centralised',  'condominium']['A']
                else:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Fan coil']['C_coefficient']
                eta_distri=1-(1-eta_distri_4)*C
        
        
        # terminale impianto radiante
        elif riga_utenza['q_3_4_3'] ==1:
            if riga_utenza['q_2_1'] == 1 :
                # villetta unifamiliare
                eta_distri_1= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'detached house  ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiant system']['C_coefficient']
                eta_distri=1-(1-eta_distri_1)*C
            
            elif riga_utenza['q_2_1'] == 2  :
                # villetta plurifamiliare
                eta_distri_2= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiant system']['C_coefficient']
                eta_distri=1-(1-eta_distri_2)*C
            
            elif riga_utenza['q_2_1'] in [3,4,5]:
                # condominio
                if riga_utenza['q_3_0'] == 1:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'centralised',  'condominium']['A']
                else:
                    eta_distri_4= matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['A']
                C= matrice_distri.loc['Other temperatures',  'terminal',   'Radiant system']['C_coefficient']
                eta_distri=1-(1-eta_distri_4)*C
    
    else: #caso particolare DA SISTEMARE
        eta_distri= None # matrice_distri.loc['M/R -->80/60°C',  'autonomous',   'multifamily house ']['D/E']
    
    
    # REGOLAZIONE
    # tramite termostati
    # if  riga_utenza['q_2_4_ric'] > 3:
    
    tipo_reg = None    
    
    n_reg = riga_utenza['q_3_13_1'] + riga_utenza['q_3_13_2']+ riga_utenza['q_3_13_3']
    if riga_utenza['q_3_13_1'] == 1:
        tipo_reg = 'Termostato'
        # Con termostato considero prima anni 90
        
        
        
        # PRIMA ANNI 90        
        if (riga_utenza['q_3_4_1'] ==1 or riga_utenza['q_3_4_2'] ==1 or n_em == 0):
            # terminali radiatori o fan coil
            eta_rego= matrice_rego.loc['thermostats',  'BF 90s']['Radiators, fc']
      
         
        elif riga_utenza['q_3_4_3'] ==1:
            # terminali radiante pannello (non accoppiato termicamente)
            eta_rego= matrice_rego.loc['thermostats',  'BF 90s']['Radiant embedded']
           
        #  # terminali radsiante embedded 
        # elif riga_utenza['q_3_4_3'] ==1:
        #    eta_rego= matrice_rego.loc['thermostats',  'BF 90s']['Radiant embedded']
        
       
       
    # DOPO ANNI 90
    # tramite timer   
    # terminali radiatori o fan coil
    
    # if riga_utenza['q_2_4_ric'] <=  3:
    elif  riga_utenza['q_3_13_2'] == 1 or n_reg == 0:
        tipo_reg = 'Cronotermostato'
        # Con cronotermostato considero dopo anni 90
        
        # DOPO ANNI 90       
        if  (riga_utenza['q_3_4_1'] ==1 or riga_utenza['q_3_4_2'] ==1 or n_em == 0):
            eta_rego= matrice_rego.loc['timer',  'AF 90s']['Radiators, fc']
          
         # terminali radiante pannello (non accoppiato termicamente)
        # elif riga_utenza['q_3_4_3'] ==1:
        #    eta_rego= matrice_rego.loc['timer',  'AF 90s']['Radiant panels']
           
         # terminali radiante embedded 
        elif riga_utenza['q_3_4_3'] ==1:
            eta_rego= matrice_rego.loc['timer',  'AF 90s']['Radiant embedded']   
    
    
        
        
    elif riga_utenza['q_3_13_3'] == 1:
        tipo_reg = 'Valvole termostatiche'
        # DOPO ANNI 90
        # tramite valvole termostatiche 
        # terminale radiatore o fan coil
        if  riga_utenza['q_3_19_5'] == 1:
            if (riga_utenza['q_3_4_1'] ==1 or riga_utenza['q_3_4_2'] ==1 or n_em == 0):
               eta_rego= matrice_rego.loc['thermostatic valve',  'retrofit']['Radiators, fc']
            
            # # terminali radiante pannello (non accoppiato termicamente)
            # elif riga_utenza['q_3_4_3'] ==1:
            #    eta_rego= matrice_rego.loc['thermostatic valve',  'AF 90s']['Radiant panels']
            
            # terminali radiante embedded
            elif riga_utenza['q_3_4_3'] ==1:
               eta_rego= matrice_rego.loc['thermostatic valve',  'retrofit']['Radiant embedded']
               
        else:
            if (riga_utenza['q_3_4_1'] ==1 or riga_utenza['q_3_4_2'] ==1 or n_em == 0):
               eta_rego= matrice_rego.loc['thermostatic valve',  'BF 90s']['Radiators, fc']
            
            # # terminali radiante pannello (non accoppiato termicamente)
            # elif riga_utenza['q_3_4_3'] ==1:
            #    eta_rego= matrice_rego.loc['thermostatic valve',  'AF 90s']['Radiant panels']
            
            # terminali radiante embedded
            elif riga_utenza['q_3_4_3'] ==1:
               eta_rego= matrice_rego.loc['thermostatic valve',  'BF 90s']['Radiant embedded']
    
    # else: #caso particolare DA SISTEMARE
    #      eta_rego = None # matrice_rego.loc['timer',  'AF 90s']['Radiant embedded']
       
    else:
        eta_rego = None
    
    
  

  # GENERAZIONE 
    # con caldaia, combustibile fossile e impianto centralizzato per più abitazioni
    pdc = None
    if riga_utenza['q_3_0'] == 1:
        if riga_utenza['q_2_1'] == 3: Pn = Pn*5 # Ipotizzati 5 Appartamenti
        elif riga_utenza['q_2_1'] == 4: Pn = Pn*15 # Ipotizzati 15 Appartamenti
        elif riga_utenza['q_2_1'] == 5: Pn = Pn*30 # Ipotizzati 30 Appartamenti
        
    anno_impianto = riga_utenza['q_2_4_ric']
    tipo_fuel = riga_utenza['q_3_1']
    fuel = None
    
    if retrofits['PDC'] and (tipo_fuel in [1,2,3,5,0,9,7]):
        tipo_fuel = 4
    
    if (tipo_fuel in [1,2,3,5,0,9,7]):
        
        fuel = {
        1 : 'NaturalGas',
        2 : 'Gasoline',
        3 : 'LPG',
        5 : 'Oil',
        0 : 'FissoPortatile',
        9 : 'DH',
        7 : 'Coke'
            }[tipo_fuel]     
        
        
        
        if retrofits['GasBoil']:
            print('Fuel boilers substitution to recent gas boilers')
            # Viene fatto per tutte le caldaie ad: olio, carbone, gpl, gasolio, gas
            # con caldaie a gas
            anno_impianto = 1
            fuel = 'NaturalGas'
            
            
        if  anno_impianto >= 4:
            c1= matrice_gene.loc['fossil fuel',  'BF 78']['c1']   # prima del 78
            c2= matrice_gene.loc['fossil fuel',  'BF 78']['c2']
           
            eta_gene= (c1+c2*math.log(Pn,10))/100       
           
        elif  anno_impianto == 3:
            c1= matrice_gene.loc['fossil fuel',  '78 TO 86']['c1']   # tra 78 e 86
            c2= matrice_gene.loc['fossil fuel',  '78 TO 86']['c2']
           
            eta_gene= (c1+c2*math.log(Pn,10))/100
           
        elif  anno_impianto == 2 :
            c1= matrice_gene.loc['fossil fuel',  '87 TO 94']['c1']   # tra 87 e 94
            c2= matrice_gene.loc['fossil fuel',  '87 TO 94']['c2']
           
            eta_gene= (c1+c2*math.log(Pn,10))/100
           
           
        elif  anno_impianto == 1 :
            c1= matrice_gene.loc['fossil fuel',  'AF 94']['c1']   # dopo 94
            c2= matrice_gene.loc['fossil fuel',  'AF 94']['c2']
           
            eta_gene = (c1+c2*math.log(Pn,10))/100   
       
        if fuel == 'FissoPortatile':
            # caso portatili o fissi (Veramente pochi hanno risposto a questo)
            if riga_utenza['q_3_5'] in [3,4]:
                fuel = 'Biomass'
            elif riga_utenza['q_3_5'] in [1,2]:
                fuel = 'Electric' 
                pdc = 'PDC aria'
                eta_gene = 1
            elif riga_utenza['q_3_5'] in [5]:
                fuel = 'NaturalGas' 
            elif riga_utenza['q_3_5'] in [6]:
                fuel = 'LPG'                
            elif riga_utenza['q_3_6'] in [1,2]:
                fuel = 'Electric' 
                pdc = 'PDC aria'
                eta_gene = 1
            elif riga_utenza['q_3_6'] in [3]:
                fuel = 'LPG'
            else:
                # Caso in cui ci sia un solo apparecchio secondario usato prevalentemente                          
                if (riga_utenza['q_2_32C'] == 1) or (riga_utenza['q_2_32D'] == 1):
                    fuel = 'Biomass'
                elif (riga_utenza['q_2_32A'] == 1) or (riga_utenza['q_2_32B'] == 1) or (riga_utenza['q_2_33A'] == 1) or (riga_utenza['q_2_33B'] == 1):
                    fuel = 'Electric' 
                    pdc = 'PDC aria'
                    eta_gene = 1
                elif (riga_utenza['q_2_32E'] == 1):
                    fuel = 'NaturalGas' 
                elif (riga_utenza['q_2_32F'] == 1) or (riga_utenza['q_2_33C'] == 1):
                    fuel = 'LPG' 
                else:
                    fuel = 'NoAnsw'  
                    
        
    
    
    elif (tipo_fuel in [6]) or fuel == 'Biomass':   
        
        fuel = {
        6 : 'Biomass'         
            }[tipo_fuel]
        # Apparecchio singolo
        # biomassa (camino)
        if riga_utenza['q_3_5'] == 4:
           
            eta_gene = 0.5  #rendimento base preso da 11300-4
            
        # biomassa (stufa)
        
        else:
            if  anno_impianto >= 4:
                c1= matrice_gene.loc['biomass',  'BF 78']['c1']   # prima del 78
                c2= matrice_gene.loc['biomass',  'BF 78']['c2']
                                                                              
                eta_gene= (c1+c2*math.log(Pn,10))/100    
            
            elif  anno_impianto in [2, 3]:
                 c1= matrice_gene.loc['biomass',  '78 TO 94']['c1']   # tra 78 e 94
                 c2= matrice_gene.loc['biomass',  '78 TO 94']['c2']
                                                                              
                 eta_gene= (c1+c2*math.log(Pn,10))/100  
            
            elif  anno_impianto == 1:
                c1= matrice_gene.loc['biomass',  'AF 94']['c1']   # dopo 94
                c2= matrice_gene.loc['biomass',  'AF 94']['c2']
                
                eta_gene= (c1+c2*math.log(Pn,10))/100 
    
            if riga_utenza['q_3_19_1'] == 1:
                c1= matrice_gene.loc['biomass',  'AF 94']['c1']   # dopo 94
                c2= matrice_gene.loc['biomass',  'AF 94']['c2']
                
                eta_gene= (c1+c2*math.log(Pn,10))/100     
                   
        # eventuale sostituzione caldaia negli ultimi 5 anni   
        
    elif tipo_fuel == 4:
        eta_gene = 1
        fuel = {
        4 : 'Electric'         
            }[tipo_fuel]
        
        
        if riga_utenza["q_3_2"] == 1:
                pdc = {
                    1 : 'PDC aria',
                    2 : 'PDC acqua falda',
                    3 : 'PDC acqua superficiale',
                    4 : 'PDC terreno',
                    9 : 'PDC aria'
                    }[riga_utenza['q_3_3']]
                
        if retrofits['PDC']:
            pdc = 'PDC aria'
            print('Fuel boilers substitution to recent gas boilers')
                
    # else:
    #     eta_gene = None
    #     fuel = None
        
    if fuel == 'Biomass' and riga_utenza['q_6_13_ric'] == 0:
        biomass = 'Wood'
    elif fuel == 'Biomass':
        biomass = 'Pellets'
    else:
        biomass = ''
        
    if fuel == 'DH':
        eta_gene = 1.
        
    
    pdc_acs = None
    if riga_utenza["q_4_3"] == 1:
        pdc_acs = {
            1 : 'PDC aria',
            2 : 'PDC acqua falda',
            3 : 'PDC acqua superficiale',
            4 : 'PDC terreno',
            9 : 'PDC aria'
            }[riga_utenza['q_4_4']]
        
    tot = eta_em * eta_distri * eta_rego * eta_gene
    
    tipo_impianto = {0:'NoRisc',
                     1:'Centralizzato',
                     2:'Autonomo',
                     3:'Singoli',
                     4:'Singoli'
                     }[riga_utenza['q_3_0']]
    
    
    # aggiungo i secondari
    fuel_sec = None
    eta_gene_sec = None
    pdc_sec = None
    if riga_utenza['q_3_0'] in [1,2]:
        # Caso apparecchio autonomo o centralizzato a combustibile fossile con impianto secondario
        if riga_utenza['q_2_32C'] == 1 or riga_utenza['q_2_32D'] == 1:
            fuel_sec = 'Wood'
            c1= matrice_gene.loc['biomass',  'AF 94']['c1']   # dopo 94
            c2= matrice_gene.loc['biomass',  'AF 94']['c2']
            eta_gene_sec = (c1+c2*math.log(Pn,10))/100 
            
        elif riga_utenza['q_2_32A'] == 1 or riga_utenza['q_2_32B'] == 1 or riga_utenza['q_2_33A'] == 1:
            fuel_sec = 'Electric'
            pdc_sec = 'PDC aria'
            eta_gene_sec = 1
            
        elif riga_utenza['q_2_32E'] == 1:
            fuel_sec = 'NaturalGas'
            c1= matrice_gene.loc['fossil fuel',  'AF 94']['c1']   # dopo 94
            c2= matrice_gene.loc['fossil fuel',  'AF 94']['c2']
            eta_gene_sec = (c1+c2*math.log(Pn,10))/100  
            
        elif riga_utenza['q_2_32F'] == 1 or riga_utenza['q_2_33C'] == 1:
            fuel_sec = 'LPG'
            c1= matrice_gene.loc['fossil fuel',  'AF 94']['c1']   # dopo 94
            c2= matrice_gene.loc['fossil fuel',  'AF 94']['c2']
            eta_gene_sec = (c1+c2*math.log(Pn,10))/100
            
        elif riga_utenza['q_2_33B'] == 1:
            fuel_sec = 'Electric'
            pdc_sec = 'Resistenza'
            eta_gene_sec = 1   
    
    return tipo_emettitore, tipo_reg, ore_acc_matt, ore_acc_pome, ore_acc_nott, eta_em, eta_distri, eta_rego, eta_gene, tot, tipo_impianto, fuel, biomass, pdc, pdc_acs,  eta_gene_sec, fuel_sec, pdc_sec


def rendimenti_acs(riga_utenza, matrice_ero_acs, matrice_gen_acs, matrice_acc_acs):
    #### ACS
    
    eta_ero = None
    eta_distr = None
    eta_gen = None
    fuel = None
    pdc = None
    
    # erogazione 11300
    eta_erog_acs = matrice_ero_acs.iloc[0,0]   # changed from matrice_ero_acs.loc['Erogazione'][0] 
    # perdite_acc_acs = matrice_acc_acs.loc['Erogazione'][0] *365 # kWh/y
    
    
    if riga_utenza['dotazione_ACS'] == 'NO': 
        tot = None # caso no ACS
        
        
    else:
        if riga_utenza['Impianto'] == 'Singoli':
            # Caso impianto autonomo
            eta_ero =  eta_erog_acs
            eta_distr = 1. # mettiamo trascurabile
            
            if riga_utenza['AccumuloIstantaneo'] == 'Accumulo':
                if riga_utenza['Fuel'] in ['NaturalGas','LPG','NoAnsw','Solar'] :
                    eta_gen = matrice_gen_acs.loc['Gas accumulo ACS','B senza pilota']['Stag']
                    fuel = 'LPG' if riga_utenza['Fuel'] == 'LPG' else 'NaturalGas'
                elif riga_utenza['Fuel'] == 'Electric':
                    eta_gen = matrice_gen_acs.loc['Elettrico','-']['Stag']
                    fuel = 'Electric'
                else:
                    if riga_utenza['Anno'] in ['20 years','10-20 years']:
                        eta_gen = matrice_gen_acs.loc['Fuoco diretto','Camera aperta']['Stag']
                    else: 
                        eta_gen = matrice_gen_acs.loc['Fuoco diretto','Condensazione']['Stag']
                    fuel = riga_utenza['Fuel']
            else:
                if riga_utenza['Fuel'] in ['NaturalGas','LPG','NoAnsw','Solar'] :
                    eta_gen = matrice_gen_acs.loc['Gas accumulo ACS','B senza pilota']['Ist']
                    fuel = 'LPG' if riga_utenza['Fuel'] == 'LPG' else 'NaturalGas'
                elif riga_utenza['Fuel'] == 'Electric':
                    eta_gen = matrice_gen_acs.loc['Elettrico','-']['Ist']
                    fuel = 'Electric'
                else:
                    if riga_utenza['Anno'] in ['20 years','10-20 years']:
                        eta_gen = matrice_gen_acs.loc['Fuoco diretto','Camera aperta']['Ist']
                    else: 
                        eta_gen = matrice_gen_acs.loc['Fuoco diretto','Condensazione']['Ist']
                    fuel = riga_utenza['Fuel']
        
        elif riga_utenza['Impianto'] in ['Centralizzato','Autonomo']:
            # Caso Impianto centralizzato e autonomo dovrebbero essere uguali
            eta_ero =  eta_erog_acs
            eta_distr = riga_utenza['Rendimento dist riscaldamento']
            # accumulo non c'è mai
            if riga_utenza['UgualeRisc'] == 'SI':
                eta_gen = riga_utenza['Rendimento gen riscaldamento']
                fuel = riga_utenza['Fuel riscaldamento']
            else:
                if riga_utenza['Fuel'] in ['DH'] and riga_utenza['q_4_2'] == 9:
                    eta_gen = 1
                    fuel = 'DH'
                elif riga_utenza['Fuel'] in ['NaturalGas','LPG','NoAnsw','DH','Solar'] :
                    eta_gen = matrice_gen_acs.loc['Gas accumulo ACS','B senza pilota']['Stag']
                    fuel = 'LPG' if riga_utenza['Fuel'] == 'LPG' else 'NaturalGas'
                elif riga_utenza['Fuel'] == 'Electric':
                    eta_gen = matrice_gen_acs.loc['Elettrico','-']['Stag']
                    fuel = 'Electric'
                else:
                    if riga_utenza['Anno'] in ['20 years','10-20 years']:
                        eta_gen = matrice_gen_acs.loc['Fuoco diretto','Camera aperta']['Stag']
                    else: 
                        eta_gen = matrice_gen_acs.loc['Fuoco diretto','Condensazione']['Stag']
                    fuel = riga_utenza['Fuel']
            
            
        
        if riga_utenza['PDC'] != None:
            eta_gen = 1.
            
        tot = eta_ero * eta_distr * eta_gen
    
    return eta_ero, eta_distr, eta_gen, tot, fuel, riga_utenza['UgualeRisc']
    
    
def crea_impianti(dataset, file_matrici, info_acs):
    # in_file = os.path.join('..','Input','DatiIstat','istat_microdata_csv.csv')
    # dataset = pd.read_csv(in_file, delimiter = ';', index_col = 0)
    
    
    #matrice emissione
    matrice_emis=pd.read_excel(file_matrici,sheet_name='Emission_eta',skiprows = 3, index_col = 0,usecols = "A:B" )
     
    #matrice regolazione
    matrice_rego=pd.read_excel(file_matrici,sheet_name='Regulation_eta',skiprows = 3, index_col = [0,1] ,usecols = "A:E" )
   
    #matrice distribuzione
    matrice_distri=pd.read_excel(file_matrici,sheet_name='Distribution_eta',skiprows = 3, index_col = [0,1,2],usecols = "A:F" )
    
    #matrice generazione
    matrice_gene=pd.read_excel(file_matrici,sheet_name='Generation_eta',skiprows = 3, index_col = [0,1],usecols = "A:D" )
    
    # Pn = pd.read_excel(file_potenze, index_col = 'id')
    dataset['istat']['P_nom_H'] = 20. # Pn['P riscaldamento unita [kW]']
    
    
    df_rendimenti = pd.DataFrame(index = dataset['istat'].index, columns = ['emissione','distribuzione','regolazione','generazione'])
    df_rendimenti[['TipoEmettitore', 'TipoRegolazione',  'OreMatt','OrePome','OreNott','emissione','distribuzione','regolazione','generazione', 'Totale', 'TipoImpianto','Fuel', 'Biomass', 'PDC','PDC acs', 'generazione_secondario', 'fuel_secondario', 'tipo_elettrico_secondario']] = dataset['istat'].apply(rendimenti, matrice_emis = matrice_emis, matrice_rego = matrice_rego, matrice_distri = matrice_distri, matrice_gene = matrice_gene, axis = 'columns',result_type ='expand')
    
    ##### ACS
    # info_acs = pd.read_excel(info_acs, header = 0, index_col = 'id')
    matrice_ero_acs=pd.read_excel(file_matrici,sheet_name='ACS',skiprows = 2, nrows = 1, index_col = [0],usecols = "A:B" )
    matrice_acc_acs=pd.read_excel(file_matrici,sheet_name='ACS',skiprows = 15, nrows = 1, index_col = [0],usecols = "A:B" )
    matrice_gen_acs=pd.read_excel(file_matrici,sheet_name='ACS',skiprows = 5, nrows = 9, index_col = [0,1],usecols = "A:D")
    #info_acs['DHW_dem'] = fabb_acs['DHW 11300 (kWh/y)']
    info_acs['Rendimento dist riscaldamento'] = df_rendimenti['distribuzione']
    info_acs['Rendimento gen riscaldamento'] = df_rendimenti['generazione']
    info_acs['PDC'] = df_rendimenti['PDC']
    info_acs['Fuel riscaldamento'] = df_rendimenti['Fuel']
    df_rendimenti[['erogazione acs','distribuzione acs','generazione acs','Totale acs', 'Fuel acs', 'StessoRisc']] = info_acs.apply(rendimenti_acs, matrice_ero_acs = matrice_ero_acs, matrice_gen_acs = matrice_gen_acs, matrice_acc_acs = matrice_acc_acs, axis = 'columns',result_type ='expand')  
    
    return df_rendimenti # .to_excel(os.path.join(outpath,'rendimenti_risc_acs.xlsx'))


##%%
if __name__ == '__main__':
    # in_file = os.path.join('..','Input','DatiIstat','istat_microdata_csv.csv')
    # dataset = pd.read_csv(in_file, delimiter = ';', index_col = 0)
    
    
    # file_matrici = os.path.join('..','Input','Impianti','Impianti.xlsx')
    
    # #matrice emissione
    # matrice_emis=pd.read_excel(file_matrici,sheet_name='Emission_eta',skiprows = 3, index_col = 0,usecols = "A:B" )
     
    # #matrice regolazione
    # matrice_rego=pd.read_excel(file_matrici,sheet_name='Regulation_eta',skiprows = 3, index_col = [0,1] ,usecols = "A:E" )
   
    # #matrice distribuzione
    # matrice_distri=pd.read_excel(file_matrici,sheet_name='Distribution_eta',skiprows = 3, index_col = [0,1,2],usecols = "A:F" )
    
    # #matrice generazione
    # matrice_gene=pd.read_excel(file_matrici,sheet_name='Generation_eta',skiprows = 3, index_col = [0,1],usecols = "A:D" )
    
    # Pn = pd.read_csv(os.path.join('..','Input','RisIntermedi','P_nom.csv'), index_col = 0)
    # dataset['P_nom_H'] = Pn['P riscaldamento [kW]']
    
    
    # df_rendimenti = pd.DataFrame(index = dataset.index, columns = ['emissione','distribuzione','regolazione','generazione'])
    # df_rendimenti[['emissione','distribuzione','regolazione','generazione', 'Totale', 'Fuel', 'PDC']] = dataset.apply(rendimenti, matrice_emis = matrice_emis, matrice_rego = matrice_rego, matrice_distri = matrice_distri, matrice_gene = matrice_gene, axis = 'columns',result_type ='expand')
    
    # df_rendimenti.to_csv(os.path.join('..','Input','RisIntermedi','rendimenti_risc.csv'))
    # eta_em_lista = []
    # eta_distri_lista = []
    # eta_rego_lista = []
    # eta_gene_lista = []
    # for i in range(1,20001):
    #     eta_em, eta_distri, eta_rego, eta_gene = rendimenti(dataset.loc[i],matrice_emis, matrice_rego, matrice_distri, matrice_gene)
    #     eta_em_lista.append(eta_em)
    #     eta_distri_lista.append(eta_distri)
    #     eta_rego_lista.append(eta_rego)
    #     eta_gene_lista.append(eta_gene)
        
    in_file = os.path.join('..','Input','DatiIstat','istat_microdata_csv.csv')
    dataset = pd.read_csv(in_file, delimiter = ';', index_col = 0)
    file_matrici = os.path.join('..','Input','Impianti','Impianti.xlsx')
    file_potenze = os.path.join('..','Input','RisIntermedi','P_nom.csv')
    fabb_acs = pd.read_csv(os.path.join('..','Output_tot','consumption.csv') ,index_col = [0])
    info_acs = os.path.join('..','Input','Acs_info.xlsx')
    outpath = os.path.join('..','Input','Impianti')
    
    crea_impianti(dataset, file_matrici, file_potenze, fabb_acs, info_acs, outpath)