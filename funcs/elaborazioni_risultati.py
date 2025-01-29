# -*- coding: utf-8 -*-

"""
File con funzioni di elaborazione dei dati
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from classes.combustibili import fattori_EP
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



global conversione 
conversione = 11630000 # kWh/ktep

def validazione_locale_costi(costi_classi, output_path = '.\Output', pfix = ''):
    # Esegue la validazione delle classi di costo come da tesi
    lista_fuels = ['Electric','NaturalGas','Gasoline','LPG','Wood','Pellets']
    
    # somme_cum = pd.DataFrame(columns = lista_fuels, index = range(12))
    # somme_cum_neg = pd.DataFrame(columns = lista_fuels, index = range(12))
    somme_cum = {}
    somme_cum_neg = {}
    
    somme_cum_df = pd.DataFrame(index = range(10), columns = lista_fuels)
    somme_cum_neg_df = pd.DataFrame(index = range(-10,10), columns = lista_fuels)
    for i in lista_fuels:
        
        filt = ~(np.isnan(costi_classi[i + ' classe sondaggio']) & (costi_classi[i + ' classe calcolata'] == 1 ))      
        
        diff_classi = abs(costi_classi[i + ' classe calcolata'] - costi_classi[i + ' classe sondaggio'])[filt]
        conteggio=(diff_classi.value_counts().sort_index()/len(diff_classi)*100).cumsum()
        diff_classi_ass = (costi_classi[i + ' classe calcolata'] - costi_classi[i + ' classe sondaggio'])[filt]
        conteggio_ass =(diff_classi_ass.value_counts().sort_index()/len(diff_classi)*100)
                
        # conteggio = conteggio.fillna(value = 0) if  
        somme_cum[i] = conteggio
        somme_cum_neg[i] = conteggio_ass
        somme_cum_df[i] = conteggio
        somme_cum_neg_df[i] = conteggio_ass
    
    
    with pd.ExcelWriter(path = os.path.join(output_path,'Validazione','ValidazioneLocaleDati.xlsx'),  mode = 'w')  as writer:
        for key in lista_fuels:
            somme_cum[i].to_excel(writer, sheet_name=f'{key}')
            somme_cum_neg[i].to_excel(writer, sheet_name=f'{key}_neg')
            
        somme_cum_df.to_excel(writer, sheet_name=f'Cumulati')
        somme_cum_neg_df.to_excel(writer, sheet_name=f'CumulatiAssoluti')
    
    fontsize = 16
    xpad = 10.
    layout = {'hspace':0.27, 'wspace': 0.05, 'top': 0.95, 'bottom': 0.08, 'left': 0.07, 'right' : 0.97}
    sharey = True
    fig, [[ax11,ax12, ax13],[ax21,ax22, ax23]] = plt.subplots(ncols = 3, nrows = 2, figsize = (15,10), gridspec_kw = layout, sharey=sharey)
    plots = {'Electric': ax11,
             'NaturalGas': ax12,
             'Gasoline': ax13,
             'LPG': ax21,
             'Wood': ax22,
             'Pellets': ax23}  
    
    fig3, [[ax311,ax312, ax313],[ax321,ax322, ax323]] = plt.subplots(ncols = 3, nrows = 2, figsize = (15,10),gridspec_kw = layout, sharey=sharey)
    plots3 = {'Electric': ax311,
             'NaturalGas': ax312,
             'Gasoline': ax313,
             'LPG': ax321,
             'Wood': ax322,
             'Pellets': ax323}  
        
    for i in lista_fuels:
        title = 'En. Elettrica'
        if i == 'NaturalGas': title = 'Gas naturale'
        if i == 'Gasoline': title = 'Gasolio'
        if i == 'LPG': title = 'GPL'
        if i == 'Wood': title = 'Legna'
        if i == 'Pellets': title = 'Pellets'  
        
        
        ax = plots[i]
        ax3 = plots3[i]
        try:
            somme_cum[i].plot(kind = 'bar',ax = ax)
            somme_cum_neg[i].plot(kind = 'bar',ax = ax3)
        except Exception:
            pass
        
        ax.set_xlabel('Errore su classe di spesa', fontsize = fontsize, labelpad = xpad)
        ax.set_ylabel('Copertura', fontsize = fontsize)
        ax.set_title(title, size = fontsize + 4)
        ax.set_ylim([0,110])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        y_labels = ax.get_yticks()
        if i == 'NaturalGas': ax.set_xlim([0-0.5,9+0.5])
        ax.set_yticklabels([f'{(y):.0f} %' for y in y_labels])   
        x_labels = ax.get_xticks()
        ax.set_xticklabels([f'\u00B1{(x):.0f}' for x in x_labels], rotation=0)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.grid(linewidth = 0.2)
        
        ax3.set_xlabel('Errore su classe di spesa', fontsize = fontsize, labelpad = xpad)
        ax3.set_ylabel('Copertura', fontsize = fontsize)
        ax3.set_title(title, size = fontsize + 4)
        ax3.set_ylim([0,35])
        if i == 'NaturalGas': ax3.set_xlim([2-0.5,20+0.5])
        x_labels = ax3.get_xticks()
        x_mean = np.round(np.array(x_labels).mean())
        ax3.set_xticklabels([f'{(x-x_mean):.0f}' for x in x_labels], rotation=90)
        y_labels = ax3.get_yticks()
        ax3.set_yticklabels([f'{(y):.0f} %' for y in y_labels])  
        
        ax3.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax3.grid(linewidth = 0.2)
    
    if not os.path.isdir(os.path.join(output_path,'Validazione')):
        os.mkdir(os.path.join(output_path,'Validazione'))
    
    fig.savefig(os.path.join(output_path,'Validazione',pfix + 'ValidazioneLocale.png'))
    plt.close()
    fig3.savefig(os.path.join(output_path,'Validazione',pfix + 'ValidazioneLocale_ass.png'))
    plt.close()
    lista_fuels = ['Electric','NaturalGas','Gasoline','LPG','Wood','Pellets']
    
    layout = {'hspace':0.27, 'wspace': 0.27, 'top': 0.95, 'bottom': 0.08, 'left': 0.07, 'right' : 0.97}
       
    fig2, [[ax211,ax212, ax213],[ax221,ax222, ax223]] = plt.subplots(ncols = 3, nrows = 2, figsize = (15,10), gridspec_kw = layout, sharey = False)
    plots = {'Electric': ax211,
             'NaturalGas': ax212,
             'Gasoline': ax213,
             'LPG': ax221,
             'Wood': ax222,
             'Pellets': ax223}    
    
    
    for i in lista_fuels:
        # diff_classi = abs(costi_classi[i + ' classe calcolata'] - costi_classi[i + ' classe sondaggio'])[costi_classi[i + ' classe sondaggio']!=0.]
        # conteggio=(diff_classi.value_counts().sort_index()/len(diff_classi)*100).cumsum()
        if i in ['Wood', 'Pellets']:
            labels = [i + ' kg',i + ' kg sondaggio']
        else:
            labels = [i + ' euro',i + ' euro sondaggio']
        
        title = 'En. Elettrica'; xlabel = 'Spesa calcolata [€]'; ylabel = 'Spesa dichiarata [€]'; 
        if i == 'NaturalGas': title = 'Gas naturale'
        if i == 'Gasoline': title = 'Gasolio'
        if i == 'LPG': title = 'GPL'
        if i == 'Wood': title = 'Legna'; xlabel = 'Quantità calcolata [kg]'; ylabel = 'Quantità dichiarata [kg]'; 
        if i == 'Pellets': title = 'Pellets'  ; xlabel = 'Quantità calcolata [kg]'; ylabel = 'Quantità dichiarata [kg]'; 
        
        x = costi_classi[labels[0]].replace(np.nan,0)
        y = costi_classi[labels[1]].replace(np.nan,0)
        
        
        
        axis_limits = np.array([0, x.max()])    
        if i == 'Wood' or i == 'Pellets': axis_limits = np.array([0, 8000])    
        ax = plots[i]
        costi_classi.plot(kind = 'scatter',x = labels[0], y = labels[1], ax = ax, s=0.1)
        regr = linear_model.LinearRegression()
        regr.fit(x.values.reshape(-1,1), y.values)
        y_linea_regr = regr.predict(axis_limits.reshape(-1,1))
        textstr = f'Retta di regressione:\ny = {regr.coef_[0]:.2f}x + {regr.intercept_:.2f}' 
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        axis_limits = [0,3000] 
        if i == 'NaturalGas': axis_limits = [0,4000]
        if i == 'Gasoline': axis_limits = [0,5000]
        if i == 'LPG': axis_limits = [0,4000]
        if i == 'Wood': axis_limits = [0,8000]
        if i == 'Pellets': axis_limits = [0,8000]
        
        
        ax.plot(axis_limits,axis_limits,'r-')
        ax.plot(axis_limits,y_linea_regr,'r:')
        ax.set_xlabel(xlabel, fontsize = fontsize)
        ax.set_ylabel(ylabel, fontsize = fontsize)
        ax.set_title(title, size = fontsize + 4)
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize = fontsize-4, bbox=props)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.grid(linewidth = 0.2)
        
    
    fig2.savefig(os.path.join(output_path,'Validazione',pfix + 'ScatterPlotCombustibili.png'))
    plt.close()
    return somme_cum, somme_cum_neg
    
def validazione_globale(info_ed, consumptions, output_path = '.\Output', nome_caso = 'Base'):
    consumption_tot = consumptions.apply(lambda x: np.asarray(x) * np.asarray(info_ed['RiportGlobale']))   
    
    tot = consumption_tot.sum()
    
    ref = pd.Series({'Electric (GWh/y)':  61379, 
                                   'NaturalGas (ktep)': 18073,
                                   'LPG (ktep)':1193,
                                   'Gasoline (ktep)':1511,
                                   'Biomass (ktep)':6633,
                                   'Oil (ktep)':0.98}, name = "Ref nazionale")
    
    
    totale_nazionale = pd.Series({'Electric (GWh/y)':  tot['Electric (kWh/y)']/1e6, # conversione kWh -> GWh
                                   'NaturalGas (ktep)': tot['NaturalGas (kWh/y)']/conversione, # conversione kWh -> tep
                                   'LPG (ktep)':        tot['LPG (kWh/y)']/conversione, # conversione kWh -> tep
                                   'Gasoline (ktep)':   tot['Gasoline (kWh/y)']/conversione, # conversione kWh -> tep
                                   'Biomass (ktep)':    (tot['Wood (kWh/y)'] + tot['Pellets (kWh/y)'])/conversione, # conversione kWh -> tep
                                   'Oil (ktep)':        tot['Oil (kWh/y)']/conversione}, # conversione kWh -> tep
                                 name = f"Sim {len(info_ed.index)} edifici")
    
    totale_nazionale.drop('Oil (ktep)')
    
    errore = pd.Series( (totale_nazionale/ref-1)*100,dtype = int,name = "Errore %")
    
    if not os.path.isdir(os.path.join(output_path,'Validazione')):
        os.mkdir(os.path.join(output_path,'Validazione'))
        
    globale = pd.concat([totale_nazionale, ref, errore], axis=1)
    with pd.ExcelWriter(path = os.path.join(output_path,'Validazione','ValidazioneGlobale.xlsx'), mode = 'w')  as writer:
        globale.to_excel(writer, sheet_name=nome_caso)
    
    return globale

def tabella_usi_finali(info_edificio, consumi_elettrodomestici, consumi_impianti, output_path = '.\Output'):
    usi = ['Illuminazione', 'FigoriferiCongelatori', 'GrandiElettrodomestici','PiccoliElettrodomestici',
           'SchermiPC', 'Cottura', 'Raffrescamento', 'Riscaldamento', 'ACS']
    combustibili = ['Electric (kWh/y)', 'NaturalGas (kWh/y)', 'Gasoline (kWh/y)', 'LPG (kWh/y)', 'Biomass (kWh/y)', 'Coke (kWh/y)', 'Oil (kWh/y)']
    
    consumi_uso_finale = pd.DataFrame(index = combustibili , columns = usi)
    
    consumi_uso_finale.loc['Electric (kWh/y)']['Illuminazione'] = np.sum(consumi_elettrodomestici['Lights Demand (kWh/y)'] * info_edificio['RiportGlobale'])
    consumi_uso_finale.loc['Electric (kWh/y)']['FigoriferiCongelatori'] = np.sum(consumi_elettrodomestici['Refrigerators Demand MOIRAE (kWh/y)'] * info_edificio['RiportGlobale'])
    consumi_uso_finale.loc['Electric (kWh/y)']['GrandiElettrodomestici'] = np.sum(consumi_elettrodomestici['Big Appliances Demand MOIRAE (kWh/y)'] * info_edificio['RiportGlobale'])
    consumi_uso_finale.loc['Electric (kWh/y)']['PiccoliElettrodomestici'] = np.sum((consumi_elettrodomestici['Little Appliances Demand (kWh/y)']  + consumi_elettrodomestici['Standby appliances (kWh/y)']) * info_edificio['RiportGlobale'])
    consumi_uso_finale.loc['Electric (kWh/y)']['SchermiPC'] = np.sum(consumi_elettrodomestici['TVs & PCs Demand (kWh/y)'] * info_edificio['RiportGlobale'])
    
    for fuel, fuel_ in zip(['Electric', 'NaturalGas', 'LPG', 'Biomass'],['Electric', 'Gas', 'GPL', 'Biomass']) :        
        consumi_uso_finale.loc[fuel + ' (kWh/y)']['Cottura'] = np.sum(consumi_elettrodomestici[fuel_ + ' Cookings Consumption (kWh/y)'] *info_edificio['RiportGlobale'])\
                                                                + np.sum(consumi_elettrodomestici[fuel_ + ' Ovens Consumption (kWh/y)'] * info_edificio['RiportGlobale'])
    
    
    consumi_impianti['Fuel H'].replace({'NoAnsw':'NaturalGas', 'Wood':'Biomass', 'Pellets':'Biomass'},inplace = True)
    consumi_impianti['Fuel H secondario'].replace({'Wood':'Biomass', 'Pellets':'Biomass'},inplace = True)
    consumi_impianti['Fuel acs'].replace({'NoAnsw':'NaturalGas', 'Wood':'Biomass', 'Pellets':'Biomass'},inplace = True)
    
    for fuel in ['Electric', 'NaturalGas', 'Coke', 'LPG', 'Biomass', 'Oil', 'Gasoline']:
        risc = consumi_impianti[consumi_impianti['Fuel H'] == fuel]['Heating Tot'] * info_edificio['RiportGlobale']
        risc_sec = consumi_impianti[consumi_impianti['Fuel H secondario'] == fuel]['SecondarioHeating'] * info_edificio['RiportGlobale']
        risc_acs = consumi_impianti[consumi_impianti['Fuel acs'] == fuel]['Acs Tot'] * info_edificio['RiportGlobale']
        
        consumi_uso_finale.loc[fuel + ' (kWh/y)']['Riscaldamento'] = np.sum(risc) + np.sum(risc_sec)
        consumi_uso_finale.loc[fuel + ' (kWh/y)']['ACS'] = np.sum(risc_acs)
        
    consumi_uso_finale.loc['Electric (kWh/y)']['Raffrescamento'] = np.sum(consumi_impianti['Cooling Tot'] * info_edificio['RiportGlobale'])
    consumi_uso_finale = consumi_uso_finale.fillna(0.).transpose()
    consumi_uso_finale.loc['Tot'] = consumi_uso_finale.sum(axis = 0)
    
    
    
    consumi_uso_finale['Electric (GWh/y)'] = consumi_uso_finale['Electric (kWh/y)'] / 1e6
    for fuel in [ 'NaturalGas', 'Coke', 'LPG', 'Gasoline', 'Biomass', 'Oil']:
        consumi_uso_finale[fuel + ' (ktep/y)'] = consumi_uso_finale[fuel + ' (kWh/y)'] / conversione
    
    moirae = pd.read_excel(os.path.join('.','Resources','RisultatiMOIRAE','UsiFinali.xlsx'), index_col = 0, header = 0)
    
    with pd.ExcelWriter(path = os.path.join(output_path,'Validazione','TabellaUsiFinali.xlsx'), mode = 'w')  as writer:
        consumi_uso_finale.to_excel(writer, sheet_name='UsiFinali')
        moirae.to_excel(writer, sheet_name='UsiFinaliMOIRAE')
    return consumi_uso_finale
    
        
def stampa_riassuntivo_istat(istat, output_path = '.\Output'):
    istat['Zona_analisi'] = istat['reg'].apply(zona_climatica)
    gruppi = istat.groupby('Zona_analisi')
    
    quesito_anno_abitazione = 'q_2_4_ric'
    codifica_anno_abitazione = {1 : 'post 2000',    
                                2 : '1990-1999',    
                                3 : '1980-1989',   
                                4 : '1970-1979',
                                5 : '1960-1969' ,   
                                6 : '1950-1959',    
                                7 : '1900-1949',    
                                8 : 'pre 1900',    
                                9 : 'Non sa'}
    
    
    gruppo_anno = raggruppa_regioni(gruppi, quesito_anno_abitazione, col_name = "Anno di costruzione", descr_dict = codifica_anno_abitazione)
    
    quesito_tipo_abitazione = 'q_2_1'
    codifica_tipo_abitazione = {1 : "Casa unifamiliare",
                                2 : "Casa plurifamiliare",   
                                3 : "Appartamento < 10 abitazioni",
                                4 : "Appartamento 10 - 27 abitazioni",
                                5 : "Appartamento > 27 abitazionii"
                                }
    
    gruppo_tipo = raggruppa_regioni(gruppi, quesito_tipo_abitazione, col_name = "Tipo di costruzione", descr_dict = codifica_tipo_abitazione)
    
    
    with pd.ExcelWriter(path = os.path.join(output_path,'Validazione','DatiRegionali.xlsx'), mode = 'w')  as writer:
        gruppo_anno.to_excel(writer, sheet_name='AnnoCostruzione')
        gruppo_tipo.to_excel(writer, sheet_name='TipoAbitazione')
    
def raggruppa_regioni(gruppi,string_variabile,col_name = None, descr_dict = None, livello_sostituzione = 1, norm = False):
    data = gruppi[string_variabile].value_counts(normalize = norm)*100
    data = data.sort_index(level =livello_sostituzione)
    if not norm:
        data = data/100
    if not col_name == None:
        data.index = data.index.rename( col_name, level = livello_sostituzione)
    data = data.unstack()
    if not descr_dict == None:
        data = data.rename(columns=descr_dict)
    return data.sort_index()

def zona_climatica_(cod_reg):
    climatica ={
            'Regione': 'Regione',
            1: '2_NO', # Piemonte    
            2: '1_M', #  Valle d'Aosta    
            3: '2_NO', #  Lombardia    
            4: '1_M', #  Trentino-Alto Adige    
            5: '3_NE', #  Veneto    
            6: '3_NE', #  Friuli-Venezia Giulia    
            7: '2_NO', #  Liguria    
            8: '3_NE', #  Emilia-Romagna    
            9: '4_C', #  Toscana    
            10: '4_C', #  Umbria    
            11: '4_C', #  Marche    
            12: '4_C', #  Lazio    
            13: '4_C', #  Abruzzo    
            14: '5_S', #  Molise    
            15: '5_S',#  Campania    
            16: '5_S',#  Puglia    
            17: '5_S',#  Basilicata    
            18: '5_S',#  Calabria    
            19: '6_I', #  Sicilia    
            20: '6_I'#  Sardegna 
            }
    return climatica[cod_reg]

def zona_climatica(cod_reg):
    climatica ={
            'Regione': 'Regione',
            1: '2 Nord Ovest', # Piemonte    
            2: '1 Montuoso', #  Valle d'Aosta    
            3: '2 Nord Ovest', #  Lombardia    
            4: '1 Montuoso', #  Trentino-Alto Adige    
            5: '3 Nord Est', #  Veneto    
            6: '3 Nord Est', #  Friuli-Venezia Giulia    
            7: '2 Nord Ovest', #  Liguria    
            8: '3 Nord Est', #  Emilia-Romagna    
            9: '4 Centro', #  Toscana    
            10: '4 Centro', #  Umbria    
            11: '4 Centro', #  Marche    
            12: '4 Centro', #  Lazio    
            13: '4 Centro', #  Abruzzo    
            14: '5 Sud', #  Molise    
            15: '5 Sud',#  Campania    
            16: '5 Sud',#  Puglia    
            17: '5 Sud',#  Basilicata    
            18: '5 Sud',#  Calabria    
            19: '6 Isole', #  Sicilia    
            20: '6 Isole'#  Sardegna 
            }
    return climatica[cod_reg]
        
        
        
def confronto_scenari(info_ed, consumi_vettore_ref_, consumi_vettore_, df_fep, colonna_per_ordinare, titolo = "", percorso_salvataggio = os.path.join(".","Scenario.png")):
    # funzione che valuta il tasso di sostituzione e il risparmio energetico
    consumi_vettore_ref_[['EPnren','EPren']] = consumi_vettore_ref_.apply(calcola_energia_primaria_df, df_fep = df_fep, axis = 'columns',result_type ='expand')
    consumi_vettore_[['EPnren','EPren']] = consumi_vettore_.apply(calcola_energia_primaria_df, df_fep = df_fep, axis = 'columns',result_type ='expand')
    
    consumi_vettore_ref_['Ordinabile'] = colonna_per_ordinare
    consumi_vettore_['Ordinabile'] = colonna_per_ordinare
    
    consumi_vettore_ref = consumi_vettore_ref_.apply(lambda x: np.asarray(x) * np.asarray(info_ed['RiportGlobale']))   
    consumi_vettore = consumi_vettore_.apply(lambda x: np.asarray(x) * np.asarray(info_ed['RiportGlobale']))   
    
    consumi_vettore_ref['RiportGlobale'] = np.array([0]*20000)
    consumi_vettore['RiportGlobale'] = info_ed['RiportGlobale']
    
    consumi_vettore_ref.sort_values(by = 'Ordinabile', ascending=False, inplace = True, na_position = 'last')
    consumi_vettore.sort_values(by = 'Ordinabile', ascending=False, inplace = True, na_position = 'last')
    #consumi_vettore_ref.drop('Ordinabile', axis = 1, inplace = True)
    #consumi_vettore.drop('Ordinabile', axis = 1, inplace = True)
    
    tot_ = (consumi_vettore-consumi_vettore_ref).cumsum(axis = 0).reset_index(drop = True)
    ref = consumi_vettore_ref.sum(axis = 0)
    ref['RiportGlobale'] = consumi_vettore['RiportGlobale'].sum()
    tot = tot_ / ref * 100
    tot.drop(['Wood (kWh/y)','Gasoline (kWh/y)','Pellets (kWh/y)', 'Oil (kWh/y)', 'Coke (kWh/y)'], inplace = True, axis = 1)
    
    # Impostare grafico tot.iloc[range(0,20000,100)]
    
    fig, ax = plt.subplots(figsize = (10,5))
    for col in tot.columns:
        if col not in ['Ordinabile','RiportGlobale']:
            ax.plot(tot['RiportGlobale'].iloc[range(0,20000,100)],tot[col].iloc[range(0,20000,100)], '-', label = col )            
            # tot.iloc[range(0,20000,100)].plot(x = 'RiportGlobale', y = col, kind = 'scatter', ax = ax, legend = True)
    ax.set_xlabel('Numero di utenze considerate nello scenario (%)')
    ax.set_ylabel("Variazione consumi ed energia primaria (%)")
    ax.set_title(titolo)
    # ax.set_xlim([0,100])
    ax.set_ylim([np.min(tot.values)-20, np.max(tot.values) + 20])
    ax.legend()
    ax.grid(linewidth = 0.2)
    
    fig.savefig(percorso_salvataggio)
        
    
    
    
def calcola_energia_primaria_df(riga_utenza, df_fep):
    
    ep_nr = (df_fep['FEPnren']['Electric']*riga_utenza['Electric (kWh/y)'] + \
    df_fep['FEPnren']['NaturalGas']*riga_utenza['NaturalGas (kWh/y)'] + \
    df_fep['FEPnren']['LPG']*riga_utenza['LPG (kWh/y)'] + \
    df_fep['FEPnren']['Gasoline']*riga_utenza['Gasoline (kWh/y)'] + \
    df_fep['FEPnren']['Wood']*riga_utenza['Wood (kWh/y)'] + \
    df_fep['FEPren']['Pellets']*riga_utenza['Pellets (kWh/y)'] + \
    df_fep['FEPnren']['Oil']*riga_utenza['Oil (kWh/y)'])
        
    ep_r = (df_fep['FEPren']['Electric']*riga_utenza['Electric (kWh/y)'] + \
    df_fep['FEPren']['NaturalGas']*riga_utenza['NaturalGas (kWh/y)'] + \
    df_fep['FEPren']['LPG']*riga_utenza['LPG (kWh/y)'] + \
    df_fep['FEPren']['Gasoline']*riga_utenza['Gasoline (kWh/y)'] + \
    df_fep['FEPren']['Wood']*riga_utenza['Wood (kWh/y)'] + \
    df_fep['FEPren']['Pellets']*riga_utenza['Pellets (kWh/y)'] + \
    df_fep['FEPren']['Oil']*riga_utenza['Oil (kWh/y)'])
        
    return ep_nr, ep_r        
        
        
        
        
        
def validazione_locale_costi_eng(costi_classi, output_path = '.\Output', pfix = ''):
    # Esegue la validazione delle classi di costo come da tesi
    lista_fuels = ['Electric','NaturalGas','Gasoline','LPG','Wood','Pellets']
    
    fontsize = 16
    xpad = 10.
    layout = {'hspace':0.27, 'wspace': 0.05, 'top': 0.95, 'bottom': 0.08, 'left': 0.07, 'right' : 0.97}
    sharey = True
    
    layout = {'hspace':0.27, 'wspace': 0.27, 'top': 0.95, 'bottom': 0.08, 'left': 0.07, 'right' : 0.97}
       
    fig2, [[ax211,ax212, ax213],[ax221,ax222, ax223]] = plt.subplots(ncols = 3, nrows = 2, figsize = (15,10), gridspec_kw = layout, sharey = False)
    plots = {'Electric': ax211,
             'NaturalGas': ax212,
             'Gasoline': ax213,
             'LPG': ax221,
             'Wood': ax222,
             'Pellets': ax223}    
    
    
    for i in lista_fuels:
        # diff_classi = abs(costi_classi[i + ' classe calcolata'] - costi_classi[i + ' classe sondaggio'])[costi_classi[i + ' classe sondaggio']!=0.]
        # conteggio=(diff_classi.value_counts().sort_index()/len(diff_classi)*100).cumsum()
        if i in ['Wood', 'Pellets']:
            labels = [i + ' kg',i + ' kg sondaggio']
        else:
            labels = [i + ' euro',i + ' euro sondaggio']
        
        title = 'Electric En.'; xlabel = 'Calculated [€]'; ylabel = 'Declared [€]'; 
        if i == 'NaturalGas': title = 'Natural Gas'
        if i == 'Gasoline': title = 'Gasoline'
        if i == 'LPG': title = 'LPG'
        if i == 'Wood': title = 'Wood'; xlabel = 'Calculated [kg]'; ylabel = 'Declared [kg]'; 
        if i == 'Pellets': title = 'Pellets'  ; xlabel = 'Calculated [kg]'; ylabel = 'Declared [kg]'; 
        
        x = costi_classi[labels[0]].replace(np.nan,0)
        y = costi_classi[labels[1]].replace(np.nan,0)
        
        
        
        axis_limits = np.array([0, x.max()])    
        if i == 'Wood' or i == 'Pellets': axis_limits = np.array([0, 8000])    
        ax = plots[i]
        costi_classi.plot(kind = 'scatter',x = labels[0], y = labels[1], ax = ax, s=0.1)
        regr = linear_model.LinearRegression()
        regr.fit(x.values.reshape(-1,1), y.values)
        y_linea_regr = regr.predict(axis_limits.reshape(-1,1))
        textstr = f'Linear regression:\ny = {regr.coef_[0]:.2f}x + {regr.intercept_:.2f}' 
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        axis_limits = [0,3000] 
        if i == 'NaturalGas': axis_limits = [0,4000]
        if i == 'Gasoline': axis_limits = [0,5000]
        if i == 'LPG': axis_limits = [0,4000]
        if i == 'Wood': axis_limits = [0,8000]
        if i == 'Pellets': axis_limits = [0,8000]
        
        
        ax.plot(axis_limits,axis_limits,'r-')
        ax.plot(axis_limits,y_linea_regr,'r:')
        ax.set_xlabel(xlabel, fontsize = fontsize)
        ax.set_ylabel(ylabel, fontsize = fontsize)
        ax.set_title(title, size = fontsize + 4)
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize = fontsize-4, bbox=props)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.grid(linewidth = 0.2)
        
    
    fig2.savefig(os.path.join(output_path,'Validazione',pfix + 'ScatterPlotCombustibili_eng.png'))
    plt.close()
      
        
def report_risultati(filtro, filtro_hvac, cartella_dati = 'Output', nome_file_output = 'sim',risultati_orari = False):
    
    dfs = pd.read_excel(os.path.join(cartella_dati, 'SummaryRisultati.xlsx'), sheet_name = None)    
    
    # Lettura dei fogli excel
    info_ed = dfs['info_edificio'].set_index('id', drop = True)
    info_hvac = dfs['info_impianti'].set_index('id', drop = True)
    consumi_el =  dfs['consumi_elettrodomestici'].set_index('id', drop = True)
    consumi_hvac =  dfs['consumi_impianti'].set_index('id', drop = True)
    consumi_vett = dfs['consumi_per_vettore'].set_index('id', drop = True)
    costi = dfs['costi'].set_index('id', drop = True)
    
    # Applicazione dei filtri in input
    filt_tot = pd.Series(index = info_ed.index, dtype = bool)
    for k_filt, filt_list in filtro.items():
        filt = info_ed[k_filt].isin(filt_list)
        filt_tot = filt_tot * filt
        
    for k_filt, filt_list in filtro_hvac.items():
        filt = info_hvac[k_filt].isin(filt_list)
        filt_tot = filt_tot * filt
        
    id_edifici = info_ed[filt_tot].index
    
    # Filtraggio dei dataframe con il filtro appena calcolato
    edifici_filtrati = info_ed[filt_tot]
    consumi_el_filtrati = consumi_el[filt_tot]
    consumi_hvac_filtrati = consumi_hvac[filt_tot]
    consumi_vett_filtrati = consumi_vett[filt_tot]
    costi = costi[filt_tot]
    riporto = edifici_filtrati['RiportGlobale']
    
    elettrici = (consumi_el_filtrati.select_dtypes(include=['float64']).mul(riporto, axis = 0) / riporto.sum()).sum(axis = 0)
    hvac = (consumi_hvac_filtrati.select_dtypes(include=['float64']).mul(riporto, axis = 0) / riporto.sum()).sum(axis = 0)
    vettore = (consumi_vett_filtrati.select_dtypes(include=['float64']).mul(riporto, axis = 0) / riporto.sum()).sum(axis = 0)
        
    with pd.ExcelWriter(path = os.path.join(nome_file_output + '.xlsx')) as writer:
        elettrici.to_excel(writer, sheet_name='ElettriciFornelliForni')
        hvac.to_excel(writer, sheet_name='HVAC')
        vettore.to_excel(writer, sheet_name='Costi')
    
    if risultati_orari:        
        heating_profile = np.zeros([8760,1])
        heating_sec_profile = np.zeros([8760,1])
        acs_profile = np.zeros([8760,1])
        cooling_profile = np.zeros([8760,1])
        appliances_profile = np.zeros([8760,1])
        
        for idx in id_edifici:
            num_forder = str(idx//1000)
            orari_edificio = pd.read_csv(os.path.join(cartella_dati,num_forder,f'{idx}.csv'))
            heating_profile += orari_edificio.loc[:, orari_edificio.columns.str.startswith('Heating (')].values * riporto.loc[idx]
            acs_profile += orari_edificio.loc[:, orari_edificio.columns.str.startswith('DHW')].values * riporto.loc[idx]
            cooling_profile += orari_edificio.loc[:, orari_edificio.columns.str.startswith('Cooling')].values * riporto.loc[idx]
            appliances_profile += orari_edificio.loc[:, orari_edificio.columns.str.startswith('Appliances')].values * riporto.loc[idx]
            try:
                heating_sec_profile += orari_edificio.loc[:, orari_edificio.columns.str.startswith('Heating sec')].values * riporto.loc[idx]
            except ValueError:
                pass
        
        heating_profile = heating_profile/riporto.sum()
        heating_sec_profile = heating_sec_profile/riporto.sum()
        acs_profile = acs_profile/riporto.sum()
        cooling_profile = cooling_profile/riporto.sum()
        appliances_profile = appliances_profile/riporto.sum()
        
        orari = pd.DataFrame({'Heating (kW)': np.reshape(heating_profile, -1),
             'Heating secondario (kW)': np.reshape(heating_sec_profile, -1),
             'Cooling  (kW)': np.reshape(cooling_profile, -1),
             'DHW (kW)': np.reshape(acs_profile, -1),
             'Appliances (kW)': np.reshape(appliances_profile, -1)}
            )
        
        with pd.ExcelWriter(path = os.path.join(nome_file_output + '.xlsx'), mode = 'a') as writer:
            orari.to_excel(
                writer, sheet_name= 'RisultatiMediOrari'
                )
                
    return id_edifici



def combina_risultati(lista_cartelle_risultati,
                      percentuale_riqualificati = 0.2,
                      tipo_retrofit = 2):
    '''Combina i risultati di diverse simulazioni per creare scenari diversi.    

    Parameters
    ----------
    lista_cartelle_risultati : TYPE
        Lista di stringhe coi nomi delle cartelle di output da considerare.
    percentuale_riqualificati : TYPE, optional
        Percentuale di edifici riqualificati rispetto al totale (default 20%).
    tipo_retrofit : TYPE, optional
        Il tipo di retrofit può essere solo involucro (2) o involucro e pdc (3).

    Returns
    -------
    None.

    '''
    
    # Initialize dicts
    info_edificio = dict()
    info_elettrodomestici = dict()
    info_impianti = dict()
    consumi_elettrodomestici = dict()
    consumi_impianti = dict()
    consumi_per_vettore = dict()
    costi = dict()
    
    # Open and read results 
    for cartella in lista_cartelle_risultati:
        # Open different output folders
        dfs = pd.read_excel(os.path.join(cartella, 'SummaryRisultati.xlsx'), 
                            sheet_name = None, index_col = 0)
        dfs['info_elettrodomestici'].columns = dfs['info_elettrodomestici'].iloc[0]  
        dfs['info_elettrodomestici'] = dfs['info_elettrodomestici'].drop(dfs['info_elettrodomestici'].index[0])
        dfs['info_elettrodomestici'] = dfs['info_elettrodomestici'].drop(dfs['info_elettrodomestici'].index[0])
        # Initialize new dataframes from scenario with no retrofits
        if cartella == 'OutputGlobaleCC-1':
            dfs_new = dfs
        # Saves results into dicts
        for df in dfs.items():
            info_edificio[cartella] = dfs['info_edificio']
            info_elettrodomestici[cartella] = dfs['info_elettrodomestici']
            info_impianti[cartella] = dfs['info_impianti']
            consumi_elettrodomestici[cartella] = dfs['consumi_elettrodomestici']
            consumi_impianti[cartella] = dfs['consumi_impianti']
            consumi_per_vettore[cartella] = dfs['consumi_per_vettore']
            costi[cartella] = dfs['costi']

    # Generate vector with id of refurbished builidngs (True)   
    id_riqualificati = np.random.choice(a=[True, False],  
                                        size=(20000, 1), 
                                        p = [percentuale_riqualificati,1-percentuale_riqualificati])
    
    # Generate new results file
    dfs_new = dfs
    for key, df in dfs_new.items():
        # pesca righe da dataframe con edifici riqualificati
        cartella_retrofit = 'OutputGlobaleCC-' + str(tipo_retrofit)
        df1 = locals()[key][cartella_retrofit]       
        # crea nuovo dataframe da combinazione di riqualificati e non riqualificati
        df2 = pd.concat([df[~id_riqualificati], df1[id_riqualificati]], axis = 0)
        df2 = df2.sort_index()
        # sostituisce dataframe con parte degli edifici riqualificati
        dfs_new[key] = df2
        
    nome_cartella_output = 'OutputGlobaleCC_tipo-' + str(tipo_retrofit) + '-perc-' + str(percentuale_riqualificati)
    cartella_output_esiste = os.path.exists(nome_cartella_output)
    # Create a new directory because it does not exist
    if not cartella_output_esiste:       
       os.makedirs(nome_cartella_output)
       print("A new output directory has been created!")
    
    # apri nuovo file excel e scrivi ogni df su foglio        
    with pd.ExcelWriter(path=os.path.join(nome_cartella_output, "SummaryRisultati.xlsx")) as writer:
        for key, df_new in dfs_new.items():
            df_new.replace({0: ""}).to_excel(writer, sheet_name=key)
        
    
    

    
    return
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        