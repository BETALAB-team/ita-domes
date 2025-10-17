# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:10:17 2025

@author: vivijac14771
"""

import pandas as pd

def read_formatted_file(file_path):
    # Legge il file usando pandas, specificando il separatore '^'
    df = pd.read_csv(file_path, sep='^', dtype=str)
    
    # Rimuove gli spazi bianchi iniziali e finali dai nomi delle colonne
    df.columns = df.columns.str.strip()
    
    # Rimuove gli spazi bianchi iniziali e finali da tutti i valori nel DataFrame
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

# Uso della funzione
file_path = 'MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt'
parsed_data = read_formatted_file(file_path)

#%% Percorso del file di output
output_file_path = 'istat_microdata_2021.csv'

# Salva il DataFrame come CSV usando ';' come separatore
parsed_data.to_csv(output_file_path, sep=';', index=False)

print(f"Il file è stato salvato come {output_file_path}")


