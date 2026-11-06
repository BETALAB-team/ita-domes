# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:56:57 2025

@author: vivijac14771
"""

import pandas as pd

def read_table_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        df = pd.read_csv(file, sep='\t', skiprows=1, index_col= 'num. ordine')
            
    return df

# Usa la funzione
df_13 = read_table_from_file('tracciato_delimited_2013.txt')
df_21 = read_table_from_file('tracciato_delimited_2021.txt')

with pd.ExcelWriter('legenda_istat.xlsx') as writer:  
    df_13.to_excel(writer, sheet_name='2013')
    df_21.to_excel(writer, sheet_name='2021')

# # Salva il DataFrame come CSV
# output_path = 'percorso/del/tuo/output.csv'
# df_13.to_csv(output_path, sep=';', index=False)
# print(f"Il file è stato salvato come {output_path}")