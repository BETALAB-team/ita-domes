from pathlib import Path
import pandas as pd,numpy as np,json
out=Path('output/temporal_extension_2013_2021_20261104/02_analysis');raw=pd.read_csv('istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt',sep='^',low_memory=False);d=pd.read_csv(out/'assigned_2021_proxy.csv');raw['eligible']=raw.ID_RECORD.isin(d.id)
raw.groupby(['ncomp_mfr','eligible']).size().unstack(fill_value=0).to_csv(out/'coverage_by_household_size_2021.csv')
raw.groupby(['RIP','eligible']).size().unstack(fill_value=0).to_csv(out/'coverage_by_macroregion_2021.csv')
ages=raw[[f'eta_mfr{i}' for i in range(1,6)]].apply(pd.to_numeric,errors='coerce');raw['has_under15']=(ages<=3).any(axis=1);raw.groupby(['has_under15','eligible']).size().unstack(fill_value=0).to_csv(out/'coverage_under15_2021.csv')
print(raw.groupby(['has_under15','eligible']).size())
