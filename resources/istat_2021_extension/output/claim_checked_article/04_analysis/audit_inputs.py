"""Read-only inventory of relevant local inputs; no model results inferred."""
from pathlib import Path
import hashlib, json, subprocess, sys, importlib.metadata as md
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRINS = Path(r'C:\Works\GRINS\GRINS-Tool')
EURECA = Path(r'C:\Works\EUReCA\EUReCA')
FORK = Path(r'C:\Works\EUReCA\Fork-Macovei\EUReCA-maco')
WX = FORK/'eureca_ubem/Input/tmy_45.408_11.873_2005_2023.epw'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def revision(p):
    return subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
def main():
    survey=GRINS/'resources/istat_data/istat_microdata_2013.csv'
    df=pd.read_csv(survey,sep=';')
    cols=['id','reg','q_3_11A','q_3_11B','q_3_11C','q_3_13_2','q_3_13_3','q_3_19_6','q_3_0','q_2_32C','q_2_32D']
    eligible=df[(df['reg']==5)&df[['q_3_11A','q_3_11B','q_3_11C']].gt(0).all(axis=1)&df[['q_3_11A','q_3_11B','q_3_11C']].lt(8).all(axis=1)]
    selected=eligible.sort_values('id').iloc[0][cols].to_dict()
    lines=WX.read_text().splitlines()
    wx=pd.read_csv(WX,skiprows=8,header=None)
    t=wx[6].to_numpy(float)
    daily=t.reshape(365,24).mean(axis=1)
    audit={'python':sys.version,'executable':sys.executable,'revisions':{str(p):revision(p) for p in [GRINS,EURECA,FORK]},
      'packages':{x:md.version(x) for x in ['numpy','pandas','scipy','pvlib','pyclipper','openpyxl']},
      'selected_survey_response':selected,'eligible_veneto_count':len(eligible),'survey_rows':len(df),
      'selection':'lowest id in reg=5 with all three reported heating-block hours strictly between 0 and 8; illustrative, unweighted, not Padova sample',
      'weather':{'path':str(WX),'header':lines[:8],'n':len(wx),'years_present':sorted(wx[0].unique().tolist()),'missing_temp':int(wx[6].isna().sum()),
       'temperature_min_C':float(t.min()),'temperature_max_C':float(t.max()),'HDD18_daily_Kd':float((18-daily).clip(0).sum()),'CDD26_daily_Kd':float((daily-26).clip(0).sum()),
       'mean_RH_percent':float(wx[8].mean()),'GHI_kWh_m2':float(wx[13].sum()/1000),'provenance':'local file, filename suggests 2005–2023 TMY; header ECMWF/ERA; generator/retrieval metadata unavailable, not historical weather'},
      'sha256':{str(p):sha(p) for p in [survey,WX,GRINS/'classes/EndUse.py',GRINS/'funcs/main_functions.py',EURECA/'eureca_building/thermal_zone.py',EURECA/'eureca_building/weather.py',EURECA/'eureca_building/config.py']}}
    gis=[]
    for parent in [Path(r'C:\Works\LivingLab Padova'),Path(r'C:\Works\PaduaDigital\PaduaDigitalModel\ANACI-processed')]:
        for p in parent.glob('*.geojson'):
            try:
                obj=json.loads(p.read_text(encoding='utf-8-sig'))
                features=obj.get('features',[])
                keys=sorted({k for f in features for k in f.get('properties',{})})
                gis.append({'path':str(p),'sha256':sha(p),'features':len(features),'property_keys':keys})
            except Exception as ex: gis.append({'path':str(p),'error':str(ex)})
    audit['GIS_inventory']=gis
    (ROOT/'04_analysis/input_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['selected_survey_response','eligible_veneto_count','weather']},indent=2))
if __name__=='__main__': main()
