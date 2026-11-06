"""Actual integration runs: changed clustering and one-reference ISTAT transfer."""
import copy,json
from pathlib import Path
import pandas as pd
from .config import default_config
from .engine import run

if __name__=='__main__':
    cfg=default_config();cfg['report']['powerpoint']=False
    cfg['analysis'].update(archetype_mode='refit',n_archetypes=3,features=['agemin','agemax','nmembers'],pca_components=2,kmeans_starts=2,heating_groups=3,electricity_groups=3,group_factors=['location','archetype'])
    cfg['simulation'].update(enabled=False,area_m2=120.,anchor_climate='Roma',anchor_context='Apartment',anchor_archetype=1)
    for climate in cfg['climates']:climate['enabled']=climate['name']=='Roma'
    for context in cfg['contexts']:context['enabled']=context['name']=='Apartment'
    first=run(cfg);df=pd.read_csv(first/'annual_results.csv')
    assert len(df)==3 and df.heating_kwh.isna().all() and df.cooling_kwh.isna().all()
    assert df.appliance_kwh.notna().all()
    assert json.loads((first/'manifest.json').read_text())['energyplus_processes_run']==0
    cfg=default_config();cfg['report']['powerpoint']=False
    cfg['simulation']['run_full_benchmark']=False
    cfg['hybrid']['method']='istat_ratio'
    # Deliberately invalid target weather paths prove they are not used.
    for climate in cfg['climates']:
        if climate['name']!='Milano':climate['epw']='not_required_for_istat_transfer.epw'
    second=run(cfg);df=pd.read_csv(second/'annual_results.csv')
    manifest=json.loads((second/'manifest.json').read_text())
    assert manifest['energyplus_processes_run']==1
    assert len(df[df.method.eq('hybrid_istat')])==36 and len(df[df.method.eq('energyplus')])==1
    assert [w['name'] for w in manifest['weather']]==['Milano']
    assert df[df.method.eq('istat_only')].heating_kwh.isna().all()
    print('Variant integration checks passed:',first,second)
