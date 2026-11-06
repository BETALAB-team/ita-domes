"""2021 appliance kWh with the verified main-branch 2013 component boundary.

Reference efficiencies are held fixed. Missing survey detail is represented by
explicit 2013 weighted averages and scenario alternatives, not recorded data.
Run with the bundled Python (pandas/openpyxl); main is read, never checked out.
"""
import io
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from audit_appliance_model import ROOT, REPO, git_blob


def estimate(d, old, db, components, scenario='reference'):
    w=old.coef_red
    out=pd.DataFrame({'id':d.ID_RECORD})
    # 2021 reports bulb totals, not hours by use band. Estimate type-specific
    # mean hours per bulb from the complete 2013 weighted survey.
    light=db['illuminazioneMOIRAE'].iloc[:,0].astype(float)
    lighting=0
    for prefix,field,power in [('q_7_2','v7_1a',light.iloc[0]),('q_7_4','v7_1b',light.iloc[1])]:
        counts=old[[prefix+c for c in 'ABC']]
        hours=((counts*np.array([3,7,14])).sum(axis=1)*w).sum()/(counts.sum(axis=1)*w).sum()
        if scenario=='lower':hours=3.
        if scenario=='upper':hours=7.
        lighting += 330*power*d[field]*hours
    out['lighting_kwh']=lighting
    # Small appliances are not individually released in 2021. A common weighted
    # background maintains the original outcome boundary without inventing ownership.
    small=np.average(components.small,weights=w)
    out['small_appliances_kwh']=0. if scenario=='lower' else small*(2 if scenario=='upper' else 1)
    # Weighted 2013 distribution of fridge capacity and multiplicity among owners.
    owners=old.q_8_2.gt(0)&old.q_8_5.isin([1,2,3,4])
    shares=old.loc[owners].groupby('q_8_5').coef_red.sum()/w[owners].sum()
    factor=np.average(old.loc[owners,'q_8_2'].map(lambda n:1 if n==1 else 1.3 if n==2 else 1.5),weights=w[owners])
    if scenario in ['lower','upper']:
        shares=pd.Series({1:1.}) if scenario=='lower' else pd.Series({3:1.})
        factor=1. if scenario=='lower' else 1.3
    # Ages map by their interval midpoint to legacy age-efficiency assumptions.
    # Unknown age uses the legacy EEI fallback; these are not 2021 energy labels.
    eei={1:27.5,2:49.5,3:49.5,4:65.,5:85.,6:85.,7:102.5,8:65.}
    age=d.v7_6.map(eei); free_age=d.v7_9.map(eei)
    box=d.v7_7.eq(1)
    fridge=0
    for cap,share in shares.items():
        volume,bvol,temp,m,n={1:(150,19,5,.233,245),2:(250,50,0,.233,245),
                             3:(400,100,-6,.643,191),4:(600,200,-12,.450,245)}[cap]
        veq=(volume-box*bvol)*(25-temp)/20+box*bvol*(25+18)/20
        fridge += share*factor*(veq*m+n+50*box)*age/100
    fridge=fridge.where(d.v7_5.eq(1),0)
    free=(220*(25+18)/20*.472+286)*free_age/100
    free=free.where(d.v7_8.eq(1),0)
    out['refrigeration_kwh']=(330/365*(fridge+free)).round(1)
    # CE model: 47 operating weeks and temperature-specific cycle coefficients.
    occasional=18 if scenario=='reference' else 0 if scenario=='lower' else 24
    cycles=d[['v7_13a','v7_13b','v7_13c']]
    wash=47*1.8*(cycles*np.array([.3,1,2.56])).sum(axis=1,min_count=3)
    wash=wash.where(d.v7_12b.ne(1),occasional*1.8)
    wash=wash.where(d.v7_10.eq(1),0)+d.v7_15.eq(1).astype(float)
    dryer=(47*d.v7_18a).where(d.v7_18b.ne(1),occasional).where(d.v7_16.eq(1),0)
    dish=(47*1.5*d.v7_21a).where(d.v7_21b.ne(1),occasional*1.5).where(d.v7_19.eq(1),0)
    out['washing_kwh']=wash;out['drying_kwh']=dryer;out['dishwashing_kwh']=dish
    screens=db['schermi_MOIRAE']
    powers=pd.to_numeric(screens.iloc[:4,0]);standby=pd.to_numeric(screens.iloc[:4,1])
    ix={1:2,2:3,3:1,4:0}
    power=d.v7_34.map({**{k:powers.iloc[v] for k,v in ix.items()},5:powers.mean()})
    sb=d.v7_34.map({**{k:standby.iloc[v] for k,v in ix.items()},5:standby.mean()})
    hours=d.v7_35.map({1:1.5,2:3.,3:5.,4:8.,5:12.})
    out['tv_kwh']=(330*(hours*power+(24-hours)*sb)).where(d.v7_32.eq(1),0).round(1)
    desktop=float(screens.iloc[5,0]);portable=float(screens.iloc[6,0])
    pc=d.v7_38.map({1:desktop,2:portable,3:(desktop+portable)/2})
    # 2021 has the primary computer type only, including tablets in portable.
    if scenario=='lower':pc=portable
    if scenario=='upper':pc=desktop
    out['computers_kwh']=(d.v7_37*pc).where(d.v7_36.eq(1),0).round(1)
    out['standby_kwh']=float(db['Standby'].iloc[0,0])*d.ncomp_mfr
    cols=[c for c in out if c.endswith('_kwh')]
    out['appliances_el_kWh']=out[cols].sum(axis=1,min_count=len(cols))
    return out


def main():
    folder=ROOT/'revised_paper_outputs/appliance_audit';folder.mkdir(exist_ok=True)
    old=pd.read_csv(ROOT/'istat_microdata_2013.csv',sep=';').set_index('id',drop=False)
    components=pd.read_csv(folder/'components_2013.csv',index_col='id').reindex(old.index)
    observed=pd.read_csv(ROOT/'results_grins.csv').set_index('id').appliances_el_kWh.reindex(old.index)
    reproduced=components[['lights','small','fridge_ce','big_ce','screens','standby']].sum(axis=1,min_count=6)
    error=float((reproduced-observed).abs().max())
    assert error<1e-8, f'2013 component boundary does not reproduce saved output: {error}'
    blob=git_blob('resources/appliances_data/Database_elettrodomestici_2013.xlsx')
    db={s:pd.read_excel(io.BytesIO(blob),sheet_name=s,index_col=None if s=='Standby' else 0)
        for s in ['illuminazioneMOIRAE','schermi_MOIRAE','Standby']}
    path=next((ROOT/'istat/Microdati21').rglob('*Microdati_2021_DELIMITED.txt'))
    raw=pd.read_csv(path,sep='^',dtype=str)
    d=raw.apply(lambda c:pd.to_numeric(c.str.strip(),errors='coerce'))
    output=estimate(d,old,db,components)
    for scenario in ['lower','upper']:
        output[scenario+'_scenario_kwh']=estimate(d,old,db,components,scenario).appliances_el_kWh
    output['electricity_model']='main_2013_CE_reference_parameters'
    output.to_csv(folder/'electricity_2021.csv',index=False)
    assert output.id.is_unique and output.appliances_el_kWh.notna().all()
    assert output.appliances_el_kWh.ge(0).all()
    weights=d.PESO
    report={'main_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','main'],text=True).strip(),
            'database_sha256':hashlib.sha256(blob).hexdigest(),
            '2013_max_absolute_replication_error_kwh':error,'2013_n':len(old),
            '2021_n':len(output),'weighted_mean_kwh':float(np.average(output.appliances_el_kWh,weights=weights)),
            'weighted_lower_scenario_kwh':float(np.average(output.lower_scenario_kwh,weights=weights)),
            'weighted_upper_scenario_kwh':float(np.average(output.upper_scenario_kwh,weights=weights)),
            'assumptions':[
                'Reference 2013 power, efficiency and operating-day parameters, not an updated 2021 efficiency database.',
                'Lighting hours per bulb type: weighted 2013 average; alternative scenarios 3 and 7 hours/day.',
                'Small-appliance background: weighted 2013 mean; alternatives zero and twice that mean.',
                'Fridge capacity/multiplicity: weighted 2013 owner distribution; alternatives small/single and large/factor 1.3.',
                '2021 age-band midpoints mapped to legacy relative-age EEI; unknown age uses EEI 65.',
                'Infrequent laundry/dishwashing: 18 cycles/year; alternatives 0 and 24.',
                'Primary TV type represents TV power. Primary PC type represents all computers/tablets; alternatives all portable/all desktop.',
                'Generic standby uses released member count; top-coded six-or-more households excluded by manuscript eligibility.',
                'Scenarios are assumptions, not confidence intervals; no measured electricity validation claimed.'
            ]}
    (folder/'electricity_model_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
