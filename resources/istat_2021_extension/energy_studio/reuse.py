"""Reuse exactly one compatible reference result from a completed local run."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .energyplus import make_model


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_reference(cfg,anchor,weather,out):
    source=Path(cfg['paths']['reference_run']).resolve()
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete':raise ValueError('Reused reference run must be complete.')
    annual=pd.read_csv(source/'annual_results.csv')
    rows=annual[annual.method.eq('energyplus') & annual.scenario.eq(anchor['key'])]
    if len(rows)!=1:raise ValueError('The selected run must contain exactly one EP result matching the reference scenario.')
    folder=source/'energyplus'/str(rows.iloc[0].scenario_id)
    expected=make_model(cfg,anchor['context'],anchor['profile'],weather)
    actual=json.loads((folder/'model.epJSON').read_text(encoding='utf-8'))
    if actual!=expected:raise ValueError('Saved reference model differs from current geometry, controls or household inputs. Clear Reference reuse and run a new reference simulation.')
    weather_hash=digest(anchor['climate']['epw'])
    if not any(w['name']==anchor['climate']['name'] and w['sha256']==weather_hash for w in manifest.get('weather',[])):
        raise ValueError('Saved reference weather differs from the configured EPW.')
    values=pd.read_csv(folder/'hourly.csv')
    columns=['heating_kwh','cooling_kwh','appliance_kwh']
    if len(values)!=8760 or not np.isfinite(values[columns]).all().all() or (values[columns]<0).any().any():raise ValueError('Invalid saved reference hourly output.')
    if not np.allclose(values[columns].sum().to_numpy(float),rows.iloc[0][columns].to_numpy(float),rtol=1e-10,atol=1e-6):raise ValueError('Saved reference hourly totals do not match its annual table.')
    destination=out/'energyplus'/anchor['id'];destination.mkdir(parents=True)
    for name in ['hourly.csv','model.epJSON']:shutil.copy2(folder/name,destination/name)
    return values,{'source_run':str(source),'scenario':anchor['key'],'model_sha256':digest(folder/'model.epJSON'),
                   'hourly_sha256':digest(folder/'hourly.csv'),'weather_sha256':weather_hash,
                   'compatibility':'Exact generated epJSON and weather hash match; reference annual/hourly totals reconciled.'}
