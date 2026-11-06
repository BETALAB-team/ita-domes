"""Acceptance check: packaged GUI and one-reference independent predictions."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd


def main():
    source=Path(sys.argv[1]).resolve();exe=Path(sys.argv[2]).resolve()
    root=source.parent/'packaged_v0_5';root.mkdir(exist_ok=True)
    cfg=json.loads((source/'config.json').read_text())
    cfg['report']['powerpoint']=False;cfg['simulation']['run_full_benchmark']=False
    cfg['paths']['output_root']=str(root/'runs')
    config=root/'acceptance_config.json';config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    for args in [['--self-test'],['--config',str(config),'--run']]:
        result=subprocess.run([str(exe),*args],timeout=240,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(f'Packaged command failed: {args}; exit={result.returncode}')
    run=sorted((root/'runs').iterdir())[-1]
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['energyplus_processes_run']==1
    expected=pd.read_csv(source/'predictions_frozen.csv').sort_values(['scenario_id','method'])
    actual=pd.read_csv(run/'predictions_frozen.csv').sort_values(['scenario_id','method'])
    for key in ['scenario_id','method']:assert list(expected[key])==list(actual[key])
    for key in ['heating_kwh','cooling_kwh','appliance_kwh']:
        np.testing.assert_allclose(expected[key],actual[key],rtol=1e-10,atol=1e-7,equal_nan=True)
    assert manifest['benchmark_uses_istat_schedules'] is False
    assert manifest['independent_schedule_sha256']==json.loads((source/'manifest.json').read_text())['independent_schedule_sha256']
    receipt={'status':'pass','gui_self_test':'pass','reference_only_ep_processes':1,'prediction_rows_matched':len(actual),
             'rtol':1e-10,'atol_kwh':1e-7,'source':str(source),'packaged_run':str(run)}
    (root/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':main()
