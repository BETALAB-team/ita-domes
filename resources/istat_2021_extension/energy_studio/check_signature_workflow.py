"""Integration acceptance against a completed reference-signature benchmark.

Run from repository root: python -m energy_studio.check_signature_workflow RUN_DIR
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from .engine import run,sha


def check(source):
    source=Path(source).resolve()
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    cfg['paths']['reference_run']=str(source)
    cfg['simulation']['run_full_benchmark']=False
    cfg['report']['powerpoint']=False
    with patch('energy_studio.engine.run_model',side_effect=AssertionError('No EP process allowed with reference reuse')), \
         patch('energy_studio.physics.simulate_rc',side_effect=AssertionError('RC prohibited')):
        out=run(copy.deepcopy(cfg))
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['energyplus_processes_run']==0 and manifest['reference_simulations_used']==1
    assert len(list((out/'energyplus').iterdir()))==1
    before=pd.read_csv(source/'hybrid_predictions_frozen.csv').sort_values(['method','scenario'])
    after=pd.read_csv(out/'hybrid_predictions_frozen.csv').sort_values(['method','scenario'])
    assert before[['method','scenario']].equals(after[['method','scenario']])
    cols=['heating_kwh','cooling_kwh','appliance_kwh']
    np.testing.assert_allclose(before[cols],after[cols],rtol=1e-12,atol=1e-9,equal_nan=True)
    assert sha(out/'hybrid_predictions_frozen.csv')==manifest['hybrid_frozen_sha256']
    annual=pd.read_csv(out/'annual_results.csv')
    assert len(annual.loc[annual.method.eq('energyplus')])==1
    assert len(annual.loc[annual.method.eq('hybrid_signature')])==36
    receipt={'source_run':str(source),'reuse_run':str(out),'new_ep_processes':0,
             'predictions_match_full_benchmark':True,'tolerance_rtol':1e-12,'tolerance_atol_kwh':1e-9}
    (out/'reuse_acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    return out


if __name__=='__main__':check(sys.argv[1])
