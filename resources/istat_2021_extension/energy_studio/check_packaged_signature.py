"""Run the built Windows application in fresh-reference and reuse modes."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .config import default_config,save


def main(source,exe):
    source=Path(source).resolve();exe=Path(exe).resolve()
    root=source.parent/'packaged_v0_4';root.mkdir(exist_ok=True)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    subprocess.run([str(exe),'--self-test'],check=True,timeout=90,creationflags=flags)
    expected=pd.read_csv(source/'hybrid_predictions_frozen.csv').sort_values(['method','scenario'])
    results=[]
    for mode,count in [('fresh',1),('reuse',0)]:
        cfg=default_config();cfg['report']['powerpoint']=False
        cfg['paths']['output_root']=str(root/mode)
        if mode=='reuse':cfg['paths']['reference_run']=str(source)
        config_path=root/(mode+'_config.json');save(cfg,config_path)
        before=set((root/mode).glob('*')) if (root/mode).exists() else set()
        subprocess.run([str(exe),'--config',str(config_path),'--run'],check=True,timeout=480,creationflags=flags)
        created=[p for p in (root/mode).iterdir() if p.is_dir() and p not in before]
        assert len(created)==1;out=created[0]
        m=json.loads((out/'manifest.json').read_text())
        assert m['status']=='complete' and m['energyplus_processes_run']==count
        actual=pd.read_csv(out/'hybrid_predictions_frozen.csv').sort_values(['method','scenario'])
        assert expected[['method','scenario']].equals(actual[['method','scenario']])
        np.testing.assert_allclose(expected[['heating_kwh','cooling_kwh','appliance_kwh']],actual[['heating_kwh','cooling_kwh','appliance_kwh']],rtol=1e-10,atol=1e-7,equal_nan=True)
        assert (out/'report.html').is_file() and list((out/'plots').glob('*.svg'))
        results.append({'mode':mode,'run':str(out),'ep_simulations':count,'matches_source_benchmark':True})
        print(results[-1],flush=True)
    (root/'acceptance.json').write_text(json.dumps({'exe':str(exe),'gui_selftest':True,'runs':results},indent=2))


if __name__=='__main__':main(*sys.argv[1:])
