"""Small real-engine integration check without a full scenario run."""
from pathlib import Path
import threading,json
from .config import default_config
from .physics import read_weather
from .energyplus import make_model,run_model

if __name__=='__main__':
    cfg=default_config();climate=cfg['climates'][1]
    profile={'members':2.,'heating_hours':8.,'cooling_hours':6.,'appliance_kwh':2000.}
    model=make_model(cfg,cfg['contexts'][0],profile,read_weather(climate['epw']))
    out=Path(cfg['paths']['output_root'])/'integration_smoke'
    result,seconds,warnings=run_model(model,climate['epw'],cfg['paths']['energyplus'],out,180,threading.Event())
    print(result.sum(numeric_only=True).to_dict(),seconds,warnings)
