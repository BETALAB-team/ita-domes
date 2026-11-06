"""Acquire fresh homogeneous candidate TMY inputs; retain source metadata and failures."""
from pathlib import Path
import requests,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_simulations/climate_inputs_v2'
OUT.mkdir(parents=True,exist_ok=True)
SITES={'Padova':(45.408,11.873),'Bolzano':(46.50,11.35),'Roma':(41.90,12.50),'Palermo':(38.12,13.36)}
BASE='https://re.jrc.ec.europa.eu/api/v5_3/tmy'
manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'provider_documentation':'https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en','purpose':'candidate homogeneous climate descriptor inventory before selection; not historical weather','requested_shared_window':[2005,2023],'sites':[]}
for name,(lat,lon) in SITES.items():
    row={'site':name,'lat':lat,'lon':lon,'files':[]}
    for fmt in ['json','epw']:
        params={'lat':lat,'lon':lon,'startyear':2005,'endyear':2023,'outputformat':fmt,'usehorizon':1}
        try:
            r=requests.get(BASE,params=params,timeout=45)
            target=OUT/f'{name}.{fmt}'
            target.write_bytes(r.content)
            row['files'].append({'format':fmt,'url':r.url,'http_status':r.status_code,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'path':str(target),'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'content_type':r.headers.get('content-type')})
            print(name,fmt,r.status_code,len(r.content),flush=True)
        except Exception as ex:
            row['files'].append({'format':fmt,'error':repr(ex)})
            print(name,fmt,repr(ex),flush=True)
    manifest['sites'].append(row)
    (OUT/'acquisition_manifest.json').write_text(json.dumps(manifest,indent=2))
