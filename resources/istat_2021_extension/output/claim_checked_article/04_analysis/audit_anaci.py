"""Inventory candidate annual-consumption fields without assuming measurement provenance."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
P=Path(r'C:\Works\PaduaDigital\PaduaDigitalModel\ANACI-processed')
result=[]
for f in sorted(P.glob('*.geojson')):
    obj=json.loads(f.read_text(encoding='utf-8-sig')); features=obj.get('features',[])
    keys=sorted({k for x in features for k in x.get('properties',{}) if 'STDMC' in k.upper() or 'VERIF' in k.upper() or k.endswith('_cal')})
    fields={}
    for k in keys:
        values=[x.get('properties',{}).get(k) for x in features]
        numbers=[v for v in values if isinstance(v,(int,float))]
        fields[k]={'nonnull':sum(v is not None for v in values),'numeric_count':len(numbers),'positive_count':sum(v>0 for v in numbers),'zero_count':sum(v==0 for v in numbers),'negative_count':sum(v<0 for v in numbers),'min':min(numbers) if numbers else None,'max':max(numbers) if numbers else None}
    result.append({'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'features':len(features),'fields':fields})
(ROOT/'04_analysis/anaci_field_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
