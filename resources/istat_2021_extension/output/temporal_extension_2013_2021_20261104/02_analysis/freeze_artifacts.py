from pathlib import Path
import hashlib,json
out=Path('output/temporal_extension_2013_2021_20261104/02_analysis');r=json.loads((out/'results.json').read_text());checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in r['source_hashes'].items()};(out/'final_source_integrity.json').write_text(json.dumps(checks,indent=2));assert all(checks.values())
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'};(out/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2));print('Frozen',len(manifest),'artifacts; all source hashes unchanged')
