"""Verify frozen source hashes without writing any source."""
import hashlib,json
from pathlib import Path
p=Path(__file__).with_name('baseline_freeze.json')
manifest=json.loads(p.read_text())['sources']
bad=[]
for r in manifest:
    h=hashlib.sha256()
    with Path(r['path']).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    if h.hexdigest()!=r['sha256']:bad.append(r['path'])
print(json.dumps({'checked':len(manifest),'changed':bad,'all_unchanged':not bad},indent=2))
raise SystemExit(bool(bad))
