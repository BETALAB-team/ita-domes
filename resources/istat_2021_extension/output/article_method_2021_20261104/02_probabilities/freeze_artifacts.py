from pathlib import Path
import hashlib,json
O=Path('output/article_method_2021_20261104/02_probabilities');m={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'};(O/'artifact_hashes.json').write_text(json.dumps(m,indent=2));print(len(m),'artifacts frozen')
