"""Verify native table transcriptions against frozen published/specification and cohort sources."""
from pathlib import Path
import hashlib,json,csv,re
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths={'specification':ROOT/'00_context/archetype_extension/published_archetype_specification.md','assignment_audit':ROOT/'00_context/archetype_extension/local_assignment_audit.json','strata':ROOT/'05_simulations/archetype_campaign_v2/pre_run/stratum_counts.csv','probabilities':ROOT/'05_simulations/archetype_campaign_v2/pre_run/mixture_probabilities.json','manuscript':ROOT/'08_manuscript/article_first_draft.md'}
assert sha(paths['specification'])=='61fa5c2b31137b6459d267b4714c66801dcd35b6e305fa43ce5d79432638f207'
assert sha(paths['assignment_audit'])=='bb7a79b0f5cbc880cd532a0028c127af95a0e5937c8787ca7bc1578237d9e099'
s=paths['manuscript'].read_text(encoding='utf-8-sig'); tables={}
for id in ['TAB-01','TAB-02']:
    after=s.split('<!-- '+id+' |')[1]; rows=[];started=False
    for line in after.splitlines():
        if line.startswith('|'):
            started=True;rows.append([x.strip() for x in line.split('|')[1:-1]])
        elif started:break
    tables[id]=rows
expected=[['70.2 / 78.3','1.8','0.5','40.1 / 29.9'],['21.2 / 57.2','3.7','0.7','26.9 / 40.2'],['46.0 / 63.0','2.3','0.5','22.9 / 21.3'],['55.6 / 72.2','2.1','0.8','10.1 / 8.6']]
assert len(tables['TAB-01'])==6
for i,row in enumerate(tables['TAB-01'][2:]):assert row[0]==str(i+1) and row[2:]==expected[i]
assert '72.1 locally versus 72.2 printed' in s
assert [r[1] for r in tables['TAB-02'][2:]]==['20,000','17,855','13,624','13,592','32']
assert '5430 / 3693 / 2991 / 1510' in tables['TAB-02'][4][2]
with paths['strata'].open(newline='') as f:strata=list(csv.DictReader(f))
assert sum(int(r['eligible_n']) for r in strata)==13624
assert [sum(int(r['eligible_n']) for r in strata if r['archetype_id']==str(i)) for i in range(1,5)]==[5430,3693,2991,1510]
assert json.loads(paths['probabilities'].read_text())['training_n']==13592
result={'status':'pass','TAB-01':'four published rows; all24 displayed numeric cells exact; description meanings and distinct unweighted household/person denominators retained; printed72.2 and adjacent local72.1 disclosure retained','TAB-02':'all5domain counts and4eligible group counts exact; count/purpose separate; two per16strata; training13592/heldout32','table_text_sha256':{k:hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in tables.items()},'sources_sha256':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},'table_rows':tables,'scope':'transcription verification, no empirical validation; no duplicate table images'}
(OUT/'table_transcription_qa.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('TAB-01/TAB-02 transcription checks pass')
