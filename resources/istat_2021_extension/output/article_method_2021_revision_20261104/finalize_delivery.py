"""Verify immutable inputs and assemble the revised code/results archive."""
from pathlib import Path
import hashlib,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'article_method_2021_20261104';REPO=ROOT.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
previous=json.loads((OLD/'PACKAGE_MANIFEST.json').read_text())['files']
assert all(sha(OLD/rel)==v['sha256'] for rel,v in previous.items()),'Previous package changed'
frozen=json.loads((ROOT/'02_probability_candidates/artifact_hashes.json').read_text())
assert all(sha(ROOT/'02_probability_candidates'/rel)==v for rel,v in frozen.items()),'New candidate files changed'
base=REPO/'output/temporal_extension_2013_2021_20261104/00_baseline'
original=json.loads((base/'baseline_freeze.json').read_text())['sources']
assert all(sha(Path(v['path']))==v['sha256'] for v in original),'Protected originals changed'
qa=json.loads((ROOT/'06_report/qa/verification.json').read_text())
assert qa['pages']==9 and qa['pngs']==9 and qa['visual_review'].startswith('passed')
assert not qa['missing_docx'] and not qa['missing_pdf'] and qa['native_equations']==1 and qa['embedded_figures_verified']==4
write(ROOT/'06_report/delivery_checks.json',qa)
ref=ROOT/'01_reference';ref.mkdir(exist_ok=True)
copy_sources=[OLD/'02_probabilities/household_archetypes_2021.csv',OLD/'02_probabilities/dwelling_samples_2021.csv',
 OLD/'02_probabilities/polynomial_coefficient_matrix.csv',OLD/'requirements-2021.txt',
 base/'published_benchmarks.json',base/'baseline_freeze.json',REPO/'probablities_only.csv']
records=[]
for p in copy_sources:
 target=ref/p.name;shutil.copyfile(p,target);assert sha(p)==sha(target)
 records.append({'source':str(p),'copy':str(target.relative_to(ROOT)),'sha256':sha(p)})
shutil.copyfile(OLD/'requirements-2021.txt',ROOT/'requirements-2021.txt')
write(ref/'reference_manifest.json',records)
write(ROOT/'delivery_manifest.json',{
 'date':'2026-11-04','status':'complete','analysis_year':2021,
 'primary_probability_matrix':'02_probability_candidates/recommended_full_quadratic_matrix.csv',
 'probability_basis':['1','age/100','area/100','age*area/10000','age^2/10000','area^2/10000'],
 'application_domain':{'age_years':[15,135],'area_m2':[40,136]},
 'probability_nodes':125,'heating_groups':4,'electricity_groups':5,'appendix_assignments':160,
 'report_pages':9,'python':sys.executable,
 'checks':{'previous_package_files_unchanged':len(previous),'protected_2013_files_unchanged':len(original),
 'new_probability_artifacts_unchanged':len(frozen),'independent_probability_check':'05_review/probability_check.json',
 'independent_energy_check':'05_review/compact_check.json','independent_report_check':'05_review/final_report_check.md',
 'visual_review':'All nine final pages inspected'},
 'selection':'Energy: smallest K within one-SE of best five-fold RMSE and at least95% between-cell mean variation retained; design choice, conditional on fixed demographic labels.',
 'references':records,'report_sha256':qa['report_sha256']})
archive=ROOT/'Italian_user_archetypes_2021_revised_results.zip'
def include(p):
 rel=p.relative_to(ROOT)
 return p.is_file() and not any(t in {'__pycache__','qa','qa_temp','.deps','.scientific'} for t in rel.parts) and p.suffix not in {'.zip','.pyc'} and p.name not in {'PACKAGE_MANIFEST.json','package_verification.json'}
members=sorted((p for p in ROOT.rglob('*') if include(p)),key=lambda p:p.relative_to(ROOT).as_posix())
index={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in members}
write(ROOT/'PACKAGE_MANIFEST.json',{'date':'2026-11-04','files':index})
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in members+[ROOT/'PACKAGE_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,v in index.items():assert hashlib.sha256(z.read(name)).hexdigest()==v['sha256'],name
result={'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
 'members_verified':len(members)+1,'previous_package_files_unchanged':len(previous),'2013_protected_files_unchanged':len(original)}
write(ROOT/'package_verification.json',result);print(json.dumps(result,indent=2))
