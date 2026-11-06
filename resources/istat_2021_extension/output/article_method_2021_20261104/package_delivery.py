"""Package verified 2021 results without modifying the historical sources."""
from pathlib import Path
import csv, hashlib, json, shutil, sys, zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASE = REPO / 'output/temporal_extension_2013_2021_20261104/00_baseline'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, d: p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')

baseline = json.loads((BASE/'baseline_freeze.json').read_text())['sources']
assert all(sha(Path(s['path'])) == s['sha256'] for s in baseline)
frozen = json.loads((ROOT/'02_probabilities/artifact_hashes.json').read_text())
assert all(sha(ROOT/'02_probabilities'/p) == h for p,h in frozen.items())
provenance = json.loads((ROOT/'01_inputs/input_provenance.json').read_text())
assert all(sha(Path(p)) == h for p,h in provenance['files'].items())

# Preserve all new assignment rows, retaining only the fields needed for reuse.
columns = ['id','weight','region','location','dwelling_type_code_2021','dwelling_type',
           'archetype','H_group','E_group','heating_daily_hours','electricity_annual_kwh',
           'cooling_system_use','building_age_years','dwelling_area_m2',
           'lower_scenario_kwh','upper_scenario_kwh']
compact = ROOT/'03_energy/household_group_assignments_2021_compact.csv'
with (ROOT/'03_energy/analysis_households_2021.csv').open(newline='',encoding='utf-8-sig') as f, compact.open('w',newline='',encoding='utf-8') as out:
    reader=csv.DictReader(f)
    assert set(columns).issubset(reader.fieldnames)
    writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
    count=0
    for row in reader:
        writer.writerow({k:row[k] for k in columns});count+=1
assert count == 52617

reference = ROOT/'07_source_reference';reference.mkdir(exist_ok=True)
names = ['01_PCA_Exploratory.py','02_Families_Analysis.py','03_PCA_Clustering.py',
         '05_00_PFE_Exploratory.py','05_01_PFE_Variable_Analysis.py',
         '05_02_PFE_Probablity.py','05_03_PFE_Analysis.py','07_ANOVA_analysis.py',
         'pfe_pipeline.py','energy_analysis.py','model_electricity_2021.py']
source_records=[]
for name in names:
    source=REPO/name;target=reference/name
    shutil.copyfile(source,target)
    assert sha(source)==sha(target)
    source_records.append({'original':str(source),'copy':str(target.relative_to(ROOT)),'sha256':sha(source)})
for name in ['baseline_freeze.json','published_benchmarks.json','published_benchmarks.csv','archive_transfer_parameters.json']:
    shutil.copyfile(BASE/name,reference/name)
shutil.copyfile(REPO/'revised_paper_outputs/appliance_audit/electricity_model_report.json',reference/'electricity_model_report.json')
write(reference/'source_copy_manifest.json',source_records)

qa=ROOT/'06_report/qa/verification.json'
checks=json.loads(qa.read_text())
assert checks['pages']==11 and checks['pngs']==11
assert not checks['missing_docx'] and not checks['missing_pdf']
assert not checks['changed_probability_artifacts'] and not checks['changed_original_sources']
checks['visual_review']='passed: all 11 final rendered pages inspected; equation, group plots and editable Appendix A verified'
checks['report_sha256']={p.name:sha(p) for p in (ROOT/'06_report').glob('Italian_user_archetypes_2021_reconstruction.*')}
write(qa,checks)
write(ROOT/'06_report/delivery_checks.json',checks)

manifest={
 'date':'2026-11-04','status':'complete','analysis_year':2021,
 'python':sys.executable,'scientific_dependencies':'requirements-2021.txt',
 'requested_outputs':{'probabilities':'02_probabilities/probability_grid_empirical.csv',
   'matrix':'02_probabilities/polynomial_coefficient_matrix.csv',
   'heating_groups':9,'electricity_groups':13,'appendix_assignment_cells':160,
   'figures':'04_figures','editable_report':'06_report/Italian_user_archetypes_2021_reconstruction.docx',
   'pdf_report':'06_report/Italian_user_archetypes_2021_reconstruction.pdf'},
 'checks':{'protected_original_files_unchanged':len(baseline),
   'frozen_probability_artifacts_unchanged':len(frozen),'source_provenance_unchanged':True,
   'independent_scientific_review':'05_review/scientific_review.md',
   'independent_report_review':'05_review/final_report_check.md','rendered_pages_visually_checked':11,
   'compact_assignment_rows':count},
 'regional_correction':'No separate physical energy multiplier was located. Multiplier 1; official PESO weights supplied as separate diagnostics.',
 'source_copies':source_records,
 'large_local_files_omitted_from_zip':[]
}
excluded={'01_inputs/analysis_input_2021.csv','03_energy/analysis_households_2021.csv',
          '03_energy/H_household_assignments_2021.csv','03_energy/E_household_assignments_2021.csv'}
for rel in sorted(excluded):
    p=ROOT/rel
    manifest['large_local_files_omitted_from_zip'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
write(ROOT/'delivery_manifest.json',manifest)

archive=ROOT/'Italian_user_archetypes_2021_results.zip'
def include(p):
    rel=p.relative_to(ROOT)
    return (p.is_file() and rel.as_posix() not in excluded
        and not any(v in {'.deps','.scientific','__pycache__','qa','qa_temp'} for v in rel.parts)
        and p.suffix not in {'.zip','.pyc'} and p.name not in {'PACKAGE_MANIFEST.json','package_verification.json'})
members=sorted((p for p in ROOT.rglob('*') if include(p)),key=lambda p:p.relative_to(ROOT).as_posix())
index={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in members}
write(ROOT/'PACKAGE_MANIFEST.json',{'date':'2026-11-04','files':index})
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in members+[ROOT/'PACKAGE_MANIFEST.json']:
        z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,record in index.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256'],name
write(ROOT/'package_verification.json',{'status':'passed','archive':archive.name,
 'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'members':len(members)+1,
 'manifest_entries_verified':len(index),'zip_integrity':'passed'})
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,
                 'members_verified':len(members)+1,'source_integrity':'passed'},indent=2))
