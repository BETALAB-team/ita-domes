"""Rebuild the 2021 equipment revision using the existing frozen demographic fit.

Run after the paired analysis exists. The assertions guard reuse of unaffected
demographic/probability results. Full paired rebuilds use revised_analysis.py.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import revised_analysis as r
from revised_figures import make_all_figures
from run_revised_papers import find_pdf_python, refresh_manifest


def main():
    config = r.ROOT/'revised_paper_config.json'
    cfg = json.loads(config.read_text())
    out = r.ROOT/cfg['output_root']; folder = out/'2021'
    evidence = json.loads((out/'results.json').read_text())
    evidence['configuration'] = cfg
    model = json.loads((out/'reference_model.json').read_text())
    path = r._resolve_raw_input(r.load_config(r.ROOT/cfg['input_config']),2021)
    assert r.sha256(path) == evidence['inputs']['2021']['sha256']
    raw, extra = r.read_wave(path,2021)
    frame, _ = r.make_frame(raw,extra,path,2021,cfg)
    frame = r.assign_archetypes(frame,model).reset_index(drop=True)
    previous = pd.read_csv(folder/'households.csv')
    columns = ['id','archetype','split','weight','heating','age_proxy','area_proxy',
               'dwelling_type','location',*r.FEATURES]
    pd.testing.assert_frame_equal(frame[columns],previous[columns],check_dtype=False)
    frame.to_csv(folder/'households.csv',index=False)
    result = evidence['years']['2021']
    result['equipment'] = r.equipment_summary(frame,folder)
    result['energy']['heating'] = r.learned_groups(frame,'heating',cfg,folder)
    result['energy']['heating']['anova'] = r.energy_anova(frame,'heating',2021,folder)
    r.insulation_group_tests(folder)
    result['energy']['electricity'] = r.learned_groups(frame,'electricity',cfg,folder)
    result['energy']['electricity']['anova'] = r.energy_anova(frame,'electricity',2021,folder)
    assert result['energy']['heating']['anova']['rank'] == result['energy']['heating']['anova']['columns']
    source_fields = r.supplement(path,2021)
    result['equipment']['source_counts'] = {
        'households':len(source_fields),
        'cooling_present':int(source_fields.cooling.eq('Present').sum()),
        'improvements_reported':int(source_fields.insulation_improvement.eq('Reported').sum()),
        'improvements_unknown':int(source_fields.insulation_improvement.eq('Unknown').sum()),
        'cooling_hours_observed':int(source_fields.cooling_hours.notna().sum())}
    metadata = path.parent.parent/'METADATI'
    for name in ['ISTAT_MFR_ECH_Tracciato_2021_DELIMITED.html','ISTAT_MFR_ECH_Questionario_2021.pdf']:
        source = metadata/name
        evidence['inputs'][name] = {'path':str(source),'sha256':r.sha256(source)}
    for source in [out/'appliance_audit/electricity_2021.csv',out/'appliance_audit/electricity_model_report.json']:
        evidence['inputs'][source.name]={'path':str(source),'sha256':r.sha256(source)}
    evidence['checks'].pop('no_2021_electricity_invented',None)
    evidence['checks']['2021_electricity_is_explicitly_modeled']=True
    r.write_json(folder/'results.json',result)
    r.write_json(out/'results.json',evidence)
    print(json.dumps(result['equipment'],indent=2),flush=True)
    make_all_figures(out,model,cfg,years=(2021,))
    pdf = find_pdf_python()
    subprocess.run([str(pdf),str(r.ROOT/'build_revised_papers.py'),'--year','2021'],check=True)
    subprocess.run([str(pdf),str(r.ROOT/'verify_revised_papers.py'),'--year','2021'],check=True)
    refresh_manifest(config)


if __name__ == '__main__':
    main()
