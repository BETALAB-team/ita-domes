from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'04_analysis/summarize_corrected_campaign_v4.py').read_text(encoding='utf-8')
text=text.replace('climate_campaign_v4','climate_campaign_v5').replace('contrasts_v4','contrasts_v5').replace('partition_v4','partition_v5').replace('run_inventory_v4','run_inventory_v5').replace('checks_v4','checks_v5').replace('interactions_v4','interactions_v5')
text=text.replace('assert len(old)==45','assert len(old)==66').replace('assert len(all_inventory)==66','assert len(all_inventory)==87')
text=text.replace('All earlier 45 annual outputs are quarantined: v1/v2 have dispatcher and sky-forcing defects; v3 fixes stepping but retained invalid opaque-sky forcing. Use only v4 tables for current conditional simulation claims. The valid horizontal IR now supplies effective sky temperature and the native annual-mean longwave scalar; no cloud filling is used.', 'All earlier 66 annual executions are quarantined: v1/v2 have dispatcher defects, v1/v2/v3 have invalid cloud forcing, and v4 still borrowed preceding-wall forcing for the zero-glazed roof. Current v5 independently initializes each surface and retains valid-IR and single-commit corrections. These are candidate results pending independent scientific review; no empirical accuracy is claimed.')
text=text.replace('Across all attempts, 66 annual runs were executed: 45 quarantined, 20 current campaign cases, and one separately identified genuine seeded replay', 'Across all attempts, 87 annual runs were executed: 66 quarantined, 20 candidate v5 campaign cases, and one separately identified genuine v5 seeded replay')
text=text.replace('match v3. The documented forcing correction replaces the invalid cloud-derived scalar with the mean air-minus-effective-sky temperature derived from valid horizontal IR.', 'match v4. The new local surface preprocessor independently computes each wall/roof plane, opaque and glazed area, shortwave and longwave term. It retains the valid-IR mean-sky correction without cloud imputation.')
text=text.replace('Corrected forcing and stepping outputs — v4','Candidate independent-surface simulation outputs — v5')
text=text.replace('Post-hoc corrected setpoint partition — v4','Post-hoc candidate setpoint partition — v5')
text=text.replace('current v4 files','candidate v5 files').replace('v4 tables','v5 candidate tables')
text=text.replace('paired_schedule_contrasts_v3.csv','paired_schedule_contrasts_v4.csv').replace('quarantined_v3_heating','quarantined_v4_heating').replace('corrected_v4_heating','candidate_v5_heating').replace('v4_minus_v3','v5_minus_v4').replace('weather_correction_diagnostic_v3_v4','surface_correction_diagnostic_v4_v5')
text=text.replace('quarantined_v3_results_report_snapshot.md','quarantined_v4_results_report_snapshot.md')
text=text.replace('Status: adapter-local correction passed declared transition checks and full annual step-balance checks.', 'Status: CANDIDATE. Local surface, sky and stepping corrections passed declared internal checks; independent scientific review of v5 remains pending.')
text=text.replace('climate_campaign_v5/Padova', 'climate_campaign_v5/Padova')
insert="""
surface_reports={}
for site in ['Padova','Bolzano','Roma','Palermo']:
    surface_reports[site]=json.loads((RUN/site/'independent_surface_forcing_checks.json').read_text())
    assert surface_reports[site]['pass']
assert json.loads((replay_dir/'independent_surface_forcing_checks.json').read_text())['pass']
(ROOT/'04_analysis/independent_surface_checks_v5.json').write_text(json.dumps(surface_reports,indent=2))
"""
text=text.replace("interactions=[]",insert+"\ninteractions=[]",1)
text=text.replace("report.extend(['','Active forcing", "report.extend(['','Every one of 43,800 exported surface-hours passes separate opaque-shortwave, longwave and window-shortwave reconstruction; all 175,200 solver hours match reconstructed node forcing. Reverse/rotate surface order preserves node forcing within 1e-8 W. Fixtures include a windowless zero-glazed roof. The preprocessing correction does not change physical areas, U-values, absorptance, glazing or network parameters.','Active forcing")
text=text.replace('Every one of 43,800 exported surface-hours','All 175,200 exported site-surface-hours across four sites')
text=text.replace(' Independent scientific review of the correction remains pending.','')
text=text.replace('check_full_step_balances_v3.py','check_full_step_balances_v4.py').replace('summarize_corrected_campaign_v3.py','summarize_corrected_campaign_v5.py')
(ROOT/'04_analysis/summarize_corrected_campaign_v5.py').write_text(text,encoding='utf-8')
v=(ROOT/'04_analysis/verify_v4_replay_interface.py').read_text(encoding='utf-8').replace('climate_campaign_v4','climate_campaign_v5').replace('run_climate_campaign_v4','run_climate_campaign_v5').replace('v4_replay_interface','v5_replay_interface')
(ROOT/'04_analysis/verify_v5_replay_interface.py').write_text(v,encoding='utf-8')
print('Prepared v5 candidate summary, inventories and refusal/replay checks')
