from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'04_analysis/summarize_corrected_campaign_v3.py').read_text(encoding='utf-8')
text=text.replace('climate_campaign_v3','climate_campaign_v4').replace('contrasts_v3','contrasts_v4').replace('partition_v3','partition_v4').replace('run_inventory_v3','run_inventory_v4').replace('checks_v3','checks_v4').replace('interactions_v3','interactions_v4')
start=text.index("old_path=ROOT/'05_simulations/run_inventory.csv'")
end=text.index("(ROOT/'04_analysis/independent_output_checks_v4.json')",start)
text=text[:start]+'''old_path=ROOT/'05_simulations/run_inventory.csv'
old=read_csv(old_path)
old=[r for r in old if r['campaign'] not in ['climate_campaign_v4','climate_campaign_v4_replay']]
assert len(old)==45
replay_dir=ROOT/'05_simulations/climate_campaign_v4_replay/Padova'
replay_summary=json.loads((replay_dir/'summary.json').read_text())[0]
assert json.loads((replay_dir/'full_step_balance_checks.json').read_text())['pass']
assert replay_summary['hourly_sha256']==next(x for x in inventory if x['site']=='Padova' and x['case']=='seeded')['hourly_sha256']
replay={'campaign':'climate_campaign_v4_replay','site':'Padova','case':'seeded','purpose':'technical_replay_not_scientific_sample','warmup_hours':168,'hourly_path':str(replay_dir/'seeded_hourly.csv'),'hourly_sha256':replay_summary['hourly_sha256'],'annual_sensible_heating_kWh':replay_summary['annual_sensible_heating_kWh'],'peak_sensible_heating_kW':replay_summary['peak_sensible_heating_kW'],'scientific_disposition':'separate genuine non-destructive replay; not an additional scientific sample'}
all_inventory=[{**x,'scientific_disposition':'QUARANTINED: original dispatcher and/or invalid opaque-sky forcing'} for x in old]+inventory+[replay]
assert len(all_inventory)==66
save_csv(ROOT/'05_simulations/run_inventory.csv',all_inventory)
''' +text[end:]
text=text.replace("paired_schedule_contrasts_v2.csv","paired_schedule_contrasts_v3.csv")
text=text.replace('quarantined_v2_heating','quarantined_v3_heating').replace('corrected_v3_heating','corrected_v4_heating').replace('v3_minus_v2','v4_minus_v3').replace('dispatcher_correction_diagnostic_v2_v3','weather_correction_diagnostic_v3_v4')
text=text.replace('Corrected new EUReCA simulation outputs — v3','Corrected forcing and stepping outputs — v4')
text=text.replace('All original 25 v1/v2 annual outputs remain quarantined because the original recursive dispatcher violates timestep state balances. Use only v3 tables for current conditional simulation claims.', 'All earlier 45 annual outputs are quarantined: v1/v2 have dispatcher and sky-forcing defects; v3 fixes stepping but retained invalid opaque-sky forcing. Use only v4 tables for current conditional simulation claims. The valid horizontal IR now supplies effective sky temperature and the native annual-mean longwave scalar; no cloud filling is used.')
text=text.replace('Across all attempts, 45 new annual simulations were executed: 25 quarantined and 20 corrected.', 'Across all attempts, 66 annual runs were executed: 45 quarantined, 20 current campaign cases, and one separately identified genuine seeded replay in a fresh directory. Replay matches the current Padova seeded hourly file byte-for-byte and is not an extra scientific sample.')
text=text.replace('match v2. The local correction changes only state stepping:', 'match v3. The documented forcing correction replaces the invalid cloud-derived scalar with the mean air-minus-effective-sky temperature derived from valid horizontal IR. Native annual-mean longwave and solar-time approximations remain limitations. The retained local stepping correction uses:')
text=text.replace('recomputed only from corrected v3 files','recomputed only from current v4 files').replace('recomputed from v3 files','recomputed from v4 files').replace('Post-hoc corrected setpoint partition — v3','Post-hoc corrected setpoint partition — v4')
text=text.replace('quarantined_v2_results_report_snapshot.md','quarantined_v3_results_report_snapshot.md')
text=text.replace('correction_protocol_v3.md; single_commit_1c.py; test_single_commit_1c.py; run_climate_campaign_v3.py; check_full_step_balances_v3.py; summarize_corrected_campaign_v3.py.', 'weather_correction_protocol_v4.md; correction_protocol_v3.md; ir_sky_forcing.py; single_commit_1c.py; run_climate_campaign_v4.py; check_full_step_balances_v4.py; summarize_corrected_campaign_v4.py.')
# Generated report remains a clear current record; preserved older files retain
# their original contents and quarantine disposition in the run inventory.
text=text.replace("report.extend(['',f'All 175,200", "report.extend(['','Active forcing fields and IR/JSON mapping passed strict QC. Palermo has one preserved invalid wind record, unused in this guarded fixed-ACH pathway; the weather file is not claimed universally valid. Native annual-mean longwave and unadjusted solar-time offsets are explicit model assumptions. No empirical validation follows.',f'All 175,200")
(ROOT/'04_analysis/summarize_corrected_campaign_v4.py').write_text(text,encoding='utf-8')
print('Prepared v4 summaries and separate replay inventory')
