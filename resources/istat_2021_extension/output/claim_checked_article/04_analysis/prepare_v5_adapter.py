from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'04_analysis/run_climate_campaign_v4.py').read_text(encoding='utf-8')
text=text.replace("    weather_audit=ROOT/", "    fixtures=json.loads((ROOT/'05_simulations/surface_forcing_v5_fixture_checks.json').read_text())\n    assert fixtures['pass'] and fixtures['source_sha256']==sha(ROOT/'04_analysis/surface_forcing_1c.py')\n    weather_audit=ROOT/",1)
text=text.replace("'weather_correction_protocol_v4.md']", "'weather_correction_protocol_v4.md','surface_correction_protocol_v5.md']")
text=text.replace("    audit=json.loads", "    (RUN/'surface_forcing_1c.py').write_bytes((ROOT/'04_analysis/surface_forcing_1c.py').read_bytes())\n    (RUN/'surface_fixture_checks_pre_run.json').write_bytes((ROOT/'05_simulations/surface_forcing_v5_fixture_checks.json').read_bytes())\n    audit=json.loads",1)
text=text.replace("    design['weather_correction']='", "    design['surface_correction']='each external surface initializes its own area, plane irradiance, construction and radiative coefficient'\n    design['weather_correction']='",1)
text=text.replace("'weather_correction_sha256':", "'surface_preprocessing_sha256':sha(RUN/'surface_forcing_1c.py'),'surface_protocol_sha256':sha(RUN/'surface_correction_protocol_v5.md'),'fixture_sha256':sha(RUN/'surface_fixture_checks_pre_run.json'),'weather_correction_sha256':",1)
text=text.replace("    from single_commit_1c import", "    from surface_forcing_1c import preprocess_1c_surface_loads,check_surface_order_invariance\n    from single_commit_1c import",1)
text=text.replace("        z.calculate_zone_loads_ISO13790(weather)", "        preprocess_1c_surface_loads(z,weather,RUN)\n        write(RUN/f'{name}_surface_order_checks.json',check_surface_order_invariance(z,weather))")
(ROOT/'04_analysis/run_climate_campaign_v5.py').write_text(text,encoding='utf-8')
print('Prepared v5; source surface forcing and all pre-run checks are frozen per site')
