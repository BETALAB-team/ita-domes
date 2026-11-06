"""Create inspectable v2 adapter while preserving exact v1 source/protocol snapshots."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
original=ROOT/'04_analysis/run_padova_pilot.py'
text=original.read_text()
(ROOT/'05_simulations/padova_pilot_v1/adapter_snapshot.py').write_bytes(original.read_bytes())
(ROOT/'05_simulations/padova_pilot_v1/protocol_pre_run_snapshot.md').write_bytes((ROOT/'03_methods/simulation_protocol.md').read_bytes())
text=text.replace("RUN=ROOT/'05_simulations/padova_pilot_v1'", "SITE=sys.argv[1]\nassert SITE in ['Padova','Bolzano','Roma','Palermo']\nRUN=ROOT/f'05_simulations/climate_campaign_v2/{SITE}'")
text=text.replace("WX=Path(r'C:\\Works\\EUReCA\\Fork-Macovei\\EUReCA-maco\\eureca_ubem\\Input\\tmy_45.408_11.873_2005_2023.epw')", "WX=ROOT/f'05_simulations/climate_inputs_v2/{SITE}.epw'")
text=text.replace("    os.chdir(RUN)", "    os.chdir(RUN)\n    (RUN/'adapter_snapshot.py').write_bytes(Path(__file__).read_bytes())\n    for pname in ['simulation_protocol.md','simulation_protocol_v2_addendum.md']:\n        (RUN/pname).write_bytes((ROOT/'03_methods'/pname).read_bytes())")
text=text.replace("    response=audit['selected_survey_response']", "    audit['sha256']={k:v for k,v in audit['sha256'].items() if not k.endswith('.epw')}\n    audit['sha256'][str(WX)]=sha(WX)\n    audit['sha256'][str(WX.with_suffix('.json'))]=sha(WX.with_suffix('.json'))\n    response=audit['selected_survey_response']")
text=text.replace("'protocol':'03_methods/simulation_protocol.md'", "'site':SITE,'protocol':'03_methods/simulation_protocol_v2_addendum.md'")
text=text.replace("'protocol_sha256':sha(ROOT/'03_methods/simulation_protocol.md')", "'protocol_sha256':sha(RUN/'simulation_protocol.md'),'protocol_v2_sha256':sha(RUN/'simulation_protocol_v2_addendum.md'),'package_source_sha256':{str(p):sha(p) for p in sorted((EURECA/'eureca_building').glob('*.py'))}")
(ROOT/'04_analysis/run_climate_campaign_v2.py').write_text(text)
print('Prepared v2 adapter with frozen v1 snapshots')
