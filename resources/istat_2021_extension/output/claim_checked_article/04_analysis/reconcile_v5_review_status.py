"""Current canonical status only; frozen run/protocol snapshots stay untouched."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/'03_methods'/name for name in ['simulation_protocol.md','validation_protocol.md','validation_matrix.md','climate_selection.md','methods_execution_report.md']]+[ROOT/'05_simulations/SCIENTIFIC_USE_HOLD.md',ROOT/'04_analysis/simulation_results.md']
for p in files:
    text=p.read_text(encoding='utf-8')
    text=text.replace('Independent scientific review of v5 remains pending','Independent technical review 03 supplement passed; fresh full-claim review 04 remains pending')
    text=text.replace('independent scientific review of v5 remains pending','independent technical review 03 supplement passed; fresh full-claim review 04 remains pending')
    p.write_text(text,encoding='utf-8')
p=ROOT/'03_methods/v5_reproduction.md';text=p.read_text(encoding='utf-8')
text=text.replace('weather_correction_protocol_v5.md','surface_correction_protocol_v5.md').replace('check_full_step_balances_v5.py','check_full_step_balances_v4.py')
text+='\nAfter the state-balance checker, run `04_analysis/check_surface_forcing_v5.py SITE --campaign NEW_ID` for independent component and actual solver-forcing checks. The reused state-balance checker retains its v4 filename and accepts the explicit new campaign path; it does not reuse old run results.\n'
p.write_text(text,encoding='utf-8')
p=ROOT/'04_analysis/simulation_results.md';text=p.read_text(encoding='utf-8').replace('check_full_step_balances_v3.py','check_full_step_balances_v4.py').replace('summarize_corrected_campaign_v3.py','summarize_corrected_campaign_v5.py')
text+='\nTechnical-review disposition: the independent reviewer 03 v5 supplement rechecked the source, per-surface forcing and all 20 campaign outputs and passed its technical checks. The fresh full-claim review 04 and manuscript gate remain separate pending tasks. Empirical validation remains absent. See `07_reviews/v5_verification_supplement.md`.\n'
p.write_text(text,encoding='utf-8')
print('Reconciled current candidate status: technical review 03 passed; full-claim review 04 pending')
