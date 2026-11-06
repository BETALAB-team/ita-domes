"""Maintain current coordination records; preserve historical snapshots."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
history = ROOT / '01_plan' / 'coordination_history'
history.mkdir(exist_ok=True)
for name in ('STATUS.md', 'agent_manifest.md', 'ARTIFACT_INDEX.md'):
    target = history / ('through_cycle03_' + name)
    if not target.exists():
        shutil.copy2(ROOT / name, target)

(ROOT / 'STATUS.md').write_text('''# Current status — 20 October 2026

Three complete serial claim-review-revision cycles are recorded. Frozen v03 has 41 claims and SHA-256 c96e3307de9b9587cc554c076f5d3527cc3ce8a6c29b12342777c07bc93c5eb2. Fresh reviewer /root/review_04 is examining v03 and primary evidence; prior verdicts and private author reasoning are withheld until its independent critique is saved. The prose gate remains closed.

The new EUReCA experiment uses one assumed 80 m² dwelling, one unweighted Veneto 2013 response, three constructed high-setpoint placements and four coordinate TMY exposures (Padova first, then Bolzano, Roma and Palermo). High-setpoint hours do not identify heater runtime. The current v5 campaign contains 20 annual runs: 12 primary comparisons, four repeats and four longer-warm-up checks. One separate seeded replay is byte-identical. All 66 earlier annual executions are quarantined and preserved, giving 87 article executions in total. Five separate reviewer diagnostic reproductions are outside that count.

Technical review 03 independently verified v5 surface forcing, state balances and output arithmetic. Full scientific review 04 and final revision closure remain required. Native annual-mean sky exchange, solar-time approximations, the fixed season boundary and uncharacterized physical inputs constrain interpretation. Bolzano's common maximum occurs at modeled season restart. Independent whole-building benchmarking and compatible measured validation remain unperformed. No population, empirical accuracy, equipment-sizing or retrofit effect is established.

Active work: /root/claim_author (gpt-6-sol requested) owns claims/responses/registers; /root/review_04 (fresh gpt-6-astra requested) owns its independent critique; /root/context_evidence (gpt-6.1-sol requested) extends local source aliases. Model-selection arguments were accepted, but backend-confirmed identity is unknown. Completed methods/literature/reviewer agents remain available for bounded followup. No prose or figures have yet been authored.

Next: save review04; author v04/response04/register04; Astra verifies final critical/major closure. Only after four complete cycles and the explicit evidence gate dispatch separate gpt-6.1-sol prose and visualization agents. Complete source, schema, link and final prose/figure fidelity audits. User clarification confirms new simulation work, context-selected areas and constructed inputs; no additional question is pending.

Historical checkpoints are preserved in 01_plan/coordination_history and status_history_through_cycle02.md. Current scientific records: 03_methods/methods_execution_report.md, 04_analysis/simulation_results.md, 05_simulations/run_inventory.csv. Full delivery navigation: ARTIFACT_INDEX.md.
''', encoding='utf-8')

manifest = '''# Agent manifest

Date: 20 October 2026. These are actual subagents in this chat. All requested model arguments were accepted using fresh contexts. The tools expose agent identifiers but no independent backend identity/version: **confirmed model metadata is unknown for every agent**. Maximum concurrent agents: four including the orchestrator. Earlier coordination snapshots are preserved in 01_plan/coordination_history.

| Actual agent identifier | Requested model | Assignment / input | Owned outputs | Current status |
|---|---|---|---|---|
| /root | Inherited; unknown | Human request; coordination and gates | 01_plan, STATUS, manifest, index | Active |
| /root/context_evidence | gpt-6.1-sol | Complete target chapter; local provenance; survey; v03 source aliases | 00_context; supplemental source registry | v03 registry followup active |
| /root/literature | gpt-6.1-sol | Targeted primary-source literature and official documentation | 02_evidence literature/bibliography | Complete, 17 primary records |
| /root/simulation_methods | gpt-6.1-sol | Protocols; new Padova-first campaigns; bounded local corrections | 03_methods, 04_analysis, 05_simulations | Handoff complete; 87 article annual executions |
| /root/claim_author | gpt-6-sol | Sole author v00–v03, responses and registers | 06_claim_drafts; 07_reviews responses/registers | v03 frozen; awaiting review04 |
| /root/review_01 | gpt-6-astra | Fresh v00; full rubric, source/scope emphasis | review01 and reviewer_01 helpers | Complete; later closure separate |
| /root/review_02 | gpt-6-astra | Fresh v01; full rubric, methods emphasis | review02 and reviewer_02 helpers | Complete; prior-issue closure separate |
| /root/review_03 | gpt-6-astra | Fresh v02; full rubric, simulation emphasis | review03, reviewer_03 helpers, separate closure/supplement | Complete; independently verified v5 |
| /root/review_04 | gpt-6-astra | Fresh v03 SHA c96e3307de9b9587cc554c076f5d3527cc3ce8a6c29b12342777c07bc93c5eb2; full red team | review04 and reviewer_04 helpers | Active; no earlier verdicts/private reasoning supplied |

## Independence and artifact accounting

Reviews 01–04 started fresh with primary evidence, claim/source ledgers, questions and the full rubric. Earlier reviews, responses, registers and private drafting reasoning were withheld until each initial critique was saved. Verbatim method excerpts omit historical review narratives while preserving technical content; code and data remain unchanged. Separate later closure passes do not replace review cycles.

Three cycles are complete: review01→v01/response01/register01; review02→v02/response02/register02; review03→v03/response03/register03. Exact input/output hashes are in 01_plan/workflow_audit.json and each review/register. Four cycles plus final closure are required before prose.

New article executions: 5 v1 +20 v2 +20 v3 +21 v4 (including replay), all quarantined; 20 v5 campaign cases plus one separate v5 replay. Total87. Reviewer01 independently reproduced five annual files as diagnostics, outside that count. Current v5 numerical verification is not empirical validation. External checkouts/environments were not changed.

Separate prose and visualization agents have not yet been dispatched. They will each request gpt-6.1-sol after the gate. No generative-image tool is permitted anywhere in this workflow.
'''
(ROOT / 'agent_manifest.md').write_text(manifest, encoding='utf-8')

p=ROOT/'ARTIFACT_INDEX.md'
s=p.read_text(encoding='utf-8-sig')
s=s.replace('Current candidate v4 simulation results', 'Current candidate v5 simulation results').replace('v4 paired contrasts','v5 paired contrasts').replace('paired_schedule_contrasts_v4.csv','paired_schedule_contrasts_v5.csv').replace('posthoc_setpoint_partition_v4.md','posthoc_setpoint_partition_v5.md').replace('run_climate_campaign_v4.py','run_climate_campaign_v5.py')
s=s.replace('All 45 original v1/v2/v3 annual outputs remain quarantined. V4 has separate immutable input/configuration/source snapshots and is undergoing independent review. Its 20 campaign cases comprise 12 primary cases and 8 technical checks; one additional isolated replay checks the new-destination interface.', 'All 66 pre-v5 annual outputs remain quarantined. V5 has separate immutable input/configuration/source snapshots:20 campaign cases (12 primary, four repeat and four warm-up checks) plus one isolated replay. Technical review03 passed; fresh full-claim review04 is underway. Total87 article annual executions.')
s=s.replace('- [Methods execution report]', '- [Per-surface correction protocol](03_methods/surface_correction_protocol_v5.md), [surface adapter](04_analysis/surface_forcing_1c.py), [independent surface checker](04_analysis/check_surface_forcing_v5.py), and [non-destructive reproduction](03_methods/v5_reproduction.md).\n- [Methods execution report]')
p.write_text(s,encoding='utf-8')

# Preserve the initial DAG while maintaining current task status.
p=ROOT/'01_plan/task_dag.md'
snap=history/'task_dag_initial.md'
if not snap.exists(): shutil.copy2(p,snap)
s=p.read_text(encoding='utf-8-sig').replace('Initial version; scientific question remains provisional until discovery is reconciled.', 'Current execution checkpoint: three complete cycles; review04 active. Novelty remains provisional.')
statuses={'A1':'Complete; target chapter and page images audited','A2':'Complete; prior results not imported as new evidence','A3':'Complete targeted search; provisional novelty','A4':'Complete; empirical references remain unverified','A5':'Complete; fixed-case schedule placement selected','B1':'Complete; pre-run protocols frozen','B2':'Complete; v5 Padova gate passed','B3':'Complete; four coordinate TMY exposures, v5 only','B4':'Implementation checks complete; independent benchmark and empirical validation pending','C0':'Complete; v00 preserved','C1':'Complete: review01 + response01 + v01 + register01','C2':'Complete: review02 + response02 + v02 + register02','C3':'Complete: review03 + response03 + v03 + register03','C4':'Running fresh independent review04 on v03','C5':'Waiting for v04 and fourth-cycle completion','D1':'Waiting for explicit gate','D2':'Waiting for prose figure briefs','D3':'Waiting for prose and figures'}
out=[]
for line in s.splitlines():
    if line.startswith('| '):
        cells=line.split('|')
        for key,value in statuses.items():
            if cells[1].strip().startswith(key+' '):
                cells[-2]=' '+value+' '
                line='|'.join(cells)
                break
    out.append(line)
p.write_text('\n'.join(out)+'\n',encoding='utf-8')
