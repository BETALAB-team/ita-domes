"""Preserve first delivery and record the human-authorized archetype extension."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
history=ROOT/'01_plan/delivery_history/pre_archetypes_20261020'
history.mkdir(parents=True,exist_ok=True)
files=['STATUS.md','ARTIFACT_INDEX.md','agent_manifest.md','01_plan/delivery_manifest.json',
       '01_plan/prose_gate.json','01_plan/research_questions.md','01_plan/decisions.md',
       '06_claim_drafts/article_claims_final.md','06_claim_drafts/final_acceptance.md',
       '06_claim_drafts/claim_evidence_ledger.json','07_reviews/final_closure.md',
       '07_reviews/issue_register_final.json','07_reviews/issue_register.json','07_reviews/issue_register.md']
records=[]
for rel in files:
    p=ROOT/rel
    dst=history/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists(): shutil.copy2(p,dst)
    records.append({'path':rel,'preserved_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
for folder in ('08_manuscript','09_figures'):
    dst=history/folder
    if not dst.exists():
        shutil.copytree(ROOT/folder,dst,ignore=shutil.ignore_patterns('.mplconfig','__pycache__'))
(history/'snapshot_manifest.json').write_text(json.dumps({'date':'2026-10-20','reason':'Preserve first delivery before user-requested archetype implementation and literature expansion.','files':records},indent=2),encoding='utf-8')
plan=ROOT/'01_plan/archetype_extension'
plan.mkdir(exist_ok=True)
(plan/'user_steering.md').write_text('''# User-authorized extension — 20 October 2026

The user asks: “ok i would like to have the user archetypes implemented soemwhere in it and obviously. get the agents up and running again. also the lit review is too scarce”. This steers the ongoing CLIMA journal-extension project: implement the existing published user archetypes in the analysis/simulations and substantially strengthen the literature synthesis. It does not authorize new sidebar chats, external contact, publication or a switch to generic clustering.

Earlier instructions persist: actual specialist agents with requested model assignments, claim-first authoring and independent review, no generative images, traceable new simulations, explicit missing-validation boundaries. The earlier four cycles cover the prior scope; they do not approve new archetype claims. Additional fresh scientific review and revision are required before updated prose.

Prior delivery is preserved in ../delivery_history/pre_archetypes_20261020. Existing v00–v04, reviews01–04, immutable run snapshots and v5 outputs remain unchanged. New worker artifacts use archetype_extension subdirectories and new campaign identities. Current final/prose pointers will change only after the extension's review gate, preserving previous snapshots.
''',encoding='utf-8')
(plan/'task_dag.md').write_text('''# Archetype implementation and literature extension DAG

| Task | Owner / requested model | Dependencies | Outputs / acceptance | Status |
|---|---|---|---|---|
| E1 Published archetypes and assignments | /root/context_evidence / gpt-6.1-sol | Full primary paper, original code/data | Page-located four-group definition, exact versus reconstructed assignment provenance; no label-invented schedules | Active |
| E2 Literature expansion | /root/literature / gpt-6.1-sol | Existing evidence plus current primary-source search | Critical thematic synthesis, source/claim/nearest-work matrix, verified locators/metadata/access depth; no reference padding | Active |
| E3 Executable design | /root/simulation_methods / gpt-6.1-sol | E1 plus existing corrected EUReCA path | Frozen archetype-to-input mapping, controlled comparator, adequate variability design, Padova-first gates and run count | Discovery active; annual runs wait for reconciled protocol |
| E4 New simulations and analyses | /root/simulation_methods / gpt-6.1-sol | E3 freeze | New configs/inputs/code/hashes/runs/checks; no old physical results substituted; interpretation matches selection/weighting | Waiting |
| E5 Extended claim checker | /root/claim_author / gpt-6-sol | E1/E2/E3 and checked E4 evidence | Immutable v05 onwards, stable existing IDs/new atomic claims, merged source ledger and current evidence boundaries | Waiting |
| E6 Fresh review and substantive revision | New independent gpt-6-astra contexts; claim author | E5 | Full rubric on archetypes/literature/new outputs, independent recalculation, responses/registers, final critical/major closure; prior four rounds not reused as approval | Waiting |
| E7 Prose and new figure briefs | Separate /root/prose_author / gpt-6.1-sol | E6 gate | Substantial literature sections and actual implemented archetype methods/results; complete paragraph/caption claim map | Waiting |
| E8 Figures and integration | Separate /root/visualization / gpt-6.1-sol, then prose owner | E7 briefs | Reproducible SVG/PNG/code/data/visual QA; final fidelity/source/link audit | Waiting |

Original CLIMA chapter remains the foundation. Published2025 archetypes supply an explicit user-representation extension, rather than rediscovery or a decorative mention. Physical-building and demographic-user archetypes must be distinguished. The selected single-case placement experiment may become a controlled component; its evidence is retained without assuming representativeness. New outcomes will not determine representative selection or change predeclared endpoints. Missing empirical validation remains explicit.
''',encoding='utf-8')
(ROOT/'STATUS.md').write_text('''# Active extension — 20 October 2026

The user reopened the manuscript to implement the published user archetypes and strengthen the scarce literature review. The previous first delivery is preserved under01_plan/delivery_history/pre_archetypes_20261020. Original four completed claim cycles and v5 evidence remain historical accepted work; they do not approve new archetype claims.

Three actual specialist agents have restarted in parallel: context_evidence is recovering the full primary archetype paper and original assignment provenance; literature is expanding the primary-source critical synthesis; simulation_methods is specifying the executable archetype-to-EUReCA bridge and small adequate matched experiment. Requested models remain gpt-6.1-sol; confirmed backend identity unknown. Root coordinates. Claim author, fresh Astra reviewers, separate prose and visualization owners follow dependencies. No annual extension run is authorized before the reconciled pre-run protocol.

New artifacts use00_context/archetype_extension,02_evidence/archetype_extension,03_methods/archetype_extension,04_analysis/archetype_extension and new simulation campaign IDs. Prior data, checkouts, runs, drafts and ledgers are preserved. No user question is pending. The full2025archetype paper was located in Downloads; original local clustering versions still require source/assignment audit.

Next: reconcile recovered archetypes and evidence with implementation options; freeze design; construct inputs and run Padova-first extension, then selected weather exposures; extend claim checker and source registry; perform fresh independent scientific review/revision/closure; only then revise English prose and figures. See01_plan/archetype_extension/task_dag.md and user_steering.md. No image-generation tools, publication or external contact.
''',encoding='utf-8')
with (ROOT/'agent_manifest.md').open('a',encoding='utf-8') as f:
    f.write('''\n## User-requested archetype and literature extension — active 20 October 2026\n\nThe human reopened the completed first-draft workflow. Three actual followup tasks restarted the existing evidence, literature and methods agents with their original requested gpt-6.1-sol model routing; backend confirmation remains unknown. E1 owns00_context/archetype_extension; E2 owns02_evidence/archetype_extension; E3/E4 own new methods/analysis/campaign directories. Claims, new independent reviewers, prose and figures follow the new DAG. Prior completion statements above refer to the preserved first delivery, not this active extension.\n''')
print('Previous delivery preserved; archetype/literature extension recorded.')

if __name__=='__main__':
    pass
