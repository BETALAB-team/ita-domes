"""Build the extension evidence pack without modifying the preceding article/evidence."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
catalog = json.loads((BASE / 'sources.json').read_text(encoding='utf-8-sig'))
sources = catalog['sources']
for s in sources:
    s['origin_record'] = '../sources.json#' + s['id']
    s['access_date'] = '2026-10-20'
    s['status'] = 'verified_for_bounded_claims'
    s['review_role'] = 'core or contextual; select by claim rather than cite automatically'
arch = next(s for s in sources if s['id'] == 'SRC-ARCH2025')
arch['access_depth'] = 'Complete local 15-page primary PDF available; selected complete methods/results/conclusion pages directly read with pypdf; no new quantitative table transcription in this pack'
arch['local_path'] = r'C:\Users\khajmoh18975\Downloads\1-s2.0-S0378778825006541-main.pdf'
arch['locator'] = 'Sections 3.1-3.3 pp.3-5; equations1-2 p.6; section4.3 pp.7-8; section4.4 pp.8-10; section6 p.13; AppendixA p.14'
arch['limitations'] = 'Declared heating on-hours and modeled appliance electricity are not energy meters. Section5 accuracy expectations are not completed dynamic-simulation validation. Table/text inconsistencies and assignment definitions require the context-owned implementation audit.'
sources += json.loads((ROOT / 'new_source_records.json').read_text(encoding='utf-8-sig'))
rug = next(s for s in sources if s['id'] == 'AE-SRC-RUGANI2024')
rug.update(authors=['Roberto Rugani','Marco Picco','Giacomo Salvadori','Fabio Fantozzi','Marco Marengo'],
    url='https://www.sciencedirect.com/science/article/pii/S0378778824002305',
    access_depth='Primary publisher abstract/highlights plus author-institution bibliographic record; institutional fullPDF fetch403',
    locator='Publisher Abstract/Highlights; Brighton and Pisa author-institution records; primary paper cited as ARCH2025 reference14',
    supports='Controlled comparison of six occupancy-profile databases in EnergyPlus across three city climates and nine building uses.',
    limitations='Abstract-level methodological appraisal; database inputs are bundled. Do not import extreme percentages, claim causal mechanisms, or infer independent measured accuracy.',
    status='verified_for_bounded_claims')
for sid in ['SRC-RAMOS2026','SRC-ISTAT-MFR','SRC-ISTAT2021']:
    next(s for s in sources if s['id']==sid)['review_role'] = 'optional temporal/data-access background; not needed for the primary archetype-implementation argument'
for s in sources:
    if s['id']=='AE-SRC-DEJAEGER2020': s['number']='3'
    if s['id']=='AE-SRC-URBEM2025': s['number']='22'
    if s['id']=='AE-SRC-WIDEN2025': s['number']='4'
    if s['id']=='AE-SRC-RICHARDSON2010': s['number']='10'
    if s['id']=='AE-SRC-WIDEN2010': s['number']='6'
    if s['id']=='SRC-KOENE2023':
        s['authors']=['F. G. H. F. Koene','B. B. Eslami-Mossallam']
        s['metadata_note']='Initials corrected in this extension from the directly inspected TNO author listing; original catalog preserved.'

(ROOT/'new_source_records.json').write_text(json.dumps([s for s in sources if s['id'].startswith('AE-SRC-')],indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

claims=json.loads((ROOT/'claim_records.json').read_text(encoding='utf-8-sig'))
index={s['id']:s for s in sources}
assert len(index)==len(sources), 'Duplicate source ID'
assert len({c['id'] for c in claims})==len(claims)
for c in claims:
    assert all(s in index for s in c['sources'])
    c['status']='supported'
    c['evidence_class']='published finding' if c['kind']=='literature' else 'hypothesis'
    c['provenance']=[{'source_id':sid,'locator':index[sid]['locator'],'access_depth':index[sid]['access_depth']} for sid in c['sources']]
    c['counterevidence']=[index[sid]['limitations'] for sid in c['sources']]
for s in sources:
    s['supports_claims']=[c['id'] for c in claims if s['id'] in c['sources']]

result={'schema_version':'archetype-extension-1.0','access_date':'2026-10-20',
    'scope':'Targeted critical narrative synthesis; original primary studies and official documentation; no systematic/exhaustive review claim.',
    'source_count':len(sources),'source_counts_are_not_a_citation_target':True,
    'baseline_catalog_preserved':'../sources.json', 'sources':sources}
(ROOT/'sources.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(ROOT/'literature_claim_ledger.json').write_text(json.dumps({'schema_version':'1.0','claims':claims},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

ledger=['# Archetype-extension primary source ledger','',
    'Accessed 20 October 2026. New owner: literature worker. This is a source-to-claim register, not article prose. The preceding source catalog is preserved. Its records are carried forward with their original access limits; ARCH2025 access is upgraded after direct local-PDF reading. Only sources used by a relevant claim should enter the manuscript. Three temporal/data-access sources remain optional background.','']
for s in sources:
    ledger += [f"## {s['id']}",'',f"- Citation: {'; '.join(s['authors'])} ({s['year']}). {s['title']}. {s['container']}, {s.get('volume','')}, {s.get('pages','')}. DOI: {s.get('doi') or 'not assigned/verified'}.",
        f"- Primary source: [{s['title']}]({s['url']})",f"- Access date/depth: 2026-10-20; {s['access_depth']}",
        f"- Supporting passage locator: {s['locator']}",f"- Passage paraphrase (no direct quotation): {s['supports']}",
        f"- Counterevidence / inference limit: {s['limitations']}",f"- Atomic claim IDs: {', '.join(s['supports_claims']) or 'background only; do not force a citation'}",'']
(ROOT/'source_ledger.md').write_text('\n'.join(ledger)+'\n',encoding='utf-8')

blocks=['# Proposed atomic literature claims for the archetype extension','',
    'Owner: literature worker. Stable NEW IDs use AE-LIT; they do not overwrite prior LIT IDs. These are atomic evidence blocks for the gpt-6-sol claim author and independent review. They are not polished manuscript prose. Supported means the bounded source statement is supported at its recorded access depth, not that the proposed model is accurate. No claim introduces a new simulation result.','']
for c in claims:
    prov='; '.join(p['source_id']+': '+p['locator'] for p in c['provenance'])
    blocks += [f"### {c['id']} — {c['label']}",f"- Exact claim: {c['exact_claim']}",
        f"- Kind: {c['kind']}",f"- Evidence class: {c['evidence_class']}",'- Status: supported (subject to the source-specific access limits).',
        '- Provenance: '+prov,f"- Population, geography, period, comparator, and units: {c['scope']}",
        '- Assumptions and uncertainty: Retain source scope and access depth; no numerical benefit is transferred to this study.',
        '- Counterevidence / alternative explanation: '+' '.join(c['counterevidence']),
        '- What would falsify or weaken the claim: A primary full-text correction, unmatched scope, or a source statement contradicted by its own methods.',
        f"- Required next action / dependency / owner: {c['implementation_implication']} Claim author incorporates; reviewer rechecks primary evidence.",
        '- Permitted manuscript wording: The exact bounded claim above, with its stated scope; no first-study or improved-accuracy inference.',
        '- Related claims / proposed tables and figures: See critical_synthesis_outline.md and nearest_work_matrix.md; citations support background or design choices, not result figures.','']
(ROOT/'proposed_literature_claims.md').write_text('\n'.join(blocks)+'\n',encoding='utf-8')

def author_bib(name):
    specials={'Lidia Premoli Vilà':'{Premoli Vilà}, Lidia','Carlos Cerezo Davila':'{Cerezo Davila}, Carlos','Ina De Jaeger':'{De Jaeger}, Ina','Bram van der Heijde':'{van der Heijde}, Bram','Istituto nazionale di statistica':'{Istituto nazionale di statistica}'}
    return specials.get(name,name)
bib=[]
for s in sources:
    fields={'author':' and '.join(author_bib(a) for a in s['authors']),'title':s['title'],'year':str(s['year'])}
    typ=s.get('type','article')
    fields['journal' if typ=='article' else 'booktitle' if typ in ('incollection','inproceedings') else 'publisher']=s['container']
    for f in ['volume','number','pages','doi','url']:
        if s.get(f): fields[f]=s[f]
    fields['urldate']='2026-10-20'
    fields['note']='Access depth and supported claims: sources.json record '+s['id']
    bib += ['@'+typ+'{'+s['id'].replace('-','')+',']+['  '+k+' = {'+v+'},' for k,v in fields.items()]+['}','']
(ROOT/'bibliography.bib').write_text('\n'.join(bib),encoding='utf-8')
print(json.dumps({'sources':len(sources),'atomic_claims':len(claims),'unresolved_source_ids':[],'original_catalog_modified':False}))
