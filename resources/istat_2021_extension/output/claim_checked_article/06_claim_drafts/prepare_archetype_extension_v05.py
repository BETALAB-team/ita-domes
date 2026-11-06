"""Prepare (or freeze) the archetype-extension claim checker from v04.

The default preview lives outside the immutable version namespace. Pass --freeze
only after the new campaign has a checked, source-linked outcome supplement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / "archetype_extension_work"
FIELDS = (
    "id", "label", "exact_claim", "kind", "evidence_class", "status",
    "source_ids", "locators", "outputs", "population_geography_period_comparator_units",
    "assumptions_and_uncertainty", "counterevidence_alternative_explanation",
    "falsification_or_weakening", "dependencies_next_action_owner",
    "permitted_manuscript_wording", "related_claims_figures",
)
KINDS = {"literature", "gap", "novelty", "method", "result", "interpretation", "limitation"}
EVIDENCE = {"published finding", "observed measurement", "survey response", "derived estimate", "simulation output", "hypothesis", "planned test"}
STATUSES = {"supported", "partially supported", "disputed", "unverified", "planned"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(claims: list[dict], cid: str, label: str, exact: str, kind: str,
        evidence: str, status: str, locators: list[str], population: str,
        assumptions: str, counter: str, falsifier: str, dependency: str,
        wording: str, related: str = "", outputs: list[str] | None = None) -> None:
    claims.append(dict(
        id=cid, label=label, exact_claim=exact, kind=kind,
        evidence_class=evidence, status=status,
        source_ids=[x.split(" | ")[0] for x in locators], locators=locators,
        outputs=outputs or [],
        population_geography_period_comparator_units=population,
        assumptions_and_uncertainty=assumptions,
        counterevidence_alternative_explanation=counter,
        falsification_or_weakening=falsifier,
        dependencies_next_action_owner=dependency,
        permitted_manuscript_wording=wording,
        related_claims_figures=related,
    ))


def build_claims() -> tuple[list[dict], dict, dict]:
    prior = json.loads((HERE / "claim_ledger_v04.json").read_text(encoding="utf-8"))
    claims = copy.deepcopy(prior["claims"])
    by_id = {c["id"]: c for c in claims}
    assert len(claims) == 41 and len(by_id) == 41

    # Historical v04 is untouched. Current v05 fields reflect the saved closure.
    c = by_id["LIT-004"]
    c.update(
        status="supported",
        locators=["ARCH2025-FULL | complete primary publisher PDF pp3–14, §§3.1–3.3, 4.1–4.4, 6, Appendix A; SHA256 8995443d5c1b92313730c75c99512f94122aac4f8e0a79b4fe00bca5b269aae2",
                  "ARCH-SPEC | 00_context/archetype_extension/published_archetype_specification.md, demographic definitions and behaviour/dwelling sections"],
        assumptions_and_uncertainty="Complete primary paper was read. Exact identity of the archived local four-group household labels with the paper's remote code branch remains unproved; this does not weaken the published-paper claim.",
        falsification_or_weakening="Contrary corrected primary-paper content would require revising the published-paper description; local assignment mismatch concerns implementation provenance separately.",
        dependencies_next_action_owner="None for the paper's reported four-class identification; retain the separate local-assignment qualification in DAT-006.",
        permitted_manuscript_wording="The published 2025 study identified four demographic user groups and explored probabilistic dwelling linkage; the present study implements an archived article-consistent assignment rather than discovering those groups.",
        related_claims_figures="DAT-005; DAT-006; LIT-021; NOV-002",
    )
    c["source_ids"] = [x.split(" | ")[0] for x in c["locators"]]
    stale = {
        "RQ-001": ("; v5 now provides a checked conditional model comparison, pending a fresh fourth scientific review", "; the v5 conditional model comparison passed the original independent scope-limited review"),
        "RQ-002": ("; v5 matched runs provide conditional weather-exposure contrasts, not local population or climate-variable causal effects, pending review 04", "; v5 matched runs provide independently reviewed conditional weather-exposure contrasts, not local population or climate-variable causal effects"),
        "RES-004": (" These are conditional model outputs awaiting fresh review 04, not observed energy or accuracy estimates.", " These are independently reviewed conditional model outputs, not observed energy or accuracy estimates."),
        "RES-005": (" These are deterministic conditional weather-exposure contrasts pending fresh review 04, not population effects or causal attribution to a weather variable.", " These are independently reviewed deterministic conditional weather-exposure contrasts, not population effects or causal attribution to a weather variable."),
        "RES-007": (" The Bolzano maxima coincide", " The Bolzano maxima coincide"),
    }
    for cid, (old, new) in stale.items():
        by_id[cid]["exact_claim"] = by_id[cid]["exact_claim"].replace(old, new)
    by_id["INT-001"]["exact_claim"] = by_id["INT-001"]["exact_claim"].replace(
        "v5 outputs show", "independently reviewed v5 outputs show")
    by_id["NOV-001"]["related_claims_figures"] += "; NOV-002 (new central implementation contribution)"
    by_id["RQ-001"]["label"] = "Original timing sensitivity pilot (linked)"
    by_id["RQ-002"]["label"] = "Original weather-exposure pilot (linked)"
    by_id["RQ-001"]["related_claims_figures"] += "; RQ-003 (primary revised article question)"
    by_id["RQ-002"]["related_claims_figures"] += "; RQ-003 (primary revised article question)"
    rq_index = "AE-RQ-INDEX | 01_plan/archetype_extension/research_questions.md, current primary RQ-003 and linked original RQ-001/002 hierarchy"
    for cid in ("RQ-001", "RQ-002"):
        by_id[cid]["locators"].append(rq_index)
        by_id[cid]["source_ids"] = [x.split(" | ")[0] for x in by_id[cid]["locators"]]
    closed_dependencies = {
        "RQ-001": "Original scope-limited review04/closure completed; current prose must retain one-case conditional boundary.",
        "RQ-002": "Original scope-limited review04/closure completed; retain fixed-input weather-exposure framing.",
        "DAT-003": "Original scope-limited final closure completed; later archetype runs must preserve and recheck the same physical configuration.",
        "MET-003": "Original scope-limited final closure completed; describe only this implementation-specific state verification.",
        "MET-005": "Original scope-limited review04/closure completed; keep numerical verification distinct from benchmark or measurement evidence.",
        "MET-006": "Original scope-limited review04/closure completed; any future wind-dependent path must resolve Palermo's unused-field exception.",
        "MET-008": "Original scope-limited review04/closure completed; avoid general EUReCA validation language.",
        "MET-009": "Original scope-limited final closure completed; retain annual-mean sky and native solar-time limits.",
        "LIM-007": "Methods/author: preserve archives and explicit version labels; original review04 established v5-only numerical provenance.",
        "MET-011": "Original scope-limited review04/closure completed; later archetype runs require their own full forcing/state checks.",
        "RES-003": "Original scope-limited final closure completed; retain encoded-versus-observed distinction.",
        "RES-004": "Original scope-limited final closure completed; report conditional v5 fixed-case model outputs without observed-accuracy claims.",
        "RES-005": "Original scope-limited final closure completed; retain coordinate-weather and no-causal-variable limits.",
        "LIM-008": "Author: retain this limit alongside original and new hourly-peak interpretations; future phase/longwave analysis requires a separate design.",
    }
    for cid, dep in closed_dependencies.items():
        by_id[cid]["dependencies_next_action_owner"] = dep

    lit = [p for p in json.loads((ROOT / "02_evidence/archetype_extension/literature_claim_ledger.json").read_text(encoding="utf-8"))["claims"] if p["id"].startswith("AE-LIT-")]
    mapping: dict[str, dict] = {}
    old = {17: "LIT-004", 21: "LIT-002", 25: "LIT-003", 32: "LIT-001; LIT-004"}
    next_lit = 5
    for p in lit:
        n = int(p["id"].split("-")[-1])
        if n in old:
            mapping[p["id"]] = {"formal_ids": old[n].split("; "), "disposition": "existing claim updated or retained", "reason": "Substance already covered by the stable original literature claim."}
            continue
        cid = f"LIT-{next_lit:03d}" if n != 33 else "GAP-003"
        if n != 33:
            next_lit += 1
        locators = [f"{q['source_id']} | {q['locator']}; access: {q['access_depth']}" for q in p["provenance"]]
        counter = "; ".join(p.get("counterevidence", [])) or "This comparator does not establish the present study's measured accuracy or generalizability."
        add(claims, cid, p["label"], p["exact_claim"],
            "gap" if n == 33 else "literature", p["evidence_class"], p["status"], locators,
            p["scope"],
            "Source access is bounded as identified in the provenance; no external effect size or Italian case outcome is imported.",
            counter,
            "Contradictory full primary methods/results, a correction, or a closer matched evaluation would narrow this statement.",
            "Literature author: keep primary locator and access depth; prose author: explain this comparison within the relevant thematic section.",
            p["exact_claim"], "LIT-004; NOV-002")
        mapping[p["id"]] = {"formal_ids": [cid], "disposition": "new atomic claim", "reason": "Distinct study or evidentiary boundary retained for substantive synthesis."}

    conference = json.loads((ROOT / "02_evidence/archetype_extension/conference_claim_records.json").read_text(encoding="utf-8"))
    existing_sources = json.loads((ROOT / "02_evidence/archetype_extension/sources.json").read_text(encoding="utf-8"))["sources"]
    new_sources = json.loads((ROOT / "02_evidence/archetype_extension/conference_source_additions.json").read_text(encoding="utf-8"))
    source_by_id = {s["id"]: s for s in existing_sources + new_sources}
    conference_map = {}
    for p in conference:
        n = int(p["id"].split("-")[-1])
        cid = f"LIT-{n+32:03d}" if n <= 8 else "INT-002"
        locators = [f"{sid} | {source_by_id[sid]['locator']}; access: {source_by_id[sid]['access_depth']}" for sid in p["sources"]]
        locators.append("CONF-BIB | 00_context/conference_complete_text.txt, printed pp1310–1311 and exact reference number in 02_evidence/archetype_extension/conference_reference_crosswalk.md")
        if n == 9:
            locators.append("AE-V2-PROTOCOL | 03_methods/archetype_extension/mixture_protocol_v2.md, shared physics/provenance: useful sensible heat only, no final-energy/DHW conversion")
        add(claims, cid, p["label"], p["exact_claim"], p["kind"],
            "derived estimate" if n == 9 else "published finding", "supported", locators,
            p["scope"],
            "The full bibliography was inspected, while individual source access is bounded as specified; no numerical performance score or current legal/standard compliance is imported.",
            p["implementation_implication"],
            "A corrected primary source, incompatible conference citation context or inspected fuller text contradicting this bounded statement would require revision.",
            "Prose author: place this conference-bibliography source in the literature lineage or methods boundary; keep actual source access and geographic/energy limits.",
            p["exact_claim"], "LIT-001; LIT-004; NOV-002")
        conference_map[p["id"]] = {"formal_ids": [cid], "disposition": "new bounded conference-backbone claim"}

    arch = "ARCH2025-FULL | complete 15-page primary PDF pp3–14, Tables 1, 3, 6, Fig. 5, §§3.1–3.3, 4.1–4.4, 6, Appendix A"
    spec = "ARCH-SPEC | 00_context/archetype_extension/published_archetype_specification.md, demographic definitions/assignment and direct archive audit"
    asn = "ARCH-ASSIGN | 00_context/archetype_extension/assignment_mapping_schema.json and archived_four_archetype_assignments.csv, SHA256 be8bc93ad8f5143f6393e9b270840a7e5a2edd90724edb60b2fb624680b09e2f"
    audit = "ARCH-AUDIT | 00_context/archetype_extension/local_assignment_audit.json, join counts and 40 rounded-cell comparisons"
    ctrl = "ARCH-CONTROL-AUDIT | 00_context/archetype_extension/survey_control_eligibility_audit.json, raw/compact routing and 16 source-control strata"
    proto = "AE-PROTOCOL | 03_methods/archetype_extension/simulation_protocol.md, frozen question/estimands, split, prototypes, weights and technical gates"
    plan = "AE-DESIGN | 01_plan/archetype_extension/design_decision.md, selected input-compression design"

    add(claims, "DAT-005", "Published demographic group definitions",
        "The 2025 paper defines four unweighted Italian household demographic groups from derived minimum/maximum age, member count and education; Table 3 and Figure 5 report class summaries and household/person shares, not threshold rules or observed operating schedules.",
        "result", "published finding", "supported", [arch, spec],
        "2013 Italian survey; four groups; published equal-household summary shares 40.1/26.9/22.9/10.1%.",
        "Released age is grouped and derived; the paper omits enough fitted-centroid/min-max/seed details to assign a new record exactly from prose.",
        "Class descriptions are examples; within-class household and heating responses vary.",
        "A full corrected assignment method or retraction of reported summaries would revise the boundary.",
        "Prose author: report means as means, not demographic ranges; use archived assignment caveat in DAT-006.",
        "The published four demographic groups describe unweighted survey households; their names do not prescribe hourly presence or heat schedules.",
        "LIT-004; DAT-006; LIM-009")
    add(claims, "DAT-006", "Archived four-group household assignment",
        "The lowercase archived clusters.csv has 17,855 unique IDs in four labels, all joined one-to-one to the official 20,000-row release and compact input; its rounded group/share/demographic summaries match 39 of 40 published cells, with archetype-4 maximum-age mean 72.1 locally versus 72.2 printed, but exact published-branch per-household identity remains unproved.",
        "result", "derived estimate", "partially supported", [asn, audit, spec],
        "Archived assignment for 17,855 of 20,000 released 2013 household IDs; 2,145 outside domain remain unassigned.",
        "Aggregate matching and local Git archival history establish article consistency, not identity to inaccessible paper-branch labels; one rounded cell differs.",
        "The later seeded reconstruction differs for 318 of the 17,855 IDs; uppercase Clustered.csv has five labels and is a distinct artifact.",
        "A direct per-ID comparison with the publisher branch could confirm or refute exact identity.",
        "Context/evidence owner: preserve hashes and the one discrepancy; methods owner: use only the archived assignment source by exact ID join.",
        "We use an archived four-group assignment whose summaries largely match the published groups; exact paper-branch household identity has not been verified.",
        "DAT-005; LIM-009", ["00_context/archetype_extension/archived_four_archetype_assignments.csv", "00_context/archetype_extension/local_assignment_audit.json"])
    add(claims, "DAT-007", "Operational-input eligibility domain",
        "Of 17,855 archived-assigned survey records, 13,624 have central/individual main heating, all three valid observed 0–8 h winter-day heating-band responses and a positive total, with group counts 5,430/3,693/2,991/1,510; all four assumed source-control branches have nonzero support in each group.",
        "result", "derived estimate", "supported", [ctrl, "AE-DOMAIN-AUDIT | 04_analysis/archetype_extension/archived_operating_domain_audit.json, eligible groups/source branches"],
        "Unweighted eligible archived 2013 response domain; no national expansion weights.",
        "Eligibility is specific to this heating-input experiment, not all published households; compact routed blanks may become zero for branch predicates.",
        "Excluded households may have different behaviour or physical systems; class proportions in this eligible cohort need not equal published full-sample shares.",
        "A rerun of raw/compact predicates yielding a different count or any unsupported control branch would require correction.",
        "Methods owner: freeze selected manifest and verify exact 16 stratum counts before annual runs.",
        "The eligible archived operating-input cohort contains 13,624 unweighted records across all four groups and four supported modeled control branches.",
        "DAT-006; MET-012; LIM-010", ["00_context/archetype_extension/survey_control_eligibility_audit.json"])
    add(claims, "DAT-008", "Pre-output prototype-profile collapse",
        "In the frozen first input-only selection, the four group medoid roles and the pooled medoid role selected different or repeated real respondent IDs but all five mapped to the same complete 8,760-hour temperature profile SHA256 1739675e45f73ab112e029291491ec18505002878a653a0c5d10ec3f237a0eab; this design could not yield distinct executable group-versus-pooled inputs and produced no annual extension result.",
        "result", "derived estimate", "supported", ["AE-V1-PROTOTYPES | 05_simulations/archetype_campaign_v1/pre_run/prototype_manifest.csv, five roles/source IDs/profile hash", "AE-V1-PROFILES | 05_simulations/archetype_campaign_v1/pre_run/profile_manifest.json, five equivalent roles and exact full-profile SHA256", proto],
        "First pre-run input selection on 13,624 eligible archived responses; no new annual thermal output.",
        "The finding is a property of the chosen 24-hour squared-distance medoid rule and this training split, not evidence that all group operational distributions coincide.",
        "Full group/control branch frequencies differ, and a conditional representation can still have distinct inputs or weights.",
        "A mismatch between prototype and full-profile hashes or evidence of annual v1 extension output would require correction.",
        "Author: retain v1 as pre-output diagnostic and link the prospective amended design without presenting a v1 group effect.",
        "The first medoid design collapsed to one identical executable profile across four groups and the pooled comparator before annual simulation.",
        "MET-013; RQ-003", ["05_simulations/archetype_campaign_v1/pre_run/prototype_manifest.csv", "05_simulations/archetype_campaign_v1/pre_run/profile_manifest.json"])
    published_groups = [
        (1, "few older members with mid-level education", "70.2/78.3/1.8/0.5", "40.1/29.9"),
        (2, "more members, commonly younger and middle-aged members", "21.2/57.2/3.7/0.7", "26.9/40.2"),
        (3, "few middle-aged members with medium education", "46.0/63.0/2.3/0.5", "22.9/21.3"),
        (4, "few middle-/older-aged members with higher education", "55.6/72.2/2.1/0.8", "10.1/8.6"),
    ]
    for group, description, means, shares in published_groups:
        add(claims, f"DAT-{group+8:03d}", f"Published archetype {group} summary",
            f"Published archetype {group} is described as {description}; Table 3 reports mean minimum age/maximum age/member count/education index {means}, and Figure 5 reports unweighted household/person shares {shares}%, respectively; these are summaries, not membership thresholds or hourly schedules.",
            "result", "published finding", "supported", ["ARCH2025-FULL | complete primary PDF p7 Table 3 and Fig. 5, archetype interpretation and printed one-decimal values", spec],
            "Italian 2013 survey households in the published unweighted four-group analysis; ages years, members persons, education index dimensionless, shares percent.",
            "Minimum and maximum ages are derived from released age classes; printed means and narrative examples do not specify the fitted centroid rule for new households.",
            ("The local archive's archetype-4 maximum-age mean rounds to 72.1 rather than the printed 72.2; exact published per-record identity is unresolved." if group == 4 else "Printed class means can conceal within-group variation; local archived per-record identity to the publisher branch remains unproved."),
            "A corrected primary Table 3/Figure 5 or direct verified source-branch summaries would require a revised table.",
            "Author: carry printed value with source and distinguish local archive audit; no label-derived thermal schedules.",
            f"Published archetype {group} has printed demographic means and unweighted survey shares as shown in Table 3 and Figure 5.",
            "DAT-005; DAT-006; TAB-ARCH")
    add(claims, "DAT-013", "Training branch distributions differ by group",
        "In the frozen 13,592-record training set, source-control branch 22/11 has group-conditional proportions 63.04%, 51.51%, 57.46% and 55.73% for groups 1–4, while branch 24/21 has 4.81%, 8.93%, 7.11% and 7.19%; these eligible-cohort frequencies create distinct conditional mixture weights without being population-weighted or observed temperatures.",
        "result", "derived estimate", "supported", ["AE-V2-MIXTURE | 05_simulations/archetype_campaign_v2/pre_run/mixture_probabilities.json, group_branch_training_counts and training_P_branch_given_group", "AUTHOR-AE-PRERUN | 06_claim_drafts/author_archetype_prerun_check_v2.json, training count/probability recalculation"],
        "13,592 eligible archived-assigned 2013 training records; four groups, four assumed source-control branches; percentages of each group training subset.",
        "Training split is frozen but conditional branch shares are observational associations shaped by source-rule assumptions and this eligible subset.",
        "Within-branch reported durations and profiles can differ even when branch code matches; demographic class is not an observed temperature preference.",
        "A training count/probability recomputation mismatch would invalidate the quoted values.",
        "Author/reviewer: verify rounded percentages from frozen counts; prose author: avoid causal wording.",
        "The eligible training groups have different frequencies of modeled control branches, providing distinct pre-output mixture weights.",
        "DAT-007; MET-015; LIM-010")
    add(claims, "GAP-004", "Paper classes do not determine 8760 behaviour",
        "The 2025 archetype paper provides demographic classes, survey/model-derived annual or daily energy-use group summaries and a dwelling-linkage method, but no complete class-specific measured 8,760-hour presence, gain, ventilation or temperature schedule; translating classes to thermal inputs therefore requires explicit additional assumptions.",
        "gap", "published finding", "supported", [arch, spec],
        "Italian 2013 survey and published 2025 analysis; no measured hourly thermal-demand comparator.",
        "Heating on-hours are declared mean daily values and appliance annual electricity is modeled; neither identifies hourly temperatures or thermal gains.",
        "A published supplementary class-specific 8760 input dataset, if verified, would narrow the absence claim.",
        "Locate and inspect any accessible official supplement before an absolute absence claim; retain this bounded paper-content wording.",
        "Methods owner: describe direct response-linked operational mapping; prose author: avoid invented label-based curves.",
        "The published demographic labels require a separate, documented bridge to hourly heating inputs.",
        "LIT-004; DAT-005; MET-012")
    add(claims, "NOV-002", "Bounded archetype implementation contribution",
        "The candidate new contribution is an auditable bridge from archived article-consistent Italian demographic household assignments and their real survey heating/control responses to a corrected ita-domes/EUReCA configuration, followed by held-out within-model input-compression comparisons against a pooled input; generic stochastic scheduling or first occupant-aware simulation is not claimed.",
        "novelty", "hypothesis", "partially supported", [plan, proto, spec, "AE-LIT-PRIOR | 02_evidence/archetype_extension/nearest_work_matrix.md, closest physical/user/schedule/stock precedents"],
        "One assumed dwelling, four coordinate TMY exposures, 2013 eligible archived survey domain; four-vs-one real-profile representation.",
        "The implementation claim requires completed, checked annual runs; targeted literature coverage cannot prove absolute priority.",
        "Buttitta, Chen, Prataviera and other sources already implement heterogeneous users, stochastic schedules or ISTAT-band reconstruction.",
        "A prior exact Italian demographic-to-same-configuration implementation, failed execution gate or invalid assignment bridge would narrow the contribution.",
        "Methods/claim authors: inspect frozen profiles, executions and output arithmetic; independent reviewer: assess bounded contribution before prose.",
        "We implement and evaluate an auditable Italian demographic-assignment-to-operational-input bridge in this specified model case.",
        "LIT-002; LIT-004; MET-012; RQ-003")
    add(claims, "RQ-003", "Archetype input-compression question",
        "For held-out real survey respondents in each archived demographic group and source-control branch, compare the same model's annual useful heating and maximum hourly sensible load under detailed respondent inputs, a real training-only group prototype and one real training-only pooled prototype, with physical case and site weather matched.",
        "method", "planned test", "planned", [proto, plan],
        "32 held-out IDs, two per four groups × four control branches; one fixed dwelling; four coordinate TMYs; kWh and kW.",
        "Detailed-input model outputs are the reference only within this model; four-vs-one capacity differs; neither arm is measured heating.",
        "The class prototype can have larger, smaller or equal model deviation than pooled; no advantage is assumed.",
        "Train/test leakage, altered physical inputs, branch imbalance without declared weights or failed technical gates would invalidate the comparison.",
        "Methods owner: freeze input-only selection and runs; author: independently recompute all deviation metrics; reviewer: assess remaining inference limits.",
        "We test within-model fidelity of four group-conditioned real-input prototypes against one pooled real-input prototype for held-out heating inputs.",
        "NOV-002; MET-012; MET-013; RES-009; RES-010")
    add(claims, "MET-012", "Response-linked archetype input bridge",
        "The archetype extension joins archived four-group IDs to actual 2013 heating-band and control responses and converts each selected respondent's inputs through the documented source-ID-seeded contiguous schedule and assumed 22/11, 22/19, 24/13 or 24/21°C high/setback branches; physical dwelling, internal gains and season remain fixed.",
        "method", "derived estimate", "partially supported", [asn, ctrl, proto, "AE-BRIDGE | 04_analysis/archetype_extension/profile_bridge.py, source-control and source-ID profile construction"],
        "Eligible archived 2013 heating respondents; one 80 m² assumed dwelling, four weather files; scheduled setpoints in °C.",
        "Source code assumptions include timer/valve/retrofit-control routing and 2 K fireplace branch; reported on-hours are not observed setpoints or actual positive-load runtime.",
        "Demographic ages and education do not enter thermal inputs directly; class differences represent joint observed responses and source-rule mappings, not an isolated causal group effect.",
        "A profile fixture discrepancy, incorrect control routing or failed source-ID preservation would invalidate the implementation.",
        "Methods owner: freeze role-to-input manifest and 24/8760-hour profile hashes; independent reviewer: inspect source route.",
        "Archived demographic groups are linked by household ID to real reported heating/control inputs; the thermal profiles then follow explicit model assumptions.",
        "DAT-006; DAT-007; RQ-003; LIM-011")
    add(claims, "MET-013", "Held-out prototypes and weighting",
        "A preselected PCG64(20261020) split holds out two IDs per group×control branch (32 total), excludes them from prototype fitting, and selects each of four group prototypes and one pooled prototype as the actual training respondent nearest the corresponding training 24-hour temperature-input centroid; descriptive losses use eligible stratum counts rather than national survey weights.",
        "method", "derived estimate", "partially supported", [proto, "AE-SELECT | 05_simulations/archetype_campaign_v1/pre_run/selection_manifest.json, seed and held-out IDs", "AE-PROTOTYPES | 05_simulations/archetype_campaign_v1/pre_run/prototype_manifest.csv and training_input_centroids.json", "AE-STRATA | 05_simulations/archetype_campaign_v1/pre_run/stratum_counts.csv, empirical weights"],
        "13,624 eligible IDs; 13,592 training and 32 test; 16 strata; 24-hour source-input vectors; no outcomes used for fitting.",
        "A real selected record is a prototype for this split/metric, not an observed group-mean schedule or intrinsically representative household.",
        "Four group profiles have more representational capacity than one pooled profile; weighting reexpresses this eligible cohort, not a population/design-weighted estimand.",
        "Any test ID in training, outcome-based selection, missing stratum or incorrect weight sum would invalidate the contrast.",
        "Methods owner: freeze and hash all selection artifacts; author: verify IDs/weights independently; reviewer: check capacity caveat.",
        "We select training-only real-input prototypes and compare them with held-out responses under declared eligible-cohort weights.",
        "DAT-007; RQ-003; MET-014")
    add(claims, "MET-014", "Within-model compression estimands",
        "For each held-out ID and site, define signed and absolute deviations of group and pooled prototype outputs from that same model's detailed-input output, plus paired absolute-deviation difference |group−detailed|−|pooled−detailed|, separately for annual useful sensible heat (kWh) and maximum hourly sensible load (kW).",
        "method", "planned test", "planned", [proto],
        "32 test IDs × four site weather exposures; annual kWh and maximum hourly kW; descriptive stratum-reweighted summaries.",
        "The detailed-input outcome is a model reference, not metered truth; negative paired difference favors group compression only in this setup.",
        "Signed errors may cancel and absolute errors may differ by group/branch/site; no result direction assumed.",
        "Arithmetic mismatch with direct hourly CSV integrations or invalid role mapping would invalidate reported losses.",
        "Methods owner: save raw hourly outputs and role map; author: recompute outputs and weighted paired deviations; reviewer: inspect.",
        "Within-model prototype deviations are reported against held-out detailed-input simulations for annual useful heating and hourly maximum.",
        "RQ-003; RES-009; RES-010")
    add(claims, "LIM-009", "Archetype identity and assignment limit",
        "The local archived labels are article-consistent but not shown identical to the publisher's inaccessible UserArchetypes branch per household; published prose means and examples cannot assign excluded or new households, and 2,145 release IDs lack archived labels.",
        "limitation", "derived estimate", "supported", [spec, asn, audit],
        "Original 2013 survey and local archived assignment; no new-household classification.",
        "One paper Table 3 rounded maximum-age cell differs; full fitted centroids and scaling details are not published in the paper.",
        "The later deterministic reconstruction differs for 318 labels and cannot silently replace the archived mapping.",
        "Direct branch per-record labels or a fully reproducible publisher assignment could resolve identity.",
        "Author: use 'archived article-consistent' wording throughout; no exact published executable reproduction.",
        "The recovered four-group assignment reproduces most printed summaries but exact paper-branch membership has not been established.",
        "DAT-006; NOV-002")
    add(claims, "LIM-010", "Prototype comparison capacity and sampling limit",
        "The 32-test-respondent, four-prototype-versus-one-pooled comparison is a small, intentionally unequal-capacity within-model fidelity experiment; its eligible-cohort descriptive weighting is not national population inference, and it cannot identify whether any difference arises from demographic grouping independently of prototype count.",
        "limitation", "derived estimate", "supported", [proto, plan, ctrl],
        "Unweighted 13,624-record eligible domain; 32 held-out IDs in 16 source-control strata; one physical case.",
        "Two test IDs per cell limit stable within-cell distribution estimates; prototype selection depends on the split.",
        "An equal-capacity pooled comparator or larger independent cohort might alter conclusions.",
        "A separately predeclared complexity-matched replication could test the grouping effect more specifically.",
        "Author: keep capacity and holdout limits adjacent to any deviation result.",
        "Four group profiles versus one pooled profile tests a specified compression design, not demographic-group superiority.",
        "RQ-003; MET-013; RES-010")
    add(claims, "LIM-011", "Operational channel and physical-case limit",
        "Only response-linked heating-hour/control inputs vary in the new archetype comparison; occupancy, internal gains, ventilation, envelope, season and ideal heating physics are fixed, so the experiment does not implement the paper's probabilistic dwelling assignment or quantify joint demographic/physical uncertainty.",
        "limitation", "planned test", "partially supported", [proto, spec, "AE-BRIDGE | 04_analysis/archetype_extension/profile_bridge.py, fixed-channel construction"],
        "Single illustrative 80 m² zone, four coordinate TMY exposures, eligible 2013 heating respondents.",
        "This isolates an operational-input bridge but limits realism and transfer; all control temperatures are source-model assumptions.",
        "The paper's probabilistic dwelling link, appliance model and within-day occupancy are distinct unused channels.",
        "Run/config comparison revealing varying gains or physical settings would invalidate this declared isolation.",
        "Methods owner: retain config/physics hashes; author: state channel exclusions in methods and limitations.",
        "We vary response-linked operational inputs while keeping physical and internal-gain assumptions common.",
        "MET-012; NOV-002; LIM-002")
    add(claims, "GAP-005", "Empirical validation remains absent",
        "A held-out comparison against detailed-input outputs from the same model evaluates input-compression fidelity, not empirical prediction accuracy; no matched independent meter or building benchmark establishes which archetype representation better predicts real heating for this case.",
        "gap", "planned test", "supported", [proto, "AE-DESIGN | 01_plan/archetype_extension/design_decision.md, model-reference versus measured-accuracy boundary", "VAL-PROTOCOL | 03_methods/validation_protocol.md, empirical benchmark and measurement criteria"],
        "One assumed dwelling; 2013 survey-derived inputs and four TMYs; model useful sensible heat, not final metered energy.",
        "The original national annual carrier comparison is a different scale and energy boundary.",
        "An independent weather/building/end-use matched meter series could evaluate actual predictive error.",
        "Independent matched data and a predeclared calibration/evaluation split would be needed for a new accuracy claim.",
        "Author: retain explicit missing-evidence marker; reviewer: prevent 'validation' for same-model holdout alone.",
        "The held-out experiment tests fidelity to detailed-input model outputs; measured heating accuracy remains untested.",
        "GAP-002; MET-007; RQ-003")

    # The post-input/pre-output amendment retains the 32 IDs but replaces v1's
    # coincident single medoids with four common branch components and distinct
    # group-conditional versus pooled TRAINING probabilities.
    v2 = "AE-V2-PROTOCOL | 03_methods/archetype_extension/mixture_protocol_v2.md, frozen post-input/pre-output shared-component mixture amendment"
    new = {c["id"]: c for c in claims}
    new["NOV-002"].update(
        exact_claim="The candidate new contribution is an auditable bridge from archived article-consistent Italian demographic household assignments and their real survey heating/control responses to a documented current GRINS/EUReCA configuration related to ita-domes, followed by a held-out within-model comparison of four shared real operational components mixed with training P(branch|archetype) versus P(branch); generic stochastic scheduling or first occupant-aware simulation is not claimed.",
        population_geography_period_comparator_units="One assumed dwelling, four coordinate TMY exposures, 2013 eligible archived survey domain; four shared real source-control branch components with conditional versus pooled weights.",
        assumptions_and_uncertainty="Implementation requires completed, checked annual runs. Both predictions share four inputs, but conditional weights add probability parameters; targeted literature coverage cannot prove absolute priority.",
        permitted_manuscript_wording="We implement and evaluate a documented Italian demographic-assignment-to-operational-input mixture bridge in this specified model case.")
    new["NOV-002"]["locators"] = [plan, v2, spec, "AE-LIT-PRIOR | 02_evidence/archetype_extension/nearest_work_matrix.md, closest physical/user/schedule/stock precedents"]
    new["RQ-003"].update(
        label="Primary archetype representation question",
        exact_claim="The primary revised-article question compares, for 32 frozen held-out real survey respondents, same-model detailed-input annual useful heating and individual annual maximum hourly load with predictions from four shared training-only real source-control branch components weighted by either P_train(branch|archived archetype) or pooled P_train(branch), under matched dwelling and site weather.",
        population_geography_period_comparator_units="32 held-out IDs, two per four groups × four control branches; one fixed dwelling; four coordinate TMYs; kWh and individual annual-maximum kW.",
        assumptions_and_uncertainty="Detailed-input model outputs are only within-model references; both predictions share four simulated components but conditional probabilities have more free parameters. Peak prediction is a weighted expectation of individual annual maxima, not the maximum of a mean curve.",
        counterevidence_alternative_explanation="Conditional branch weights may have larger, smaller or equal held-out squared deviation than pooled weights; no benefit is assumed.",
        permitted_manuscript_wording="We compare held-out within-model fidelity of shared operational components mixed by group-conditional versus pooled training branch frequencies.")
    new["RQ-003"]["locators"] = [v2, plan, rq_index]
    new["MET-013"].update(
        label="Frozen holdout and shared real branch components",
        exact_claim="The unchanged PCG64(20261020) split holds out two IDs per group×source-control branch (32 total), excluding them from fitting; the amended pre-output design selects one real training respondent per each of four source-control branches by nearest 24-hour temperature-input centroid with smallest-ID ties, then uses those four components for both conditional and pooled predictions.",
        population_geography_period_comparator_units="13,624 eligible IDs; 13,592 training and 32 test; 16 strata; four shared training-only real branch profiles.",
        assumptions_and_uncertainty="The v1 four group medoids and pooled medoid collapsed to one executable input before annual simulation; v2 keeps the held-out IDs but replaces that ineffective prototype rule. Branch centroids are never simulated.",
        counterevidence_alternative_explanation="One branch component cannot retain all within-branch duration/timing variation; component selection uses no heating outcomes.",
        falsification_or_weakening="Test leakage, outcome-fitted components, altered test IDs, unsupported branch, nonreal component or incorrect profile hash would invalidate the comparison.",
        dependencies_next_action_owner="Methods owner: freeze new branch-component manifests and probabilities; author: independently verify IDs/profiles/counts; reviewer: assess input-only amendment.",
        permitted_manuscript_wording="The amended comparison retains the preselected held-out IDs and uses the same four real training branch inputs in both representations.",
        related_claims_figures="DAT-007; DAT-008; RQ-003; MET-014; MET-015")
    new["MET-013"]["locators"] = [v2, "AE-V1-SELECT | 05_simulations/archetype_campaign_v1/pre_run/selection_manifest.json, unchanged seed/holdout", "AE-V2-COMPONENTS | 05_simulations/archetype_campaign_v2/pre_run/component_manifest.csv and component_fit_details.json, real branch choices"]
    new["MET-014"].update(
        label="Conditional-mixture loss estimands",
        exact_claim="For each site and each primary endpoint, predict with the weighted sum of the same four separately simulated real branch-component endpoints, using either P_train(branch|group) or pooled P_train(branch); compare both with every held-out detailed-input model endpoint by eligible-stratum-weighted squared deviation and RMSE, including their paired squared-deviation difference.",
        population_geography_period_comparator_units="32 test IDs × four site weather exposures; annual useful heating kWh and expected individual annual-maximum kW; squared deviations in kWh²/kW², RMSE in kWh/kW.",
        assumptions_and_uncertainty="The detailed-input output is not metered truth. The prediction for peak is expected individual annual maximum, not the maximum of an averaged hourly curve or diversified coincident peak; no averaged setpoint input is simulated.",
        counterevidence_alternative_explanation="The four methods share components, but conditional mixture has twelve free branch-probability parameters versus three pooled; lower RMSE would not isolate demographic information from added capacity.",
        falsification_or_weakening="Incorrect training probability/count normalization, comparison weights, endpoint integration or role mapping would invalidate reported loss metrics.",
        dependencies_next_action_owner="Methods owner: save raw hourly outputs/component role map; author: independently recompute endpoint sums and losses; reviewer: inspect weights and peak definition.",
        permitted_manuscript_wording="We report held-out within-model squared deviations and RMSE for conditional and pooled mixtures of shared real operating components.",
        related_claims_figures="RQ-003; MET-013; MET-015; RES-009; RES-010")
    new["MET-014"]["locators"] = [v2]
    new["LIM-010"].update(
        exact_claim="The 32-test-respondent group-conditional-versus-pooled mixture comparison is a small within-model fidelity experiment using the same four simulated branch inputs, but conditional weights have more free parameters and omit within-branch group differences; eligible-cohort descriptive weighting is not national population inference or an isolated general demographic-group advantage.",
        assumptions_and_uncertainty="Two test IDs per group×control cell limit stable within-cell estimates; training component and probability estimates depend on the split.",
        counterevidence_alternative_explanation="Equal component count removes the v1 profile-count confound but not conditioning/parameter-count differences, within-branch heterogeneity or the single physical case.",
        permitted_manuscript_wording="The mixtures share branch inputs, while group conditioning adds probability parameters; any deviation difference is limited to this small within-model test.")
    new["LIM-010"]["locators"] = [v2, ctrl, plan]
    new["GAP-005"]["exact_claim"] = "A held-out comparison against detailed-input outputs from the same model evaluates operational-mixture fidelity, not empirical prediction accuracy; no matched independent meter or building benchmark establishes which representation better predicts real heating for this case."
    new["GAP-005"]["locators"] = [v2, "VAL-PROTOCOL | 03_methods/validation_protocol.md, empirical benchmark and measurement criteria"]
    add(claims, "MET-015", "Shared-component conditional mixtures",
        "The conditional archetype representation uses P_train(branch|group)=N_train,g,b/N_train,g and the pooled representation uses P_train(branch)=N_train,b/N_train, both applied to the same four real branch-component model endpoints; no averaged input trajectory or outcome-fitted weight is used.",
        "method", "planned test", "planned", [v2, "AE-V2-MIXTURE | 05_simulations/archetype_campaign_v2/pre_run/mixture_probabilities.json, frozen training counts and probabilities"],
        "Four archived classes and four modeled source-control branches; probabilities conditional on the 13,592 training-record eligible domain.",
        "The branch probabilities are survey-cohort frequencies, not spatial prevalence, assignment probabilities or measured operation probabilities; conditional model has twelve free probabilities versus three pooled.",
        "Different branch distributions by group can alter predictions, but within-branch duration/timing differences are not represented.",
        "A mismatch between frozen training counts and weights, weights not summing to one or outcome-tuned probabilities would invalidate the comparison.",
        "Methods owner: freeze mixture weights before annual outputs; author/reviewer: check every group/pooled normalization and shared-component identity.",
        "Both methods mix the same four real branch simulations; archetypes enter through their training branch frequencies.",
        "MET-013; MET-014; LIM-010")
    new["MET-013"]["status"] = "supported"
    new["MET-013"]["locators"].append("AUTHOR-AE-PRERUN | 06_claim_drafts/author_archetype_prerun_check_v2.json, frozen split/36 profile hashes/branch components pass")
    met15 = claims[-1]
    assert met15["id"] == "MET-015"
    met15["status"] = "supported"
    met15["locators"].append("AUTHOR-AE-PRERUN | 06_claim_drafts/author_archetype_prerun_check_v2.json, training probabilities and evaluation-weight arithmetic pass")
    met15["source_ids"] = [x.split(" | ")[0] for x in met15["locators"]]
    for cid in ("NOV-002", "RQ-003", "MET-013", "MET-014", "LIM-010", "GAP-005"):
        new[cid]["source_ids"] = [x.split(" | ")[0] for x in new[cid]["locators"]]

    # All new annual executions and the author-owned raw-hour integration now
    # exist. They remain candidates until independent review05.
    run_summary = "AE-CAMPAIGN | 05_simulations/archetype_campaign_v2/campaign_completion_manifest.json, 144 scientific plus eight technical annual executions and four site gates"
    result_check = "AUTHOR-AE-RESULT | 06_claim_drafts/author_archetype_result_check_v2.json, direct integration of all 152 hourly outputs and conditional/pooled loss reconstruction"
    new["NOV-002"]["locators"].extend([run_summary, result_check])
    new["RQ-003"].update(status="partially supported", evidence_class="simulation output",
        dependencies_next_action_owner="Campaign and author arithmetic checks complete; fresh reviewer05 must inspect new source/input/run claims before scientific promotion.")
    new["RQ-003"]["locators"].extend([run_summary, result_check])
    new["MET-012"]["locators"].extend([run_summary, result_check])
    new["MET-012"]["dependencies_next_action_owner"] = "Frozen profile/role and run gates completed; fresh reviewer05 to inspect the response-to-input bridge and source control mapping."
    new["MET-014"].update(status="partially supported", evidence_class="simulation output",
        dependencies_next_action_owner="Campaign and author arithmetic checks complete; fresh reviewer05 to assess all eight endpoint/site metrics and individual maximum definition.")
    new["MET-014"]["locators"].extend(["AE-METRICS | 04_analysis/archetype_extension/results_v2/weighted_metrics.csv and individual_deviations.csv, all site/group loss metrics", result_check])
    new["LIM-011"].update(status="supported", evidence_class="simulation output",
        dependencies_next_action_owner="Author: retain fixed physical/internal-gain and unused dwelling-allocation boundary; new reviewer05 may audit run configs.")
    new["LIM-011"]["locators"].append(run_summary)
    for cid in ("NOV-002", "RQ-003", "MET-012", "MET-014", "LIM-011"):
        new[cid]["source_ids"] = [x.split(" | ")[0] for x in new[cid]["locators"]]

    # Scientific outcomes are added by the result supplement only after the frozen
    # annual campaign and an author-owned independent arithmetic check exist.
    supplement_path = WORK / "checked_result_claims_v05.json"
    if supplement_path.exists():
        supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
        assert supplement.get("campaign") == "archetype_campaign_v2"
        claims.extend(supplement["claims"])

    # The fresh review receives primary protocols, manifests, raw executions and
    # analysis tables; author-only arithmetic files remain separate audit work.
    for c in claims:
        c["locators"] = [x for x in c["locators"] if not x.startswith("AUTHOR-AE-")]
        c["source_ids"] = [x.split(" | ")[0] for x in c["locators"]]
        c["outputs"] = [x for x in c["outputs"] if not x.startswith("06_claim_drafts/author_archetype_")]

    assert len(claims) == len({c["id"] for c in claims}), [c["id"] for c in claims if [x["id"] for x in claims].count(c["id"]) > 1]
    for c in claims:
        for key in FIELDS:
            assert key in c and c[key] is not None, (c["id"], key)
        assert c["kind"] in KINDS and c["evidence_class"] in EVIDENCE and c["status"] in STATUSES, c["id"]
        assert c["source_ids"] == [x.split(" | ")[0] for x in c["locators"]], c["id"]
    assert len(mapping) == 33 and len(conference_map) == 9
    return claims, mapping, conference_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    claims, mapping, conference_map = build_claims()
    WORK.mkdir(exist_ok=True)
    map_path = HERE / "archetype_extension_literature_id_map.json"
    if not map_path.exists():
        map_path.write_text(json.dumps({"source_packet": "02_evidence/archetype_extension/literature_claim_ledger.json", "mapping": mapping}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cf_map_path = HERE / "archetype_extension_conference_id_map.json"
    if not cf_map_path.exists():
        cf_map_path.write_text(json.dumps({"source_packet": "02_evidence/archetype_extension/conference_claim_records.json", "mapping": conference_map}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sections = [("Literature review", ("LIT",)), ("Gap and novelty", ("GAP", "NOV")),
                ("Research questions", ("RQ",)), ("Data", ("DAT",)),
                ("Methodological claims", ("MET",)), ("Results", ("RES",)),
                ("Interpretation", ("INT",)), ("Limitations", ("LIM",))]
    fields = [("Exact claim", "exact_claim"), ("Kind", "kind"),
              ("Evidence class", "evidence_class"), ("Status", "status"),
              ("Provenance", "locators"),
              ("Population, geography, period, comparator, and units", "population_geography_period_comparator_units"),
              ("Assumptions and uncertainty", "assumptions_and_uncertainty"),
              ("Counterevidence / alternative explanation", "counterevidence_alternative_explanation"),
              ("What would falsify or weaken the claim", "falsification_or_weakening"),
              ("Required next action / dependency / owner", "dependencies_next_action_owner"),
              ("Permitted manuscript wording", "permitted_manuscript_wording"),
              ("Related claims / proposed tables and figures", "related_claims_figures")]
    note = ("Extension candidate before new independent review. The original v04 remains immutable and accepted only for its original scope. "
            "New archetype claims are source-bounded; any new numerical result remains conditional pending fresh review 05.")
    lines = ["# Article claim checker v05 — archetype extension", "", note, ""]
    for heading, prefixes in sections:
        lines.extend((f"## {heading}", ""))
        selected = [c for c in claims if c["id"].split("-")[0] in prefixes]
        if heading == "Literature review":
            backbone = ["LIT-001", *[f"LIT-{n:03d}" for n in range(33, 41)], "LIT-002", "LIT-003", "LIT-004"]
            selected.sort(key=lambda c: (backbone.index(c["id"]) if c["id"] in backbone else 1000 + int(c["id"].split("-")[1])))
        elif heading == "Research questions":
            selected.sort(key=lambda c: {"RQ-003": 0, "RQ-001": 1, "RQ-002": 2}.get(c["id"], 99))
        for c in selected:
            lines.extend((f"### {c['id']} — {c['label']}", ""))
            for label, key in fields:
                value = c[key]
                if isinstance(value, list):
                    value = "; ".join(value)
                lines.append(f"- {label}: {value}")
            lines.append("")
    target_md = HERE / "article_claims_v05.md" if args.freeze else WORK / "candidate_article_claims_v05.md"
    target_json = HERE / "claim_ledger_v05.json" if args.freeze else WORK / "candidate_claim_ledger_v05.json"
    if args.freeze and (target_md.exists() or target_json.exists()):
        raise SystemExit("Refusing to overwrite immutable v05")
    target_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    digest = sha(target_md)
    prior_hash = sha(HERE / "article_claims_v04.md")
    ledger = dict(schema_version="1.0", version="v05", draft_filename=target_md.name,
                  draft_sha256=digest, prior_input_sha256=prior_hash,
                  extension_design_sha256=sha(ROOT / "01_plan/archetype_extension/design_decision.md"),
                  proposal_map_sha256=sha(map_path), conference_map_sha256=sha(cf_map_path), claim_count=len(claims), claims=claims)
    target_json.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.freeze:
        index = HERE / "claim_evidence_ledger_v05.json"
        if index.exists():
            raise SystemExit("Refusing to overwrite immutable v05 claim index")
        index.write_text(json.dumps({
            "version": "v05", "draft_filename": target_md.name, "draft_sha256": digest,
            "claim_ledger_filename": target_json.name, "claim_ledger_sha256": sha(target_json),
            "claims": {c["id"]: {k: c[k] for k in ("exact_claim", "kind", "evidence_class", "status", "source_ids", "locators", "outputs", "dependencies_next_action_owner", "permitted_manuscript_wording")} for c in claims}
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": args.freeze, "claims": len(claims), "draft_sha256": digest, "draft": str(target_md), "ledger": str(target_json)}))


if __name__ == "__main__":
    main()
