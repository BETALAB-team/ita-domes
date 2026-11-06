"""Publish canonical claim pointers only after independent v07 closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = ROOT / "01_plan/delivery_history/pre_archetypes_20261020"
REVIEWS = ROOT / "07_reviews/archetype_extension"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


v07 = HERE / "article_claims_v07.md"
v07_ledger = HERE / "claim_ledger_v07.json"
v07_index = HERE / "claim_evidence_ledger_v07.json"
closure = REVIEWS / "final_closure.md"
assert sha(v07) == "9aa90444db044ab1480872fbf496db171f359e39e10cf1207f76a4c586757df9"
assert sha(v07_ledger) == "a768e5d022bb117c5a185c8a40500bbac52387910167ad65d439d759d27dd468"
assert sha(v07_index) == "f458f7b2b811d7be924533a22bc892eb53550c84f3ebc900e2f56442c482999a"
assert sha(closure) == "d8730da7ffb3dfd922764f12e10746317985a4ca606158c00d199c4f1a0d27e8"

# Ensure the initial accepted delivery is preserved byte-for-byte before
# changing only its mutable current pointers.
for relative in ("06_claim_drafts/article_claims_final.md",
                 "06_claim_drafts/claim_evidence_ledger.json",
                 "06_claim_drafts/final_acceptance.md",
                 "07_reviews/issue_register.json"):
    assert sha(ROOT / relative) == sha(HISTORY / relative), relative

v07_full = json.loads(v07_ledger.read_text(encoding="utf-8"))
v07_machine = json.loads(v07_index.read_text(encoding="utf-8"))
assert v07_full["claim_count"] == 106 and v07_machine["version"] == "v07"
assert v07_full["draft_sha256"] == v07_machine["draft_sha256"] == sha(v07)
registry = json.loads((ROOT / "02_evidence/archetype_extension/registry_audit_v07.json").read_text(encoding="utf-8"))
assert registry["claim_ledger"]["sha256"] == sha(v07_ledger)
assert all(registry["counts"][key] == 0 for key in ("hash_mismatch_count", "missing_file_count", "unresolved_alias_count"))

(HERE / "article_claims_final.md").write_bytes(v07.read_bytes())
(HERE / "claim_evidence_ledger.json").write_bytes(v07_index.read_bytes())
assert sha(HERE / "article_claims_final.md") == sha(v07)
assert sha(HERE / "claim_evidence_ledger.json") == sha(v07_index)

historical = json.loads((ROOT / "07_reviews/issue_register_final.json").read_text(encoding="utf-8"))
current = json.loads((REVIEWS / "issue_register_06.json").read_text(encoding="utf-8"))
assert len(historical["entries"]) == 10
assert {entry["issue_id"] for entry in current["entries"]} == {"R5-01", "R5-02"}
entries = historical["entries"]
for entry in entries:
    assert not entry["still_open"]
    entry["retained_historical_closure"] = True
    entry["extension_final_closure_path"] = "07_reviews/archetype_extension/final_closure.md"
for entry in current["entries"]:
    entry["still_open"] = False
    entry["disposition"] = ("Closed after independent review06 and separate exact-v07 final closure. "
                            + entry["disposition"].split(" Independent closure awaits review06.")[0])
    entry["evidence_paths"].append("07_reviews/archetype_extension/final_closure.md")
    entry["closure_evidence_sha256"] = sha(closure)
    entries.append(entry)
assert len(entries) == 12 and len({e["issue_id"] for e in entries}) == 12
final_register = {
    "register_type": "final archetype-extension issue register; historical bounded closures retained",
    "through_round": 6, "claim_version": "v07", "claim_count": 106,
    "review_06_sha256": sha(REVIEWS / "review_06.md"),
    "response_06_sha256": sha(REVIEWS / "response_06.md"),
    "revision_sha256": sha(v07), "final_closure_path": "07_reviews/archetype_extension/final_closure.md",
    "final_closure_sha256": sha(closure),
    "historical_register_path": "07_reviews/issue_register_final.json",
    "historical_register_sha256": sha(ROOT / "07_reviews/issue_register_final.json"),
    "unresolved_critical_or_major_scientific_issues": [],
    "standing_limitations": [
        "Exact remote published per-household assignment identity unproved",
        "No matched external building benchmark or independent compatible meter validation",
        "Single fixed physical case, one held-out split and two respondents per group-control cell",
        "Conditional mixture has twelve free probabilities versus three pooled",
        "Sky, native solar-time, seasonal-restart and physical-input sensitivity unquantified",
        "No national/city representativeness, measured accuracy, causal group effect or sizing inference"
    ],
    "entries": entries
}
final_path = REVIEWS / "issue_register_final.json"
assert not final_path.exists(), final_path
final_path.write_text(json.dumps(final_register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(REVIEWS / "issue_register.json").write_bytes(final_path.read_bytes())
(ROOT / "07_reviews/issue_register.json").write_bytes(final_path.read_bytes())
lines = ["# Current scientific issue register — archetype extension v07", "",
         f"Independent final closure: `07_reviews/archetype_extension/final_closure.md`, SHA256 `{sha(closure)}`. All 12 recorded issues are closed for their bounded stated scopes. The original ten-issue final register remains unchanged at `07_reviews/issue_register_final.json`; its initial-delivery copy is also in dated history.", "",
         "The current scientific limits remain substantive: exact published-branch membership, matched benchmark and meters, population transfer, representation capacity, physical-input sensitivity and causal peak explanation are unestablished.", ""]
for entry in entries:
    lines.extend((f"- **{entry['issue_id']}** ({entry['severity']}): closed for its bounded scope.",))
md = "\n".join(lines).rstrip() + "\n"
(REVIEWS / "issue_register.md").write_text(md, encoding="utf-8")
(ROOT / "07_reviews/issue_register.md").write_text(md, encoding="utf-8")

acceptance = f"""# Accepted claim-checker handoff — revised archetype extension

Date: 20 October 2026. The canonical scientific claim file `article_claims_final.md` is an exact byte copy of immutable `article_claims_v07.md`; both have SHA256 `{sha(v07)}` and contain 106 atomic claims. `claim_ledger_v07.json` SHA256 `{sha(v07_ledger)}` is the full machine ledger; current `claim_evidence_ledger.json` is an exact byte copy of versioned `claim_evidence_ledger_v07.json` SHA256 `{sha(v07_index)}`. No accepted-copy edit changes a claim, status, source locator, figure permission or number.

Independent full `review_06.md` SHA256 `{sha(REVIEWS / 'review_06.md')}` found no new scientific correction. The separate `final_closure.md` SHA256 `{sha(closure)}` closed R5-01/R5-02 and opened the revised English prose-development gate for this **bounded, incomplete first manuscript**. Frozen v07's stage note says final closure remained required when that version was authored; this dated note and the closed `issue_register_final.json` supersede only that procedural stage label. They do not promote partially supported claims to observed or empirically validated results. The v07 supplemental source registry SHA256 `{sha(ROOT / '02_evidence/archetype_extension/local_sources.json')}` resolves its declared aliases; source binding is an identity check, not validation of physical truth.

The central implemented experiment uses archived article-consistent four-group demographic assignments to condition training frequencies across the same four real source-derived heating-control components. It compares group-conditional and pooled mixtures against **same-model** detailed-input outputs for 32 held-out respondents, one assumed 80 m² dwelling and four coordinate TMY exposures. Conditional overall model-reference RMSE is lower for annual useful sensible heating and expected individual annual maximum at each exposure, while Bolzano groups 2 and 4 worsen on the latter. The conditional mixture has twelve free probabilities against three pooled. The Bolzano 27/32 respondent and 2/4 component restart-day maxima are a **post-hoc model-timing description**; no peak or primary loss was removed. The original one-respondent timing/weather v5 pilot remains a separate secondary result.

The review's literature framework begins with all seventeen references and the actual modelling lineage in the original CLIMA chapter, then compares relevant survey, physical/demographic archetype, occupant-schedule, aggregation and validation precedents at their recorded access depths. The 2025 archetype article is a distinct linked demographic representation, not a chronological successor to the 2026 CLIMA proceedings publication. This is a specific input-bridge/evaluation contribution, not a claim to generic first occupancy integration or exhaustive literature priority.

Prose and figures must preserve the unproved exact remote published-assignment identity, one-cell demographic rounding discrepancy, assumptions in the source heating-control mapping, one physical case, small one-split holdout, unequal probability-parameter counts, within-branch information loss, adverse subgroup results, post-hoc seasonal context and weather/sky/time limits. No matched independent building benchmark, compatible independently documented meter series, population weighting/representativeness or quantified broader uncertainty was supplied. The current claims therefore do not show measured predictive accuracy, causal demographic benefit, actual equipment runtime, equipment sizing, urban coincident peak or national-stock performance.

The original 41-claim v04 accepted delivery and its old final pointers are preserved byte-for-byte under `01_plan/delivery_history/pre_archetypes_20261020`; historical review/register snapshots and extension v05–v07 snapshots remain intact. The current `07_reviews/issue_register.json` points to the extension's final closed register, while historical `07_reviews/issue_register_final.json` is unchanged. This handoff authorizes article drafting only; it does not authorize submission, publication or external contact.
"""
(HERE / "final_acceptance.md").write_text(acceptance, encoding="utf-8")
print(json.dumps({"canonical_claim_sha256": sha(HERE / "article_claims_final.md"),
                  "canonical_index_sha256": sha(HERE / "claim_evidence_ledger.json"),
                  "final_acceptance_sha256": sha(HERE / "final_acceptance.md"),
                  "extension_final_register_sha256": sha(final_path),
                  "closure_sha256": sha(closure)}))
