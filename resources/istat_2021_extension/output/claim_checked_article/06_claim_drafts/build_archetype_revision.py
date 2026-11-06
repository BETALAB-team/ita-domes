"""Render immutable extension claim revisions from a reviewed prior snapshot."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
version = int(sys.argv[1])
assert version >= 6
v = f"v{version:02d}"
prior = f"v{version-1:02d}"
patch = json.loads((HERE / f"revision_{version:02d}.json").read_text(encoding="utf-8"))
prior_md = HERE / f"article_claims_{prior}.md"
prior_json = HERE / f"claim_ledger_{prior}.json"
review = ROOT / f"07_reviews/archetype_extension/review_{version-1:02d}.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert patch["prior_draft_sha256"] == sha(prior_md)
assert patch["review_sha256"] == sha(review)
prior_ledger = json.loads(prior_json.read_text(encoding="utf-8"))
assert prior_ledger["draft_sha256"] == sha(prior_md)
claims = prior_ledger["claims"]
by_id = {claim["id"]: claim for claim in claims}
assert len(by_id) == len(claims)
for cid, updates in patch["updates"].items():
    assert cid in by_id and "id" not in updates
    by_id[cid].update(updates)
    if "locators" in updates:
        by_id[cid]["source_ids"] = [loc.split(" | ", 1)[0] for loc in updates["locators"]]
for claim in patch.get("new_claims", []):
    assert claim["id"] not in by_id
    claims.append(claim)
    by_id[claim["id"]] = claim

required = ("id", "label", "exact_claim", "kind", "evidence_class", "status", "source_ids",
            "locators", "outputs", "population_geography_period_comparator_units",
            "assumptions_and_uncertainty", "counterevidence_alternative_explanation",
            "falsification_or_weakening", "dependencies_next_action_owner",
            "permitted_manuscript_wording", "related_claims_figures")
allowed_status = {"supported", "partially supported", "disputed", "unverified", "planned"}
for claim in claims:
    assert all(key in claim and claim[key] is not None for key in required), claim["id"]
    assert claim["status"] in allowed_status, claim["id"]
    assert claim["source_ids"] == [loc.split(" | ", 1)[0] for loc in claim["locators"]], claim["id"]

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
lines = [f"# Article claim checker {v} — archetype extension", "", patch["status_note"], ""]
for heading, prefixes in sections:
    lines.extend((f"## {heading}", ""))
    selected = [c for c in claims if c["id"].split("-")[0] in prefixes]
    if heading == "Literature review":
        backbone = ["LIT-001", *[f"LIT-{n:03d}" for n in range(33, 41)], "LIT-002", "LIT-003", "LIT-004"]
        selected.sort(key=lambda c: backbone.index(c["id"]) if c["id"] in backbone else 1000 + int(c["id"].split("-")[1]))
    elif heading == "Research questions":
        selected.sort(key=lambda c: {"RQ-003": 0, "RQ-001": 1, "RQ-002": 2}.get(c["id"], 99))
    for claim in selected:
        lines.extend((f"### {claim['id']} — {claim['label']}", ""))
        for label, key in fields:
            value = claim[key]
            if isinstance(value, list):
                value = "; ".join(value)
            lines.append(f"- {label}: {value}")
        lines.append("")

md = HERE / f"article_claims_{v}.md"
ledger_path = HERE / f"claim_ledger_{v}.json"
index_path = HERE / f"claim_evidence_ledger_{v}.json"
assert not any(p.exists() for p in (md, ledger_path, index_path)), "Refusing to overwrite immutable revision"
md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
ledger = {"schema_version": "1.0", "version": v, "draft_filename": md.name,
          "draft_sha256": sha(md), "prior_input_sha256": sha(prior_md),
          "review_sha256": sha(review), "claim_count": len(claims), "claims": claims}
ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
index = {"version": v, "draft_filename": md.name, "draft_sha256": sha(md),
         "claim_ledger_filename": ledger_path.name, "claim_ledger_sha256": sha(ledger_path),
         "claims": {c["id"]: {k: c[k] for k in ("exact_claim", "kind", "evidence_class", "status", "source_ids", "locators", "outputs", "dependencies_next_action_owner", "permitted_manuscript_wording")} for c in claims}}
index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"version": v, "claims": len(claims), "draft_sha256": sha(md),
                  "ledger_sha256": sha(ledger_path), "index_sha256": sha(index_path)}))
