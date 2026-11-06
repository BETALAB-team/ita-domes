"""Render a reviewed claim-ledger revision without changing earlier snapshots.

Usage: python -B build_revision.py NN
Requires revision_NN.json with `updates` keyed by stable claim ID and optional
`new_claims`. It verifies review/prior hashes before writing vNN files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
nn = int(sys.argv[1])
assert nn >= 1
v = f"v{nn:02d}"
prior = f"v{nn-1:02d}"
prior_md = HERE / f"article_claims_{prior}.md"
prior_json = HERE / f"claim_ledger_{prior}.json"
review = ROOT / f"07_reviews/review_{nn:02d}.md"
patch = json.loads((HERE / f"revision_{nn:02d}.json").read_text(encoding="utf-8"))
prior_hash = hashlib.sha256(prior_md.read_bytes()).hexdigest()
review_hash = hashlib.sha256(review.read_bytes()).hexdigest()
assert patch["prior_draft_sha256"] == prior_hash
assert patch["review_sha256"] == review_hash
prior_ledger = json.loads(prior_json.read_text(encoding="utf-8"))
claims = prior_ledger["claims"]
by_id = {c["id"]: c for c in claims}
for cid, changes in patch["updates"].items():
    assert cid in by_id, cid
    assert "id" not in changes
    by_id[cid].update(changes)
    if "locators" in changes:
        by_id[cid]["source_ids"] = [p.split(" | ")[0] for p in changes["locators"]]
for c in patch.get("new_claims", []):
    assert c["id"] not in by_id
    assert c["source_ids"] == [p.split(" | ")[0] for p in c["locators"]]
    claims.append(c)
assert len(claims) == len({c["id"] for c in claims})
for c in claims:
    for k in ("id", "label", "exact_claim", "kind", "evidence_class", "status", "source_ids", "locators", "outputs", "population_geography_period_comparator_units", "assumptions_and_uncertainty", "counterevidence_alternative_explanation", "falsification_or_weakening", "dependencies_next_action_owner", "permitted_manuscript_wording", "related_claims_figures"):
        assert k in c and c[k] is not None, (c["id"], k)

sections = [
    ("Literature review", ("LIT",)), ("Gap and novelty", ("GAP", "NOV")),
    ("Research questions", ("RQ",)), ("Data", ("DAT",)),
    ("Methodological claims", ("MET",)), ("Results", ("RES",)),
    ("Interpretation", ("INT",)), ("Limitations", ("LIM",)),
]
fields = [
    ("Exact claim", "exact_claim"), ("Kind", "kind"),
    ("Evidence class", "evidence_class"), ("Status", "status"),
    ("Provenance", "locators"),
    ("Population, geography, period, comparator, and units", "population_geography_period_comparator_units"),
    ("Assumptions and uncertainty", "assumptions_and_uncertainty"),
    ("Counterevidence / alternative explanation", "counterevidence_alternative_explanation"),
    ("What would falsify or weaken the claim", "falsification_or_weakening"),
    ("Required next action / dependency / owner", "dependencies_next_action_owner"),
    ("Permitted manuscript wording", "permitted_manuscript_wording"),
    ("Related claims / proposed tables and figures", "related_claims_figures"),
]
lines = [f"# Article claim checker {v}", "", patch["status_note"], ""]
for heading, prefixes in sections:
    lines += [f"## {heading}", ""]
    for c in claims:
        if c["id"].split("-")[0] not in prefixes:
            continue
        lines += [f"### {c['id']} — {c['label']}", ""]
        for label, key in fields:
            value = c[key]
            if isinstance(value, list):
                value = "; ".join(value)
            lines += [f"- {label}: {value}"]
        lines += [""]
md_path = HERE / f"article_claims_{v}.md"
json_path = HERE / f"claim_ledger_{v}.json"
for p in (md_path, json_path):
    if p.exists():
        raise SystemExit(f"Refusing to overwrite immutable snapshot: {p}")
md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
digest = hashlib.sha256(md_path.read_bytes()).hexdigest()
ledger = {
    "schema_version": "1.0", "version": v, "draft_filename": md_path.name,
    "draft_sha256": digest, "prior_input_sha256": prior_hash,
    "review_sha256": review_hash, "claim_count": len(claims), "claims": claims,
}
json_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(HERE / "claim_evidence_ledger.json").write_text(json.dumps({
    "version": v, "draft_filename": md_path.name, "draft_sha256": digest,
    "claim_ledger_filename": json_path.name,
    "claim_ledger_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
    "claims": {c["id"]: {k: c[k] for k in ("exact_claim", "kind", "evidence_class", "status", "source_ids", "locators", "outputs", "dependencies_next_action_owner", "permitted_manuscript_wording")} for c in claims},
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"version": v, "claim_count": len(claims), "draft_sha256": digest, "prior_input_sha256": prior_hash, "review_sha256": review_hash, "draft": str(md_path), "ledger": str(json_path)}))
