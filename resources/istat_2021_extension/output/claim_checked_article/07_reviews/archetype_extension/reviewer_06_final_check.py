"""Independent final revision identity/delta check. No producer code imported."""
from pathlib import Path
import hashlib, json, re, difflib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8-sig"))
def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

old = read("06_claim_drafts/claim_ledger_v06.json")
new = read("06_claim_drafts/claim_ledger_v07.json")
evidence = read("06_claim_drafts/claim_evidence_ledger_v07.json")
a = {x["id"]: x for x in old["claims"]}
b = {x["id"]: x for x in new["claims"]}
assert list(a) == list(b) and len(b) == 106
changes = {}
for cid in a:
    delta = {key: {"old": a[cid].get(key), "new": b[cid].get(key)}
             for key in set(a[cid]) | set(b[cid])
             if a[cid].get(key) != b[cid].get(key)}
    if delta:
        changes[cid] = delta

md = (ROOT/"06_claim_drafts/article_claims_v07.md").read_text(encoding="utf-8-sig")
blocks = {}
for match in re.finditer(r"^### ([A-Z]+-\d+) .*?\n(.*?)(?=^### |\Z)", md, re.M | re.S):
    blocks[match[1]] = match[2]
assert set(blocks) == set(b)
parity = []
for cid, claim in b.items():
    exact = re.search(r"^- Exact claim: (.*)$", blocks[cid], re.M)[1]
    wording = re.search(r"^- Permitted manuscript wording: (.*)$", blocks[cid], re.M)[1]
    assert exact == claim["exact_claim"] == evidence["claims"][cid]["exact_claim"]
    assert wording == claim["permitted_manuscript_wording"] == evidence["claims"][cid]["permitted_manuscript_wording"]
    for key, val in evidence["claims"][cid].items():
        if key in claim:
            assert val == claim[key], (cid, key)
    parity.append(cid)
assert new["draft_sha256"] == evidence["draft_sha256"] == sha("06_claim_drafts/article_claims_v07.md")
assert evidence["claim_ledger_sha256"] == sha("06_claim_drafts/claim_ledger_v07.json")
files = [
    "06_claim_drafts/article_claims_v06.md", "06_claim_drafts/claim_ledger_v06.json",
    "06_claim_drafts/claim_evidence_ledger_v06.json", "06_claim_drafts/article_claims_v07.md",
    "06_claim_drafts/claim_ledger_v07.json", "06_claim_drafts/claim_evidence_ledger_v07.json",
    "07_reviews/archetype_extension/review_06.md", "07_reviews/archetype_extension/response_06.md",
    "07_reviews/archetype_extension/review_05.md", "07_reviews/archetype_extension/response_05.md",
    "07_reviews/archetype_extension/issue_register_05.json",
    "03_methods/archetype_extension/validation_matrix.md",
    "03_methods/archetype_extension/posthoc_peak_context_report.md",
    "07_reviews/final_closure.md", "07_reviews/issue_register_final.json",
]
if (ROOT/"07_reviews/archetype_extension/issue_register_06.json").exists():
    files.append("07_reviews/archetype_extension/issue_register_06.json")
report = {"claim_count": len(b), "same_ordered_ids": True,
          "all_markdown_and_evidence_ledger_claim_parity": True,
          "claim_changes": changes, "hashes": {p: sha(p) for p in files},
          "metadata_changes": {k: {"old": old.get(k), "new": new.get(k)}
              for k in set(old)|set(new) if k != "claims" and old.get(k) != new.get(k)}}
cycles = []
for number in range(1, 7):
    base = "07_reviews" if number <= 4 else "07_reviews/archetype_extension"
    register_path = f"{base}/issue_register_{number:02}.json"
    register = read(register_path)
    review_path = f"{base}/review_{number:02}.md"
    response_path = f"{base}/response_{number:02}.md"
    revision_path = "06_claim_drafts/" + register["revision_filename"]
    input_version = number - 1 if number <= 4 else number
    input_path = f"06_claim_drafts/article_claims_v{input_version:02}.md"
    assert sha(review_path) == register["review_sha256"]
    assert sha(response_path) == register["response_sha256"]
    assert sha(revision_path) == register["revision_sha256"]
    assert sha(input_path) != sha(revision_path)
    cycles.append({"round": number, "input": input_path, "input_sha256": sha(input_path),
                   "review": review_path, "review_sha256": sha(review_path),
                   "response": response_path, "response_sha256": sha(response_path),
                   "revision": revision_path, "revision_sha256": sha(revision_path),
                   "register": register_path, "register_sha256": sha(register_path)})
report["six_complete_artifact_cycles"] = cycles
(OUT/"reviewer_06_final_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
oldmd = (ROOT/"06_claim_drafts/article_claims_v06.md").read_text(encoding="utf8").splitlines()
(OUT/"reviewer_06_v07_delta.diff").write_text("\n".join(difflib.unified_diff(oldmd, md.splitlines(), fromfile="v06", tofile="v07"))+"\n", encoding="utf8")
print(json.dumps(report, indent=2, ensure_ascii=True))
