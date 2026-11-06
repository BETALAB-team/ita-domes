"""Record a no-new-finding review06 without inventing an issue."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


review = HERE / "review_06.md"
response = HERE / "response_06.md"
revision = ROOT / "06_claim_drafts/article_claims_v07.md"
assert sha(review) == "67015c5f00462ee246098d637821a789fd6ce6ce7eba4014a43d44780883bdef"
assert sha(revision) == "9aa90444db044ab1480872fbf496db171f359e39e10cf1207f76a4c586757df9"
previous = json.loads((HERE / "issue_register_05.json").read_text(encoding="utf-8"))
entries = previous["entries"]
assert {item["issue_id"] for item in entries} == {"R5-01", "R5-02"}
for item in entries:
    item["disposition"] += " Fresh review06 reported no new scientific correction and independently checked the current v2 documentation, raw peak timing, losses and limits; a separately requested formal prior-issue closure is still pending."
    item["evidence_paths"].append("07_reviews/archetype_extension/review_06.md")
    item["revision_locations"].append("06_claim_drafts/article_claims_v07.md")
register = {"through_round": 6, "review_sha256": sha(review),
            "response_sha256": sha(response), "revision_filename": revision.name,
            "revision_sha256": sha(revision), "new_issues_this_round": [],
            "entries": entries}
snapshot = HERE / "issue_register_06.json"
assert not snapshot.exists(), snapshot
snapshot.write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(HERE / "issue_register.json").write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["# Archetype-extension issue register after review 06", "",
         f"Review06 SHA256 `{sha(review)}`; response06 SHA256 `{sha(response)}`; immutable v07 SHA256 `{sha(revision)}`.", "",
         "Review06 found no new scientific correction. V07 changes elapsed procedural dependencies only. The two review05 reporting issues await the separately requested formal closure pass.", ""]
for entry in entries:
    lines.extend((f"## {entry['issue_id']} — {entry['severity']}", "", entry["disposition"], "",
                  f"Closure condition: {entry['closure_condition']}", ""))
(HERE / "issue_register.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(json.dumps({"review": sha(review), "response": sha(response), "revision": sha(revision), "register": sha(snapshot)}))
