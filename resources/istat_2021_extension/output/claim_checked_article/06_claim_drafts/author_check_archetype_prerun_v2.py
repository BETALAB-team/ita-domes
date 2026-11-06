"""Independent arithmetic/identity checks of the frozen archetype v2 inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "05_simulations/archetype_campaign_v1/pre_run"
V2 = ROOT / "05_simulations/archetype_campaign_v2/pre_run"
OUT = Path(__file__).resolve().parent / "author_archetype_prerun_check_v2.json"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


v1 = json.loads((V1 / "selection_manifest.json").read_text(encoding="utf-8"))
v2 = json.loads((V2 / "selection_manifest.json").read_text(encoding="utf-8"))
mix = json.loads((V2 / "mixture_probabilities.json").read_text(encoding="utf-8"))
roles = read_csv(V2 / "role_manifest.csv")
components = read_csv(V2 / "component_manifest.csv")
strata = read_csv(V2 / "stratum_counts.csv")
tests = [r for r in roles if r["role"] == "test"]

assert v1["test_ids"] == v2["test_ids"]
assert len(tests) == 32 == len(v2["test_ids"])
assert sorted(int(r["household_id"]) for r in tests) == v2["test_ids"]
assert len(components) == 4
assert set(r["control_branch"] for r in components) == set(mix["branch_order"])
assert not ({int(r["household_id"]) for r in tests} & {int(r["household_id"]) for r in components})
assert len({r["profile_sha256"] for r in tests + components}) == 36
assert len(strata) == 16
assert sum(int(r["eligible_n"]) for r in strata) == 13624
assert sum(float(r["overall_empirical_weight"]) for r in tests) - 1 < 1e-12
assert abs(sum(float(r["overall_empirical_weight"]) for r in tests) - 1) < 1e-12
by_cell = Counter((int(r["archetype_id"]), r["control_branch"]) for r in tests)
assert len(by_cell) == 16 and set(by_cell.values()) == {2}
for r in tests:
    assert abs(float(r["inclusion_probability"]) - 2 / float(r["stratum_N"])) < 1e-12
    assert abs(float(r["overall_empirical_weight"]) - float(r["stratum_N"]) / (2 * 13624)) < 1e-12
    group_n = next(int(x["group_n"]) for x in strata if int(x["archetype_id"]) == int(r["archetype_id"]))
    assert abs(float(r["within_group_weight"]) - float(r["stratum_N"]) / (2 * group_n)) < 1e-12

for g in ("1", "2", "3", "4"):
    counts = mix["group_branch_training_counts"][g]
    assert sum(counts.values()) == sum(int(r["eligible_n"]) for r in strata if str(r["archetype_id"]) == g) - 8
    assert abs(sum(mix["training_P_branch_given_group"][g].values()) - 1) < 1e-12
    for b in mix["branch_order"]:
        assert abs(mix["training_P_branch_given_group"][g][b] - counts[b] / sum(counts.values())) < 1e-12
assert sum(sum(x.values()) for x in mix["group_branch_training_counts"].values()) == 13592
assert abs(sum(mix["pooled_P_branch"].values()) - 1) < 1e-12
for b in mix["branch_order"]:
    count = sum(mix["group_branch_training_counts"][g][b] for g in ("1", "2", "3", "4"))
    assert abs(mix["pooled_P_branch"][b] - count / 13592) < 1e-12

for r in tests + components:
    manifest_profile = V2 / f"{r['profile_id']}.csv"
    assert manifest_profile.exists()
    with manifest_profile.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8760
    assert [int(x["hour"]) for x in rows] == list(range(8760))
    raw = b"".join(struct.pack("<d", float(x["lower_C"])) for x in rows)
    assert hashlib.sha256(raw).hexdigest() == r["profile_sha256"]

result = {
    "status": "pass: independent frozen-input arithmetic and identity checks; no annual-output certification",
    "v1_selection_manifest_sha256": digest(V1 / "selection_manifest.json"),
    "v2_selection_manifest_sha256": digest(V2 / "selection_manifest.json"),
    "v2_mixture_probabilities_sha256": digest(V2 / "mixture_probabilities.json"),
    "v2_component_manifest_sha256": digest(V2 / "component_manifest.csv"),
    "test_n": len(tests),
    "test_ids_unchanged": True,
    "test_per_group_branch": 2,
    "training_n": 13592,
    "component_ids": {r["control_branch"]: int(r["household_id"]) for r in components},
    "distinct_test_and_component_full_profile_sha256_n": len({r["profile_sha256"] for r in tests + components}),
    "weight_sum": sum(float(r["overall_empirical_weight"]) for r in tests),
    "branch_probabilities_recomputed": True,
    "all_36_csv_profiles_8760_ordered_hours": True,
    "all_36_numeric_profile_hashes_recomputed": True,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
