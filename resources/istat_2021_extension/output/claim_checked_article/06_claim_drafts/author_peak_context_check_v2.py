"""Post-hoc Bolzano restart-day context from frozen primary hourly files.

This descriptive check never modifies the predeclared peak endpoint or losses.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "05_simulations/archetype_campaign_v2/pre_run"
SITE = ROOT / "05_simulations/archetype_campaign_v2/Bolzano"
OUT = Path(__file__).with_suffix(".json")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


roles = [r for r in read_rows(PRE / "role_manifest.csv") if r["role"] in {"test", "component"}]
assert Counter(r["role"] for r in roles) == {"test": 32, "component": 4}
summaries = {r["profile_id"]: r for r in json.loads((SITE / "summary.json").read_text(encoding="utf-8"))
             if r["case"] == r["profile_id"]}
records = []
for role in roles:
    path = SITE / f'{role["profile_id"]}_hourly.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == summaries[role["profile_id"]]["hourly_sha256"].lower(), (role["profile_id"], digest, summaries[role["profile_id"]]["hourly_sha256"])
    rows = read_rows(path)
    assert len(rows) == 8760
    assert [int(r["hour"]) for r in rows] == list(range(8760))
    peak = max(float(r["sensible_load_W"]) for r in rows)
    first_hour = next(int(r["hour"]) for r in rows if float(r["sensible_load_W"]) == peak)
    assert abs(peak / 1000 - summaries[role["profile_id"]]["peak_sensible_heating_kW"]) < 1e-9
    records.append({
        "role": role["role"], "household_id": int(role["household_id"]),
        "profile_id": role["profile_id"], "peak_kW": peak / 1000,
        "first_peak_hour": first_hour, "on_restart_day": 6912 <= first_hour <= 6935,
        "at_restart_hour": first_hour == 6912, "hourly_sha256": digest,
    })

counts = {}
for kind in ("test", "component"):
    these = [r for r in records if r["role"] == kind]
    counts[kind] = {"denominator": len(these),
                    "restart_day_first_maxima": sum(r["on_restart_day"] for r in these),
                    "restart_hour_first_maxima": sum(r["at_restart_hour"] for r in these)}
assert counts == {
    "test": {"denominator": 32, "restart_day_first_maxima": 27,
             "restart_hour_first_maxima": 18},
    "component": {"denominator": 4, "restart_day_first_maxima": 2,
                  "restart_hour_first_maxima": 2},
}
OUT.write_text(json.dumps({"analysis_type": "post_hoc_descriptive_peak_timing",
                           "site": "Bolzano", "model_restart_day_hours": [6912, 6935],
                           "counts": counts, "records": records}, indent=2) + "\n", encoding="utf-8")
print(OUT)
