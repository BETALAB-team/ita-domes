"""Independent standard-library arithmetic check of the Padova pilot outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "05_simulations/padova_pilot_v1"
SUM = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
CHECKS = json.loads((RUN / "checks.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((RUN / "completion_manifest.json").read_text(encoding="utf-8"))
assert MANIFEST["status"] == "complete" and CHECKS["pass"]

cases = {}
for s in SUM:
    p = RUN / f"{s['case']}_hourly.csv"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == s["hourly_sha256"]
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    q = [float(r["sensible_load_W"]) for r in rows]
    assert len(q) == 8760 and all(math.isfinite(x) for x in q)
    assert [int(r["hour"]) for r in rows] == list(range(8760))
    annual = math.fsum(max(x, 0.0) for x in q) / 1000
    peak = max(max(x, 0.0) for x in q) / 1000
    assert math.isclose(annual, s["annual_sensible_heating_kWh"], abs_tol=1e-8)
    assert math.isclose(peak, s["peak_sensible_heating_kW"], abs_tol=1e-12)
    prof = s["daily_on_profile"]
    assert len(prof) == 24 and set(prof) <= {0.0, 1.0}
    # Clock-hour blocks; the night block wraps midnight.
    counts = [sum(prof[5:13]), sum(prof[13:21]), sum(prof[21:24] + prof[0:5])]
    assert counts == [4.0, 4.0, 1.0]
    cases[s["case"]] = {"annual_kWh": annual, "peak_kW": peak,
                         "block_counts_h": counts, "hourly_sha256": s["hourly_sha256"]}

base = cases["seeded"]
deltas = {}
for name in ("early", "late"):
    v = cases[name]
    deltas[name] = {
        "annual_minus_seeded_kWh": v["annual_kWh"] - base["annual_kWh"],
        "annual_minus_seeded_percent": 100 * (v["annual_kWh"] / base["annual_kWh"] - 1),
        "peak_minus_seeded_kW": v["peak_kW"] - base["peak_kW"],
        "peak_minus_seeded_percent": 100 * (v["peak_kW"] / base["peak_kW"] - 1),
    }

result = {
    "status": "arithmetic/profile check passed; no empirical validation",
    "source_run": "05_simulations/padova_pilot_v1",
    "summary_sha256": hashlib.sha256((RUN / "summary.json").read_bytes()).hexdigest(),
    "completion_manifest_sha256": hashlib.sha256((RUN / "completion_manifest.json").read_bytes()).hexdigest(),
    "cases": cases, "deltas": deltas,
    "caveat": "Single assumed dwelling, survey row, code adapter and Padova-coordinate TMY; thermal demand only.",
}
out = Path(__file__).resolve().parent / "author_result_check_v00.json"
if out.exists():
    raise SystemExit(f"Immutable check exists: {out}")
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "deltas": deltas}))
