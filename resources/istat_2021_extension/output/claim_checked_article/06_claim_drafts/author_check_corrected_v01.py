"""Author-side independent arithmetic check of corrected Padova v3 outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "05_simulations/climate_campaign_v3/Padova"
OUT = Path(__file__).resolve().parent / "author_corrected_check_v01.json"
if OUT.exists():
    raise SystemExit(f"Refusing overwrite: {OUT}")
summary = {x["case"]: x for x in json.loads((RUN / "summary.json").read_text(encoding="utf-8"))}
full = json.loads((RUN / "full_step_balance_checks.json").read_text(encoding="utf-8"))
pre = json.loads((RUN / "transition_checks_pre_run.json").read_text(encoding="utf-8"))
manifest = json.loads((RUN / "completion_manifest.json").read_text(encoding="utf-8"))
assert full["pass"] and pre["pass"] and manifest["status"] == "complete" and manifest["checks_pass"]
cases = {}
for name in ("seeded", "early", "late", "repeat_seeded", "warmup_seeded"):
    path = RUN / f"{name}_hourly.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == summary[name]["hourly_sha256"]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8760
    q = [float(r["sensible_load_W"]) for r in rows]
    assert all(math.isfinite(x) for x in q)
    annual = math.fsum(max(0, x) for x in q) / 1000
    peak = max(max(0, x) for x in q) / 1000
    assert math.isclose(annual, summary[name]["annual_sensible_heating_kWh"], abs_tol=1e-8)
    assert math.isclose(peak, summary[name]["peak_sensible_heating_kW"], abs_tol=1e-12)
    prof = summary[name]["daily_on_profile"]
    assert [sum(prof[5:13]), sum(prof[13:21]), sum(prof[21:24] + prof[0:5])] == [4, 4, 1]
    check = next(x for x in full["cases"] if x["case"] == name)
    assert check["n"] == 8760 and check["n_residual_failure_gt_1e_minus6W"] == 0
    cases[name] = {"annual_kWh": annual, "peak_kW": peak,
                   "hourly_sha256": summary[name]["hourly_sha256"],
                   "max_node_residual_W": max(check["maximum_air_surface_mass_abs_residual_W"])}
base = cases["seeded"]
deltas = {}
for name in ("early", "late"):
    alt = cases[name]
    deltas[name] = {"annual_minus_seeded_kWh": alt["annual_kWh"] - base["annual_kWh"],
                    "annual_minus_seeded_percent": 100 * (alt["annual_kWh"] / base["annual_kWh"] - 1),
                    "peak_minus_seeded_kW": alt["peak_kW"] - base["peak_kW"],
                    "peak_minus_seeded_percent": 100 * (alt["peak_kW"] / base["peak_kW"] - 1)}
result = {"status": "corrected v3 Padova CSV arithmetic and declared checks independently inspected by author; independent reviewer still required",
          "run_path": "05_simulations/climate_campaign_v3/Padova",
          "completion_manifest_sha256": hashlib.sha256((RUN / "completion_manifest.json").read_bytes()).hexdigest(),
          "transition_checks_sha256": hashlib.sha256((RUN / "transition_checks_pre_run.json").read_bytes()).hexdigest(),
          "full_step_checks_sha256": hashlib.sha256((RUN / "full_step_balance_checks.json").read_bytes()).hexdigest(),
          "cases": cases, "deltas": deltas}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "deltas": deltas}))
