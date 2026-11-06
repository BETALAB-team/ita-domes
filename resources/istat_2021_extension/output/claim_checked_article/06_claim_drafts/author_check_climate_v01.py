"""Independent arithmetic/hash checks for the four-site primary output table."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "05_simulations/climate_campaign_v3"
TABLE = ROOT / "04_analysis/paired_schedule_contrasts_v3.csv"
out = Path(__file__).resolve().parent / "author_result_check_v01.json"
if out.exists():
    raise SystemExit(f"Refusing overwrite: {out}")

rows = list(csv.DictReader(TABLE.open(newline="", encoding="utf-8")))
assert len(rows) == 12
cases = {}
for site in ("Padova", "Bolzano", "Roma", "Palermo"):
    run = CAMPAIGN / site
    manifest = json.loads((run / "completion_manifest.json").read_text(encoding="utf-8"))
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    full = json.loads((run / "full_step_balance_checks.json").read_text(encoding="utf-8"))
    transition = json.loads((run / "transition_checks_pre_run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and checks["pass"] and full["pass"] and transition["pass"], site
    summaries = {x["case"]: x for x in json.loads((run / "summary.json").read_text(encoding="utf-8"))}
    site_cases = {}
    for name in ("seeded", "early", "late"):
        p = run / f"{name}_hourly.csv"
        s = summaries[name]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == s["hourly_sha256"]
        with p.open(newline="", encoding="utf-8") as f:
            hourly = list(csv.DictReader(f))
        assert len(hourly) == 8760
        q = [float(r["sensible_load_W"]) for r in hourly]
        assert all(math.isfinite(x) for x in q)
        annual = math.fsum(max(0, x) for x in q) / 1000
        peak = max(max(0, x) for x in q) / 1000
        assert math.isclose(annual, s["annual_sensible_heating_kWh"], abs_tol=1e-8)
        assert math.isclose(peak, s["peak_sensible_heating_kW"], abs_tol=1e-12)
        profile = s["daily_on_profile"]
        assert [sum(profile[5:13]), sum(profile[13:21]), sum(profile[21:24] + profile[0:5])] == [4, 4, 1]
        tab = next(r for r in rows if r["site"] == site and r["schedule"] == name)
        assert math.isclose(annual, float(tab["annual_sensible_heating_kWh"]), abs_tol=1e-8)
        assert math.isclose(peak, float(tab["peak_sensible_heating_kW"]), abs_tol=1e-12)
        site_cases[name] = {"annual_kWh": annual, "peak_kW": peak, "hourly_sha256": s["hourly_sha256"]}
    for name in ("early", "late"):
        base, alt = site_cases["seeded"], site_cases[name]
        tab = next(r for r in rows if r["site"] == site and r["schedule"] == name)
        delta = alt["annual_kWh"] - base["annual_kWh"]
        peak_delta = alt["peak_kW"] - base["peak_kW"]
        assert math.isclose(delta, float(tab["delta_annual_kWh"]), abs_tol=1e-8)
        assert math.isclose(peak_delta, float(tab["delta_peak_kW"]), abs_tol=1e-12)
        assert math.isclose(100 * delta / base["annual_kWh"], float(tab["delta_annual_percent"]), abs_tol=1e-10)
        assert math.isclose(100 * peak_delta / base["peak_kW"], float(tab["delta_peak_percent"]), abs_tol=1e-10)
    cases[site] = {"runs": site_cases, "checks_pass": True,
                   "full_step_checks_sha256": hashlib.sha256((run / "full_step_balance_checks.json").read_bytes()).hexdigest(),
                   "completion_manifest_sha256": hashlib.sha256((run / "completion_manifest.json").read_bytes()).hexdigest()}

out.write_text(json.dumps({
    "status": "12 primary outputs independently integrated and table deltas checked; no empirical validation",
    "source_table": "04_analysis/paired_schedule_contrasts_v3.csv",
    "source_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
    "sites": cases,
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "sites": list(cases), "primary_runs": 12}))
