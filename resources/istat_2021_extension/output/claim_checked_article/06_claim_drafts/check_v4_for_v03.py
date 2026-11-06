"""Author-side independent arithmetic/input audit of frozen v4 files.

This is not an independent reviewer verdict or empirical validation.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = ("Padova", "Bolzano", "Roma", "Palermo")
CASES = ("seeded", "early", "late", "repeat_seeded", "warmup_seeded")
SIGMA = 5.6697e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def epw_sky(site):
    path = ROOT / f"05_simulations/climate_inputs_v2/{site}.epw"
    with path.open(newline="", encoding="utf-8") as f:
        epw = list(csv.reader(f))[8:]
    assert len(epw) == 8760
    cloud = [float(x[23]) for x in epw]
    assert set(cloud) == {99.0}
    delta = []
    for x in epw:
        ir_wh_m2 = float(x[12])
        tair_c = float(x[6])
        assert math.isfinite(ir_wh_m2) and ir_wh_m2 > 0
        tsky_c = (ir_wh_m2 / (1.0 * SIGMA)) ** 0.25 - 273.15
        delta.append(tair_c - tsky_c)
    return {"epw_sha256": sha(path), "rows": len(epw), "cloud_missing_count": sum(x == 99 for x in cloud), "mean_air_minus_sky_K": sum(delta) / len(delta)}


checks = []
baselines = {}
weather = {}
for site in SITES:
    folder = ROOT / f"05_simulations/climate_campaign_v4/{site}"
    summary = {x["case"]: x for x in json.loads((folder / "summary.json").read_text(encoding="utf-8"))}
    correction = json.loads((folder / "weather_forcing_correction.json").read_text(encoding="utf-8"))
    weather[site] = epw_sky(site)
    assert abs(weather[site]["mean_air_minus_sky_K"] - correction["corrected_IR_mean_delta_K"]) < 1e-10
    assert weather[site]["epw_sha256"] == correction["raw_epw_sha256"]
    assert correction["active_forcing_gate_pass"] is True
    for case in CASES:
        file = folder / f"{case}_hourly.csv"
        records = rows(file)
        assert len(records) == 8760
        assert sha(file) == summary[case]["hourly_sha256"]
        sensible = [float(r["sensible_load_W"]) for r in records]
        annual = sum(max(0.0, p) for p in sensible) / 1000.0
        peak = max(max(0.0, p) for p in sensible) / 1000.0
        err_ann = annual - summary[case]["annual_sensible_heating_kWh"]
        err_peak = peak - summary[case]["peak_sensible_heating_kW"]
        assert abs(err_ann) < 1e-7 and abs(err_peak) < 1e-10
        assert all(math.isfinite(float(r["air_temperature_C"])) for r in records)
        if case == "seeded":
            baselines[site] = {"annual": annual, "peak": peak, "hash": sha(file)}
        if case == "repeat_seeded":
            assert sha(file) == baselines[site]["hash"]
        checks.append({"site": site, "case": case, "rows": len(records), "sha256": sha(file), "annual_kWh": annual, "peak_kW": peak, "annual_summary_error_kWh": err_ann, "peak_summary_error_kW": err_peak})

table = rows(ROOT / "04_analysis/paired_schedule_contrasts_v4.csv")
assert len(table) == 12
by_case = {(x["site"], x["case"]): x for x in checks}
table_errors = []
for row in table:
    site, case = row["site"], row["schedule"]
    actual = by_case[(site, case)]
    base = baselines[site]
    expected = {
        "annual_sensible_heating_kWh": actual["annual_kWh"],
        "peak_sensible_heating_kW": actual["peak_kW"],
        "delta_annual_kWh": actual["annual_kWh"] - base["annual"],
        "delta_annual_percent": 100 * (actual["annual_kWh"] - base["annual"]) / base["annual"],
        "delta_peak_kW": actual["peak_kW"] - base["peak"],
        "delta_peak_percent": 100 * (actual["peak_kW"] - base["peak"]) / base["peak"],
    }
    maxerr = max(abs(float(row[k]) - v) for k, v in expected.items())
    assert maxerr < 1e-8
    table_errors.append({"site": site, "case": case, "max_abs_difference": maxerr})

replay = ROOT / "05_simulations/climate_campaign_v4_replay/Padova/seeded_hourly.csv"
assert replay.exists() and sha(replay) == baselines["Padova"]["hash"]
replay_manifest = json.loads((replay.parent / "completion_manifest.json").read_text(encoding="utf-8"))
assert replay_manifest["status"] == "complete_replay_seeded_only"
assert "--replay-only-seeded" in replay_manifest["command_argv"]

out = {
    "scope": "author-side independent standard-library EPW and hourly arithmetic check; not a reviewer verdict, full radiation audit, benchmark, or empirical validation",
    "sites": weather,
    "case_checks": checks,
    "paired_table_max_abs_error": max(x["max_abs_difference"] for x in table_errors),
    "padova_replay_seeded_byte_identical": True,
    "passed": True,
}
target = Path(__file__).with_name("author_v4_check_for_v03.json")
if target.exists():
    raise SystemExit(f"Refusing to overwrite {target}")
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"cases": len(checks), "sites": len(weather), "paired_table_max_abs_error": out["paired_table_max_abs_error"], "output": str(target)}))
