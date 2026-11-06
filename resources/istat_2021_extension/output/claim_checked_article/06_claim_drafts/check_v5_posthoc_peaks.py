"""Author-side check of descriptive setpoint partitions and peak context."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


partition = read(root / "04_analysis/posthoc_setpoint_partition_v5.csv")
assert len(partition) == 36
rows = []
peaks = []
for site in ("Padova", "Bolzano", "Roma", "Palermo"):
    for case in ("seeded", "early", "late"):
        csvrows = read(root / f"05_simulations/climate_campaign_v5/{site}/{case}_hourly.csv")
        assert len(csvrows) == 8760
        positive_total = sum(max(0.0, float(x["sensible_load_W"])) for x in csvrows) / 1000
        for temp in (0, 19, 22):
            subgroup = [x for x in csvrows if abs(float(x["heating_setpoint_C"]) - temp) < 1e-9]
            positive = [max(0.0, float(x["sensible_load_W"])) for x in subgroup]
            row = next(x for x in partition if x["site"] == site and x["schedule"] == case and float(x["setpoint_C"]) == temp)
            expected_hours = sum(p > 1e-6 for p in positive)
            expected_energy = sum(positive) / 1000
            assert len(subgroup) == int(row["calendar_hours_at_setpoint"])
            assert expected_hours == int(row["positive_heating_hours_gt_1e_minus6W"])
            assert abs(expected_energy - float(row["positive_sensible_heating_kWh"])) < 1e-8
            assert abs(100 * expected_energy / positive_total - float(row["share_annual_sensible_heating_percent"])) < 1e-9
            rows.append({"site": site, "case": case, "setpoint_C": temp, "calendar_hours": len(subgroup), "positive_hours": expected_hours, "heating_kWh": expected_energy})
        hour = max(range(len(csvrows)), key=lambda i: max(0, float(csvrows[i]["sensible_load_W"])))
        peak = csvrows[hour]
        prev = csvrows[hour - 1] if hour else None
        peaks.append({"site": site, "case": case, "hour": hour, "peak_kW": max(0, float(peak["sensible_load_W"])) / 1000, "setpoint_C": float(peak["heating_setpoint_C"]), "previous_setpoint_C": float(prev["heating_setpoint_C"]) if prev else None, "old_air_C": float(peak["old_air_C"]), "previous_air_C": float(prev["air_temperature_C"]) if prev else None})

bolzano = [x for x in peaks if x["site"] == "Bolzano"]
assert len({x["hour"] for x in bolzano}) == 1
assert bolzano[0]["hour"] == 6912
assert all(x["previous_setpoint_C"] == 0 and x["setpoint_C"] == 19 for x in bolzano)
assert max(x["peak_kW"] for x in bolzano) - min(x["peak_kW"] for x in bolzano) < 1e-12

out = {"scope": "author-side post-hoc arithmetic and peak-context check; no new endpoint or empirical/physical validation", "setpoint_partitions": rows, "annual_peak_context": peaks, "bolzano_same_season_onset_peak_all_profiles": True, "pass": True}
target = Path(__file__).with_name("author_v5_posthoc_peak_check_v03.json")
if target.exists():
    raise SystemExit(f"Refusing overwrite: {target}")
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"partition_rows": len(rows), "peak_rows": len(peaks), "output": str(target)}))
