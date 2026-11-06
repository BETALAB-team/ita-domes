"""Independent setpoint partition check from corrected v3 hourly CSVs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "05_simulations/climate_campaign_v3"
TABLE = ROOT / "04_analysis/posthoc_setpoint_partition_v3.csv"
OUT = Path(__file__).resolve().parent / "author_posthoc_check_v01.json"
if OUT.exists():
    raise SystemExit(f"Refusing overwrite: {OUT}")
published = {(r["site"], r["schedule"], float(r["setpoint_C"])): r
             for r in csv.DictReader(TABLE.open(newline="", encoding="utf-8"))}
assert len(published) == 36
max_error = 0.0
for site in ("Padova", "Bolzano", "Roma", "Palermo"):
    for name in ("seeded", "early", "late"):
        p = CAMPAIGN / site / f"{name}_hourly.csv"
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 8760
        for temp in (0.0, 19.0, 22.0):
            selected = [float(r["sensible_load_W"]) for r in rows
                        if float(r["heating_setpoint_C"]) == temp]
            count = sum(q > 1e-6 for q in selected)
            energy = math.fsum(max(q, 0.0) for q in selected) / 1000
            tab = published[(site, name, temp)]
            assert len(selected) == int(tab["calendar_hours_at_setpoint"])
            assert count == int(tab["positive_heating_hours_gt_1e_minus6W"])
            error = abs(energy - float(tab["positive_sensible_heating_kWh"]))
            max_error = max(max_error, error)
            assert error < 1e-8
result = {"status": "post-hoc partition independently checked from 12 corrected hourly CSVs; not a primary endpoint",
          "table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
          "rows_checked": len(published), "max_energy_discrepancy_kWh": max_error,
          "padova_seeded_19C": {k: published[("Padova", "seeded", 19.0)][k]
                                for k in ("calendar_hours_at_setpoint", "positive_heating_hours_gt_1e_minus6W", "positive_sensible_heating_kWh")}}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "rows_checked": len(published), "max_error_kWh": max_error}))
