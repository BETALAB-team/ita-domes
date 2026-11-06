"""Independently integrate the archetype v2 hourly outputs and mixture losses.

Run only after all four site completion manifests and technical checks exist.
This arithmetic audit does not itself certify building physics or measured accuracy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "05_simulations/archetype_campaign_v2/pre_run"
CAMPAIGN = ROOT / "05_simulations/archetype_campaign_v2"
OUT = Path(__file__).resolve().parent / "author_archetype_result_check_v2.json"
SITES = ("Padova", "Bolzano", "Roma", "Palermo")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def endpoint_from_hourly(path: Path) -> tuple[float, float, int, float]:
    n = 0
    pos = []
    peak = 0.0
    min_load = 0.0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert int(row["hour"]) == n, (path, n)
            p = float(row["sensible_load_W"])
            assert math.isfinite(p)
            if p > 0:
                pos.append(p)
                peak = max(peak, p)
            min_load = min(min_load, p)
            n += 1
    assert n == 8760, path
    return math.fsum(pos) / 1000, peak / 1000, n, min_load


mix = json.loads((PRE / "mixture_probabilities.json").read_text(encoding="utf-8"))
roles = read_csv(PRE / "role_manifest.csv")
strata = read_csv(PRE / "stratum_counts.csv")
tests = [r for r in roles if r["role"] == "test"]
components = [r for r in roles if r["role"] == "component"]
assert len(tests) == 32 and len(components) == 4
assert Counter((r["archetype_id"], r["control_branch"]) for r in tests) == Counter({(str(g), b): 2 for g in range(1, 5) for b in mix["branch_order"]})
stratum_count = {(r["archetype_id"], r["control_branch"]): int(r["eligible_n"]) for r in strata}
assert len(stratum_count) == 16 and sum(stratum_count.values()) == 13624
component_profile = {r["control_branch"]: r["profile_id"] for r in components}
assert set(component_profile) == set(mix["branch_order"])
assert len(set(component_profile.values())) == 4

result = {
    "status": "pass if every site completes; same-model arithmetic and provenance check only",
    "campaign": "archetype_campaign_v2",
    "source_hashes": {"selection_manifest": sha(PRE / "selection_manifest.json"),
                      "role_manifest": sha(PRE / "role_manifest.csv"),
                      "mixture_probabilities": sha(PRE / "mixture_probabilities.json")},
    "sites": {},
}
for site in SITES:
    run = CAMPAIGN / site
    manifest = json.loads((run / "completion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["checks_pass"]
    assert sha(run / "role_manifest.csv") == result["source_hashes"]["role_manifest"]
    assert sha(run / "mixture_probabilities.json") == result["source_hashes"]["mixture_probabilities"]
    for fname in ("checks.json", "full_step_balance_checks.json", "independent_surface_forcing_checks.json", "profile_preservation_checks.json", "site_gate.json"):
        assert json.loads((run / fname).read_text(encoding="utf-8"))["pass"], (site, fname)
    summary = {x["case"]: x for x in json.loads((run / "summary.json").read_text(encoding="utf-8"))}
    assert len(summary) == 38 and set(summary) == {r["profile_id"] for r in roles} | {"repeat_component", "warmup_component"}
    assert manifest["summary_sha256"] == sha(run / "summary.csv")
    integrated = {}
    for name, report in summary.items():
        path = run / f"{name}_hourly.csv"
        assert sha(path) == report["hourly_sha256"], (site, name, "hourly hash")
        heat, peak, n, min_load = endpoint_from_hourly(path)
        assert math.isclose(heat, report["annual_sensible_heating_kWh"], abs_tol=1e-7), (site, name, "heat")
        assert math.isclose(peak, report["peak_sensible_heating_kW"], abs_tol=1e-9), (site, name, "peak")
        assert min_load >= -1e-8, (site, name, "cooling boundary")
        integrated[name] = {"heat_kWh": heat, "peak_kW": peak, "rows": n, "hourly_sha256": report["hourly_sha256"]}
    reference = component_profile["22/19"]
    assert integrated[reference]["hourly_sha256"] != ""
    assert (run / f"{reference}_hourly.csv").read_bytes() == (run / "repeat_component_hourly.csv").read_bytes()
    comp_endpoint = {
        endpoint: {b: integrated[pid][endpoint] for b, pid in component_profile.items()}
        for endpoint in ("heat_kWh", "peak_kW")
    }
    losses = {}
    details = []
    for endpoint in ("heat_kWh", "peak_kW"):
        pooled = math.fsum(mix["pooled_P_branch"][b] * comp_endpoint[endpoint][b] for b in mix["branch_order"])
        conditional = {g: math.fsum(mix["training_P_branch_given_group"][g][b] * comp_endpoint[endpoint][b] for b in mix["branch_order"]) for g in ("1", "2", "3", "4")}
        square_pool = []
        square_cond = []
        signed_pool = []
        signed_cond = []
        abs_pool = []
        abs_cond = []
        by_group = {g: {"pool_sq": [], "cond_sq": []} for g in ("1", "2", "3", "4")}
        for row in tests:
            g, b = row["archetype_id"], row["control_branch"]
            y = integrated[row["profile_id"]][endpoint]
            w = stratum_count[(g, b)] / (2 * 13624)
            gw = stratum_count[(g, b)] / (2 * sum(stratum_count[(g, z)] for z in mix["branch_order"]))
            ep = pooled - y
            ec = conditional[g] - y
            square_pool.append(w * ep * ep)
            square_cond.append(w * ec * ec)
            signed_pool.append(w * ep)
            signed_cond.append(w * ec)
            abs_pool.append(w * abs(ep))
            abs_cond.append(w * abs(ec))
            by_group[g]["pool_sq"].append(gw * ep * ep)
            by_group[g]["cond_sq"].append(gw * ec * ec)
            details.append({"site": site, "endpoint": endpoint, "role_id": row["role_id"], "household_id": int(row["household_id"]),
                            "archetype": int(g), "branch": b, "eligible_weight": w, "detailed_model": y,
                            "conditional_prediction": conditional[g], "pooled_prediction": pooled,
                            "conditional_signed_deviation": ec, "pooled_signed_deviation": ep,
                            "paired_squared_deviation_difference": ec * ec - ep * ep})
        mse_pool, mse_cond = math.fsum(square_pool), math.fsum(square_cond)
        losses[endpoint] = {
            "pooled_prediction": pooled,
            "conditional_predictions": conditional,
            "pooled_weighted_mse": mse_pool,
            "conditional_weighted_mse": mse_cond,
            "paired_weighted_mse_difference_conditional_minus_pooled": mse_cond - mse_pool,
            "pooled_weighted_rmse": math.sqrt(mse_pool),
            "conditional_weighted_rmse": math.sqrt(mse_cond),
            "pooled_weighted_signed_deviation": math.fsum(signed_pool),
            "conditional_weighted_signed_deviation": math.fsum(signed_cond),
            "pooled_weighted_absolute_deviation": math.fsum(abs_pool),
            "conditional_weighted_absolute_deviation": math.fsum(abs_cond),
            "group_rmse": {g: {"pooled": math.sqrt(math.fsum(by_group[g]["pool_sq"])), "conditional": math.sqrt(math.fsum(by_group[g]["cond_sq"]))} for g in by_group},
            "component_endpoints": comp_endpoint[endpoint],
        }
    result["sites"][site] = {"summary_sha256": sha(run / "summary.csv"), "manifest_sha256": sha(run / "completion_manifest.json"),
                             "integrated_case_count": len(integrated), "losses": losses, "individual": details}

published = read_csv(ROOT / "04_analysis/archetype_extension/results_v2/weighted_metrics.csv")
endpoint_alias = {"annual_sensible_heating_kWh": "heat_kWh", "peak_sensible_heating_kW": "peak_kW"}
for row in published:
    ours = result["sites"][row["site"]]["losses"][endpoint_alias[row["endpoint"]]]
    if row["domain"].startswith("archetype_"):
        group = row["domain"].split("_")[-1]
        assert math.isclose(float(row["conditional_RMSE"]), ours["group_rmse"][group]["conditional"], rel_tol=1e-12, abs_tol=1e-8)
        assert math.isclose(float(row["pooled_RMSE"]), ours["group_rmse"][group]["pooled"], rel_tol=1e-12, abs_tol=1e-8)
        continue
    assert row["domain"] == "all_eligible_empirical_groups"
    comparisons = {
        "conditional_MSE": ours["conditional_weighted_mse"],
        "conditional_RMSE": ours["conditional_weighted_rmse"],
        "conditional_MAE": ours["conditional_weighted_absolute_deviation"],
        "conditional_signed_deviation": ours["conditional_weighted_signed_deviation"],
        "pooled_MSE": ours["pooled_weighted_mse"],
        "pooled_RMSE": ours["pooled_weighted_rmse"],
        "pooled_MAE": ours["pooled_weighted_absolute_deviation"],
        "pooled_signed_deviation": ours["pooled_weighted_signed_deviation"],
        "paired_MSE_difference_conditional_minus_pooled": ours["paired_weighted_mse_difference_conditional_minus_pooled"],
    }
    for field, value in comparisons.items():
        assert math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-7), (row["site"], row["endpoint"], field)
assert sum(r["domain"] == "all_eligible_empirical_groups" for r in published) == 8
assert sum(r["domain"].startswith("archetype_") for r in published) == 32
campaign_manifest = json.loads((CAMPAIGN / "campaign_completion_manifest.json").read_text(encoding="utf-8"))
assert campaign_manifest["all_site_gates_pass"]
assert (campaign_manifest["scientific_executions"], campaign_manifest["technical_executions"], campaign_manifest["annual_executions_total"]) == (144, 8, 152)
assert campaign_manifest["exported_hourly_rows"] == 152 * 8760
result["cross_check"] = {"overall_weighted_metric_rows": 8, "subgroup_rmse_rows": 32, "published_weighted_metrics_sha256": sha(ROOT / "04_analysis/archetype_extension/results_v2/weighted_metrics.csv"),
                         "campaign_manifest_sha256": sha(CAMPAIGN / "campaign_completion_manifest.json"),
                         "all_five_site_check_files_pass": True, "all_method_overall_metrics_match": True}

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "sites": {s: {"heat": x["losses"]["heat_kWh"], "peak": x["losses"]["peak_kW"]} for s, x in result["sites"].items()}}, ensure_ascii=False))
