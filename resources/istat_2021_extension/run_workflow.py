"""Regenerate and verify the complete ISTAT archetype workflow in one command."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from archetype_pipeline import (
    ARCHETYPE_INPUT_COLUMNS,
    PipelineConfig,
    run_pipeline as run_archetypes,
)
from column_mapping import normalize_mapping
from energy_analysis import (
    ENERGY_COLUMNS,
    RAW_COLUMNS,
    EnergyConfig,
    run_pipeline as run_energy,
)
from pfe_pipeline import PFEConfig, run_pipeline as run_pfe
from report_pipeline import ReportConfig, run_pipeline as run_report


BASELINE_SCHEMA = [
    "id",
    "agemin",
    "agemax",
    "nmembers",
    "education",
    "pc1",
    "pc2",
    "pc1_scaled",
    "pc2_scaled",
    "education_scaled",
    "archetype",
]
PFE_SCHEMA = [
    "building_age_years",
    "dwelling_area_m2",
    "age_window_lower",
    "age_window_upper",
    "area_window_lower",
    "area_window_upper",
    "sample_count",
    "meets_minimum_support",
    "archetype_1_count",
    "prob_archetype_1",
    "archetype_2_count",
    "prob_archetype_2",
    "archetype_3_count",
    "prob_archetype_3",
    "archetype_4_count",
    "prob_archetype_4",
    "probability_sum",
]
ENERGY_SCHEMA = [
    "id",
    "archetype",
    "dwelling_type",
    "location",
    "insulation",
    "insulation_level",
    "building_age_years",
    "dwelling_area_m2",
    "cooling_daily_hours",
    "cooling_system_use",
    "heating_daily_hours",
    "electricity_annual_kwh",
    "heating_tukey_group",
    "electricity_tukey_group",
]
ANOVA_SCHEMA = [
    "response",
    "term",
    "model_term",
    "degrees_of_freedom",
    "sum_squares",
    "f_statistic",
    "p_value",
    "significant_at_alpha",
    "paper_f_statistic",
    "paper_p_label",
    "paper_significant_at_0_05",
    "paper_significance_match",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_sections = {"paths", "seeds", "parameters", "validation"}
    missing_sections = sorted(required_sections - set(config))
    if missing_sections:
        raise ValueError(f"Workflow config is missing sections: {missing_sections}")

    required_paths = {
        "raw_istat",
        "energy_results",
        "baseline_output_dir",
        "pfe_output_dir",
        "energy_output_dir",
        "workflow_output_dir",
        "report_output_file",
        "report_figure_dir",
        "report_validation_file",
    }
    missing_paths = sorted(required_paths - set(config["paths"]))
    if missing_paths:
        raise ValueError(f"Workflow config is missing paths: {missing_paths}")

    mappings = config.setdefault("column_mappings", {})
    if not isinstance(mappings, dict):
        raise TypeError("Workflow column_mappings must be a JSON object")
    raw_mapping = normalize_mapping(
        mappings.get("raw_istat", {}), "Raw ISTAT config"
    )
    energy_mapping = normalize_mapping(
        mappings.get("energy_results", {}), "Energy-results config"
    )
    unknown_raw_targets = sorted(
        set(raw_mapping.values())
        - (set(ARCHETYPE_INPUT_COLUMNS) | set(RAW_COLUMNS) | {"reg"})
    )
    unknown_energy_targets = sorted(
        set(energy_mapping.values()) - set(ENERGY_COLUMNS)
    )
    if unknown_raw_targets:
        raise ValueError(
            "Raw ISTAT mapping targets columns not used by the workflow: "
            f"{unknown_raw_targets}"
        )
    if unknown_energy_targets:
        raise ValueError(
            "Energy mapping targets columns not used by the workflow: "
            f"{unknown_energy_targets}"
        )
    config["column_mappings"] = {
        "raw_istat": raw_mapping,
        "energy_results": energy_mapping,
    }

    base = config_path.parent
    resolved_paths: dict[str, Path] = {}
    for key, value in config["paths"].items():
        candidate = Path(value)
        resolved_paths[key] = (
            candidate if candidate.is_absolute() else base / candidate
        ).resolve()
    config["_config_path"] = config_path
    config["_paths"] = resolved_paths
    return config


def workflow_files(config: dict[str, Any]) -> dict[str, Path]:
    paths = config["_paths"]
    return {
        "baseline_households": paths["baseline_output_dir"]
        / "household_archetypes.csv",
        "baseline_summary": paths["baseline_output_dir"] / "archetype_summary.csv",
        "baseline_exclusions": paths["baseline_output_dir"]
        / "preprocessing_exclusions.csv",
        "baseline_model": paths["baseline_output_dir"] / "archetype_model.json",
        "baseline_validation": paths["baseline_output_dir"] / "validation_report.json",
        "pfe_probabilities": paths["pfe_output_dir"]
        / "dwelling_archetype_probabilities.csv",
        "pfe_unsupported_windows": paths["pfe_output_dir"]
        / "unsupported_windows.csv",
        "pfe_model": paths["pfe_output_dir"] / "pfe_model.json",
        "pfe_validation": paths["pfe_output_dir"] / "validation_report.json",
        "energy_dataset": paths["energy_output_dir"] / "energy_analysis_dataset.csv",
        "energy_exclusions": paths["energy_output_dir"] / "energy_exclusions.csv",
        "anova_results": paths["energy_output_dir"] / "anova_results.csv",
        "heating_groups": paths["energy_output_dir"] / "heating_tukey_groups.csv",
        "electricity_groups": paths["energy_output_dir"]
        / "electricity_tukey_groups.csv",
        "energy_group_summary": paths["energy_output_dir"]
        / "energy_group_summary.csv",
        "energy_factor_summary": paths["energy_output_dir"]
        / "energy_factor_summary.csv",
        "tukey_merge_history": paths["energy_output_dir"]
        / "tukey_merge_history.csv",
        "energy_model": paths["energy_output_dir"] / "energy_model.json",
        "energy_validation": paths["energy_output_dir"] / "validation_report.json",
        "word_report": paths["report_output_file"],
        "report_validation": paths["report_validation_file"],
        **{
            f"report_figure_{stem}_{extension}": paths["report_figure_dir"]
            / f"{stem}.{extension}"
            for stem in [
                "study_process_flow",
                "pca_explained_variance",
                "clustering_elbow",
                "regional_cluster_centroids",
                "archetype_population_shares",
                "dwelling_probability_profiles",
                "probability_model_validation",
                "energy_group_boxplots",
                "electricity_standard_comparison",
                "group_assignment_matrix",
            ]
            for extension in ["svg", "png"]
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_outputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["_paths"]
    files = workflow_files(config)
    checks: dict[str, bool] = {
        "raw_istat_input_exists": paths["raw_istat"].is_file(),
        "energy_input_exists": paths["energy_results"].is_file(),
        "all_required_outputs_exist": all(path.is_file() for path in files.values()),
    }
    if not all(checks.values()):
        missing = [str(path) for path in files.values() if not path.is_file()]
        return {"passed": False, "checks": checks, "missing_outputs": missing}

    baseline = pd.read_csv(files["baseline_households"])
    probabilities = pd.read_csv(files["pfe_probabilities"])
    energy = pd.read_csv(files["energy_dataset"])
    anova = pd.read_csv(files["anova_results"])
    baseline_report = _read_json(files["baseline_validation"])
    pfe_report = _read_json(files["pfe_validation"])
    energy_report = _read_json(files["energy_validation"])
    report_validation = _read_json(files["report_validation"])

    probability_columns = [f"prob_archetype_{index}" for index in range(1, 5)]
    probability_sum = probabilities[probability_columns].sum(axis=1).to_numpy(float)
    checks.update(
        {
            "baseline_stage_passed": bool(baseline_report["passed"]),
            "pfe_stage_passed": bool(pfe_report["passed"]),
            "energy_stage_passed": bool(energy_report["passed"]),
            "word_report_stage_passed": bool(report_validation["passed"]),
            "baseline_schema_is_exact": baseline.columns.tolist() == BASELINE_SCHEMA,
            "pfe_schema_is_exact": probabilities.columns.tolist() == PFE_SCHEMA,
            "energy_schema_is_exact": energy.columns.tolist() == ENERGY_SCHEMA,
            "anova_schema_is_exact": anova.columns.tolist() == ANOVA_SCHEMA,
            "baseline_ids_are_unique": not baseline["id"].duplicated().any(),
            "energy_ids_are_unique": not energy["id"].duplicated().any(),
            "energy_ids_are_archetyped": set(energy["id"]).issubset(
                set(baseline["id"])
            ),
            "all_four_archetypes_are_present": set(baseline["archetype"].unique())
            == {1, 2, 3, 4},
            "pfe_has_63_published_windows": len(probabilities) == 63,
            "pfe_probabilities_sum_to_one": bool(
                np.allclose(probability_sum, 1.0, rtol=0.0, atol=1e-12)
            ),
            "pfe_probability_sum_column_is_consistent": bool(
                np.allclose(
                    probabilities["probability_sum"].to_numpy(float),
                    probability_sum,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "anova_has_two_complete_14_term_models": (
                anova.groupby("response")["term"].nunique().to_dict()
                == {"electricity_annual_kwh": 14, "heating_daily_hours": 14}
            ),
            "anova_p_values_are_bounded": bool(
                anova["p_value"].between(0.0, 1.0).all()
            ),
            "all_energy_rows_have_tukey_groups": bool(
                energy[
                    ["heating_tukey_group", "electricity_tukey_group"]
                ].notna().all().all()
            ),
            "configured_seeds_reached_baseline": (
                baseline_report["configuration"]["age_seed"]
                == config["seeds"]["age"]
                and baseline_report["configuration"]["cluster_seed"]
                == config["seeds"]["cluster"]
            ),
            "configured_seed_reached_pfe": (
                pfe_report["configuration"]["dwelling_seed"]
                == config["seeds"]["dwelling"]
            ),
            "configured_seed_reached_energy": (
                energy_report["configuration"]["dwelling_seed"]
                == config["seeds"]["dwelling"]
            ),
            "configured_mapping_reached_baseline": (
                baseline_report["configuration"].get("source_to_canonical", {})
                == config["column_mappings"]["raw_istat"]
            ),
            "configured_mapping_reached_pfe": (
                pfe_report["configuration"].get("source_to_canonical", {})
                == config["column_mappings"]["raw_istat"]
            ),
            "configured_mappings_reached_energy": (
                energy_report["configuration"].get(
                    "raw_source_to_canonical", {}
                )
                == config["column_mappings"]["raw_istat"]
                and energy_report["configuration"].get(
                    "energy_source_to_canonical", {}
                )
                == config["column_mappings"]["energy_results"]
            ),
        }
    )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "row_counts": {
            "archetyped_households": int(len(baseline)),
            "pfe_windows": int(len(probabilities)),
            "energy_analysis_households": int(len(energy)),
            "anova_rows": int(len(anova)),
        },
        "paper_alignment": energy_report["validation"]["paper_alignment"],
    }


def write_workflow_reports(config: dict[str, Any], verification: dict[str, Any]) -> None:
    output_dir = config["_paths"]["workflow_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    files = workflow_files(config)
    manifest = {
        "configuration": str(config["_config_path"]),
        "inputs": {
            "raw_istat": {
                "path": str(config["_paths"]["raw_istat"]),
                "sha256": sha256_file(config["_paths"]["raw_istat"]),
            },
            "energy_results": {
                "path": str(config["_paths"]["energy_results"]),
                "sha256": sha256_file(config["_paths"]["energy_results"]),
            },
        },
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in files.items()
        },
    }
    (output_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )


def run_complete_workflow(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["_paths"]
    raw_path = paths["raw_istat"]
    energy_path = paths["energy_results"]
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw ISTAT input does not exist: {raw_path}")
    if not energy_path.is_file():
        raise FileNotFoundError(f"Energy input does not exist: {energy_path}")

    baseline_report = run_archetypes(
        raw_path,
        paths["baseline_output_dir"],
        PipelineConfig(
            age_seed=int(config["seeds"]["age"]),
            cluster_seed=int(config["seeds"]["cluster"]),
        ),
        config["column_mappings"]["raw_istat"],
    )
    if not baseline_report["passed"] and config["validation"].get(
        "require_paper_archetype_tolerances", True
    ):
        raise RuntimeError("Four-archetype baseline failed paper tolerances")

    archetypes_path = paths["baseline_output_dir"] / "household_archetypes.csv"
    pfe_report = run_pfe(
        raw_path,
        archetypes_path,
        paths["pfe_output_dir"],
        PFEConfig(
            dwelling_seed=int(config["seeds"]["dwelling"]),
            minimum_support=int(config["parameters"]["pfe_minimum_support"]),
            allow_unsupported_windows=bool(
                config["parameters"].get("allow_unsupported_pfe_windows", False)
            ),
        ),
        config["column_mappings"]["raw_istat"],
    )
    if not pfe_report["passed"]:
        raise RuntimeError("Windowed PFE stage failed validation")

    energy_report = run_energy(
        raw_path,
        archetypes_path,
        energy_path,
        paths["energy_output_dir"],
        EnergyConfig(
            dwelling_seed=int(config["seeds"]["dwelling"]),
            alpha=float(config["parameters"]["anova_alpha"]),
            expected_heating_groups=int(
                config["validation"]["paper_heating_groups"]
            ),
            accepted_electricity_groups=tuple(
                int(value)
                for value in config["validation"][
                    "accepted_paper_electricity_group_counts"
                ]
            ),
        ),
        config["column_mappings"]["raw_istat"],
        config["column_mappings"]["energy_results"],
    )
    if not energy_report["passed"]:
        raise RuntimeError("Energy ANOVA/Tukey stage failed structural validation")

    report_settings = config.get("report", {})
    report_report = run_report(
        raw_path,
        paths["baseline_output_dir"],
        paths["pfe_output_dir"],
        paths["energy_output_dir"],
        paths["report_figure_dir"],
        paths["report_output_file"],
        paths["report_validation_file"],
        ReportConfig(
            title=str(
                report_settings.get(
                    "title", "Italian Household Archetypes and Energy Analysis"
                )
            ),
            subtitle=str(
                report_settings.get(
                    "subtitle", "Automated methods and results report"
                )
            ),
            cluster_seed=int(config["seeds"]["cluster"]),
        ),
        config["column_mappings"]["raw_istat"],
    )
    if not report_report["passed"]:
        raise RuntimeError("Word report stage failed structural validation")

    verification = verify_outputs(config)
    write_workflow_reports(config, verification)
    return verification


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=script_dir / "workflow_config.json"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing outputs without regenerating them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.verify_only:
        verification = verify_outputs(config)
        write_workflow_reports(config, verification)
    else:
        verification = run_complete_workflow(config)
    status = "PASS" if verification["passed"] else "FAIL"
    counts = verification.get("row_counts", {})
    print(
        f"Complete workflow: {status} | "
        f"{counts.get('archetyped_households', 0)} archetyped households | "
        f"{counts.get('pfe_windows', 0)} PFE windows | "
        f"{counts.get('energy_analysis_households', 0)} energy rows"
    )
    print(f"Report: {(config['_paths']['workflow_output_dir'] / 'validation_report.json')}")
    return 0 if verification["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
