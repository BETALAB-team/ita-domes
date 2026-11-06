"""Run the comparable ISTAT archetype/PFE workflow for 2013 and 2021."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from archetype_pipeline import (
    AGE_RANGES_2021,
    EDUCATION_YEARS,
    EDUCATION_YEARS_2021,
    PipelineConfig,
    run_pipeline as run_archetypes,
)
from energy_analysis import EnergyConfig, run_pipeline as run_energy
from istat_harmonizer import harmonize_release
from pfe_pipeline import PFEConfig, run_pipeline as run_pfe
from report_pipeline import FIGURE_STEMS, generate_figures


BUILDING_AGE_RANGES_2021 = {
    1: (0, 11),
    2: (12, 21),
    3: (22, 31),
    4: (32, 41),
    5: (42, 51),
    6: (52, 61),
    7: (62, 71),
    8: (72, 150),
    9: (0, 70),
}

DWELLING_AREA_RANGES_2021 = {
    1: (10, 39),
    2: (40, 49),
    3: (50, 59),
    4: (60, 69),
    5: (70, 79),
    6: (80, 89),
    7: (90, 99),
    8: (100, 109),
    9: (110, 119),
    10: (120, 129),
    11: (130, 149),
    12: (150, 169),
    13: (170, 199),
    14: (200, 249),
    15: (250, 299),
    16: (300, 400),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    config["_base"] = base
    config["_output_root"] = (base / config["paths"]["output_root"]).resolve()
    config["_energy_results"] = (
        base / config["paths"]["energy_results"]
    ).resolve()
    config["_legend"] = (base / config["paths"]["legend"]).resolve()
    config["_crosswalk"] = (base / config["paths"]["crosswalk"]).resolve()
    return config


def _resolve_raw_input(config: dict[str, Any], year: int) -> Path:
    entry = config["years"][str(year)]
    base = config["_base"]
    if "raw_input" in entry:
        path = (base / entry["raw_input"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{year} raw input does not exist: {path}")
        return path
    matches = sorted(base.glob(entry["raw_glob"]))
    files = [path.resolve() for path in matches if path.is_file()]
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one {year} raw input for {entry['raw_glob']!r}, found {files}"
        )
    return files[0]


def _archetype_config(config: dict[str, Any], year: int) -> PipelineConfig:
    seeds = config["seeds"]
    if year == 2013:
        return PipelineConfig(
            age_seed=int(seeds["age"]),
            cluster_seed=int(seeds["cluster"]),
            survey_year=year,
            available_member_columns=12,
            education_years=dict(EDUCATION_YEARS),
            require_paper_tolerances=True,
        )
    return PipelineConfig(
        age_seed=int(seeds["age"]),
        cluster_seed=int(seeds["cluster"]),
        survey_year=year,
        available_member_columns=5,
        age_ranges=dict(AGE_RANGES_2021),
        education_years=dict(EDUCATION_YEARS_2021),
        unknown_education_codes=(0,),
        education_age_basis="realized_age",
        require_paper_tolerances=False,
    )


def _pfe_config(config: dict[str, Any], year: int) -> PFEConfig:
    parameters = config["parameters"]
    common = {
        "dwelling_seed": int(config["seeds"]["dwelling"]),
        "minimum_support": int(parameters["pfe_minimum_support"]),
        "allow_unsupported_windows": bool(
            parameters.get("allow_unsupported_pfe_windows", False)
        ),
        "survey_year": year,
    }
    if year == 2013:
        return PFEConfig(**common)
    return PFEConfig(
        **common,
        building_age_ranges=dict(BUILDING_AGE_RANGES_2021),
        dwelling_area_ranges=dict(DWELLING_AREA_RANGES_2021),
    )


def _year_files(year_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in year_dir.rglob("*")
        if path.is_file() and path.name not in {"workflow_manifest.json"}
    )


def run_year(config: dict[str, Any], year: int) -> dict[str, Any]:
    raw_input = _resolve_raw_input(config, year)
    year_dir = config["_output_root"] / str(year)
    harmonized = year_dir / "harmonized_istat.csv"
    baseline_dir = year_dir / "baseline"
    pfe_dir = year_dir / "pfe"
    energy_dir = year_dir / "energy"
    figure_dir = year_dir / "figures"

    harmonization = harmonize_release(
        year,
        raw_input,
        harmonized,
        config["_legend"],
    )
    archetypes = run_archetypes(
        harmonized,
        baseline_dir,
        _archetype_config(config, year),
    )
    if not archetypes["passed"]:
        raise RuntimeError(f"{year} archetype stage failed validation")

    household_archetypes = baseline_dir / "household_archetypes.csv"
    pfe = run_pfe(
        harmonized,
        household_archetypes,
        pfe_dir,
        _pfe_config(config, year),
    )
    if not pfe["passed"]:
        raise RuntimeError(f"{year} PFE stage failed validation")

    run_energy_stage = bool(config["years"][str(year)].get("run_energy", False))
    energy: dict[str, Any] | None = None
    if run_energy_stage:
        energy = run_energy(
            harmonized,
            household_archetypes,
            config["_energy_results"],
            energy_dir,
            EnergyConfig(
                dwelling_seed=int(config["seeds"]["dwelling"]),
                alpha=float(config["parameters"]["anova_alpha"]),
            ),
        )
        if not energy["passed"]:
            raise RuntimeError(f"{year} energy stage failed validation")

    figure_paths = generate_figures(
        harmonized,
        household_archetypes,
        baseline_dir / "archetype_summary.csv",
        baseline_dir / "archetype_model.json",
        pfe_dir / "dwelling_archetype_probabilities.csv",
        energy_dir / "energy_analysis_dataset.csv" if energy else None,
        energy_dir / "heating_tukey_groups.csv" if energy else None,
        energy_dir / "electricity_tukey_groups.csv" if energy else None,
        figure_dir,
        int(config["seeds"]["cluster"]),
        demographic_only=not run_energy_stage,
    )
    expected_stems = FIGURE_STEMS if run_energy_stage else FIGURE_STEMS[:7]
    figure_validation = {
        "expected_stems": expected_stems,
        "actual_stems": sorted(figure_paths),
        "all_svg_and_png_exist": all(
            paths[extension].is_file()
            for paths in figure_paths.values()
            for extension in ("svg", "png")
        ),
    }
    figure_validation["passed"] = (
        set(figure_paths) == set(expected_stems)
        and figure_validation["all_svg_and_png_exist"]
    )

    limitations = []
    if year == 2021:
        limitations.append(
            "The energy ANOVA/Tukey stage and its three figures are not run: "
            "the 2021 release has expenditure rather than the joined GRINS kWh "
            "response and lacks the 2013 thermal-insulation rating."
        )
        limitations.append(
            "Households coded as six or more members are excluded because the "
            "2021 microdata expose only five member-level age/education columns."
        )

    report = {
        "passed": bool(
            harmonization["passed"]
            and archetypes["passed"]
            and pfe["passed"]
            and (energy is None or energy["passed"])
            and figure_validation["passed"]
        ),
        "survey_year": year,
        "stages": {
            "harmonization": harmonization,
            "archetypes": archetypes,
            "pfe": pfe,
            "energy": energy,
            "figures": figure_validation,
        },
        "limitations": limitations,
    }
    validation_path = year_dir / "validation_report.json"
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "survey_year": year,
        "source_crosswalk": {
            "path": str(config["_crosswalk"]),
            "sha256": sha256_file(config["_crosswalk"]),
        },
        "outputs": {
            str(path.relative_to(year_dir)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in _year_files(year_dir)
        },
    }
    (year_dir / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_all(config: dict[str, Any], years: list[int]) -> dict[str, Any]:
    output_root = config["_output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    reports = {str(year): run_year(config, year) for year in years}

    summaries = []
    for year in years:
        summary = pd.read_csv(
            output_root / str(year) / "baseline" / "archetype_summary.csv"
        )
        summary.insert(0, "survey_year", year)
        summaries.append(summary)
    comparison_path = output_root / "archetype_summary_2013_2021.csv"
    pd.concat(summaries, ignore_index=True).to_csv(comparison_path, index=False)

    report = {
        "passed": all(item["passed"] for item in reports.values()),
        "years": reports,
        "comparison_summary": {
            "path": str(comparison_path.resolve()),
            "sha256": sha256_file(comparison_path),
        },
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "multiyear_workflow_config.json",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        choices=(2013, 2021),
        default=[2013, 2021],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = run_all(config, args.years)
    status = "PASS" if report["passed"] else "FAIL"
    for year in args.years:
        stages = report["years"][str(year)]["stages"]
        counts = stages["archetypes"]["row_counts"]
        print(
            f"{year}: {counts['eligible_households']} archetyped households | "
            f"{len(stages['figures']['actual_stems'])} figures"
        )
    print(f"Multi-year workflow: {status} | outputs: {config['_output_root']}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
