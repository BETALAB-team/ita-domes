"""Deterministic windowed probabilistic feature extraction (PFE).

This stage joins the raw ISTAT dwelling classes to the restored household
archetypes, converts dwelling age and area classes to seeded continuous values,
and calculates empirical archetype probabilities inside centered windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from column_mapping import read_mapped_csv


BUILDING_AGE_COLUMN = "q_2_4_ric"
DWELLING_AREA_COLUMN = "q_2_7_class"

# Inclusive ranges recovered from the analysis that underlies the article.
BUILDING_AGE_RANGES = {
    1: (0, 13),
    2: (14, 23),
    3: (24, 33),
    4: (34, 43),
    5: (44, 53),
    6: (54, 63),
    7: (64, 113),
    8: (113, 150),
    9: (0, 70),
}
DWELLING_AREA_RANGES = {
    1: (10, 19),
    2: (20, 39),
    3: (40, 59),
    4: (60, 89),
    5: (90, 119),
    6: (120, 149),
    7: (150, 400),
}

# Published discretization and window configuration.
BUILDING_AGE_GRID = np.arange(20, 141, 20, dtype=int)
DWELLING_AREA_GRID = np.arange(40, 361, 40, dtype=int)
BUILDING_AGE_HALF_WINDOW = 6
DWELLING_AREA_HALF_WINDOW = 8
N_ARCHETYPES = 4


@dataclass(frozen=True)
class PFEConfig:
    dwelling_seed: int = 42
    minimum_support: int = 5
    allow_unsupported_windows: bool = False
    survey_year: int = 2013
    building_age_ranges: dict[int, tuple[int, int]] | None = None
    dwelling_area_ranges: dict[int, tuple[int, int]] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(raw: pd.DataFrame, archetypes: pd.DataFrame) -> None:
    raw_required = {"id", BUILDING_AGE_COLUMN, DWELLING_AREA_COLUMN}
    archetype_required = {"id", "archetype"}
    raw_missing = sorted(raw_required - set(raw.columns))
    archetype_missing = sorted(archetype_required - set(archetypes.columns))
    if raw_missing:
        raise ValueError(f"Raw ISTAT data is missing columns: {raw_missing}")
    if archetype_missing:
        raise ValueError(f"Archetype data is missing columns: {archetype_missing}")
    if raw["id"].duplicated().any():
        raise ValueError("Raw ISTAT household IDs must be unique")
    if archetypes["id"].duplicated().any():
        raise ValueError("Archetype household IDs must be unique")

    labels = set(archetypes["archetype"].dropna().astype(int).unique())
    expected = set(range(1, N_ARCHETYPES + 1))
    if labels != expected:
        raise ValueError(f"Expected archetypes {sorted(expected)}, found {sorted(labels)}")


def deterministic_values_from_classes(
    classes: np.ndarray,
    ranges: dict[int, tuple[int, int]],
    rng: np.random.RandomState,
    variable_name: str,
) -> np.ndarray:
    values = np.empty(len(classes), dtype=int)
    for index, raw_class in enumerate(classes):
        raw_class = int(raw_class)
        if raw_class not in ranges:
            raise ValueError(
                f"Unsupported {variable_name} class {raw_class} at row {index}"
            )
        lower, upper = ranges[raw_class]
        values[index] = int(rng.randint(lower, upper + 1))
    return values


def build_dwelling_samples(
    raw: pd.DataFrame,
    archetypes: pd.DataFrame,
    dwelling_seed: int,
    building_age_ranges: dict[int, tuple[int, int]] | None = None,
    dwelling_area_ranges: dict[int, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Join inputs and deterministically expand dwelling age and area classes."""
    validate_inputs(raw, archetypes)
    dwelling_columns = raw[["id", BUILDING_AGE_COLUMN, DWELLING_AREA_COLUMN]]
    samples = archetypes[["id", "archetype"]].merge(
        dwelling_columns,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if samples[[BUILDING_AGE_COLUMN, DWELLING_AREA_COLUMN]].isna().any().any():
        missing_ids = samples.loc[
            samples[[BUILDING_AGE_COLUMN, DWELLING_AREA_COLUMN]].isna().any(axis=1),
            "id",
        ].astype(int).tolist()
        raise ValueError(f"Dwelling information is missing for IDs: {missing_ids[:10]}")

    # One generator and a fixed call order reproduce the original two sequential
    # randomization passes while making the result stable across runs.
    rng = np.random.RandomState(dwelling_seed)
    resolved_age_ranges = BUILDING_AGE_RANGES if building_age_ranges is None else building_age_ranges
    resolved_area_ranges = (
        DWELLING_AREA_RANGES if dwelling_area_ranges is None else dwelling_area_ranges
    )
    samples["building_age_years"] = deterministic_values_from_classes(
        samples[BUILDING_AGE_COLUMN].to_numpy(dtype=int),
        resolved_age_ranges,
        rng,
        "building-age",
    )
    samples["dwelling_area_m2"] = deterministic_values_from_classes(
        samples[DWELLING_AREA_COLUMN].to_numpy(dtype=int),
        resolved_area_ranges,
        rng,
        "dwelling-area",
    )
    return samples


def calculate_window_probabilities(
    samples: pd.DataFrame,
    minimum_support: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Count archetypes in every published age/area window."""
    ages = samples["building_age_years"].to_numpy(dtype=int)
    areas = samples["dwelling_area_m2"].to_numpy(dtype=int)
    archetypes = samples["archetype"].to_numpy(dtype=int)
    ids = samples["id"].to_numpy(dtype=int)

    supported_rows: list[dict[str, Any]] = []
    unsupported_rows: list[dict[str, int]] = []
    contributing_ids: set[int] = set()

    for area_center in DWELLING_AREA_GRID:
        for age_center in BUILDING_AGE_GRID:
            age_lower = int(age_center - BUILDING_AGE_HALF_WINDOW)
            age_upper = int(age_center + BUILDING_AGE_HALF_WINDOW)
            area_lower = int(area_center - DWELLING_AREA_HALF_WINDOW)
            area_upper = int(area_center + DWELLING_AREA_HALF_WINDOW)
            selected = (
                (ages >= age_lower)
                & (ages <= age_upper)
                & (areas >= area_lower)
                & (areas <= area_upper)
            )
            total = int(selected.sum())
            base = {
                "building_age_years": int(age_center),
                "dwelling_area_m2": int(area_center),
                "age_window_lower": age_lower,
                "age_window_upper": age_upper,
                "area_window_lower": area_lower,
                "area_window_upper": area_upper,
            }
            if total == 0:
                unsupported_rows.append(base)
                continue

            contributing_ids.update(ids[selected].tolist())
            counts = np.bincount(archetypes[selected], minlength=N_ARCHETYPES + 1)[1:]
            if int(counts.sum()) != total:
                raise RuntimeError("Archetype counts do not equal the window sample count")
            probabilities = counts.astype(float) / total
            # Normalize once more to make the invariant explicit and robust to
            # floating-point division order.
            probabilities /= probabilities.sum()

            row: dict[str, Any] = {
                **base,
                "sample_count": total,
                "meets_minimum_support": total >= minimum_support,
            }
            for archetype in range(1, N_ARCHETYPES + 1):
                row[f"archetype_{archetype}_count"] = int(counts[archetype - 1])
                row[f"prob_archetype_{archetype}"] = float(
                    probabilities[archetype - 1]
                )
            row["probability_sum"] = float(probabilities.sum())
            supported_rows.append(row)

    supported = pd.DataFrame.from_records(supported_rows)
    unsupported = pd.DataFrame.from_records(
        unsupported_rows,
        columns=[
            "building_age_years",
            "dwelling_area_m2",
            "age_window_lower",
            "age_window_upper",
            "area_window_lower",
            "area_window_upper",
        ],
    )
    return supported, unsupported, contributing_ids


def validate_probability_dataset(probabilities: pd.DataFrame) -> dict[str, Any]:
    probability_columns = [
        f"prob_archetype_{archetype}"
        for archetype in range(1, N_ARCHETYPES + 1)
    ]
    count_columns = [
        f"archetype_{archetype}_count"
        for archetype in range(1, N_ARCHETYPES + 1)
    ]
    probability_matrix = probabilities[probability_columns].to_numpy(dtype=float)
    count_matrix = probabilities[count_columns].to_numpy(dtype=int)
    sample_counts = probabilities["sample_count"].to_numpy(dtype=int)

    row_sums = probability_matrix.sum(axis=1)
    expected_probabilities = count_matrix / sample_counts[:, None]
    unique_grid = not probabilities.duplicated(
        ["building_age_years", "dwelling_area_m2"]
    ).any()
    checks = {
        "dataset_is_nonempty": not probabilities.empty,
        "all_sample_counts_are_positive": bool(np.all(sample_counts > 0)),
        "archetype_counts_equal_sample_counts": bool(
            np.array_equal(count_matrix.sum(axis=1), sample_counts)
        ),
        "probabilities_are_finite": bool(np.isfinite(probability_matrix).all()),
        "probabilities_are_bounded": bool(
            ((probability_matrix >= 0.0) & (probability_matrix <= 1.0)).all()
        ),
        "probabilities_sum_to_one": bool(
            np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12)
        ),
        "probabilities_equal_count_ratios": bool(
            np.allclose(
                probability_matrix,
                expected_probabilities,
                rtol=0.0,
                atol=1e-15,
            )
        ),
        "grid_coordinates_are_unique": bool(unique_grid),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_probability_sum_error": float(np.max(np.abs(row_sums - 1.0))),
        "minimum_probability": float(probability_matrix.min()),
        "maximum_probability": float(probability_matrix.max()),
    }


def run_pipeline(
    raw_path: Path,
    archetypes_path: Path,
    output_dir: Path,
    config: PFEConfig,
    source_to_canonical: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = read_mapped_csv(
        raw_path,
        ["id", BUILDING_AGE_COLUMN, DWELLING_AREA_COLUMN],
        source_to_canonical,
        "Raw ISTAT dwelling input",
        sep=";",
    )
    archetypes = pd.read_csv(archetypes_path, usecols=["id", "archetype"])
    samples = build_dwelling_samples(
        raw,
        archetypes,
        config.dwelling_seed,
        config.building_age_ranges,
        config.dwelling_area_ranges,
    )
    probabilities, unsupported, contributing_ids = calculate_window_probabilities(
        samples,
        config.minimum_support,
    )
    validation = validate_probability_dataset(probabilities)

    if not unsupported.empty and not config.allow_unsupported_windows:
        coordinates = unsupported[["building_age_years", "dwelling_area_m2"]]
        raise RuntimeError(
            "The configured grid contains unsupported windows: "
            f"{coordinates.to_dict(orient='records')}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dwelling_archetype_probabilities.csv"
    model_path = output_dir / "pfe_model.json"
    validation_path = output_dir / "validation_report.json"
    unsupported_path = output_dir / "unsupported_windows.csv"

    probabilities.to_csv(dataset_path, index=False, float_format="%.15g")
    unsupported.to_csv(unsupported_path, index=False)

    resolved_age_ranges = (
        BUILDING_AGE_RANGES
        if config.building_age_ranges is None
        else config.building_age_ranges
    )
    resolved_area_ranges = (
        DWELLING_AREA_RANGES
        if config.dwelling_area_ranges is None
        else config.dwelling_area_ranges
    )
    model = {
        "method": "windowed empirical probabilistic feature extraction",
        "survey_year": config.survey_year,
        "dwelling_seed": config.dwelling_seed,
        "building_age_class_ranges_inclusive": resolved_age_ranges,
        "dwelling_area_class_ranges_inclusive": resolved_area_ranges,
        "grid": {
            "building_age_years": BUILDING_AGE_GRID.tolist(),
            "dwelling_area_m2": DWELLING_AREA_GRID.tolist(),
        },
        "windows": {
            "building_age_half_width_years": BUILDING_AGE_HALF_WINDOW,
            "dwelling_area_half_width_m2": DWELLING_AREA_HALF_WINDOW,
            "bounds_are_inclusive": True,
        },
        "minimum_support": config.minimum_support,
    }
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    report = {
        "passed": validation["passed"] and (
            unsupported.empty or config.allow_unsupported_windows
        ),
        "source": {
            "title": "Identification of Italian user archetypes for building energy simulations based on a national survey",
            "doi": "10.1016/j.enbuild.2025.115924",
            "raw_input": str(raw_path.resolve()),
            "raw_input_sha256": sha256_file(raw_path),
            "archetype_input": str(archetypes_path.resolve()),
            "archetype_input_sha256": sha256_file(archetypes_path),
        },
        "configuration": {
            "dwelling_seed": config.dwelling_seed,
            "survey_year": config.survey_year,
            "minimum_support": config.minimum_support,
            "source_to_canonical": source_to_canonical or {},
            "building_age_grid_step_years": 20,
            "dwelling_area_grid_step_m2": 40,
            "building_age_half_window_years": BUILDING_AGE_HALF_WINDOW,
            "dwelling_area_half_window_m2": DWELLING_AREA_HALF_WINDOW,
        },
        "row_counts": {
            "archetyped_households": int(len(samples)),
            "configured_windows": int(
                len(BUILDING_AGE_GRID) * len(DWELLING_AREA_GRID)
            ),
            "supported_windows": int(len(probabilities)),
            "unsupported_windows": int(len(unsupported)),
            "windows_below_minimum_support": int(
                (~probabilities["meets_minimum_support"]).sum()
            ),
            "unique_households_contributing_to_windows": len(contributing_ids),
        },
        "sample_count_range": {
            "minimum": int(probabilities["sample_count"].min()),
            "maximum": int(probabilities["sample_count"].max()),
        },
        "probability_validation": validation,
        "outputs": {
            "probabilities": str(dataset_path.resolve()),
            "model": str(model_path.resolve()),
            "validation": str(validation_path.resolve()),
            "unsupported_windows": str(unsupported_path.resolve()),
        },
    }
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=script_dir / "istat_microdata_2013.csv",
    )
    parser.add_argument(
        "--archetypes-input",
        type=Path,
        default=script_dir / "baseline_outputs" / "household_archetypes.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "pfe_outputs",
    )
    parser.add_argument("--dwelling-seed", type=int, default=42)
    parser.add_argument("--minimum-support", type=int, default=5)
    parser.add_argument("--allow-unsupported-windows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PFEConfig(
        dwelling_seed=args.dwelling_seed,
        minimum_support=args.minimum_support,
        allow_unsupported_windows=args.allow_unsupported_windows,
    )
    report = run_pipeline(
        args.raw_input,
        args.archetypes_input,
        args.output_dir,
        config,
    )
    status = "PASS" if report["passed"] else "FAIL"
    counts = report["row_counts"]
    print(
        f"Windowed PFE: {status} | {counts['supported_windows']} supported windows | "
        f"{counts['archetyped_households']} archetyped households"
    )
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
