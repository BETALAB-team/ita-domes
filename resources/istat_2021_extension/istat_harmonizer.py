"""Harmonize the raw ISTAT 2013 and 2021 energy-survey releases.

The original workflow consumes a compact semicolon-delimited table with stable
canonical names. ISTAT publishes the two waves with different delimiters,
member-column counts, names, and category definitions. This module performs
only source-faithful name/format harmonization; year-specific category ranges
remain explicit in the downstream model configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from archetype_pipeline import AGE_COLUMNS, EDUCATION_COLUMNS


REGION_2021_TO_2013 = {
    10: 1,
    20: 2,
    30: 3,
    41: 4,
    42: 4,
    50: 5,
    60: 6,
    70: 7,
    80: 8,
    90: 9,
    100: 10,
    110: 11,
    120: 12,
    130: 13,
    140: 14,
    150: 15,
    160: 16,
    170: 17,
    180: 18,
    190: 19,
    200: 20,
}

CORE_COLUMNS = [
    "id",
    "survey_year",
    "q_1_1_sq1",
    *AGE_COLUMNS,
    *EDUCATION_COLUMNS,
    "reg",
    "rip",
    "q_2_4_ric",
    "q_2_7_class",
]

ENERGY_2013_COLUMNS = [
    "q_2_1",
    "q_3_17",
    "q_3_11A",
    "q_3_11B",
    "q_3_11C",
    "q_5_8A",
    "q_5_8B",
    "q_5_8C",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_selected(
    input_path: Path,
    delimiter: str,
    columns: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(
        input_path,
        sep=delimiter,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame


def _as_integer(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric.fillna(0).astype(int)


def _validate_harmonized(frame: pd.DataFrame, year: int) -> dict[str, Any]:
    checks = {
        "schema_is_exact": frame.columns.tolist()
        == CORE_COLUMNS + (ENERGY_2013_COLUMNS if year == 2013 else []),
        "ids_are_nonmissing": bool(frame["id"].notna().all()),
        "ids_are_unique": not frame["id"].duplicated().any(),
        "household_counts_are_nonnegative": bool((frame["q_1_1_sq1"] >= 0).all()),
        "regions_are_canonical": bool(frame["reg"].isin(range(1, 21)).all()),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _harmonize_2013(input_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    source_columns = [
        "nprog",
        "q_1_1_sq1",
        *AGE_COLUMNS,
        *EDUCATION_COLUMNS,
        "reg",
        "rip",
        "q_2_4_ric",
        "q_2_7_class",
        *ENERGY_2013_COLUMNS,
    ]
    raw = _read_selected(input_path, "\t", source_columns)
    raw = raw.rename(columns={"nprog": "id"})
    raw.insert(1, "survey_year", 2013)
    numeric_columns = [column for column in raw.columns if column != "survey_year"]
    _as_integer(raw, numeric_columns)
    result = raw[CORE_COLUMNS + ENERGY_2013_COLUMNS].copy()
    return result, {"nprog": "id"}


def _harmonize_2021(input_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    age_sources = [f"eta_mfr{index}" for index in range(1, 6)]
    education_sources = [f"istr_mfr{index}" for index in range(1, 6)]
    source_columns = [
        "ID_RECORD",
        "ncomp_mfr",
        *age_sources,
        *education_sources,
        "REGIO",
        "RIP",
        "v1_2_new",
        "v1_7_mfr",
    ]
    raw = _read_selected(input_path, "^", source_columns)
    rename = {
        "ID_RECORD": "id",
        "ncomp_mfr": "q_1_1_sq1",
        "REGIO": "reg",
        "RIP": "rip",
        "v1_2_new": "q_2_4_ric",
        "v1_7_mfr": "q_2_7_class",
        **{source: AGE_COLUMNS[index] for index, source in enumerate(age_sources)},
        **{
            source: EDUCATION_COLUMNS[index]
            for index, source in enumerate(education_sources)
        },
    }
    raw = raw.rename(columns=rename)
    raw.insert(1, "survey_year", 2021)
    for column in AGE_COLUMNS[5:] + EDUCATION_COLUMNS[5:]:
        raw[column] = 0
    numeric_columns = [column for column in raw.columns if column != "survey_year"]
    _as_integer(raw, numeric_columns)
    source_regions = raw["reg"].copy()
    raw["reg"] = source_regions.map(REGION_2021_TO_2013).fillna(0).astype(int)
    result = raw[CORE_COLUMNS].copy()
    return result, rename


def harmonize_release(
    year: int,
    input_path: Path,
    output_path: Path,
    legend_path: Path | None = None,
) -> dict[str, Any]:
    """Create one canonical, analysis-ready table without changing the source."""
    if year == 2013:
        frame, mapping = _harmonize_2013(input_path)
        available_member_columns = 12
    elif year == 2021:
        frame, mapping = _harmonize_2021(input_path)
        available_member_columns = 5
    else:
        raise ValueError(f"Unsupported ISTAT survey year: {year}")

    validation = _validate_harmonized(frame, year)
    if not validation["passed"]:
        failed = [
            name for name, passed in validation["checks"].items() if not passed
        ]
        raise ValueError(f"{year} harmonization failed checks: {failed}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep=";", index=False)
    report_path = output_path.with_name("harmonization_report.json")
    report = {
        "passed": True,
        "survey_year": year,
        "source": {
            "microdata": str(input_path.resolve()),
            "microdata_sha256": sha256_file(input_path),
            "legend": str(legend_path.resolve()) if legend_path else None,
            "legend_sha256": sha256_file(legend_path) if legend_path else None,
        },
        "mapping": mapping,
        "available_member_columns": available_member_columns,
        "row_counts": {
            "source_households": int(len(frame)),
            "households_with_zero_members": int((frame["q_1_1_sq1"] == 0).sum()),
            "households_exceeding_available_member_columns": int(
                (frame["q_1_1_sq1"] > available_member_columns).sum()
            ),
        },
        "validation": validation,
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "columns": frame.columns.tolist(),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
