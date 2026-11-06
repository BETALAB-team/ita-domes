"""Paper-aligned energy ANOVA and iterative Tukey grouping.

This module consumes the restored household archetypes plus the raw ISTAT and
GRINS energy data.  It produces one analysis-ready household table, Type-II
ANOVA tables for daily heating-system use and annual electricity consumption,
and deterministic Tukey-Kramer group reductions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from column_mapping import read_mapped_csv
from pfe_pipeline import build_dwelling_samples


RAW_COLUMNS = [
    "id",
    "q_2_1",
    "q_2_4_ric",
    "q_2_7_class",
    "q_3_17",
    "rip",
    "q_3_11A",
    "q_3_11B",
    "q_3_11C",
    "q_5_8A",
    "q_5_8B",
    "q_5_8C",
]
ENERGY_COLUMNS = ["id", "appliances_el_kWh"]

DWELLING_TYPES = {
    1: "single_family_house",
    2: "multi_family_house",
    3: "apartment_small_building",
    4: "apartment_medium_building",
    5: "apartment_large_building",
}
LOCATIONS = {
    1: "north_west",
    2: "north_east",
    3: "central",
    4: "south_and_islands",
}
INSULATION = {
    1: "well_insulated",
    2: "sufficiently_insulated",
    3: "slightly_insulated",
    4: "not_insulated",
}
INSULATION_LEVEL = {
    1: "sufficient_or_high",
    2: "sufficient_or_high",
    3: "none_or_slight",
    4: "none_or_slight",
}

# Table 5 in the article.  ``paper_significant`` uses the paper's stated
# alpha=0.05 decision, while ``paper_p_label`` preserves the published value.
PAPER_ANOVA = {
    "Usercluster": {
        "heating_daily_hours": (11.74, "<0.01", True),
        "electricity_annual_kwh": (215.36, "<0.01", True),
    },
    "Dwellingtype": {
        "heating_daily_hours": (60.93, "<0.01", True),
        "electricity_annual_kwh": (5.44, "<0.01", True),
    },
    "Location": {
        "heating_daily_hours": (553.45, "<0.01", True),
        "electricity_annual_kwh": (5.98, "<0.01", True),
    },
    "Insulationlevel": {
        "heating_daily_hours": (10.42, "<0.01", True),
        "electricity_annual_kwh": (0.43, "0.92", False),
    },
    "Age_dwel": {
        "heating_daily_hours": (12.61, "<0.01", True),
        "electricity_annual_kwh": (62.35, "<0.01", True),
    },
    "Area_dwel": {
        "heating_daily_hours": (27.98, "<0.01", True),
        "electricity_annual_kwh": (241.70, "<0.01", True),
    },
    "CoolingSystemUse": {
        "heating_daily_hours": (0.20, "0.65", False),
        "electricity_annual_kwh": (19.06, "<0.01", True),
    },
    "Usercluster x Dwellingtype": {
        "heating_daily_hours": (2.41, "<0.01", True),
        "electricity_annual_kwh": (4.12, "<0.01", True),
    },
    "Usercluster x Location": {
        "heating_daily_hours": (2.18, "0.02", True),
        "electricity_annual_kwh": (5.11, "<0.01", True),
    },
    "Usercluster x InsulationLevel": {
        "heating_daily_hours": (1.45, "0.16", False),
        "electricity_annual_kwh": (0.48, "0.88", False),
    },
    "InsulationLevel x Location": {
        "heating_daily_hours": (1.09, "0.36", False),
        "electricity_annual_kwh": (0.95, "0.49", False),
    },
    "Dwellingtype x Location": {
        "heating_daily_hours": (10.02, "<0.01", True),
        "electricity_annual_kwh": (3.51, "0.02", True),
    },
    "Dwellingtype x InsulationLevel": {
        "heating_daily_hours": (0.77, "0.67", False),
        "electricity_annual_kwh": (1.43, "0.16", False),
    },
    "Age_dwel x Area_dwel": {
        "heating_daily_hours": (0.18, "0.06", False),
        "electricity_annual_kwh": (0.38, "0.54", False),
    },
}

PAPER_GROUP_STATS = {
    "heating_daily_hours": [
        (4.67, 3.54),
        (5.25, 3.77),
        (6.91, 4.43),
        (7.53, 4.41),
        (8.53, 4.00),
        (9.14, 4.17),
        (9.67, 4.28),
    ],
    "electricity_annual_kwh": [
        (1801.0, 1502.0),
        (2023.0, 1631.0),
        (2320.0, 1502.0),
        (2646.0, 1844.0),
        (2895.0, 1853.0),
    ],
}


@dataclass(frozen=True)
class EnergyConfig:
    dwelling_seed: int = 42
    alpha: float = 0.05
    expected_heating_groups: int = 7
    # The article text says six electricity groups; Table 6 lists five.
    accepted_electricity_groups: tuple[int, ...] = (5, 6)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def require_unique_ids(frame: pd.DataFrame, label: str) -> None:
    if frame["id"].isna().any():
        raise ValueError(f"{label} contains missing household IDs")
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"{label} contains duplicate household IDs: {duplicates}")


def build_energy_dataset(
    raw_path: Path,
    archetypes_path: Path,
    energy_path: Path,
    dwelling_seed: int,
    raw_source_to_canonical: dict[str, str] | None = None,
    energy_source_to_canonical: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    raw = read_mapped_csv(
        raw_path,
        RAW_COLUMNS,
        raw_source_to_canonical,
        "Raw ISTAT energy-analysis input",
        sep=";",
    )
    archetypes = pd.read_csv(archetypes_path, usecols=["id", "archetype"])
    energy = read_mapped_csv(
        energy_path,
        ENERGY_COLUMNS,
        energy_source_to_canonical,
        "GRINS energy results",
    )
    require_columns(raw, RAW_COLUMNS, "Raw ISTAT data")
    require_columns(archetypes, ["id", "archetype"], "Archetype assignments")
    require_columns(energy, ENERGY_COLUMNS, "GRINS energy results")
    require_unique_ids(raw, "Raw ISTAT data")
    require_unique_ids(archetypes, "Archetype assignments")
    require_unique_ids(energy, "GRINS energy results")

    labels = set(archetypes["archetype"].astype(int).unique())
    if labels != {1, 2, 3, 4}:
        raise ValueError(f"Expected archetypes [1, 2, 3, 4], found {sorted(labels)}")

    dwelling = build_dwelling_samples(
        raw[["id", "q_2_4_ric", "q_2_7_class"]],
        archetypes,
        dwelling_seed,
    )[["id", "archetype", "building_age_years", "dwelling_area_m2"]]
    joined = dwelling.merge(raw, on="id", how="left", validate="one_to_one")
    joined = joined.merge(energy, on="id", how="left", validate="one_to_one")

    invalid_reason = pd.Series("", index=joined.index, dtype=object)
    invalid_reason.loc[~joined["q_2_1"].isin(DWELLING_TYPES)] = (
        "unsupported_dwelling_type"
    )
    invalid_reason.loc[
        invalid_reason.eq("") & ~joined["rip"].isin(LOCATIONS)
    ] = "unsupported_location"
    invalid_reason.loc[
        invalid_reason.eq("") & ~joined["q_3_17"].isin(INSULATION)
    ] = "missing_or_unsupported_insulation"
    invalid_reason.loc[
        invalid_reason.eq("") & ~np.isfinite(joined["appliances_el_kWh"])
    ] = "missing_or_nonfinite_electricity"
    invalid_reason.loc[
        invalid_reason.eq("") & (joined["appliances_el_kWh"] <= 0)
    ] = "nonpositive_electricity"

    exclusions = joined.loc[invalid_reason.ne(""), ["id"]].copy()
    exclusions["reason"] = invalid_reason.loc[invalid_reason.ne("")].to_numpy()
    data = joined.loc[invalid_reason.eq("")].copy()

    data["archetype"] = data["archetype"].astype(int)
    data["dwelling_type"] = data["q_2_1"].map(DWELLING_TYPES)
    data["location"] = data["rip"].map(LOCATIONS)
    data["insulation"] = data["q_3_17"].map(INSULATION)
    data["insulation_level"] = data["q_3_17"].map(INSULATION_LEVEL)
    heating_columns = ["q_3_11A", "q_3_11B", "q_3_11C"]
    cooling_columns = ["q_5_8A", "q_5_8B", "q_5_8C"]
    data["heating_daily_hours"] = data[heating_columns].sum(axis=1)
    data["cooling_daily_hours"] = data[cooling_columns].sum(axis=1)
    data["cooling_system_use"] = np.where(
        data["cooling_daily_hours"] > 0,
        "present_or_used",
        "absent_or_unused",
    )
    data["electricity_annual_kwh"] = data["appliances_el_kWh"].astype(float)
    output_columns = [
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
    ]
    data = data[output_columns].sort_values("id").reset_index(drop=True)

    if data["id"].duplicated().any():
        raise AssertionError("Energy analysis dataset contains duplicate IDs")
    numeric = data[
        [
            "building_age_years",
            "dwelling_area_m2",
            "cooling_daily_hours",
            "heating_daily_hours",
            "electricity_annual_kwh",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Energy analysis dataset contains non-finite values")

    counts = {
        "raw_households": int(len(raw)),
        "archetyped_households": int(len(archetypes)),
        "analysis_households": int(len(data)),
        "energy_stage_exclusions": int(len(exclusions)),
        "raw_households_without_archetype": int(len(raw) - len(archetypes)),
    }
    return data, exclusions, counts


def anova_formula(response: str) -> str:
    terms = [
        "C(archetype)",
        "C(dwelling_type)",
        "C(location)",
        "C(insulation)",
        "building_age_years",
        "dwelling_area_m2",
        "C(cooling_system_use)",
        "C(archetype):C(dwelling_type)",
        "C(archetype):C(location)",
        "C(archetype):C(insulation)",
        "C(insulation):C(location)",
        "C(dwelling_type):C(location)",
        "C(dwelling_type):C(insulation)",
        "building_age_years:dwelling_area_m2",
    ]
    return f"{response} ~ " + " + ".join(terms)


MODEL_TERM_NAMES = {
    "C(archetype)": "Usercluster",
    "C(dwelling_type)": "Dwellingtype",
    "C(location)": "Location",
    "C(insulation)": "Insulationlevel",
    "building_age_years": "Age_dwel",
    "dwelling_area_m2": "Area_dwel",
    "C(cooling_system_use)": "CoolingSystemUse",
    "C(archetype):C(dwelling_type)": "Usercluster x Dwellingtype",
    "C(archetype):C(location)": "Usercluster x Location",
    "C(archetype):C(insulation)": "Usercluster x InsulationLevel",
    "C(insulation):C(location)": "InsulationLevel x Location",
    "C(dwelling_type):C(location)": "Dwellingtype x Location",
    "C(dwelling_type):C(insulation)": "Dwellingtype x InsulationLevel",
    "building_age_years:dwelling_area_m2": "Age_dwel x Area_dwel",
}


def run_type2_anova(data: pd.DataFrame, response: str, alpha: float) -> pd.DataFrame:
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "The energy stage requires statsmodels. Install requirements-workflow.txt."
        ) from exc

    formula = anova_formula(response)
    model = ols(formula, data=data).fit()
    table = anova_lm(model, typ=2).reset_index().rename(
        columns={
            "index": "model_term",
            "sum_sq": "sum_squares",
            "df": "degrees_of_freedom",
            "F": "f_statistic",
            "PR(>F)": "p_value",
        }
    )
    table = table[table["model_term"].isin(MODEL_TERM_NAMES)].copy()
    table["term"] = table["model_term"].map(MODEL_TERM_NAMES)
    table["response"] = response
    table["significant_at_alpha"] = table["p_value"] < alpha
    table["paper_f_statistic"] = table["term"].map(
        lambda term: PAPER_ANOVA[term][response][0]
    )
    table["paper_p_label"] = table["term"].map(
        lambda term: PAPER_ANOVA[term][response][1]
    )
    table["paper_significant_at_0_05"] = table["term"].map(
        lambda term: PAPER_ANOVA[term][response][2]
    )
    table["paper_significance_match"] = (
        table["significant_at_alpha"] == table["paper_significant_at_0_05"]
    )
    order = {name: index for index, name in enumerate(MODEL_TERM_NAMES.values())}
    table["term_order"] = table["term"].map(order)
    table = table.sort_values("term_order").drop(columns="term_order")
    return table[
        [
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
    ].reset_index(drop=True)


def _tukey_pairwise_pvalues(
    values: np.ndarray,
    groups: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    frame = pd.DataFrame({"value": values, "group": groups})
    stats = frame.groupby("group", sort=True)["value"].agg(
        count="size", total="sum", mean="mean"
    )
    squared = frame.assign(square=frame["value"] ** 2).groupby(
        "group", sort=True
    )["square"].sum()
    stats["sse"] = squared - stats["total"] ** 2 / stats["count"]
    labels = stats.index.astype(str).tolist()
    counts = stats["count"].to_numpy(dtype=float)
    means = stats["mean"].to_numpy(dtype=float)
    k = len(labels)
    error_df = int(len(frame) - k)
    if k < 2 or error_df <= 0:
        return (
            labels,
            np.array([], int),
            np.array([], int),
            np.array([], float),
            means,
            error_df,
        )
    mse = float(stats["sse"].sum() / error_df)
    left, right = np.triu_indices(k, k=1)
    standard_error = np.sqrt(
        (mse / 2.0) * (1.0 / counts[left] + 1.0 / counts[right])
    )
    difference = np.abs(means[left] - means[right])
    q_statistic = np.divide(
        difference,
        standard_error,
        out=np.full_like(difference, np.inf),
        where=standard_error > 0,
    )
    # The adjusted p-value is monotone in q.  Returning q avoids thousands of
    # expensive distribution integrations per reduction step; only the least
    # separated pair needs a p-value.
    return labels, left, right, q_statistic, means, error_df


def iterative_tukey_groups(
    data: pd.DataFrame,
    response: str,
    factor_columns: list[str],
    alpha: float,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = data[[response, *factor_columns]].copy()
    original = working[factor_columns].astype(str).agg(" | ".join, axis=1)
    current = original.copy()
    merge_rows: list[dict[str, Any]] = []

    try:
        from scipy.integrate import IntegrationWarning
        from scipy.stats import studentized_range
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "The energy stage requires scipy. Install requirements-workflow.txt."
        ) from exc

    for iteration in range(1, int(original.nunique()) + 1):
        labels, left, right, q_statistics, means, error_df = _tukey_pairwise_pvalues(
            working[response].to_numpy(dtype=float), current.to_numpy(dtype=str)
        )
        if len(q_statistics) == 0:
            terminal_adjusted_p_value = 0.0
            break
        minimum = float(np.min(q_statistics))
        tied = np.flatnonzero(
            np.isclose(q_statistics, minimum, rtol=0.0, atol=1e-15)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            adjusted_p_value = float(
                np.clip(
                    studentized_range.sf(minimum, len(labels), error_df),
                    0.0,
                    1.0,
                )
            )
        if adjusted_p_value < alpha:
            terminal_adjusted_p_value = adjusted_p_value
            break
        choices = sorted(
            (
                labels[int(left[index])],
                labels[int(right[index])],
                int(index),
            )
            for index in tied
        )
        group_left, group_right, pair_index = choices[0]
        merged_label = f"merge_{iteration:03d}"
        merge_rows.append(
            {
                "response": response,
                "iteration": iteration,
                "groups_before": len(labels),
                "left_group": group_left,
                "right_group": group_right,
                "adjusted_p_value": adjusted_p_value,
                "left_mean": float(means[int(left[pair_index])]),
                "right_mean": float(means[int(right[pair_index])]),
                "merged_group": merged_label,
            }
        )
        current = current.replace([group_left, group_right], merged_label)
    else:  # pragma: no cover - defensive guard
        raise RuntimeError("Tukey reduction exceeded its deterministic iteration limit")

    assigned = pd.DataFrame(
        {"original_combination": original, "internal_group": current}
    ).drop_duplicates()
    row_groups = pd.DataFrame({response: working[response], "internal_group": current})
    final_stats = row_groups.groupby("internal_group", sort=True)[response].agg(
        sample_count="size", mean="mean", standard_deviation="std"
    )
    ordered_internal = final_stats.sort_values(["mean", "internal_group"]).index.tolist()
    labels = {group: f"{prefix}{index + 1}" for index, group in enumerate(ordered_internal)}
    assigned["group"] = assigned["internal_group"].map(labels)
    components = assigned["original_combination"].str.split(r" \| ", expand=True)
    components.columns = factor_columns
    mapping = pd.concat(
        [assigned[["group", "original_combination"]], components], axis=1
    ).sort_values(["group", *factor_columns])

    final_stats = final_stats.reset_index()
    final_stats["group"] = final_stats["internal_group"].map(labels)
    final_stats["response"] = response
    final_stats["terminal_max_pairwise_adjusted_p_value"] = (
        terminal_adjusted_p_value
    )
    reference = PAPER_GROUP_STATS[response]
    paper_means = {f"{prefix}{i + 1}": item[0] for i, item in enumerate(reference)}
    paper_sds = {f"{prefix}{i + 1}": item[1] for i, item in enumerate(reference)}
    final_stats["paper_mean"] = final_stats["group"].map(paper_means)
    final_stats["paper_standard_deviation"] = final_stats["group"].map(paper_sds)
    final_stats["mean_delta_from_paper"] = final_stats["mean"] - final_stats["paper_mean"]
    final_stats["sd_delta_from_paper"] = (
        final_stats["standard_deviation"] - final_stats["paper_standard_deviation"]
    )
    final_stats = final_stats[
        [
            "response",
            "group",
            "sample_count",
            "mean",
            "standard_deviation",
            "terminal_max_pairwise_adjusted_p_value",
            "paper_mean",
            "paper_standard_deviation",
            "mean_delta_from_paper",
            "sd_delta_from_paper",
        ]
    ].sort_values("group")
    history = pd.DataFrame(merge_rows)
    return mapping.reset_index(drop=True), final_stats.reset_index(drop=True), history


def validate_energy_outputs(
    data: pd.DataFrame,
    anova: pd.DataFrame,
    group_summary: pd.DataFrame,
    config: EnergyConfig,
) -> dict[str, Any]:
    heating_groups = int(
        group_summary.loc[
            group_summary["response"].eq("heating_daily_hours"), "group"
        ].nunique()
    )
    electricity_groups = int(
        group_summary.loc[
            group_summary["response"].eq("electricity_annual_kwh"), "group"
        ].nunique()
    )
    anova_numeric = anova[
        ["degrees_of_freedom", "sum_squares", "f_statistic", "p_value"]
    ].to_numpy(dtype=float)
    checks = {
        "analysis_dataset_is_nonempty": not data.empty,
        "analysis_household_ids_are_unique": not data["id"].duplicated().any(),
        "all_four_archetypes_are_present": set(data["archetype"].unique())
        == {1, 2, 3, 4},
        "responses_are_finite": bool(
            np.isfinite(
                data[["heating_daily_hours", "electricity_annual_kwh"]].to_numpy(
                    dtype=float
                )
            ).all()
        ),
        "all_28_anova_rows_are_present": len(anova) == 28,
        "anova_statistics_are_finite": bool(np.isfinite(anova_numeric).all()),
        "anova_p_values_are_bounded": bool(anova["p_value"].between(0.0, 1.0).all()),
        "heating_tukey_reduction_is_nonempty": heating_groups > 0,
        "electricity_tukey_reduction_is_nonempty": electricity_groups > 0,
        "all_remaining_tukey_pairs_are_significant": bool(
            (
                group_summary["terminal_max_pairwise_adjusted_p_value"]
                < config.alpha
            ).all()
        ),
    }
    paper_matches = int(anova["paper_significance_match"].sum())
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "paper_alignment": {
            "anova_significance_matches": paper_matches,
            "anova_terms_compared": int(len(anova)),
            "all_anova_significance_decisions_match": paper_matches == len(anova),
            "heating_groups": heating_groups,
            "paper_heating_groups": config.expected_heating_groups,
            "heating_group_count_matches_paper": heating_groups
            == config.expected_heating_groups,
            "electricity_groups": electricity_groups,
            "paper_text_electricity_groups": 6,
            "paper_table_6_electricity_groups_listed": 5,
            "electricity_group_count_matches_a_published_count": electricity_groups
            in config.accepted_electricity_groups,
        },
    }


def significant_factor_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the separate categorical effects retained by the paper."""
    requests = [
        ("heating_daily_hours", "insulation_level"),
        ("electricity_annual_kwh", "archetype"),
        ("electricity_annual_kwh", "cooling_system_use"),
    ]
    rows: list[pd.DataFrame] = []
    for response, factor in requests:
        summary = data.groupby(factor, sort=True)[response].agg(
            sample_count="size", mean="mean", standard_deviation="std"
        ).reset_index()
        summary = summary.rename(columns={factor: "level"})
        summary.insert(0, "factor", factor)
        summary.insert(0, "response", response)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_pipeline(
    raw_path: Path,
    archetypes_path: Path,
    energy_path: Path,
    output_dir: Path,
    config: EnergyConfig,
    raw_source_to_canonical: dict[str, str] | None = None,
    energy_source_to_canonical: dict[str, str] | None = None,
) -> dict[str, Any]:
    data, exclusions, row_counts = build_energy_dataset(
        raw_path,
        archetypes_path,
        energy_path,
        config.dwelling_seed,
        raw_source_to_canonical,
        energy_source_to_canonical,
    )
    heating_anova = run_type2_anova(data, "heating_daily_hours", config.alpha)
    electricity_anova = run_type2_anova(
        data, "electricity_annual_kwh", config.alpha
    )
    anova = pd.concat([heating_anova, electricity_anova], ignore_index=True)

    heating_mapping, heating_summary, heating_history = iterative_tukey_groups(
        data,
        "heating_daily_hours",
        ["location", "archetype", "dwelling_type"],
        config.alpha,
        "H",
    )
    electricity_mapping, electricity_summary, electricity_history = (
        iterative_tukey_groups(
            data,
            "electricity_annual_kwh",
            ["location", "dwelling_type"],
            config.alpha,
            "E",
        )
    )
    group_summary = pd.concat(
        [heating_summary, electricity_summary], ignore_index=True
    )
    factor_summary = significant_factor_summary(data)
    merge_history = pd.concat(
        [heating_history, electricity_history], ignore_index=True
    )

    heating_lookup = heating_mapping.set_index("original_combination")["group"]
    electricity_lookup = electricity_mapping.set_index("original_combination")["group"]
    heating_combinations = data[["location", "archetype", "dwelling_type"]].astype(
        str
    ).agg(" | ".join, axis=1)
    electricity_combinations = data[["location", "dwelling_type"]].astype(str).agg(
        " | ".join, axis=1
    )
    data["heating_tukey_group"] = heating_combinations.map(heating_lookup)
    data["electricity_tukey_group"] = electricity_combinations.map(
        electricity_lookup
    )
    if data[["heating_tukey_group", "electricity_tukey_group"]].isna().any().any():
        raise AssertionError("Some households were not assigned to final Tukey groups")
    validation = validate_energy_outputs(data, anova, group_summary, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis_dataset": output_dir / "energy_analysis_dataset.csv",
        "exclusions": output_dir / "energy_exclusions.csv",
        "anova": output_dir / "anova_results.csv",
        "heating_groups": output_dir / "heating_tukey_groups.csv",
        "electricity_groups": output_dir / "electricity_tukey_groups.csv",
        "group_summary": output_dir / "energy_group_summary.csv",
        "factor_summary": output_dir / "energy_factor_summary.csv",
        "merge_history": output_dir / "tukey_merge_history.csv",
        "model": output_dir / "energy_model.json",
        "validation": output_dir / "validation_report.json",
    }
    data.to_csv(paths["analysis_dataset"], index=False, float_format="%.12g")
    exclusions.to_csv(paths["exclusions"], index=False)
    anova.to_csv(paths["anova"], index=False, float_format="%.12g")
    heating_mapping.to_csv(paths["heating_groups"], index=False)
    electricity_mapping.to_csv(paths["electricity_groups"], index=False)
    group_summary.to_csv(paths["group_summary"], index=False, float_format="%.12g")
    factor_summary.to_csv(paths["factor_summary"], index=False, float_format="%.12g")
    merge_history.to_csv(paths["merge_history"], index=False, float_format="%.12g")

    model = {
        "method": "Type-II ANOVA followed by iterative Tukey-Kramer HSD reduction",
        "responses": {
            "heating_daily_hours": "q_3_11A + q_3_11B + q_3_11C",
            "electricity_annual_kwh": "results_grins.appliances_el_kWh",
        },
        "formula": anova_formula("<response>"),
        "alpha": config.alpha,
        "dwelling_seed": config.dwelling_seed,
        "tukey_factors": {
            "heating_daily_hours": ["location", "archetype", "dwelling_type"],
            "electricity_annual_kwh": ["location", "dwelling_type"],
        },
        "separate_paper_factors": {
            "heating": {"insulation_level": sorted(set(INSULATION_LEVEL.values()))},
            "electricity": {
                "archetypes": [1, 2, 3, 4],
                "cooling_system_use": ["absent_or_unused", "present_or_used"],
            },
        },
        "published_group_count_note": (
            "The article text reports six electricity groups, while Table 6 lists "
            "only E1-E5; validation accepts either published count."
        ),
    }
    paths["model"].write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    report = {
        "passed": validation["passed"],
        "source": {
            "title": "Identification of Italian user archetypes for building energy simulations based on a national survey",
            "doi": "10.1016/j.enbuild.2025.115924",
            "raw_input": str(raw_path.resolve()),
            "raw_input_sha256": sha256_file(raw_path),
            "archetypes_input": str(archetypes_path.resolve()),
            "archetypes_input_sha256": sha256_file(archetypes_path),
            "energy_input": str(energy_path.resolve()),
            "energy_input_sha256": sha256_file(energy_path),
        },
        "configuration": {
            "dwelling_seed": config.dwelling_seed,
            "alpha": config.alpha,
            "expected_heating_groups": config.expected_heating_groups,
            "accepted_electricity_groups": list(config.accepted_electricity_groups),
            "raw_source_to_canonical": raw_source_to_canonical or {},
            "energy_source_to_canonical": energy_source_to_canonical or {},
        },
        "row_counts": row_counts,
        "validation": validation,
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["validation"].write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input", type=Path, default=script_dir / "istat_microdata_2013.csv"
    )
    parser.add_argument(
        "--archetypes-input",
        type=Path,
        default=script_dir / "baseline_outputs" / "household_archetypes.csv",
    )
    parser.add_argument(
        "--energy-input", type=Path, default=script_dir / "results_grins.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "energy_outputs"
    )
    parser.add_argument("--dwelling-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_pipeline(
        args.raw_input,
        args.archetypes_input,
        args.energy_input,
        args.output_dir,
        EnergyConfig(dwelling_seed=args.dwelling_seed, alpha=args.alpha),
    )
    validation = report["validation"]["paper_alignment"]
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"Energy ANOVA/Tukey: {status} | {report['row_counts']['analysis_households']} "
        f"households | H={validation['heating_groups']} groups | "
        f"E={validation['electricity_groups']} groups"
    )
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
