"""Deterministic reproduction of the four ISTAT household archetypes.

The pipeline reconstructs household demographic features directly from
``istat_microdata_2013.csv``, applies the two published PCA transformations,
min-max scales the clustering features, and assigns four k-means archetypes.

Only NumPy and pandas are required.  The k-means++ implementation is local so
the result does not depend on changes to scikit-learn defaults across versions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from column_mapping import read_mapped_csv


N_ARCHETYPES = 4
AGE_COLUMNS = [f"cletad_{i}" for i in range(1, 13)]
EDUCATION_COLUMNS = [f"titstu_{i}" for i in range(1, 13)]
MEMBER_COUNT_COLUMN = "q_1_1_sq1"
ARCHETYPE_INPUT_COLUMNS = [
    "id",
    MEMBER_COUNT_COLUMN,
    *AGE_COLUMNS,
    *EDUCATION_COLUMNS,
]

# Effective mapping used by the analysis that produced the published table.
# The legacy mutable-array code first changed category 3 to 5 and subsequently
# changed all 5 values to 12, so category 3 effectively corresponds to 12.
EDUCATION_YEARS = {
    0: 0.0,
    1: 0.0,
    3: 12.0,
    4: 8.0,
    5: 12.0,
    7: 16.0,
    9: 18.0,
    10: 21.0,
}
UNKNOWN_EDUCATION_CODE = 99

# The 2021 survey publishes finer age classes and a consolidated education
# classification. These ranges/years come from the ISTAT 2021 classifications.
AGE_RANGES_2021 = {
    1: (0, 5),
    2: (6, 10),
    3: (11, 14),
    4: (15, 19),
    5: (20, 29),
    6: (30, 39),
    7: (40, 49),
    8: (50, 59),
    9: (60, 69),
    10: (70, 79),
    11: (80, 100),
}
EDUCATION_YEARS_2021 = {
    # Code 1 combines no qualification and primary school. Five years is the
    # upper bound of that published aggregate and is recorded in the model
    # metadata so the assumption remains visible.
    1: 5.0,
    3: 8.0,
    4: 12.0,
    6: 18.0,
}

PAPER_TARGETS = {
    1: {
        "agemin_mean": 70.2,
        "agemin_sd": 11.2,
        "agemax_mean": 78.3,
        "agemax_sd": 5.0,
        "nmembers_mean": 1.8,
        "nmembers_sd": 0.7,
        "education_mean": 0.5,
        "education_sd": 0.1,
        "household_share_pct": 40.1,
        "people_share_pct": 29.9,
    },
    2: {
        "agemin_mean": 21.2,
        "agemin_sd": 7.6,
        "agemax_mean": 57.2,
        "agemax_sd": 8.6,
        "nmembers_mean": 3.7,
        "nmembers_sd": 0.9,
        "education_mean": 0.7,
        "education_sd": 0.1,
        "household_share_pct": 26.9,
        "people_share_pct": 40.2,
    },
    3: {
        "agemin_mean": 46.0,
        "agemin_sd": 12.8,
        "agemax_mean": 63.0,
        "agemax_sd": 8.0,
        "nmembers_mean": 2.3,
        "nmembers_sd": 0.8,
        "education_mean": 0.5,
        "education_sd": 0.1,
        "household_share_pct": 22.9,
        "people_share_pct": 21.3,
    },
    4: {
        "agemin_mean": 55.6,
        "agemin_sd": 17.1,
        "agemax_mean": 72.2,
        "agemax_sd": 9.0,
        "nmembers_mean": 2.1,
        "nmembers_sd": 0.8,
        "education_mean": 0.8,
        "education_sd": 0.1,
        "household_share_pct": 10.1,
        "people_share_pct": 8.6,
    },
}

# The paper reports one-decimal values after randomizing within age classes.
# Tolerances account for that rounding and for using a fixed random realization.
VALIDATION_TOLERANCES = {
    "agemin_mean": 0.75,
    "agemin_sd": 1.00,
    "agemax_mean": 0.75,
    "agemax_sd": 1.00,
    "nmembers_mean": 0.15,
    "nmembers_sd": 0.15,
    "education_mean": 0.10,
    "education_sd": 0.05,
    "household_share_pct": 0.50,
    "people_share_pct": 0.50,
}


@dataclass(frozen=True)
class PipelineConfig:
    age_seed: int = 42
    cluster_seed: int = 1
    max_iterations: int = 300
    convergence_tolerance: float = 1e-8
    survey_year: int = 2013
    available_member_columns: int = 12
    age_ranges: dict[int, tuple[int, int]] | None = None
    education_years: dict[int, float] = field(
        default_factory=lambda: dict(EDUCATION_YEARS)
    )
    unknown_education_codes: tuple[int, ...] = (UNKNOWN_EDUCATION_CODE,)
    education_age_basis: str = "class_code"
    require_paper_tolerances: bool = True


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_possible_study_years(age_class: int) -> float:
    """Return the effective potential-study-years value from the legacy model."""
    if age_class < 10:
        return 0.0
    if age_class < 15:
        return 5.0
    if age_class < 19:
        return 8.0
    if age_class < 23:
        return 12.0
    if age_class < 25:
        return 16.0
    if age_class < 29:
        return 18.0
    return 21.0


def randomize_age_classes(
    age_classes: np.ndarray,
    rng: np.random.RandomState,
    age_ranges: dict[int, tuple[int, int]] | None = None,
) -> np.ndarray:
    """Convert survey age classes to ages using a deterministic legacy-style draw."""
    if age_ranges is not None:
        ages = np.empty(len(age_classes), dtype=float)
        for index, raw_class in enumerate(age_classes):
            age_class = int(raw_class)
            if age_class not in age_ranges:
                raise ValueError(f"Unsupported age-class code: {age_class}")
            lower, upper = age_ranges[age_class]
            ages[index] = rng.randint(lower, upper + 1)
        return ages

    ages = age_classes.astype(float, copy=True)

    youngest = ages == 0
    ages[youngest] = rng.randint(11, 15, size=int(youngest.sum()))

    for age_class in range(15, 106, 10):
        selected = ages == age_class
        ages[selected] = rng.randint(
            age_class + 5,
            age_class + 10,
            size=int(selected.sum()),
        )

    valid_codes = (age_classes == 0) | (
        (age_classes >= 15) & ((age_classes - 15) % 10 == 0)
    )
    if not np.all(valid_codes):
        invalid = sorted(np.unique(age_classes[~valid_codes]).astype(int).tolist())
        raise ValueError(f"Unsupported age-class codes: {invalid}")
    return ages


def education_index(
    age_values: np.ndarray,
    education_codes: np.ndarray,
    education_years: dict[int, float] | None = None,
    unknown_codes: tuple[int, ...] = (UNKNOWN_EDUCATION_CODE,),
) -> float:
    """Compute the household-average education index used in the publication."""
    years_by_code = EDUCATION_YEARS if education_years is None else education_years
    ratios: list[float] = []
    for age_value, education_code in zip(age_values, education_codes, strict=True):
        education_code = int(education_code)
        if education_code in unknown_codes:
            continue
        if education_code not in years_by_code:
            raise ValueError(f"Unsupported education code: {education_code}")

        years_studied = years_by_code[education_code]
        potential_years = maximum_possible_study_years(int(age_value))
        if potential_years == 0:
            # This preserves the effective legacy behavior: positive/zero became
            # infinity/NaN, after which infinity was capped to one.
            ratio = math.nan if years_studied == 0 else 1.0
        else:
            ratio = min(years_studied / potential_years, 1.0)
        ratios.append(ratio)

    if not ratios:
        return math.nan
    return float(np.mean(ratios))


def validate_raw_schema(raw: pd.DataFrame) -> None:
    required = {"id", MEMBER_COUNT_COLUMN, *AGE_COLUMNS, *EDUCATION_COLUMNS}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw ISTAT file is missing required columns: {missing}")
    if raw["id"].duplicated().any():
        raise ValueError("Raw ISTAT household IDs must be unique")


def build_household_features(
    raw: pd.DataFrame,
    age_seed: int,
    available_member_columns: int = 12,
    age_ranges: dict[int, tuple[int, int]] | None = None,
    education_years: dict[int, float] | None = None,
    unknown_education_codes: tuple[int, ...] = (UNKNOWN_EDUCATION_CODE,),
    education_age_basis: str = "class_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build agemin, agemax, nmembers, and education from the raw survey."""
    validate_raw_schema(raw)
    rng = np.random.RandomState(age_seed)
    ids = raw["id"].to_numpy()
    member_counts = raw[MEMBER_COUNT_COLUMN].to_numpy(dtype=int)
    age_matrix = raw[AGE_COLUMNS].to_numpy(dtype=int)
    education_matrix = raw[EDUCATION_COLUMNS].to_numpy(dtype=int)

    records: list[dict[str, float | int]] = []
    exclusions: list[dict[str, int | str]] = []

    for position, household_id in enumerate(ids):
        nmembers = int(member_counts[position])
        if nmembers <= 0:
            exclusions.append({"id": int(household_id), "reason": "zero_members"})
            continue
        if nmembers > available_member_columns:
            exclusions.append(
                {
                    "id": int(household_id),
                    "reason": "member_count_exceeds_available_columns",
                }
            )
            continue

        age_classes = age_matrix[position, :nmembers]
        ages = randomize_age_classes(age_classes, rng, age_ranges)
        if education_age_basis == "class_code":
            education_age_values = age_classes
        elif education_age_basis == "realized_age":
            education_age_values = ages
        else:
            raise ValueError(
                "education_age_basis must be 'class_code' or 'realized_age'"
            )
        education = education_index(
            education_age_values,
            education_matrix[position, :nmembers],
            education_years,
            unknown_education_codes,
        )
        if not np.isfinite(education):
            exclusions.append(
                {"id": int(household_id), "reason": "undefined_education_index"}
            )
            continue

        records.append(
            {
                "id": int(household_id),
                "agemin": float(ages.min()),
                "agemax": float(ages.max()),
                "nmembers": nmembers,
                "education": education,
            }
        )

    features = pd.DataFrame.from_records(records)
    excluded = pd.DataFrame.from_records(exclusions, columns=["id", "reason"])
    if features.empty:
        raise ValueError("Preprocessing produced no eligible households")
    return features, excluded


def add_published_components(features: pd.DataFrame) -> pd.DataFrame:
    """Apply equations (1) and (2) from the paper and min-max normalize them."""
    result = features.copy()
    result["pc1"] = (
        -0.25 * result["agemin"]
        - 0.75 * result["agemax"]
        + 10.0 * result["nmembers"]
    )
    result["pc2"] = (
        (1.0 / 3.0) * result["agemin"]
        + (2.0 / 3.0) * result["agemax"]
        + 10.0 * result["nmembers"]
    )

    for column in ("pc1", "pc2", "education"):
        minimum = float(result[column].min())
        maximum = float(result[column].max())
        if maximum <= minimum:
            raise ValueError(f"Cannot min-max scale constant column {column}")
        result[f"{column}_scaled"] = (result[column] - minimum) / (maximum - minimum)
    return result


def pca_explained_variance(features: pd.DataFrame) -> list[float]:
    """Calculate exploratory PCA variance ratios for the three raw demographics."""
    matrix = features[["agemin", "agemax", "nmembers"]].to_numpy(dtype=float)
    spans = np.ptp(matrix, axis=0)
    if np.any(spans == 0):
        raise ValueError("Cannot calculate PCA with a constant demographic feature")
    matrix = (matrix - matrix.min(axis=0)) / spans
    matrix -= matrix.mean(axis=0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    variances = singular_values**2
    return (variances / variances.sum()).tolist()


def kmeans_plus_plus_initial_centers(
    matrix: np.ndarray,
    n_clusters: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Initialize centers with the greedy k-means++ procedure used by sklearn."""
    n_samples, n_features = matrix.shape
    centers = np.empty((n_clusters, n_features), dtype=float)
    uniform_weights = np.full(n_samples, 1.0 / n_samples)
    centers[0] = matrix[rng.choice(n_samples, p=uniform_weights)]

    closest_distances = np.sum((matrix - centers[0]) ** 2, axis=1)
    current_potential = float(closest_distances.sum())
    local_trials = 2 + int(math.log(n_clusters))

    for center_index in range(1, n_clusters):
        candidates = rng.choice(
            n_samples,
            size=local_trials,
            replace=True,
            p=closest_distances / current_potential,
        )
        candidate_distances = np.array(
            [
                np.minimum(
                    closest_distances,
                    np.sum((matrix - matrix[candidate]) ** 2, axis=1),
                )
                for candidate in candidates
            ]
        )
        potentials = candidate_distances.sum(axis=1)
        best = int(np.argmin(potentials))
        centers[center_index] = matrix[candidates[best]]
        closest_distances = candidate_distances[best]
        current_potential = float(potentials[best])
    return centers


def fit_kmeans(
    matrix: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iterations: int,
    tolerance: float,
) -> KMeansResult:
    """Fit deterministic Lloyd k-means from one deterministic k-means++ start."""
    if len(matrix) < n_clusters:
        raise ValueError("The number of observations is smaller than n_clusters")
    rng = np.random.RandomState(seed)
    centers = kmeans_plus_plus_initial_centers(matrix, n_clusters, rng)

    for iteration in range(1, max_iterations + 1):
        distances = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        if len(np.unique(labels)) != n_clusters:
            raise RuntimeError("K-means produced an empty cluster")
        new_centers = np.array(
            [matrix[labels == cluster].mean(axis=0) for cluster in range(n_clusters)]
        )
        center_shift = float(np.sum((new_centers - centers) ** 2))
        centers = new_centers
        if center_shift <= tolerance:
            break
    else:
        raise RuntimeError(f"K-means did not converge in {max_iterations} iterations")

    distances = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    inertia = float(distances[np.arange(len(matrix)), labels].sum())
    return KMeansResult(labels, centers, inertia, iteration)


def canonical_archetype_mapping(features: pd.DataFrame, labels: np.ndarray) -> dict[int, int]:
    """Give arbitrary k-means labels stable semantic archetype numbers."""
    labeled = features.assign(_label=labels)
    means = labeled.groupby("_label")[["agemin", "nmembers", "education"]].mean()
    remaining = set(int(label) for label in means.index)

    cluster_2 = int(means.loc[list(remaining), "nmembers"].idxmax())
    remaining.remove(cluster_2)
    cluster_4 = int(means.loc[list(remaining), "education"].idxmax())
    remaining.remove(cluster_4)
    cluster_1 = int(means.loc[list(remaining), "agemin"].idxmax())
    remaining.remove(cluster_1)
    cluster_3 = remaining.pop()

    return {cluster_1: 1, cluster_2: 2, cluster_3: 3, cluster_4: 4}


def calinski_harabasz_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    n_samples = len(matrix)
    overall = matrix.mean(axis=0)
    within = 0.0
    between = 0.0
    for label in unique:
        group = matrix[labels == label]
        center = group.mean(axis=0)
        within += float(np.sum((group - center) ** 2))
        between += len(group) * float(np.sum((center - overall) ** 2))
    return (between / (len(unique) - 1)) / (within / (n_samples - len(unique)))


def summarize_archetypes(features: pd.DataFrame) -> pd.DataFrame:
    grouped = features.groupby("archetype", sort=True)
    total_people = float(features["nmembers"].sum())
    rows: list[dict[str, float | int]] = []
    for archetype, group in grouped:
        rows.append(
            {
                "archetype": int(archetype),
                "households": int(len(group)),
                "people": int(group["nmembers"].sum()),
                "household_share_pct": 100.0 * len(group) / len(features),
                "people_share_pct": 100.0 * group["nmembers"].sum() / total_people,
                "agemin_mean": group["agemin"].mean(),
                "agemin_sd": group["agemin"].std(ddof=1),
                "agemax_mean": group["agemax"].mean(),
                "agemax_sd": group["agemax"].std(ddof=1),
                "nmembers_mean": group["nmembers"].mean(),
                "nmembers_sd": group["nmembers"].std(ddof=1),
                "education_mean": group["education"].mean(),
                "education_sd": group["education"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows).sort_values("archetype").reset_index(drop=True)


def validate_against_paper(summary: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    by_archetype = summary.set_index("archetype")
    for archetype, targets in PAPER_TARGETS.items():
        if archetype not in by_archetype.index:
            checks.append(
                {
                    "archetype": archetype,
                    "metric": "cluster_exists",
                    "passed": False,
                }
            )
            continue
        for metric, target in targets.items():
            actual = float(by_archetype.loc[archetype, metric])
            tolerance = VALIDATION_TOLERANCES[metric]
            difference = actual - target
            checks.append(
                {
                    "archetype": archetype,
                    "metric": metric,
                    "actual": actual,
                    "paper": target,
                    "difference": difference,
                    "absolute_difference": abs(difference),
                    "tolerance": tolerance,
                    "passed": abs(difference) <= tolerance,
                }
            )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def add_reference_columns(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for metric in VALIDATION_TOLERANCES:
        result[f"paper_{metric}"] = result["archetype"].map(
            {cluster: values[metric] for cluster, values in PAPER_TARGETS.items()}
        )
        result[f"delta_{metric}"] = result[metric] - result[f"paper_{metric}"]
    return result


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    source_to_canonical: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = read_mapped_csv(
        input_path,
        ARCHETYPE_INPUT_COLUMNS,
        source_to_canonical,
        "Raw ISTAT archetype input",
        sep=";",
    )
    features, exclusions = build_household_features(
        raw,
        config.age_seed,
        config.available_member_columns,
        config.age_ranges,
        config.education_years,
        config.unknown_education_codes,
        config.education_age_basis,
    )
    features = add_published_components(features)

    cluster_columns = ["pc1_scaled", "pc2_scaled", "education_scaled"]
    matrix = features[cluster_columns].to_numpy(dtype=float)
    kmeans = fit_kmeans(
        matrix,
        N_ARCHETYPES,
        config.cluster_seed,
        config.max_iterations,
        config.convergence_tolerance,
    )
    mapping = canonical_archetype_mapping(features, kmeans.labels)
    features["archetype"] = pd.Series(kmeans.labels, index=features.index).map(mapping)
    features["archetype"] = features["archetype"].astype(int)

    summary = summarize_archetypes(features)
    validation = validate_against_paper(summary)
    explained_variance = pca_explained_variance(features)
    chi = calinski_harabasz_score(matrix, kmeans.labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    household_path = output_dir / "household_archetypes.csv"
    summary_path = output_dir / "archetype_summary.csv"
    exclusions_path = output_dir / "preprocessing_exclusions.csv"
    model_path = output_dir / "archetype_model.json"
    validation_path = output_dir / "validation_report.json"

    features.to_csv(household_path, index=False, float_format="%.12g")
    add_reference_columns(summary).to_csv(summary_path, index=False, float_format="%.12g")
    exclusions.to_csv(exclusions_path, index=False)

    inverse_mapping = {
        archetype: raw_label for raw_label, archetype in mapping.items()
    }
    canonical_centers = {
        str(archetype): {
            column: float(kmeans.centers[inverse_mapping[archetype], index])
            for index, column in enumerate(cluster_columns)
        }
        for archetype in range(1, N_ARCHETYPES + 1)
    }
    model = {
        "method": "published PCA equations, min-max scaling, k-means++ and Lloyd iteration",
        "pca_equations": {
            "pc1": "-0.25*agemin - 0.75*agemax + 10*nmembers",
            "pc2": "(1/3)*agemin + (2/3)*agemax + 10*nmembers",
        },
        "cluster_features": cluster_columns,
        "age_seed": config.age_seed,
        "cluster_seed": config.cluster_seed,
        "survey_year": config.survey_year,
        "available_member_columns": config.available_member_columns,
        "age_class_ranges_inclusive": config.age_ranges,
        "education_years_by_code": config.education_years,
        "unknown_education_codes": list(config.unknown_education_codes),
        "education_age_basis": config.education_age_basis,
        "iterations": kmeans.iterations,
        "inertia": kmeans.inertia,
        "calinski_harabasz_score": chi,
        "exploratory_pca_explained_variance_ratio": explained_variance,
        "exploratory_pca_first_two_cumulative": sum(explained_variance[:2]),
        "feature_scaling": {
            column: {
                "minimum": float(features[column].min()),
                "maximum": float(features[column].max()),
            }
            for column in ("pc1", "pc2", "education")
        },
        "centers": canonical_centers,
    }
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    report = {
        "passed": validation["passed"] or not config.require_paper_tolerances,
        "source": {
            "title": "Identification of Italian user archetypes for building energy simulations based on a national survey",
            "doi": "10.1016/j.enbuild.2025.115924",
            "input_file": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
        },
        "configuration": {
            "age_seed": config.age_seed,
            "cluster_seed": config.cluster_seed,
            "n_archetypes": N_ARCHETYPES,
            "survey_year": config.survey_year,
            "available_member_columns": config.available_member_columns,
            "age_class_ranges_inclusive": config.age_ranges,
            "education_years_by_code": config.education_years,
            "unknown_education_codes": list(config.unknown_education_codes),
            "education_age_basis": config.education_age_basis,
            "require_paper_tolerances": config.require_paper_tolerances,
            "source_to_canonical": source_to_canonical or {},
        },
        "row_counts": {
            "raw_households": int(len(raw)),
            "eligible_households": int(len(features)),
            "excluded_households": int(len(exclusions)),
            "exclusions_by_reason": {
                str(key): int(value)
                for key, value in exclusions["reason"].value_counts().items()
            },
        },
        "model_metrics": {
            "inertia": kmeans.inertia,
            "calinski_harabasz_score": chi,
            "iterations": kmeans.iterations,
            "exploratory_pca_explained_variance_ratio": explained_variance,
            "exploratory_pca_first_two_cumulative": sum(explained_variance[:2]),
        },
        "paper_validation": validation,
        "outputs": {
            "households": str(household_path.resolve()),
            "summary": str(summary_path.resolve()),
            "exclusions": str(exclusions_path.resolve()),
            "model": str(model_path.resolve()),
            "validation": str(validation_path.resolve()),
        },
    }
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "istat_microdata_2013.csv",
        help="Raw semicolon-delimited ISTAT microdata CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "baseline_outputs",
        help="Directory for the reconstructed baseline outputs",
    )
    parser.add_argument("--age-seed", type=int, default=42)
    parser.add_argument("--cluster-seed", type=int, default=1)
    parser.add_argument(
        "--allow-validation-drift",
        action="store_true",
        help="Return success even when results exceed paper-validation tolerances",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(age_seed=args.age_seed, cluster_seed=args.cluster_seed)
    report = run_pipeline(args.input, args.output_dir, config)
    status = "PASS" if report["passed"] else "FAIL"
    counts = report["row_counts"]
    print(
        f"Four-archetype baseline: {status} | "
        f"{counts['eligible_households']} eligible of {counts['raw_households']} households"
    )
    print(f"Outputs: {args.output_dir.resolve()}")
    if not report["passed"] and not args.allow_validation_drift:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
