"""Generate the workflow's captioned, paper-style Word results report."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


FIGURE_STEMS = [
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

LEGACY_FIGURE_STEMS = [
    "archetype_pca_scatter",
    "heating_tukey_groups",
    "electricity_tukey_groups",
]

CLUSTER_COLORS = {
    1: "#4169E1",
    2: "#50C878",
    3: "#FFBF00",
    4: "#D50032",
}

DWELLING_TYPE_LABELS = {
    "apartment_small_building": "Apartment in Small Building",
    "apartment_medium_building": "Apartment in Medium Building",
    "apartment_large_building": "Apartment in Large Building",
    "single_family_house": "Single Family House",
    "multi_family_house": "Multi Family House",
}

LOCATION_LABELS = {
    "north_east": "North East",
    "north_west": "North West",
    "central": "Central",
    "south_and_islands": "South and Islands",
}


@dataclass(frozen=True)
class ReportConfig:
    title: str = "Italian Household Archetypes and Energy Analysis"
    subtitle: str = "Automated methods and results report"
    cluster_seed: int = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_report_dependencies() -> None:
    try:
        import docx  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Word report generation requires python-docx and matplotlib. "
            "Install requirements-workflow.txt."
        ) from exc


def _save_figure(fig: Any, figure_dir: Path, stem: str) -> dict[str, Path]:
    svg_path = figure_dir / f"{stem}.svg"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "GRINS deterministic workflow"},
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "GRINS deterministic workflow"},
    )
    return {"svg": svg_path, "png": png_path}


def _pca_cumulative(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = frame[list(columns)].to_numpy(dtype=float)
    spans = np.ptp(matrix, axis=0)
    if np.any(spans == 0):
        raise ValueError(f"Cannot calculate PCA for constant columns: {columns}")
    matrix = (matrix - matrix.min(axis=0)) / spans
    matrix -= matrix.mean(axis=0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    variances = singular_values**2
    return np.cumsum(variances / variances.sum()) * 100.0


def _probability_design(age: np.ndarray, area: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    area = np.asarray(area, dtype=float)
    return np.column_stack(
        [
            np.ones(len(age)),
            age / 100.0,
            area / 100.0,
            age**2 / 10_000.0,
            area**2 / 10_000.0,
            age * area / 10_000.0,
        ]
    )


def _fit_probability_models(
    probabilities: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    design = _probability_design(
        probabilities["building_age_years"].to_numpy(float),
        probabilities["dwelling_area_m2"].to_numpy(float),
    )
    coefficients: dict[int, np.ndarray] = {}
    fitted = probabilities[
        ["building_age_years", "dwelling_area_m2", "sample_count"]
    ].copy()
    for archetype in range(1, 5):
        actual = probabilities[f"prob_archetype_{archetype}"].to_numpy(float)
        weights = np.sqrt(probabilities["sample_count"].to_numpy(float))
        weighted_design = design * weights[:, None]
        weighted_actual = actual * weights
        coefficient = np.linalg.lstsq(
            weighted_design, weighted_actual, rcond=None
        )[0]
        coefficients[archetype] = coefficient
        fitted[f"actual_{archetype}"] = actual
        fitted[f"predicted_{archetype}"] = np.clip(
            design @ coefficient, 0.0, 1.0
        )
    return coefficients, fitted


def _group_sort_key(value: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", str(value))
    return (str(value)[:1], int(match.group(1)) if match else 0)


def _best_center_order(
    centers: np.ndarray, global_centers: np.ndarray
) -> np.ndarray:
    best_order: tuple[int, ...] | None = None
    best_distance = math.inf
    for order in itertools.permutations(range(len(centers))):
        distance = float(
            np.sum((centers[np.asarray(order)] - global_centers) ** 2)
        )
        if distance < best_distance:
            best_distance = distance
            best_order = order
    if best_order is None:  # pragma: no cover - four centers always produce orders
        raise RuntimeError("Could not align regional cluster centers")
    return centers[np.asarray(best_order)]


def _draw_process_flow(plt: Any, demographic_only: bool = False) -> Any:
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        face: str = "white",
        fontsize: float = 8.5,
    ) -> None:
        ax.add_patch(
            Rectangle((x, y), width, height, facecolor=face, edgecolor="black", lw=1)
        )
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.1,
        )

    def arrow(
        start: tuple[float, float], end: tuple[float, float], label: str = ""
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="simple,head_length=8,head_width=8,tail_width=1",
                facecolor="white",
                edgecolor="black",
                lw=0.8,
                mutation_scale=1,
            )
        )
        if label:
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.42,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    box(0.15, 0.45, 1.65, 6.9, "")
    ax.text(0.42, 3.9, "Raw Dataset", rotation=90, ha="center", va="center", fontsize=10)
    box(0.65, 5.05, 1.15, 2.3, "Socio-demographic\nvariables", "#F4F4F4", 7.0)
    box(0.65, 2.75, 1.15, 2.3, "Construction-related\nvariables", "#F4F4F4", 7.0)
    energy_label = (
        "Energy-specific\nstage not\ncomparable"
        if demographic_only
        else "Energy-pattern\nvariables"
    )
    box(0.65, 0.45, 1.15, 2.3, energy_label, "#F4F4F4", 7.0)
    box(2.75, 5.75, 1.65, 1.05, "Define\ninformative variables")
    box(5.15, 5.75, 1.75, 1.05, "Reduced\ndimensionality")
    box(7.75, 5.75, 1.75, 1.05, "Clustering\nanalysis")
    box(10.35, 5.75, 2.05, 1.05, "Socio-demographic\nuser types")
    arrow((1.8, 6.28), (2.75, 6.28))
    arrow((4.4, 6.28), (5.15, 6.28))
    ax.text(4.78, 7.12, "Principal Component Analysis", ha="center", va="bottom", fontsize=8.5)
    arrow((6.9, 6.28), (7.75, 6.28))
    arrow((9.5, 6.28), (10.35, 6.28))

    box(4.3, 3.15, 3.0, 1.15, "Assign user-type probabilities\nfrom dwelling information", fontsize=8.2)
    arrow((1.8, 3.72), (4.3, 3.72), "Probabilistic Feature Extraction")
    ax.plot([11.38, 11.38, 7.3], [5.75, 3.72, 3.72], color="black", lw=1)
    arrow((8.1, 3.72), (7.3, 3.72))
    ax.text(9.45, 4.15, "Probabilistic Feature Extraction", ha="center", va="bottom", fontsize=8.5)

    if not demographic_only:
        box(8.2, 0.95, 4.2, 1.25, "Significant dwelling-user groups based on\nheating-system use and electricity consumption", fontsize=8.1)
        ax.plot([1.8, 6.05, 6.05], [1.15, 1.15, 2.6], color="black", lw=1)
        ax.plot([5.8, 5.8, 7.9], [3.15, 1.57, 1.57], color="black", lw=1)
        arrow((7.9, 1.57), (8.2, 1.57))
        ax.text(2.55, 1.38, "Analysis of Variance", fontsize=8.5, va="bottom")
    fig.tight_layout(pad=0.2)
    return fig


def _draw_group_assignment_matrix(
    plt: Any, heating_groups_path: Path, electricity_groups_path: Path
) -> Any:
    from matplotlib.patches import Rectangle

    heating = pd.read_csv(heating_groups_path)
    electricity = pd.read_csv(electricity_groups_path)
    heating_lookup = {
        (row["location"], int(row["archetype"]), row["dwelling_type"]): row["group"]
        for _, row in heating.iterrows()
    }
    electricity_lookup = {
        (row["location"], row["dwelling_type"]): row["group"]
        for _, row in electricity.iterrows()
    }
    locations = ["north_east", "north_west", "central", "south_and_islands"]
    dwelling_types = [
        "apartment_small_building",
        "apartment_medium_building",
        "apartment_large_building",
        "single_family_house",
        "multi_family_house",
    ]
    heat_labels = sorted(set(heating["group"]), key=_group_sort_key)
    electricity_labels = sorted(set(electricity["group"]), key=_group_sort_key)
    heat_colors = {
        label: plt.cm.Oranges(value)
        for label, value in zip(
            heat_labels, np.linspace(0.18, 0.92, len(heat_labels)), strict=True
        )
    }
    electricity_colors = {
        label: plt.cm.Blues(value)
        for label, value in zip(
            electricity_labels,
            np.linspace(0.18, 0.92, len(electricity_labels)),
            strict=True,
        )
    }

    widths = [0.7, 3.25] + [0.88] * 8
    x_edges = np.cumsum([0.0] + widths)
    n_rows = len(locations) * len(dwelling_types)
    fig, ax = plt.subplots(figsize=(11.8, 10.8))
    ax.set_xlim(0, x_edges[-1])
    ax.set_ylim(-0.15, n_rows + 2.15)
    ax.axis("off")

    def cell(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        face: Any = "white",
        rotation: float = 0,
        bold: bool = False,
        fontsize: float = 8.5,
    ) -> None:
        ax.add_patch(
            Rectangle((x, y), width, height, facecolor=face, edgecolor="#A8A8A8", lw=0.55)
        )
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            rotation=rotation,
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
        )

    cell(x_edges[0], n_rows, widths[0], 2, "Location", "#F3F3F3", 90, True)
    cell(x_edges[1], n_rows, widths[1], 2, "Dwelling Type", "#F3F3F3", 0, True)
    cell(x_edges[2], n_rows + 1, sum(widths[2:6]), 1, "Heating Group", "#F3F3F3", 0, True)
    cell(x_edges[6], n_rows + 1, sum(widths[6:10]), 1, "Electricity Group", "#F3F3F3", 0, True)
    for index in range(8):
        cell(
            x_edges[index + 2],
            n_rows,
            widths[index + 2],
            1,
            f"User\nArchetype\n{index % 4 + 1}",
            "#F3F3F3",
            fontsize=7.6,
        )

    row_index = 0
    for location in locations:
        location_y = n_rows - row_index - len(dwelling_types)
        cell(
            x_edges[0],
            location_y,
            widths[0],
            len(dwelling_types),
            LOCATION_LABELS[location],
            "#F5F5F5",
            90,
            True,
            8.2,
        )
        for dwelling_type in dwelling_types:
            y = n_rows - row_index - 1
            cell(
                x_edges[1],
                y,
                widths[1],
                1,
                DWELLING_TYPE_LABELS[dwelling_type],
                "#FAFAFA",
                fontsize=8.1,
            )
            electricity_group = electricity_lookup.get((location, dwelling_type), "-")
            for archetype in range(1, 5):
                heating_group = heating_lookup.get(
                    (location, archetype, dwelling_type), "-"
                )
                cell(
                    x_edges[archetype + 1],
                    y,
                    widths[archetype + 1],
                    1,
                    heating_group,
                    heat_colors.get(heating_group, "white"),
                    fontsize=8.5,
                )
                electricity_x_index = archetype + 5
                cell(
                    x_edges[electricity_x_index],
                    y,
                    widths[electricity_x_index],
                    1,
                    electricity_group,
                    electricity_colors.get(electricity_group, "white"),
                    fontsize=8.5,
                )
            row_index += 1
    fig.tight_layout(pad=0.15)
    return fig


def generate_figures(
    raw_path: Path,
    archetypes_path: Path,
    archetype_summary_path: Path,
    archetype_model_path: Path,
    probabilities_path: Path,
    energy_dataset_path: Path | None,
    heating_groups_path: Path | None,
    electricity_groups_path: Path | None,
    figure_dir: Path,
    cluster_seed: int,
    raw_source_to_canonical: dict[str, str] | None = None,
    demographic_only: bool = False,
) -> dict[str, dict[str, Path]]:
    _require_report_dependencies()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.7,
            "grid.color": "#D5D5D5",
            "grid.linewidth": 0.6,
            "svg.fonttype": "none",
            "svg.hashsalt": "grins-istat-paper-replica",
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from archetype_pipeline import calinski_harabasz_score, fit_kmeans
    from column_mapping import read_mapped_csv

    figure_dir.mkdir(parents=True, exist_ok=True)
    for stem in LEGACY_FIGURE_STEMS:
        for extension in ("svg", "png"):
            stale_path = figure_dir / f"{stem}.{extension}"
            if stale_path.is_file():
                stale_path.unlink()
    if demographic_only:
        for stem in FIGURE_STEMS[7:]:
            for extension in ("svg", "png"):
                stale_path = figure_dir / f"{stem}.{extension}"
                if stale_path.is_file():
                    stale_path.unlink()

    outputs: dict[str, dict[str, Path]] = {}
    households = pd.read_csv(archetypes_path)
    summary = pd.read_csv(archetype_summary_path).sort_values("archetype")
    probabilities = pd.read_csv(probabilities_path)
    energy = None if demographic_only else pd.read_csv(energy_dataset_path)
    archetype_model = json.loads(archetype_model_path.read_text(encoding="utf-8"))

    fig = _draw_process_flow(plt, demographic_only)
    outputs[FIGURE_STEMS[0]] = _save_figure(fig, figure_dir, FIGURE_STEMS[0])
    plt.close(fig)

    all_cumulative = _pca_cumulative(
        households, ["agemin", "agemax", "nmembers", "education"]
    )
    demographic_cumulative = _pca_cumulative(
        households, ["agemin", "agemax", "nmembers"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=True)
    for axis, values, panel in [
        (axes[0], all_cumulative, "(a)"),
        (axes[1], demographic_cumulative, "(b)"),
    ]:
        components = np.arange(1, len(values) + 1)
        axis.plot(components, values, color="black", marker="o", ms=3.5, lw=1.2)
        axis.set_xticks(components)
        axis.set_xlabel("Number of Components")
        axis.set_ylim(60, 100)
        axis.grid(True)
        axis.text(0.5, -0.22, panel, transform=axis.transAxes, ha="center", va="top")
    axes[0].set_ylabel("Cumulative Explained Variance [%]")
    fig.subplots_adjust(wspace=0.04, bottom=0.23)
    outputs[FIGURE_STEMS[1]] = _save_figure(fig, figure_dir, FIGURE_STEMS[1])
    plt.close(fig)

    cluster_matrix = households[
        ["pc1_scaled", "pc2_scaled", "education_scaled"]
    ].to_numpy(float)
    wcss = [
        fit_kmeans(cluster_matrix, k, cluster_seed, 300, 1e-8).inertia
        for k in range(1, 10)
    ]
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(range(1, 10), wcss, color="black", marker="o", ms=4.2, lw=1.3)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Within Cluster Sum of Squares")
    ax.set_xticks(range(1, 10))
    ax.grid(True)
    fig.tight_layout()
    outputs[FIGURE_STEMS[2]] = _save_figure(fig, figure_dir, FIGURE_STEMS[2])
    plt.close(fig)

    regions = read_mapped_csv(
        raw_path,
        ["id", "reg"],
        raw_source_to_canonical,
        "Raw ISTAT regional figure input",
        sep=";",
    )
    regional_households = households.merge(regions, on="id", how="left", validate="one_to_one")
    global_centers = (
        households.groupby("archetype")[["pc1_scaled", "pc2_scaled", "education_scaled"]]
        .mean()
        .sort_index()
        .to_numpy(float)
    )
    regional_results: list[tuple[np.ndarray, float]] = []
    for region_code, region in regional_households.dropna(subset=["reg"]).groupby("reg"):
        matrix = region[["pc1_scaled", "pc2_scaled", "education_scaled"]].to_numpy(float)
        if len(matrix) < 8:
            continue
        fitted = fit_kmeans(matrix, 4, cluster_seed + int(region_code), 300, 1e-8)
        ordered = _best_center_order(fitted.centers, global_centers)
        score = calinski_harabasz_score(matrix, fitted.labels)
        regional_results.append((ordered, score))

    fig, ax = plt.subplots(figsize=(6.0, 4.9))
    jitter_rng = np.random.RandomState(731)
    for centers, score in regional_results:
        x = np.r_[centers[:, 2], centers[0, 2]]
        y = np.r_[1.0 - centers[:, 0], 1.0 - centers[0, 0]]
        for _ in range(5):
            jitter = jitter_rng.normal(0.0, 0.006, size=len(x))
            ax.plot(x + jitter, y + jitter, color="#B8B8B8", alpha=0.055, lw=0.7)
        ax.scatter(
            centers[:, 2],
            1.0 - centers[:, 0],
            s=max(8.0, math.sqrt(score) * 0.45),
            color="#A8A8A8",
            alpha=0.8,
            edgecolors="none",
        )
    global_x = np.r_[global_centers[:, 2], global_centers[0, 2]]
    global_y = np.r_[1.0 - global_centers[:, 0], 1.0 - global_centers[0, 0]]
    ax.plot(global_x, global_y, color="#D50032", lw=1.4, alpha=0.7)
    for archetype, center in enumerate(global_centers, start=1):
        ax.scatter(
            center[2],
            1.0 - center[0],
            s=170,
            color=CLUSTER_COLORS[archetype],
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
    ax.set_xlabel("Normalized Study Index")
    ax.set_ylabel("Normalized PC1")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(
        handles=[
            Patch(color=CLUSTER_COLORS[index], label=f"Cluster {index}")
            for index in (4, 3, 2, 1)
        ],
        loc="lower left",
        frameon=True,
    )
    fig.tight_layout()
    outputs[FIGURE_STEMS[3]] = _save_figure(fig, figure_dir, FIGURE_STEMS[3])
    plt.close(fig)

    figure_values = pd.DataFrame(
        {
            "households": summary["household_share_pct"].to_numpy(float) / 100.0,
            "people": summary["people_share_pct"].to_numpy(float) / 100.0,
        },
        index=summary["archetype"].astype(int),
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    bottoms = np.zeros(2)
    for archetype in range(1, 5):
        heights = figure_values.loc[archetype, ["households", "people"]].to_numpy(float)
        bars = ax.bar(
            [0, 1],
            heights,
            width=0.4,
            bottom=bottoms,
            color=CLUSTER_COLORS[archetype],
        )
        for bar, height, bottom in zip(bars, heights, bottoms, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + height / 2,
                f"{height * 100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        bottoms += heights
    ax.set_xticks([0, 1], ["households", "people"])
    ax.set_ylabel("Proportion of Total")
    ax.set_ylim(0, 1)
    ax.legend(
        handles=[
            Patch(color=CLUSTER_COLORS[index], label=f"Cluster {index}")
            for index in (4, 3, 2, 1)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )
    fig.tight_layout()
    outputs[FIGURE_STEMS[4]] = _save_figure(fig, figure_dir, FIGURE_STEMS[4])
    plt.close(fig)

    probability_coefficients, probability_fitted = _fit_probability_models(probabilities)
    age_profiles = [
        (100.0, "before 1945"),
        (60.0, "1945-1970"),
        (40.0, "1970-1990"),
        (20.0, "After 1990"),
    ]
    area_line = np.linspace(40.0, 140.0, 101)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
    for archetype, ax in enumerate(axes.flat, start=1):
        plotted_values: list[float] = []
        line_labels: list[tuple[float, float, str]] = []
        for line_index, (age_value, label) in enumerate(age_profiles):
            prediction = np.clip(
                _probability_design(
                    np.full(len(area_line), age_value), area_line
                )
                @ probability_coefficients[archetype],
                0.0,
                1.0,
            )
            plotted_values.extend(prediction.tolist())
            ax.plot(area_line, prediction, color="#A7A7A7", ls="--", lw=1.0)
            label_x = 62.0 + line_index * 7.0
            label_y = float(np.interp(label_x, area_line, prediction))
            line_labels.append((label_x, label_y, label))
            nearest_age = float(
                probabilities.iloc[
                    (probabilities["building_age_years"] - age_value).abs().argsort()[:1]
                ]["building_age_years"].iloc[0]
            )
            empirical = probabilities[
                probabilities["building_age_years"].eq(nearest_age)
                & probabilities["dwelling_area_m2"].between(35, 145)
            ]
            ax.scatter(
                empirical["dwelling_area_m2"],
                empirical[f"prob_archetype_{archetype}"],
                marker="x",
                s=20,
                linewidths=0.8,
                color=CLUSTER_COLORS[archetype],
                alpha=0.65,
            )
            plotted_values.extend(empirical[f"prob_archetype_{archetype}"].tolist())
        for vertical_offset, (label_x, label_y, label) in zip(
            [-22, -7, 7, 22],
            sorted(line_labels, key=lambda item: item[1]),
            strict=True,
        ):
            ax.annotate(
                label,
                xy=(label_x, label_y),
                xytext=(4, vertical_offset),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.5},
            )
        ax.set_title(f"Cluster {archetype}")
        ax.set_xlim(35, 145)
        low = max(0.0, min(plotted_values) - 0.035)
        high = min(1.0, max(plotted_values) + 0.035)
        if high - low < 0.12:
            midpoint = (high + low) / 2
            low = max(0.0, midpoint - 0.07)
            high = min(1.0, midpoint + 0.07)
        ax.set_ylim(low, high)
        ax.grid(True)
    fig.supxlabel("Dwelling Area (m$^2$)", y=0.02)
    fig.supylabel("Predicted Probability", x=0.02)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.10, hspace=0.28, wspace=0.25)
    outputs[FIGURE_STEMS[5]] = _save_figure(fig, figure_dir, FIGURE_STEMS[5])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 5.5))
    for archetype in range(1, 5):
        ax.scatter(
            probability_fitted[f"actual_{archetype}"],
            probability_fitted[f"predicted_{archetype}"],
            color=CLUSTER_COLORS[archetype],
            marker="x",
            s=21,
            linewidths=0.8,
            alpha=0.7,
            label=f"Cluster {archetype}",
        )
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=0.9, label="y = x")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual Proportion")
    ax.set_ylabel("Predicted Probability")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    outputs[FIGURE_STEMS[6]] = _save_figure(fig, figure_dir, FIGURE_STEMS[6])
    plt.close(fig)

    if demographic_only:
        expected = set(FIGURE_STEMS[:7])
        if set(outputs) != expected:
            raise RuntimeError(
                "Demographic/PFE figures are incomplete: "
                f"{sorted(expected - set(outputs))}"
            )
        return outputs

    if energy is None or heating_groups_path is None or electricity_groups_path is None:
        raise ValueError("Energy figure inputs are required unless demographic_only=True")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2))
    boxplot_specs = [
        (
            axes[0],
            "heating_tukey_group",
            "heating_daily_hours",
            "insulation_level",
            ["sufficient_or_high", "none_or_slight"],
            ["#E8D0B2", "#D5A374"],
            ["Sufficiently insulated", "Not sufficiently insulated"],
            "Daily On-Time of Heating System",
            "Insulation Type",
        ),
        (
            axes[1],
            "electricity_tukey_group",
            "electricity_annual_kwh",
            "cooling_system_use",
            ["absent_or_unused", "present_or_used"],
            ["#A8BED5", "#243B5A"],
            ["Cooling System Not Used", "Cooling System Used"],
            "Annual Appliance Consumption (kWh)",
            "Cooling System",
        ),
    ]
    for (
        ax,
        group_column,
        value_column,
        hue_column,
        hue_levels,
        colors,
        legend_labels,
        y_label,
        legend_title,
    ) in boxplot_specs:
        group_labels = sorted(energy[group_column].unique(), key=_group_sort_key)
        centers = np.arange(1, len(group_labels) + 1, dtype=float)
        offsets = [-0.18, 0.18]
        for hue_index, hue_level in enumerate(hue_levels):
            for group_index, group_label in enumerate(group_labels):
                values = energy.loc[
                    energy[group_column].eq(group_label)
                    & energy[hue_column].eq(hue_level),
                    value_column,
                ].to_numpy(float)
                if len(values) == 0:
                    continue
                ax.boxplot(
                    [values],
                    positions=[centers[group_index] + offsets[hue_index]],
                    widths=0.34,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    boxprops={"facecolor": colors[hue_index], "edgecolor": "#444444", "linewidth": 0.65},
                    whiskerprops={"color": "#444444", "linewidth": 0.65},
                    capprops={"color": "#444444", "linewidth": 0.65},
                    medianprops={"color": "#444444", "linewidth": 0.75},
                )
        ax.set_xticks(centers, [f"Group {label}" for label in group_labels])
        ax.set_ylabel(y_label)
        ax.grid(axis="y", ls="--", alpha=0.75)
        ax.legend(
            handles=[Patch(facecolor=color, edgecolor="#444444", label=label) for color, label in zip(colors, legend_labels, strict=True)],
            title=legend_title,
            loc="upper left",
            frameon=True,
        )
    axes[0].text(0.5, -0.20, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontsize=12)
    axes[1].text(0.5, -0.20, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontsize=12)
    fig.subplots_adjust(hspace=0.42, left=0.12, right=0.98, top=0.98, bottom=0.08)
    outputs[FIGURE_STEMS[7]] = _save_figure(fig, figure_dir, FIGURE_STEMS[7])
    plt.close(fig)

    electricity_groups = sorted(
        energy["electricity_tukey_group"].unique(), key=_group_sort_key
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    blue_values = np.linspace(0.25, 0.85, len(electricity_groups))
    electricity_box_stats: list[dict[str, float | str]] = []
    for group_label in electricity_groups:
        selected = energy[energy["electricity_tukey_group"].eq(group_label)]
        electricity_values = selected["electricity_annual_kwh"].to_numpy(float)
        x_q1, x_median, x_q3 = np.quantile(electricity_values, [0.25, 0.5, 0.75])
        iqr = x_q3 - x_q1
        lower = max(float(electricity_values.min()), x_q1 - 1.5 * iqr)
        upper = min(float(electricity_values.max()), x_q3 + 1.5 * iqr)
        electricity_box_stats.append(
            {
                "group": group_label,
                "q1": float(x_q1),
                "median": float(x_median),
                "q3": float(x_q3),
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    reference_area_max = max(
        400.0, float(energy["dwelling_area_m2"].quantile(0.995))
    )
    group_positions = np.linspace(
        45.0, reference_area_max - 45.0, len(electricity_groups)
    )
    box_height = min(55.0, reference_area_max / (len(electricity_groups) * 1.9))
    for stats, blue_value, y_position in zip(
        electricity_box_stats, blue_values, group_positions, strict=True
    ):
        color = plt.cm.Blues(blue_value)
        ax.add_patch(
            Rectangle(
                (float(stats["q1"]), y_position - box_height / 2),
                float(stats["q3"]) - float(stats["q1"]),
                box_height,
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.75,
                alpha=0.95,
            )
        )
        ax.plot(
            [float(stats["median"]), float(stats["median"])],
            [y_position - box_height / 2, y_position + box_height / 2],
            color="#333333",
            lw=0.9,
        )
        ax.plot(
            [float(stats["lower"]), float(stats["upper"])],
            [y_position, y_position],
            color="#555555",
            lw=0.8,
        )
        cap_height = box_height * 0.32
        ax.plot(
            [float(stats["lower"]), float(stats["lower"])],
            [y_position - cap_height / 2, y_position + cap_height / 2],
            color="#555555",
            lw=0.8,
        )
        ax.plot(
            [float(stats["upper"]), float(stats["upper"])],
            [y_position - cap_height / 2, y_position + cap_height / 2],
            color="#555555",
            lw=0.8,
        )
    x_max = max(
        5000.0,
        max(float(stats["upper"]) for stats in electricity_box_stats) * 1.12,
    )
    line_x = np.linspace(0, x_max, 200)
    ax.plot(line_x, line_x * (320.0 / 5000.0), color="#229A2E", ls="--", lw=1.2, label="EN 16798 - Apartment")
    ax.plot(line_x, line_x * (400.0 / 5000.0), color="#FF9800", ls="--", lw=1.2, label="EN 16798 - Detached Houses")
    ax.axvline(1717.0, color="#FF3333", ls="--", lw=1.2, label="ISO 18523-2")
    ax.set_xlim(0, x_max * 1.03)
    ax.set_ylim(0, reference_area_max)
    ax.set_xlabel("Yearly Electricity Consumption [kWh]")
    ax.set_ylabel("Dwelling Area [m$^2$]")
    group_axis = ax.twinx()
    group_axis.set_ylim(0, reference_area_max)
    group_axis.set_yticks(group_positions, electricity_groups)
    group_axis.set_ylabel("Electricity Group")
    group_axis.tick_params(axis="y", length=0)
    ax.grid(True, ls="--", alpha=0.65)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    outputs[FIGURE_STEMS[8]] = _save_figure(fig, figure_dir, FIGURE_STEMS[8])
    plt.close(fig)

    fig = _draw_group_assignment_matrix(
        plt, heating_groups_path, electricity_groups_path
    )
    outputs[FIGURE_STEMS[9]] = _save_figure(fig, figure_dir, FIGURE_STEMS[9])
    plt.close(fig)

    if set(outputs) != set(FIGURE_STEMS):
        raise RuntimeError(
            f"Paper-replica figures are incomplete: {sorted(set(FIGURE_STEMS) - set(outputs))}"
        )
    return outputs


def _set_repeat_table_header(row: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = row._tr.get_or_add_trPr()
    header = properties.find(qn("w:tblHeader"))
    if header is None:
        header = OxmlElement("w:tblHeader")
        properties.append(header)
    header.set(qn("w:val"), "true")


def _set_table_geometry(table: Any, widths_dxa: Sequence[int]) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    total = sum(widths_dxa)
    table.autofit = False
    table_properties = table._tbl.tblPr
    for tag, attributes in [
        ("w:tblW", {"w:w": str(total), "w:type": "dxa"}),
        ("w:tblInd", {"w:w": "120", "w:type": "dxa"}),
        ("w:tblLayout", {"w:type": "fixed"}),
    ]:
        element = table_properties.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            table_properties.append(element)
        for key, value in attributes.items():
            element.set(qn(key), value)

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[index]
            cell.width = width
            properties = cell._tc.get_or_add_tcPr()
            cell_width = properties.find(qn("w:tcW"))
            if cell_width is None:
                cell_width = OxmlElement("w:tcW")
                properties.append(cell_width)
            cell_width.set(qn("w:w"), str(width))
            cell_width.set(qn("w:type"), "dxa")
            margins = properties.find(qn("w:tcMar"))
            if margins is None:
                margins = OxmlElement("w:tcMar")
                properties.append(margins)
            for side, value in [
                ("top", 80),
                ("bottom", 80),
                ("start", 120),
                ("end", 120),
            ]:
                margin = margins.find(qn(f"w:{side}"))
                if margin is None:
                    margin = OxmlElement(f"w:{side}")
                    margins.append(margin)
                margin.set(qn("w:w"), str(value))
                margin.set(qn("w:type"), "dxa")


def _add_number_field(paragraph: Any, sequence: str, number: int) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), f"SEQ {sequence} \\* ARABIC")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = str(number)
    run.append(text)
    field.append(run)
    paragraph._p.append(field)


def _caption(document: Any, label: str, number: int, text: str, before: bool) -> Any:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    paragraph = document.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4 if before else 3)
    paragraph.paragraph_format.space_after = Pt(3 if before else 8)
    paragraph.paragraph_format.keep_with_next = before
    paragraph.add_run(f"{label} ")
    _add_number_field(paragraph, label, number)
    paragraph.add_run(f". {text}")
    return paragraph


def _set_run_font(run: Any, size: float, bold: bool = False, italic: bool = False) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.name = "Calibri"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Calibri")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def _add_table(
    document: Any,
    number: int,
    caption: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    widths_dxa: Sequence[int],
    font_size: float = 8.5,
) -> Any:
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import RGBColor

    _caption(document, "Table", number, caption, before=True)
    rows = list(rows)
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    _set_table_geometry(table, widths_dxa)
    header_row = table.rows[0]
    _set_repeat_table_header(header_row)
    for index, value in enumerate(headers):
        cell = header_row.cells[index]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "E8EEF5")
        cell._tc.get_or_add_tcPr().append(shading)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(str(value))
        _set_run_font(run, font_size, bold=True)
        run.font.color.rgb = RGBColor(31, 77, 120)
    for row_values in rows:
        row = table.add_row()
        for index, value in enumerate(row_values):
            cell = row.cells[index]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = paragraph.add_run(str(value))
            _set_run_font(run, font_size)
    _set_table_geometry(table, widths_dxa)
    document.add_paragraph().paragraph_format.space_after = 0
    return table


def _add_figure(
    document: Any,
    number: int,
    png_path: Path,
    caption: str,
    width_inches: float = 6.15,
) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    shape = run.add_picture(str(png_path), width=Inches(width_inches))
    shape._inline.docPr.set("descr", caption)
    shape._inline.docPr.set("title", f"Figure {number}")
    _caption(document, "Figure", number, caption, before=False)


def _add_appendix_figure_caption(document: Any, text: str) -> Any:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    paragraph = document.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.add_run(f"Figure A.1. {text}")
    return paragraph


def _embed_svg_fallbacks(docx_path: Path, svg_paths: Sequence[Path]) -> None:
    """Embed native SVGs alongside the PNG fallbacks inserted by python-docx."""
    from lxml import etree

    namespaces = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
        "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
        "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    }
    with zipfile.ZipFile(docx_path, "r") as source:
        archive = {name: source.read(name) for name in source.namelist()}

    document_root = etree.fromstring(archive["word/document.xml"])
    blips = document_root.xpath("//a:blip", namespaces=namespaces)
    if len(blips) != len(svg_paths):
        raise RuntimeError(
            f"Expected {len(svg_paths)} Word image fallbacks, found {len(blips)}"
        )

    relationships_root = etree.fromstring(
        archive["word/_rels/document.xml.rels"]
    )
    relationship_tag = f"{{{namespaces['pr']}}}Relationship"
    existing_ids = {
        element.get("Id")
        for element in relationships_root.findall(relationship_tag)
    }
    for index, (blip, svg_path) in enumerate(zip(blips, svg_paths, strict=True), start=1):
        relationship_id = f"rIdPaperSvg{index}"
        if relationship_id in existing_ids:
            raise RuntimeError(f"Duplicate SVG relationship ID: {relationship_id}")
        media_name = f"paper_figure_{index:02d}_{svg_path.stem}.svg"
        relationship = etree.SubElement(relationships_root, relationship_tag)
        relationship.set("Id", relationship_id)
        relationship.set(
            "Type",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
        )
        relationship.set("Target", f"media/{media_name}")

        extension_list = etree.SubElement(
            blip, f"{{{namespaces['a']}}}extLst"
        )
        extension = etree.SubElement(
            extension_list, f"{{{namespaces['a']}}}ext"
        )
        extension.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
        svg_blip = etree.SubElement(
            extension, f"{{{namespaces['asvg']}}}svgBlip"
        )
        svg_blip.set(f"{{{namespaces['r']}}}embed", relationship_id)
        archive[f"word/media/{media_name}"] = svg_path.read_bytes()

    content_types_root = etree.fromstring(archive["[Content_Types].xml"])
    default_tag = f"{{{namespaces['ct']}}}Default"
    if not any(
        element.get("Extension", "").lower() == "svg"
        for element in content_types_root.findall(default_tag)
    ):
        svg_default = etree.SubElement(content_types_root, default_tag)
        svg_default.set("Extension", "svg")
        svg_default.set("ContentType", "image/svg+xml")

    archive["word/document.xml"] = etree.tostring(
        document_root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    archive["word/_rels/document.xml.rels"] = etree.tostring(
        relationships_root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    archive["[Content_Types].xml"] = etree.tostring(
        content_types_root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )

    temporary = docx_path.with_suffix(".svg-embedded.docx")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for name in sorted(archive):
            target.writestr(name, archive[name])
    os.replace(temporary, docx_path)


def _format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _normalize_docx_archive(path: Path) -> None:
    """Rewrite ZIP metadata so identical report inputs produce identical DOCX bytes."""
    temporary = path.with_suffix(".normalized.docx")
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for name in sorted(source.namelist()):
            data = source.read(name)
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            target.writestr(info, data)
    os.replace(temporary, path)


def build_report(
    baseline_dir: Path,
    pfe_dir: Path,
    energy_dir: Path,
    figure_paths: dict[str, dict[str, Path]],
    output_path: Path,
    config: ReportConfig,
) -> None:
    _require_report_dependencies()
    from docx import Document
    from docx.enum.section import WD_ORIENT, WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    archetype_summary = pd.read_csv(baseline_dir / "archetype_summary.csv")
    probabilities = pd.read_csv(
        pfe_dir / "dwelling_archetype_probabilities.csv"
    )
    anova = pd.read_csv(energy_dir / "anova_results.csv")
    groups = pd.read_csv(energy_dir / "energy_group_summary.csv")
    factors = pd.read_csv(energy_dir / "energy_factor_summary.csv")
    baseline_report = json.loads(
        (baseline_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    pfe_report = json.loads(
        (pfe_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    energy_report = json.loads(
        (energy_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    probability_coefficients, probability_fitted = _fit_probability_models(
        probabilities
    )

    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for style_name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 18, 10),
        ("Heading 2", 13, "2E74B5", 14, 7),
        ("Heading 3", 12, "1F4D78", 10, 5),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    caption_style = styles["Caption"]
    caption_style.font.name = "Calibri"
    caption_style.font.size = Pt(9)
    caption_style.font.italic = True
    caption_style.font.color.rgb = RGBColor.from_string("475569")

    header = section.header.paragraphs[0]
    header.text = "GRINS | ISTAT archetype workflow"
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in header.runs:
        _set_run_font(run, 8.5)
        run.font.color.rgb = RGBColor.from_string("64748B")
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Page ")
    page_field = OxmlElement("w:fldSimple")
    page_field.set(qn("w:instr"), "PAGE")
    page_run = OxmlElement("w:r")
    page_text = OxmlElement("w:t")
    page_text.text = "1"
    page_run.append(page_text)
    page_field.append(page_run)
    footer._p.append(page_field)
    for run in footer.runs:
        _set_run_font(run, 8.5)
        run.font.color.rgb = RGBColor.from_string("64748B")

    document.add_paragraph().paragraph_format.space_after = Pt(92)
    kicker = document.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = kicker.add_run("AUTOMATED TECHNICAL REPORT")
    _set_run_font(run, 10, bold=True)
    run.font.color.rgb = RGBColor.from_string("7A5A00")
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(9)
    run = title.add_run(config.title)
    _set_run_font(run, 27, bold=True)
    run.font.color.rgb = RGBColor.from_string("203748")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(30)
    run = subtitle.add_run(config.subtitle)
    _set_run_font(run, 14)
    run.font.color.rgb = RGBColor.from_string("2B5163")
    scope = document.add_paragraph()
    scope.alignment = WD_ALIGN_PARAGRAPH.CENTER
    scope.paragraph_format.space_after = Pt(88)
    run = scope.add_run(
        "Household clustering, dwelling probabilities, Type-II ANOVA, "
        "and Tukey-Kramer groups"
    )
    _set_run_font(run, 10.5, italic=True)
    run.font.color.rgb = RGBColor.from_string("475569")
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = note.add_run(
        "Generated from workflow outputs. Statistical interpretation is intentionally omitted."
    )
    _set_run_font(run, 9.5)
    run.font.color.rgb = RGBColor.from_string("64748B")
    document.add_page_break()

    table_number = 1
    figure_number = 1

    document.add_heading("1. Data and workflow configuration", level=1)
    document.add_paragraph(
        "The report is generated from the raw ISTAT household survey, the restored "
        "four-archetype assignments, the windowed dwelling probability dataset, and "
        "the GRINS annual electricity results."
    )
    metadata_rows = [
        ("Raw ISTAT households", baseline_report["row_counts"]["raw_households"]),
        (
            "Archetyped households",
            baseline_report["row_counts"]["eligible_households"],
        ),
        ("PFE windows", pfe_report["row_counts"]["supported_windows"]),
        (
            "Energy-analysis households",
            energy_report["row_counts"]["analysis_households"],
        ),
        ("Age seed", baseline_report["configuration"]["age_seed"]),
        ("Cluster seed", baseline_report["configuration"]["cluster_seed"]),
        ("Dwelling seed", pfe_report["configuration"]["dwelling_seed"]),
        ("ANOVA alpha", energy_report["configuration"]["alpha"]),
    ]
    _add_table(
        document,
        table_number,
        "Workflow inputs, row counts, and deterministic configuration.",
        ["Item", "Value"],
        metadata_rows,
        [2700, 6660],
        9,
    )
    table_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["study_process_flow"]["png"],
        "Process flow of the study.",
    )
    figure_number += 1

    document.add_heading("2. Household archetypes", level=1)
    document.add_heading("2.1 Feature construction and clustering", level=2)
    document.add_paragraph(
        "Household member ages, household size, and the education index are used "
        "to construct two published PCA components. The three normalized clustering "
        "features are PCA component 1, PCA component 2, and the education index. "
        "Four clusters are estimated with seeded k-means++ initialization and Lloyd iteration."
    )
    equation = document.add_paragraph()
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = equation.add_run(
        "PC1 = -0.25 x minimum age - 0.75 x maximum age + 10 x household members"
    )
    _set_run_font(run, 10, italic=True)
    equation = document.add_paragraph()
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = equation.add_run(
        "PC2 = (1/3) x minimum age + (2/3) x maximum age + 10 x household members"
    )
    _set_run_font(run, 10, italic=True)
    _add_figure(
        document,
        figure_number,
        figure_paths["pca_explained_variance"]["png"],
        "Explained variance of principal components considering (a) all variables and (b) excluding education.",
    )
    figure_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["clustering_elbow"]["png"],
        "Within Cluster Sum of Squares as a function of the number of clusters.",
        width_inches=5.55,
    )
    figure_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["regional_cluster_centroids"]["png"],
        "Cluster centroids for individual Italian regions and the complete dataset. Gray polygons represent regional solutions; colored points and the red polygon represent the complete dataset.",
        width_inches=5.65,
    )
    figure_number += 1

    archetype_rows = []
    for _, row in archetype_summary.sort_values("archetype").iterrows():
        archetype_rows.append(
            [
                int(row["archetype"]),
                int(row["households"]),
                f"{row['household_share_pct']:.1f}",
                f"{row['people_share_pct']:.1f}",
                f"{row['agemin_mean']:.1f} ({row['agemin_sd']:.1f})",
                f"{row['agemax_mean']:.1f} ({row['agemax_sd']:.1f})",
                f"{row['nmembers_mean']:.1f} ({row['nmembers_sd']:.1f})",
                f"{row['education_mean']:.2f} ({row['education_sd']:.2f})",
            ]
        )
    _add_table(
        document,
        table_number,
        "Restored four-archetype household statistics. Values in parentheses are standard deviations.",
        [
            "Archetype",
            "Households",
            "Household share (%)",
            "People share (%)",
            "Minimum age",
            "Maximum age",
            "Members",
            "Education index",
        ],
        archetype_rows,
        [650, 850, 1050, 950, 1400, 1400, 1300, 1760],
        7.4,
    )
    table_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["archetype_population_shares"]["png"],
        "Proportion of each cluster among households and people.",
        width_inches=5.65,
    )
    figure_number += 1

    document.add_heading("3. Dwelling-to-archetype probabilities", level=1)
    document.add_heading("3.1 Windowed probabilistic feature extraction", level=2)
    document.add_paragraph(
        "Building-age and dwelling-area classes are expanded with the configured "
        "dwelling seed. Archetype counts are calculated in centered age and area "
        "windows. Each probability equals the archetype count divided by the total "
        "sample count in its window."
    )
    probability_rows = [
        ("Configured windows", len(probabilities)),
        ("Building-age centers", probabilities["building_age_years"].nunique()),
        ("Dwelling-area centers", probabilities["dwelling_area_m2"].nunique()),
        ("Minimum sample count", int(probabilities["sample_count"].min())),
        ("Maximum sample count", int(probabilities["sample_count"].max())),
        (
            "Maximum probability-sum error",
            f"{np.abs(probabilities['probability_sum'] - 1.0).max():.3g}",
        ),
    ]
    _add_table(
        document,
        table_number,
        "Windowed dwelling probability dataset summary.",
        ["Item", "Value"],
        probability_rows,
        [4200, 5160],
        9,
    )
    table_number += 1
    coefficient_names = [
        "intercept",
        "age / 100",
        "area / 100",
        "age squared / 10,000",
        "area squared / 10,000",
        "age x area / 10,000",
    ]
    regression_rows: list[list[str]] = []
    for coefficient_index, coefficient_name in enumerate(coefficient_names):
        regression_rows.append(
            [coefficient_name]
            + [
                f"{probability_coefficients[archetype][coefficient_index]:.3f}"
                for archetype in range(1, 5)
            ]
        )
    adjusted_r_squared: list[float] = []
    root_mean_squared_error: list[float] = []
    for archetype in range(1, 5):
        actual = probability_fitted[f"actual_{archetype}"].to_numpy(float)
        predicted = probability_fitted[f"predicted_{archetype}"].to_numpy(float)
        residual_sum = float(np.sum((actual - predicted) ** 2))
        total_sum = float(np.sum((actual - actual.mean()) ** 2))
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
        n = len(actual)
        predictors = 5
        adjusted_r_squared.append(
            1.0 - (1.0 - r_squared) * (n - 1) / (n - predictors - 1)
        )
        root_mean_squared_error.append(math.sqrt(residual_sum / n))
    regression_rows.append(
        ["Adjusted R squared"] + [f"{value:.3f}" for value in adjusted_r_squared]
    )
    regression_rows.append(
        ["RMSE"] + [f"{value:.3f}" for value in root_mean_squared_error]
    )
    _add_table(
        document,
        table_number,
        "Windowed probability regression coefficients and goodness-of-fit statistics.",
        ["Term", "Prob. 1", "Prob. 2", "Prob. 3", "Prob. 4"],
        regression_rows,
        [3000, 1590, 1590, 1590, 1590],
        8.2,
    )
    table_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["dwelling_probability_profiles"]["png"],
        "Probability of each cluster as dwelling age and area change. Points are calculated proportions; dashed lines are regression-model predictions for different construction-age classes.",
    )
    figure_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["probability_model_validation"]["png"],
        "Alignment between predicted and actual values across clusters.",
        width_inches=5.55,
    )
    figure_number += 1

    document.add_heading("4. Energy analysis", level=1)
    document.add_heading("4.1 Response and explanatory variables", level=2)
    document.add_paragraph(
        "Daily heating-system on-time is the sum of q_3_11A, q_3_11B, and q_3_11C. "
        "Annual electricity consumption is appliances_el_kWh. Explanatory variables "
        "are user archetype, dwelling type, location, insulation level, building age, "
        "dwelling area, and cooling-system use."
    )
    document.add_heading("4.2 Type-II ANOVA", level=2)
    term_order = anova[anova["response"].eq("heating_daily_hours")]["term"].tolist()
    heating = anova[anova["response"].eq("heating_daily_hours")].set_index("term")
    electricity = anova[anova["response"].eq("electricity_annual_kwh")].set_index(
        "term"
    )
    anova_rows = []
    for term in term_order:
        anova_rows.append(
            [
                term,
                f"{heating.loc[term, 'f_statistic']:.2f}",
                _format_p(float(heating.loc[term, "p_value"])),
                f"{electricity.loc[term, 'f_statistic']:.2f}",
                _format_p(float(electricity.loc[term, "p_value"])),
            ]
        )
    _add_table(
        document,
        table_number,
        "Type-II ANOVA results for daily heating-system on-time and annual electricity consumption.",
        ["Variable or interaction", "Heating F", "Heating p", "Electricity F", "Electricity p"],
        anova_rows,
        [3600, 1440, 1440, 1440, 1440],
        8,
    )
    table_number += 1

    document.add_heading("4.3 Iterative Tukey-Kramer groups", level=2)
    for response, heading in [
        (
            "heating_daily_hours",
            "Heating-system on-time groups",
        ),
        (
            "electricity_annual_kwh",
            "Annual electricity groups",
        ),
    ]:
        document.add_heading(heading, level=3)
        selected = groups[groups["response"].eq(response)].sort_values("group")
        group_rows = [
            [
                row["group"],
                int(row["sample_count"]),
                f"{row['mean']:.2f}",
                f"{row['standard_deviation']:.2f}",
                "" if pd.isna(row["paper_mean"]) else f"{row['paper_mean']:.2f}",
                ""
                if pd.isna(row["paper_standard_deviation"])
                else f"{row['paper_standard_deviation']:.2f}",
            ]
            for _, row in selected.iterrows()
        ]
        _add_table(
            document,
            table_number,
            f"{heading}: generated and published summary statistics.",
            ["Group", "n", "Generated mean", "Generated SD", "Paper mean", "Paper SD"],
            group_rows,
            [900, 900, 1890, 1890, 1890, 1890],
            8.3,
        )
        table_number += 1

    _add_figure(
        document,
        figure_number,
        figure_paths["energy_group_boxplots"]["png"],
        "Variation of (a) daily heating-system operating hours and (b) annual electricity consumption across the generated significant groups. Box plots are drawn without outliers; boxes span the second and third quartiles and the internal line is the median.",
    )
    figure_number += 1
    _add_figure(
        document,
        figure_number,
        figure_paths["electricity_standard_comparison"]["png"],
        "Yearly electricity consumption, dwelling area, generated electricity groups, and the reference assignments from EN 16798 and ISO 18523-2.",
    )
    figure_number += 1

    factor_rows = [
        [
            row["response"],
            row["factor"],
            row["level"],
            int(row["sample_count"]),
            f"{row['mean']:.2f}",
            f"{row['standard_deviation']:.2f}",
        ]
        for _, row in factors.iterrows()
    ]
    _add_table(
        document,
        table_number,
        "Separate categorical factor summaries retained by the paper methodology.",
        ["Response", "Factor", "Level", "n", "Mean", "SD"],
        factor_rows,
        [1800, 1500, 2200, 900, 1480, 1480],
        8,
    )
    table_number += 1

    document.add_heading("5. Workflow validation", level=1)
    alignment = energy_report["validation"]["paper_alignment"]
    validation_rows = [
        ("Four-archetype paper-tolerance checks", "PASS" if baseline_report["passed"] else "FAIL"),
        ("Probability checks", "PASS" if pfe_report["passed"] else "FAIL"),
        ("Energy structural checks", "PASS" if energy_report["passed"] else "FAIL"),
        (
            "ANOVA significance matches",
            f"{alignment['anova_significance_matches']} of {alignment['anova_terms_compared']}",
        ),
        (
            "Generated heating groups / paper count",
            f"{alignment['heating_groups']} / {alignment['paper_heating_groups']}",
        ),
        (
            "Generated electricity groups / paper text / Table 6",
            f"{alignment['electricity_groups']} / {alignment['paper_text_electricity_groups']} / "
            f"{alignment['paper_table_6_electricity_groups_listed']}",
        ),
    ]
    _add_table(
        document,
        table_number,
        "Automated workflow and paper-alignment validation record.",
        ["Check", "Result"],
        validation_rows,
        [6100, 3260],
        8.8,
    )
    table_number += 1

    appendix_section = document.add_section(WD_SECTION.NEW_PAGE)
    appendix_section.orientation = WD_ORIENT.PORTRAIT
    appendix_section.page_width = Inches(8.5)
    appendix_section.page_height = Inches(11)
    appendix_section.top_margin = Inches(1)
    appendix_section.right_margin = Inches(1)
    appendix_section.bottom_margin = Inches(1)
    appendix_section.left_margin = Inches(1)
    appendix_section.different_first_page_header_footer = False
    document.add_heading(
        "Appendix A. Assignment of user groups by dwelling type, location, and archetype",
        level=1,
    )
    appendix_figure = document.add_paragraph()
    appendix_figure.alignment = WD_ALIGN_PARAGRAPH.CENTER
    appendix_figure.paragraph_format.keep_with_next = True
    appendix_figure.paragraph_format.space_after = Pt(0)
    appendix_run = appendix_figure.add_run()
    appendix_shape = appendix_run.add_picture(
        str(figure_paths["group_assignment_matrix"]["png"]),
        width=Inches(6.15),
    )
    appendix_shape._inline.docPr.set(
        "descr",
        "Assignment of generated heating and electricity groups by dwelling type, location, and user archetype.",
    )
    appendix_shape._inline.docPr.set("title", "Figure A.1")
    _add_appendix_figure_caption(
        document,
        "Assignment of generated heating and electricity groups by dwelling type, location, and user archetype.",
    )

    probability_section = document.add_section(WD_SECTION.NEW_PAGE)
    probability_section.orientation = WD_ORIENT.PORTRAIT
    probability_section.page_width = Inches(8.5)
    probability_section.page_height = Inches(11)
    probability_section.top_margin = Inches(1)
    probability_section.right_margin = Inches(1)
    probability_section.bottom_margin = Inches(1)
    probability_section.left_margin = Inches(1)
    probability_section.different_first_page_header_footer = False
    document.add_heading("Appendix B. Windowed archetype probabilities", level=1)
    appendix_rows = []
    for _, row in probabilities.sort_values(
        ["dwelling_area_m2", "building_age_years"]
    ).iterrows():
        appendix_rows.append(
            [
                int(row["building_age_years"]),
                int(row["dwelling_area_m2"]),
                int(row["sample_count"]),
                f"{row['prob_archetype_1']:.3f}",
                f"{row['prob_archetype_2']:.3f}",
                f"{row['prob_archetype_3']:.3f}",
                f"{row['prob_archetype_4']:.3f}",
            ]
        )
    _add_table(
        document,
        table_number,
        "Complete building-age and dwelling-area probability grid.",
        ["Age", "Area", "n", "P(A1)", "P(A2)", "P(A3)", "P(A4)"],
        appendix_rows,
        [1100, 1100, 1100, 1515, 1515, 1515, 1515],
        7.7,
    )

    document.add_heading("Source", level=1)
    document.add_paragraph(
        "Khajedehi, M. H. et al. Identification of Italian user archetypes for "
        "building energy simulations based on a national survey. Energy & Buildings "
        "344 (2025) 115924. https://doi.org/10.1016/j.enbuild.2025.115924"
    )

    properties = document.core_properties
    properties.title = config.title
    properties.subject = "Automated methods and results report"
    properties.author = "GRINS deterministic workflow"
    properties.last_modified_by = "GRINS deterministic workflow"
    properties.keywords = "ISTAT, archetypes, PFE, ANOVA, Tukey"
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    properties.created = fixed_time
    properties.modified = fixed_time

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    _embed_svg_fallbacks(
        output_path,
        [figure_paths[stem]["svg"] for stem in FIGURE_STEMS],
    )
    _normalize_docx_archive(output_path)


def validate_report(
    output_path: Path,
    figure_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    from docx import Document

    document = Document(output_path)
    captions = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.style.name == "Caption"
    ]
    table_captions = [text for text in captions if text.startswith("Table ")]
    figure_captions = [text for text in captions if text.startswith("Figure ")]
    with zipfile.ZipFile(output_path, "r") as archive:
        document_xml = archive.read("word/document.xml")
        embedded_svg_media = [
            name
            for name in archive.namelist()
            if name.startswith("word/media/") and name.lower().endswith(".svg")
        ]
    expected_figure_count = len(FIGURE_STEMS)
    checks = {
        "docx_exists": output_path.is_file() and output_path.stat().st_size > 0,
        "all_paper_replica_svg_figures_exist": all(
            paths["svg"].is_file() for paths in figure_paths.values()
        )
        and set(figure_paths) == set(FIGURE_STEMS),
        "all_png_fallbacks_exist": all(
            paths["png"].is_file() for paths in figure_paths.values()
        )
        and len(figure_paths) == expected_figure_count,
        "all_figures_are_embedded": len(document.inline_shapes)
        == expected_figure_count,
        "native_svgs_are_embedded_with_png_fallbacks": (
            document_xml.count(b"svgBlip") == expected_figure_count
            and len(embedded_svg_media) == expected_figure_count
        ),
        "all_figures_have_captions": len(figure_captions)
        == expected_figure_count,
        "appendix_assignment_figure_is_present": any(
            caption.startswith("Figure A.1.") for caption in figure_captions
        ),
        "all_tables_have_captions": len(table_captions) == len(document.tables),
        "report_contains_multiple_result_tables": len(document.tables) >= 8,
        "report_contains_no_interpretation_heading": not any(
            re.search(r"interpretation|discussion|implications", paragraph.text, re.I)
            for paragraph in document.paragraphs
            if paragraph.style.name.startswith("Heading")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "tables": len(document.tables),
            "table_captions": len(table_captions),
            "figures": len(document.inline_shapes),
            "figure_captions": len(figure_captions),
        },
    }


def run_pipeline(
    raw_path: Path,
    baseline_dir: Path,
    pfe_dir: Path,
    energy_dir: Path,
    figure_dir: Path,
    output_path: Path,
    validation_path: Path,
    config: ReportConfig,
    raw_source_to_canonical: dict[str, str] | None = None,
) -> dict[str, Any]:
    figure_paths = generate_figures(
        raw_path,
        baseline_dir / "household_archetypes.csv",
        baseline_dir / "archetype_summary.csv",
        baseline_dir / "archetype_model.json",
        pfe_dir / "dwelling_archetype_probabilities.csv",
        energy_dir / "energy_analysis_dataset.csv",
        energy_dir / "heating_tukey_groups.csv",
        energy_dir / "electricity_tukey_groups.csv",
        figure_dir,
        config.cluster_seed,
        raw_source_to_canonical,
    )
    build_report(
        baseline_dir,
        pfe_dir,
        energy_dir,
        figure_paths,
        output_path,
        config,
    )
    validation = validate_report(output_path, figure_paths)
    report = {
        "passed": validation["passed"],
        "configuration": {
            "title": config.title,
            "subtitle": config.subtitle,
            "cluster_seed": config.cluster_seed,
            "raw_source_to_canonical": raw_source_to_canonical or {},
        },
        "validation": validation,
        "outputs": {
            "docx": str(output_path.resolve()),
            "docx_sha256": sha256_file(output_path),
            "figures": {
                stem: {
                    kind: str(path.resolve()) for kind, path in paths.items()
                }
                for stem, paths in figure_paths.items()
            },
        },
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
