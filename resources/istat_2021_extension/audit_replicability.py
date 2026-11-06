"""Audit paper reproduction and compare the 2013/2021 waves without changing the workflow.

Run: .venv/Scripts/python.exe audit_replicability.py
All new results go to replicability_outputs/. Production outputs are only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from archetype_pipeline import (
    AGE_COLUMNS, AGE_RANGES_2021, EDUCATION_COLUMNS, EDUCATION_YEARS,
    add_published_components, build_household_features, canonical_archetype_mapping,
    fit_kmeans, maximum_possible_study_years, run_pipeline as run_archetypes,
    summarize_archetypes, validate_against_paper,
)
from energy_analysis import EnergyConfig, run_pipeline as run_energy
from istat_harmonizer import _harmonize_2013, _harmonize_2021
from pfe_pipeline import (
    build_dwelling_samples, calculate_window_probabilities, run_pipeline as run_pfe,
)
from report_pipeline import _fit_probability_models, _probability_design
from run_multiyear_workflow import (
    _archetype_config, _pfe_config, _resolve_raw_input, load_config,
)

ROOT = Path(__file__).resolve().parent
COMPONENTS = ["pc1", "pc2", "education"]
FEATURES = ["agemin", "agemax", "nmembers", "education"]
PAPER_SHARES = [40.1, 26.9, 22.9, 10.1]
AGE_RANGES_2013 = {0: (0, 14), **{c: (c, c + 9) for c in range(15, 75, 10)}, 75: (75, 100)}
EDUCATION_GROUPS = {
    2013: {1: 0, 3: 0, 4: 1, 5: 2, 7: 3, 9: 3, 10: 3},
    2021: {1: 0, 3: 1, 4: 2, 6: 3},
}
AREA_BANDS = ["<40", "40-59", "60-89", "90-119", "120-149", "150+"]
PERIOD_BANDS = ["before 1950", "1950-1969", "1970-1989", "1990+"]
COLORS = ["#3965a6", "#d17a22"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def read_wave(path: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read weights/heating independently; retain missing hours rather than filling zero."""
    canonical = (_harmonize_2013 if year == 2013 else _harmonize_2021)(path)[0]
    if year == 2013:
        id_col, weight_col, frequency = "nprog", "coef_red", "q_3_10"
        hours = ["q_3_11A", "q_3_11B", "q_3_11C"]
    else:
        id_col, weight_col, frequency = "ID_RECORD", "PESO", "v3_7"
        hours = ["v3_8a", "v3_8b", "v3_8c"]
    cols = [id_col, weight_col, frequency, *hours] + (["TECNICA"] if year == 2021 else [])
    raw = pd.read_csv(path, sep="\t" if year == 2013 else "^", usecols=cols, dtype=str)
    extra = pd.DataFrame({"id": numeric(raw[id_col]).astype(int), "weight": numeric(raw[weight_col])})
    extra["heating_frequency"] = numeric(raw[frequency])
    h = raw[hours].apply(numeric)
    extra["heating_hours"] = h.sum(axis=1, min_count=3)
    # The 2013 question was asked for frequency 1/2, 2021 for 1/2/3.
    extra["heating_comparable"] = (
        extra["heating_frequency"].isin([1, 2]) & h.notna().all(axis=1)
        & h.ge(0).all(axis=1) & h.le(8).all(axis=1)
    )
    extra["survey_mode"] = raw["TECNICA"].str.strip() if year == 2021 else "CATI"
    assert extra.id.is_unique and extra.weight.gt(0).all()
    canonical = canonical.sort_values("id").reset_index(drop=True)
    return canonical, extra


def common_features(
    raw: pd.DataFrame, year: int, seed: int | None = None,
    primary_years: float = 5.0, tertiary_years: float = 18.0, age_cap: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparable conditional sample: 1-5 members, coarsened education, codebook ages.

    Midpoints give a transparent deterministic reference; seeded uniform draws
    measure interval-imputation sensitivity. Years within broad education groups
    are explicit assumptions, not recovered observations.
    """
    rng = np.random.RandomState(seed) if seed is not None else None
    ranges = dict(AGE_RANGES_2013 if year == 2013 else AGE_RANGES_2021)
    last = 75 if year == 2013 else 11
    ranges[last] = (ranges[last][0], age_cap)
    study = [primary_years, 8.0, 12.0, tertiary_years]
    rows, excluded = [], []
    for row in raw.itertuples(index=False):
        record = row._asdict()
        count, household_id = int(record["q_1_1_sq1"]), int(record["id"])
        reason = ""
        if count < 1 or count > 5:
            reason = "outside_common_1_to_5_member_sample"
        if reason:
            excluded.append({"id": household_id, "reason": reason})
            continue
        ages, ratios = [], []
        for i in range(1, count + 1):
            code, edu = int(record[f"cletad_{i}"]), int(record[f"titstu_{i}"])
            if code not in ranges:
                reason = "invalid_member_age"
                break
            lo, hi = ranges[code]
            age = (lo + hi) / 2 if rng is None else float(rng.randint(lo, hi + 1))
            ages.append(age)
            potential = maximum_possible_study_years(int(age))
            # Children with no potential completed schooling are outside this
            # denominator. Missing education is omitted, never treated as zero.
            if potential > 0 and edu in EDUCATION_GROUPS[year]:
                ratios.append(min(study[EDUCATION_GROUPS[year][edu]] / potential, 1.0))
        if reason or not ratios:
            excluded.append({"id": household_id, "reason": reason or "no_defined_member_education"})
            continue
        rows.append({"id": household_id, "agemin": min(ages), "agemax": max(ages),
                     "nmembers": count, "education": float(np.mean(ratios))})
    features = pd.DataFrame(rows, columns=["id", *FEATURES]).astype({"id": int, **{c: float for c in FEATURES}})
    assert features.id.is_unique and np.isfinite(features[FEATURES]).all().all()
    return features, pd.DataFrame(excluded, columns=["id", "reason"])


def component_matrix(features: pd.DataFrame) -> np.ndarray:
    # Equations 1/2 are fixed engineered features, not a claim of refitting PCA.
    return np.column_stack([
        -.25 * features.agemin - .75 * features.agemax + 10 * features.nmembers,
        features.agemin / 3 + 2 * features.agemax / 3 + 10 * features.nmembers,
        features.education,
    ])


def fit_reference(features: pd.DataFrame, seeds=range(10)) -> tuple[dict, pd.DataFrame]:
    unscaled = component_matrix(features)
    lo, span = unscaled.min(axis=0), np.ptp(unscaled, axis=0)
    if np.any(span <= 0):
        raise ValueError("Constant reference feature")
    matrix = (unscaled - lo) / span
    fitted = [fit_kmeans(matrix, 4, int(s), 300, 1e-8) for s in seeds]
    best = min(fitted, key=lambda model: model.inertia)
    mapping = canonical_archetype_mapping(features, best.labels)
    centers = np.array([best.centers[next(k for k, v in mapping.items() if v == a)] for a in range(1, 5)])
    model = {"minimum": lo.tolist(), "span": span.tolist(), "centers": centers.tolist(),
             "initialization_seeds": list(seeds), "selected_inertia": best.inertia,
             "all_inertias": [m.inertia for m in fitted], "training_year": 2013,
             "training_weighting": "unweighted; reported shares also use official survey weights"}
    return model, assign_reference(features, model)


def assign_reference(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    matrix = (component_matrix(features) - np.array(model["minimum"])) / np.array(model["span"])
    out = features.copy()
    out["archetype"] = np.argmin(((matrix[:, None] - np.array(model["centers"])[None]) ** 2).sum(axis=2), axis=1) + 1
    out["outside_reference_range"] = ((matrix < 0) | (matrix > 1)).any(axis=1)
    return out


def aligned_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    joined = a[["id", "archetype"]].merge(b[["id", "archetype"]], on="id", validate="one_to_one")
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (joined.archetype_x.to_numpy(int) - 1, joined.archetype_y.to_numpy(int) - 1), 1)
    rows, cols = linear_sum_assignment(-counts)
    return {"shared_households": len(joined), "best_permutation_agreement_pct": float(100 * counts[rows, cols].sum() / len(joined)),
            "contingency": counts.tolist()}


def summary(frame: pd.DataFrame, year: int, method: str) -> pd.DataFrame:
    rows = []
    for weighted in [False, True]:
        weights = frame.weight.to_numpy(float) if weighted else np.ones(len(frame))
        for a in range(1, 5):
            selected = frame.archetype.eq(a).to_numpy()
            group = frame[selected]
            w = weights[selected]
            rows.append({"method": method, "year": year, "archetype": a,
                "weighting": "survey" if weighted else "sample", "households": len(group),
                "household_share_pct": 100 * w.sum() / weights.sum(),
                "people_share_pct": 100 * np.sum(w * group.nmembers) / np.sum(weights * frame.nmembers),
                **{f"{c}_mean": float(np.average(group[c], weights=w)) if len(w) else np.nan for c in FEATURES}})
    return pd.DataFrame(rows)


def common_dwelling_groups(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    if year == 2013:
        area = {1: "<40", 2: "<40", 3: "40-59", 4: "60-89", 5: "90-119", 6: "120-149", 7: "150+"}
        period = {1: "1990+", 2: "1990+", 3: "1970-1989", 4: "1970-1989", 5: "1950-1969", 6: "1950-1969", 7: "before 1950", 8: "before 1950"}
    else:
        area = {1: "<40", 2: "40-59", 3: "40-59", 4: "60-89", 5: "60-89", 6: "60-89", 7: "90-119", 8: "90-119", 9: "90-119", 10: "120-149", 11: "120-149", **{i: "150+" for i in range(12, 17)}}
        period = {1: "1990+", 2: "1990+", 3: "1990+", 4: "1970-1989", 5: "1970-1989", 6: "1950-1969", 7: "1950-1969", 8: "before 1950"}
    return pd.DataFrame({"id": raw.id, "area_band": raw.q_2_7_class.map(area), "construction_period": raw.q_2_4_ric.map(period), "region": raw.reg})


def dwelling_probabilities(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for (area, period), group in frame.dropna(subset=["area_band", "construction_period"]).groupby(["area_band", "construction_period"]):
        weights = group.weight.to_numpy(float)
        row = {"year": year, "area_band": area, "construction_period": period, "sample_count": len(group),
               "effective_n": weights.sum() ** 2 / np.square(weights).sum()}
        for a in range(1, 5):
            row[f"prob_{a}"] = float(weights[group.archetype.eq(a)].sum() / weights.sum())
        row["meets_support_30"] = len(group) >= 30 and row["effective_n"] >= 30
        rows.append(row)
    result = pd.DataFrame(rows)
    assert np.allclose(result[[f"prob_{a}" for a in range(1, 5)]].sum(axis=1), 1)
    return result


def heating_summary(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    eligible = frame[frame.heating_comparable]
    for a in [0, 1, 2, 3, 4]:
        group = eligible if a == 0 else eligible[eligible.archetype.eq(a)]
        for weighting in ["sample", "survey"]:
            w = group.weight.to_numpy(float) if weighting == "survey" else np.ones(len(group))
            rows.append({"year": year, "archetype": a, "weighting": weighting, "sample_count": len(group),
                         "mean_daily_hours": float(np.average(group.heating_hours, weights=w))})
    return pd.DataFrame(rows)


def direct_survey_summary(raw: pd.DataFrame, extra: pd.DataFrame, year: int) -> pd.DataFrame:
    """Observable comparisons that do not depend on age draws or cluster boundaries."""
    sample = raw[raw.q_1_1_sq1.between(1, 5)].merge(extra, on="id", validate="one_to_one")
    members = sample.q_1_1_sq1.to_numpy(int)
    ages = sample[AGE_COLUMNS[:5]].to_numpy(int)
    ed = sample[EDUCATION_COLUMNS[:5]].to_numpy(int)
    active = np.arange(5)[None] < members[:, None]
    young = ages == 0 if year == 2013 else np.isin(ages, [1, 2, 3])
    tertiary = np.isin(ed, [7, 9, 10] if year == 2013 else [6])
    education_known = np.all(~active | young | np.isin(ed, list(EDUCATION_GROUPS[year])), axis=1)
    dwelling = common_dwelling_groups(sample, year)
    measures = {"mean_household_size": (members.astype(float), np.ones(len(sample), bool), 1),
                "households_with_child_under_15_pct": ((young & active).any(axis=1), np.ones(len(sample), bool), 100),
                "households_with_tertiary_educated_member_pct": ((tertiary & active).any(axis=1), education_known, 100),
                "dwelling_at_least_150m2_pct": (dwelling.area_band.eq("150+"), dwelling.area_band.notna(), 100),
                "construction_before_1950_pct": (dwelling.construction_period.eq("before 1950"), dwelling.construction_period.notna(), 100),
                "construction_unknown_pct": (dwelling.construction_period.isna(), np.ones(len(sample), bool), 100)}
    measures.update({f"households_{n}_member_pct": (members == n, np.ones(len(sample), bool), 100) for n in range(1, 6)})
    rows = []
    for name, (values, valid, factor) in measures.items():
        for weighting in ["sample", "survey"]:
            w = sample.weight.to_numpy(float) if weighting == "survey" else np.ones(len(sample))
            rows.append({"year": year, "measure": name, "weighting": weighting, "denominator_records": int(np.sum(valid)),
                         "value": float(factor * np.average(np.asarray(values)[valid], weights=w[valid]))})
    return pd.DataFrame(rows)


def probability_transfer_score(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Score genuine held-out households, conditional on the fixed archetype model.

    Four probabilities are estimated from common categorical cells with five
    average-weight pseudo-households of shrinkage to the training marginal.
    """
    frames = {y: f.dropna(subset=["area_band", "construction_period"]).sort_values("id").reset_index(drop=True) for y, f in frames.items()}
    scores = []
    training_2013 = None
    for training_year in [2013, 2021]:
        frame = frames[training_year]
        indices = np.random.RandomState(2026).permutation(len(frame))
        cut = int(.8 * len(frame))
        train, test = frame.iloc[indices[:cut]], frame.iloc[indices[cut:]]
        marginal = np.bincount(train.archetype.to_numpy(int)-1, weights=train.weight, minlength=4)
        marginal /= marginal.sum()
        normalized_weights = train.weight / train.weight.mean()
        training = train.assign(_weight=normalized_weights)
        lookup = {}
        for key, group in training.groupby(["area_band", "construction_period"]):
            count = np.bincount(group.archetype.to_numpy(int)-1, weights=group._weight, minlength=4)
            lookup[key] = (count + 5*marginal) / (count.sum() + 5)
        datasets = [(f"{training_year}_household_holdout", test)]
        if training_year == 2013:
            datasets.append(("2013_to_2021_external_wave", frames[2021]))
        for evaluation, target in datasets:
            truth = np.eye(4)[target.archetype.to_numpy(int)-1]
            pred = np.array([lookup.get(key, marginal) for key in zip(target.area_band, target.construction_period)])
            baseline = np.tile(marginal, (len(target), 1))
            assert np.allclose(pred.sum(axis=1), 1) and np.all(pred > 0)
            for name, probs in [("area_and_construction_period", pred), ("training_marginal_only", baseline)]:
                brier = np.square(probs-truth).sum(axis=1)
                logloss = -(truth*np.log(probs)).sum(axis=1)
                scores.append({"evaluation": evaluation, "training_households": len(train), "test_households": len(target),
                    "model": name, "brier_score": float(np.average(brier, weights=target.weight)),
                    "log_loss": float(np.average(logloss, weights=target.weight))})
    return pd.DataFrame(scores)


def regression_audit(probabilities: pd.DataFrame, year: int) -> pd.DataFrame:
    coeff, fit = _fit_probability_models(probabilities)
    design = _probability_design(probabilities.building_age_years.to_numpy(), probabilities.dwelling_area_m2.to_numpy())
    masks = {1: [0, 1, 3, 4], 2: [0, 1, 2, 3], 3: [0, 1, 2, 3], 4: [0, 1, 3, 5]}
    rows = []
    # Compare current report fit, paper's sparse terms, and the <=140m2 visible domain.
    for scope in ["all_63_windows", "area_at_most_140"]:
        subset = np.ones(len(probabilities), bool) if scope == "all_63_windows" else probabilities.dwelling_area_m2.le(140).to_numpy()
        subset &= probabilities.sample_count.ge(5).to_numpy()
        for a in range(1, 5):
            y = probabilities[f"prob_archetype_{a}"].to_numpy(float)[subset]
            for method in ["paper_terms_OLS", "current_full_quadratic_WLS"]:
                x = design[subset][:, masks[a]] if method == "paper_terms_OLS" else design[subset]
                w = np.ones(len(y)) if method == "paper_terms_OLS" else np.sqrt(probabilities.sample_count.to_numpy(float)[subset])
                beta = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]
                pred = x @ beta
                sse = np.square(y - pred).sum()
                r2 = 1 - sse / np.square(y - y.mean()).sum()
                loo = []
                for i in range(len(y)):
                    keep = np.arange(len(y)) != i
                    b = np.linalg.lstsq(x[keep] * w[keep, None], y[keep] * w[keep], rcond=None)[0]
                    loo.append(float(x[i] @ b))
                rows.append({"year": year, "archetype": a, "scope": scope, "method": method,
                    "windows": len(y), "r2_unweighted": r2, "adjusted_r2_unweighted": 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]),
                    "rmse": float(np.sqrt(sse / len(y))), "leave_one_window_out_rmse": float(np.sqrt(np.mean((y-np.array(loo))**2))),
                    "predictions_outside_0_1": int(((pred < 0) | (pred > 1)).sum())})
    current_pred = fit[[f"predicted_{a}" for a in range(1, 5)]].to_numpy()
    print(f"{year}: current clipped probability sum max error {np.max(abs(current_pred.sum(axis=1)-1)):.6g}", flush=True)
    return pd.DataFrame(rows)


def paper_equation_audit() -> tuple[pd.DataFrame, dict]:
    # Column order follows Eq.3 exactly: intercept, age, area, age*area, age^2, area^2.
    equation = np.array([[.289,.692,0,0,-.372,-.084], [.323,-.523,.104,0,.269,0],
                         [.335,-.230,-.028,0,.127,0], [.033,.039,0,.014,.152,0]])
    table = np.array([[.29,.69,0,0,-.37,-.08], [.32,-.52,.10,0,.27,0],
                      [.34,-.23,-.03,0,.13,0], [.03,.04,0,.01,.02,0]])
    rows = []
    for age, area in [(100, 100), (40, 120), (20, 360), (140, 360)]:
        x = np.array([1,age/100,area/100,age*area/1e4,age**2/1e4,area**2/1e4])
        for name, matrix in [("Equation_3", equation), ("Table_4", table)]:
            p = matrix @ x
            rows.append({"source": name, "age": age, "area": area, **{f"prob_{a+1}": p[a] for a in range(4)}, "sum": p.sum()})
    # Table7's second-row -60% is inconsistent with the stated denominator.
    return pd.DataFrame(rows), {"iso_deviation_for_E5_pct": 100 * (1717-2895)/2895,
                              "paper_printed_iso_deviation_for_E5_pct": -60,
                              "eq3_prob4_age_squared": .152, "table4_prob4_age_squared": .02}


def recover_historical_probabilities(raw: pd.DataFrame, out: Path) -> dict:
    """Recover the published regressions from saved empirical counts, then try raw recovery.

    The archived September 2024 routine uses age grid 15:5:135, area grid
    40:24:352, and half windows 15 years / 40m2. This differs from Section 4.3.
    """
    import statsmodels.api as sm
    source = ROOT / "probablities_only.csv"
    saved = pd.read_csv(source)
    masks = {1:[0,1,3,4],2:[0,1,2,3],3:[0,1,2,3],4:[0,1,3,5]}
    paper_f = [648.9,571.0,282.8,219.7]
    paper_r2 = [.94,.93,.87,.84]
    def fit(frame, name):
        selected=frame[(frame.total_number>=5)&(frame.age<150)&(frame.area<150)].copy()
        design=_probability_design(selected.age.to_numpy(),selected.area.to_numpy())
        rows=[]
        for a,mask in masks.items():
            y=selected[f"prob{a-1}"].to_numpy()
            model=sm.OLS(y,design[:,mask]).fit()
            coefficients=np.zeros(6); coefficients[mask]=model.params
            selected[f"predicted_{a}"]=model.predict()
            rows.append({"input":name,"archetype":a,"windows":len(selected),"f_statistic":float(model.fvalue),
                "paper_f":paper_f[a-1],"f_matches_paper_rounding":abs(model.fvalue-paper_f[a-1])<.05,
                "adjusted_r2":float(model.rsquared_adj),"paper_adjusted_r2":paper_r2[a-1],
                "rmse":float(np.sqrt(np.mean(model.resid**2))),
                **{f"coefficient_{term}":v for term,v in zip(["intercept","age","area","age_squared","area_squared","age_area"],coefficients)}})
        return pd.DataFrame(rows),selected
    stats,pred=fit(saved,"saved_2013_intermediate")
    pred.to_csv(out/"historical_probability_predictions.csv",index=False)
    original=pd.read_csv(ROOT/"clusters.csv",usecols=["identi","segment"]).rename(columns={"identi":"id","segment":"archetype"})
    original.archetype+=1
    samples=build_dwelling_samples(raw,original,42)
    recovered=[]
    for area in np.arange(40,361,24):
        for age in np.arange(15,136,5):
            group=samples[samples.building_age_years.between(age-15,age+15)&samples.dwelling_area_m2.between(area-40,area+40)]
            if group.empty:
                continue
            recovered.append({"age":age,"area":area,"total_number":len(group),
                **{f"prob{a-1}":float(group.archetype.eq(a).mean()) for a in range(1,5)}})
    recovered=pd.DataFrame(recovered)
    recovered.to_csv(out/"historical_raw_window_recovery.csv",index=False)
    raw_stats,_=fit(recovered,"raw_2013_archived_windows_seed42_original_labels")
    pd.concat([stats,raw_stats],ignore_index=True).to_csv(out/"historical_regression_recovery.csv",index=False)
    aligned=saved.merge(recovered,on=["age","area"],validate="one_to_one",suffixes=("_saved","_raw"))
    return {"saved_table_sha256":sha256(source),"saved_cluster_sha256":sha256(ROOT/"clusters.csv"),
        "saved_windows":len(saved),"regression_windows":len(pred),"recovered_raw_windows":len(recovered),
        "paper_f_statistics_recovered_at_printed_precision":int(stats.f_matches_paper_rounding.sum()),
        "mean_absolute_probability_difference_raw_vs_saved":float(np.mean([abs(aligned[f"prob{i}_saved"]-aligned[f"prob{i}_raw"]).mean() for i in range(4)])),
        "historical_generator_git_ref":"e7094f4:resources/istat_data/classification_analysis_bayes.py",
        "historical_window_half_widths":[15,40],"historical_grid_steps":[5,24],
        "source_warning":"Archived routine predates final paper; raw recovery retains saved cluster labels and uses a new fixed dwelling seed. It is not an exact raw-to-paper reproduction."}


def regional_centroid_diagnostic(regional: pd.DataFrame, out: Path) -> None:
    from statsmodels.multivariate.manova import MANOVA
    columns=["pc1_scaled","pc2_scaled","education_scaled"]
    global_centers=regional.groupby("archetype")[columns].mean().to_numpy()
    rows=[]
    for region,group in regional.groupby("reg"):
        fitted=fit_kmeans(group[columns].to_numpy(),4,int(region)+1,300,1e-8)
        r,c=linear_sum_assignment(np.sum((global_centers[:,None]-fitted.centers[None])**2,axis=2))
        for label,center in zip(r,fitted.centers[c]):
            rows.append({"reg":region,"archetype":label+1,**dict(zip(columns,center))})
    centers=pd.DataFrame(rows)
    centers.to_csv(out/"regional_centroids_audit.csv",index=False)
    result=[]
    for formula,label in [("pc1_scaled + pc2_scaled + education_scaled ~ C(reg)","centroids_region_only"),
                          ("pc1_scaled + pc2_scaled + education_scaled ~ C(archetype) + C(reg)","centroids_region_adjusted_for_archetype")]:
        table=MANOVA.from_formula(formula,centers).mv_test().results["C(reg)"]["stat"].copy()
        table.insert(0,"model",label);result.append(table)
    pd.concat(result).to_csv(out/"regional_centroid_manova.csv")


def make_plots(out: Path, comparisons: pd.DataFrame, heating: pd.DataFrame, cells: pd.DataFrame,
               variants: pd.DataFrame, regressions: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "istat-replicability-audit"})
    def save(fig, name):
        fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, method, weighting, title in zip(axes, ["current_separate_fits", "common_fixed_2013"], ["sample", "survey"],
        ["Existing workflow: separate definitions", "Comparable coding + fixed 2013 model"]):
        for i, year in enumerate([2013, 2021]):
            s = comparisons.query("method == @method and year == @year and weighting == @weighting").sort_values("archetype")
            bars = ax.bar(np.arange(4) + (i-.5)*.34, s.household_share_pct, width=.34, color=COLORS[i], label=str(year))
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set(title=title, ylabel="Households (%)", xticks=np.arange(4), xticklabels=["A1", "A2", "A3", "A4"], ylim=(0, 55))
        ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.5, -.02, "Left: unweighted retained samples. Right: survey-weighted, 1-5 members; labels belong to each method's reference model.", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "archetype_comparison")

    fig, ax = plt.subplots(figsize=(8, 4.3))
    for i, year in enumerate([2013, 2021]):
        s = heating.query("year == @year and weighting == 'survey'").sort_values("archetype")
        bars=ax.bar(np.arange(5)+(i-.5)*.34, s.mean_daily_hours, width=.34, color=COLORS[i], label=str(year))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=np.arange(5), xticklabels=["All eligible", "A1", "A2", "A3", "A4"], ylabel="Declared hours per winter day", ylim=(0, max(heating.mean_daily_hours)+2), title="Heating: matched question eligibility and fixed archetypes")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.5, -.02, "Survey-weighted. Frequency categories 1/2 only; all three time bands observed. 2021 responses refer to winter 2020.", ha="center", fontsize=9)
    fig.tight_layout(); save(fig, "heating_comparison")

    a=cells[cells.year.eq(2013)].set_index(["area_band","construction_period"])
    b=cells[cells.year.eq(2021)].set_index(["area_band","construction_period"])
    change=(b[[f"prob_{i}" for i in range(1,5)]]-a[[f"prob_{i}" for i in range(1,5)]])*100
    adequate=a.meets_support_30 & b.meets_support_30
    fig, axes=plt.subplots(2,2,figsize=(11,7), layout="constrained")
    magnitude=max(10, float(abs(change.to_numpy()).max()))
    for i, ax in enumerate(axes.flat,1):
        grid=change[f"prob_{i}"].unstack("area_band").reindex(index=PERIOD_BANDS,columns=AREA_BANDS)
        image=ax.imshow(grid, vmin=-magnitude, vmax=magnitude, cmap="RdBu_r", aspect="auto")
        for r, period in enumerate(PERIOD_BANDS):
            for c, area in enumerate(AREA_BANDS):
                value=float(grid.loc[period,area]); ok=bool(adequate.loc[(area,period)])
                ax.text(c,r,f"{value:+.1f}"+("*" if not ok else ""), ha="center",va="center", fontsize=9,
                        color="white" if abs(value)>.65*magnitude else "black")
        ax.set(title=f"A{i}", xticks=np.arange(6),xticklabels=AREA_BANDS,yticks=np.arange(4),yticklabels=PERIOD_BANDS,xlabel="Dwelling floor-area band (m²)")
    fig.colorbar(image,ax=axes,label="Change in conditional probability (percentage points)",shrink=.8)
    fig.suptitle("2021 minus 2013: weighted probabilities in common dwelling categories\n* fewer than 30 records or effective observations in either wave", fontsize=12)
    save(fig,"dwelling_probability_changes")

    fig, ax=plt.subplots(figsize=(9,4.5))
    for i, (method, group) in enumerate(variants.groupby("scenario",sort=False)):
        ax.plot(np.arange(1,5), group.sort_values("archetype").household_share_pct, marker="o",label=method)
    ax.plot(np.arange(1,5),PAPER_SHARES, "k--", label="Paper Figure 5")
    ax.set(xticks=np.arange(1,5),xlabel="Cluster label after semantic ordering (definitions can change)",ylabel="Unweighted household share (%)",title="Same 2013 records: sensitivity to preprocessing")
    ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.2);fig.tight_layout();save(fig,"preprocessing_sensitivity")

    recovered=pd.read_csv(out/"historical_probability_predictions.csv")
    fig,ax=plt.subplots(figsize=(6,5))
    for a,color in enumerate(["#4164bc","#50b86c","#eea91b","#be3158"],1):
        ax.scatter(recovered[f"prob{a-1}"],recovered[f"predicted_{a}"],color=color,marker="x",s=15,label=f"A{a}")
    ax.plot([0,.7],[0,.7],"k--",lw=1)
    ax.set(xlabel="Original saved window proportion",ylabel="OLS prediction using printed terms",title="Recovery from the original 125-window subset",xlim=(0,.7),ylim=(0,.7))
    ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.text(.5,-.01,"A1-A3 reproduce Table 4 F-statistics. A4 does not. In-sample fit; windows overlap.",ha="center",fontsize=9)
    fig.tight_layout();save(fig,"historical_probability_recovery")

    s=pd.read_csv(out/"comparison_sensitivity.csv")
    p=s.pivot(index=["method","archetype"],columns="year",values="household_share_pct")
    delta=p[2021]-p[2013]
    fig,ax=plt.subplots(figsize=(8,4))
    for a in range(1,5):
        values=delta.xs(a,level="archetype")
        ax.plot([values.min(),values.max()],[a,a],color="#3965a6",lw=4)
        ax.plot(values,[a]*len(values),"|",color="#173c74",markersize=12)
        ax.plot(values.loc["midpoints"],a,"o",color="#d17a22",ms=8)
    ax.axvline(0,color="#666666",ls="--")
    ax.set(yticks=range(1,5),yticklabels=["A1","A2","A3","A4"],xlabel="2021 minus 2013 household share (percentage points)",title="Sensitivity of changes to age and education assumptions")
    ax.grid(axis="x",alpha=.2)
    fig.text(.5,-.01,"Blue: range across ten scenarios, not a confidence interval. Orange: midpoint reference. Scenario labels aligned on 2013 IDs.",ha="center",fontsize=8)
    fig.tight_layout();save(fig,"comparison_assumption_sensitivity")


def write_review(out: Path, evidence: dict) -> None:
    """Generate a review whose numeric tables always come from this audit run."""
    def table(frame: pd.DataFrame) -> str:
        def fmt(value):
            if isinstance(value, (float, np.floating)):
                return f"{value:.4g}"
            return str(value).replace("|", "/")
        lines=["| " + " | ".join(map(str,frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
        lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)]
        return "\n".join(lines)
    def link(name):
        return (out/name).as_posix()
    shares=pd.read_csv(out/"archetype_comparison.csv")
    fixed=shares[shares.method.eq("common_fixed_2013")&shares.weighting.eq("survey")]
    fixed=fixed.pivot(index="archetype",columns="year",values="household_share_pct")
    fixed["Change (percentage points)"]=fixed[2021]-fixed[2013]
    direct=pd.read_csv(out/"direct_survey_comparison.csv")
    direct=direct[direct.weighting.eq("survey")].pivot(index="measure",columns="year",values="value")
    heat=pd.read_csv(out/"heating_comparison.csv")
    heat=heat[heat.weighting.eq("survey")].pivot(index="archetype",columns="year",values="mean_daily_hours")
    heat["Change (hours)"]=heat[2021]-heat[2013]
    historical=pd.read_csv(out/"historical_regression_recovery.csv")
    historic_view=historical[historical.input.eq("saved_2013_intermediate")][["archetype","f_statistic","paper_f","adjusted_r2","paper_adjusted_r2","rmse"]]
    variation=pd.DataFrame([{ "Scenario":name,"Retained":v["retained"],"Paper checks / 40":v["paper_checks_passed"],
                              "Reassigned after best label alignment (%)":100-v["best_permutation_agreement_pct"]}
                             for name,v in evidence["preprocessing_sensitivity"].items()])
    sensitivity=pd.read_csv(out/"comparison_sensitivity.csv").pivot(index=["method","archetype"],columns="year",values="household_share_pct")
    sensitivity["delta"]=sensitivity[2021]-sensitivity[2013]
    sensitivity_range=sensitivity.groupby("archetype").delta.agg(["min","max"]).reset_index()
    scores=pd.read_csv(out/"heldout_probability_scores.csv")
    score_pairs=scores.pivot(index="evaluation",columns="model",values="brier_score")
    score_pairs["Brier improvement (%)"]=100*(1-score_pairs.area_and_construction_period/score_pairs.training_marginal_only)
    household_manova=pd.read_csv(out/"regional_household_manova.csv").rename(columns={"Unnamed: 0":"Statistic"}).iloc[[0,3]]
    centroid_manova=pd.read_csv(out/"regional_centroid_manova.csv").rename(columns={"Unnamed: 0":"Statistic"})
    centroid_manova=centroid_manova[centroid_manova.iloc[:,0].isin(["Wilks' lambda","Roy's greatest root"])]
    alignment=evidence["energy"]["validation"]["paper_alignment"]
    raw_recovery=evidence["historical_recovery"]
    pdf="C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main (1).pdf"
    text=f"""# Replicability audit and 2013/2021 comparison

The article is **partly numerically reproducible**, and its broad workflow can be rebuilt. It is not currently an exact, scientifically verified raw-data reproduction. The strongest positive evidence is that the saved original probability table reproduces the first three Table 4 regressions at the published F-statistic precision. The strongest concern is that the matching demographic reconstruction preserves confirmed age and education errors. Reproducing a faulty calculation and producing a defensible new analysis are different objectives.

This audit reran both raw releases and the 2013 energy stage in this folder's `reconstructed/` directory. It added a comparison with consistent coding, official survey weights, a common household-size restriction, and a frozen 2013 reference model. Existing production scripts, configurations, Word reports, and outputs were not overwritten.

## Evidence and provenance

- [Article, all 15 pages](<{pdf}>), especially Sections 3.1-3.3 and 4.1-4.4, Equations 1-3, Tables 2-7, and Figures 2-9.
- [Public code cited in the article](https://github.com/BETALAB-team/ita-domes/tree/1b5972b1e052f00a2fdd9fa30d46bb442205c83d/resources/istat_data/UserClusters/istat_data), inspected at commit `1b5972b1e052f00a2fdd9fa30d46bb442205c83d`. Its household and clustering scripts contain the same defects found locally; this is not evidence of the exact unpublished execution that created every paper result.
- Local git history: `e7094f4` (2024-10-03) contains `classification_analysis_bayes.py`, including the older empirical window routine. `e07cd69` is the 2025 analysis snapshot; `a129009` contains the 2026 reconstruction. Later local changes were already present before this audit.
- ISTAT's supplied 2013/2021 classification HTML files and questionnaires define the age, education, household-size, construction, floor-area, and heating-question mappings. The 2021 methodological note identifies the survey frame, mixed CAWI/CATI collection, final weights, and 2020 reference year for consumption.
- Input paths and SHA-256 hashes, assumptions, results, and output hashes are recorded in [audit_results.json]({link('audit_results.json')}) and [manifest.json]({link('manifest.json')}). No private survey records were sent to a remote service.

## What can be recovered

### Four archetypes

The current 2013 reconstruction retains 17,855 of 20,000 households and satisfies all 40 declared Table 3/Figure 5 tolerances. Its unweighted shares are 40.129%, 26.777%, 22.767%, and 10.328%, close to the paper's 40.1%, 26.9%, 22.9%, and 10.1%. This is approximate numerical recovery under legacy assumptions, not exact agreement or proof of methodological validity.

The effect of changing only preprocessing, while keeping the published component equations and the same k-means seed, is:

{table(variation)}

The age-only and both-recoding scenarios use the full published age intervals and an explicit upper limit of 100 for the open-ended 75+ group. Those scenarios retain the legacy class-code education denominator to isolate recoding effects. They are diagnostic alternatives, not a uniquely identified correction. Label agreement uses the best permutation of labels on shared household IDs.

![Preprocessing sensitivity]({link('preprocessing_sensitivity.svg')})

*Figure 1. Unweighted 2013 cluster shares under alternative preprocessing. The dashed line is the published Figure 5 reference; cluster membership and meaning may change between scenarios.*

### Probability regressions: a substantial recovery, with a clear discrepancy

The file `probablities_only.csv` contains 350 empirical windows. The saved analysis restricts it to `total_number >= 5`, `age < 150`, and `area < 150`, giving 125 observations on a 25-by-5 grid. Fitting the sparse terms printed in Table 4 gives:

{table(historic_view)}

The first three F-statistics reproduce the paper at its printed precision. For A4, the printed terms do not reproduce its coefficients or F-statistic. The corresponding recovered predictions are included separately; the fourth model has not been relabelled as an exact match.

The archived routine uses age centers spaced 5 years apart, area centers spaced 24 m² apart, and **half windows of 15 years and 40 m²**. The paper instead describes 20-year/40-m² bins and window widths of 6 years/8 m². The reconstruction adopted a 63-window grid with half widths 6 and 8, then fitted a sample-count-weighted full quadratic model. These are materially different datasets and regressions. My earlier claim that the reconstructed probability graphs replicated the paper was too strong.

Using the older grid directly with raw 2013 dwelling data, saved original cluster labels, and a new fixed random seed produces {raw_recovery['recovered_raw_windows']} windows. Its mean absolute probability difference from the saved table is {raw_recovery['mean_absolute_probability_difference_raw_vs_saved']:.4f}. That is a partial raw-data recovery, not an exact rerun: the source random realization is not recorded, and the original cluster assignments remain an input. Results are in [historical_regression_recovery.csv]({link('historical_regression_recovery.csv')}).

![Recovered original probability regressions]({link('historical_probability_recovery.svg')})

*Figure 2. Fits to the original saved 125-window subset. A1-A3 recover the paper's reported F-statistics. These are in-sample fits to overlapping, smoothed proportions, not independent household predictions; high R² is not evidence of equivalent out-of-sample predictive accuracy.*

### Energy analysis

The rerun agrees with {alignment['anova_significance_matches']} of {alignment['anova_terms_compared']} Table 5 significance decisions. It yields {alignment['heating_groups']} heating groups and {alignment['electricity_groups']} electricity groups. The paper reports seven heating groups, while its text says six electricity groups and Table 6 lists five. The current deterministic grouping is computationally reproducible but does not reproduce all published group definitions.

The electricity response is model-derived appliance kWh from `results_grins.csv`, not metered household consumption. This audit verifies its existing ID join; it does not regenerate MODENA's appliance estimates from the full external model/database chain. The 2021 data do not supply the same modeled response, so no 2013-versus-2021 kWh comparison has been manufactured.

## Problems in the original version and the reconstruction

| Layer | Finding | Consequence |
| --- | --- | --- |
| Original demographic code | In-place education recoding changes primary code 3 to 5, then changes all 5s to 12. Primary schooling is effectively assigned 12 years. | The error changes the education geometry and roughly 19% of assignments when corrected alone. The reconstruction explicitly preserved it. |
| Original age conversion | The 15-24 band is sampled only at 20-24; the under-15 class only at 11-14; the open 75+ band only at 80-84. | Systematic age distortion and falsely narrow ages, especially in the elderly cluster. |
| Education denominator and exclusions | The old denominator receives age-class codes rather than realized ages. The current reconstruction excludes 2,050 households for undefined education and 95 for zero members. | Sample selection is consequential. The 2021 path uses a different denominator and excludes 6+ households instead. |
| Original executable chain | Public `02_Families_Analysis.py` uses `.at[i]` on an ID-indexed frame starting at 1 and never writes the required `family_trees.csv`; `03_PCA_Clustering.py` requests 20 clusters and consumes that saved file. | The numbered files are not a clean, complete executable record of the four-cluster paper. Earlier scripts contain alternate paths, so these defects alone do not identify the exact run used for publication. |
| Naive Bayes replacement | `05_02_PFE_Probablity.py` writes `probabilities_only.csv` with held-out `Prob_*` columns, while the next script reads `probablities_only.csv` and expects age, area, counts, and `prob0`-`prob3`. | Both a filename mismatch and a changed statistical target broke the chain. |
| PCA and clustering | Published equations are hard-coded for clustering; the reported exploratory PCA is calculated separately. Seeds, full loadings, centering, and initialization details are not fully specified in the paper. | Applying the equations is reproducible; calling it a refit of the paper's exact PCA is not justified. In Figure 2, the final cumulative values shown for all listed components are below 100%, another item requiring the original plotting data. |
| Probability reconstruction | The 63-window grid differs from the recovered 350-window source; 12 of its 2013 windows and 9 of its 2021 windows have fewer than 5 observations. Algorithm 1's iterative categorical-variable removal and regrouping is absent. | It cannot substantiate the paper's variable-selection result or replicate its reported fit merely because its probabilities sum to one. |
| Probability outputs | The empirical count probabilities sum to one, but the Word plotting regression clips each fitted probability independently. | The 2013 fitted sums deviate from one by as much as 0.01334. Counts and regression outputs need separate validation. |
| Previous figure replica | The standards-comparison figure places energy-group boxes at evenly spaced positions on a dwelling-area axis. These positions are not observed group areas. | Visual similarity is not a data-faithful reproduction of a relationship with dwelling size. |
| Multi-year comparison | The existing runner changes education coding, household eligibility, min-max limits, and independently fits centers for each wave, then gives them the same labels. | Its concatenated summary cannot be read directly as temporal change in fixed archetypes. |
| Weighting | The paper explicitly reports unweighted sample proportions in Section 4.2. The two surveys have different frames and modes. | Unweighted changes do not establish national demographic change. Applying official weights helps but cannot fully remove mode/frame differences. |
| Inference | Overlapping probability windows reuse households. Repeated data-dependent Tukey merging uses the same response for selection and subsequent significance testing. | Nominal regression p-values and one-round Tukey family-wise guarantees do not validate the complete adaptive process. |

## Inconsistencies in the article itself

Equation 3 gives A4 an age-squared coefficient of 0.152, while Table 4 prints 0.02. At building age 100 years and area 100 m², Equation 3 gives probabilities `[0.525, 0.173, 0.204, 0.238]`, summing to **1.14**. At age 20 and area 360 m² it gives a negative A1 probability. The latter is outside Figure 6's displayed domain and demonstrates that the polynomial must not be extrapolated unrestrictedly.

Table 4's F-statistics can partly be recovered as shown above, but A4 remains unresolved. Figure 6's horizontal axis is dwelling area even though its caption describes age changing. Table 5's electricity interaction decisions and F-values conflict with the surrounding text; Table 6 repeats the same example for H6 and H7; the text says six electricity groups while the table lists five. Table 7 prints a -60% ISO deviation for E5, but `(1717-2895)/2895` is -40.69%. These are reporting inconsistencies, not discrepancies that code should conceal by tuning results.

### Regional effects depend on the unit of analysis

Applying the Section 3.1 description literally at the household level (`PC1 + PC2 + education ~ region`) gives:

{table(household_manova)}

For comparison, applying the same test to four fitted centroids per region, with and without conditioning on archetype, gives:

{table(centroid_manova)}

These are diagnostic, unweighted tests, not survey-design-adjusted national inference. They test different hypotheses. A non-significant test that pools four very different archetype centroids within each region does not establish that household distributions or archetype prevalences are geographically identical. The exact Table 2 input table is needed to settle its intended interpretation.

## Comparison of the two actual datasets

The releases contain 20,000 and 54,069 records. The common analysis retains 19,688 and 53,520 households of size 1-5, respectively. The 2021 release exposes only five members and top-codes the sixth size category as 6+. Official `coef_red` and `PESO` weights are retained. Coverage and exclusions are in [sample_comparison.csv]({link('sample_comparison.csv')}). Household IDs are local to each release; this is a repeated cross-section, not a panel of matched families.

### Direct observable comparisons

These survey-weighted quantities do not depend on randomized age values or fitted cluster boundaries. Percentages are conditional on the shared 1-5-member sample; qualification and construction measures use known-answer denominators, recorded in the CSV. Construction unknowns are reported separately, not assigned random ages. The open upper floor-area category has a small boundary ambiguity at exactly 150 m² in the 2013 label.

{table(direct.reset_index())}

These are descriptive differences, not causal estimates. The rise in unknown construction year affects the selection of households usable for dwelling-conditioned probabilities.

### Fixed-reference archetypes

The reference uses age-class midpoints; common education categories are assigned 5, 8, 12, and 18 years in both waves. No qualification and primary schooling must be combined because 2021 combines them; five years is an upper-bound assumption. Members with zero potential schooling are omitted from the education denominator. The 2013 unweighted model is chosen from ten starts by minimum inertia, then its scaling and centers are applied unchanged to 2021. There are 2,983 2021 households outside at least one 2013 min-max bound; their transformed values are retained without clipping.

{table(fixed.reset_index())}

The A1-A4 names here describe this new reference model, not guaranteed one-to-one replicas of the paper's archetypes. The existing workflow's A1 comparison is 40.13% to 25.68% (unweighted and independently refitted); this controlled example is 26.98% to 25.06% (weighted and fixed). The difference between those comparisons combines coding, eligibility, weighting, and model-definition effects and must not be attributed to one correction alone.

![Archetype comparison]({link('archetype_comparison.svg')})

*Figure 3. Existing unweighted separate fits versus the common weighted fixed-reference analysis. Each panel has its own model definitions.*

Across ten age/education scenarios, after aligning reference labels on shared 2013 IDs, the changes span:

{table(sensitivity_range)}

*These are scenario ranges in percentage points, not confidence intervals. A2 reverses direction between midpoints and random draws; A3 reverses under the open-age-cap scenario; A4 reverses under alternative no/primary education assumptions. Precise temporal archetype trends are not robust enough to claim from this exercise.*

![Assumption sensitivity]({link('comparison_assumption_sensitivity.svg')})

*Figure 4. Scenario sensitivity of the cross-wave share changes. Blue ranges are not sampling uncertainty intervals.*

### Dwelling probabilities and prediction across waves

The new comparison uses 24 common construction-period/floor-area cells. It uses observed category membership, rather than generating precise ages and areas from differently sized intervals. Each cell records sample count, effective sample size, four weighted probabilities, and a support flag. Construction periods hold building vintages comparable; the '1990+' group naturally also includes post-2013 buildings in 2021. A missing construction year is excluded from this conditional analysis.

![Dwelling-conditioned probability changes]({link('dwelling_probability_changes.svg')})

*Figure 5. Weighted changes in conditional archetype probability, 2021 minus 2013. Asterisks identify cells below 30 records or effective observations in either wave. All cells use the midpoint reference model; the age/education sensitivity still applies.*

As an independent prediction check, the conditional probability table is trained on 80% of eligible households and scored on the remaining 20%, then the 2013 table is transferred to 2021. Five average-weight pseudo-households shrink each cell towards the training marginal. Brier scores sum the squared errors of all four probabilities; lower is better. Survey weights are used in fitting cell proportions and evaluating scores.

{table(score_pairs.reset_index())}

This tests prediction from the common dwelling categories conditional on a frozen archetype definition. The clustering itself used the 2013 demographic sample, so this is not nested validation of the entire discovery pipeline. It is a simple probabilistic comparator, not a reproduction of Algorithm 1. A modest gain over the marginal baseline is more informative about practical predictive value than R² on overlapping fitted proportions.

### Heating hours

The previous runner skips all 2021 energy analysis, but the raw release contains `v3_8a`, `v3_8b`, and `v3_8c`, corresponding to the three eight-hour bands in 2013 `q_3_11A/B/C`. The routing differs: 2013 asks hours for frequency categories 1/2, 2021 for 1/2/3. This comparison restricts both to 1/2, with all three hours present and in 0-8. It has 18,332 and 49,545 observations. Row 0 below is the overall eligible group.

{table(heat.reset_index())}

![Heating comparison]({link('heating_comparison.svg')})

*Figure 6. Weighted mean winter-day operating hours for the matched frequent-use sample and fixed reference archetypes. These are neither annual heating energy nor means over every surveyed household. The 2021 answers explicitly reference winter 2020; pandemic conditions, survey mode/frame, climate, and equipment changes are not controlled.*

## Recommended next implementation

1. Preserve a documented legacy-reproduction mode, including the recovered probability intermediates and exact historical settings. Label paper matches and unresolved discrepancies separately.
2. Implement a scientifically revised preprocessing mode using the codebooks, explicit missing-value rules, shared education categories, configurable open-ended age assumptions, saved transformations, and stable reference centers. Do not silently change the legacy baseline.
3. Use valid joint probabilities and household-level validation for the dwelling model. Carry uncertainty from coarse inputs and sparse cells. Revisit the regional hypothesis and adaptive energy grouping with explicit analysis units and validation data.

The present comparison is a concrete audit and reproducible prototype of those choices. It does not claim that a single repaired implementation has been scientifically validated or that the paper's conclusions have all been replicated.

## Re-run and outputs

From `resources/istat_data`:

```powershell
.\\.venv\\Scripts\\python.exe audit_replicability.py
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

The runner accepts `--config` and `--output-dir`; all new results default to `replicability_outputs/`. Main tables include `archetype_comparison.csv`, `direct_survey_comparison.csv`, `heating_comparison.csv`, `dwelling_probability_comparison.csv`, `heldout_probability_scores.csv`, `historical_regression_recovery.csv`, `comparison_sensitivity.csv`, and `preprocessing_sensitivity.csv`. All six plots have SVG and PNG versions. This report is regenerated from the audit outputs.
"""
    (out/"REPLICABILITY_REVIEW.md").write_text(text,encoding="utf-8")


def run(config_path: Path, output: Path) -> dict:
    config=load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    waves, extras, reconstructed, probabilities = {}, {}, {}, {}
    evidence={"source_config":str(config_path),"production_pipeline_modified":False,"inputs":{},"current_runs":{}}
    for year in [2013,2021]:
        source=_resolve_raw_input(config,year)
        print(f"Reading {year} raw release",flush=True)
        waves[year],extras[year]=read_wave(source,year)
        evidence["inputs"][str(year)]={"path":str(source),"sha256":sha256(source),"raw_rows":len(waves[year])}
        year_out=output/"reconstructed"/str(year)
        year_out.mkdir(parents=True,exist_ok=True)
        harmonized=year_out/"harmonized_istat.csv"
        waves[year].to_csv(harmonized,sep=";",index=False)
        ar=run_archetypes(harmonized,year_out/"baseline",_archetype_config(config,year))
        pf=run_pfe(harmonized,year_out/"baseline"/"household_archetypes.csv",year_out/"pfe",_pfe_config(config,year))
        evidence["current_runs"][str(year)]={"archetypes_passed":ar["passed"],"paper_tolerance_matches":sum(c["passed"] for c in ar["paper_validation"]["checks"]),
             "rows":ar["row_counts"],"pfe_rows":pf["row_counts"]}
        reconstructed[year]=pd.read_csv(year_out/"baseline"/"household_archetypes.csv")
        probabilities[year]=pd.read_csv(year_out/"pfe"/"dwelling_archetype_probabilities.csv")
        print(f"Reconstructed {year}: {len(reconstructed[year])} households",flush=True)
    print("Re-running 2013 ANOVA/Tukey in isolated audit output",flush=True)
    energy=run_energy(output/"reconstructed/2013/harmonized_istat.csv",output/"reconstructed/2013/baseline/household_archetypes.csv",
                      config["_energy_results"],output/"reconstructed/2013/energy",EnergyConfig())
    evidence["energy"]=energy

    summaries=[]
    for year in [2013,2021]:
        summaries.append(summary(reconstructed[year].merge(extras[year],on="id",validate="one_to_one"),year,"current_separate_fits"))
    print("Building common definitions and fixed reference model",flush=True)
    common, exclusions={},{}
    for year in [2013,2021]:
        common[year],exclusions[year]=common_features(waves[year],year)
        exclusions[year].to_csv(output/f"common_exclusions_{year}.csv",index=False)
    model, assigned13=fit_reference(common[2013])
    write_json(output/"comparison_reference_model.json",model)
    assigned={2013:assigned13,2021:assign_reference(common[2021],model)}
    heating,cells,inventory=[],[],[]
    comparison_frames={}
    for year in [2013,2021]:
        frame=assigned[year].merge(extras[year],on="id",validate="one_to_one").merge(common_dwelling_groups(waves[year],year),on="id",validate="one_to_one")
        comparison_frames[year]=frame
        frame.to_csv(output/f"comparison_households_{year}.csv",index=False,float_format="%.12g")
        summaries.append(summary(frame,year,"common_fixed_2013"))
        heating.append(heating_summary(frame,year)); cells.append(dwelling_probabilities(frame,year))
        inventory.append({"year":year,"raw_households":len(waves[year]),"current_retained":len(reconstructed[year]),
             "common_retained":len(frame),"common_excluded":len(exclusions[year]),"weighted_population_coverage_pct":100*frame.weight.sum()/extras[year].weight.sum(),
             "heating_comparable_n":int(frame.heating_comparable.sum()),"unknown_construction_n":int(frame.construction_period.isna().sum()),
             "outside_reference_range_n":int(frame.outside_reference_range.sum())})
    comparisons=pd.concat(summaries,ignore_index=True); heating=pd.concat(heating,ignore_index=True); cells=pd.concat(cells,ignore_index=True)
    comparisons.to_csv(output/"archetype_comparison.csv",index=False)
    heating.to_csv(output/"heating_comparison.csv",index=False)
    cells.to_csv(output/"dwelling_probability_comparison.csv",index=False)
    pd.DataFrame(inventory).to_csv(output/"sample_comparison.csv",index=False)
    evidence["sample_comparison"]=inventory
    direct=pd.concat([direct_survey_summary(waves[y],extras[y],y) for y in [2013,2021]],ignore_index=True)
    direct.to_csv(output/"direct_survey_comparison.csv",index=False)
    probability_transfer_score(comparison_frames).to_csv(output/"heldout_probability_scores.csv",index=False)
    evidence["comparison_assumptions"]={"sample":"1-5 members; defined education among members with positive potential schooling",
        "age_reference":"codebook class midpoints, open age bands capped at 100; 2013 and 2021 retain different class boundaries",
        "education_years_common_categories":[5,8,12,18],"education_primary_warning":"no qualification and primary combined in BOTH waves; 5 is upper-bound assumption",
        "clustering":"2013 unweighted four-cluster best of ten starts; same feature scaling and centers applied to 2021 without clipping",
        "weights":"coef_red (2013), PESO (2021), used for conditional summaries, not cluster fitting",
        "dwelling":"six area bins x four construction-period bins; no invented continuous ages/areas; unknown construction excluded",
        "heating":"frequency categories 1/2 and all 3 eight-hour bands observed, both waves; not all-household mean",
        "limits":["different survey frames and CATI/CAWI modes", "2021 energy reference is 2020 (COVID period)",
                  "no directly comparable modeled appliance kWh for 2021", "point estimates; no survey-design confidence intervals"]}

    variants=[]; variant_evidence={}
    for name, age_ranges, years in [("Current legacy coding",None,dict(EDUCATION_YEARS)),
                                   ("Education recoding fixed",None,{**EDUCATION_YEARS,3:5}),
                                   ("Age intervals fixed",AGE_RANGES_2013,dict(EDUCATION_YEARS)),
                                   ("Both recodings fixed",AGE_RANGES_2013,{**EDUCATION_YEARS,3:5})]:
        f,ex=build_household_features(waves[2013],42,age_ranges=age_ranges,education_years=years)
        f=add_published_components(f); x=f[[f"{c}_scaled" for c in COMPONENTS]].to_numpy()
        k=fit_kmeans(x,4,1,300,1e-8); mapping=canonical_archetype_mapping(f,k.labels)
        f["archetype"]=pd.Series(k.labels).map(mapping)
        s=summarize_archetypes(f); s.insert(0,"scenario",name);variants.append(s)
        variant_evidence[name]={"retained":len(f),"paper_checks_passed":sum(c["passed"] for c in validate_against_paper(s)["checks"]),
            **aligned_agreement(reconstructed[2013],f)}
    variants=pd.concat(variants,ignore_index=True);variants.to_csv(output/"preprocessing_sensitivity.csv",index=False)
    evidence["preprocessing_sensitivity"]=variant_evidence

    print("Testing uncertain age and education assumptions",flush=True)
    sensitivity=[]
    scenarios=[("midpoints",None,5.,18.,100)]+[(f"age_draw_{s}",s,5.,18.,100) for s in [11,23,42,67,101]]+[
        ("no_primary_zero_years",None,0.,18.,100),("no_primary_2_5_years",None,2.5,18.,100),
        ("tertiary_16_years",None,5.,16.,100),("open_age_cap_90",None,5.,18.,90)]
    for name,seed,primary,tertiary,cap in scenarios:
        frames={year:common_features(waves[year],year,seed,primary,tertiary,cap)[0] for year in [2013,2021]}
        m,ref=(model,assigned13) if name=="midpoints" else fit_reference(frames[2013],range(5))
        # Align scenario reference clusters to main reference on shared IDs.
        j=assigned13[["id","archetype"]].merge(ref[["id","archetype"]],on="id",validate="one_to_one")
        ct=pd.crosstab(j.archetype_x,j.archetype_y).reindex(index=range(1,5),columns=range(1,5),fill_value=0).to_numpy()
        rr,cc=linear_sum_assignment(-ct);label_map={int(c+1):int(r+1) for r,c in zip(rr,cc)}
        for year in [2013,2021]:
            f=assign_reference(frames[year],m);f.archetype=f.archetype.map(label_map)
            s=summary(f.merge(extras[year],on="id",validate="one_to_one"),year,name)
            sensitivity.append(s[s.weighting.eq("survey")])
    pd.concat(sensitivity,ignore_index=True).to_csv(output/"comparison_sensitivity.csv",index=False)

    print("Auditing probability models and regional MANOVA",flush=True)
    regressions=pd.concat([regression_audit(probabilities[y],y) for y in [2013,2021]],ignore_index=True)
    regressions.to_csv(output/"probability_fit_audit.csv",index=False)
    eq,checks=paper_equation_audit();eq.to_csv(output/"paper_equation_checks.csv",index=False);evidence["paper_arithmetic"]=checks
    from statsmodels.multivariate.manova import MANOVA
    regional=reconstructed[2013].merge(waves[2013][["id","reg"]],on="id",validate="one_to_one")
    manova=MANOVA.from_formula("pc1_scaled + pc2_scaled + education_scaled ~ C(reg)",regional).mv_test().results["C(reg)"]["stat"]
    manova.to_csv(output/"regional_household_manova.csv")
    regional_centroid_diagnostic(regional,output)
    evidence["historical_recovery"]=recover_historical_probabilities(waves[2013],output)
    evidence["regional_manova_caution"]="Literal household-level model in Section 3.1; Table 2 may instead test regional centroids (different unit of analysis)."
    source_pfe=[]
    for seed in [11,23,42,67,101]:
        samples=build_dwelling_samples(waves[2013],reconstructed[2013],seed)
        pp,_,_=calculate_window_probabilities(samples,5);pp.insert(0,"seed",seed);source_pfe.append(pp)
    pd.concat(source_pfe,ignore_index=True).to_csv(output/"legacy_pfe_seed_sensitivity.csv",index=False)
    make_plots(output,comparisons,heating,cells,variants,regressions)
    evidence["checks"]={"common_shares_sum_to_100":bool(np.allclose(comparisons.groupby(["method","year","weighting"]).household_share_pct.sum(),100)),
         "common_cell_probabilities_sum_to_one":bool(np.allclose(cells[[f"prob_{i}" for i in range(1,5)]].sum(axis=1),1)),
         "comparison_model_frozen_to_2013":True,"both_raw_waves_rerun":True,"no_2021_kwh_fabricated":True}
    evidence["software"]={"numpy":np.__version__,"pandas":pd.__version__}
    write_json(output/"audit_results.json",evidence)
    write_review(output,evidence)
    write_json(output/"manifest.json",{"inputs":evidence["inputs"],"script_sha256":sha256(Path(__file__)),
        "outputs":{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="manifest.json"}})
    print(pd.DataFrame(inventory).to_string(index=False),flush=True)
    print(comparisons.query("method=='common_fixed_2013' and weighting=='survey'")[["year","archetype","household_share_pct"]].to_string(index=False),flush=True)
    print(f"Audit complete: {output}",flush=True)
    return evidence


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"multiyear_workflow_config.json")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"replicability_outputs")
    args=parser.parse_args()
    run(args.config.resolve(),args.output_dir.resolve())
