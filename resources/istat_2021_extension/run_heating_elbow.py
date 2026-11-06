"""IDE entry point: 2021 heating-group diminishing returns, without insulation.

Original sources: 01_PCA_Exploratory.py (k-means WCSS curve) and
07_ANOVA_analysis.py:476-517 (iterative Tukey reduction of factor cells).
This adapts the elbow idea to weighted heating cell means, not demographic PCA.
The exact 1-D k-means objective is solved by dynamic programming to avoid
random-start local minima. Household hours are NOT used to assign test groups.
"""
import json
import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from revised_analysis import ROOT, group_mean_variance
from revised_figures import plt, save, weighted_box
from energy_analysis import iterative_tukey_groups

YEAR = 2021
ELBOW_MAX_K = 12  # Display/selection range; sensitivity checks also use 9, 16, 24.
OUT = ROOT / 'revised_paper_outputs' / 'heating_elbow' / str(YEAR)
FACTORS = ['location', 'dwelling_type', 'archetype']


def exact_partitions(x, w, max_k):
    """Globally minimum weighted SSE partitions of sorted one-dimensional means."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    assert np.all(np.diff(x) >= 0) and np.all(w > 0)
    n = len(x)
    a, b, c = [np.r_[0., np.cumsum(v)] for v in [w, w*x, w*x*x]]
    cost = np.full((n, n+1), np.inf)
    for i in range(n):
        for j in range(i+1, n+1):
            cost[i, j] = max(0., c[j]-c[i]-(b[j]-b[i])**2/(a[j]-a[i]))
    dp = np.full((max_k+1, n+1), np.inf)
    prev = np.zeros_like(dp, dtype=int)
    dp[0, 0] = 0.
    partitions = {}
    for k in range(1, max_k+1):
        for j in range(k, n+1):
            candidates = dp[k-1, k-1:j] + cost[k-1:j, j]
            i = k-1 + int(np.argmin(candidates))
            dp[k, j], prev[k, j] = candidates.min(), i
        bounds, j = [], n
        for step in range(k, 0, -1):
            i = prev[step, j]
            bounds.append((i, j))
            j = i
        partitions[k] = list(reversed(bounds))
    return partitions, dp[1:, n]


def knee(curve, maximum):
    """Transparent elbow heuristic: largest gap below the normalized endpoint chord."""
    sub = curve[curve.k.le(maximum)]
    x = (sub.k.to_numpy()-1)/(maximum-1)
    y = sub.cell_wcss.to_numpy()
    if y[0] == y[-1]:
        return 1
    y = (y-y[-1])/(y[0]-y[-1])
    return int(sub.iloc[np.argmax(1-x-y)].k)


def main(outcome='heating', output=None):
    if outcome not in ('heating', 'electricity'):
        raise ValueError('Expected heating or electricity')
    OUT = output if output is not None else ROOT / 'revised_paper_outputs' / f'{outcome}_elbow' / str(YEAR)
    prefix = 'H' if outcome == 'heating' else 'E'
    ylabel = ('Daily heating-system on-time (hours)' if outcome == 'heating'
              else 'Annual appliance electricity (kWh)')
    units = 'hours/day' if outcome == 'heating' else 'kWh/year'
    color = '#ecd3b8' if outcome == 'heating' else '#abc2db'
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'revised_paper_outputs' / str(YEAR) / 'households.csv')
    frame = frame.drop(columns=['insulation', 'insulation_improvement'])
    frame = frame.dropna(subset=[outcome, *FACTORS]).copy()
    frame['cell'] = frame[FACTORS].astype(str).agg('|'.join, axis=1)
    train, tune, test = [frame[frame.split.eq(s)].copy() for s in ['train', 'tune', 'test']]
    cells = pd.DataFrame([{'cell': key, 'n': len(g), 'weight': g.weight.sum(),
                           'mean': np.average(g[outcome], weights=g.weight)}
                          for key, g in train.groupby('cell') if len(g) >= 30])
    cells = cells.sort_values(['mean', 'cell']).reset_index(drop=True)
    max_k = min(24, len(cells))
    partitions, sse = exact_partitions(cells['mean'], cells.weight, max_k)
    fallback = np.average(train[outcome], weights=train.weight)
    mappings, predictions, rows = {}, {}, []
    for k, bounds in partitions.items():
        mapping, estimates = {}, {}
        for label, (i, j) in enumerate(bounds, 1):
            block = cells.iloc[i:j]
            mean = np.average(block['mean'], weights=block.weight)
            for cell in block.cell:
                mapping[cell], estimates[cell] = f'{prefix}{label}', mean
        mappings[k], predictions[k] = mapping, estimates
        def mse(data):
            prediction = data.cell.map(estimates).fillna(fallback)
            return float(np.average((data[outcome]-prediction)**2, weights=data.weight))
        rows.append({'k': k, 'cell_wcss': sse[k-1],
                     'training_household_mse': mse(train), 'tuning_rmse': np.sqrt(mse(tune))})
    curve = pd.DataFrame(rows)
    curve['cell_variation_explained_percent'] = 100*(1-curve.cell_wcss/curve.cell_wcss.iloc[0])
    curve['additional_explained_percentage_points'] = curve.cell_variation_explained_percent.diff()
    curve.to_csv(OUT / 'elbow_curve.csv', index=False)
    sensitivity = {str(m): knee(curve, m) for m in [9, 12, 16, 24] if m <= max_k}
    chosen = knee(curve, min(ELBOW_MAX_K, max_k))
    cells['group'] = cells.cell.map(mappings[chosen])
    cells.to_csv(OUT / f'{outcome}_assignment.csv', index=False)
    test['group'] = test.cell.map(mappings[chosen]).fillna('Unsupported')
    test['prediction'] = test.cell.map(predictions[chosen]).fillna(fallback)
    test.to_csv(OUT / f'{outcome}_test.csv', index=False)
    summary = []
    for label, g in test[test.group.ne('Unsupported')].groupby('group'):
        mean, sd, var = group_mean_variance(g, outcome)
        summary.append({'group': label, 'n': len(g), 'mean': mean, 'sd': sd, 'se': np.sqrt(var)})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / f'{outcome}_groups.csv', index=False)
    pairs = []
    for a, b in itertools.combinations(summary.itertuples(), 2):
        diff, se = b.mean-a.mean, np.hypot(a.se, b.se)
        pairs.append({'group1': a.group, 'group2': b.group, 'difference': diff,
                      'p': 2*norm.sf(abs(diff/se))})
    pairs = pd.DataFrame(pairs)
    pairs['p_holm'] = multipletests(pairs.p, method='holm')[1]
    pairs.to_csv(OUT / f'{outcome}_robust_pairs.csv', index=False)
    # Original unweighted Tukey criterion, applied to training only; diagnostic,
    # not used to select the elbow or claim confirmatory training p-values.
    original_response = 'heating_daily_hours' if outcome == 'heating' else 'electricity_annual_kwh'
    original_factors = FACTORS if outcome == 'heating' else ['location', 'dwelling_type']
    reference = train.rename(columns={outcome: original_response})
    mapping, stats, history = iterative_tukey_groups(
        reference, original_response, original_factors, .05, prefix)
    mapping.to_csv(OUT / 'original_tukey_training_mapping.csv', index=False)
    stats.drop(columns=[c for c in stats if 'paper' in c]).to_csv(
        OUT / 'original_tukey_training_summary.csv', index=False)
    history.to_csv(OUT / 'original_tukey_training_merges.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    display = curve[curve.k.le(ELBOW_MAX_K)]
    axes[0].plot(display.k, display.cell_wcss/display.cell_wcss.iloc[0], 'o-', color='black')
    axes[0].set_ylabel('Remaining between-cell dispersion (fraction)')
    axes[1].bar(display.k.iloc[1:], display.additional_explained_percentage_points.iloc[1:], color='#daac81')
    axes[1].set_ylabel('Additional variation explained (percentage points)')
    axes[2].plot(display.k, display.tuning_rmse, 'o-', color='#233755')
    axes[2].set_ylabel(f'Tuning RMSE ({units})')
    for ax in axes:
        ax.axvline(chosen, color='#ab4c83', linestyle='--', label=f'Elbow candidate: {chosen}')
        ax.set_xlabel(f'Number of {outcome} groups'); ax.set_xticks(display.k)
    axes[0].legend()
    fig.tight_layout(); save(fig, OUT / 'figures', f'{outcome}_elbow')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    boxes = [weighted_box(g, outcome, label) for label, g in
             test[test.group.ne('Unsupported')].groupby('group')]
    artists = ax.bxp(boxes, patch_artist=True, showfliers=False)
    for box in artists['boxes']: box.set_facecolor(color)
    ax.set(xlabel=f'{outcome.capitalize()} group', ylabel=ylabel)
    save(fig, OUT / 'figures', f'{outcome}_elbow_groups')
    # Retain all range-sensitive elbow candidates for transparent comparison.
    candidate_ks = sorted(set(sensitivity.values()))
    fig, axes = plt.subplots(1, len(candidate_ks), figsize=(5*len(candidate_ks), 4.5),
                             sharey=True, squeeze=False)
    for ax, k in zip(axes[0], candidate_ks):
        candidate = test.copy()
        candidate['group'] = candidate.cell.map(mappings[k]).fillna('Unsupported')
        mapping = cells.drop(columns='group').copy()
        mapping['group'] = mapping.cell.map(mappings[k])
        mapping.to_csv(OUT / f'candidate_{k}_assignment.csv', index=False)
        boxes = [weighted_box(g, outcome, label) for label, g in
                 candidate[candidate.group.ne('Unsupported')].groupby('group')]
        artists = ax.bxp(boxes, patch_artist=True, showfliers=False)
        for box in artists['boxes']: box.set_facecolor(color)
        ax.set(xlabel=f'{outcome.capitalize()} group', title=f'{k} groups')
    axes[0, 0].set_ylabel(ylabel)
    fig.tight_layout(); save(fig, OUT / 'figures', f'{outcome}_candidate_comparison')
    results = {'year': YEAR, 'outcome': outcome, 'grouping_factors': FACTORS,
               'original_tukey_factors': original_factors,
               'selected_k': chosen, 'range_sensitivity': sensitivity,
               'training_cells': len(cells), 'original_tukey_training_groups': len(stats),
               'test_n': len(test), 'unsupported_test': int(test.group.eq('Unsupported').sum()),
               'test_rmse': float(np.sqrt(np.average((test[outcome]-test.prediction)**2, weights=test.weight))),
               'selection': 'Maximum normalized chord gap on training weighted cell-mean WCSS, k=1..12',
               'caution': 'Exploratory elbow candidate, not proof of a unique optimum. Cell variance explained is not household variance explained. Pairwise tests are model-based, not a full survey-design correction.'}
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    main()
