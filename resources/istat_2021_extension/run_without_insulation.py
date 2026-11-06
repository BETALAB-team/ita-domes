"""Run in an IDE or with .venv/Scripts/python.exe run_without_insulation.py.

Uses saved 2021 household features and unchanged train/tune/test splits.
Original results are never overwritten. No insulation variable or missingness
restriction enters the regression; grouping already excludes insulation.
"""
import json
from pathlib import Path

import pandas as pd

from revised_analysis import ROOT, energy_anova, learned_groups
from revised_figures import plt, save, weighted_box

YEAR = 2021
OUTPUT = ROOT / 'revised_paper_outputs' / 'no_insulation' / str(YEAR)


def main():
    source = ROOT / 'revised_paper_outputs' / str(YEAR)
    frame = pd.read_csv(source / 'households.csv')
    # Drop the columns entirely: a guard against accidental downstream use.
    frame = frame.drop(columns=['insulation', 'insulation_improvement'])
    cfg = json.loads((ROOT / 'revised_paper_config.json').read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {'year': YEAR, 'include_insulation': False,
               'grouping_inputs': ['location', 'dwelling_type', 'archetype']}
    for outcome in ['heating', 'electricity']:
        grouped = learned_groups(frame, outcome, cfg, OUTPUT)
        anova = energy_anova(frame, outcome, YEAR, OUTPUT, include_insulation=False)
        old = pd.read_csv(source / f'{outcome}_assignment.csv').set_index('cell')['group'].sort_index()
        new = pd.read_csv(OUTPUT / f'{outcome}_assignment.csv').set_index('cell')['group'].sort_index()
        results[outcome] = {'grouping': grouped, 'anova': anova,
                            'assignments_unchanged': old.equals(new)}
        test = pd.read_csv(OUTPUT / f'{outcome}_test.csv')
        fig, ax = plt.subplots(figsize=(9, 4.8))
        boxes = [weighted_box(g, outcome, label)
                 for label, g in test[test.group.ne('Unsupported')].groupby('group')]
        artists = ax.bxp(boxes, showfliers=False, patch_artist=True)
        for box in artists['boxes']:
            box.set_facecolor('#ecd3b8' if outcome == 'heating' else '#abc2db')
        ax.set_ylabel('Heating hours/day' if outcome == 'heating' else 'Appliance electricity (kWh/year)')
        ax.set_xlabel('Heating group' if outcome == 'heating' else 'Electricity group')
        save(fig, OUTPUT / 'figures', f'{outcome}_without_insulation')
    tuning = pd.read_csv(OUTPUT / 'heating_group_tuning.csv')
    best = tuning.tune_rmse.min()
    tuning['rmse_above_best_percent'] = 100 * (tuning.tune_rmse / best - 1)
    tuning.to_csv(OUTPUT / 'heating_complexity_comparison.csv', index=False)
    results['interpretation'] = (
        'Groups optimize tuning prediction, not pairwise statistical separation. '
        'No groups were merged using test-set p-values. Similar H7/H8 test '
        'distributions are evidence against interpreting every label as a distinct '
        'heating regime. Fewer groups require a separate complexity-selection rule.')
    (OUTPUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
