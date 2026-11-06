"""Corrected, split-sample analysis for the paired ISTAT manuscripts.

This module never changes the legacy workflow or its outputs. Run through
run_revised_papers.py, or run this file for analysis/figures only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linear_sum_assignment
from scipy.special import logsumexp, softmax
from scipy.stats import norm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from audit_replicability import (
    AGE_RANGES_2013, AGE_RANGES_2021, EDUCATION_GROUPS, FEATURES,
    common_dwelling_groups, read_wave, sha256, write_json,
)
from run_multiyear_workflow import load_config, _resolve_raw_input

ROOT = Path(__file__).resolve().parent
DEMOGRAPHIC = ['agemin', 'agemax', 'nmembers']
AREA_CENTERS = {'<40': 20., '40-59': 49.5, '60-89': 74.5,
                '90-119': 104.5, '120-149': 134.5, '150+': 200.}
YEAR_CENTERS = {'before 1950': 1924.5, '1950-1969': 1959.5,
                '1970-1989': 1979.5, '1990+': 2001.5}
LOCATIONS = ['North-west', 'North-east', 'Centre', 'South/islands']
TYPES = ['Apartment', 'Multi-family', 'Single-family']
TERMS = ['Intercept', 'Age/100', 'Area/100', 'Age squared/10000',
         'Area squared/10000', 'Age x area/10000']


def weighted_mean_sd(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    mean = np.average(values, weights=weights)
    return float(mean), float(np.sqrt(np.average((values-mean)**2, weights=weights)))


def effective_n(weights):
    w = np.asarray(weights, float)
    return float(w.sum()**2/(w@w))


def features_from_codes(raw, year, cfg, draw_seed=None):
    """Ages are interval proxies; schooling is a bounded attainment proxy, not UNDP."""
    ranges = dict(AGE_RANGES_2013 if year == 2013 else AGE_RANGES_2021)
    last = 75 if year == 2013 else 11
    ranges[last] = (ranges[last][0], cfg['age_cap'])
    study = [cfg['no_or_primary_years'], 8., 12., cfg['tertiary_years']]
    rng = np.random.default_rng(draw_seed) if draw_seed is not None else None
    rows, exclusions = [], []
    for record in raw.to_dict('records'):
        household_id, count = int(record['id']), int(record['q_1_1_sq1'])
        reason = None
        if not 1 <= count <= 5:
            reason = 'outside_common_1_to_5_members'
        ages, education = [], []
        if reason is None:
            for i in range(1, count+1):
                code, edu = int(record[f'cletad_{i}']), int(record[f'titstu_{i}'])
                if code not in ranges:
                    reason = 'invalid_age_code'; break
                lo, hi = ranges[code]
                age = (lo+hi)/2 if rng is None else float(rng.integers(lo, hi+1))
                ages.append(age)
                # Use the known lower bound, not a randomly imputed adult/child status.
                if lo >= 15:
                    if edu not in EDUCATION_GROUPS[year]:
                        reason = 'missing_education_for_member_15_plus'; break
                    education.append(study[EDUCATION_GROUPS[year][edu]]/18.)
        if reason is None and not education:
            reason = 'no_member_15_plus_with_education'
        if reason:
            exclusions.append({'id': household_id, 'reason': reason}); continue
        rows.append({'id': household_id, 'agemin': min(ages), 'agemax': max(ages),
                     'nmembers': count, 'education': float(np.mean(education))})
    result = pd.DataFrame(rows, columns=['id', *FEATURES]).astype({'id':int, **{name:float for name in FEATURES}})
    assert result.id.is_unique and np.isfinite(result[FEATURES]).all().all()
    return result, pd.DataFrame(exclusions, columns=['id', 'reason'])


def assign_splits(ids, cfg, year):
    """Split once, before fitting anything. Stable for the sorted release IDs."""
    order = np.random.default_rng(cfg['seed']+year).permutation(len(ids))
    ntrain, ntune = int(len(ids)*cfg['train_fraction']), int(len(ids)*cfg['tune_fraction'])
    split = np.full(len(ids), 'test', dtype=object)
    split[order[:ntrain]] = 'train'; split[order[ntrain:ntrain+ntune]] = 'tune'
    return split


def fit_pca(matrix, weights):
    matrix, weights = np.asarray(matrix, float), np.asarray(weights, float)
    mean = np.average(matrix, axis=0, weights=weights)
    scale = np.sqrt(np.average((matrix-mean)**2, axis=0, weights=weights))
    if np.any(scale <= 0):
        raise ValueError('Constant PCA input')
    z = (matrix-mean)/scale
    covariance = (z*weights[:, None]).T@z/weights.sum()
    val, vec = np.linalg.eigh(covariance)
    order = np.argsort(val)[::-1]; val, vec = val[order], vec[:, order]
    for j in range(vec.shape[1]):
        if vec[np.argmax(abs(vec[:, j])), j] < 0: vec[:, j] *= -1
    return {'mean': mean.tolist(), 'scale': scale.tolist(),
            'loadings': vec.T.tolist(), 'explained_ratio': (val/val.sum()).tolist()}


def pca_scores(matrix, pca):
    return ((np.asarray(matrix)-pca['mean'])/pca['scale'])@np.array(pca['loadings']).T


def weighted_kmeans(matrix, weights, k, seed, starts=1):
    x, w = np.asarray(matrix, float), np.asarray(weights, float)
    best = None
    for start in range(starts):
        rng = np.random.default_rng(seed+start)
        centers = [x[rng.choice(len(x), p=w/w.sum())]]
        while len(centers) < k:
            dist = ((x[:, None]-np.array(centers)[None])**2).sum(2).min(1)
            sampling = w*dist
            if sampling.sum() == 0: raise ValueError('Too few distinct observations')
            centers.append(x[rng.choice(len(x), p=sampling/sampling.sum())])
        centers = np.array(centers)
        for iteration in range(500):
            distances = ((x[:, None]-centers[None])**2).sum(2)
            label = distances.argmin(1)
            updated = np.array([np.average(x[label==a], axis=0, weights=w[label==a])
                                if np.any(label==a) else x[np.argmax(w*distances.min(1))]
                                for a in range(k)])
            shift = ((updated-centers)**2).sum(); centers = updated
            if shift < 1e-12: break
        else: raise RuntimeError('Weighted k-means failed to converge')
        distances = ((x[:, None]-centers[None])**2).sum(2)
        label = distances.argmin(1); inertia = float(w@distances.min(1)/w.sum())
        if best is None or inertia < best['inertia']:
            best = {'centers': centers, 'labels': label, 'inertia': inertia,
                    'iterations': iteration+1, 'seed': seed+start}
    return best


def cluster_matrix(features, model):
    pcs = pca_scores(features[DEMOGRAPHIC], model['pca'])[:, :2]
    z = np.column_stack([pcs, features.education])
    return (z-model['minimum'])/model['span']


def fit_demographics(train, cfg):
    pca = fit_pca(train[DEMOGRAPHIC], train.weight)
    pcs = pca_scores(train[DEMOGRAPHIC], pca)[:, :2]
    components = np.column_stack([pcs, train.education])
    lo, span = components.min(0), np.ptp(components, axis=0)
    model = {'pca': pca, 'minimum': lo.tolist(), 'span': span.tolist()}
    x = cluster_matrix(train, model)
    result = weighted_kmeans(x, train.weight, 4, cfg['seed'], cfg['kmeans_starts'])
    descriptions = []
    for k in range(4):
        g = train[result['labels']==k]
        descriptions.append([k, np.average(g.nmembers, weights=g.weight),
                             np.average(g.education, weights=g.weight),
                             np.average(g.agemin, weights=g.weight)])
    # Semantic ordering is frozen from 2013 training data, not recreated in 2021.
    remaining = list(descriptions)
    a2 = max(remaining, key=lambda r:(r[1],-r[0])); remaining.remove(a2)
    a4 = max(remaining, key=lambda r:(r[2],-r[0])); remaining.remove(a4)
    a1 = max(remaining, key=lambda r:(r[3],-r[0])); remaining.remove(a1)
    ordered = [a1[0], a2[0], remaining[0][0], a4[0]]
    model.update(centers=result['centers'][ordered].tolist(), inertia=result['inertia'],
                 selected_seed=result['seed'], training_n=len(train), training_year=2013)
    return model


def assign_archetypes(frame, model):
    x = cluster_matrix(frame, model)
    out = frame.copy()
    out['archetype'] = ((x[:, None]-np.array(model['centers'])[None])**2).sum(2).argmin(1)+1
    out[['PC1', 'PC2']] = pca_scores(frame[DEMOGRAPHIC], model['pca'])[:, :2]
    out['outside_reference_range'] = ((x < 0)|(x > 1)).any(1)
    return out


def supplement(raw_path, year):
    if year == 2013:
        ident, typ, freq = 'nprog', 'q_2_1', 'q_5_7'
        cool = [f'q_2_37{c}' for c in 'ABCDEF']
        extra = ['q_3_17']
    else:
        ident, typ, freq = 'ID_RECORD', 'v1_1', 'v5_6'
        cool = [f'v2_11{c}' for c in 'abcde']
        extra = ['v3_13', 'v3_14gh_new', 'v5_7a', 'v5_7b', 'v5_7c',
                 'v2_11conta', 'v5_0', 'v5_4_mfr']
    source = pd.read_csv(raw_path, sep='\t' if year==2013 else '^',
                         usecols=[ident, typ, freq, *cool, *extra], dtype=str)
    source = source.apply(lambda c:pd.to_numeric(c.str.strip(), errors='coerce'))
    out = pd.DataFrame({'id': source[ident].astype(int),
                        'dwelling_type': source[typ].map({1:'Single-family',2:'Multi-family',
                                                        3:'Apartment',4:'Apartment',5:'Apartment'})})
    # Apartment floor/building-size classifications are not interchangeable.
    present = source[cool].eq(1).any(axis=1)
    # In the 2013 questionnaire, all inventory items are skipped after Q2.36=No.
    # Keep a separate count of this routed absence in the released-data audit.
    routed_absence = (source[cool].isna().all(axis=1) & source[freq].isna()
                      if year == 2013 else pd.Series(False, index=source.index))
    absent = source[cool].eq(2).all(axis=1) | routed_absence
    out['cooling_routed_absence'] = routed_absence
    out['cooling'] = np.select([present, absent], ['Present','Absent'], default='Unknown')
    if year == 2013:
        out['insulation'] = source.q_3_17.map({1:'High/sufficient',2:'High/sufficient',3:'Low/draughty',4:'Low/draughty'}).fillna('Unknown')
    else:
        out['insulation'] = 'Unavailable'  # Subjective quality rating, not renovation.
        out['insulation_improvement'] = np.select(
            [source.v3_13.eq(1) & source.v3_14gh_new.eq(1),
             source.v3_13.eq(2) | (source.v3_13.eq(1) & source.v3_14gh_new.isna())],
            ['Reported', 'Not reported'], default='Unknown')
        out['cooling_frequency'] = source[freq]
        hours = source[['v5_7a', 'v5_7b', 'v5_7c']]
        valid = hours.ge(0).all(axis=1) & hours.le(8).all(axis=1)
        out['cooling_hours'] = hours.sum(axis=1, min_count=3).where(
            present & source[freq].isin([1, 2, 3]) & valid)
        out['cooling_type_count'] = source.v2_11conta
        out['cooling_main_type'] = source.v5_0
        out['cooling_age_class'] = source.v5_4_mfr
    return out


def make_frame(raw, extra, path, year, cfg):
    feature, excluded = features_from_codes(raw, year, cfg)
    frame = feature.merge(extra, on='id', validate='one_to_one')
    frame = frame.merge(common_dwelling_groups(raw, year), on='id', validate='one_to_one')
    frame = frame.merge(supplement(path, year), on='id', validate='one_to_one')
    regmap = {r:('North-west' if r in [1,2,3,7] else 'North-east' if r in [4,5,6,8]
                else 'Centre' if r in [9,10,11,12] else 'South/islands') for r in range(1,21)}
    frame['location'] = frame.region.map(regmap)
    frame['area_proxy'] = frame.area_band.map(AREA_CENTERS)
    # Closed-period midpoints; the upper period ends at each survey year.
    anchors = {**YEAR_CENTERS, '1990+':(1990+year)/2}
    frame['age_proxy'] = year-frame.construction_period.map(anchors)
    frame['split'] = assign_splits(frame.id, cfg, year)
    if year == 2013:
        electricity = pd.read_csv(ROOT/'results_grins.csv', usecols=['id','appliances_el_kWh'])
        assert electricity.id.is_unique
        frame = frame.merge(electricity, on='id', how='left', validate='one_to_one')
        frame['electricity'] = frame.appliances_el_kWh.where(frame.appliances_el_kWh.gt(0))
    else:
        electricity = pd.read_csv(ROOT/'revised_paper_outputs/appliance_audit/electricity_2021.csv')
        frame = frame.merge(electricity[['id','appliances_el_kWh']],on='id',how='left',validate='one_to_one')
        frame['electricity'] = frame.appliances_el_kWh.where(frame.appliances_el_kWh.gt(0))
    frame['heating'] = frame.heating_hours.where(frame.heating_comparable)
    return frame, excluded


def probability_design(frame, extended=False):
    a, s = frame.age_proxy.to_numpy(float)/100, frame.area_proxy.to_numpy(float)/100
    x = np.column_stack([np.ones(len(frame)), a, s, a*a, s*s, a*s])
    names = TERMS.copy()
    if extended:
        # Explicit common categories only: no 2013-only insulation or unknown-as-zero.
        for column, levels in [('location', LOCATIONS), ('dwelling_type', TYPES)]:
            for level in levels[1:]:
                x = np.column_stack([x, frame[column].eq(level).astype(float)])
                names.append(f'{column}: {level}')
    return x, names


def fit_softmax(x, y, weights, penalty):
    """Case-weighted multinomial likelihood, aggregated only for numerical speed."""
    x, y, w = np.asarray(x,float), np.asarray(y,int), np.asarray(weights,float)
    ux, inv = np.unique(x, axis=0, return_inverse=True)
    counts = np.zeros((len(ux),4)); np.add.at(counts, (inv,y-1), w/w.sum())
    total = counts.sum(1)
    mask = np.ones((x.shape[1],3)); mask[0] = 0
    def objective(theta):
        beta = theta.reshape(x.shape[1],3)
        logits = np.column_stack([ux@beta, np.zeros(len(ux))])
        probs = softmax(logits,axis=1)
        loss = -(counts*(logits-logsumexp(logits,axis=1)[:,None])).sum()
        loss += penalty/2*np.square(beta*mask).sum()
        gradient = ux.T@(probs[:,:3]*total[:,None]-counts[:,:3])+penalty*beta*mask
        return loss, gradient.ravel()
    fit = minimize(objective, np.zeros(x.shape[1]*3), jac=True, method='L-BFGS-B',
                   options={'maxiter':3000, 'ftol':1e-12, 'gtol':1e-8})
    if not fit.success: raise RuntimeError(f'Multinomial fit failed: {fit.message}')
    beta = np.column_stack([fit.x.reshape(x.shape[1],3), np.zeros(x.shape[1])])
    return beta


def score_probabilities(frame, pred):
    truth = np.eye(4)[frame.archetype.to_numpy(int)-1]
    return {'n':len(frame), 'brier':float(np.average(np.square(pred-truth).sum(1),weights=frame.weight)),
            'log_loss':float(np.average(-np.log(pred[np.arange(len(pred)),frame.archetype.to_numpy(int)-1]),weights=frame.weight))}


def probability_analysis(frame, cfg, out):
    f = frame.dropna(subset=['age_proxy','area_proxy','dwelling_type','location']).copy()
    train, tune, test = [f[f.split.eq(s)] for s in ['train','tune','test']]
    marginal = np.bincount(train.archetype.to_numpy(int)-1, weights=train.weight, minlength=4)
    marginal /= marginal.sum()
    scores, choices, models = [], [], {}
    for extended, name in [(False,'quadratic'),(True,'quadratic_plus_location_type')]:
        xtrain, names = probability_design(train, extended)
        xtune, _ = probability_design(tune, extended)
        candidates = []
        for penalty in cfg['probability_penalties']:
            b = fit_softmax(xtrain, train.archetype, train.weight, penalty)
            metrics = score_probabilities(tune, softmax(xtune@b,axis=1))
            choices.append({'model':name,'penalty':penalty,**metrics})
            candidates.append((metrics['log_loss'],penalty,b))
        _, penalty, beta = min(candidates,key=lambda row:row[0])
        models[name] = {'beta':beta.tolist(),'penalty':penalty,'names':names}
        xtest,_ = probability_design(test, extended)
        scores.append({'model':name,**score_probabilities(test,softmax(xtest@beta,axis=1))})
    scores.append({'model':'training_marginal',**score_probabilities(test,np.tile(marginal,(len(test),1)))})
    beta = np.array(models['quadratic']['beta'])
    xall,_ = probability_design(f); pred = softmax(xall@beta,axis=1)
    for a in range(4): f[f'prob_{a+1}'] = pred[:,a]
    assert np.allclose(pred.sum(1),1,atol=1e-12) and np.all((pred>0)&(pred<1))
    f.to_csv(out/'probability_households.csv',index=False)
    cells = []
    for split in ['train','tune','test','all']:
        sub = f if split=='all' else f[f.split.eq(split)]
        for key,g in sub.groupby(['construction_period','area_band']):
            row = {'split':split,'period':key[0],'area_band':key[1],
                   'age_proxy':g.age_proxy.iloc[0],'area_proxy':g.area_proxy.iloc[0],
                   'n':len(g),'weight_sum':float(g.weight.sum()),'effective_n':effective_n(g.weight)}
            row['supported'] = len(g)>=cfg['minimum_cell_count'] and row['effective_n']>=cfg['minimum_cell_count']
            for a in range(1,5):
                row[f'count_{a}'] = int(g.archetype.eq(a).sum())
                row[f'weight_{a}'] = float(g.loc[g.archetype.eq(a),'weight'].sum())
                row[f'observed_{a}'] = float(np.average(g.archetype.eq(a),weights=g.weight))
                row[f'predicted_{a}'] = float(np.average(g[f'prob_{a}'],weights=g.weight))
            cells.append(row)
    pd.DataFrame(cells).to_csv(out/'probability_cells.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'probability_scores.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'probability_tuning.csv',index=False)
    pd.DataFrame(beta,index=TERMS,columns=['A1','A2','A3','A4']).rename_axis('term').to_csv(out/'probability_coefficients.csv')
    models['marginal'] = marginal.tolist()
    write_json(out/'probability_models.json',models)
    return {'eligible':len(f),'train':len(train),'tune':len(tune),'test':len(test),
            'scores':scores,'penalty':models['quadratic']['penalty'], 'models':models}


def group_mean_variance(g, outcome):
    mean, sd = weighted_mean_sd(g[outcome],g.weight)
    w = g.weight.to_numpy(float); residual = g[outcome].to_numpy(float)-mean
    variance = float(np.sum(w*w*residual**2)/w.sum()**2*len(g)/max(len(g)-1,1))
    return mean, sd, variance


def learned_groups(frame, outcome, cfg, out):
    """Train outcome compression; select its complexity on tune; test once."""
    data = frame.dropna(subset=[outcome,'dwelling_type','location']).copy()
    train, tune, test = [data[data.split.eq(s)].copy() for s in ['train','tune','test']]
    keys = ['location','dwelling_type','archetype']
    train['cell'] = train[keys].astype(str).agg('|'.join,axis=1)
    tune['cell'] = tune[keys].astype(str).agg('|'.join,axis=1)
    test['cell'] = test[keys].astype(str).agg('|'.join,axis=1)
    stats = []
    for key,g in train.groupby('cell'):
        if len(g)<cfg['group_minimum_train_count']: continue
        stats.append({'cell':key,'n':len(g),'weight':g.weight.sum(),
                      'mean':np.average(g[outcome],weights=g.weight)})
    stats = pd.DataFrame(stats).sort_values('mean').reset_index(drop=True)
    groups = [{r.cell} for r in stats.itertuples()]
    lookup_stats = stats.set_index('cell')
    def group_stats(group):
        rows = lookup_stats.loc[sorted(group)]
        return float(rows.weight.sum()), float(np.average(rows['mean'],weights=rows.weight))
    hierarchy = {}
    while groups:
        if len(groups)<=cfg['group_maximum_count']:
            hierarchy[len(groups)] = [set(g) for g in groups]
        if len(groups)==1: break
        values = [group_stats(g) for g in groups]
        costs = [values[i][0]*values[i+1][0]/(values[i][0]+values[i+1][0])*(values[i][1]-values[i+1][1])**2 for i in range(len(groups)-1)]
        j = int(np.argmin(costs)); groups[j] |= groups[j+1]; groups.pop(j+1)
    baseline = float(np.average(train[outcome],weights=train.weight))
    candidates = []
    for k,gs in hierarchy.items():
        estimates = {cell:group_stats(group)[1] for group in gs for cell in group}
        pred = tune.cell.map(estimates).fillna(baseline)
        rmse = float(np.sqrt(np.average((tune[outcome]-pred)**2,weights=tune.weight)))
        candidates.append({'groups':k,'tune_rmse':rmse})
    chosen = min(candidates,key=lambda r:(r['tune_rmse'],r['groups']))['groups']
    prefix = 'H' if outcome=='heating' else 'E'
    groups = hierarchy[chosen]
    assignments, estimates = {}, {}
    for i,group in enumerate(groups,1):
        for cell in group:
            assignments[cell] = f'{prefix}{i}'; estimates[cell] = group_stats(group)[1]
    test['group'] = test.cell.map(assignments).fillna('Unsupported')
    test['prediction'] = test.cell.map(estimates).fillna(baseline)
    test.to_csv(out/f'{outcome}_test.csv',index=False)
    summary = []
    for label,g in test[test.group.ne('Unsupported')].groupby('group',sort=True):
        mean,sd,var = group_mean_variance(g,outcome)
        candidates_cells = stats[stats.cell.map(assignments).eq(label)]
        example = candidates_cells.sort_values(['n','cell'],ascending=[False,True]).iloc[0].cell
        summary.append({'group':label,'n_test':len(g),'mean':mean,'sd':sd,'se':np.sqrt(var),
                        'effective_n':effective_n(g.weight),'example_cell':example,
                        'training_mean':estimates[example]})
    summary = pd.DataFrame(summary)
    summary.to_csv(out/f'{outcome}_groups.csv',index=False)
    pairs = []
    for first,second in itertools.combinations(summary.itertuples(),2):
        diff = second.mean-first.mean; se = np.hypot(first.se,second.se)
        pairs.append({'group1':first.group,'group2':second.group,'mean_difference':diff,
                      'p_unadjusted':float(2*norm.sf(abs(diff/se)))})
    if pairs:
        pairs = pd.DataFrame(pairs)
        pairs['p_holm'] = multipletests(pairs.p_unadjusted,method='holm')[1]
        pairs.to_csv(out/f'{outcome}_robust_pairs.csv',index=False)
        valid = test[test.group.ne('Unsupported')]
        tukey = pairwise_tukeyhsd(valid[outcome],valid.group)
        pd.DataFrame(tukey._results_table.data[1:],columns=tukey._results_table.data[0]).to_csv(out/f'{outcome}_tukey.csv',index=False)
    mapping = stats.copy(); mapping['group'] = mapping.cell.map(assignments)
    mapping.to_csv(out/f'{outcome}_assignment.csv',index=False)
    pd.DataFrame(candidates).sort_values('groups').to_csv(out/f'{outcome}_group_tuning.csv',index=False)
    rmse = float(np.sqrt(np.average((test[outcome]-test.prediction)**2,weights=test.weight)))
    base_rmse = float(np.sqrt(np.average((test[outcome]-baseline)**2,weights=test.weight)))
    return {'n':len(data),'train_n':len(train),'tune_n':len(tune),'test_n':len(test),
            'groups':int(chosen),'unsupported_test':int(test.group.eq('Unsupported').sum()),
            'rmse':rmse,'baseline_rmse':base_rmse,'mean':weighted_mean_sd(data[outcome],data.weight)[0],
            'test_mean':weighted_mean_sd(test[outcome],test.weight)[0]}


ANOVA_TERMS = [('Archetype','C(archetype)'),('Dwelling type','C(dwelling_type)'),
               ('Location','C(location)'),('Insulation','C(insulation)'),
               ('Building-age proxy','age100'),('Floor-area proxy','area100'),
               ('Cooling presence','C(cooling)'),
               ('Archetype x type','C(archetype):C(dwelling_type)'),
               ('Archetype x location','C(archetype):C(location)'),
               ('Archetype x insulation','C(archetype):C(insulation)'),
               ('Insulation x location','C(insulation):C(location)'),
               ('Type x location','C(dwelling_type):C(location)'),
               ('Type x insulation','C(dwelling_type):C(insulation)'),
               ('Age x area','age100:area100')]


def energy_anova(frame, outcome, year, out, include_insulation=True):
    data = frame[frame.split.eq('test')].dropna(subset=[outcome,'age_proxy','area_proxy','dwelling_type','location']).copy()
    data = data[data.cooling.ne('Unknown')]
    if include_insulation:
        if year==2013: data = data[data.insulation.ne('Unknown')]
        else: data = data[data.insulation_improvement.ne('Unknown')]
    data['age100'] = data.age_proxy/100; data['area100'] = data.area_proxy/100
    active = ANOVA_TERMS if year == 2013 else [
        (label.replace('Insulation', 'Insulation improvement').replace('x insulation', 'x insulation improvement'),
         term.replace('C(insulation)', 'C(insulation_improvement)')) for label, term in ANOVA_TERMS]
    if not include_insulation:
        active = [(label, term) for label, term in active if 'insulation' not in term]
    formula = outcome+' ~ '+' + '.join(term for _,term in active)
    fitted = smf.wls(formula,data=data,weights=data.weight/data.weight.mean()).fit(cov_type='HC3')
    # Joint Wald F tests for prespecified coefficient blocks in one full model.
    design = fitted.model.data.design_info
    rows = []
    for label,term in active:
        if term not in design.term_name_slices or design.term_name_slices[term].start == design.term_name_slices[term].stop:
            rows.append({'term':label,'F':np.nan,'p':np.nan,'df_num':0,'n':len(data),'status':'Unavailable'}); continue
        sl = design.term_name_slices[term]
        restriction = np.eye(len(fitted.params))[sl]
        test = fitted.f_test(restriction)
        rows.append({'term':label,'F':float(np.asarray(test.fvalue).item()),
                     'p':float(np.asarray(test.pvalue).item()),'df_num':float(test.df_num),
                     'n':len(data),'status':'Model-based HC3'})
    result = pd.DataFrame(rows); valid = result.p.notna()
    result.loc[valid,'p_holm'] = multipletests(result.loc[valid,'p'],method='holm')[1]
    result.to_csv(out/f'{outcome}_anova.csv',index=False)
    return {'n':len(data),'formula':formula,'rank':int(np.linalg.matrix_rank(fitted.model.exog)),
            'columns':len(fitted.params),'weighted_r2':float(fitted.rsquared),
            'holm_significant_terms':result.loc[result.p_holm.lt(.05),'term'].tolist()}


def insulation_group_tests(out):
    """Yes/no renovation contrasts within fixed heating groups; exclude unknowns."""
    data=pd.read_csv(out/'heating_test.csv')
    rows=[]
    for label, g in data[data.group.ne('Unsupported')].groupby('group'):
        valid=g[g.insulation_improvement.isin(['Reported','Not reported'])].copy()
        yes=valid.insulation_improvement.eq('Reported')
        row={'group':label,'yes_n':int(yes.sum()),'no_n':int((~yes).sum()),
             'excluded_unknown_n':len(g)-len(valid),'p':np.nan,'difference_hours':np.nan}
        if yes.sum()>=2 and (~yes).sum()>=2:
            valid['improved']=yes.astype(int)
            fit=smf.wls('heating ~ improved',valid,weights=valid.weight/valid.weight.mean()).fit(cov_type='HC3')
            row.update(p=float(fit.pvalues['improved']),difference_hours=float(fit.params['improved']))
        rows.append(row)
    result=pd.DataFrame(rows);valid=result.p.notna()
    result.loc[valid,'p_holm']=multipletests(result.loc[valid,'p'],method='holm')[1]
    result.to_csv(out/'heating_insulation_pairs.csv',index=False)


def equipment_summary(frame, out):
    """Weighted descriptive estimates with questionnaire-specific denominators."""
    rows = []
    for a, g in [('All', frame), *list(frame.groupby('archetype'))]:
        for measure, column, positive in [
            ('AC ownership', 'cooling', 'Present'),
            ('Insulation/window improvements', 'insulation_improvement', 'Reported')]:
            valid = g[g[column].ne('Unknown')]
            rows.append({'archetype':a, 'measure':measure, 'n':len(valid),
                         'positive_n':int(valid[column].eq(positive).sum()),
                         'estimate':100*np.average(valid[column].eq(positive), weights=valid.weight),
                         'unit':'percent'})
        valid = g.dropna(subset=['cooling_hours'])
        rows.append({'archetype':a, 'measure':'Cooling hours (frequency 1-3)',
                     'n':len(valid), 'positive_n':None,
                     'estimate':np.average(valid.cooling_hours, weights=valid.weight), 'unit':'hours/day'})
    pd.DataFrame(rows).to_csv(out/'equipment_summary.csv', index=False)
    return {'eligible_n':len(frame),
            'cooling_present_n':int(frame.cooling.eq('Present').sum()),
            'cooling_hours_n':int(frame.cooling_hours.notna().sum()),
            'improvement_reported_n':int(frame.insulation_improvement.eq('Reported').sum()),
            'improvement_unknown_n':int(frame.insulation_improvement.eq('Unknown').sum())}


def demographic_tables(frame, model, cfg, out):
    rows = []
    for a,g in frame.groupby('archetype'):
        row = {'archetype':int(a),'n':len(g),'effective_n':effective_n(g.weight),
               'household_share':100*g.weight.sum()/frame.weight.sum(),
               'people_share':100*(g.weight*g.nmembers).sum()/(frame.weight*frame.nmembers).sum()}
        for feat in FEATURES:
            row[feat+'_mean'], row[feat+'_sd'] = weighted_mean_sd(g[feat],g.weight)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'archetypes.csv',index=False)
    test = frame[frame.split.eq('test')]
    manova = MANOVA.from_formula('PC1 + PC2 + education ~ C(region)',data=test).mv_test().results['C(region)']['stat']
    manova.rename_axis('Statistic').to_csv(out/'regional_manova.csv')
    # Regional centroid geometry is purely descriptive; no centroid p-value claim.
    region_rows = []
    x = cluster_matrix(frame,model)
    for region,g in frame.groupby('region'):
        idx = frame.index.get_indexer(g.index)
        local = weighted_kmeans(x[idx],g.weight,4,cfg['seed']+int(region),3)
        distance = ((local['centers'][:,None]-np.array(model['centers'])[None])**2).sum(2)
        rr,cc = linear_sum_assignment(distance)
        for r,c in zip(rr,cc):
            region_rows.append({'region':int(region),'archetype':int(c+1),'n':len(g),
                                **{f'coordinate_{j+1}':float(v) for j,v in enumerate(local['centers'][r])}})
    pd.DataFrame(region_rows).to_csv(out/'regional_centroids.csv',index=False)
    train = frame[frame.split.eq('train')]
    elbow = []
    # For 2021 this is diagnostic refitting only, never used for its assignments.
    xtrain = cluster_matrix(train,model)
    for k in range(1,11):
        fit = weighted_kmeans(xtrain,train.weight,k,cfg['seed'],3)
        elbow.append({'k':k,'weighted_mean_squared_distance':fit['inertia']})
    pd.DataFrame(elbow).to_csv(out/'elbow.csv',index=False)
    region_hours = []
    for region,g in frame.dropna(subset=['heating']).groupby('region'):
        region_hours.append({'region':int(region),'n':len(g),
                             'heating_hours':weighted_mean_sd(g.heating,g.weight)[0]})
    pd.DataFrame(region_hours).to_csv(out/'regional_heating.csv',index=False)
    return {'raw_n':None,'eligible_n':len(frame),'split_counts':frame.split.value_counts().to_dict(),
            'outside_reference_range':int(frame.outside_reference_range.sum()),
            'household_weight_total':float(frame.weight.sum()),'effective_n':effective_n(frame.weight),
            'construction_unknown_pct':100*float(frame.loc[frame.construction_period.isna(),'weight'].sum()/frame.weight.sum())}


def run(config_path=ROOT/'revised_paper_config.json'):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    out = ROOT/cfg['output_root']; out.mkdir(parents=True,exist_ok=True)
    input_cfg = load_config(ROOT/cfg['input_config'])
    raw_waves, frames, extras, evidence = {}, {}, {}, {'configuration':cfg,'years':{},'inputs':{}}
    for year in [2013,2021]:
        path = _resolve_raw_input(input_cfg,year)
        raw, extra = read_wave(path,year)
        raw_waves[year], extras[year] = raw, extra
        folder = out/str(year); folder.mkdir(exist_ok=True)
        frame, excluded = make_frame(raw,extra,path,year,cfg)
        excluded.to_csv(folder/'exclusions.csv',index=False)
        frames[year] = frame
        evidence['inputs'][str(year)] = {'path':str(path),'sha256':sha256(path),'raw_n':len(raw)}
        print(f'{year}: {len(frame)} eligible of {len(raw)}; {len(excluded)} excluded',flush=True)
    model = fit_demographics(frames[2013].query("split=='train'"),cfg)
    model['all_variable_pca'] = fit_pca(frames[2013].query("split=='train'")[FEATURES],frames[2013].query("split=='train'").weight)
    write_json(out/'reference_model.json',model)
    for year in [2013,2021]:
        folder = out/str(year)
        frames[year] = frame = assign_archetypes(frames[year],model).reset_index(drop=True)
        frame.to_csv(folder/'households.csv',index=False)
        result = demographic_tables(frame,model,cfg,folder)
        result['raw_n'] = len(raw_waves[year])
        result['probability'] = probability_analysis(frame,cfg,folder)
        result['energy'] = {}
        if year == 2021:
            result['equipment'] = equipment_summary(frame, folder)
        for outcome in ['heating','electricity']:
            print(f'{year}: {outcome} split-sample groups and robust tests',flush=True)
            result['energy'][outcome] = learned_groups(frame,outcome,cfg,folder)
            result['energy'][outcome]['anova'] = energy_anova(frame,outcome,year,folder)
        if year==2021: insulation_group_tests(folder)
        evidence['years'][str(year)] = result
        write_json(folder/'results.json',result)
    # Explicit input-assumption sensitivity; models retrained only on 2013 train.
    sensitivities = []
    scenarios = [('reference',{} ,None),('no_primary_0',{'no_or_primary_years':0.},None),
                 ('no_primary_5',{'no_or_primary_years':5.},None),('age_cap_90',{'age_cap':90},None),
                 ('tertiary_16',{'tertiary_years':16.},None)]
    scenarios += [(f'age_draw_{s}',{},s) for s in range(3)]
    for name,changes,seed in scenarios:
        print(f'Input sensitivity: {name}',flush=True)
        alternative = {}
        for year in [2013,2021]:
            f,_ = features_from_codes(raw_waves[year],year,{**cfg,**changes},seed)
            alternative[year] = f.merge(frames[year][['id','weight','split']],on='id',validate='one_to_one')
        fitted = model if name=='reference' else fit_demographics(alternative[2013].query("split=='train'"),cfg)
        alternatives = {y:assign_archetypes(f,fitted) for y,f in alternative.items()}
        match = frames[2013][['id','archetype','weight']].merge(alternatives[2013][['id','archetype']],on='id')
        counts = np.zeros((4,4)); np.add.at(counts,(match.archetype_x-1,match.archetype_y-1),match.weight)
        r,c = linear_sum_assignment(-counts); mapping = {int(b+1):int(a+1) for a,b in zip(r,c)}
        for year,f in alternatives.items():
            f['archetype'] = f.archetype.map(mapping)
            for a,g in f.groupby('archetype'):
                sensitivities.append({'scenario':name,'year':year,'archetype':int(a),
                                      'share':100*g.weight.sum()/f.weight.sum()})
    pd.DataFrame(sensitivities).to_csv(out/'sensitivity.csv',index=False)
    evidence['inputs']['electricity_2013'] = {'path':str(ROOT/'results_grins.csv'),'sha256':sha256(ROOT/'results_grins.csv')}
    evidence['checks'] = {'no_production_outputs_overwritten':True,'test_split_unused_in_fitting':True,
                          'frozen_2013_pca_and_centers':True,'softmax_simplex':True,
                          '2021_electricity_is_explicitly_modeled':True,'no_missing_heating_filled_with_zero':True}
    write_json(out/'results.json',evidence)
    from revised_figures import make_all_figures
    make_all_figures(out,model,cfg)
    write_json(out/'manifest.json',{'inputs':evidence['inputs'],
               'code':{p.name:sha256(p) for p in [Path(__file__),ROOT/'revised_figures.py',Path(config_path)]},
               'outputs':{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print('Corrected analyses and vector figures complete.',flush=True)
    return evidence


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'revised_paper_config.json')
    run(parser.parse_args().config)
