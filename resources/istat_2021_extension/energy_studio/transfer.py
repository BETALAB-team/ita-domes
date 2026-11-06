"""Annual ISTAT-ratio transfer. No weather, thermal solver or target EP inputs."""
import math

END_USES = {'heating_kwh': 'heating', 'cooling_kwh': 'cooling', 'appliance_kwh': 'appliance'}


def transfer_one(reference_energy, reference_index, target_index, is_anchor=False):
    """Return estimate, ratio and status; never invent unsupported denominators."""
    if any(not math.isfinite(float(v)) for v in [reference_energy, reference_index, target_index]):
        return math.nan, math.nan, 'missing_or_unsupported_index_or_anchor'
    if min(reference_energy, reference_index, target_index) < 0:
        return math.nan, math.nan, 'invalid_negative_value'
    if reference_index <= 0:
        return math.nan, math.nan, 'zero_reference_index'
    ratio = target_index / reference_index
    if reference_energy <= 1e-6 and target_index > 0 and not is_anchor:
        return math.nan, ratio, 'zero_energy_anchor_cannot_identify_transfer'
    return reference_energy * ratio, ratio, 'anchor_identity' if is_anchor else 'estimated'


def predict(scenarios, anchor, anchor_energy):
    """Only anchor EP totals are accepted: target benchmark outputs cannot leak in."""
    results, audit = {}, []
    for sc in scenarios:
        values = {}
        for use, index in END_USES.items():
            r, t = anchor['profile'], sc['profile']
            energy = anchor_energy.get(use, math.nan)
            estimate, ratio, status = transfer_one(energy, r[index+'_index'], t[index+'_index'], sc['id']==anchor['id'])
            values[use] = estimate
            audit.append({'scenario_id':sc['id'], 'scenario':sc['key'], 'end_use':use,
                          'reference_scenario':anchor['key'], 'reference_energy_kwh':energy,
                          'reference_index':r[index+'_index'], 'target_index':t[index+'_index'],
                          'reference_support_n':r[index+'_index_n'], 'target_support_n':t[index+'_index_n'],
                          'reference_coverage':r[index+'_index_coverage'], 'target_coverage':t[index+'_index_coverage'],
                          'index_source':t[index+'_index_source'], 'ratio':ratio,
                          'estimate_kwh':estimate, 'status':status})
        results[sc['id']] = values
    return results, audit
