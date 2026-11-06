"""Read-only checks of the archived archetype input packet; no clustering or simulation."""
from pathlib import Path
import csv, hashlib, json, collections
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
def read(path, delimiter=','):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    specification=json.loads((HERE/'assignment_mapping_schema.json').read_text(encoding='utf-8'))
    archived=read(ROOT/'clusters.csv')
    exported=read(HERE/'archived_four_archetype_assignments.csv')
    baseline=read(ROOT/'baseline_outputs/household_archetypes.csv')
    raw=read(ROOT/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt','\t'.encode().decode('unicode_escape'))
    compact=read(ROOT/'istat_microdata_2013.csv',';')
    original={int(r['identi']):int(r['segment'])+1 for r in archived}
    mapped={int(r['household_id']):int(r['archetype_id']) for r in exported}
    old_reconstruction={int(r['id']):int(r['archetype']) for r in baseline}
    rawids=[int(r['nprog']) for r in raw]
    compactids=[int(r['id']) for r in compact]
    assert len(archived)==len(original)==len(exported)==len(mapped)==17855
    assert original==mapped and set(original.values())=={1,2,3,4}
    assert set(original)<=set(rawids) and set(original)<=set(compactids)
    assert len(rawids)==len(set(rawids))==len(compactids)==len(set(compactids))==20000
    disagreement=sum(original[i]!=old_reconstruction[i] for i in original)
    assert disagreement==318
    assert digest(ROOT/'clusters.csv')==specification['input_sha256']
    assert digest(HERE/'archived_four_archetype_assignments.csv')==specification['assignment_sha256']
    results={'date':'2026-10-20','n':len(mapped),'group_counts':dict(sorted(collections.Counter(mapped.values()).items())),
             'source_to_mapping_exact':True,'raw_compact_id_join_complete':True,
             'baseline_disagreement_n':disagreement,'baseline_disagreement_denominator':len(mapped),
             'script_sha256':digest(Path(__file__)),'mapping_sha256':digest(HERE/'archived_four_archetype_assignments.csv'),
             'archived_source_sha256':digest(ROOT/'clusters.csv'),'pass':True,
             'boundary':'Archive consistency and ID provenance only; not exact published branch identity or empirical validation.'}
    (HERE/'assignment_packet_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
