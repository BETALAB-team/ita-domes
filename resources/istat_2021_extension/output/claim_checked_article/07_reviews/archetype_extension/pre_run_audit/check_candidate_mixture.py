"""Read-only candidate-v2 checks; reuse independently reconstructed v1 inputs."""
import runpy,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
v=runpy.run_path(str(HERE/'independent_input_check.py'))
ROOT=v['ROOT'];train=v['train'];indexed=v['indexed']
p=ROOT/'04_analysis/archetype_extension/mixture_design_assessment.json'
c=json.loads(p.read_text())
assert c['test_ids_unchanged']==sorted(v['picked'])
assert c['training_n']==len(train)==13592
for component in c['components']:
    b=component['branch'];t=train[train.branch==b]
    mat=np.array([indexed[i] for i in t.household_id]);mu=mat.mean(axis=0)
    cost=((mat-mu)**2).sum(axis=1)
    ties=t.household_id.to_numpy()[abs(cost-cost.min())<=1e-10]
    assert int(min(ties))==component['household_id']
    assert len(ties)==component['tie_n']
    assert len(t)==component['training_n']
    assert abs(c['pooled_P_branch'][b]-len(t)/len(train))<1e-15
    for g in [1,2,3,4]:
        ng=sum(train.archetype_id==g);n=sum(t.archetype_id==g)
        assert c['counts'][str(g)][b]==n
        assert abs(c['training_P_branch_given_group'][str(g)][b]-n/ng)<1e-15
assert len({z['profile_sha256'] for z in c['components']})==4
out={'pass':True,'scope':'candidate mixture input assessment, not finalized v2 protocol/runner approval',
     'assessment_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
     'component_ids':[z['household_id'] for z in c['components']],
     'component_profiles_distinct':4,'training_n':len(train),
     'checks':['same held-out IDs','independent branch-specific squared-distance medoids/ties','training-only pooled and conditional probabilities']}
(HERE/'candidate_mixture_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
