"""Analytical loss fixtures including a zero reference and mixture expectation."""
import json,math
from pathlib import Path
from analyze_mixtures import loss
cases=[]
r=[{'w':.25,'detailed_reference':0.,'conditional_prediction':2.,'pooled_prediction':1.},{'w':.75,'detailed_reference':4.,'conditional_prediction':2.,'pooled_prediction':1.}]
a=loss(r,'w');assert a['conditional_MSE']==4 and a['pooled_MSE']==7 and a['paired_MSE_difference_conditional_minus_pooled']==-3
assert a['conditional_RMSE']==2 and a['conditional_signed_deviation']==-1 and a['pooled_signed_deviation']==-2
cases.append('unequal weights, zero reference, signed/squared/absolute arithmetic')
r=[{'w':1.,'detailed_reference':0.,'conditional_prediction':0.,'pooled_prediction':0.}];assert loss(r,'w')['conditional_RMSE']==0
cases.append('all zero without denominator replacement')
# Averaging annual individual maxima differs from maximizing an average curve.
a=[10,0];b=[0,20];expected=.5*max(a)+.5*max(b);coincident=max(.5*x+.5*y for x,y in zip(a,b));assert expected==15 and coincident==10
cases.append('expected individual maximum distinguished from coincident maximum')
result={'pass':True,'cases':cases,'thermal_runs':0}
Path(__file__).with_name('analysis_fixture_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
