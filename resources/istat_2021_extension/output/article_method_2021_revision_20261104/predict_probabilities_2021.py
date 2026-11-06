"""Evaluate the revised national dwelling probability matrix, without clipping."""
from pathlib import Path
import argparse,csv,json
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--age',type=float,required=True)
parser.add_argument('--area',type=float,required=True)
parser.add_argument('--allow-extrapolation',action='store_true')
args=parser.parse_args()
domain=15<=args.age<=135 and 40<=args.area<=136
if not domain and not args.allow_extrapolation:
    parser.error('Checked domain: dwelling age 15–135 years at 2021, area 40–136 m². Use --allow-extrapolation to inspect a flagged raw estimate outside it.')
path=Path(__file__).resolve().parent/'02_probability_candidates/recommended_full_quadratic_matrix.csv'
with path.open(newline='',encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
terms=['intercept','age/100','area/100','age*area/10000','age^2/10000','area^2/10000']
a=args.age/100;s=args.area/100;basis=[1,a,s,a*s,a*a,s*s]
p={f'A{r["archetype"]}':sum(float(r[t])*v for t,v in zip(terms,basis)) for r in rows}
print(json.dumps({'age_at_2021':args.age,'area_m2':args.area,'inside_checked_domain':domain,
                 'probabilities':p,'sum':sum(p.values()),'components_in_0_1':all(0<=v<=1 for v in p.values())},indent=2))
