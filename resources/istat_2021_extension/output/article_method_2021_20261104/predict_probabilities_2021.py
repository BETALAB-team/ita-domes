"""Evaluate the exported 2021 matrix without silently changing its estimates."""
from pathlib import Path
import argparse,csv,json
ROOT=Path(__file__).resolve().parent
def predict(age_years,area_m2):
 with (ROOT/'02_probabilities/polynomial_coefficient_matrix.csv').open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f);columns=reader.fieldnames[1:];B=[[float(r[c]) for c in columns] for r in reader]
 a=age_years;s=area_m2;x=[1,a/100,s/100,a*s/10000,a*a/10000,s*s/10000]
 p=[sum(c*v for c,v in zip(row,x)) for row in B];valid=all(0<=v<=1 for v in p);total=sum(p)
 return {'age_years':a,'area_m2':s,'raw_polynomial_estimates':p,'raw_sum':total,'individual_estimates_within_0_1':valid,'inside_fitting_center_hull':20<=a<=140 and 40<=s<=120,'inside_article_style_area_domain':20<=a<=140 and 40<=s<150,'explicitly_normalized_for_sampling':([v/total for v in p] if valid and total>0 else None),'normalization_note':'The normalized vector is a separately labelled calculation; the raw matrix estimates above are preserved.'}
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--age',required=True,type=float);parser.add_argument('--area',required=True,type=float);args=parser.parse_args();print(json.dumps(predict(args.age,args.area),indent=2))
