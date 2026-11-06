from pathlib import Path
import pandas as pd,json,hashlib
O=Path('output/article_method_2021_20261104/02_probabilities');b=pd.read_csv(O/'polynomial_coefficient_matrix.csv');cols=['intercept','age/100','area/100','age*area/10000','age^2/10000','area^2/10000'];assert b.columns.tolist()==['archetype']+cols
matrix=' \\\\ \n'.join(' & '.join(f'{v:.7f}' for v in row) for row in b[cols].to_numpy())
s='# New2021probabilitymatrix\n\nLet a be building age in years at2021 and s dwelling area in m2. The ordered basis is exactly the CSV column order:\n\n$$x=[1, a/100, s/100, as/10000, a^2/10000, s^2/10000]^T.$$\n\n$$\\hat p = Bx, \\quad B=\\begin{bmatrix}\n'+matrix+'\n\\end{bmatrix}.$$\n\nRows are aligned archetypesA1,A2,A3,A4. Zeros represent terms removed by backwards elimination. This is a new2021fit, not the published2013matrix. Primary fit uses equal-household empirical window ratios, support>=5,age<150 andarea<150. The expression is unconstrained OLS; use saved range/sum flags before interpreting it as probabilities, particularly outside the primary domain. No automatic clipping or renormalization is applied.\n'
(O/'matrix_equation_2021.md').write_text(s)
r=json.loads((O/'results.json').read_text());r['probability_basis_order']=cols;(O/'results.json').write_text(json.dumps(r,indent=2));print(s)
