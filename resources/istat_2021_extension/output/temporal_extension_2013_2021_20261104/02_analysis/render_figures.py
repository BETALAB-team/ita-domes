from pathlib import Path
import pandas as pd,json,re
from PIL import Image,ImageDraw
OUT=Path('output/temporal_extension_2013_2021_20261104/02_analysis');summ=pd.read_csv(OUT/'2021_fixed_proxy_summary.csv');refits=pd.read_csv(OUT/'2021_refit_k4.csv').to_dict('records');res=json.loads((OUT/'results.json').read_text())
s=(OUT/'run_analysis.py').read_text();s=s[s.index('def bars('):];s=re.sub(r"'Household share \(%\).*?cross-sectional samples'","'Household share (%) - proxy labels; cross-sectional samples'",s);s=s.replace('Eligible 2021 proxy','2021 no-under11 profile proxy');exec(s)
