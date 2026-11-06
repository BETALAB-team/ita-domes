from pathlib import Path
import pandas as pd, json
from pypdf import PdfReader
r=Path.cwd(); o=r/'output/temporal_extension_2013_2021_20261104/01_comparability'
for p in (r/'istat').rglob('*.pdf'):
 t=[]
 for i,page in enumerate(PdfReader(p).pages):
  text=page.extract_text() or ''
  t.append(f'\nPAGE {i+1}\n{text}')
 (o/(p.stem+'.txt')).write_text('\n'.join(t),encoding='utf-8')
for p in [r/'istat/legenda_istat.xlsx',r/'istat/legenda_istat-old.xlsx',r/'domande istat.xlsx']:
 x=pd.ExcelFile(p); d={s:pd.read_excel(p,sheet_name=s).fillna('').astype(str).values.tolist() for s in x.sheet_names}
 (o/(p.stem+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
p=Path('C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main.pdf')
reader=PdfReader(p)
(o/'original_paper_all_15_pages.txt').write_text('\n'.join(f'PAGE {i+1}\n{page.extract_text()}' for i,page in enumerate(reader.pages)),encoding='utf8')
print('Extracted PDF and workbook evidence',len(reader.pages))
