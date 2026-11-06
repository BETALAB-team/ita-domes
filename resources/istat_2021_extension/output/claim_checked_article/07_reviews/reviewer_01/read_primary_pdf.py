from pathlib import Path
from pypdf import PdfReader
root=Path.cwd()
conf=Path(r'C:\Users\khajmoh18975\Downloads\Proceedings of the 15th REHVA HVAC World Congress - CLIMA 2025.pdf')
r=PdfReader(conf)
t='\n'.join(f'\nPAGE {i+1}\n'+r.pages[i].extract_text() for i in range(1,11))
(root/'07_reviews/reviewer_01/conference_primary_text.txt').write_text(t,encoding='utf-8')
q=Path(r'C:\Works\GRINS\GRINS-Tool\resources\istat_data\istat\ECH_2013_IT\METADATI\ECH_Questionario_Anno 2013.pdf')
r=PdfReader(q)
t='\n'.join(f'\nPAGE {i+1}\n'+r.pages[i].extract_text() for i in range(39,43))
(root/'07_reviews/reviewer_01/questionnaire_primary_text.txt').write_text(t,encoding='utf-8')
print('Saved primary chapter pages 2-11 and questionnaire pages 40-43.')

