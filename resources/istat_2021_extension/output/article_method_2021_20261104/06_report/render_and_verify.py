from pathlib import Path
import sys,importlib.util,functools,shutil,tempfile,json,re,hashlib,uuid,unicodedata
from zipfile import ZipFile
from lxml import etree
from pdf2image import convert_from_path
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPO=ROOT.parents[1]
SKILL=Path('C:/Users/khajmoh18975/.codex/plugins/cache/openai-primary-runtime/documents/26.1103.10131/skills/documents/render_docx.py')
POPPLER='C:/Users/khajmoh18975/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
DOCX=HERE/'Italian_user_archetypes_2021_reconstruction.docx';PDF=DOCX.with_suffix('.pdf')
spec=importlib.util.spec_from_file_location('packaged_render_docx',SKILL);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def use_word_export(input_path,user_profile,convert_tmp_dir,stem,verbose=False):
 assert Path(input_path).resolve()==DOCX.resolve();target=Path(convert_tmp_dir)/(stem+'.pdf');shutil.copy2(PDF,target)
 return str(target),'Native Microsoft Word export in place of unavailable LibreOffice; packaged PNG rasterizer retained.'
renderer.convert_to_pdf=use_word_export;renderer.convert_from_path=functools.partial(convert_from_path,poppler_path=POPPLER);renderer.pdfinfo_from_path=functools.partial(renderer.pdfinfo_from_path,poppler_path=POPPLER)
temp=HERE/'qa_temp';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
class TaskDirectory:
 def __init__(self,prefix='',**kwargs):self.path=temp/(prefix+uuid.uuid4().hex[:10]);self.path.mkdir(mode=0o777);self.name=str(self.path)
 def __enter__(self):return self.name
 def __exit__(self,*args):return False
renderer.tempfile.TemporaryDirectory=TaskDirectory
pngs=renderer.rasterize(str(DOCX),str(HERE/'qa'),144,False,False)
reader=PdfReader(PDF);pages=[p.extract_text() or '' for p in reader.pages];full='\n'.join(pages)
blocks=json.loads((HERE/'report_content.json').read_text(encoding='utf-8'))
with ZipFile(DOCX) as z:
 xml=etree.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
 docxtext='\n'.join(''.join(p.xpath('.//w:t/text() | .//m:t/text()',namespaces=ns)) for p in xml.xpath('.//w:p',namespaces=ns))
 media={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
 eqs=len(xml.xpath('.//m:oMath',namespaces=ns))
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s)).replace('−','-').replace('ﬁ','fi').replace('ﬂ','fl'))
missing_docx=[];missing_pdf=[];images=0
for b in blocks:
 if b['kind'] in ['paragraph','link']:texts=[b['text']]
 elif b['kind']=='table':texts=[*b['headers'],*[v for row in b['rows'] for v in row]]
 elif b['kind']=='equation':texts=[v for row in b['coefficients'] for v in row]
 else:
  texts=[]
  if b['kind']=='image':assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest() in media;images+=1
 for text in texts:
  if norm(text) not in norm(docxtext):missing_docx.append(text)
  if norm(text) not in norm(full):missing_pdf.append(text)
assert not missing_docx,missing_docx
assert not missing_pdf,missing_pdf
assert eqs==1 and images==5
assert len(pages)==len(pngs)
man=json.loads((ROOT/'02_probabilities/artifact_hashes.json').read_text());changed=[n for n,h in man.items() if hashlib.sha256((ROOT/'02_probabilities'/n).read_bytes()).hexdigest()!=h];assert not changed,changed
baseline=json.loads((REPO/'output/temporal_extension_2013_2021_20261104/00_baseline/baseline_freeze.json').read_text())['sources'];source_changed=[v['path'] for v in baseline if hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()!=v['sha256']];assert not source_changed,source_changed
q={'pages':len(pages),'pngs':len(pngs),'content_blocks':len(blocks),'missing_docx':missing_docx,'missing_pdf':missing_pdf,'embedded_figures_verified':images,'native_equations':eqs,'frozen_probability_artifacts_verified':len(man),'changed_probability_artifacts':changed,'protected_original_sources_verified':len(baseline),'changed_original_sources':source_changed,'characters_by_page':[len(p) for p in pages],'visual_review':'pending','renderer':'Native Word PDF export and current packaged render_docx.py with bundled Poppler'}
(HERE/'qa/pdf_extracted_text.txt').write_text(full,encoding='utf-8');(HERE/'qa/verification.json').write_text(json.dumps(q,indent=2),encoding='utf-8');print(json.dumps(q,indent=2))
