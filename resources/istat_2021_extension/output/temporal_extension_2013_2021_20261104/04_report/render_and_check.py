from pathlib import Path
import sys, importlib.util, functools, shutil, tempfile, json, re, hashlib, uuid
from zipfile import ZipFile
from lxml import etree
from pdf2image import convert_from_path
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
SKILL=Path('C:/Users/khajmoh18975/.codex/plugins/cache/openai-primary-runtime/documents/26.1103.10131/skills/documents/render_docx.py')
POPPLER='C:/Users/khajmoh18975/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
DOCX=HERE/'Household_archetypes_temporal_extension.docx';PDF=DOCX.with_suffix('.pdf')
spec=importlib.util.spec_from_file_location('packaged_render_docx',SKILL)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def use_word_export(input_path,user_profile,convert_tmp_dir,stem,verbose=False):
    assert Path(input_path).resolve()==DOCX.resolve()
    target=Path(convert_tmp_dir)/(stem+'.pdf');shutil.copy2(PDF,target)
    return str(target),'Native Word export used in place of unavailable LibreOffice; packaged PNG rasterizer retained.'
renderer.convert_to_pdf=use_word_export
renderer.convert_from_path=functools.partial(convert_from_path,poppler_path=POPPLER)
renderer.pdfinfo_from_path=functools.partial(renderer.pdfinfo_from_path,poppler_path=POPPLER)
temp=HERE/'qa_temp';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
class TaskDirectory:
    def __init__(self,prefix='',**kwargs):
        self.path=temp/(prefix+uuid.uuid4().hex[:10]);self.path.mkdir(mode=0o777);self.name=str(self.path)
    def __enter__(self):return self.name
    def __exit__(self,*args):return False
renderer.tempfile.TemporaryDirectory=TaskDirectory
pngs=renderer.rasterize(str(DOCX),str(HERE/'qa'),144,False,False)
reader=PdfReader(PDF);pages=[p.extract_text() or '' for p in reader.pages];full='\n'.join(pages)
blocks=json.loads((HERE/'report_content.json').read_text(encoding='utf-8'))
with ZipFile(DOCX) as z:
    xml=etree.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    docxtext='\n'.join(''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in xml.xpath('.//w:p',namespaces=ns))
def norm(s):return re.sub(r'\s+','',s)
missing_docx=[];missing_pdf=[]
for b in blocks:
    texts=[b['text']] if b['kind'] in ['paragraph','link'] else ([*b['headers'],*[v for r in b['rows'] for v in r]] if b['kind']=='table' else [])
    for text in texts:
        if norm(text) not in norm(docxtext):missing_docx.append(text)
        if norm(text) not in norm(full):missing_pdf.append(text)
assert not missing_docx,missing_docx
assert not missing_pdf,missing_pdf
assert len(pages)==len(pngs)==8,(len(pages),len(pngs))
assert all(len(p)>100 for p in pages)
manifest=json.loads((HERE.parent/'02_analysis/artifact_hashes.json').read_text())
changed=[n for n,h in manifest.items() if hashlib.sha256((HERE.parent/'02_analysis'/n).read_bytes()).hexdigest()!=h]
assert not changed,changed
qa={'pages':len(pages),'pngs':len(pngs),'content_blocks':len(blocks),'missing_docx':missing_docx,'missing_pdf':missing_pdf,'analysis_artifacts_verified':len(manifest),'analysis_artifacts_changed':changed,'characters_by_page':[len(p) for p in pages],'visual_review':'pending','renderer':'Microsoft Word PDF export and packaged render_docx.py with bundled Poppler'}
(HERE/'qa/pdf_extracted_text.txt').write_text(full,encoding='utf-8')
(HERE/'qa/verification.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))
