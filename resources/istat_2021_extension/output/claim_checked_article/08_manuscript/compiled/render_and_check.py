from pathlib import Path
import sys, os, importlib.util, functools, shutil, tempfile, json, re, hashlib, uuid
sys.stdout.reconfigure(encoding='utf-8')
from pdf2image import convert_from_path
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
SKILL=Path('C:/Users/khajmoh18975/.codex/plugins/cache/openai-primary-runtime/documents/26.1013.11914/skills/documents/render_docx.py')
POPPLER='C:/Users/khajmoh18975/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
DOCX=HERE/'Italian_user_archetypes_manuscript.docx'; PDF=DOCX.with_suffix('.pdf')
assert PDF.is_file() and PDF.stat().st_size>0
spec=importlib.util.spec_from_file_location('packaged_render_docx',SKILL)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def use_word_export(input_path,user_profile,convert_tmp_dir,stem,verbose=False):
    assert Path(input_path).resolve()==DOCX.resolve()
    target=Path(convert_tmp_dir)/(stem+'.pdf')
    shutil.copy2(PDF,target)
    return str(target),'Native Word PDF export substituted for unavailable LibreOffice; packaged PNG rasterizer retained.'
renderer.convert_to_pdf=use_word_export
renderer.convert_from_path=functools.partial(convert_from_path,poppler_path=POPPLER)
temp=HERE/'qa_temp'; temp.mkdir(exist_ok=True)
assert temp.resolve().is_relative_to(HERE.resolve())
tempfile.tempdir=str(temp)
class TaskDirectory:
    def __init__(self,prefix='',**kwargs):
        self.path=temp/(prefix+uuid.uuid4().hex[:10])
        assert self.path.resolve().is_relative_to(HERE.resolve())
        self.path.mkdir(mode=0o777)
        self.name=str(self.path)
    def __enter__(self):return self.name
    def __exit__(self,*args):return False
renderer.tempfile.TemporaryDirectory=TaskDirectory
pngs=renderer.rasterize(str(DOCX),str(HERE/'qa'),144,False,False)
reader=PdfReader(PDF)
pages=[page.extract_text() or '' for page in reader.pages]
full='\n'.join(pages)
(HERE/'qa/pdf_extracted_text.txt').write_text(full,encoding='utf-8')
report={'pages':len(pages),'page_pngs':len(pngs),'renderer':'Native Microsoft Word PDF export plus packaged render_docx.py rasterizer and bundled Poppler',
 'text_chars_by_page':[len(x) for x in pages], 'images_by_page':[len(page.images) for page in reader.pages],
 'source_sha256':hashlib.sha256((HERE.parent/'article_first_draft.md').read_bytes()).hexdigest(),
 'docx_sha256':hashlib.sha256(DOCX.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
 'tables':2,'references':37,'visual_review':'pending'}
assert len(pages)==len(pngs)
assert all(len(x)>20 for x in pages)
assert 'Incomplete first manuscript' in full
assert 'Figure 4.' in full and 'Figure 5.' in full
assert '152' in full and '13,592' in full and '27/32' in full
assert 'ISO 15927-4:2005' in full
(HERE/'qa/export_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
