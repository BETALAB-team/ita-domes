from pathlib import Path
import re,json,zipfile,hashlib,unicodedata
from collections import Counter
from lxml import etree
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'build_manifest.json').read_text(encoding='utf-8'))
docx=HERE/'Italian_user_archetypes_manuscript.docx';pdf=docx.with_suffix('.pdf')
W={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(docx) as z:
 tree=etree.fromstring(z.read('word/document.xml'))
 paragraphs=[''.join(p.xpath('.//w:t/text()',namespaces=W)) for p in tree.xpath('.//w:p',namespaces=W)]
 borders=len(tree.xpath('.//w:pBdr',namespaces=W))
 styles=etree.fromstring(z.read('word/styles.xml'))
 used_style_borders=len(styles.xpath('.//w:style[@w:styleId="Title" or @w:styleId="Normal" or @w:styleId="Heading1" or @w:styleId="Heading2"]//w:pBdr',namespaces=W))
 media=[n for n in z.namelist() if n.startswith('word/media/')]
 media_hashes=[hashlib.sha256(z.read(n)).hexdigest() for n in media]
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s).replace('\u00ad',''))
token=re.compile(r'\[(\[[^\]]+\]|[^\]]*)\]\(([^)]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*')
def plain(s):
 return token.sub(lambda m:next(x for x in [m[1],m[3],m[4]] if x is not None),s)
expected=[]
for b in manifest['blocks']:
 if b['type']=='image':continue
 if b['type']=='table':expected.extend(plain(x) for row in b['rows'] for x in row)
 else:expected.append(plain(b.get('source_markdown',b.get('text',''))))
para_counts=Counter(norm(x) for x in paragraphs)
needed=Counter(norm(x) for x in expected)
missing_docx=[x for x,n in needed.items() if para_counts[x]<n]
reader=PdfReader(pdf)
pdf_text='\n'.join(re.sub(r'(?m)^\s*\d+\s*$','',p.extract_text() or '') for p in reader.pages)
pdf_norm=norm(pdf_text)
missing_pdf=[x for x in expected if norm(x) not in pdf_norm]
assert not missing_docx,missing_docx
assert not missing_pdf,missing_pdf
assert len(media)==2
assert borders==0 and used_style_borders==0
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() in media_hashes for p in manifest['images'])
assert hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()==manifest['source_sha256']
report={'source_unchanged':True,'visible_source_blocks_and_table_cells_checked':len(expected),
 'missing_docx_text':missing_docx,'missing_pdf_text':missing_pdf,'embedded_source_images_byte_identical':True,
 'paragraph_borders':borders,'pages':len(reader.pages),'pdf_link_annotations':sum(len(p.get('/Annots',[])) for p in reader.pages),
 'body_paragraph_wording':'preserved exactly after Markdown formatting removal','figure_placement':'floated within methods/results for pagination'}
(HERE/'qa/content_fidelity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
