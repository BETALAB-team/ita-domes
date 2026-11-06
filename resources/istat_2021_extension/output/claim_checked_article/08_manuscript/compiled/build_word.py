"""Compile the current reviewed Markdown without altering scientific wording."""
from pathlib import Path
import re, json, hashlib
from docx import Document
from docx.shared import Mm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'article_first_draft.md'
OUT=HERE/'Italian_user_archetypes_manuscript.docx'
text=SOURCE.read_text(encoding='utf-8-sig')
doc=Document()
sec=doc.sections[0]
sec.page_width=Mm(210); sec.page_height=Mm(297)
sec.top_margin=Mm(20); sec.bottom_margin=Mm(20)
sec.left_margin=Mm(20); sec.right_margin=Mm(20)
sec.footer_distance=Mm(9)
for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
 s=doc.styles[name]; s.font.name='Times New Roman'; s.font.color.rgb=RGBColor(0,0,0)
 fonts=s.element.get_or_add_rPr().get_or_add_rFonts()
 for key in list(fonts.attrib):
  if 'Theme' in key:del fonts.attrib[key]
 for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Times New Roman')
 s.paragraph_format.widow_control=True
normal=doc.styles['Normal']; normal.font.size=Pt(11)
normal.paragraph_format.line_spacing=1.10; normal.paragraph_format.space_after=Pt(6)
for name,size,before,after in [('Title',18,0,12),('Heading 1',14,12,7),('Heading 2',12,10,5),('Heading 3',11,8,4),('Caption',9.5,4,7)]:
 s=doc.styles[name]; s.font.size=Pt(size)
 s.font.bold=name!='Caption'; s.font.italic=name=='Caption'
 s.paragraph_format.space_before=Pt(before); s.paragraph_format.space_after=Pt(after)
 s.paragraph_format.keep_with_next=name!='Caption'
 s.paragraph_format.keep_together=True
doc.core_properties.title=text.splitlines()[0].removeprefix('# ')
doc.core_properties.subject='Italian demographic user archetypes and operational mixtures'
doc.core_properties.author=''; doc.core_properties.last_modified_by=''
doc.core_properties.comments='Compiled from the Markdown manuscript; scientific text preserved.'
foot=sec.footer.paragraphs[0]; foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE')
foot._p.append(fld)
foot.style=doc.styles['Normal']; foot.paragraph_format.space_after=Pt(0)
for run in foot.runs: run.font.size=Pt(9)

link_re=r'\[(\[[^\]]+\]|[^\]]*)\]\(([^)]+)\)'
token=re.compile(link_re+r'|\*\*([^*]+)\*\*|\*([^*]+)\*')
links=[]; images=[]; blocks=[]
def inline(p,s,base_size=None):
 pos=0
 for m in token.finditer(s):
  if m.start()>pos: p.add_run(s[pos:m.start()])
  label,url,bold,italic=m.groups()
  if label is not None:
   if not re.match(r'https?://',url):
    url=(SOURCE.parent/url.split('#')[0]).resolve().as_uri()+(('#'+url.split('#',1)[1]) if '#' in url else '')
   rel=p.part.relate_to(url,RT.HYPERLINK,is_external=True)
   hl=OxmlElement('w:hyperlink'); hl.set(qn('r:id'),rel)
   run=OxmlElement('w:r'); props=OxmlElement('w:rPr')
   fonts=OxmlElement('w:rFonts'); fonts.set(qn('w:ascii'),'Times New Roman'); fonts.set(qn('w:hAnsi'),'Times New Roman');props.append(fonts)
   color=OxmlElement('w:color');color.set(qn('w:val'),'000000');props.append(color)
   if base_size:
    sz=OxmlElement('w:sz');sz.set(qn('w:val'),str(round(base_size*2)));props.append(sz)
   run.append(props); t=OxmlElement('w:t');t.text=label;run.append(t);hl.append(run);p._p.append(hl)
   links.append({'text':label,'url':url})
  else:
   r=p.add_run(bold if bold is not None else italic);r.bold=bold is not None;r.italic=italic is not None
  pos=m.end()
 if pos<len(s):p.add_run(s[pos:])
 if base_size:
  for run in p.runs:run.font.size=Pt(base_size)

def table(lines):
 rows=[[x.strip() for x in line.strip().strip('|').split('|')] for line in lines]
 rows=[row for row in rows if not all(re.fullmatch(r':?-+:?',x) for x in row)]
 t=doc.add_table(rows=0,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 widths=([14,45,28,20,27,36] if len(rows[0])==6 else [50,20,100])
 for col,w in zip(t.columns,widths):col.width=Mm(w)
 pr=t._tbl.tblPr
 borders=OxmlElement('w:tblBorders')
 for side in ['top','left','bottom','right','insideH','insideV']:
  el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
 pr.append(borders)
 for ri,row in enumerate(rows):
  cells=t.add_row().cells
  trpr=cells[0]._tc.getparent().get_or_add_trPr()
  no=OxmlElement('w:cantSplit');trpr.append(no)
  if ri==0:trpr.append(OxmlElement('w:tblHeader'))
  for ci,(cell,value,w) in enumerate(zip(cells,row,widths)):
   cell.width=Mm(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   tcpr=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
   for side in ['top','bottom','left','right']:
    edge=OxmlElement('w:'+side);edge.set(qn('w:w'),'85');edge.set(qn('w:type'),'dxa');mar.append(edge)
   tcpr.append(mar)
   shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'E8EBEE' if ri==0 else ('F6F7F8' if ri%2==0 else 'FFFFFF'));tcpr.append(shd)
   p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
   p.paragraph_format.keep_with_next=ri<len(rows)-1
   p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ((len(row)==6 and ci==1) or (len(row)==3 and ci!=1)) else WD_ALIGN_PARAGRAPH.CENTER
   inline(p,value,9 if len(row)==6 else 10)
   for r in p.runs:r.bold=ri==0
 blocks.append({'type':'table','rows':rows})
 spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(0)
 spacer.paragraph_format.space_before=Pt(0);spacer.paragraph_format.line_spacing=Pt(4)
 spacer.add_run().font.size=Pt(1)

clean=re.sub(r'<!--.*?-->','',text,flags=re.S)
chunks=re.split(r'\n\s*\n',clean.strip())
refs=False
for chunk in chunks:
 chunk=chunk.strip()
 if not chunk:continue
 if chunk.startswith('|'):
  table(chunk.splitlines());continue
 h=re.match(r'^(#{1,6}) (.*)$',chunk,re.S)
 if h:
  n=len(h[1]);value=h[2].strip()
  if n==1:p=doc.add_paragraph(style='Title')
  else:p=doc.add_paragraph(style='Heading '+str(min(n-1,3)))
  if value=='References':refs=True
  inline(p,value);blocks.append({'type':'heading','text':value});continue
 im=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',chunk)
 if im:
  path=Path(im[2]);assert path.is_file()
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(6)
  shape=p.add_run().add_picture(str(path),width=Mm(170))
  shape._inline.docPr.set('descr',im[1]);images.append(str(path));blocks.append({'type':'image','path':str(path)});continue
 caption=chunk.startswith('*Table ') or chunk.startswith('*Figure ')
 p=doc.add_paragraph(style='Caption' if caption else 'Normal')
 if caption:
  p.paragraph_format.keep_with_next=chunk.startswith('*Table ')
 if refs:
  p.paragraph_format.left_indent=Mm(6);p.paragraph_format.first_line_indent=Mm(-6)
  p.paragraph_format.keep_together=True;p.paragraph_format.line_spacing=1.0
  p.paragraph_format.space_after=Pt(3)
 inline(p,chunk,9.5 if refs else None)
 blocks.append({'type':'caption' if caption else ('reference' if refs else 'paragraph'),'source_markdown':chunk})
# Let the first figure float after the endpoint and weight definitions, avoiding
# a half-empty page while preserving every paragraph's wording and body order.
pars=doc.paragraphs
image_p=next(p for p in pars if p._p.xpath('.//wp:docPr[@descr="Figure 4: Actual shared heating-control component profiles and group-conditional versus pooled training probabilities"]'))
caption_p=next(p for p in pars if p.text.startswith('Figure 4.'))
weights_p=next(p for p in pars if p.text.startswith('With eligible stratum size'))
weights_p._p.addnext(caption_p._p);weights_p._p.addnext(image_p._p)
metrics_image=next(p for p in pars if p._p.xpath('.//wp:docPr[@descr="Figure 5: Overall within-model annual-heating and individual-maximum RMSE with all four Bolzano subgroups"]'))
metrics_caption=next(p for p in pars if p.text.startswith('Figure 5.'))
pilot_end=next(p for p in pars if p.text.startswith("The secondary pilot's same-site early/late annual changes"))
pilot_end._p.addnext(metrics_caption._p);pilot_end._p.addnext(metrics_image._p)
# Remove inherited theme fonts and title rules from the Word base template.
for el in doc.styles.element.iter():
 if el.tag==qn('w:rFonts'):
  for key in list(el.attrib):
   if 'Theme' in key:del el.attrib[key]
 if el.tag in (qn('w:pBdr'),qn('w:tblpPr')):
  el.getparent().remove(el)
for el in list(doc.element.iter(qn('w:pBdr'))):el.getparent().remove(el)
OUT.parent.mkdir(parents=True,exist_ok=True)
doc.save(OUT)
manifest={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'docx':str(OUT),'docx_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
 'images':images,'tables':sum(x['type']=='table' for x in blocks),'references':sum(x['type']=='reference' for x in blocks),
 'conversion':'Markdown comments omitted; scientific wording, titles, labels, figures, tables, links and explicit incomplete markers retained. Figures float after method definitions and results narrative to avoid large pagination gaps; body paragraph order unchanged.',
 'blocks':blocks,'hyperlinks':links}
(HERE/'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['blocks','hyperlinks']},indent=2))
