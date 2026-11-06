from pathlib import Path
import json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

class Report:
 def __init__(self,here):
  self.here=Path(here);self.doc=Document();self.blocks=[];s=self.doc.sections[0]
  s.page_width=Inches(8.5);s.page_height=Inches(11);s.top_margin=s.bottom_margin=Inches(.66);s.left_margin=s.right_margin=Inches(.7);s.header_distance=s.footer_distance=Inches(.28)
  for style in self.doc.styles:
   if style.type==1:style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
   for ff in style.element.xpath('.//w:rFonts'):
    for attr in list(ff.attrib):
     if 'theme' in attr.lower():del ff.attrib[attr]
   for el in style.element.xpath('.//w:pBdr'):el.getparent().remove(el)
   for el in style.element.xpath('.//w:color'):
    for attr in list(el.attrib):
     if 'theme' in attr.lower():del el.attrib[attr]
    el.set(qn('w:val'),'000000')
  n=self.doc.styles['Normal'];n.font.size=Pt(10.5);n.paragraph_format.line_spacing=1.04;n.paragraph_format.space_after=Pt(7)
  for name,size in [('Title',22),('Heading 1',16),('Heading 2',12)]:
   st=self.doc.styles[name];st.font.bold=True;st.font.size=Pt(size);st.paragraph_format.space_before=Pt(7);st.paragraph_format.space_after=Pt(6);st.paragraph_format.keep_with_next=True
  c=self.doc.styles['Caption'];c.font.size=Pt(9.5);c.font.bold=False;c.font.italic=False
  h=s.header.paragraphs[0];h.text='Italian user archetypes   |   2021 reconstruction';h.runs[0].font.size=Pt(8)
  f=s.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.RIGHT;f.add_run('4 November 2026   |   ').font.size=Pt(8)
  field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
 def p(self,text,style=None):
  pp=self.doc.add_paragraph(text,style);self.blocks.append({'kind':'paragraph','text':text});return pp
 def h(self,text):return self.p(text,'Heading 1')
 def sub(self,text):return self.p(text,'Heading 2')
 def page(self):self.doc.add_page_break()
 def note(self,text):
  pp=self.p(text)
  for r in pp.runs:r.font.size=Pt(9)
  return pp
 def caption(self,label,title,before=False):
  pp=self.doc.add_paragraph(style='Caption');pp.paragraph_format.keep_with_next=before
  pp.add_run(label+'  ').bold=True;pp.add_run(title);self.blocks.append({'kind':'paragraph','text':label+'  '+title});return pp
 def figure(self,path,width=7):
  path=Path(path);pp=self.doc.add_paragraph();pp.alignment=WD_ALIGN_PARAGRAPH.CENTER;pp.paragraph_format.keep_with_next=True;pp.paragraph_format.space_after=Pt(3)
  rr=pp.add_run();rr.add_picture(str(path),width=Inches(width))
  for el in rr._r.xpath('.//wp:docPr'):el.set('descr',path.stem.replace('_',' '))
  self.blocks.append({'kind':'image','path':str(path)})
 def table(self,headers,rows,widths,size=9.5):
  tt=self.doc.add_table(rows=1,cols=len(headers));tt.alignment=WD_TABLE_ALIGNMENT.CENTER;tt.autofit=False
  for c,w in zip(tt.columns,widths):c.width=Inches(w)
  for c,v,w in zip(tt.rows[0].cells,headers,widths):c.width=Inches(w);c.text=v
  tt.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
  for row in rows:
   for c,v,w in zip(tt.add_row().cells,row,widths):c.width=Inches(w);c.text=str(v)
  bd=OxmlElement('w:tblBorders')
  for edge in ['top','bottom','left','right','insideH','insideV']:
   el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil');el.set(qn('w:sz'),'5');el.set(qn('w:color'),'000000');bd.append(el)
  tt._tbl.tblPr.append(bd)
  for i,row in enumerate(tt.rows):
   row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
   for j,c in enumerate(row.cells):
    c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;pr=c._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
    for side in ['top','left','bottom','right']:
     el=OxmlElement('w:'+side);el.set(qn('w:w'),'48' if side in ['top','bottom'] else '65');el.set(qn('w:type'),'dxa');mar.append(el)
    pr.append(mar)
    if i==0:
     bb=OxmlElement('w:tcBorders');ee=OxmlElement('w:bottom');ee.set(qn('w:val'),'single');ee.set(qn('w:sz'),'4');ee.set(qn('w:color'),'000000');bb.append(ee);pr.append(bb)
    for pp in c.paragraphs:
     pp.paragraph_format.space_after=Pt(0);pp.paragraph_format.line_spacing=1
     pp.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
     for rr in pp.runs:rr.font.size=Pt(size);rr.font.name='Times New Roman'
  pp=self.doc.add_paragraph();pp.paragraph_format.space_after=Pt(0);pp.paragraph_format.space_before=Pt(0);pp.add_run().font.size=Pt(2)
  self.blocks.append({'kind':'table','headers':headers,'rows':[[str(v) for v in row] for row in rows]});return tt
 def link(self,label,url):
  pp=self.doc.add_paragraph();rid=pp.part.relate_to(url,RT.HYPERLINK,is_external=True);hl=OxmlElement('w:hyperlink');hl.set(qn('r:id'),rid)
  rr=OxmlElement('w:r');pr=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'19');pr.append(sz);rr.append(pr);text=OxmlElement('w:t');text.text=label;rr.append(text);hl.append(rr);pp._p.append(hl)
  self.blocks.append({'kind':'link','text':label,'url':url})
 def matrix(self,B):
  para=self.doc.add_paragraph();para.paragraph_format.space_before=Pt(5);para.paragraph_format.space_after=Pt(8)
  op=OxmlElement('m:oMathPara');opp=OxmlElement('m:oMathParaPr');jc=OxmlElement('m:jc');jc.set(qn('m:val'),'center');opp.append(jc);op.append(opp);om=OxmlElement('m:oMath');op.append(om)
  def run(text):
   rr=OxmlElement('m:r');mp=OxmlElement('m:rPr');sty=OxmlElement('m:sty');sty.set(qn('m:val'),'p');mp.append(sty);rr.append(mp)
   wp=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'20');wp.append(sz);rr.append(wp);tt=OxmlElement('m:t');tt.text=text;rr.append(tt);return rr
  def mat(data):
   delim=OxmlElement('m:d');dp=OxmlElement('m:dPr')
   for tag,val in [('begChr','['),('endChr',']')]:
    el=OxmlElement('m:'+tag);el.set(qn('m:val'),val);dp.append(el)
   delim.append(dp);ee=OxmlElement('m:e');mm=OxmlElement('m:m')
   for row in data:
    mr=OxmlElement('m:mr')
    for value in row:
     ce=OxmlElement('m:e')
     if value in ['p1','p2','p3','p4']:
      sub=OxmlElement('m:sSub');base=OxmlElement('m:e');base.append(run('p'));index=OxmlElement('m:sub');index.append(run(value[-1]));sub.append(base);sub.append(index);ce.append(sub)
     else:ce.append(run(value))
     mr.append(ce)
    mm.append(mr)
   ee.append(mm);delim.append(ee);return delim
  om.append(mat([['p1'],['p2'],['p3'],['p4']]))
  om.append(run(' = '));vals=[[f'{v:.5f}' for v in row] for row in B];om.append(mat(vals));om.append(mat([['1'],['a / 100'],['s / 100'],['a s / 10000'],['a² / 10000'],['s² / 10000']]))
  om.append(run(' + ε'))
  para._p.append(op);self.blocks.append({'kind':'equation','coefficients':vals})
 def save(self,name):
  self.doc.core_properties.title='2021 reconstruction of Italian user archetypes and energy groups';self.doc.core_properties.author='GRINS research working report'
  self.doc.save(self.here/name);(self.here/'report_content.json').write_text(json.dumps(self.blocks,ensure_ascii=False,indent=2),encoding='utf-8')
