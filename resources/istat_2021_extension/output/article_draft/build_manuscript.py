from pathlib import Path
import re,json,html,hashlib
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).parent
text=(ROOT/'first_draft.md').read_text(encoding='utf-8')
def blocks():
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip(): i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in row):rows.append(row)
                i+=1
            yield 'table',rows;continue
        m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        if m: yield 'image',m.group(2)
        elif line.startswith('#'):yield 'heading',(len(line)-len(line.lstrip('#')),line.lstrip('# ').strip())
        else:yield 'paragraph',line
        i+=1
doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.8);sec.left_margin=sec.right_margin=Inches(.85)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
    st.font.size=Pt(11 if name in ['Normal','Caption'] else {'Title':19,'Subtitle':13,'Heading 1':14,'Heading 2':12,'Heading 3':11}.get(name,11))
normal=doc.styles['Normal'];normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(7)
for name in ['Heading 1','Heading 2','Heading 3']:
    doc.styles[name].paragraph_format.space_before=Pt(14);doc.styles[name].paragraph_format.space_after=Pt(7)
footer=sec.footer.paragraphs[0];footer.alignment=2
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
for kind,val in blocks():
    if kind=='heading':
        level,label=val;doc.add_paragraph(label,'Title' if level==1 else 'Heading '+str(level-1))
    elif kind=='paragraph':
        p=doc.add_paragraph(val,'Caption' if val.startswith(('Figure ','Table ')) else 'Normal')
        if val.startswith('Table '):p.paragraph_format.keep_with_next=True
        if val.startswith('['):
            for r in p.runs:r.italic=True;r.font.color.rgb=RGBColor.from_string('8B3B21')
    elif kind=='image':
        p=doc.add_paragraph();p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(ROOT/val),width=Inches(6.7))
    else:
        table=doc.add_table(rows=1,cols=len(val[0]));table.autofit=False
        for j,h in enumerate(val[0]):table.rows[0].cells[j].text=h
        for row in val[1:]:
            for cell,v in zip(table.add_row().cells,row):cell.text=v
        for ri,row in enumerate(table.rows):
            trpr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
            if ri==0:trpr.append(OxmlElement('w:tblHeader'))
            for cell in row.cells:
                tc=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
                for edge in ['top','left','bottom','right']:
                    el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');b.append(el)
                tc.append(b)
                if ri==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EFF3');tc.append(sh)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(5);p.paragraph_format.space_before=Pt(5)
                    for r in p.runs:r.font.size=Pt(9);r.bold=ri==0
        doc.add_paragraph()
temp=ROOT.parents[1]/'tmp'/'article_draft';temp.mkdir(parents=True,exist_ok=True)
doc.save(temp/'first_draft.docx')
print('DOCX candidate',temp/'first_draft.docx')
print('Body words including tables/captions',len(re.findall(r'\b\w+\b',text.split('## References')[0])))
