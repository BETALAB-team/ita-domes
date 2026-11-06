from pathlib import Path
import re,html,json,hashlib,sys
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
ROOT=Path(__file__).parent
pdfmetrics.registerFont(TTFont('TNR','C:/Windows/Fonts/times.ttf'))
pdfmetrics.registerFont(TTFont('TNRB','C:/Windows/Fonts/timesbd.ttf'))
pdfmetrics.registerFont(TTFont('TNRI','C:/Windows/Fonts/timesi.ttf'))
pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNRB',italic='TNRI',boldItalic='TNRB')
stem=sys.argv[1] if len(sys.argv)>1 else 'first_draft'
source=(ROOT/(stem+'.md')).read_text(encoding='utf-8')
styles={
 'body':ParagraphStyle('body',fontName='TNR',fontSize=11,leading=14,spaceAfter=7),
 'title':ParagraphStyle('title',fontName='TNRB',fontSize=19,leading=23,spaceAfter=15),
 'h1':ParagraphStyle('h1',fontName='TNRB',fontSize=14,leading=17,spaceBefore=13,spaceAfter=8,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='TNRB',fontSize=12,leading=15,spaceBefore=11,spaceAfter=7,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='TNRI',fontSize=9.5,leading=12,spaceAfter=10),
 'pending':ParagraphStyle('pending',fontName='TNRI',fontSize=11,leading=14,spaceAfter=8,textColor=colors.HexColor('#8B3B21')),
 'cell':ParagraphStyle('cell',fontName='TNR',fontSize=9,leading=11),
 'head':ParagraphStyle('head',fontName='TNRB',fontSize=9,leading=11),
}
def clean(s):return html.escape(s).replace('–','-').replace('—','-')
def para(s,key='body'):return Paragraph(clean(s),styles[key])
story=[];lines=source.splitlines();i=0;pending_caption=None
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('Table '):pending_caption=para(line,'caption');i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            row=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',c) for c in row):rows.append(row)
            i+=1
        n=len(rows[0]);width=489.6
        if n==3:ws=[width*.43,width*.23,width*.34]
        elif n==5:ws=[width*.18]+[width*.205]*4
        elif n==6:ws=[width*.16]+[width*.168]*5
        elif n==7:ws=[width*.16]+[width*.14]*6
        else:ws=[width/n]*n
        t=Table([[para(c,'head' if ri==0 else 'cell') for c in row] for ri,row in enumerate(rows)],colWidths=ws,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EFF3')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        block=([pending_caption] if pending_caption else [])+[t,Spacer(1,10)]
        story.append(KeepTogether(block));pending_caption=None;continue
    m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
    if m:
        path=ROOT/m.group(2);w,h=PILImage.open(path).size;im=Image(str(path),width=480,height=480*h/w)
        j=i+1
        while j<len(lines) and not lines[j].strip():j+=1
        if j<len(lines) and lines[j].startswith('Figure '):story.append(KeepTogether([im,Spacer(1,6),para(lines[j],'caption')]));i=j+1
        else:story.append(im);i+=1
        continue
    if line.startswith('#'):
        level=len(line)-len(line.lstrip('#'));story.append(para(line.lstrip('# ').strip(),'title' if level==1 else ('h1' if level==2 else 'h2')))
    else:story.append(para(line,'pending' if line.startswith('[') else 'body'))
    i+=1
def footer(canvas,doc):
    canvas.setFont('TNR',9);canvas.setFillColor(colors.HexColor('#555555'));canvas.drawRightString(550,29,str(doc.page))
out=ROOT/(stem+'.pdf')
SimpleDocTemplate(str(out),pagesize=(612,792),rightMargin=61.2,leftMargin=61.2,topMargin=48,bottomMargin=46,title='Temporal transferability of Italian household energy archetypes between 2013 and 2021',author='').build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
r=PdfReader(out)
main=source.split('## References')[0]
manifest={'body_words_including_tables_and_captions':len(re.findall(r'\b\w+\b',main)),'pages':len(r.pages),'figures':5,'tables':8,'references':17,'pdf_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'pending_sections':[s for s in lines if s.startswith('[')],'docx_status':'Not delivered because required renderer is unavailable'}
manifest['references']=len(re.findall(r'^\d+\. ',source.split('## References')[1],re.M))
(ROOT/('draft_manifest.json' if stem=='first_draft' else stem+'_manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='pending_sections'},indent=2))
