from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
from pypdf import PdfReader
ROOT=Path(__file__).parent
QA=ROOT.parent.parent/'tmp'/'article_literature'
pages=sorted(QA.glob('page-*.png'))
for batch in range(0,len(pages),6):
    sheet=Image.new('RGB',(1260,1680),'#bdbdbd')
    draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(pages[batch:batch+6]):
        im=Image.open(path).convert('RGB');im.thumbnail((610,790))
        x=10+(j%2)*630;y=25+(j//2)*560
        im.thumbnail((610,530));sheet.paste(im,(x,y))
        draw.text((x,y-18),path.stem,fill='black')
    sheet.save(QA/f'contact-{batch//6+1}.png')
r=PdfReader(ROOT/'literature_revised_draft.pdf')
for i,p in enumerate(r.pages):
    t=p.extract_text()
    print(i+1,len(t),t[:70].replace('\n',' '))
