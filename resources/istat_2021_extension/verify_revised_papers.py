"""Structural checks and PDF page renders/contact sheets for visual inspection."""
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
from pypdf import PdfReader
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent


def main(render=True,config_path=ROOT/'revised_paper_config.json',years=(2013,2021)):
    cfg=json.loads(Path(config_path).read_text())
    root=ROOT/cfg['output_root'];pdf_dir=ROOT/cfg['pdf_directory']
    poppler=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
    qa=[]
    for year in years:
        path=pdf_dir/f'italian_user_archetypes_corrected_{year}.pdf'
        reader=PdfReader(str(path));text='\n'.join(p.extract_text() for p in reader.pages)
        figures=[str(i) for i in range(1,11)]+['A.1','B.1']
        if year == 2021: figures.append('11')
        assert all(f'Figure {n}.' in text for n in figures)
        assert all(f'Table {n}.' in text for n in range(1,9))
        assert 'Algorithm 1.' in text and 'Algorithm 2.' in text
        assert '6. Conclusions' in text and 'References' in text
        assert '\ufffd' not in text
        if year == 2021:
            assert 'Table 9.' in text and 'v3_14gh_new' in text
            assert 'insulation unavailable' not in text.lower()
            assert 'Insulation-stratified heating and appliance-electricity distributions are unavailable' not in text
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        record={'year':year,'pages':len(reader.pages),'words':len(text.split()),
                'sha256':digest,'figure_table_checks':True,'rasterized':False,'visual_review_complete':False}
        if render:
            folder=ROOT/f'tmp/pdfs/revised/qa_{year}';folder.mkdir(parents=True,exist_ok=True)
            subprocess.run([str(poppler),'-r','110','-png',str(path),str(folder/'page')],check=True)
            # Ignore renders from a previous, longer draft; only inspect pages
            # emitted for the current PDF. Cleanup is limited to this QA folder.
            digits=len(str(len(reader.pages)))
            pages=[folder/f'page-{i:0{digits}d}.png' for i in range(1,len(reader.pages)+1)]
            assert all(page.is_file() for page in pages)
            for first in range(0,len(pages),6):
                canvas=Image.new('RGB',(1200,1150),'#cbd0d5');draw=ImageDraw.Draw(canvas)
                for j,page in enumerate(pages[first:first+6]):
                    tile=Image.open(page).convert('RGB');tile.thumbnail((382,536))
                    x=(j%3)*400+(400-tile.width)//2;y=(j//3)*575+24
                    canvas.paste(tile,(x,y));draw.text(((j%3)*400+12,(j//3)*575+5),f'{year} - page {first+j+1}',fill='black')
                canvas.save(folder/f'contact_{first+1:02d}.png')
            record['rasterized']=True
        qa.append(record)
    (root/'pdf_verification.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'revised_paper_config.json')
    parser.add_argument('--year',type=int,choices=[2013,2021])
    args=parser.parse_args();main(not args.no_render,args.config,(args.year,) if args.year else (2013,2021))
