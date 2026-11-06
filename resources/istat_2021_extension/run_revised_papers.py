"""Regenerate corrected 2013 and 2021 analyses and both vector PDF manuscripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def find_pdf_python():
    configured=os.environ.get('ISTAT_PDF_PYTHON')
    candidates=[Path(configured)] if configured else []
    candidates += [Path(sys.executable),Path(os.environ['USERPROFILE'])/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe']
    for path in candidates:
        if not path.is_file():continue
        check=subprocess.run([str(path),'-c','import reportlab,pypdf,PIL'],capture_output=True)
        if check.returncode==0:return path
    raise RuntimeError('Install reportlab and pypdf, or set ISTAT_PDF_PYTHON to a Python environment containing them.')


def refresh_manifest(config_path):
    """Snapshot current evidence after manuscript generation and/or visual QA."""
    config_path=Path(config_path).resolve()
    cfg=json.loads(config_path.read_text())
    out=ROOT/cfg['output_root'];pdf_dir=ROOT/cfg['pdf_directory']
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    evidence=json.loads((out/'results.json').read_text())
    code=[ROOT/name for name in ['revised_analysis.py','revised_figures.py',
          'build_revised_papers.py','run_revised_papers.py','verify_revised_papers.py',
          'revise_2021_equipment.py','audit_appliance_model.py','model_electricity_2021.py']]+[config_path]
    manifest={'inputs':evidence['inputs'],'code':{str(path):digest(path) for path in code},
              'outputs':{str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name!='manifest.json'},
              'pdfs':{str(path):digest(path) for path in sorted(pdf_dir.glob('italian_user_archetypes_corrected_*.pdf'))}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'revised_paper_config.json')
    parser.add_argument('--manuscripts-only',action='store_true',help='Reuse existing corrected result tables and figures.')
    args=parser.parse_args()
    config=args.config.resolve()
    pdf_python=find_pdf_python()
    if not args.manuscripts_only:
        subprocess.run([str(pdf_python),str(ROOT/'audit_appliance_model.py')],cwd=ROOT,check=True)
        subprocess.run([str(pdf_python),str(ROOT/'model_electricity_2021.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'revised_analysis.py'),'--config',str(config)],cwd=ROOT,check=True)
    subprocess.run([str(pdf_python),str(ROOT/'build_revised_papers.py'),'--config',str(config)],cwd=ROOT,check=True)
    subprocess.run([str(pdf_python),str(ROOT/'verify_revised_papers.py'),'--config',str(config),'--no-render'],cwd=ROOT,check=True)
    refresh_manifest(config)
    cfg=json.loads(config.read_text())
    print('PDF outputs:',ROOT/cfg['pdf_directory'],flush=True)


if __name__=='__main__':main()
