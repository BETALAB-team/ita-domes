"""Rebuild only the isolated 2021 analysis from the existing repository inputs.

Run with the bundled Python from the repository resources/istat_data directory.
Scientific packages are task-local; the 2013 scripts and output paths are never run.
"""
from pathlib import Path
import sys,runpy,os,json,hashlib
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'.scientific'));sys.path.insert(0,str(REPO));os.chdir(REPO)
stages=['01_inputs/build_inputs.py','02_probabilities/run_probabilities.py','02_probabilities/plot_probabilities.py','02_probabilities/export_equation.py','03_energy/run_energy_2021.py','04_figures/build_article_figures.py']
for relative in stages:
 print('Running '+relative,flush=True);runpy.run_path(str(ROOT/relative),run_name='__main__')
print('2021 analysis finished. Rebuild 06_report/build_report.py and export_word_pdf.ps1 for the document.',flush=True)
