"""Refresh only display labels and plot layout from the fitted energy outputs."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'.scientific'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch,Rectangle
F=ROOT/'04_figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.linewidth':.6,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none','savefig.facecolor':'white'})
factors=['location','dwelling_type','archetype']
maps={p:pd.read_csv(HERE/f'{p}_group_lookup_2021.csv') for p in ['H','E']}
stats={p:pd.read_csv(HERE/f'{p}_group_summary_2021.csv') for p in ['H','E']}
domains={r:pd.read_csv(HERE/f'{p}_household_assignments_2021.csv',low_memory=False) for r,p in [('heating_daily_hours','H'),('electricity_annual_kwh','E')]}
d=pd.read_csv(HERE/'analysis_households_2021.csv',low_memory=False)
results=json.loads((HERE/'results.json').read_text())['groups']
source=(HERE/'run_energy_2021.py').read_text(encoding='utf-8')
import hashlib
exec(source[source.index('# Appendix A:'):])
