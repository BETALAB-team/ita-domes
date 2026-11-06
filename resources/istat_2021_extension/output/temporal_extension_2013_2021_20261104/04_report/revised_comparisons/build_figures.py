from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,Color,black,white
from pdf2image import convert_from_path
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];A=ROOT/'02_analysis';REPO=ROOT.parents[1]
F=HERE/'figures';F.mkdir(exist_ok=True)
POP='C:/Users/khajmoh18975/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
pdfmetrics.registerFont(TTFont('Paper','C:/Windows/Fonts/times.ttf'))
pdfmetrics.registerFont(TTFont('PaperBold','C:/Windows/Fonts/timesbd.ttf'))
COL=['#4169E1','#50C878','#FFBF00','#D50032'];GRAY=HexColor('#D4D4D4')
data={}
def text(c,x,y,s,size=10,align='left',bold=False):
    c.setFillColor(black);c.setFont('PaperBold' if bold else 'Paper',size)
    getattr(c,{'left':'drawString','center':'drawCentredString','right':'drawRightString'}[align])(x,y,str(s))
def finish(c,name):
    c.save();convert_from_path(str(F/(name+'.pdf')),dpi=240,poppler_path=POP)[0].save(F/(name+'.png'))
def chart(name,w,h):return canvas.Canvas(str(F/(name+'.pdf')),pagesize=(w,h))
def axes(c,x,y,w,h,yticks,ymax,xlabel=None,ylabel=None):
    c.setLineWidth(.35)
    for v in yticks:
        yy=y+h*v/ymax;c.setStrokeColor(GRAY);c.line(x,yy,x+w,yy)
        text(c,x-5,yy-3,f'{v:g}',8.5,'right')
    c.setStrokeColor(black);c.rect(x,y,w,h,fill=0,stroke=1)
    if xlabel:text(c,x+w/2,y-27,xlabel,10,'center')
    if ylabel:
        c.saveState();c.translate(x-34,y+h/2);c.rotate(90);text(c,0,0,ylabel,10,'center');c.restoreState()
def legend(c,x,y,size=9):
    for j,i in enumerate([3,2,1,0]):
        c.setFillColor(HexColor(COL[i]));c.rect(x,y-j*14,14,8,fill=1,stroke=0)
        text(c,x+20,y-j*14,f'Cluster {i+1}',size)

# Figure 5 reproduces the original colour order, stacked bars and household/person split.
s=pd.read_csv(A/'2021_fixed_proxy_summary.csv');name='cluster_proportions';c=chart(name,570,300)
for panel,year,vals in [(0,2013,[[40.1,26.9,22.9,10.1],[29.9,40.2,21.3,8.6]]),(1,2021,[s.share_pct.tolist(),s.people_share_pct.tolist()])]:
    x=45+panel*286;y=48;w=220;h=192
    axes(c,x,y,w,h,[0,.2,.4,.6,.8,1],1,ylabel='Proportion of Total' if panel==0 else None)
    text(c,x+w/2,257,str(year),12,'center',True)
    for j,v in enumerate(vals):
        xx=x+32+j*100;bottom=y
        for i,pct in enumerate(v):
            hh=h*pct/100;c.setFillColor(HexColor(COL[i]));c.rect(xx,bottom,56,hh,stroke=0,fill=1)
            c.setFillColor(white);c.setFont('Paper',9);c.drawCentredString(xx+28,bottom+hh/2-3,f'{pct:.1f}%');bottom+=hh
        text(c,xx+28,y-13,['households','people'][j],10,'center')
for j,i in enumerate([3,2,1,0]):
    c.setFillColor(HexColor(COL[i]));c.rect(112+j*95,281,14,8,fill=1,stroke=0)
    text(c,132+j*95,281,f'Cluster {i+1}',9)
finish(c,name);data[name]={'2013_households':[40.1,26.9,22.9,10.1],'2013_people':[29.9,40.2,21.3,8.6],'2021_households':s.share_pct.tolist(),'2021_people':s.people_share_pct.tolist()}

# Published cooling series follows the paper's thin black line, markers and grid.
name='cooling_availability';c=chart(name,570,190);x,y,w,h=48,37,493,126
axes(c,x,y,w,h,[0,20,40,60],60,ylabel='Households (%)')
yrs=[2013,2021,2024];v=[29.4,48.8,56.0];points=[]
for yr,val in zip(yrs,v):
    xx=x+20+(w-40)*(yr-2013)/11;yy=y+h*val/60;points.append((xx,yy))
    text(c,xx,y-15,str(yr),10,'center');text(c,xx,yy+10,f'{val:.1f}%',10,'center')
c.setStrokeColor(black);c.setLineWidth(.8)
for aa,bb in zip(points[:-1],points[1:]):c.line(*aa,*bb)
for xx,yy in points:c.setFillColor(black);c.circle(xx,yy,2.2,fill=1,stroke=0)
finish(c,name);data[name]={'years':yrs,'official_pct':v}

# Geographic figures use the paper's same four-category colour scale.
regional=pd.read_csv(A/'regional_common_cohort_changes.csv')
regions=['Piemonte',"Valle d'Aosta",'Lombardia','Trentino-Alto Adige','Veneto','Friuli-Venezia Giulia','Liguria','Emilia-Romagna','Toscana','Umbria','Marche','Lazio','Abruzzo','Molise','Campania','Puglia','Basilicata','Calabria','Sicilia','Sardegna']
name='regional_proportions';c=chart(name,570,370)
for panel,year in enumerate([2013,2021]):
    x=124+panel*222;y=35;w=208;h=285
    text(c,x+w/2,343,str(year),12,'center',True)
    for i,rname in enumerate(regions):
        yy=y+h-(i+1)*14.2
        if panel==0:text(c,x-7,yy+2,rname,8.6,'right')
        rr=regional[regional.region==i+1].sort_values('archetype');left=x
        for k,row in enumerate(rr.itertuples()):
            pct=getattr(row,f'weighted{year}');ww=w*pct/100
            c.setFillColor(HexColor(COL[k]));c.rect(left,yy,ww,11.2,fill=1,stroke=0);left+=ww
    for v in [0,.25,.5,.75,1]:
        xx=x+w*v;text(c,xx,y-6,f'{v:g}',8.5,'center')
    text(c,x+w/2,8,'Proportion of Total',10,'center')
for i in range(4):
    c.setFillColor(HexColor(COL[i]));c.rect(145+i*90,359,12,7,fill=1,stroke=0);text(c,161+i*90,358,f'Cluster {i+1}',8.5)
finish(c,name);data[name]=regional.to_dict(orient='records')

# Native hour responses, same frequency and completeness rule as the frozen analysis.
boxstats={};oldpath=REPO/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt'
newpath=REPO/'istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt'
for year,path,sep,sizecol,types in [
 (2013,oldpath,'\t','q_1_1_sq1',[('heating','q_3_10',['q_3_11A','q_3_11B','q_3_11C']),('cooling','q_5_7',['q_5_8A','q_5_8B','q_5_8C'])]),
 (2021,newpath,'^','ncomp_mfr',[('heating','v3_7',['v3_8a','v3_8b','v3_8c']),('cooling','v5_6',['v5_7a','v5_7b','v5_7c'])])]:
    cols=[sizecol]+[v for _,freq,cs in types for v in [freq,*cs]]
    df=pd.read_csv(path,sep=sep,usecols=cols,low_memory=False)
    df=df[pd.to_numeric(df[sizecol],errors='coerce').between(1,5)]
    for kind,freq,cs in types:
        b=df[cs].apply(pd.to_numeric,errors='coerce');ok=pd.to_numeric(df[freq],errors='coerce').isin([1,2])&b.notna().all(axis=1)&b.ge(0).all(axis=1)&b.le(8).all(axis=1)
        vals=b.loc[ok].sum(axis=1).to_numpy();q1,med,q3=np.quantile(vals,[.25,.5,.75]);iqr=q3-q1
        lo=vals[vals>=q1-1.5*iqr].min();hi=vals[vals<=q3+1.5*iqr].max()
        boxstats[f'{year}_{kind}']={'n':len(vals),'q1':float(q1),'median':float(med),'q3':float(q3),'low':float(lo),'high':float(hi),'mean_unweighted':float(vals.mean())}
        source=pd.read_csv(A/'full_size1_5_direct_regional.csv');rr=source[(source.year==year)&(source.region==0)&(source.metric==kind+'_conditional_hours')].iloc[0]
        assert len(vals)==rr.valid_n and abs(vals.mean()-rr.unweighted)<1e-9
name='operating_hours';c=chart(name,570,253)
for panel,kind in enumerate(['heating','cooling']):
    x=45+panel*285;y=44;w=220;h=166
    axes(c,x,y,w,h,[0,4,8,12,16,20,24],24,ylabel='Daily On-Time (hours)' if panel==0 else None)
    text(c,x+w/2,233,'(a) Heating' if panel==0 else '(b) Cooling',11,'center')
    for j,yr in enumerate([2013,2021]):
        b=boxstats[f'{yr}_{kind}'];xx=x+62+j*96;sc=lambda v:y+h*v/24
        c.setStrokeColor(HexColor('#555555'));c.setLineWidth(.5);c.line(xx,sc(b['low']),xx,sc(b['high']))
        for v in ['low','high']:c.line(xx-10,sc(b[v]),xx+10,sc(b[v]))
        fills=['#EFD5B8','#DFAD83'] if kind=='heating' else ['#ABC2D8','#233854']
        c.setFillColor(HexColor(fills[j]));c.rect(xx-22,sc(b['q1']),44,sc(b['q3'])-sc(b['q1']),fill=1,stroke=1)
        c.line(xx-22,sc(b['median']),xx+22,sc(b['median']));text(c,xx,y-15,str(yr),10,'center')
finish(c,name);data[name]=boxstats

# Figure 3: original 2013 panel rendered directly, followed by 2021 values.
name='within_cluster_sum_squares';c=chart(name,570,235)
c.drawImage(str(HERE/'reference/paper_wcss_2013.png'),5,18,width=271,height=196,preserveAspectRatio=True,mask='auto')
text(c,145,222,'2013',12,'center',True)
x,y,w,h=338,44,214,153
axes(c,x,y,w,h,[800,1200,1600,2000,2400],2500,xlabel='Number of Clusters',ylabel='Within Cluster Sum of Squares')
kdf=pd.read_csv(A/'2021_k_search.csv');points=[]
for r in kdf.itertuples():
    xx=x+8+(w-16)*(r.k-2)/6;yy=y+h*r.inertia/2500;points.append((xx,yy));text(c,xx,y-13,str(r.k),9,'center')
for aa,bb in zip(points[:-1],points[1:]):c.setStrokeColor(black);c.setLineWidth(.8);c.line(*aa,*bb)
for xx,yy in points:c.setFillColor(black);c.circle(xx,yy,2.1,fill=1,stroke=0)
text(c,x+w/2,222,'2021',12,'center',True);finish(c,name)
data[name]={'2013':'Unaltered graphical result from original paper Figure 3; no refit','2021':kdf.to_dict(orient='records')}

# A report-only demographic table calculated from the unchanged assignment output.
assigned=pd.read_csv(A/'assigned_2021_proxy.csv');demo=[]
for aa,g in assigned.groupby('archetype'):
    row={'archetype':int(aa),'n':len(g)}
    for key in ['min_age','max_age','number','studio_ind']:
        row[key+'_mean']=float(g[key].mean());row[key+'_sd']=float(g[key].std(ddof=1))
        target=float(s.loc[s.archetype==aa,key].iloc[0]);assert abs(row[key+'_mean']-target)<1e-9
    demo.append(row)
(HERE/'table3_2021_statistics.json').write_text(json.dumps(demo,indent=2),encoding='utf-8')
(HERE/'figure_data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps({'figures':len(list(F.glob('*.png'))),'new_boxplot_statistics_verified_against_frozen_means':True,'table3_2021_statistics':demo},indent=2))
