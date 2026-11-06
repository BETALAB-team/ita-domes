from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'.scientific'))
import numpy as np,pandas as pd
from statsmodels.api import OLS
from document_helpers import Report,OxmlElement,qn,Inches,Pt,RGBColor,WD_ALIGN_PARAGRAPH,WD_CELL_VERTICAL_ALIGNMENT
import matplotlib
sys.stdout.reconfigure(encoding='utf-8')
P=ROOT/'02_probabilities';E=ROOT/'03_energy';F=ROOT/'04_figures'
r=Report(HERE)
S=pd.read_csv(P/'archetype_summary_2021.csv').sort_values('archetype')
B=pd.read_csv(P/'polynomial_coefficient_matrix.csv',index_col=0).to_numpy()
diag=json.loads((P/'polynomial_fit_diagnostics.json').read_text());ps=json.loads((P/'results.json').read_text());es=json.loads((E/'results.json').read_text())
H=pd.read_csv(E/'H_group_summary_2021.csv');EE=pd.read_csv(E/'E_group_summary_2021.csv')
for df in [H,EE]:df['rank']=df.group.str[1:].astype(int);df.sort_values('rank',inplace=True)
BASE=REPO/'output/temporal_extension_2013_2021_20261104/00_baseline'
published=json.loads((BASE/'published_benchmarks.json').read_text())
def pub(group,key):return float(next(row['value'] for row in published if row['group']==group and row['metric']==key))
def f_p(p):return '<0.001' if p<.001 else f'{p:.3f}'
def figure(name,width=7):r.figure(F/f'{name}.png',width)
def primary_design(df):
 a=df.building_age_years.to_numpy()/100;s=df.dwelling_area_m2.to_numpy()/100
 return np.column_stack((np.ones(len(df)),a,s,a*s,a*a,s*s))
t=pd.read_csv(P/'empirical_vs_fitted_training.csv');X=primary_design(t)
regression=[];models=[]
for a,b in enumerate(B,1):
 keep=np.flatnonzero(b!=0);model=OLS(t[f'prob_archetype_{a}'].to_numpy(),X[:,keep]).fit();models.append(model)
 assert np.max(np.abs(model.params-b[keep]))<1e-10
 for j in range(6):regression.append({'archetype':a,'term_index':j,'coefficient':b[j],'p':float(model.pvalues[list(keep).index(j)]) if j in keep else np.nan})
pd.DataFrame(regression).to_csv(HERE/'regression_coefficients_and_pvalues_2021.csv',index=False)

# Page 1
r.p('2021 reconstruction of Italian user archetypes and energy groups','Title')
r.p('Dwelling probabilities coefficient matrix heating and electricity groups and assignment tables')
r.p('4 November 2026')
r.p('We applied the article’s analysis sequence to the 2021 survey: household archetypes, dwelling-conditioned probabilities, polynomial regression, energy-group reduction and an Appendix A assignment matrix. Every 2021 statistical figure in this report is generated from that new run. The published 2013 results remain the historical reference.')
r.caption('Table 1','The published 2013 results and the 2021 reconstruction.',True)
r.table(['Output','2013 published result','2021 fitted result'],[
 ['Household archetype shares A1-A4','40.1 / 26.9 / 22.9 / 10.1%','24.20 / 32.87 / 21.76 / 21.16%'],
 ['Heating groups','H1-H7','H1-H9'],
 ['Electricity groups listed in Table 6','E1-E5','E1-E13'],
 ['Heating-group mean range','4.67-9.67 h/day',f'{H["mean"].min():.2f}-{H["mean"].max():.2f} h/day'],
 ['Appliance electricity group mean range','1801-2895 kWh/year',f'{EE["mean"].min():.0f}-{EE["mean"].max():.0f} kWh/year'],
],[2.7,2.2,2.2],10)
r.p('In 2013, A1 was the largest published household type. In the 2021 fit, A2 is largest at 32.87%, while A4 accounts for 21.16%. The new grouping procedure produces nine heating groups and thirteen electricity groups. Group names are ordered by each year’s fitted means; the same suffix denotes a rank within that year.')
r.caption('Table 2','Households contributing to the 2021 reconstruction.',True)
r.table(['Stage','Households'],[
 ['Raw 2021 release','54,069'],['Demographic fit','52,617'],['Dwelling probability stage','46,712'],['Heating-group stage','48,690'],['Electricity-group stage','52,617']
],[5.2,1.9],10)
r.p('The appendix covers all four locations, five 2021 dwelling categories and four user archetypes: 80 combinations and 160 H/E assignments. For example, a North West single-family dwelling occupied by user type A2 maps to H9 and E13 in the new table. Their group means are 11.11 h/day and 2843 kWh/year, respectively.')
r.note('Households contribute equally to the primary clustering and empirical cell ratios; the polynomial gives equal weight to each of the 21 fitting windows. Separate PESO-weighted shares, group means and probability regressions are supplied. The 2021 apartment categories describe floor position; the 2013 apartment categories describe building size. Appliance electricity is an ownership/use model with fixed reference coefficients.')

# Page 2
r.page();r.h('Definition of user archetypes')
r.p('The 2021 four-cluster fit uses the article’s loneliness and generation coordinates and an age-adjusted schooling index. Its labels are aligned with the historical interpretations. In A1, mean maximum age is 81.2 years, compared with the paper’s 78.3. In A2, mean household size is 3.2 members, compared with 3.7 in 2013. Table 3 gives the corresponding profile summaries.')
r.caption('Table 3','User archetypes derived from k-means clustering.',True)
r.note('2013 published values. Average (standard deviation).')
fields=['min_age','max_age','number','studio_ind']
r.table(['cluster','age min','age max','n members','education'],[[a]+[f'{pub("A"+str(a),f+"_mean"):.1f} ({pub("A"+str(a),f+"_sd"):.1f})' for f in fields] for a in range(1,5)],[.55,1.65,1.65,1.55,1.7],9.5)
r.note('2021 fitted values. Average (standard deviation).')
r.table(['cluster','age min','age max','n members','education'],[[int(z.archetype)]+[f'{getattr(z,f+"_mean"):.1f} ({getattr(z,f+"_sd"):.1f})' for f in ['min_age','max_age','number','education']] for z in S.itertuples()],[.55,1.65,1.65,1.55,1.7],9.5)
figure('archetype_population_shares_2021')
r.caption('Fig. 5.','Proportion of each cluster among households and people.')
r.note('The 2013 panel retains the printed shares. The 2021 panel uses the new 52,617-household fit. Its survey-weighted household shares are 28.31%, 27.70%, 27.65% and 16.35%. The schooling and age assumptions are stated on page 10; all 2021 outputs use these same primary labels.')

# Page 3
r.page();r.h('Associating user archetypes to building types')
r.p('We reconstructed windowed empirical probabilities using the article’s 20-year building-age grid, 40 m² area grid and centred half-windows of 6 years and 8 m². The eligible dwelling cohort has 46,712 records; 12,334 fall inside at least one of the 63 grid windows. Fifty-seven windows have at least five observations. The primary polynomial uses the 21 supported windows centred at ages 20-140 years and areas 40, 80 and 120 m².')
r.p('Let a be building age in years and s dwelling area in m². The fitted 2021 matrix, rounded to five decimal places, is')
r.matrix(B)
r.note('The matrix product is the fitted part; ε denotes the four residuals. The full-precision coefficient CSV is authoritative. Basis order is 1, a/100, s/100, a s/10000, a²/10000, s²/10000. The six-term starting basis follows the article’s matrix; backward elimination retains terms at p≤0.05, with the intercept retained.')
r.caption('Table 4','Regression results.',True)
regdf=pd.DataFrame(regression);terms=['intercept','age/100','area/100','age × area/10000','age²/10000','area²/10000']
regrows=[]
for j,term in enumerate(terms):
 row=[term]
 for a in range(1,5):
  rr=regdf[(regdf.archetype==a)&(regdf.term_index==j)].iloc[0]
  row.extend([f'{rr.coefficient:.5f}' if rr.coefficient else '0',f_p(rr.p) if pd.notna(rr.p) else '-'])
 regrows.append(row)
regrows.extend([['F-statistic']+sum(([f'{m.fvalue:.2f}',''] for m in models),[]),['Adjusted R²']+sum(([f'{z["adjusted_R2"]:.3f}',''] for z in diag),[]),['RMSE']+sum(([f'{z["RMSE"]:.4f}',''] for z in diag),[])])
r.table(['Variable']+sum(([f'Prob {a}\ncoefficient','p-value'] for a in range(1,5)),[]),regrows,[1.25]+[.73125]*8,8.2)
r.p('The selected 2021 equation relates A1 mainly to building age, A2 to age and area, A3 to a linear combination of age and area, and A4 to squared area. This changes the fitted shapes from the 2013 equation while preserving the same six-column representation.')
r.note('At the 21 fitting centres, individual estimates lie within 0-1 and their four-value sums range from 0.98044 to 1.03517. On the full 63-point grid, 35 points have an individual estimate outside 0-1, all beyond the primary area domain. The saved grid preserves those raw range and sum flags. The following probability figures use the primary fit domain.')

# Page 4
r.page();r.h('Dwelling probability profiles')
r.p('In the 2021 equation, A2 probability rises with dwelling area and A3 probability falls with area at a fixed building age. A1 changes with building age. After variable selection, the A4 curve depends on area. Figure 6 places the empirical proportions and fitted curves on the article’s four-panel axes.')
figure('dwelling_probability_profiles_2021')
r.caption('Fig. 6.','Probability of each cluster of users as the dwelling’s age changes. Points are calculated proportions. Lines are predictions of the regression model.')
r.note('Crosses are the 21 empirical fitting windows. Gray lines correspond to building ages 20, 60, 100 and 140 years over areas 40-140 m². In Cluster 4 the selected equation gives coincident age curves. The horizontal axis is dwelling area, matching the original figure’s layout.')
r.p('The full empirical grid and its household counts are exported for inspection. The survey-weighted grid and a separately fitted weighted coefficient matrix are also available, so the weighting choice can be examined without changing the paper-style primary fit.')

# Page 5
r.page();r.h('Alignment between fitted and empirical probabilities')
figure('probability_model_validation_2021',5.2)
r.caption('Fig. 7.','Alignment Between Predicted and Actual Values Across Clusters.')
r.note('The plot uses the 21 fitting windows. Crosses retain the paper’s four cluster colours; the dashed line is y=x. These are apparent fitted values. The household holdout scores below provide a separate diagnostic conditional on the demographic fit.')
hold=pd.read_csv(P/'household_holdout_probability_scores.csv')
r.caption('Table 4a','Published and new probability regression fit summaries.',True)
r.table(['Type','2013\nAdjusted R²','2021\nAdjusted R²','2013\nRMSE','2021\nRMSE','2021 holdout\nRMSE'],[
 [f'A{a}',f'{pub("A"+str(a),"adjusted_R2"):.2f}',f'{diag[a-1]["adjusted_R2"]:.3f}',f'{pub("A"+str(a),"RMSE"):.3f}',f'{diag[a-1]["RMSE"]:.4f}',f'{hold.loc[hold.archetype==a,"RMSE"].iloc[0]:.4f}'] for a in range(1,5)
],[.55,1.25,1.25,1.1,1.1,1.85],9.4)
r.p('The 2013 paper reports adjusted R² values of 0.94, 0.93, 0.87 and 0.84. In the new 2021 window regressions, they are 0.565, 0.556, 0.694 and 0.385. The new residuals are larger, and the household-held-out RMSEs range from 0.0511 to 0.0983. Thus the 2021 reconstruction shows a less tight relationship between the two dwelling predictors and the fitted user types on its supported windows.')

# Page 6
r.page();r.h('Deriving energy patterns for specific occupant groups')
r.p('The energy stage starts from location × dwelling type × user archetype cells. Each response has 80 observed starting cells. We successively merge the pair with the largest Tukey-Kramer adjusted p-value, then order the resulting groups by their response mean. The 2021 reduction makes 71 heating merges and 67 electricity merges, producing H1-H9 and E1-E13.')
r.caption('Table 5','ANOVA results (single variables and interactions).',True)
anova=pd.read_csv(E/'anova_results_2021.csv')
termmap=[('C(archetype)','Usercluster'),('C(dwelling_type)','Dwellingtype'),('C(location)','Location'),('building_age_years','Age dwel'),('dwelling_area_m2','Area dwel'),('C(cooling_system_use)','CoolingSystemUse'),('C(archetype):C(dwelling_type)','Usercluster × Dwellingtype'),('C(archetype):C(location)','Usercluster × Location'),('C(dwelling_type):C(location)','Dwellingtype × Location'),('building_age_years:dwelling_area_m2','Age dwel × Area dwel')]
ar=[]
for term,label in termmap:
 row=[label]
 for response in ['heating_daily_hours','electricity_annual_kwh']:
  z=anova[(anova.response==response)&(anova.term==term)].iloc[0];row.extend([f'{z.F:.2f}',f_p(z.p)])
 ar.append(row)
r.table(['Term','Heating\nF-statistic','p-value','Electricity\nF-statistic','p-value'],ar,[2.8,1.25,.9,1.25,.9],9.4)
r.note('Type-II OLS ANOVAs use 43,569 heating records and 46,712 electricity records with known construction age and area. The original subjective insulation question is absent in 2021, so insulation terms are omitted. The corresponding heating model has R²=0.111; the electricity model has R²=0.266. Regional and dwelling factors retain their 2021 definitions.')
r.p('Location has the largest heating F-statistic among the main categorical factors. The user-archetype effect is pronounced in the appliance electricity model. Interactions among user type, dwelling type and location also contribute to the fitted response patterns. The group procedure below uses those three cell identifiers for both responses, following the final saved group structure and the published Appendix A layout.')
r.p('The final partition contains nine heating groups with means from 6.17 to 11.11 h/day, compared with seven published groups from 4.67 to 9.67 h/day in 2013. It contains thirteen modelled appliance-electricity groups with means from 1621 to 2843 kWh/year, compared with the five groups listed in the paper from 1801 to 2895 kWh/year.')
r.note('The iterative procedure reuses the same observations as groups are merged. Its terminal pairwise tests describe the selected partition. The implementation uses full-precision p-values with a deterministic tie rule. Group labels are ranks within 2021 and are linked to their actual cell memberships in Appendix A.')

# Page 7
r.page();r.h('Heating and electricity group summaries')
r.p('The following means and standard deviations come from the new 2021 household assignments. Each example is an observed location, dwelling and user-type cell belonging to that group. The complete assignments appear in Appendix A.')
abbr={'north_east':'NE','north_west':'NW','central':'Central','south_and_islands':'South/Islands'}
dn={'apartment_ground_basement_raised':'ground/semi-basement/raised-floor flat','apartment_intermediate_floor':'intermediate-floor flat','apartment_top_floor':'top-floor flat','single_family_house':'single-family house','multi_family_house':'multi-family house'}
def group_rows(prefix,summary):
 z=pd.read_csv(E/f'{prefix}_household_assignments_2021.csv',usecols=['group','location','dwelling_type','archetype'])
 counts=z.groupby(['group','location','dwelling_type','archetype']).size().rename('n').reset_index();rows=[]
 for row in summary.itertuples():
  cell=counts[counts.group==row.group].sort_values('n',ascending=False).iloc[0]
  example=f'A{int(cell.archetype)}, {dn[cell.dwelling_type]}, {abbr[cell.location]}'
  rows.append([row.group,example,f'{row.mean:.2f} ({row.standard_deviation:.2f})' if prefix=='H' else f'{row.mean:.0f} ({row.standard_deviation:.0f})',f'{row.sample_count:,}'])
 return rows
r.caption('Table 6a','Significant groups for heating system usage.',True)
r.table(['Group','Example','Average (SD)\nh/day','n'],group_rows('H',H),[.5,3.6,1.7,1.3],9)
r.caption('Table 6b','Significant groups for annual electricity consumption',True)
r.table(['Group','Example','Average (SD)\nkWh/year','n'],group_rows('E',EE),[.5,3.6,1.7,1.3],9)
r.note('NE=North East; NW=North West. Sample sizes are unweighted. PESO-weighted group means are supplied in separate CSVs. Electricity includes the audited appliance-model components and reference assumptions described on page 11.')

# Page 8
r.page();r.h('Daily heating on time and annual appliance electricity')
figure('energy_group_boxplots_2021',6.85)
r.caption('Fig. 8.','Variation of (a) daily heating system operating hours and (b) annual electricity consumption based on significant groups of dwelling types, geographical location, and user type.')
r.note('All boxes use the new 2021 assignments. Declared heating hours refer to cold months of 2020. Heating shows H1-H9 without an insulation split because that subjective rating is absent in the 2021 release. Electricity shows E1-E13 split by declared cooling use. Boxes span the 25th-75th percentiles, with medians and non-outlying whiskers; outliers are hidden. The displayed outcomes and group axes follow the paper.')

# Page 9
r.page();r.h('Appendix A')
r.caption('Figure A.1.','Assignment of user groups based on building type, location and user archetype.',True)
wide=pd.read_csv(E/'appendix_A_assignment_2021_wide.csv');headers=['Location','Dwelling Type']+[f'User Cluster {a}' for a in range(1,5)]*2
data=wide.astype(str).values.tolist();tt=r.table(headers,data,[.45,2.05]+[.575]*8,8.1)
# Turn the first row into the paper's two-tier merged header.
top=OxmlElement('w:tr');tt._tbl.insert(2,top)
# python-docx row insertion is safer through copied header cells before merges.
from copy import deepcopy
top_parent=top.getparent();top_parent.remove(top)
copied=deepcopy(tt.rows[0]._tr);tt._tbl.insert(list(tt._tbl).index(tt.rows[0]._tr),copied)
for cell in tt.rows[0].cells:cell.text=''
tt.cell(0,0).merge(tt.cell(1,0)).text='Location';tt.cell(0,1).merge(tt.cell(1,1)).text='Dwelling Type'
tt.cell(0,2).merge(tt.cell(0,5)).text='Heating Group';tt.cell(0,6).merge(tt.cell(0,9)).text='Electricity Group'
for j in range(8):tt.cell(1,j+2).text=f'User\nCluster\n{j%4+1}'
direction=OxmlElement('w:textDirection');direction.set(qn('w:val'),'btLr');tt.cell(0,0)._tc.get_or_add_tcPr().append(direction)
for row in tt.rows[:2]:
 for c in row.cells:
  pr=c._tc.get_or_add_tcPr();bd=OxmlElement('w:tcBorders')
  for edge in ['top','bottom','left','right']:
   el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'3');el.set(qn('w:color'),'AAAAAA');bd.append(el)
  pr.append(bd)
  for pp in c.paragraphs:pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
heats=H.group.tolist();elecs=EE.group.tolist();hc={g:matplotlib.colormaps['Oranges'](v) for g,v in zip(heats,np.linspace(.18,.9,len(heats)))};ec={g:matplotlib.colormaps['Blues'](v) for g,v in zip(elecs,np.linspace(.18,.9,len(elecs)))}
for i in range(20):
 row=tt.rows[i+2]
 for j,c in enumerate(row.cells):
  pr=c._tc.get_or_add_tcPr();bd=OxmlElement('w:tcBorders')
  for edge in ['top','bottom','left','right']:
   el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'3');el.set(qn('w:color'),'AAAAAA');bd.append(el)
  pr.append(bd)
  if j>=2:
   g=c.text;rgb=np.array((hc if j<6 else ec)[g][:3]);fill=''.join(f'{round(x*255):02X}' for x in rgb);sh=OxmlElement('w:shd');sh.set(qn('w:fill'),fill);pr.append(sh)
   for pp in c.paragraphs:
    for rr in pp.runs:rr.font.color.rgb=RGBColor(255,255,255) if rgb@np.array([.2126,.7152,.0722])<.43 else RGBColor(0,0,0)
 for pp in row.cells[1].paragraphs:
  pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
for i in [0,5,10,15]:
 cell=tt.cell(i+2,0).merge(tt.cell(i+6,0));cell.text=wide.iloc[i]['Location'];pr=cell._tc.get_or_add_tcPr();direction=OxmlElement('w:textDirection');direction.set(qn('w:val'),'btLr');pr.append(direction)
 for pp in cell.paragraphs:
  pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
  for rr in pp.runs:rr.font.size=Pt(8.5)
for row in tt.rows:
 for c in row.cells:
  c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
  for pp in c.paragraphs:
   for rr in pp.runs:rr.font.name='Times New Roman';rr.font.size=Pt(8.1)
r.note('New 2021 fitted lookup. The three apartment rows follow the released ground/semi-basement/raised, intermediate and top-floor categories. Group means are in Table 6. H and E labels are within-year ranks. All 160 assignments are also supplied as long and wide CSV tables, and as standalone SVG and PNG graphics.')

# Page 10
r.page();r.h('Model construction and sensitivity')
figure('clustering_elbow_2021',5.9)
r.caption('Fig. 3.','Within Cluster Sum of Squares as a function of number of clusters.')
r.p('The fresh 2021 fit uses K=4 to retain the article’s four-type resolution. K-means uses 20 initializations, seed 42, a 500-iteration limit and tolerance 10^-8. Seed 42 draws integer ages uniformly within released intervals, bounding ages 80+ at 100. Loneliness and generation use a new 2021 min-max transform; the bounded schooling index is used directly. An orthogonal full PCA is recorded. Semantic labels are aligned to historical reference coordinates. All original 2013 assignments remain fixed.')
r.p('The released 0-10 age groups have blank schooling. Their ages and member counts are retained, while these blank entries are omitted from the schooling mean. For members with supported schooling, the index uses education years divided by age-specific potential schooling, capped at one. Primary coding assigns five years to the merged no-credential/primary category and twelve years to the upper-secondary category.')
sens=pd.read_csv(P/'archetype_sensitivities.csv')
r.caption('Table 7','Archetype shares under schooling assumptions.',True)
rows=[['Primary']+[f'{x:.2f}%' for x in S.household_share]]
for key,label in [('schooling_code1_0','Lowest schooling category at 0 years'),('diploma13','Upper-secondary category at 13 years')]:
 ss=sens[sens.scenario==key].sort_values('archetype');rows.append([label]+[f'{x:.2f}%' for x in ss.share])
r.table(['Coding','A1','A2','A3','A4'],rows,[3.1]+[1]*4,9.3)
r.note('Primary demographic coverage is 52,617 households, including 7,737 with a released child aged 0-10. Questionnaire F.10 routes schooling to age 10 or older and relationship codes 1-5. The primary fit excludes 903 otherwise size-eligible profiles with unsupported schooling and 549 households with 6+ members. The lowest schooling-category assumption materially affects the fitted shares; all H/E and PFE outputs use the declared primary coding.')

# Page 11
r.page();r.h('Sources and reproducibility')
r.p('The analysis follows the saved article code and the final published output structure. Family features trace to 02_Families_Analysis.py; the four-type fit to 01_PCA_Exploratory.py and the clustering representation; dwelling windows and polynomial regressions to the 05-series files and pfe_pipeline.py; energy-group reduction and Figure 8 to 07_ANOVA_analysis.py and the recovered iterative grouping function. The final saved electricity input and published Appendix A determine the three-factor cell structure.')
r.p('The legacy files contain exploratory cells with different K values, restricted regression trials and two-factor electricity trials. The new run records the selected article structure: four demographic types, the six-column polynomial basis and location × dwelling × user cells. The new Tukey implementation uses full-precision p-values and deterministic ties. It writes new group memberships and lookup tables while leaving every 2013 source and result unchanged.')
r.sub('Energy input assumptions')
r.p('The annual appliance outcome is rebuilt from 2021 ownership and use using the audited reference model. It includes lighting, small appliances, refrigeration, washing, drying, dishwashing, screens, computers and standby. Reference power and efficiency parameters are held fixed. Missing bulb-hour, small-appliance and refrigerator-detail fields use documented 2013-derived reference assumptions, with lower and upper model scenarios saved in the input package. The 2013 component reconstruction agrees with the stored reference totals within 1.82 × 10^-12 kWh.')
r.p('Unknown construction age is kept out of the probability and age/area ANOVA stages and is retained for H/E grouping. Heating uses frequent-user responses with all three valid daily blocks. Cooling use is established from declared frequencies or explicit absence. The regional expansion weights are PESO; weighted diagnostics are separate from the equal-household fits. Statistical tests describe these selected survey/model records, with survey-mode, reference-period and imputation differences recorded in the methods files.')
r.sub('Primary sources')
r.link('Khajedehi, Vivian and Zarrella (2025). Identification of Italian user archetypes for building energy simulations based on a national survey. Energy and Buildings 344, 115924.','https://doi.org/10.1016/j.enbuild.2025.115924')
r.link('ISTAT. Indagine sui consumi energetici delle famiglie. Native 2021 microdata, record layout, classifications and questionnaire.','https://www.istat.it/microdati/indagine-sui-consumi-energetici-delle-famiglie/')
r.sub('Analysis package')
r.p('The output/article_method_2021_20261104 folder contains audited inputs, primary household labels, empirical and weighted probability grids, coefficient matrices, fit and holdout diagnostics, H/E merge histories and terminal tests, household assignments, group summaries and all Appendix A cells. Figure outputs include SVG, PNG and PDF versions. The report supplies the numerical equation as editable Word mathematics and the appendix as an editable Word table.')
r.p('The independent review reproduces the raw-data cohort, probability calculations and regressions, all H/E merges, ANOVAs and Appendix assignments. The final manifest records protected-source integrity, package versions and output hashes. The original 2013 baseline and prior reports are retained.')

r.save('Italian_user_archetypes_2021_reconstruction.docx')
print(json.dumps({'blocks':len(r.blocks),'tables':len(r.doc.tables),'figures':len(r.doc.inline_shapes),'equations':1},indent=2))
