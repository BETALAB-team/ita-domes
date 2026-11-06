from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OLD=ROOT.parent/'article_method_2021_20261104';REPO=ROOT.parents[1]
sys.path.insert(0,str(OLD/'.scientific'));sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd,matplotlib
from document_helpers import Report,OxmlElement,qn,Pt,RGBColor,WD_ALIGN_PARAGRAPH,WD_CELL_VERTICAL_ALIGNMENT
P=ROOT/'02_probability_candidates';E=ROOT/'03_energy_candidates';F=ROOT/'04_figures'
r=Report(HERE)
r.doc.sections[0].header.paragraphs[0].text='Italian user archetypes   |   Revised 2021 reconstruction'
B=pd.read_csv(P/'recommended_full_quadratic_matrix.csv',index_col=0).to_numpy()
fit=pd.read_csv(P/'recommended_apparent_fit_statistics.csv')
H=pd.read_csv(E/'H_selected_group_summary.csv');EE=pd.read_csv(E/'E_selected_group_summary.csv')
energy=json.loads((E/'results.json').read_text())['results']
cv=pd.read_csv(E/'candidate_curves.csv')
fair=pd.read_csv(P/'fair_geometry_validation_summary.csv').set_index('candidate')
shares=pd.read_csv(OLD/'02_probabilities/archetype_summary_2021.csv')
check=json.loads((ROOT/'05_review/probability_check.json').read_text())
compactcheck=json.loads((ROOT/'05_review/compact_check.json').read_text())
old_rmse=fair.loc['old21_backward','RMSE_mean'];new_rmse=fair.loc['new125_full_quadratic','RMSE_mean']
def picture(name,width=7):r.figure(F/f'{name}.png',width)
def dec(x,k=3):return f'{float(x):.{k}f}'

# 1 Results and the historical comparator
r.p('Italian user archetypes in 2013 and 2021','Title')
r.p('Revised dwelling probabilities and compact heating and electricity groups')
r.p('4 November 2026')
r.p('The 2021 reconstruction now follows the numerical window settings recovered from the saved 2013 regression workflow. It supplies a four-by-six quadratic probability matrix, construction-period curves, four practical heating groups and five electricity groups. The 2013 published results remain fixed throughout the comparison.')
r.caption('Table 1','The published 2013 results and the revised 2021 reconstruction.',True)
r.table(['Output','2013 published result','2021 revised result'],[
 ['Household shares A1–A4','40.1 / 26.9 / 22.9 / 10.1%', ' / '.join(dec(v,2) for v in shares.household_share)+'%'],
 ['Heating groups','H1–H7','H1–H4 compact resolution'],
 ['Electricity groups in Table 6','E1–E5','E1–E5 compact resolution'],
 ['Heating-group mean range','4.67–9.67 h/day',f'{H["mean"].min():.2f}–{H["mean"].max():.2f} h/day'],
 ['Electricity-group mean range','1801–2895 kWh/year',f'{EE["mean"].min():.0f}–{EE["mean"].max():.0f} kWh/year'],
 ['Regression representation','Quadratic matrix','Full quadratic matrix'],
 ],[2.3,2.4,2.4],9.6)
r.p('In 2013, the largest household archetype was A1 at 40.1%. In the 2021 demographic fit, A2 is largest at 32.87%. The new compact energy lookup spans 6.42–10.53 daily heating hours and 1664–2691 modeled annual appliance kWh. Group suffixes rank the means within each year and resolution; the underlying memberships and dwelling definitions are recorded separately.')
r.caption('Table 2','Households contributing to the 2021 analysis.',True)
r.table(['Stage','Households'],[['Raw release','54,069'],['Fixed demographic labels','52,617'],['Dwelling probability stage','46,712'],['Heating-group stage','48,690'],['Electricity-group stage','52,617']],[5.3,1.8],9.5)
r.p(f'The revised regression uses 125 supported overlapping windows. Matched household holdout RMSE falls from {old_rmse:.5f} for the previous reconstruction to {new_rmse:.5f}, a {100*(1-new_rmse/old_rmse):.1f}% reduction. The compact groups retain 95.31% and 95.54% of variation between the 80 starting heating and electricity cell means.')
r.note('The four demographic archetypes are unchanged from the previous 2021 run. This revision updates the dwelling-probability stage and the practical energy-group resolution. The full 9H/13E Tukey results remain available in the previous analysis. Appendix A supplies the revised 160 H/E assignments.')

# 2 Figure 6
r.page();r.h('Associating user archetypes to building types')
picture('dwelling_probability_profiles_2021',7.05)
r.caption('Fig. 6.','Probability of each cluster of users by dwelling area and construction period.')
r.p('The 2021 profiles show A2 becoming more likely as floor area increases, while A3 becomes less likely. A1 generally declines across the larger areas, and A4 rises. The age-by-area term allows each construction-period curve to have its own slope within the common quadratic equation.')
r.note('Crosses are fitted probabilities at the 125 age/area nodes, following the saved plotting implementation. Gray curves average fitted values over the age nodes assigned to each construction period. Periods refer to window centres at survey year 2021; the overlapping ±15-year age windows include neighbouring periods. The 2021 source supports a pre-1950 category. Actual proportions are checked against the fitted values in Fig. 7.')

# 3 Equation and regression table
r.page();r.h('Regression results')
r.p('The saved 2013 regression input contains 25 age positions at five-year steps and five area positions: 40, 64, 88, 112 and 136 m². The recovered generator uses inclusive half-widths of 15 years and 40 m². The revised 2021 fit restores this 125-node primary geometry and gives equal weight to each window.')
r.matrix(B)
r.note('Here a is dwelling age in years at 2021 and s is floor area in m². The CSV matrix retains full numerical precision. The fitted mean is Bφ; ε denotes the empirical-window residual. All four responses use the same six-term basis, which makes their fitted sum equal to one algebraically.')
r.caption('Table 4','Regression results.',True)
terms=['Intercept','Age / 100','Area / 100','Age × Area / 10000','Age² / 10000','Area² / 10000']
rows=[[term]+[dec(v,5) for v in B[:,j]] for j,term in enumerate(terms)]
rows += [['Adjusted R²']+[dec(v,3) for v in fit.adjusted_R2],['RMSE']+[dec(v,5) for v in fit.RMSE],['Fitting windows']+['125']*4]
r.table(['Term','Prob₁','Prob₂','Prob₃','Prob₄'],rows,[2.4]+[1.175]*4,9.5)
r.p('In 2013, the printed adjusted R² values were 0.94, 0.93, 0.87 and 0.84. The corresponding 2021 values are 0.901, 0.855, 0.946 and 0.913. The 2021 RMSE values are 0.01847, 0.02022, 0.01009 and 0.01014.')
r.note('The checked application rectangle is age 15–135 years and area 40–136 m². Analytic extrema confirm that every component lies in [0,1] throughout that rectangle. The full basis and backward-selection versions have essentially tied holdout error; the full basis supplies coherent sums and retains area dependence. Overlapping windows make nominal independent-row coefficient significance unsuitable for selecting the final equation. Wider-area predictions are supplied with explicit range/domain flags.')

# 4 Figure 7 and validation
r.page();r.h('Alignment between predicted and actual values')
picture('probability_model_validation_2021',5.6)
r.caption('Fig. 7.','Alignment Between Predicted and Actual Values Across Clusters.')
r.caption('Table 4a','Matched household holdout comparison.',True)
r.table(['Training geometry and model','RMSE','Count-weighted RMSE'],[
 ['Previous 21-node backward fit',dec(old_rmse,5),dec(fair.loc['old21_backward','countweighted_RMSE_mean'],5)],
 ['Revised 125-node full quadratic',dec(new_rmse,5),dec(fair.loc['new125_full_quadratic','countweighted_RMSE_mean'],5)],
 ['Revised 125-node backward fit',dec(fair.loc['new125_backward','RMSE_mean'],5),dec(fair.loc['new125_backward','countweighted_RMSE_mean'],5)],
 ],[4.1,1.3,1.7],9.3)
r.p('Fig. 7 shows apparent fit to all 125 empirical nodes. Table 4a uses five repeated 70/30 household splits, rebuilding training and test probabilities separately. Every model is assessed against the same 125-node held-out target; the revised full quadratic improves RMSE in all five splits.')
r.note('The comparison assesses the probability stage conditional on fixed demographic labels. The repeated splits support model comparison; they do not provide an untouched external evaluation. Training windows overlap, so 125 nodes do not mean 125 independent samples. The broader historical windows smooth the empirical probability target; the matched comparison keeps that target common to both models.')

# 5 Energy compression selection
r.page();r.h('Selecting compact energy groups')
picture('compact_group_fidelity_2021',7.05)
r.caption('Fig. S1.','Variation retained as the number of practical energy groups increases.')
r.p('The practical partition starts with the same 80 location × dwelling × user cells. For each candidate count K from 1 to 15, the cell means are ordered and divided into K contiguous ranges. Exact dynamic programming minimizes the count-weighted loss of the cell means within each range.')
r.p('The declared selection rule chooses the smallest K whose five-fold household RMSE is within one standard error of the best tested count and that retains at least 95% of the full-data variation between cell means. The 95% floor is a chosen fidelity target. It selects four heating groups and five electricity groups.')
r.caption('Table 5a','Compact resolution and held-out prediction cost.',True)
rows=[]
for p,label,unit in [('H','Heating','h/day'),('E','Electricity','kWh/year')]:
 v=energy[p];z=v['selected_curve'];best=cv[(cv.response==p)&(cv.K==v['best_cv_K'])].iloc[0]
 rows.append([label,str(v['selected_K']),f'{100*z["between_cell_variance_retained"]:.2f}%',f'{z["cv_rmse_mean"]:.4f} {unit}',f'+{z["cv_rmse_mean"]-best.cv_rmse_mean:.4f} {unit}'])
r.table(['Outcome','Groups','Mean variation retained','CV RMSE','Cost vs best tested'],rows,[1.0,.6,1.55,1.95,2.0],9)
r.note('Every validation fold recomputes cell means, ordering, cut points and predictions using training households only. The full-data lookup is then refitted for application. Fold standard errors describe split variability. These are practical approximations of the cells; their within-group distributions remain broad. The compact fit is a separate cell partition and can split previous Tukey memberships.')
r.note('The fidelity choice is visible: at floors of 90%, 95%, 97.5% and 99%, the same rule gives 3/4/6/9 heating groups and 5/5/7/10 electricity groups. Heating’s best tested count is the search boundary K=15. Electricity E5 passes the one-standard-error rule narrowly, so the chosen count can vary with the folds.')

# 6 Group means and examples
r.page();r.h('Deriving energy use patterns for specific occupant groups')
r.p('In 2013, the published heating table contained seven mean-ranked groups and the electricity table contained five. The revised 2021 practical table contains four heating groups and five electricity groups. The new assignments preserve the main location, dwelling and user contrasts with fewer profiles to pass to simulation.')
names={'north_east':'North East','north_west':'North West','central':'Central','south_and_islands':'South and Islands','apartment_ground_basement_raised':'ground/semi-basement/raised-floor apartment','apartment_intermediate_floor':'intermediate-floor apartment','apartment_top_floor':'top-floor apartment','single_family_house':'single-family house','multi_family_house':'multi-family house'}
for p,summary,title,unit in [('H',H,'Daily heating system on-time','h/day'),('E',EE,'Annual appliance electricity','kWh/year')]:
 look=pd.read_csv(E/f'{p}_selected_cell_lookup.csv');rows=[]
 for row in summary.itertuples():
  ex=look[look.group==row.group].sort_values(['location','dwelling_type','archetype']).iloc[0]
  desc=f'{names[ex.location]}, {names[ex.dwelling_type]}, A{int(ex.archetype)}'
  rows.append([row.group,f'{row.mean:.2f} ({row.sd:.2f})',f'{row.n:,}',str(row.cell_count),desc])
 r.caption('Table 6a' if p=='H' else 'Table 6b',f'{title}: selected groups and examples ({unit}).',True)
 r.table(['Group','Mean (SD)','Households','Cells','Example'],rows,[.5,1.35,.8,.45,4.0],9.1)
r.p('The four heating means are 6.42, 7.74, 8.96 and 10.53 h/day. The five appliance-electricity means are 1664, 1856, 2088, 2275 and 2691 kWh/year. The appendix assigns each of the 80 combinations to one heating and one electricity group.')
r.note('A seven-group heating sensitivity retains 98.65% of between-cell mean variance and has CV RMSE 4.7872 h/day; the selected four-group version has RMSE 4.7914 h/day. These measured tradeoffs allow a higher-resolution simulation if needed. The new five-group electricity result shares the historical group count; the membership is fitted to 2021. PESO-weighted means are supplied beside the primary unweighted means.')

# 7 Figure 8
r.page();r.h('Heating and electricity distributions')
picture('energy_group_boxplots_2021',7.0)
r.caption('Fig. 8.','Differences in (a) daily heating system on-time and (b) annual appliance electricity across the selected groups.')
r.note('The panels use the new 2021 H1–H4 and E1–E5 assignments. Heating boxes pool insulation levels because the 2021 release has no matching subjective rating. Electricity boxes retain the article’s cooling-system-use split. The appliance outcome uses reported ownership and use with fixed reference power/efficiency coefficients. Heating responses concern the cold months of 2020.')

# 8 Editable Appendix A
r.page();r.h('Appendix A')
r.caption('Figure A.1.','Assignment of user groups based on building type, location and user archetype.',True)
wide=pd.read_csv(E/'appendix_A_assignment_2021_wide_flat.csv');headers=['Location','Dwelling Type']+[f'User Cluster {a}' for a in range(1,5)]*2
data=wide.astype(str).values.tolist();tt=r.table(headers,data,[.45,2.05]+[.575]*8,8.1)
from copy import deepcopy
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
hc={g:matplotlib.colormaps['Oranges'](v) for g,v in zip(H.group,np.linspace(.18,.9,len(H)))};ec={g:matplotlib.colormaps['Blues'](v) for g,v in zip(EE.group,np.linspace(.18,.9,len(EE)))}
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
 for pp in row.cells[1].paragraphs:pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
for i in [0,5,10,15]:
 cell=tt.cell(i+2,0).merge(tt.cell(i+6,0));cell.text=wide.iloc[i]['Location'];direction=OxmlElement('w:textDirection');direction.set(qn('w:val'),'btLr');cell._tc.get_or_add_tcPr().append(direction)
 for pp in cell.paragraphs:pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
for row in tt.rows:
 for c in row.cells:
  c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
  for pp in c.paragraphs:
   for rr in pp.runs:rr.font.name='Times New Roman';rr.font.size=Pt(8.1)
r.note('New compact 2021 lookup: all 160 assignments are also available as long and flat wide CSV tables and standalone SVG/PNG/PDF graphics. The three apartment rows use the released ground/semi-basement/raised, intermediate and top-floor categories. Heating and electricity means are in Tables 6a and 6b.')

# 9 Method and provenance
r.page();r.h('Methods and reproducibility')
r.sub('Recovering the 2013 probability workflow')
r.p('The archived generator at Git revision e7094f4 and the saved probablities_only.csv establish the numerical workflow: age centres 15–135 at five-year steps, area centres 40–352 at 24 m² steps, inclusive half-widths of 15 years and 40 m², and a minimum of five households. Restricting centres to age and area below 150 gives 125 primary nodes. This is the numerical procedure used for the revised 2021 equation.')
r.p('The earlier reconstruction followed the different spacing and narrow windows described in the paper’s prose and the recovered helper module. The saved workflow supplies the denser, more strongly smoothed empirical target. To measure the improvement fairly, the earlier geometry is refitted within each training split and both geometries predict the same held-out 125-node target. No 2013 outcome is recalculated or replaced.')
r.sub('Household and energy assumptions')
r.p('All 52,617 demographic labels and the 46,712 known-age/area dwelling draws are held fixed. The existing schooling assumptions, included younger children and uniform within-class age draws therefore remain the same. The new matrix is nationally pooled over location and dwelling type; its context variables are dwelling age and area. Construction-period curves average the same age-node sets for observed and fitted diagnostics.')
r.p('The energy partition uses location × dwelling × archetype cells. Heating uses 48,690 valid frequent-user responses; appliance electricity uses all 52,617 demographic profiles. Modeled kWh retain the audited 2013 reference power and efficiency assumptions, with 2021 ownership and use. The raw 2021 apartment classes describe floor position. The 2013 apartment classes describe building size. Group mean differences therefore describe the stated survey/model definitions.')
r.p('Households have equal weight in primary cell means and energy validation. The probability regression weights its windows equally. Official PESO expansion weights remain a separate diagnostic. No separately calibrated regional energy multiplier was located; the multiplier is set to 1 as a modelling convention. The 95% energy fidelity floor is a design choice for compact simulation inputs; simulation-level equivalence has not yet been tested.')
r.sub('Source and audit package')
r.p('The accompanying package includes all candidate models, the selected full-precision matrix, matched node/curve data, repeated household validation, group-count sensitivity, compact household assignments and all Appendix A cells. Independent scripts reconstruct the probability matrix and every group assignment. Original 2013 source hashes and the previous 2021 output hashes are checked at delivery.')
r.p('Khajedehi, Vivian and Zarrella (2025). Identification of Italian user archetypes for building energy simulations based on a national survey. Energy and Buildings 344, 115924.')
r.link('Original article DOI','https://doi.org/10.1016/j.enbuild.2025.115924')
r.note('The package README identifies each result file and the generating scripts. Validation is internal and conditional on the existing demographic fit. Group count and fidelity sensitivities remain available for choosing a different simulation resolution.')
r.doc.core_properties.title='Italian user archetypes in 2013 and 2021 revised dwelling probabilities and compact energy groups'
r.save('Italian_user_archetypes_2021_reconstruction.docx')
print(json.dumps({'blocks':len(r.blocks),'tables':len(r.doc.tables),'figures':len(r.doc.inline_shapes),'equations':1}),flush=True)
