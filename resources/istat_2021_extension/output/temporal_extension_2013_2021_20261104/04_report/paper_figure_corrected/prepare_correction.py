from pathlib import Path
import shutil,json,hashlib,re,sys
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'revised_comparisons';ROOT=HERE.parents[1];REPO=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['cluster_proportions','cooling_availability','regional_proportions','within_cluster_sum_squares']:
    for ext in ['png','pdf']:shutil.copy2(PREV/'figures'/f'{name}.{ext}',HERE/'figures'/f'{name}.{ext}')
shutil.copy2(PREV/'table3_2021_statistics.json',HERE/'table3_2021_statistics.json')
s=(PREV/'build_report.py').read_text(encoding='utf-8')
start=s.index('# Page 5\n');end=s.index("caption('Table 5'",start)
replacement='''# Page 5
page();h('Heating and appliance electricity use in 2013')
p('The paper groups heating use into H1-H7 and annual appliance electricity into E1-E5. Heating is split by insulation level, and electricity by cooling-system use. Figure 4 reproduces the published 2013 Figure 8 directly, preserving its groups, variables, distributions, legends and axes.')
fig('published_figure8_2013',width=5.55)
doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
caption('Fig. 4.','Variation of (a) daily heating system operating hours and (b) annual electricity consumption based on significant groups of dwelling types, geographical location, and user type.')
note('Published 2013 Figure 8 reproduced unchanged from page 11 of the original article [1]. Boxes show the interquartile range and median; outliers are omitted. The source figure retains its original group definitions and units.')

# New direct survey table page
page();h('Heating and cooling system usage in 2013 and 2021')
p('In 2013, frequent users reported a weighted average of 7.90 heating hours per day. In the 2021 survey responses referring to 2020, the average is 8.55 hours. Cooling on-time changes from 4.80 to 6.26 hours. In the unweighted frequent-user responses, median heating on-time is 8 hours in both waves and median cooling on-time rises from 4 to 5 hours. Tables 5 and 6 give these newer comparisons on their stated response domains.')
'''
s=s[:start]+replacement+s[end:]
s=s.replace('described on page 5','described on page 6').replace('described on pages 1 and 7','described on pages 1 and 8').replace('comparisons on page 5','comparisons on page 6')
old="p('Report Figure 2 retains the title and four-colour stacked-bar design of original Figure 5. Report Figure 5 retains the title of original Figure 3 and reproduces its historical panel directly from the publisher PDF. Table 2 retains the original Table 3 title and mean-with-standard-deviation layout. Tables 9a and 9b retain the corresponding original Table 6 titles. New graphics use the same serif labels, thin axes and restrained grids. Times New Roman substitutes for the paper\\'s Charis SIL font.')"
new="p('Report Figure 2 retains the title and four-colour stacked-bar design of original Figure 5. Report Figure 4 reproduces original Figure 8 directly from the publisher PDF, including H1-H7 with insulation and E1-E5 with cooling-system use. Report Figure 5 retains the title of original Figure 3 and reproduces its historical panel. Table 2 retains the original Table 3 title and mean-with-standard-deviation layout. Tables 9a and 9b retain the corresponding original Table 6 titles. Times New Roman substitutes for the paper\\'s Charis SIL font in the report text.')"
assert old in s;s=s.replace(old,new)
s=s.replace('The revised report, chart source data and figure-building scripts are in 04_report/revised_comparisons.','The corrected report and Figure 8 source trace are in 04_report/paper_figure_corrected. Chart source data and figure-building scripts for the other comparisons are retained in 04_report/revised_comparisons.')
s=s.replace('The new graphical box-plot summaries reproduce the frozen operating-domain counts and unweighted means.','The restored Figure 8 is rendered directly from the protected publisher PDF. The newer numerical comparisons retain their independently checked domains and values.')
s=s.replace('Household_archetypes_2013_2021_comparative_report','Household_archetypes_2013_2021_corrected_report')
(HERE/'build_report.py').write_text(s,encoding='utf-8')
for name in ['export_word_pdf.ps1','render_and_check.py']:
    text=(PREV/name).read_text(encoding='utf-8-sig').replace('Household_archetypes_2013_2021_comparative_report','Household_archetypes_2013_2021_corrected_report')
    if name.endswith('.py'):text=text.replace('==len(pngs)==9','==len(pngs)==10')
    (HERE/name).write_text(text,encoding='utf-8')
script=(REPO/'07_ANOVA_analysis.py').read_text(encoding='utf-8-sig').splitlines()
def find(needle):return [i for i,line in enumerate(script,1) if needle in line]
trace={'request':'Restore the actual paper plot and trace its source code, without changing 2013 results','source_paper':{'path':'C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main.pdf','figure':8,'page':11,'crop_pdf_points':[115,150,480,606],'dpi':300},'source_code':{'path':str(REPO/'07_ANOVA_analysis.py'),'sha256':sha(REPO/'07_ANOVA_analysis.py'),'heating_plot_line':find("y='riscaldament',"),'electricity_plot_line':find("y='elettricita',"),'electricity_outcome_label_line':find('Annual Appliance Consumption (kWh)'),'heating_outcome_label_line':find('Daily On-Time of Heating System'),'electricity_palette_line':find("palette=['#A3C1E3', '#1B365D']"),'heating_palette_line':find("palette=['#F4D4B0', '#E9AC72']")},'heating_panel':{'x':'new_group: Group H1 through H7','y':'riscaldament: Daily On-Time of Heating System','hue':'isolamento: Sufficiently insulated / Not sufficiently insulated','saved_input':'Significant_groups.csv'},'electricity_panel':{'x':'new_group_e: Group E1 through E5','y':'elettricita: Annual Appliance Consumption (kWh)','hue':'Cooling System: not used / used, derived from raffrescament','saved_input':'new_elet.csv'},'read_only_supporting_sources':[{'path':str(REPO/n),'sha256':sha(REPO/n)} for n in ['Significant_groups.csv','new_elet.csv','Heating.svg','Elec.svg','Elec_Times_New_Roman.svg']],'restoration':'Direct rendering of the published Figure 8 preserves the published historical distributions; no historical group fitting or new 2013 calculation is used for the replacement','removed':'Two yearly boxes for heating hours and two for cooling hours; this substituted a new design and outcome for the original grouped Figure 8','figure_png_sha256':sha(HERE/'figures/published_figure8_2013.png')}
(HERE/'figure8_code_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
print(json.dumps({'prepared':str(HERE),'source_code':trace['source_code'],'replacement':'Actual Figure 8 from the paper'},indent=2))
