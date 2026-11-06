from pathlib import Path
import json,csv,sys
HERE=Path(__file__).resolve().parent
helper=(HERE.parent/'build_report.py').read_text(encoding='utf-8').split('# Page 1')[0]
exec(helper.replace('ROOT=HERE.parent','ROOT=HERE.parents[1]'))
for style in doc.styles:
    if style.type==1:style.font.name='Times New Roman'
    for fonts in style.element.xpath('.//w:rFonts'):
        for attr in list(fonts.attrib):
            if 'theme' in attr.lower():del fonts.attrib[attr]
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Normal'].paragraph_format.line_spacing=1.04
doc.styles['Normal'].paragraph_format.space_after=Pt(7)
for name,size in [('Title',22),('Heading 1',16),('Heading 2',12)]:
    doc.styles[name].font.size=Pt(size)
doc.styles['Caption'].font.name='Times New Roman';doc.styles['Caption'].font.size=Pt(9.5)
doc.styles['Caption'].font.bold=False;doc.styles['Caption'].font.italic=False
header.text='Italian household archetypes   |   2013 and 2021'
header.runs[0].font.name='Times New Roman';header.runs[0].font.size=Pt(8)
def table(headers,data,widths,size=10):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for cc,w in zip(t.columns,widths):cc.width=Inches(w)
    for c,v,w in zip(t.rows[0].cells,headers,widths):c.width=Inches(w);c.text=v
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in data:
        for c,v,w in zip(t.add_row().cells,row,widths):c.width=Inches(w);c.text=str(v)
    bd=OxmlElement('w:tblBorders')
    for edge in ['top','bottom','left','right','insideH','insideV']:
        el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil')
        el.set(qn('w:sz'),'5');el.set(qn('w:color'),'000000');bd.append(el)
    t._tbl.tblPr.append(bd)
    for i,row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for j,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'55' if side in ['top','bottom'] else '70');el.set(qn('w:type'),'dxa');mar.append(el)
            pr.append(mar)
            if i==0:
                b=OxmlElement('w:tcBorders');e=OxmlElement('w:bottom');e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'000000');b.append(e);pr.append(b)
            for pp in c.paragraphs:
                pp.paragraph_format.space_after=Pt(0);pp.paragraph_format.line_spacing=1.0
                pp.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in pp.runs:r.font.name='Times New Roman';r.font.size=Pt(size);r.font.bold=False
    blocks.append({'kind':'table','headers':headers,'rows':[[str(v) for v in rr] for rr in data]})
    pp=doc.add_paragraph();pp.paragraph_format.space_after=Pt(0);pp.paragraph_format.space_before=Pt(0);pp.paragraph_format.line_spacing=1;pp.paragraph_format.line_spacing_rule=None
    pp.paragraph_format.space_after=Pt(1);pp.paragraph_format.space_before=Pt(0)
    pp.add_run().font.size=Pt(2)
def caption(label,title):
    pp=doc.add_paragraph(style='Caption');pp.paragraph_format.keep_with_next=label.startswith('Table')
    pp.add_run(label+'  ').bold=True;pp.add_run(title)
    blocks.append({'kind':'paragraph','text':label+'  '+title})
def note(text):
    pp=doc.add_paragraph(text);pp.paragraph_format.space_after=Pt(5)
    for r in pp.runs:r.font.size=Pt(9)
    blocks.append({'kind':'paragraph','text':text})
def fig(name,width=7.02):
    path=HERE/'figures'/f'{name}.png';pp=doc.add_paragraph();pp.paragraph_format.space_after=Pt(3);pp.paragraph_format.keep_with_next=True
    rr=pp.add_run();rr.add_picture(str(path),width=Inches(width))
    for el in rr._r.xpath('.//wp:docPr'):el.set('descr',name.replace('_',' '))
    blocks.append({'kind':'image','path':str(path)})
bench=json.loads((ROOT/'00_baseline/published_benchmarks.json').read_text(encoding='utf-8'))
demo=json.loads((HERE/'table3_2021_statistics.json').read_text())
s=rows('2021_fixed_proxy_summary.csv');reg=rows('regional_common_cohort_changes.csv')
def pb(group,metric):return float(next(r['value'] for r in bench if r['group']==group and r['metric']==metric))
def signed(v,d=2):return f'{v:+.{d}f}'

# Page 1
p('Italian household archetypes in 2013 and 2021','Title')
p('Changes in household profiles operating patterns and regional distributions')
p('4 November 2026')
p('In 2013, A1 was the largest published household group at 40.1%. In the eligible 2021 profiles, A2 is the largest unweighted group at 33.92%, followed by A1 at 27.61% and A3 at 27.42%. Applying ISTAT household weights makes A3 the largest newer group at 31.88%. The following pages show the 2013 and 2021 values together, using the original paper\'s cluster colours, table conventions and matching captions.')
caption('Table 1','Household characteristics and system use in 2013 and 2021.')
table(['Indicator','2013','2021','Change'],[
 ['Households with air conditioning','29.4%','48.8%','+19.4 pp'],
 ['Mean household size','2.281','2.276','-0.005'],
 ['Households with a member under 15','22.01%','18.50%','-3.51 pp'],
 ['Daily heating on-time among frequent users','7.90 h','8.55 h','+0.65 h'],
 ['Daily cooling on-time among frequent users','4.80 h','6.26 h','+1.46 h'],
],[3.65,1.12,1.12,1.17],10.5)
note('Air conditioning: official all-household publication estimates [2-4]. Other rows: survey-weighted calculations for households with 1 to 5 members [5]. Operating-hour means use the frequent-user response domain described on page 5. pp denotes percentage points.')
fig('cooling_availability')
caption('Fig. 1.','Households with air conditioning in 2013, 2021 and 2024.')
p('Cooling availability rose from 29.4% in 2013 to 48.8% in 2021 and reached 56.0% in the 2024 published results. Heating and cooling on-time also rose among the analysed frequent users. Survey-weighted household size remained close to 2.28 members in both microdata waves.')
note('Archetype comparisons use the printed, unweighted 2013 results and 44,880 complete 2021 profiles. The newer profile domain covers households with 1 to 5 members and excludes all released households with a child aged 0-10 because their schooling fields are blank. Direct survey comparisons in Table 1 retain these families. All historical 2013 results are preserved.')

# Page 2
page();h('Definition of user archetypes')
p('The four types retain their historical meanings. A1 describes smaller older households, A2 larger households with younger and middle-aged members, A3 smaller middle-aged households, and A4 smaller middle-aged or older households with higher education. In the newer profiles, mean household size is lower in each group, while the mean maximum age in A1 is higher.')
caption('Table 2','User archetypes derived from k-means clustering.')
note('2013 published values. Average (standard deviation).')
fields=['min_age','max_age','number','studio_ind']
headers=['cluster','age min','age max','n members','education']
table(headers,[[str(i)]+[f'{pb("A"+str(i),f+"_mean"):.1f} ({pb("A"+str(i),f+"_sd"):.1f})' for f in fields] for i in range(1,5)],[.55,1.65,1.65,1.55,1.66],10)
note('2021 complete profiles under the fixed archive classifier. Average (standard deviation).')
table(headers,[[str(r['archetype'])]+[f'{r[f+"_mean"]:.1f} ({r[f+"_sd"]:.1f})' for f in fields] for r in demo],[.55,1.65,1.65,1.55,1.66],10)
fig('cluster_proportions')
caption('Fig. 2.','Proportion of each cluster among households and people.')
note('Both panels are unweighted. The 2013 panel reproduces Figure 5 of the paper; the 2021 panel uses the 44,880 complete profiles. Age values represent released age intervals. The education index follows the archive coding convention. Table 2 and Figure 2 compare these respective domains.')

# Page 3
page();h('Regional distribution of user archetypes')
p('Across the 20 harmonized regions, the weighted share assigned to A1 is lower in the newer eligible profiles and the share assigned to A3 is higher. A2 and A4 have more varied regional changes. The same blue, green, amber and red order identifies A1-A4 in both years.')
fig('regional_proportions')
caption('Fig. 3.','Proportion of each cluster among households by region.')
note('Survey-weighted proportions within each region. Domains: 17,756 archived 2013 households with 1 to 5 members and 44,880 complete 2021 profiles. The 2021 schooling and age coverage described on pages 1 and 7 applies to every regional comparison. Bolzano and Trento are pooled into Trentino-Alto Adige from record-level sums.')
caption('Table 3','Proportion of each cluster among households with survey weighting and regional standardization.')
std=rows('regional_composition_standardization.csv')
table(['Type','2013 weighted\nArchive domain','2021 weighted\nComplete profiles','2021 at 2013\nRegional mix'],[
 [f'A{i+1}',f'{float(r["archive2013_surveyweighted"]):.2f}%',f'{float(s[i]["weighted_share_pct"]):.2f}%',f'{float(r["proxy2021_at2013_surveyweighted_region_mix"]):.2f}%'] for i,r in enumerate(std)
],[.62,2.15,2.15,2.14],10)
p('Holding the 2013 regional mix fixed gives 2021 shares of 30.46%, 29.19%, 31.79% and 8.56%. Each is within 0.09 percentage points of the newer weighted estimate. The regional-composition adjustment is small for this eligible-profile comparison.')

# Page 4
page();h('Regional values in 2013 and 2021')
def regional_pair(region,aa):
    rr=next(r for r in reg if int(r['region'])==region and int(r['archetype'])==aa)
    return f'{float(rr["weighted2013"]):.2f}% to {float(rr["weighted2021"]):.2f}%'
p(f'Table 4 places each regional share beside its newer value. In Piemonte, for example, A1 changes from {regional_pair(1,1)} and A3 from {regional_pair(1,3)}. In Campania, A1 changes from {regional_pair(15,1)}, while A3 changes from {regional_pair(15,3)}. The pattern across regions is clearer in Figure 3, and the table gives the corresponding values.')
regions=['Piemonte',"Valle d'Aosta",'Lombardia','Trentino-Alto Adige','Veneto','Friuli-Venezia Giulia','Liguria','Emilia-Romagna','Toscana','Umbria','Marche','Lazio','Abruzzo','Molise','Campania','Puglia','Basilicata','Calabria','Sicilia','Sardegna']
rd=[]
for i,name in enumerate(regions,1):
    rr=sorted([r for r in reg if int(r['region'])==i],key=lambda x:int(x['archetype']))
    rd.append([name]+[f'{float(r["weighted2013"]):.2f} / {float(r["weighted2021"]):.2f}' for r in rr])
caption('Table 4','Regional proportions of user archetypes in 2013 and 2021.')
table(['Region','A1\n2013 / 2021','A2\n2013 / 2021','A3\n2013 / 2021','A4\n2013 / 2021'],rd,[2.02,1.26,1.26,1.26,1.26],9.5)
note('Each cell gives survey-weighted percentages as 2013 / 2021. The 2013 values use saved archive memberships within the 1 to 5 member domain. The 2021 values use complete profiles. Thus the rows describe the two stated domains, including their different age and schooling coverage. The regional source CSV retains record counts and percentage-point differences.')
sub('Regional correction factors')
p('The repository audit identified coef_red for 2013 and PESO for 2021 as the applicable supplied household expansion weights. We used them under the agreed fallback. Each regional proportion is the sum of household weights for a type divided by the sum for the regional analysis domain. National standardization combines the 2021 regional proportions using the 2013 archive domain\'s weighted regional shares.')
p('The factor inventory records the search bounds and distinguishes these household weights from model energy-calibration multipliers. The original paper\'s published national proportions remain the unweighted reference shown on page 2.')

# Page 5
page();h('Heating and cooling system usage')
p('In 2013, frequent users reported a weighted average of 7.90 heating hours per day. In the 2021 survey responses referring to 2020, this average is 8.55 hours. Cooling on-time changes from 4.80 to 6.26 hours. In the unweighted distributions, median heating on-time is 8 hours in both waves; median cooling on-time rises from 4 to 5 hours. Figure 4 shows these daily-response distributions in the broad 1 to 5 member domains.')
fig('operating_hours')
caption('Fig. 4.','Variation of (a) daily heating system operating hours and (b) daily cooling system operating hours.')
note('Unweighted box plots: boxes span the 25th to 75th percentiles; the line marks the median. Whiskers reach the most extreme observations within 1.5 interquartile ranges; outliers are omitted. Colour tones follow the original Figure 8. Each panel shows declared daily hours in its frequent-user samples.')
caption('Table 5','Daily on-time of heating and cooling systems.')
op=rows('full_size1_5_direct_regional.csv');oper=[]
for metric,label in [('heating_conditional_hours','Heating'),('cooling_conditional_hours','Cooling')]:
    rr=[next(r for r in op if r['year']==str(y) and r['region']=='0' and r['metric']==metric) for y in [2013,2021]]
    oper.append([label]+[f'{float(r["surveyweighted"]):.2f}' for r in rr]+[f'{int(r["valid_n"]):,}' for r in rr])
table(['System','2013 mean\nh/day','2021 mean\nh/day','2013\nValid n','2021\nValid n'],oper,[1.78,1.32,1.32,1.32,1.32],10)
caption('Table 6','Daily on-time by user archetype in 2013 and 2021.')
co=rows('conditional_heating_cooling_new_diagnostic.csv')
table(['Type','Heating\n2013 / 2021','Cooling\n2013 / 2021'],[
 [f'A{aa}']+[ ' / '.join(f'{float(next(r["mean_hours_surveyweighted"] for r in co if r["year"]==str(y) and r["outcome"]==kind and int(r["archetype"])==aa)):.2f}' for y in [2013,2021]) for kind in ['heating','cooling']] for aa in range(1,5)
],[.7,3.18,3.18],10)
note('Tables 5-6 are survey weighted. Frequency codes 1 or 2 and three complete time blocks in the range 0-8 hours define each valid sample. The 2013 questions concern a typical winter or summer day; the 2021 questions concern cold or hot months of 2020. Table 6 uses archived 2013 and complete-profile 2021 assignments. These are cross-sectional reported hours; weather, equipment mix and survey mode also vary across waves.')

# Page 6
page();h('Cluster structure in the newer profiles')
p('The 2013 study selected four clusters. In 2021, a four-cluster fit on the same archive coordinates yields aligned shares of 31.64%, 25.19%, 26.68% and 16.50%. The fixed historical classifier gives 27.61%, 33.92%, 27.42% and 11.05% on those same newer households.')
fig('within_cluster_sum_squares')
caption('Fig. 5.','Within Cluster Sum of Squares as a function of number of clusters.')
note('Left: the published 2013 Figure 3, rendered directly from the article. Right: the new 2021 exploratory fits for K=2 to 8. The panels retain their own sample sizes and vertical scales. The historical figure is unchanged.')
caption('Table 7','Proportion of each cluster in the fixed transfer and the 2021 fit.')
refit=rows('2021_refit_k4.csv');r42=[r for r in refit if r['seed']=='42'];r42.sort(key=lambda r:int(r['archetype']))
table(['Type','2013\nPublished','2021\nFixed classifier','2021\nAligned refit'],[
 [f'A{i+1}',f'{pb("A"+str(i+1),"household_share_pct"):.1f}%',f'{float(s[i]["share_pct"]):.2f}%',f'{float(r42[i]["share_pct"]):.2f}%'] for i in range(4)
],[.62,2.15,2.15,2.14],10.5)
p('The two 2021 partitions agree on 78.62% of household labels. Their adjusted Rand index is 0.6694 and normalized mutual information is 0.6609. Agreement is lower for fixed A4: 1,611 of its 4,958 households, or 32.5%, retain the aligned A4 label in the new fit. These comparisons describe the two classifications applied to the same newer records.')
p('Across the 2021 candidate fits, K=2 has the highest sampled silhouette, 0.408, while K=4 gives 0.345. The historical four-group resolution provides the comparison used throughout this report. The K search describes the geometry of the complete-profile sample.')
note('K-means used 10 starts, up to 150 iterations, tolerance 10^-8 and seeds 0, 42 and 2021. The displayed refit uses seed 42. Alignment minimizes total squared centroid distance over all 24 permutations; silhouettes use a fixed 1,500-record sample.')

# Page 7
page();h('Data coverage and sensitivity')
p('The 2013 raw release contains 20,000 households and the newer release contains 54,069. The broad domains with 1 to 5 members contain 19,688 and 53,520 households. Saved 2013 archive labels cover 17,855 households in total, including 17,756 in the broad size domain. The complete-profile 2021 transfer covers 44,880 households, representing 83.0% of newer records and 77.16% of the full-wave household expansion weight.')
p('Among newer households with 1 to 5 members, 7,803 contain a released child aged 0-10 and have incomplete schooling profiles; another 837 fail other profile requirements. All released members in the 0-5 and 6-10 age bands have blank schooling. Questionnaire F.10 routes schooling to age 10 or older and relationship codes 1-5; the released bands combine age 10 with younger children. The direct household and operating comparisons use the broader size domain, retaining these families.')
sub('Age and education coding')
p('Primary assignments use age-band midpoints, with ages 80 and over represented by 90 under an assumed 80-100 range. The merged no-credential/primary category is represented by 12 years to approximate the archive\'s sequential recoding convention. The index is therefore a harmonized coding assumption. The following checks show how assignments respond to alternative mappings.')
caption('Table 8','Proportion of each cluster under alternative age and education mappings.')
sens=rows('transfer_sensitivities.csv');post=rows('review_triggered_age_alignment.csv');sr=[]
for title,rr in [('Primary age midpoints',[r for r in sens if r['scenario']=='primary']),('Aligned historical age imputation',post)]:
    rr.sort(key=lambda r:int(r['archetype']));sr.append([title]+[f'{float(r["share_pct"]):.2f}%' for r in rr])
table(['Mapping','A1','A2','A3','A4'],sr,[2.62,1.11,1.11,1.11,1.11],10)
table(['Sensitivity','Household labels changed'],[
 ['Within-age-band random draws, seeds 0, 42, 2021','7.86-8.12%'],
 ['Lowest education category represented by 5 years','2.41%'],
 ['Age mapping aligned with historical imputation','10.18%'],
],[4.5,2.56],10)
p('Under primary midpoints, A3 changes from the printed 2013 share of 22.9% to 27.42% in the eligible 2021 profiles. Under the aligned historical age imputation, its newer share is 23.61%. This additional check was introduced during independent review. The example shows the range of the age-mapping effect on the temporal comparison.')
sub('Fixed historical classification')
p('The extension uses the saved 2013 labels and fixed coordinate means, scaling and centroids. Replaying that classifier reproduces all 17,855 archived labels. The reconstructed archive proxy supplies the operational transfer to 2021. Of the eligible newer profiles, 2,985, or 6.65%, lie outside at least one historical coordinate range, mostly in A1. The published 2013 results and original files retain their historical values.')

# Page 8
page();h('Implications for building simulations')
p('The newer survey evidence supports input scenarios with more widespread cooling and longer daily operating periods among frequent users. The 2013 study supplies the historical user and operating groups. The 2021 extension supplies updated equipment context, reported-hour distributions and a conditional comparison of demographic profiles. The 2024 published series extends the cooling-availability context to 56.0%.')
caption('Table 9a','Significant groups for heating system usage.')
table(['Group name','Average (standard deviation)','Unit'],[
 [r['group'],f'{float(r["value"]):.2f} ({float(r["sd"]):.2f})','h/day'] for r in bench if r['kind']=='heating' and r['metric']=='mean'
],[1.3,3.46,2.3],10.5)
caption('Table 9b','Significant groups for annual electricity consumption')
table(['Group name','Average (standard deviation)','Unit'],[
 [r['group'],f'{float(r["value"]):.0f} ({float(r["sd"]):.0f})','kWh/year'] for r in bench if r['kind']=='electricity' and r['metric']=='mean'
],[1.3,3.46,2.3],10.5)
note('Published 2013 Table 6 means and standard deviations, preserved exactly. H1-H7 concern declared heating on-time; E1-E5 concern modelled appliance electricity. Their definitions combine user type, dwelling and location. The 2021 daily-hour comparisons on page 5 have their own response domains.')
sub('Next extension of the model inputs')
p('For the simulation stage, keep the historical A1-A4 definitions and evaluate newer equipment and operating scenarios alongside them. Match geography and dwelling characteristics before assigning the paper\'s H and E groups. Establish a compatible dwelling crosswalk for the newer survey, including apartment size and building-floor information.')
p('A further demographic extension should retain families with young children in a shared age and schooling representation. Show the resulting weighted coverage in both waves, then repeat the fixed-classifier comparison and the exploratory clustering. This would put the change in archetype proportions on a more comparable household domain.')
p('The next microdata wave can extend the household analysis to 2024 when obtained through an appropriate access channel. The current local household releases cover 2013 and 2021; the 2024 values used here are published aggregates. Regional physical energy calibration would require separate compatible energy benchmarks. The simulations themselves remain a subsequent study stage.')

# Page 9
page();h('Sources and analytical record')
p('The extension combines the original published results, the local ISTAT 2013 and 2021 releases and the official 2024 equipment report. All temporal estimates are descriptive repeated-cross-section comparisons. Household identifiers are wave specific. Differences in collection mode, question reference period, age intervals and schooling availability are documented in the analysis package.')
sub('Primary references')
link('[1] Khajedehi, Vivian and Zarrella (2025). Identification of Italian user archetypes for building energy simulations based on a national survey. Energy and Buildings 344, 115924. Historical Table 3, Figure 5, Figure 3 and Table 6.', 'https://doi.org/10.1016/j.enbuild.2025.115924')
link('[2] ISTAT (2014). I consumi energetici delle famiglie, anno 2013.', 'https://www.istat.it/wp-content/uploads/2014/12/StatReport_Consumi_energetici.pdf')
link('[3] ISTAT (2022). I consumi energetici delle famiglie, anno 2021. Release and linked metadata.', 'https://www.istat.it/comunicato-stampa/i-consumi-energetici-delle-famiglie-anno-2021/')
link('[4] ISTAT (2025). Dotazioni energetiche delle famiglie, anno 2024. Released 9 October 2025. Cooling section gives the 2013, 2021 and 2024 availability series.', 'https://www.istat.it/wp-content/uploads/2025/10/Report_Dotazioni-energetiche-delle-famiglie_Anno-2024.pdf')
p('[5] Local ISTAT releases: ECH_Microdati_Anno_2013.txt with coef_red and ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt with PESO. Definitions were checked against the released record layouts, classifications and questionnaires. New 2021 user profiles are transferred using the fixed archive classifier.')
link('[6] ISTAT. Indagine sui consumi energetici delle famiglie. The MFR listing inspected on 4 November 2026 contains 2013 and 2021.', 'https://www.istat.it/microdati/indagine-sui-consumi-energetici-delle-famiglie/')
sub('Figure and table correspondence')
p('Report Figure 2 retains the title and four-colour stacked-bar design of original Figure 5. Report Figure 5 retains the title of original Figure 3 and reproduces its historical panel directly from the publisher PDF. Table 2 retains the original Table 3 title and mean-with-standard-deviation layout. Tables 9a and 9b retain the corresponding original Table 6 titles. New graphics use the same serif labels, thin axes and restrained grids. Times New Roman substitutes for the paper\'s Charis SIL font.')
sub('Reproducibility and verification')
p('The package root is output/temporal_extension_2013_2021_20261104. Folder 00_baseline contains printed benchmarks and source hashes; 01_comparability contains primary-definition evidence and the factor inventory; 02_analysis contains the fixed analysis, CSV results and scripts; 03_review contains independent scientific checks. The revised report, chart source data and figure-building scripts are in 04_report/revised_comparisons.')
p('Independent checks reproduced the eligible assignments, all 80 regional cells, regional standardization, direct outcomes and principal clustering diagnostics. The protected original files and 40 frozen analysis artifacts are checked again at delivery. The new graphical box-plot summaries reproduce the frozen operating-domain counts and unweighted means. The 2013 analysis and published results remain unchanged.')

doc.core_properties.title='Italian household archetypes in 2013 and 2021'
doc.core_properties.subject='Direct comparative report with paper-style graphics and tables'
doc.core_properties.author='GRINS research working report'
doc.save(HERE/'Household_archetypes_2013_2021_comparative_report.docx')
(HERE/'report_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'blocks':len(blocks),'tables':len(doc.tables),'figures':len(doc.inline_shapes)},indent=2))
