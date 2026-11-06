from pathlib import Path
import csv, json, hashlib, sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
A=ROOT/'02_analysis'
def rows(name):
    with (A/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
doc=Document(); sec=doc.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.68)
sec.left_margin=sec.right_margin=Inches(.72)
sec.header_distance=sec.footer_distance=Inches(.28)
for s in doc.styles:
    if s.type==1:
        s.font.name='Calibri'; s.font.color.rgb=RGBColor(0,0,0)
    for el in s.element.xpath('.//w:pBdr'):
        el.getparent().remove(el)
    for el in s.element.xpath('.//w:color'):
        for attr in list(el.attrib):
            if 'theme' in attr: del el.attrib[attr]
        el.set(qn('w:val'),'000000')
normal=doc.styles['Normal']; normal.font.size=Pt(11)
normal.paragraph_format.line_spacing=1.08; normal.paragraph_format.space_after=Pt(7)
for name,size in [('Title',24),('Heading 1',17),('Heading 2',12)]:
    s=doc.styles[name]; s.font.size=Pt(size); s.font.bold=True
    s.paragraph_format.space_before=Pt(10 if name!='Title' else 0)
    s.paragraph_format.space_after=Pt(7)
    s.paragraph_format.keep_with_next=True
doc.styles['Caption'].font.size=Pt(9)
doc.styles['Caption'].font.color.rgb=RGBColor(0,0,0)
header=sec.header.paragraphs[0]; header.text='GRINS   |   Italian household archetype extension'
header.runs[0].font.size=Pt(8)
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('4 November 2026   |   ')
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
for r in footer.runs:r.font.size=Pt(8)
blocks=[]
def p(text,style=None):
    doc.add_paragraph(text,style); blocks.append({'kind':'paragraph','text':text})
def h(text):p(text,'Heading 1')
def sub(text):p(text,'Heading 2')
def page():doc.add_page_break()
def table(headers,data,widths,size=9.5):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for c,v,w in zip(t.rows[0].cells,headers,widths):c.width=Inches(w);c.text=v
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in data:
        for c,v,w in zip(t.add_row().cells,row,widths):c.width=Inches(w);c.text=str(v)
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
    t._tbl.tblPr.append(borders)
    for i,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
        for j,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'65' if side in ['top','bottom'] else '85');el.set(qn('w:type'),'dxa');mar.append(el)
            pr.append(mar)
            if i==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E9EDF1');pr.append(shade)
            for pp in c.paragraphs:
                pp.paragraph_format.space_after=Pt(0);pp.paragraph_format.line_spacing=1.02
                pp.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in pp.runs:r.font.size=Pt(size);r.font.bold=(i==0)
    blocks.append({'kind':'table','headers':headers,'rows':[[str(v) for v in x] for x in data]})
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
def link(label,url):
    pp=doc.add_paragraph(); rel=pp.part.relate_to(url,RT.HYPERLINK,is_external=True)
    hl=OxmlElement('w:hyperlink');hl.set(qn('r:id'),rel)
    rr=OxmlElement('w:r');pr=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'19');pr.append(sz);rr.append(pr)
    tt=OxmlElement('w:t');tt.text=label;rr.append(tt);hl.append(rr);pp._p.append(hl)
    pp.paragraph_format.space_after=Pt(6)
    blocks.append({'kind':'link','text':label,'url':url})

# Page 1
p('Extending the Italian household archetypes with newer survey data','Title')
p('A comparison of 2013 and 2021 with regional weighting and 2024 published context')
p('Research report  |  4 November 2026')
h('Main findings')
p('The published 2013 archetypes remain the historical reference, with all results preserved. The newer evidence shows a substantial increase in cooling availability and longer reported heating and cooling hours among frequent users. It also reveals that a simple comparison of cluster shares would mix real household differences with changes in measurement and sample coverage.')
table(['Indicator','2013','2021','2024'],[
 ['Households with air conditioning\nOfficial all-household estimate','29.4%','48.8%','56.0%'],
 ['Mean household size\nSurvey weighted, 1 to 5 members','2.281','2.276','Not analysed'],
 ['Households with a member under 15\nSurvey weighted, 1 to 5 members','22.01%','18.50%','Not analysed'],
 ['Reported heating hours per day\nWeighted frequent users, 1 to 5 members','7.90','8.55','Not analysed'],
 ['Reported cooling hours per day\nWeighted frequent users, 1 to 5 members','4.80','6.26','Not analysed'],
],[3.2,1.28,1.28,1.3],10)
p('Sources: ISTAT official cooling series [2-4]; new raw-release calculations in this extension [5]. The operating-hour questions have different reference periods, detailed on page 6. The 1 to 5 member domain is narrower than the official all-household cooling series. These rows have separate denominators.')
p('The 2021 fixed classifier assigns 27.61%, 33.92%, 27.42% and 11.05% of eligible households to A1-A4. These differ from the printed 2013 shares of 40.1%, 26.9%, 22.9% and 10.1%. However, the strict 2021 profile excludes every released household containing a child aged 0-10. The difference therefore cannot establish a national change in archetype prevalence.')
p('No qualified separate regional calibration table was found in the repository search. Following the agreed fallback, the extension uses ISTAT household expansion weights. Holding the 2013 regional mix fixed changes the eligible 2021 weighted shares by less than 0.09 percentage points each. This small geographic effect does not resolve the missing household types or the age and education crosswalk.')

# Page 2
page();h('The frozen 2013 reference')
p('The reference is Khajedehi, Vivian and Zarrella, Identification of Italian user archetypes for building energy simulations based on a national survey, Energy and Buildings 344, 115924 (2025) [1]. Its data refer to 2013. Table 3 and Figure 5 provide the values below; household and person shares are unweighted.')
table(['Type','Household share','Person share','Mean minimum age','Mean maximum age','Mean members','Mean education index'],[
 ['A1','40.1%','29.9%','70.2','78.3','1.8','0.5'],
 ['A2','26.9%','40.2%','21.2','57.2','3.7','0.7'],
 ['A3','22.9%','21.3%','46.0','63.0','2.3','0.5'],
 ['A4','10.1%','8.6%','55.6','72.2','2.1','0.8'],
],[.44,.96,.85,1.18,1.18,.96,1.49],9.3)
p('These are group means rather than classification thresholds. A1 represents smaller older households; A2 larger households with younger and middle-aged members; A3 smaller middle-aged households; and A4 smaller middle-aged or older households with higher education. The printed values, including A4 maximum age 72.2, are retained exactly.')
sub('How the new assignments were made')
p('The lowercase clusters.csv archive supplies 17,855 existing 2013 household memberships. We used those labels as saved. No 2013 PCA or k-means model was refitted, and none of the original source files or published output tables was edited. Recovering fixed coordinate means within the saved groups reproduced all 17,855 archived assignments without disagreement.')
p('That replay establishes an operational archive proxy. It does not certify recovery of the original fitted estimator used for the paper. The 2021 transfer therefore uses fixed archive coordinates, scaling and centroids, with nearest-centroid assignment and no new-year rescaling or forced group sizes.')
sub('Data domains')
table(['Domain','2013 records','2021 records'],[
 ['Raw survey release','20,000','54,069'],
 ['Broad domain with 1 to 5 members','19,688','53,520'],
 ['Archived labels within 1 to 5 members','17,756','Not applicable'],
 ['Complete profile for the 2021 transfer','Not a matched domain','44,880'],
],[4.16,1.45,1.45],10)
p('The broad domain supports direct household and operating comparisons while retaining families with children. The stricter transfer domain additionally requires complete age and schooling profiles. It covers 83.0% of the raw 2021 records and 77.16% of their household expansion weight. The archived 2013 and eligible 2021 domains share geography and size bounds, but not equivalent demographic coverage.')

# Page 3
page();h('What changes in the transferred archetypes')
summary=rows('2021_fixed_proxy_summary.csv')
printed=[40.1,26.9,22.9,10.1]
table(['Type','Printed 2013\nUnweighted','2021 proxy\nUnweighted','2021 proxy\nSurvey weighted'],[
 [f'A{i+1}',f'{printed[i]:.1f}%',f'{float(r["share_pct"]):.2f}%',f'{float(r["weighted_share_pct"]):.2f}%'] for i,r in enumerate(summary)
],[.66,2.0,2.2,2.2],10)
p('The unweighted juxtaposition shows fewer A1 and more A2 and A3 assignments in the selected 2021 sample. Applying PESO changes the newer shares materially, particularly A2 and A3. The weighted column estimates the distribution within the eligible 2021 domain; it is not a weighted replacement for the paper and does not have a like-for-like published 2013 comparator.')
sub('Why this is not yet a population trend')
p('Of the 53,520 newer households with 1 to 5 members, 7,803 contain a child in the released 0-5 or 6-10 age band and necessarily fail the complete-schooling filter. Another 837 fail for other incomplete or unsupported profiles. All 549 households with 6 or more members also fall outside the usable released roster domain. Retention falls to 59.3% of four-person and 51.3% of five-person households.')
p('In the released 2021 file, all members aged 0-5 and 6-10 have blank schooling. The questionnaire rule is more precise: F.10 is routed to age 10 or older and relationship codes 1-5. Age 10 cannot be isolated inside the released age band. We therefore describe the observed exclusion of every household with a released 0-10 child, rather than claiming that the questionnaire excludes everyone under 11. Weighting cannot restore a household stratum with zero included records.')
sub('Sensitivity to the crosswalk')
p('Primary assignments use age-band midpoints, with the 80 and over band treated as 80-100 and represented by 90. The merged no-credential/primary category is mapped to 12 years to approximate the archive\'s sequential recoding collision. This is a coding convention rather than observed schooling.')
table(['Check on the same eligible 2021 records','Result'],[
 ['Random draws within released age intervals\nSeeds 0, 42 and 2021','7.86-8.12% of labels change'],
 ['Lowest schooling category mapped to 5 years\nAlternative to archive-compatible 12 years','2.41% of labels change'],
 ['Age mapping aligned to expected old imputation\nAdded after independent review','10.18% of labels change'],
 ['Outside one or more archived coordinate ranges','2,985 records or 6.65%'],
],[4.65,2.41],10)
p('The review-triggered age mapping produces A1-A4 shares of 32.50%, 30.03%, 23.61% and 13.87%. Under this plausible alternative, A3 is only 0.71 percentage points above the printed 2013 value, compared with 4.52 points under the primary mapping. Most out-of-range records are assigned A1. These results make the magnitude of apparent cluster change sensitive to harmonization choices.')

# Page 4
page();h('Does the newer sample support the same partition')
p('An exploratory k-means fit uses only the 44,880 eligible 2021 profiles in the frozen archive representation. Four clusters were aligned to the reference centroids by the minimum total squared distance over all 24 possible label permutations. This tests whether the same observations naturally form the same partition; it does not estimate household movement between years.')
doc.add_picture(str(A/'fixed_vs_refit_2021.png'),width=Inches(7.02))
blocks.append({'kind':'image','path':str(A/'fixed_vs_refit_2021.png')})
p('Figure 1  Unweighted shares from the fixed transfer and the aligned 2021 refit with seed 42. Both series use the same 44,880 selected 2021 households. The bars compare classification methods within one survey wave.','Caption')
table(['Diagnostic','Result','Meaning'],[
 ['Aligned label agreement','78.62%','Same label on the same 2021 record'],
 ['Adjusted Rand index','0.6694','Agreement between two partitions'],
 ['Normalized mutual information','0.6609','Shared information between partitions'],
 ['Fixed A4 retaining aligned A4','32.5%','1,611 of 4,958 fixed A4 records'],
],[2.2,1.0,3.86],9.7)
p('The refit shares are 31.64%, 25.19%, 26.68% and 16.50%. A4 is particularly unstable under refitting even though total agreement is substantial. These are alternative labels on the same newer households, with no panel linkage to 2013.')
p('The exploratory K=2 to 8 search gives the highest sampled silhouette to K=2, about 0.408, compared with 0.345 for K=4. That ranking concerns this selected sample and feature geometry. It does not establish that Italy now has two household archetypes. The original four types remain the historical resolution for this extension.')
p('The analysis used 10 initializations, a maximum of 150 iterations, tolerance 10^-8 and seeds 0, 42 and 2021. Silhouette calculations used a fixed 1,500-record sample. Independent calculations reproduced the seed 42 fit, alignment, agreement statistics and the full K search.')

# Page 5
page();h('Regional weighting and the geographic comparison')
p('The repository search identified the official final household weights coef_red in 2013 and PESO in 2021. It did not identify a qualified region-by-year physical calibration table. Means use sum(w times value) divided by sum(w); shares use the corresponding weighted indicator. The weights expand households and do not multiply an individual household\'s reported hours or energy demand.')
p('We harmonized geography to 20 regions. The two 2021 provincial codes 41 and 42 were pooled into Trentino-Alto Adige using record-level weighted numerators and denominators. The table shows weighted differences between the 17,756 archived 2013 households with 1 to 5 members and the 44,880 eligible 2021 profiles. It is a geographic diagnostic across differently selected domains.')
regions=['Piemonte',"Valle d'Aosta",'Lombardia','Trentino-Alto Adige','Veneto','Friuli-Venezia Giulia','Liguria','Emilia-Romagna','Toscana','Umbria','Marche','Lazio','Abruzzo','Molise','Campania','Puglia','Basilicata','Calabria','Sicilia','Sardegna']
regional=rows('regional_common_cohort_changes.csv');rd=[]
for i,name in enumerate(regions,1):
    rr=[r for r in regional if int(r['region'])==i];rr.sort(key=lambda x:int(x['archetype']))
    rd.append([name,f'{int(rr[0]["n2013"]):,}',f'{int(rr[0]["n2021"]):,}']+[f'{float(r["weighted_change_pp"]):+.2f}' for r in rr])
table(['Region','n 2013','n 2021','A1 pp','A2 pp','A3 pp','A4 pp'],rd,[2.2,.65,.65,.89,.89,.89,.89],8.5)
p('pp denotes percentage points; n denotes unweighted eligible records. Regional differences inherit the missing young-child households and the coding assumptions. They are not region-specific population trend estimates or energy correction factors. Source: frozen regional comparison table [5].')
p('Replacing the eligible 2021 regional mix with the archive domain\'s 2013 weighted mix yields shares of 30.46%, 29.19%, 31.79% and 8.56%. Relative to the actual eligible weighted shares, the largest change is 0.087 percentage points. Geographic composition alone explains little of this conditional comparison; selection within regions remains unresolved.')

# Page 6
page();h('Direct household and operating changes')
p('The broader 1 to 5 member domains retain 19,688 households in 2013 and 53,520 in 2021, with valid ages for all listed members. Unlike the archetype transfer, these comparisons retain children aged 0-10. They are new descriptive calculations from the raw releases, separate from the paper\'s frozen cluster results.')
direct=rows('full_size1_5_direct_survey.csv')
def dval(year,metric,key):return float(next(r[key] for r in direct if int(r['year'])==year and r['metric']==metric))
table(['Measure','2013\nUnweighted','2021\nUnweighted','2013\nWeighted','2021\nWeighted'],[
 ['Mean household size']+[f'{dval(y,"household_size_mean",k):.3f}' for k,y in [('unweighted',2013),('unweighted',2021),('surveyweighted',2013),('surveyweighted',2021)]],
 ['Any member under 15']+[f'{dval(y,"has_under15_pct",k):.2f}%' for k,y in [('unweighted',2013),('unweighted',2021),('surveyweighted',2013),('surveyweighted',2021)]],
 ['One-member household']+[f'{dval(y,"household_size1_pct",k):.2f}%' for k,y in [('unweighted',2013),('unweighted',2021),('surveyweighted',2013),('surveyweighted',2021)]],
 ['Two-member household']+[f'{dval(y,"household_size2_pct",k):.2f}%' for k,y in [('unweighted',2013),('unweighted',2021),('surveyweighted',2013),('surveyweighted',2021)]],
],[2.2,1.215,1.215,1.215,1.215],9.4)
p('Weighting changes the interpretation. Mean size falls in the raw sample but is almost unchanged after weighting, from 2.28095 to 2.27590. The share with a member under 15 increases unweighted but falls by 3.51 percentage points when weighted. Within this size-restricted domain, one-member households decrease and two-member households increase under the supplied weights. These examples caution against reading sample composition as population change.')
op=rows('full_size1_5_direct_regional.csv');oper=[]
for metric,label in [('heating_conditional_hours','Heating'),('cooling_conditional_hours','Cooling')]:
    r0=next(r for r in op if r['year']=='2013' and r['region']=='0' and r['metric']==metric)
    r1=next(r for r in op if r['year']=='2021' and r['region']=='0' and r['metric']==metric)
    oper.append([label,f'{float(r0["surveyweighted"]):.2f}',f'{float(r1["surveyweighted"]):.2f}',f'{int(r0["valid_n"]):,}',f'{int(r1["valid_n"]):,}'])
table(['Reported hours per day','2013\nWeighted','2021\nWeighted','Valid n\n2013','Valid n\n2021'],oper,[2.2,1.215,1.215,1.215,1.215],10)
p('These means apply to frequency codes 1 or 2 with all three daily time-block responses valid between 0 and 8 hours. The 2013 questions refer to a typical winter or summer day; the 2021 questions refer to cold or hot months of 2020. Equipment mix, weather, respondent composition and survey mode can differ. Longer declared hours do not establish a rise in annual metered consumption.')
p('Official air-conditioning availability rises by 19.4 percentage points from 2013 to 2021 and by another 7.2 points to 2024 [4]. The independently reconstructed affirmative cooling inventory gives weighted 29.43% and 48.85% in the broad local domains. In 2013, blanks among non-affirmative inventory records reflect routing and cannot all be read as explicit absence. The official publication series is therefore used for the headline comparison.')

# Page 7
page();h('Implications for the extension and energy modelling')
p('Retain A1-A4 as the published historical reference. The current evidence supports updating equipment context and testing operating assumptions. It does not yet support replacing the historical national shares with the selected 2021 proxy distribution or presenting new clusters as an observed national transition.')
sub('Published operating benchmarks remain intact')
p('Table 6 in the paper reports the following means. Heating groups H1-H7 and appliance groups E1-E5 combine demographic type with dwelling and location information. They are not one heating or electricity value for each A type.')
table(['Published group','Mean','Unit'],[
 ['H1 / H2 / H3','4.67 / 5.25 / 6.91','heating h/day'],
 ['H4 / H5 / H6 / H7','7.53 / 8.53 / 9.14 / 9.67','heating h/day'],
 ['E1 / E2 / E3','1801 / 2023 / 2320','modelled appliance kWh/year'],
 ['E4 / E5','2646 / 2895','modelled appliance kWh/year'],
],[1.7,2.9,2.46],10)
p('No new H or E group allocation is asserted. The dwelling crosswalk needed for the paper\'s Appendix A was not sufficiently established, including the distinction between apartment size and building floors. The new daily-hour means on page 6 have their own conditional survey denominators and do not overwrite these historical benchmarks.')
sub('Recommended analytical sequence')
p('First, build a cross-wave age and schooling representation that retains families with young children. Document the treatment of children, the merged lowest education category and open-ended age bins. Use a common supported domain and show its weighted coverage in both waves before estimating prevalence differences.')
p('Second, retain the fixed historical four-type assignment as one comparison and examine new clustering only as an explicitly separate exploratory analysis. Report the dependence on age mapping, education coding and support restrictions. Select a new number of clusters only after coverage and harmonization are defensible.')
p('Third, obtain the 2024 household microdata and metadata when available through an appropriate access channel, then repeat the same audited crosswalk. The locally available household waves are 2013 and 2021; the official MFR page inspected for this report lists those years. The 2024 results used here are published aggregates, not a household-level cluster extension [4,6].')
p('For a subsequent EUReCA study, use the verified equipment and conditional operating responses to define input scenarios, with geography and dwelling characteristics explicitly matched. Independent energy benchmarks would be needed to estimate physical regional calibration multipliers. Household expansion weights do not provide that calibration; expenditure in euros cannot be substituted for energy in kWh. No energy simulation was run for this temporal report.')

# Page 8
page();h('Sources and reproducibility')
p('The extension is descriptive. The waves are repeated cross-sections without linked households. The 2013 CATI and 2021 CAWI/CATI collection modes differ. Compatible cross-wave design variance and harmonization-error estimates have not been established, so this report makes no claim of statistical significance, causal change or individual transitions.')
p('Four bounded specialist tasks covered the frozen baseline, variable comparability and regional factors, numerical analysis, and independent scientific review. The reviewer independently parsed the raw releases and reproduced the eligible assignments, all 80 regional cells, standardization, direct outcomes and principal clustering diagnostics. The analysis package contains 40 frozen artifacts. All 14 protected original source hashes were unchanged at scientific review; they are checked again at delivery.')
sub('Primary references')
link('[1] Khajedehi, Vivian and Zarrella (2025). Identification of Italian user archetypes for building energy simulations based on a national survey. Energy and Buildings 344, 115924. Tables 3 and 6; Figure 5.', 'https://doi.org/10.1016/j.enbuild.2025.115924')
link('[2] ISTAT (2014). I consumi energetici delle famiglie, anno 2013. Equipment and heating context.', 'https://www.istat.it/wp-content/uploads/2014/12/StatReport_Consumi_energetici.pdf')
link('[3] ISTAT (2022). I consumi energetici delle famiglie, anno 2021. Release and linked survey metadata.', 'https://www.istat.it/comunicato-stampa/i-consumi-energetici-delle-famiglie-anno-2021/')
link('[4] ISTAT (2025). Dotazioni energetiche delle famiglie, anno 2024. Cooling section reports 29.4%, 48.8% and 56.0% for 2013, 2021 and 2024. Released 9 October 2025.', 'https://www.istat.it/wp-content/uploads/2025/10/Report_Dotazioni-energetiche-delle-famiglie_Anno-2024.pdf')
p('[5] Native local ISTAT releases and metadata: ECH_Microdati_Anno_2013.txt with coef_red; ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt with PESO. Definitions checked against the released record layouts, classifications and questionnaires. All derived tables are new extension outputs, not replacements for the published 2013 results.')
link('[6] ISTAT. Indagine sui consumi energetici delle famiglie. MFR microdata listing, inspected 4 November 2026. The inspected page lists 2013 and 2021.', 'https://www.istat.it/microdati/indagine-sui-consumi-energetici-delle-famiglie/')
sub('Supporting files in the isolated report package')
p('The package root is output/temporal_extension_2013_2021_20261104. The 00_baseline folder records the printed benchmarks, fixed transfer parameters and source hashes; 01_comparability records codebook evidence, source links and the regional-factor search. The 02_analysis folder contains the prespecified analysis, runnable scripts, assignments, exclusions, coverage and result CSVs. The 03_review folder contains the independent critique and independently calculated evidence. This document and its PDF are in 04_report.')
p('The report resolves two wording issues without altering the frozen numerical package: the released 0-10 exclusion is distinguished from the exact F.10 questionnaire routing, and the one-sided exact-thirds perturbation is omitted from the main evidence. The review-triggered age alignment is explicitly identified as an additional sensitivity rather than a prespecified result.')

doc.core_properties.title='Extending the Italian household archetypes with newer survey data'
doc.core_properties.subject='2013 frozen baseline and descriptive 2021 extension with regional weighting'
doc.core_properties.author='GRINS research working report'
doc.core_properties.keywords='ISTAT; household archetypes; 2013; 2021; regional weights'
doc.save(HERE/'Household_archetypes_temporal_extension.docx')
(HERE/'report_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'docx':str(HERE/'Household_archetypes_temporal_extension.docx'),'content_blocks':len(blocks)},indent=2))
