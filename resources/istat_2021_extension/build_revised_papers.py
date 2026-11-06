"""Build two evidence-bound manuscripts using ReportLab and vector PDF figures.

Run with the bundled PDF Python (reportlab/pypdf). Scientific calculations are
performed separately by revised_analysis.py in the project's environment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, Flowable, CondPageBreak
from pypdf import PdfReader, PdfWriter, Transformation

ROOT=Path(__file__).resolve().parent
WIDTH=A4[0]-96
INK=colors.HexColor('#182837');ACCENT=colors.HexColor('#355d7d')


def load_csv(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def fmt(value, digits=2):
    try:
        v=float(value)
        if v!=v:return 'Unavailable'
        return f'{v:,.{digits}f}'
    except (ValueError,TypeError):return str(value)


def pvalue(value):
    try:
        v=float(value)
        if v!=v:return 'Unavailable'
        return '<0.001' if v<.001 else f'{v:.3f}'
    except (ValueError,TypeError):return 'Unavailable'


def register_fonts():
    folder=Path('C:/Windows/Fonts')
    for name,file in [('Paper','times.ttf'),('Paper-Bold','timesbd.ttf'),('Paper-Italic','timesi.ttf'),('Paper-BoldItalic','timesbi.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(folder/file)))
    pdfmetrics.registerFontFamily('Paper',normal='Paper',bold='Paper-Bold',italic='Paper-Italic',boldItalic='Paper-BoldItalic')


class VectorFigure(Flowable):
    def __init__(self,path,placements,width=WIDTH,max_height=400):
        super().__init__();self.path=Path(path);self.placements=placements
        page=PdfReader(str(path)).pages[0]
        self.source_width=float(page.mediabox.width);self.source_height=float(page.mediabox.height)
        self.scale=min(width/self.source_width,max_height/self.source_height)
        self.width=width;self.height=self.source_height*self.scale
    def drawOn(self,canvas,x,y,_sW=0):
        self.placements.append({'page':canvas.getPageNumber()-1,'path':str(self.path),
                                'x':x+(self.width-self.source_width*self.scale)/2,'y':y,'scale':self.scale})
    def draw(self):pass


class Paper:
    def __init__(self,year,source,output):
        self.year=year;self.source=source;self.output=output;self.story=[];self.placements=[]
        self.prose=[];self.figure_names=[];self.table_names=[]
        self.styles={
            'body':ParagraphStyle('Body',fontName='Paper',fontSize=10.5,leading=14,alignment=TA_JUSTIFY,spaceAfter=7,textColor=INK,allowWidows=0,allowOrphans=0),
            'small':ParagraphStyle('Small',fontName='Paper',fontSize=9,leading=11.6,spaceAfter=5,textColor=INK,allowWidows=0,allowOrphans=0),
            'title':ParagraphStyle('Title',fontName='Paper-Bold',fontSize=21,leading=25,spaceAfter=13,textColor=INK),
            'subtitle':ParagraphStyle('Subtitle',fontName='Paper',fontSize=12,leading=16,spaceAfter=12,textColor=ACCENT),
            'h1':ParagraphStyle('H1',fontName='Paper-Bold',fontSize=14,leading=18,spaceBefore=14,spaceAfter=8,keepWithNext=True,textColor=INK),
            'h2':ParagraphStyle('H2',fontName='Paper-Bold',fontSize=11.5,leading=15,spaceBefore=10,spaceAfter=6,keepWithNext=True,textColor=INK),
            'caption':ParagraphStyle('Caption',fontName='Paper',fontSize=9,leading=11.5,spaceBefore=6,spaceAfter=11,textColor=INK),
            'tablecap':ParagraphStyle('TableCaption',fontName='Paper',fontSize=9,leading=11.5,spaceBefore=8,spaceAfter=5,textColor=INK),
            'cell':ParagraphStyle('Cell',fontName='Paper',fontSize=8.5,leading=10.5,textColor=INK),
            'equation':ParagraphStyle('Equation',fontName='Paper',fontSize=11,leading=16,alignment=TA_CENTER,spaceBefore=7,spaceAfter=10,textColor=INK),
        }
    def para(self,text,style='body'):
        self.story.append(Paragraph(text,self.styles[style]));self.prose.append(text)
    def heading(self,text,level=1):self.para(text,'h1' if level==1 else 'h2')
    def equation(self,text,number=None):self.para(text+(f' &nbsp;&nbsp; ({number})' if number else ''),'equation')
    def table(self,name,caption,header,rows,widths=None):
        self.table_names.append(name)
        # Short tables stay intact with their heading/caption. Long tables may
        # split, but retain a heading plus caption and initial rows on one page.
        short=len(rows)<=8
        if not short:
            if self.story and isinstance(self.story[-1],Paragraph) and self.story[-1].style.name in ['H1','H2']:
                heading=self.story.pop();self.story.extend([CondPageBreak(170),heading])
            else:self.story.append(CondPageBreak(110))
        caption_style=ParagraphStyle('LocalTableCaption',parent=self.styles['tablecap'],keepWithNext=short)
        text=f'<b>{name}.</b> {caption}'
        self.story.append(Paragraph(text,caption_style));self.prose.append(text)
        def cell(v,bold=False):
            text=escape(str(v)).replace('\n','<br/>')
            return Paragraph(('<b>'+text+'</b>') if bold else text,self.styles['cell'])
        data=[[cell(v,True) for v in header]]+[[cell(v) for v in row] for row in rows]
        table=Table(data,colWidths=widths or [WIDTH/len(header)]*len(header),repeatRows=1,hAlign='LEFT')
        padding=4 if name=='Table C.1' else 5
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f4')),
                                  ('LINEABOVE',(0,0),(-1,0),.7,ACCENT),('LINEBELOW',(0,0),(-1,0),.5,ACCENT),
                                  ('LINEBELOW',(0,-1),(-1,-1),.6,ACCENT),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f8f9')]),
                                  ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                                  ('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding)]))
        self.story.extend([table,Spacer(1,7)])
    def figure(self,stem,number,caption,max_height=400):
        self.figure_names.append(number)
        flow=VectorFigure(self.source/'figures'/f'{stem}.pdf',self.placements,max_height=max_height)
        cap=Paragraph(f'<b>Figure {number}.</b> {caption}',self.styles['caption'])
        self.story.append(KeepTogether([Spacer(1,7),flow,cap]));self.prose.append(f'Figure {number}. {caption}')
    def algorithm(self,number,title,steps):
        self.table(f'Algorithm {number}',title,['Step','Operation'],[[str(i),step] for i,step in enumerate(steps,1)],[35,WIDTH-35])
    def build(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        scratch=ROOT/'tmp/pdfs/revised';scratch.mkdir(parents=True,exist_ok=True)
        base=scratch/f'manuscript_{self.year}_text.pdf'
        title=f'Italian household archetypes: corrected {self.year} analysis'
        doc=SimpleDocTemplate(str(base),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=44,
                              title=title,author='Research reanalysis draft; authorship subject to author review',pageCompression=1,invariant=1)
        def furniture(canvas,document):
            canvas.saveState();canvas.setFillColor(ACCENT);canvas.setFont('Paper',8)
            canvas.drawString(48,A4[1]-25,f'ITALIAN HOUSEHOLD ARCHETYPES | {self.year} REANALYSIS')
            canvas.setStrokeColor(colors.HexColor('#c6d2dc'));canvas.setLineWidth(.4)
            canvas.line(48,32,A4[0]-48,32)
            canvas.drawString(48,20,'Corrected methods | Reproducible research draft')
            canvas.drawRightString(A4[0]-48,20,str(document.page));canvas.restoreState()
        doc.build(self.story,onFirstPage=furniture,onLaterPages=furniture)
        reader=PdfReader(str(base));writer=PdfWriter()
        for page in reader.pages:writer.add_page(page)
        for placement in self.placements:
            figure=PdfReader(placement['path']).pages[0]
            transform=Transformation().scale(placement['scale']).translate(placement['x'],placement['y'])
            writer.pages[placement['page']].merge_transformed_page(figure,transform,over=True)
        writer.add_metadata({'/Title':title,'/Subject':'Corrected ISTAT analysis with explicit evidence limitations',
                             '/Author':'Reanalysis prepared for review of the source article','/Creator':'GRINS revised manuscript workflow'})
        with self.output.open('wb') as stream:writer.write(stream)
        extracted='\n'.join(p.extract_text() for p in PdfReader(str(self.output)).pages)
        expected = 13 if self.year == 2021 else 12
        assert len(self.figure_names)==expected and len(set(self.figure_names))==expected
        assert all(f'Figure {n}.' in extracted for n in self.figure_names)
        assert all(f'Table {i}.' in extracted for i in range(1,9))
        assert 'modeled' in extracted.lower()
        assert '\ufffd' not in extracted
        (self.source/'manuscript_text.html').write_text('\n\n'.join(self.prose),encoding='utf-8')
        qa={'year':self.year,'pages':len(reader.pages),'figures':self.figure_names,'tables':self.table_names,
            'vector_placements':len(self.placements),'text_characters':len(extracted),
            'pdf_sha256':hashlib.file_digest(self.output.open('rb'),'sha256').hexdigest(),
            'automated_checks_passed':True,'visual_review_required':True}
        (self.source/'manuscript_checks.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
        print(f'{self.output}: {len(reader.pages)} pages; {len(self.placements)} vector figures/equations',flush=True)


def manuscript(year,root,pdf_dir):
    source=root/str(year);r=json.loads((source/'results.json').read_text())
    model=json.loads((root/'reference_model.json').read_text())
    global_results=json.loads((root/'results.json').read_text())
    shares=load_csv(source/'archetypes.csv');coef=load_csv(source/'probability_coefficients.csv')
    scores={row['model']:row for row in r['probability']['scores']}
    gain=100*(1-scores['quadratic']['brier']/scores['training_marginal']['brier'])
    energy=r['energy'];heat=energy['heating'];hg=load_csv(source/'heating_groups.csv')
    eg=load_csv(source/'electricity_groups.csv')
    p=Paper(year,source,pdf_dir/f'italian_user_archetypes_corrected_{year}.pdf')
    p.para('Identification of Italian user archetypes for building energy simulations','title')
    p.para('Corrected reanalysis of the 2013 national survey' if year==2013 else 'Application of the corrected framework to the 2021 national survey','subtitle')
    p.para('Research manuscript prepared for review. Adapted from Khajedehi, Vivian and Zarrella, <i>Energy &amp; Buildings</i> 344 (2025), 115924 [1]. New calculations replace the source article\'s numerical results. This document does not represent an approved correction, a published article, or endorsement by the original co-authors.','small')
    p.heading('Abstract')
    p.para(f'Occupant-related inputs in building energy simulations require a transparent link between household characteristics, dwelling information and energy-related observations. This study reanalyzes the {year} Italian household energy survey using a corrected and reproducible version of an archetype-based framework. Of {r["raw_n"]:,} released households, {r["eligible_n"]:,} satisfy the common restriction to one to five members and complete education information for members aged at least 15. Age-class midpoints are treated as proxies rather than observed ages. Survey-weighted principal component analysis and four-cluster k-means are fitted only to the 2013 training subset; the same transformations and centroids are used throughout the paired analyses. A multinomial model links dwelling-age and floor-area proxies to four probabilities that are jointly constrained to sum to one. On {r["probability"]["test"]:,} held-out households with known dwelling categories, its multiclass Brier score is {scores["quadratic"]["brier"]:.4f}, compared with {scores["training_marginal"]["brier"]:.4f} for the training-marginal baseline. Training-derived energy groups are selected on a separate tuning subset and evaluated without further merging on the test subset. The heating model retains {heat["groups"]} groups and yields a test root mean squared error of {heat["rmse"]:.2f} hours, compared with {heat["baseline_rmse"]:.2f} hours for a constant predictor. '+
           ('Annual appliance electricity is analyzed as an existing model-derived response, not as measured consumption. ' if year==2013 else 'Cooling ownership and operating hours are analyzed alongside insulation and window improvements during the previous five years. The improvement indicator enters the heating model. Appliance electricity is estimated from ownership and use with a verified 2013 component model and explicit assumptions for coarser 2021 inputs. ')+
           'The findings support a bounded, testable representation of household heterogeneity, but do not demonstrate causal behavioral effects or improved building-simulation accuracy. Sensitivity to coarse age and education inputs remains a central limitation.')
    p.para('<b>Keywords:</b> household archetypes; occupant-related inputs; ISTAT; probabilistic modeling; survey weights; reproducibility; heating schedules.','small')

    p.heading('1. Introduction')
    p.para('The operation of residential buildings depends on more than envelope properties and technical-system efficiency. Occupancy, appliance use, temperature preferences and the timing of heating interact with the physical characteristics of the dwelling. Consequently, a simulation supplied with one generic household profile can differ from a simulation supplied with information about the actual residents. The broader literature on the energy performance gap and occupant behavior provides the motivation for studying these inputs [2,3]. It does not, by itself, establish that any particular demographic clustering improves prediction.')
    p.para('National household surveys offer a practical source of demographic and equipment information at a scale that is rarely available in monitored-building studies. Their strengths are coverage, common questionnaires and documented coding. Their limitations include coarse categories, self-reporting, survey routing and incomplete information about the building and its operation. An archetype method must preserve those distinctions: a demographic label is not an observation of a behavioral mechanism, and a reported operating duration is not a measurement of delivered heat.')
    p.heading('1.1. International standards',2)
    p.para('Schedule-based standards provide reference inputs for energy calculations. ISO 18523-2:2018 specifies formats and categories for residential schedules and conditions, including occupancy, technical-system operation, appliance usage and internal gains [8]. Its publicly available scope identifies the included values as informative. EN 16798-1:2019 is the source of a second set of examples discussed by the original article [1,9]. These sources have different purposes from a fitted household prediction model. A difference between a reference schedule and a survey mean is therefore not sufficient evidence that the standard is erroneous.')
    p.para('The corrected comparison retains the source article\'s interest in diversity, but removes unsupported claims about the motives or composition of standardization committees. It also separates appliance electricity from heat released within occupied space. Electrical energy may be converted into sensible heat, latent heat, heat removed by water, or energy dissipated elsewhere. A model of appliance kWh cannot be inserted unchanged as an internal-gain schedule without assumptions about that allocation and its timing.')
    p.heading('1.2. Literature review',2)
    p.para('Reviews describe a wide range of occupant-modeling approaches, from fixed schedules to stochastic representations and explicit interactions with building systems [3]. Monitored occupant-behavior databases provide information about presence and interactions with equipment that is more direct than demographic variables [4]. Nevertheless, such observations are not routinely available for every dwelling in a building-stock model. Survey-derived archetypes may therefore be useful as uncertain inputs when detailed information is absent.')
    p.para('Studies of alternative occupancy-profile assumptions show that schedule selection can materially change simulated demand [5]. This motivates sensitivity analysis rather than a presumption that more archetypes necessarily produce a more accurate model. A useful archetype framework should provide a reproducible feature definition, a clearly identified training population, probabilities for unknown occupants, and an evaluation against simple alternatives. The present analysis addresses those requirements while retaining the source article\'s sequence of demographic clustering, dwelling association and energy-pattern analysis [1].')
    p.heading('1.3. Research gap and objectives',2)
    p.para('Three questions guide this reanalysis. First, can a fixed four-archetype representation summarize the available household information under explicit preprocessing assumptions? Second, does dwelling information improve out-of-sample probability prediction beyond the overall archetype distribution? Third, do combinations of archetype, location and dwelling type improve prediction of the available energy-related response? The questions are associative and predictive. They do not ask whether archetype membership causes energy use, nor whether a resulting schedule has already reduced a building simulation\'s performance gap.')
    p.para(('The 2013 manuscript establishes the corrected reference model and documents the changes from the published analysis. ' if year==2013 else 'The 2021 manuscript applies the frozen 2013 demographic model rather than independently redefining and renaming four new clusters. ')+
           'The manuscript retains corresponding figure types, coefficient tables, group summaries and discussion sections. Where the available evidence cannot support the original statistic or comparison, the replacement is stated explicitly rather than generating a visually similar but scientifically different result without explanation.')

    p.heading('2. Dataset')
    p.para(('The 2013 release contains 20,000 household records from the ISTAT survey of household energy consumption. Its telephone-based collection and landline sampling frame are described in the supplied methodological documentation [6]. ' if year==2013 else 'The 2021 research release contains 54,069 household records. ISTAT documents a mixed web/telephone collection strategy and a different sampling frame from the 2013 survey [7]. Equipment and dwelling questions describe the survey context, whereas the heating-hours questions refer to winter 2020. These periods must not be collapsed into an unqualified 2021 energy-demand observation. ')+
           'The released record identifier is used only within its own survey. No attempt is made to link individual households between years. The paired analyses are repeated cross-sections, not a panel.')
    p.para(f'To define the same accessible household-size domain in both years, households with fewer than one or more than five members are excluded. The 2021 release supplies demographic fields for at most five members and top-codes its larger-household category. Each included member must have a valid age category; all members in categories beginning at age 15 or above must also have a known education category. Children are not assigned missing adult education values of zero. These rules retain {r["eligible_n"]:,} households and exclude {r["raw_n"]-r["eligible_n"]:,}. Exclusion reasons are exported by record identifier. This complete-case domain is not the entire national household population.')
    p.para('The official household expansion weights are retained. They define weighted means, distributions, PCA, clustering and probability-model losses. People-weighted archetype shares multiply each household weight by its recorded number of members. Reported household counts remain actual sample counts, not rounded expansion totals. Because the analysis does not reconstruct the survey\'s complete variance-estimation design, its standard errors and tests are explicitly model-based; using expansion weights is not a substitute for strata, primary sampling units or replicate weights.')
    p.para(f'Unknown construction period accounts for {r["construction_unknown_pct"]:.2f}% of the eligible household weight. Such records remain in the demographic analysis but are omitted when a dwelling-age predictor is required. The dwelling model therefore describes a conditional sample whose composition may differ from that of households with an unknown construction year. '+
           ('The existing appliance-electricity file is joined one-to-one by the 2013 household identifier. It contains modeled annual appliance electricity; the external model chain that produced it is not rerun here. Its uncertainty and possible overlap with the survey predictors limit the interpretation of electricity associations.' if year==2013 else 'The 2021 appliance-electricity response is constructed from survey ownership and use using the main-branch appliance model. Its component boundary reproduces all 20,000 saved 2013 household totals to numerical precision: lighting, small appliances, CE refrigerators/freezers, CE washing/drying/dishwashing, TVs/computers and standby. Cooking, space heating, hot water and cooling electricity are excluded. This is a modeled annual reference estimate, not a metered observation or an expenditure conversion.'))
    p.para('Building classifications also require semantic harmonization. Both releases distinguish single-family and multi-family dwellings, but their apartment subcategories do not describe the same attribute. The common classification therefore uses single-family, multi-family and apartment. Geographic divisions are consolidated into north-west, north-east, centre, and south/islands. The 2013 subjective insulation rating is retained for that wave. In 2021, v3_14gh_new records insulation of walls, ceilings or floors and/or replacement doors, windows or frames in the last five years. This renovation measure is analyzed explicitly as insulation/window improvements; it does not measure current thermal performance. The comparable 2013 renovation items are q_3_19_7 and q_3_19_8, rather than q_3_17.')

    p.heading('3. Methods')
    p.para('Figure 1 summarizes the corrected workflow. All record-level decisions precede model fitting. Eligible households in each release are assigned to training (70%), tuning (15%) and test (15%) subsets using a saved seed and sorted identifiers. The 2013 training subset is the only input to the demographic PCA, normalization and reference clustering. Tuning selects probability regularization and energy-group complexity. Test households are not used for those choices or for subsequent group merging.')
    p.figure('fig01_process','1','Corrected analysis sequence. The split is made before fitting. For 2021, demographic transformations and archetype centers are imported unchanged from the 2013 training analysis. Appliance kWh are modeled with explicit parameter assumptions; unknown insulation responses are excluded from its comparisons.',max_height=290)
    p.heading('3.1. PCA and clustering',2)
    p.table('Table 1','Household variables and corrected operational definitions.',
            ['Variable','Definition','Interpretation / restriction'],[
                ['Minimum age','Minimum of the released age-category midpoint proxies','Not an observed exact age; children retain their published interval.'],
                ['Maximum age','Maximum of the same member-level proxies','Open oldest interval capped at 100 years; cap 90 tested.'],
                ['Household size','Recorded number of members, restricted to 1-5','Same accessible household-size domain in both releases.'],
                ['Schooling proxy','Mean assigned completed-schooling years / 18 among members in age classes beginning at 15+','Known education required. Not the UNDP education index and not an age-normalized opportunity measure.']
            ],[88,210,WIDTH-298])
    p.para('The midpoint convention avoids a single undocumented random realization, but does not turn interval data into exact ages. The releases use non-nested age bands: for example, a 2013 band beginning at 25 cannot be exactly reconstructed from the 2021 bands beginning at 20 and 30. Their midpoints are therefore transparent approximations, with seeded within-interval draws used in sensitivity analysis. The oldest categories are bounded at 100 years in the reference calculation. Neither the cap nor the uniform-draw scenarios is asserted to describe the true within-band distribution.')
    p.para('Education is harmonized to four shared categories: no qualification/primary, lower secondary, upper secondary, and tertiary. Assigned schooling values are 2.5, 8, 12 and 18 years. The first category combines two qualifications in 2021; its midpoint of the possible 0-5-year range is a working proxy, not an estimated mean. Values of 0 and 5 are tested separately. The fixed denominator removes the erroneous dependence on age-class codes and avoids claiming that the bespoke ratio reproduces the UNDP method. Students can have lower completed attainment because they have not finished education; the variable is not interpreted as ability or opportunity.')
    p.equation('E<sub>h</sub> = (1 / m<sub>h</sub>) Σ<sub>i in M(h)</sub> (s<sub>i</sub> / 18), &nbsp; 0 ≤ E<sub>h</sub> ≤ 1')
    p.para('Here M(h) contains members in categories beginning at age 15 or above, m(h) is their count, and s is the assigned schooling value. A weighted correlation-matrix PCA is fitted after centering and scaling each demographic variable with its training-weighted mean and standard deviation. A four-variable exploratory PCA is also reported, but education is kept as a separate clustering coordinate. The fitted first two demographic components are:')
    for number in [1,2]:
        flow=VectorFigure(source/'figures'/f'eq{number:02d}_pca.pdf',p.placements,max_height=55)
        p.story.append(KeepTogether([flow,Paragraph(f'({number})',p.styles['equation'])]))
    centers=model['pca']['mean'];scales=model['pca']['scale']
    p.para(f'In Equations (1)-(2), z denotes the centered and standardized minimum-age, maximum-age and household-size proxy, respectively. The training means are {centers[0]:.6f}, {centers[1]:.6f} and {centers[2]:.6f}; the corresponding scales are {scales[0]:.6f}, {scales[1]:.6f} and {scales[2]:.6f}. Full-precision loadings are exported with the model. Component signs are fixed by requiring the largest absolute loading to be positive. Unlike the source article\'s rounded engineering expressions, these components are generated by an orthonormal eigenvector decomposition of the stated covariance matrix.')
    p.para('The two component scores and E are min-max normalized using the 2013 training limits only. A weighted k-means objective minimizes the sum of weighted squared distances to the four centroids. Twelve deterministic k-means++ starts are attempted and the minimum-inertia fit is retained. Four clusters are chosen to preserve the study\'s intended level of resolution, not because an elbow plot proves that four is uniquely optimal. Labels are fixed from the reference training summaries: A2 has the greatest mean household size; among the remainder A4 has the highest schooling proxy, A1 the highest minimum age, and A3 is the remaining cluster. These labels do not guarantee correspondence to the original paper\'s clusters.')
    p.equation('J = Σ<sub>h</sub> w<sub>h</sub> ||u<sub>h</sub> - c<sub>g(h)</sub>||² / Σ<sub>h</sub> w<sub>h</sub>')
    p.para('Regional refits are aligned to the fixed reference by minimum centroid distance and shown descriptively. The inferential regional comparison uses individual test households as its unit, with PC1, PC2 and E as outcomes and region as the factor. Classical Wilks and Roy statistics are provided as unweighted diagnostics, not survey-design-adjusted national tests. Failure to reject a regional test is not evidence of equivalence, and resemblance among four centroid locations does not imply equal archetype prevalence.')

    p.heading('3.2. Probabilistic feature extraction',2)
    p.para('Dwelling-age and area information is available in categories. The corrected empirical extraction uses 24 non-overlapping combinations of four construction-period groups and six common area groups. This avoids reusing the same household in many overlapping windows and calling those windows independent observations. For each cell, the workflow exports the actual count, sum of weights, effective sample size and four observed weighted proportions. Support is flagged when either the actual count or effective sample size is below 30; the threshold is a display and caution rule, not a significance test.')
    p.equation('p̂<sub>jk</sub> = Σ<sub>h in j</sub> w<sub>h</sub> 1[g(h)=k] / Σ<sub>h in j</sub> w<sub>h</sub>; &nbsp; n<sub>eff,j</sub> = (Σ w<sub>h</sub>)² / Σ w<sub>h</sub>²')
    p.para('The area proxies are 20, 49.5, 74.5, 104.5, 134.5 and 200 m². The first and last values require explicit bounds of 0-39 and 150-250 m². Construction proxies use 1924.5, 1959.5 and 1979.5 for the three older periods; the 1990+ group uses the midpoint between 1990 and the survey year. Thus the oldest group is represented by an assumed 1900-1949 interval. These values support a compact plotting and regression coordinate system, not a recovery of the unknown continuous values. Extrapolation outside the represented support is not recommended.')
    p.para('A multinomial logistic model replaces four independently fitted polynomial probability regressions. A quadratic feature vector retains the source article\'s age, area, squares and interaction terms, but a softmax link guarantees a valid joint distribution. Archetype A4 is the reference, with its coefficient column fixed to zero. The coefficient matrix is fitted by weighted household-level cross-entropy, aggregated over identical predictor vectors only to accelerate computation:')
    p.equation('φ(x) = [1, a/100, A/100, a²/10⁴, A²/10⁴, aA/10⁴]<sup>T</sup>')
    p.equation('p<sub>k</sub>(x) = exp[β<sub>k</sub><sup>T</sup>φ(x)] / Σ<sub>l=1</sub><sup>4</sup> exp[β<sub>l</sub><sup>T</sup>φ(x)], &nbsp; β<sub>4</sub> = 0',3)
    p.para('A small ridge penalty is applied to slopes, not intercepts. Its value is selected from four prespecified candidates using tuning-set log loss. Coefficient p-values, Gaussian F-statistics and adjusted R² are not reported for this penalized multinomial model; they are not valid substitutes for probabilistic prediction scores. A secondary model adds the common location and dwelling-type indicators. It is evaluated as an alternative, not used in an iterative p-value removal chain. The original claim that all omitted variables have no predictive value is therefore not carried forward.')
    p.algorithm('1','Corrected empirical extraction and probability fitting.',[
        'Create a household split before PCA, archetype discovery or probability fitting. Use the fixed 2013 archetype definition.',
        'Retain known construction and area categories; form the 24 common cells and export counts, effective counts and four weighted proportions.',
        'Fit the household-weighted multinomial quadratic model on training households. Choose the ridge penalty by tuning log loss.',
        'Freeze the coefficients. Compute all four probabilities jointly; require positivity and a row sum of one without clipping.',
        'Score test households using multiclass Brier score and log loss against the training-marginal baseline. Plot test-cell calibration separately.',
        'Report sparse-cell flags, alternative predictors and interval-assumption limitations. Do not retune from test results.'
    ])
    p.equation('BS = Σ<sub>h in test</sub> w<sub>h</sub> Σ<sub>k=1</sub><sup>4</sup> (p<sub>hk</sub> - 1[g(h)=k])² / Σ<sub>h in test</sub> w<sub>h</sub>')
    p.para('The Brier score is the sum over four classes, not the class-averaged convention. Lower Brier score and lower log loss are better. The calibration plot compares predictions with weighted proportions among test households in each cell. Its points are not independent estimates of a continuous probability surface, and sparse cells should not be interpreted as precise failures or successes. The entire test evaluation is conditional on the stated demographic measurement model.')

    p.heading('3.3. Regression analysis and energy groups',2)
    p.para('The heating response is the sum of reported operation in three eight-hour daily bands. To respect the differing question routes, both years are restricted to use-frequency categories 1 or 2, with all three hours observed and each value between 0 and 8. Missing hours are not zeros. The resulting mean describes a typical reported winter day among frequent users; it is neither an annual heating duration nor the mean over all households. '+
           ('The electricity response is the existing annual appliance-kWh estimate associated with the 2013 record.' if year==2013 else 'The 2021 annual appliance-kWh response is modeled using the verified component boundary and the assumptions documented below.'))
    p.para('For correspondence with the original ANOVA table, the weighted linear model includes archetype, common dwelling type, location, the wave-specific insulation measure (subjective rating in 2013; recent improvements in 2021), age proxy, area proxy, cooling-system presence and the listed interactions. The 2021 specification was revised after the source-variable audit and is not a preregistered analysis. HC3 heteroscedasticity-robust covariance is used for joint coefficient-block Wald F tests on test households [10]. Main-effect block tests in a model containing interactions are conditional on its reference coding. They should not be read as total causal effects. Holm correction is applied across the available blocks within each response. The use of final survey weights does not make these full design-based survey tests.')
    p.para('Cooling presence is reconstructed from the equipment inventory. In 2013, complete skipped inventory blocks with no use-frequency response are treated as routed absence; partial unexplained missingness remains unknown. In 2021, v2_11a-e are answered for every household: any code 1 establishes ownership and five code-2 responses establish absence. The variable v2_11conta counts types of equipment, not individual devices. Daily cooling hours sum v5_7a-c only for v5_6 codes 1-3, the questionnaire routing for frequent use. Occasional, never-use and absent-system records do not receive imputed daily hours. These operating questions refer to summer 2020.')
    if year == 2021:
        p.para('The heating model includes insulation/window improvements and its interactions with archetype, location and dwelling type. A reported improvement requires v3_13=1 and v3_14gh_new=1. An explicit no to investments (v3_13=2), or an investment respondent with the improvement item unselected, is classified as not reported. Records without a response to the routed investment question remain unknown. No recent improvement does not imply poor insulation: work may predate the five-year window or the dwelling may already be well insulated. The 2021 investment question is routed to households with heating equipment; the 2013 question was asked of everyone.')
        p.para('Appliance electricity uses the verified 2013 component boundary with power, efficiency and operating-day assumptions held fixed. Washing uses the observed temperature mix and 47 weeks/year, at 1.8 kWh per reference cycle with factors 0.3, 1 and 2.56. Separate drying and dishwashing use 1 and 1.5 kWh/cycle. The infrequent-use category is represented by 18 cycles/year. Lighting hours and refrigerator size/multiplicity use weighted 2013 reference distributions; unlisted small appliances receive a common weighted 2013 background. Refrigerator age intervals map by their midpoint to the legacy efficiency assumptions. TV power uses the primary type, while all computers/tablets use the primary computer category with legacy annual rates. These approximations are modeled inputs, not observations of missing 2021 characteristics. They can affect variance and correlations, especially where a common background replaces household variation.')
        p.para('The supplied June 2022 report workbook supports the measurement definitions: Table 16 summarizes lighting, Table 18 appliance ownership and Table 19 cooking fuels. These are aggregate tables, not appliance power measurements. The December workbook Table 12 describes washing temperatures. Neither workbook supplies the missing household-specific refrigerator capacities or individual small-appliance inventory. The electricity model therefore includes a parameter sensitivity analysis, reported in Appendix B. These scenarios hold reference efficiencies fixed; the analysis does not estimate technological efficiency change between 2013 and 2021.')
    p.para('Energy-group construction is changed more substantially. Repeatedly merging the least significant Tukey pair on the same observations does not preserve the guarantee associated with one fixed family of comparisons. The corrected procedure starts from training cells defined by location, common dwelling type and archetype. Cells with fewer than 30 training observations are unsupported. Adjacent cells ordered by their training-weighted means are merged by minimum increase in weighted squared error. Candidate partitions contain one to eight groups; tuning-set RMSE selects the number. The selected map is frozen before test evaluation.')
    p.algorithm('2','Split-sample grouping and verification, replacing adaptive Tukey merging.',[
        'Use training households to estimate outcome means for supported location x dwelling-type x archetype cells.',
        'Order cells by mean and merge adjacent cells by minimum weighted squared-error increase; retain partitions with one to eight groups.',
        'Choose the partition with lowest tuning-set RMSE. For unsupported cells use the training-wide mean, not a guessed group.',
        'Freeze memberships and training predictions. Evaluate RMSE on all eligible test households, including fallback predictions.',
        'Summarize test-group distributions and perform one unweighted Tukey-Kramer comparison family only after group selection is complete.',
        'Also report survey-weighted, heteroscedasticity-robust pairwise contrasts with Holm correction. Do not merge groups again or call all groups significant.'
    ])
    p.para('Tukey-Kramer results are supplied as a classical sample-level diagnostic conditional on the learned partition, with their equal-variance and independence assumptions [11]. The weighted robust contrasts provide a different, explicitly model-based check. Neither establishes equality merely because a contrast is not significant. Group labels are ordered by training means, so their test means need not remain strictly ordered. Weighted boxplots show the median, quartiles and non-outlying whiskers of the test distributions; suppressing plotted outliers does not remove observations from fitting or evaluation.')

    p.heading('4. Results and discussion')
    p.heading('4.1. Principal components and clusters',2)
    pcvar=100*sum(model['pca']['explained_ratio'][:2]);allvar=100*sum(model['all_variable_pca']['explained_ratio'][:3])
    p.para(f'The first three components of the four-variable training PCA explain {allvar:.2f}% of weighted standardized variance. The first two demographic components explain {pcvar:.2f}% (Figure 2). Every cumulative curve reaches 100% when all its components are included. '+
           ('These are estimates from the 2013 training subset, not values selected to reproduce the percentages in the published figure.' if year==2013 else 'The curves are shown from the 2013 reference fit. They are not mislabeled as a newly fitted 2021 transformation; their purpose here is to document the basis used to assign 2021 households.'))
    p.figure('fig02_pca','2','Cumulative explained variance for the 2013 training-weighted correlation-matrix PCA: (a) all four household variables; (b) the three demographic variables. These fitted bases are reused unchanged in 2021.',max_height=240)
    p.para('Figure 3 reports within-cluster weighted mean squared distance for one to ten clusters. The curve is a resolution diagnostic: increasing the number of centers generally reduces this objective even when no unique natural partition exists. A universal Calinski-Harabasz threshold is not invoked, and choosing four clusters is presented as a modeling convention consistent with the research question. '+
           ('The four-center reference is fitted once to the training data and used for all subsequent subsets.' if year==2013 else 'The 2021 elbow curve is an exploratory refit in the frozen reference coordinate system; those refitted centers are not used for the reported assignments.'))
    p.figure('fig03_elbow','3',f'Weighted within-cluster dispersion for the {year} training subset. The vertical line marks the prespecified four-archetype resolution, not a statistically established optimum.',max_height=260)
    p.para(f'There are {r["outside_reference_range"]:,} eligible households outside at least one 2013 training min-max limit. Their coordinates are retained rather than clipped. Figure 4 compares regional descriptive refits with the national training reference. Regional symbols are scaled by sample count rather than by an unsupported interpretation of a clustering-validity index. The connecting segments identify four aligned centers; they do not represent trajectories or individual household transitions.')
    p.figure('fig04_regional_centroids','4',f'Regional four-center refits for the {year} eligible sample (grey) and the frozen 2013 training centers (colored). The source-style two-dimensional projection shows normalized schooling and PC1; PC2 is omitted from the display, not from clustering. Regional fits share the reference coordinates and are aligned by minimum distance. Grey marker area increases with regional sample count.',max_height=340)
    manova=load_csv(source/'regional_manova.csv');selected=[row for row in manova if row['Statistic'] in ["Wilks' lambda","Roy's greatest root"]]
    p.table('Table 2','Regional MANOVA on individual held-out households; unweighted diagnostic, not centroid-level or survey-design inference.',
            ['Statistic','Value','F','Numerator df','Denominator df','p'],
            [[row['Statistic'],fmt(row['Value'],4),fmt(row['F Value'],3),fmt(row['Num DF'],0),fmt(row['Den DF'],0),pvalue(row['Pr > F'])] for row in selected],
            [110,70,70,75,90,WIDTH-415])
    p.para('The unit of analysis in Table 2 is a household, not one of four fitted centroids per region. The table tests equality of multivariate conditional means under the classical model assumptions. It cannot establish regional equivalence of full demographic distributions or of archetype shares. Moreover, large samples can identify small departures from mean equality; the p-values should be considered alongside centroid geometry and the substantive size of regional differences, rather than turned into a binary assertion that geography does or does not matter.')
    p.heading('4.2. Definition of user archetypes',2)
    p.table('Table 3','Weighted demographic summaries by fixed-reference archetype. Entries are mean (population-weighted standard deviation); ages are category proxies.',
            ['Archetype','n','Minimum age','Maximum age','Members','Schooling proxy'],
            [[f'A{row["archetype"]}',fmt(row['n'],0)]+[f'{fmt(row[f+"_mean"],2)} ({fmt(row[f+"_sd"],2)})' for f in ['agemin','agemax','nmembers','education']] for row in shares],
            [62,60,98,98,90,WIDTH-408])
    for row in shares:
        a=int(row['archetype'])
        p.para(f'<b>Archetype A{a}.</b> The weighted mean minimum and maximum age proxies are {fmt(row["agemin_mean"],1)} and {fmt(row["agemax_mean"],1)} years, with {fmt(row["nmembers_mean"],2)} members and a schooling proxy of {fmt(row["education_mean"],3)}. This group accounts for {fmt(row["household_share"],2)}% of eligible household weight and {fmt(row["people_share"],2)}% of people weight within those households. These summaries describe the fitted group; they do not identify employment, retirement, parenthood or a particular domestic relationship unless those characteristics are separately observed.')
    p.para('Figure 5 distinguishes households from people. A cluster with a larger average household size can represent a greater share of people than of households. All shares use the official weights and the same eligible domain. They should not be directly compared with the original article\'s unweighted full-sample figures, because feature definitions, exclusions, weighting and fitted centers have changed. Similarly, a fixed cluster label describes proximity in the modeled feature space rather than a diagnostic category of occupant behavior.')
    p.figure('fig05_shares','5',f'Weighted shares of the four fixed-reference archetypes in the {year} eligible sample. Stacked bars distinguish households from people within those households, matching the source figure type. Both bars are conditional on one to five members and complete required demographic information.',max_height=280)
    p.heading('4.3. Associating user archetypes with building types',2)
    p.para(f'The dwelling-probability analysis includes {r["probability"]["eligible"]:,} households: {r["probability"]["train"]:,} for fitting, {r["probability"]["tune"]:,} for tuning and {r["probability"]["test"]:,} for final evaluation. The selected ridge penalty is {r["probability"]["penalty"]:g}. Table 4 reports the actual coefficient matrix used in Equation (3), followed by household-level test scores. Its A4 reference column is zero by construction; A4 nevertheless has a nonzero probability through the softmax denominator.')
    p.table('Table 4','Multinomial quadratic coefficient matrix. Values are log-odds coefficients relative to A4, not direct probability effects. All computations use unrounded coefficients.',
            ['Term','A1 vs A4','A2 vs A4','A3 vs A4','A4 reference'],
            [[row['term']]+[fmt(row[a],6) for a in ['A1','A2','A3','A4']] for row in coef],[160,85,85,85,WIDTH-415])
    p.table('Table 4 (continued)','Held-out household predictive performance. Lower is better; ridge settings were chosen without using the test subset.',
            ['Model','Test n','Multiclass Brier','Log loss'],
            [[{'quadratic':'Age-area quadratic','quadratic_plus_location_type':'Quadratic + location/type','training_marginal':'Training marginal only'}[name],str(row['n']),fmt(row['brier'],4),fmt(row['log_loss'],4)] for name,row in scores.items()],
            [220,65,110,WIDTH-395])
    p.para(f'The quadratic model changes the Brier score by {gain:+.2f}% relative to the marginal baseline, where a positive value denotes improvement. The secondary location/type model has a test Brier score of {scores["quadratic_plus_location_type"]["brier"]:.4f}. This comparison is more directly relevant to assigning an unknown household than a high in-sample R² on smoothed proportions. It also prevents interpreting probabilistic regression as an exception to classification: a multinomial classifier itself produces probabilities, and a poor hard-label confusion matrix does not establish that probabilistic classification is unsuitable.')
    p.para('Figure 6 shows smooth model profiles against the floor-area proxy, with one line for each represented building-age proxy. The dots are empirical proportions from test households, not values from the fitting subset. The curves interpolate between categorical proxy coordinates and must not be read as evidence of an observed within-category area response. In particular, their oldest and largest categories depend on chosen finite bounds. Crosses identify cells with low actual or effective support.')
    p.figure('fig06_probability_profiles','6','Archetype probability versus dwelling floor-area proxy. Grey line styles distinguish building-age proxies in the jointly constrained quadratic multinomial model. Archetype-colored points are weighted held-out cell proportions; crosses have fewer than 30 actual or effective observations. The axes match the caption.',max_height=380)
    p.para('In Figure 7, the diagonal is the calibration reference. Departure from it reflects differences between the model and a finite test-cell proportion, including sampling variability and model misspecification. Large and small cells are not equally informative, so marker sizes reflect effective sample size. No claim of near-perfect individual prediction is inferred from visual alignment. The predicted probability vector is checked for positivity and unit sum on every eligible household without clipping individual components.')
    p.figure('fig07_calibration','7','Predicted probabilities (vertical axis) against held-out weighted cell proportions (horizontal axis), with four archetype colors in one source-style plot. The diagonal denotes calibration; marker area increases with effective sample size and crosses identify weakly supported cells. This replaces in-sample fitted-proportion validation.',max_height=360)
    p.heading('4.4. Deriving energy-use patterns for occupant groups',2)
    p.para(f'The matched frequent-heating sample contains {heat["n"]:,} households, including {heat["test_n"]:,} test observations. Its weighted mean reported operating duration is {heat["mean"]:.2f} hours per winter day. The energy-group procedure retains {heat["groups"]} heating groups. Predictions fall back to the training-wide mean for {heat["unsupported_test"]} test households whose categorical cell lacks the minimum training support. These observations remain in the predictive score rather than being discarded to improve apparent performance.')
    ah=load_csv(source/'heating_anova.csv');ae={row['term']:row for row in load_csv(source/'electricity_anova.csv')}
    p.para(f'The complete-covariate heating regression in Table 5 uses {heat["anova"]["n"]:,} test households. '+
           (f'The corresponding electricity regression uses {energy["electricity"]["anova"]["n"]:,}. ')+
           'These counts are smaller than the group-evaluation samples because the regression additionally requires known age/area proxies and the available equipment and insulation covariates. The fitted design matrices are checked for full column rank.')
    p.table('Table 5','Energy-model coefficient-block tests on held-out households. HC3 Wald F statistics and Holm-adjusted p-values; unavailable variables/responses are not substituted.',
            ['Term','Heating F','Heating p(Holm)','Appliance kWh F','kWh p(Holm)'],
            [[row['term'],fmt(row['F'],2),pvalue(row.get('p_holm')),
              fmt(ae[row['term']]['F'],2) if row['term'] in ae else 'Unavailable',
              pvalue(ae[row['term']].get('p_holm')) if row['term'] in ae else 'Unavailable'] for row in ah],
            [170,65,90,85,WIDTH-410])
    significant=heat['anova']['holm_significant_terms']
    p.para('For heating, the blocks passing the within-response Holm threshold of 0.05 are '+(', '.join(significant) if significant else 'none')+'. The table and this statement are generated from the same result objects. Significance does not quantify practical predictive gain, establish causal effects or validate all interactions. A group can have a distinct distribution without every pairwise difference being statistically established, and different grouping criteria can produce a different number of groups.')
    for outcome,groups in [('heating',hg),('electricity',eg)]:
        suffix='(a)' if outcome=='heating' else '(b)'
        if not groups:
            p.table('Table 6 (continued)','(b) Appliance-electricity groups.',['Response','Status','Reason'],[['Annual appliance kWh','Unavailable','No matching 2021 outcome in the supplied files.']], [120,85,WIDTH-205]);continue
        p.table('Table 6' if outcome=='heating' else 'Table 6 (continued)',f'{suffix} Training-defined {outcome} groups, summarized on test households. Examples are the highest-count training cell assigned to each group, not duplicated illustrative scenarios.',
                ['Group','Example: location / dwelling / archetype','Test n','Mean (SD)','SE'],
                [[row['group'],row['example_cell'].replace('|',' / '),fmt(row['n_test'],0),f'{fmt(row["mean"],2)} ({fmt(row["sd"],2)})',fmt(row['se'],2)] for row in groups],
                [42,220,55,110,WIDTH-427])
    p.para(f'The heating-group test RMSE is {heat["rmse"]:.3f} hours, compared with {heat["baseline_rmse"]:.3f} hours for the training-wide mean. '+
           (f'The electricity model selects {energy["electricity"]["groups"]} groups and has a test RMSE of {energy["electricity"]["rmse"]:.1f} kWh/year, compared with {energy["electricity"]["baseline_rmse"]:.1f} kWh/year for its constant baseline. Its target is model-derived, so this is a test of prediction of that modeled response, not validation against household meters. ')+
           'Full group maps and fixed-family pairwise tests are exported. The groups are called training-defined groups rather than significant groups, because the procedure does not enforce significance by repeated merging on the test response.')
    p.figure('fig08_energy_groups','8','Weighted held-out distributions of (a) heating operating hours and (b) modeled annual appliance electricity. Boxes show quartiles, the median and 1.5-IQR whiskers; plotted outliers are suppressed but retained in calculations. '+('Heating is stratified by reported insulation, electricity by cooling-system presence.' if year==2013 else 'Heating is stratified by reported versus not-reported insulation/window improvements in the last five years. Unknown improvement responses are omitted from the stratified boxes. Electricity is stratified by cooling ownership. Heating p-values compare yes/no improvements within each fixed group using weighted HC3 contrasts, Holm-adjusted over the displayed groups.'),max_height=440)
    p.para('Differences between the displayed subgroups are descriptive associations. For example, conditioning on reported insulation does not isolate an insulation intervention: dwelling condition, climate, resources and operation may all differ. Cooling presence is an equipment characteristic and should not be described as a measured cooling-energy effect. The original narrative\'s claims about building transmission losses, occupant thermal comfort and appliances causing the group differences are not retained without corresponding measurements or an identification strategy.')

    def equipment_section():
        p.heading('5.3. Cooling equipment, operating hours and envelope improvements',2)
        eq = load_csv(source/'equipment_summary.csv')
        overall = {row['measure']:row for row in eq if row['archetype']=='All'}
        p.para(f'In the eligible demographic sample, {r["equipment"]["cooling_present_n"]:,} households report air conditioning, corresponding to {fmt(overall["AC ownership"]["estimate"],1)}% after weighting. Insulation/window improvements are reported by {r["equipment"]["improvement_reported_n"]:,} households, or {fmt(overall["Insulation/window improvements"]["estimate"],1)}% of eligible households with an interpretable investment response. There are {r["equipment"]["improvement_unknown_n"]:,} unknown investment responses. The mean daily cooling duration is {fmt(overall["Cooling hours (frequency 1-3)"]["estimate"],2)} hours among {r["equipment"]["cooling_hours_n"]:,} eligible frequent users. This is a conditional duration, not an all-household daily or annual mean.')
        p.table('Table 9','Weighted equipment and improvement estimates in the eligible sample. Counts are unweighted denominators; percentages and hours are weighted. Each measure uses its own responding domain.',
                ['Archetype','AC n / %','Improvement n / %','Cooling-use n / hours'],
                [[str(a) if a=='All' else 'A'+str(a)]+[
                    f'{int(next(row for row in eq if row["archetype"]==str(a) and row["measure"]==m)["n"]):,} / '+
                    fmt(next(row for row in eq if row['archetype']==str(a) and row['measure']==m)['estimate'],2)
                    for m in ['AC ownership','Insulation/window improvements','Cooling hours (frequency 1-3)']]
                 for a in ['All',1,2,3,4]], [65,140,145,WIDTH-350])
        p.figure('fig11_equipment','11','Weighted descriptive estimates by frozen-reference archetype: AC ownership; insulation/window improvements during the preceding five years; and daily cooling hours among frequency codes 1-3. Denominators are given in Table 9. Bars are descriptive estimates without confidence intervals; they are not causal effects.',max_height=255)
        pairs=load_csv(source/'heating_insulation_pairs.csv')
        p.table('Table 10','Within-heating-group comparison of households reporting insulation/window improvements (yes) versus not reporting them (no). Unknown responses are excluded. Differences are weighted yes-minus-no means in hours/day; HC3 p-values are Holm-adjusted over these fixed groups.',
                ['Group','Yes n','No n','Unknown excluded','Difference (h)','p (Holm)'],
                [[row['group'],row['yes_n'],row['no_n'],row['excluded_unknown_n'],fmt(row['difference_hours'],2),pvalue(row['p_holm'])] for row in pairs],
                [45,65,65,105,110,WIDTH-390])
        p.para('These data support equipment and operating-profile comparisons alongside modeled appliance electricity. Cooling ownership includes functioning systems that are never used. Renovation decisions may reflect income, dwelling condition and heating needs; their association with operating duration does not identify energy savings. The heating model retains its training-defined groups and held-out evaluation. The additional descriptive estimates use the full eligible sample, and are not presented as independent predictive validation.')

    p.heading('5. Implications for building simulations')
    p.para('The corrected outputs provide a way to represent uncertainty about the occupants of a dwelling. A simulator can draw an archetype from the four modeled probabilities and then use a compatible distribution of operating parameters. Repeated draws represent occupant-assignment uncertainty more faithfully than treating the most probable archetype as known. However, this study does not validate the resulting building-energy predictions. Any claimed reduction in simulation error requires a separate test using buildings with measured energy and comparable weather, systems and operation.')
    p.para('For a regional or national stock model, the conditional sampling domain and weights matter. The present results do not cover households of six or more members, households with required missing education, or unknown dwelling categories in the probability stage. An operational implementation must state how it handles those cases and should not silently reuse the last available category. It should also preserve probability uncertainty instead of making deterministic assignments appear more certain than the evidence allows.')
    p.heading('5.1. Assumptions on internal heat gains',2)
    p.para('The source article compares a constant ISO-based example with an area-dependent EN-based example. Its constants, 1717 kWh/year and 1879 kWh/year at 120 m², are reproduced here as calculations reported by that article [1], not newly verified universal requirements of the standards. The public ISO scope supports the interpretation of schedules as informative examples [8]. The original derivations and appliance boundaries would have to be independently checked against the complete standards before treating either as a like-for-like benchmark.')
    p.para(('Figure 9 preserves horizontal electricity-group boxplots, but their equally spaced vertical positions now denote ordered groups, not floor area. The right-hand labels report each group\'s weighted mean area proxy; they do not define a continuous area axis. The EN illustration is evaluated at those means, and its connecting segments are a visual guide across groups, not a fitted area-response curve. ')+
           'A group mean is not the response of a dwelling with exactly that area. The 150+ category is especially broad, and its assigned proxy of 200 m² is not an observed mean.')
    p.figure('fig09_benchmarks','9',('Horizontal weighted held-out appliance-electricity distributions by training-defined group. The vertical ISO example is 1717 kWh/year; EN markers scale 1879 kWh/year at 120 m² to each group\'s mean area proxy (right labels). The vertical axis denotes groups, not area. ')+
              'Source-paper benchmarks are attributed illustrations, not current legal requirements or demonstrated prediction errors.',max_height=310)
    household_rows=load_csv(source/'households.csv')
    bench=[]
    for a in range(1,5):
        rows=[row for row in household_rows if int(row['archetype'])==a and row['split']=='test' and row['area_proxy']]
        rows=[row for row in rows if row['electricity']]
        w=sum(float(row['weight']) for row in rows)
        area=sum(float(row['weight'])*float(row['area_proxy']) for row in rows)/w
        mean=sum(float(row['weight'])*float(row['electricity']) for row in rows)/w
        en=1879/120*area
        bench.append([f'A{a}',str(len(rows)),fmt(area,1),fmt(mean,0) if mean is not None else 'Unavailable',
                      f'1717 ({100*(1717-mean)/mean:+.1f}%)' if mean is not None else '1717',
                      f'{en:.0f} ({100*(en-mean)/mean:+.1f}%)' if mean is not None else f'{en:.0f}'])
    p.table('Table 7','Illustrative benchmark comparison by archetype on held-out households with known area. Percentage differences use the modeled group mean as denominator; they are not validation errors. The 2021 values are conditional on the appliance-model assumptions.',
            ['Archetype','n','Area proxy mean (m²)','Modeled mean (kWh/y)','ISO example (difference)','EN example (difference)'],bench,[55,45,90,100,105,WIDTH-395])
    p.equation('Relative difference (%) = 100 × (reference value - group mean) / group mean')
    p.para('The denominator is stated because changing it changes the percentage. This definition also corrects the arithmetic inconsistency in the source article\'s E5 example: comparing 1717 with 2895 gives -40.69%, not -60%. The new table recalculates every entry from the new response summaries. More importantly, a numerical difference is not an energy-performance gap unless the same outcome, boundary, weather and operating conditions have been evaluated. No percentage in Table 7 is presented as a measured improvement of the proposed method over a standard.')
    p.heading('5.2. Assumptions on heating-system schedules',2)
    p.para('The original discussion compares reported heating hours with maximum daily durations associated with climatic zones. A maximum duration is not an estimate of typical use. The corrected Table 8 therefore treats the values 6, 8, 10, 12 and 14 hours as explicit schedule scenarios quoted from the source paper, not as current legal advice or household-level compliance classifications. The available survey geography does not identify each household\'s municipal climatic zone, so no region is assigned a single zone and no household is classified as exceeding a legal limit.')
    p.table('Table 8','Scenario duration relative to held-out heating-group mean: 100 x (scenario - mean) / mean. All entries are hypothetical comparisons, not geographically matched errors or compliance assessments.',
            ['Group','Test mean (h)']+[f'{i} h scenario' for i in [6,8,10,12,14]],
            [[row['group'],fmt(row['mean'],2)]+[f'{100*(h-float(row["mean"]))/float(row["mean"]):+.1f}%' for h in [6,8,10,12,14]] for row in hg],
            [45,80]+[(WIDTH-125)/5]*5)
    p.para('Figure 10 retains the geographic-map figure type but displays the statistic that can actually be estimated from the released geography: weighted regional heating hours among the eligible frequent users. The supplied 2024 regional boundaries are used only for cartography. The map is not a municipal climate-zone map, and it does not imply that a region is climatically uniform. Regional means can reflect the composition of households, buildings and systems as well as weather.')
    p.figure('fig10_regional_map','10',f'Weighted mean reported daily heating hours by region, {year} eligible frequent-use sample. Regional polygons come from the supplied ISTAT 2024 boundary file. '+('The heating responses refer to winter 2020. ' if year==2021 else '')+'This map replaces the unreconstructable municipal climate-zone layer; it is not a map of legal limits.',max_height=390)
    p.para('Operating duration alone is insufficient to calculate heat demand: setpoints, heated floor area, intermittency, envelope heat transfer, weather and system efficiency are also needed. The appropriate next simulation experiment would propagate both archetype-assignment and schedule uncertainty, compare resulting predictions with independent measured energy, and report whether the gain exceeds the uncertainty caused by those other inputs. The present analysis supplies candidate inputs for that experiment, not its outcome.')

    if year == 2021:
        equipment_section()
    p.heading('6. Conclusions')
    if year == 2021:
        p.para('The 2021 release supports explicit analyses of cooling ownership, summer operating duration and recent envelope improvements. Including the renovation indicator restores the corresponding heating-model tests and stratified distributions. Its interpretation remains recent investment activity rather than insulation quality or measured energy savings. Cooling-use estimates describe frequent users in summer 2020; extending them to annual demand requires additional modeling.')
    p.para(f'This corrected {year} analysis retains a four-archetype framework while replacing the source implementation\'s recursive education recoding, undocumented age distortions and hard-coded component expressions. The reference model is based on a documented eligible sample and a fitted weighted PCA. '+
           ('The resulting 2013 archetypes provide a reproducible baseline for later releases.' if year==2013 else 'The 2021 households are assigned through the unchanged 2013 reference, allowing the meaning of a label to remain fixed rather than being recreated independently.'))
    p.para(f'The dwelling model yields a valid four-category probability distribution by construction. Its test Brier score of {scores["quadratic"]["brier"]:.4f}, compared with {scores["training_marginal"]["brier"]:.4f} for the marginal baseline, provides a direct assessment of predictive value under the proxy assumptions. Neither high in-sample R² nor visual calibration alone is used to claim household-level accuracy. The secondary location/type model and sparse-cell diagnostics remain available for comparison, without a claim that all omitted variables are irrelevant.')
    p.para(f'The training-and-tuning process selects {heat["groups"]} heating groups. Their test prediction error is {heat["rmse"]:.2f} hours, while the constant training predictor has error {heat["baseline_rmse"]:.2f} hours. Group memberships are not repeatedly modified to obtain significant test differences. '+
           ('The modeled appliance-electricity analysis is reported with the explicit limitation that its response is generated by an external model rather than measured by meters.' if year==2013 else 'The 2021 heating distributions are stratified by recent insulation/window improvements. Appliance electricity is modeled under reference 2013 parameters and explicit assumptions; cooling operating hours and ownership provide separate observed results.'))
    p.para('Three limitations remain unresolved by code corrections alone. First, non-nested age bands and combined education categories prevent exact measurement equivalence between releases; the scenario appendix shows the resulting sensitivity. Second, the uncertainty calculations do not reconstruct the full survey design. Third, no independent building-simulation validation has been performed. The manuscripts therefore support a reproducible, explicitly conditional reanalysis, not a claim that all original conclusions have been confirmed or that the revised framework is ready for adoption as a standard.')
    if year==2021:
        p.para('Cross-year differences require additional restraint because the later heating answers refer to pandemic-era winter 2020 and the surveys differ in frame and mode. Even with a fixed archetype model and expansion weights, changes can combine demographic composition, measurement, selection and period effects. The paired release is useful for transfer testing and methodological comparison, but does not identify a causal temporal change in occupant behavior.')

    p.heading('Appendix A. Assignment of energy groups')
    p.para('Figure A.1 gives the complete mapping from the three common dwelling types, four geographic divisions and four archetypes to the selected training-defined energy groups. Each row is an input combination, not an observed household trajectory. A dash denotes insufficient training support and invokes the training-wide mean in predictive evaluation. The mapping must be used with the same fixed archetype definition and category crosswalk; replacing a category label without preserving its meaning invalidates the assignment.')
    p.figure('figA1_assignment','A.1','Energy-group assignment by location, common dwelling type and fixed-reference archetype. Group labels are selected on training/tuning data only. Dashes denote unsupported training cells.',max_height=420)
    p.table('Table A.1','Core crosswalk and availability checks.',
            ['Concept','2013 source','2021 source / correction'],[
                ['Household identifier','nprog','ID_RECORD; identifiers are not linked between years'],
                ['Expansion weight','coef_red','PESO'],
                ['Member age','cletad_1 ... cletad_5','eta_mfr1 ... eta_mfr5; non-nested age bands'],
                ['Member education','titstu_1 ... titstu_5','istr_mfr1 ... istr_mfr5; common four-category proxy'],
                ['Dwelling type','q_2_1','v1_1; apartment subtypes collapsed before comparison'],
                ['Construction / area','q_2_4_ric / q_2_7_class','v1_2_new / v1_7_mfr; four by six common cells'],
                ['Heating frequency / hours','q_3_10 / q_3_11A,B,C','v3_7 / v3_8a,b,c; restrict both frequencies to 1 or 2'],
                ['Cooling inventory','q_2_37A ... F','v2_11a ... e; routed absence distinguished from partial missingness'],
                ['Insulation rating','q_3_17','No equivalent rating in the released file'],
                ['Insulation/window improvements','q_3_18; q_3_19_7 or q_3_19_8','v3_13; v3_14gh_new; last five years, with different routing'],
                ['Cooling operating hours','q_5_7; q_5_8A,B,C','v5_6 codes 1-3; sum v5_7a,b,c; summer 2020'],
                ['Appliance electricity','results_grins.csv: appliances_el_kWh','Ownership/use model; reference 2013 parameters; assumptions audited']
            ],[98,164,WIDTH-262])
    p.heading('Appendix B. Sensitivity and reproducibility')
    p.para('Eight scenarios are evaluated: the midpoint reference; no/primary schooling values of zero and five years; an upper age cap of 90 years; a tertiary-schooling value of 16 years; and three seeded uniform draws within each released age interval. Every alternative demographic model is fitted only to the 2013 training subset and then applied unchanged to both years. Alternative labels are aligned to the reference by maximum weighted household agreement. The ranges describe choices about uncertain measurement, not confidence intervals and not a posterior distribution.')
    sensitivity=load_csv(root/'sensitivity.csv');ranges=[]
    for a in range(1,5):
        relevant=[row for row in sensitivity if int(row['year'])==year and int(row['archetype'])==a]
        values=[float(row['share']) for row in relevant];reference=next(float(row['share']) for row in relevant if row['scenario']=='reference')
        ranges.append([f'A{a}',fmt(reference,2),fmt(min(values),2),fmt(max(values),2)])
    p.table('Table B.1',f'Scenario sensitivity of {year} weighted household shares (%).',['Archetype','Reference','Scenario minimum','Scenario maximum'],ranges)
    p.figure('figB1_sensitivity','B.1',f'Input-assumption sensitivity: (a) {year} household shares; (b) changes between 2021 and 2013 under each common scenario. Dots mark the midpoint reference. Vertical spans are scenario ranges, not confidence intervals.',max_height=270)
    p.para('A complete uncertainty assessment would additionally vary the construction and floor-area bounds, the probability functional form, the household-size restriction and the missing-data model, and would repeat the entire analysis with valid survey-design resampling. Those analyses are not claimed to have been completed. The present scenarios identify a material limitation without implying that their envelope exhausts uncertainty. In particular, a small observed share change that is sensitive to age or education coding should not be described as an established temporal shift.')
    p.para('The executable workflow regenerates the scientific tables, vector SVG/PDF plots and both manuscripts from the raw releases and the existing 2013 appliance-energy input. It writes into a separate output directory and does not overwrite the legacy pipeline. The JSON configuration records paths, seeds, split proportions, proxy assumptions, support thresholds and penalty candidates. Input and output hashes, fitted transforms, full-precision coefficients, household splits and exclusion logs are retained. Figures use the same result files as the narrative and tables.')
    p.para('<b>Reproduction command:</b> <font name="Courier">.\\.venv\\Scripts\\python.exe run_revised_papers.py</font>','small')
    p.heading('Appendix C. Corrections relative to the source article')
    if year==2021:
        # Keep the assumption table in Appendix B before starting C.
        heading=p.story.pop();p.prose.pop()
        report=json.loads((root/'appliance_audit/electricity_model_report.json').read_text())
        p.table('Table B.2','Appliance-model assumptions. Alternatives are illustrative parameter scenarios, not confidence limits.',
                ['Input','Reference','Alternative scenarios'],[
                    ['Lighting hours','2013 weighted hours per bulb type','3 / 7 hours per day'],
                    ['Small-appliance background','2013 weighted mean','Zero / twice the reference mean'],
                    ['Fridge size / count factor','2013 weighted owner distribution','Small, factor 1 / large, factor 1.3'],
                    ['Infrequent cycles','18 per year','0 / 24 per year'],
                    ['Computer mix','Primary type applied to device count','All portable / all desktop'],
                    ['Power / efficiency','Verified legacy 2013 parameters','Held fixed; no 2021 efficiency update']
                ],[115,180,WIDTH-295])
        p.para(f'The reference weighted appliance estimate over all {report["2021_n"]:,} released households is {report["weighted_mean_kwh"]:.0f} kWh/year. Joint alternative scenarios give {report["weighted_lower_scenario_kwh"]:.0f} and {report["weighted_upper_scenario_kwh"]:.0f} kWh/year. These full-release descriptive means differ in domain from the eligible/test samples used above. The scenarios vary multiple assumptions together and are neither household prediction intervals nor a complete uncertainty distribution. The model matches the saved 2013 totals with maximum absolute discrepancy {report["2013_max_absolute_replication_error_kwh"]:.2g} kWh. This validates code reconstruction, not physical accuracy. Source model revision: {report["main_commit"][:12]}.')
        p.story.append(heading);p.prose.append('Appendix C. Corrections relative to the source article')
    corrections=[
        ['Education recoding','Recursive replacement could turn primary education into 12 years.','One immutable code-to-category map; explicit bounded schooling proxy.'],
        ['Education interpretation','A custom age-dependent ratio was attributed to the UNDP method.','No such equivalence claimed; formula and eligible members stated.'],
        ['Age conversion','Restricted draws misrepresented several released intervals.','Published bounds, midpoint proxies and stated sensitivity draws.'],
        ['PCA equations / variance','Rounded engineering expressions were not a fully specified fitted PCA; cumulative curves were inconsistent.','Fitted centered/scaled weighted PCA; orthonormality and 100% total-variance checks.'],
        ['Number of clusters','A visual elbow and general index thresholds were treated as stronger evidence than warranted.','Four clusters retained by design; diagnostic elbow shown without optimality claim.'],
        ['Regional inference','Household and centroid units were conflated; non-significance treated as invariance.','Household-level diagnostic MANOVA and separate descriptive centroid geometry.'],
        ['Probability model','Four independent polynomials could give negative probabilities or sums above one.','Multinomial softmax; consistent full-precision coefficients and simplex checks.'],
        ['Probability validation','Overlapping fitted proportions were interpreted as accurate prediction.','Non-overlapping cells for display; untouched household-level test scores.'],
        ['Variable selection','Iterative p-value filtering did not establish predictive irrelevance.','Prespecified quadratic and extended comparators; tuning isolated from testing.'],
        ['Energy groups','Adaptive Tukey merging did not retain a simple fixed-family testing guarantee.','Training group construction, separate complexity tuning, one test evaluation.'],
        ['Energy outcome','Modeled appliance kWh was sometimes described as expenditure or consumption measurement.','Response provenance and energy boundary stated consistently.'],
        ['Missing / new-wave variables','Available cooling and insulation-investment fields were incompletely used.','2021 cooling ownership/use and envelope improvements analyzed with routing; subjective insulation quality stays distinct from renovation. Appliance kWh reconstructed with explicit model assumptions.'],
        ['Figure 6 / Figure 9','Caption-axis inconsistency; arbitrary group positions on an area axis.','Correct captions; horizontal boxes on a group axis with actual mean-area annotations.'],
        ['Tables and narrative','Conflicting group counts, repeated examples, coefficients and percentage arithmetic.','Narrative and tables generated from one set of computed results.'],
        ['Standards and climate','Reference schedules treated as prediction errors; regional data used beyond its resolution.','Attributed illustrative scenarios; no compliance inference or municipal-zone reconstruction.'],
        ['Simulation benefit','Improved simulation accuracy asserted without direct validation.','Proposed use distinguished from demonstrated predictive/simulation performance.']
    ]
    p.table('Table C.1','Correction register. Methodological changes are explicit; unavailable evidence is not described as repaired data.',
            ['Topic','Problem','Correction / boundary'],corrections,[84,200,WIDTH-284])
    p.heading('Data and code availability')
    p.para('The analyses use the locally supplied ISTAT research microdata and their accompanying questionnaires, classification files and methodological notes [6,7]. Public information about access is available from ISTAT; the workflow does not redistribute household-level records or imply that research-access files can be published without checking their conditions. The source article\'s public code [12] and local history informed the correction register. The revised scripts and aggregate outputs are stored with this manuscript; all household-level outputs remain local.')
    p.heading('Authorship, declarations and acknowledgements')
    p.para('This working manuscript was prepared as a computational and methodological revision of the cited article at the requester\'s direction. It requires review and approval by the responsible authors before submission or dissemination as a scientific paper. The source article\'s authorship contributions, competing-interest declaration, funding statement and institutional endorsements are not automatically carried over to this new analysis. No new author approvals, conflicts declarations or funding claims are asserted here.')
    p.heading('References')
    references=[
        'Khajedehi, M.H., Vivian, J., and Zarrella, A. (2025). Identification of Italian user archetypes for building energy simulations based on a national survey. Energy &amp; Buildings 344, 115924. <link href="https://doi.org/10.1016/j.enbuild.2025.115924" color="#355d7d">doi:10.1016/j.enbuild.2025.115924</link>. Source article, distributed under CC BY; this manuscript is an adaptation with changed analyses and conclusions.',
        'De Wilde, P. (2014). The gap between predicted and measured energy performance of buildings: a framework for investigation. Automation in Construction 41, 40-49. doi:10.1016/j.autcon.2014.02.009. Background reference identified in the source article.',
        'Yan, D., O\'Brien, W., Hong, T., Feng, X., Gunay, H.B., Tahmasebi, F., et al. (2015). Occupant behavior modeling for building performance simulation: current state and future challenges. Energy &amp; Buildings 107, 264-278. doi:10.1016/j.enbuild.2015.08.032. Background reference identified in the source article.',
        'Dong, B., Liu, Y., Mu, W., Jiang, Z., Pandey, P., Hong, T., et al. (2022). A global building occupant behavior database. Scientific Data 9, 369. doi:10.1038/s41597-022-01475-3. Background reference identified in the source article.',
        'Rugani, R., Picco, M., Salvadori, G., Fantozzi, F., and Marengo, M. (2024). A numerical analysis of occupancy profile databases impact on dynamic energy simulation of buildings. Energy &amp; Buildings 310, 114114. doi:10.1016/j.enbuild.2024.114114. Background reference identified in the source article; its numerical effects are not imported as results here.',
        'ISTAT. Indagine sui consumi energetici delle famiglie, 2013 research release. Locally supplied microdata, questionnaire, classifications, record layout and methodological note. <link href="https://www.istat.it/microdati/indagine-sui-consumi-energetici-delle-famiglie/" color="#355d7d">Official research-file information</link>. Access information checked September 2026.',
        'ISTAT. Indagine sui consumi energetici delle famiglie, 2021 research release. Locally supplied microdata, questionnaire, classifications, record layout and methodological notes. <link href="https://www.istat.it/comunicato-stampa/consumi-energetici-famiglie-anni-2020-2021/" color="#355d7d">Consumi energetici delle famiglie - Anni 2020-2021</link> (21 December 2022).',
        'International Organization for Standardization (2018). ISO 18523-2:2018. Energy performance of buildings - Schedule and condition of building, zone and space usage for energy calculation - Part 2: Residential buildings. <link href="https://www.iso.org/standard/69633.html" color="#355d7d">Official scope and publication information</link>. The public scope was checked; full normative tables were not independently rederived.',
        'European Committee for Standardization (2019). EN 16798-1:2019. Energy performance of buildings - Ventilation for buildings - Part 1: Indoor environmental input parameters. Bibliographic and illustrative benchmark information reproduced from the source article [1]; no assertion about the current edition or universal appliance-energy requirements.',
        'statsmodels documentation. Linear-model inference and heteroscedasticity-robust covariance; <link href="https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html" color="#355d7d">ANOVA documentation</link>. The implementation uses prespecified WLS coefficient-block Wald tests with HC3 covariance, not an unqualified survey-design ANOVA.',
        'statsmodels documentation. <link href="https://www.statsmodels.org/stable/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html" color="#355d7d">pairwise_tukeyhsd</link>. Used for a single classical test family on fixed held-out groups; weighted robust contrasts are exported separately.',
        'BETALAB-team. ita-domes, UserArchetypes source code, inspected at commit 1b5972b1e052f00a2fdd9fa30d46bb442205c83d. <link href="https://github.com/BETALAB-team/ita-domes/tree/1b5972b1e052f00a2fdd9fa30d46bb442205c83d/resources/istat_data/UserClusters/istat_data" color="#355d7d">Archived code cited by the source article</link>. This is evidence about available implementation files, not proof of the exact unpublished execution used for every published result.'
    ]
    for i,text in enumerate(references,1):p.para(f'[{i}] {text}','small')
    p.build()


def main(config, years=(2013,2021)):
    register_fonts();cfg=json.loads(Path(config).read_text())
    for year in years:manuscript(year,ROOT/cfg['output_root'],ROOT/cfg['pdf_directory'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'revised_paper_config.json')
    parser.add_argument('--year',type=int,choices=[2013,2021])
    args=parser.parse_args()
    main(args.config, (args.year,) if args.year else (2013,2021))
