"""Add claim-specific sources to the literature revision without changing results."""
from pathlib import Path
import re
ROOT=Path(__file__).parent
source=(ROOT/'literature_revised_draft.md').read_text(encoding='utf-8')
text=source
def replace(old,new):
    global text
    assert text.count(old)==1, old[:80]
    text=text.replace(old,new)
def before(heading,paragraph):
    replace(heading,paragraph+'\n\n'+heading)

replace('These differences matter when models are used to compare interventions across a region, because an intervention acts on a building that is occupied and operated rather than on an envelope in isolation.','These differences matter in regional intervention assessment. The urban modelling literature treats input organization, thermal calculation and validation as connected tasks [26,27]; reviews of multi-domain tools also emphasize the interaction between demand, generation and distribution [28]. Household inputs must therefore be assessed in relation to the modelling chain in which they are used.')
before('### 1 2 Foundations', 'Evidence about household determinants also cautions against treating demographic variables as direct behavioral explanations. Huebner et al. [30] examine building, demographic, attitudinal and self-reported behavioral predictors in English households, highlighting overlapping explanatory information. For our application, a demographic archetype is a useful assignment device, not a substitute for dwelling characteristics or an independently identified mechanism of consumption.')
before('### 1 3 Occupant', 'Besagni and Borgarello [33] provide a further Italian foundation by relating expenditure to dwelling, demographic and appliance characteristics and developing household segmentation. Their expenditure outcome should remain distinct from the electricity reconstructed here: monetary spending combines quantities with prices and billing conditions. This supports continuity with Italian segmentation research while making the cross-wave measurement target explicit.')
replace('The three', 'The three') if text.count('The three')==1 else None
before('### 1 4 Objectives', 'The broader predictive literature describes changing input-target relationships as concept drift [39]. That terminology helps motivate out-of-period evaluation, but the present design is not an online drift detector: two independent surveys cannot reveal the timing of change between waves. Its contribution is an application-specific test of temporal adequacy under documented measurement constraints.')
replace('It is not a complete design effect because the full sampling design may introduce additional dependence.','It is not a complete design effect because the full sampling design may introduce additional dependence. Design-aware variance estimation requires more than the weights alone, as discussed by Lumley [38].')
replace('Weighted log loss and the multiclass Brier score summarize predictive performance, while calibration examines agreement between predicted probabilities and observed frequencies.','Weighted log loss and the multiclass Brier score summarize predictive performance using proper scoring rules [35], while calibration examines agreement between predicted probabilities and observed frequencies.')
replace('The planned transfer experiment distinguishes three strategies.','The planned transfer experiment distinguishes three strategies, motivated by the need to evaluate prediction under changing data relationships [39].')
replace('heteroskedasticity-consistent HC3 covariance.','heteroskedasticity-consistent HC3 covariance [37].')
replace('Holm adjustment is applied to control multiplicity within that family.','Holm adjustment [36] is applied to control multiplicity within that family. This controls the testing family subject to the validity of the underlying p-values; it does not remedy omitted survey-design dependence.')
before('### 3 7 Heating groups', 'High-resolution appliance models provide a methodological contrast rather than replacement coefficients. Richardson et al. [31] combine active occupancy and activities with simulated appliance use; Widén and Wäckelgård [32] generate activity and demand sequences from time-use information. Their temporal construction requires inputs absent from our annual reconstruction. These sources justify keeping annual energy comparability and hourly profile realism as separate validation questions.')
replace('Uncertainty arises from survey sampling, proxy construction, missing-detail assumptions, conditional assignment and physical model parameters.','Uncertainty arises from survey sampling, proxy construction, missing-detail assumptions, conditional assignment and physical model parameters. Tian et al. [29] distinguish propagation of uncertain inputs from inverse inference of model inputs using energy observations. The present sensitivity scenarios belong to the former category; they are not a calibration against metered household demand.')
replace('None of these checks automatically substitutes for the others.','None of these checks automatically substitutes for the others, consistent with the separation of data, simulation and validation in urban modelling reviews [26,27].')
before('### 5 5 Appliance electricity', 'The distinction is also consistent with broader performance-gap evidence. Majcen et al. [34] compare theoretical and actual consumption in Dutch dwellings and find discrepancies that differ across energy-label categories. Sunikka-Blank and Galvin [40] describe the prebound effect, in which actual consumption is below calculated demand before an intervention. Such evidence cautions against assuming that theoretical baseline energy is fully realized. Neither mechanism can be inferred from our improvement indicator and reported hours alone; measured energy and comfort conditions would be needed to investigate it.')
replace('The aggregation dependence examined by Ferrando et al. [25] reinforces','The aggregation dependence examined by Ferrando et al. [25], alongside application-oriented urban modelling reviews [26,27], reinforces')
replace('Any later use in self-consumption, storage sizing or flexibility assessment would require time-resolved evidence and validation beyond agreement with an annual total.','Any later use in self-consumption, storage sizing or flexibility assessment would require time-resolved evidence and validation beyond agreement with an annual total, including an explicit activity-to-demand model [31,32] and its connection to the wider energy system [28].')
replace('These uncertainties are complementary, not competing explanations.','These uncertainties are complementary, not competing explanations; the distinction follows the broader separation of uncertain inputs and their propagation in building-energy assessment [29].')
text+='''
26. Reinhart CF, Cerezo Davila C. Urban building energy modeling - A review of a nascent field. Building and Environment. 2016;97:196-202. https://doi.org/10.1016/j.buildenv.2015.12.001

27. Hong T, Chen Y, Luo X, Luo N, Lee SH. Ten questions on urban building energy modeling. Building and Environment. 2020;168:106508. https://doi.org/10.1016/j.buildenv.2019.106508

28. Sola A, Corchero C, Salom J, Sanmarti M. Multi-domain urban-scale energy modelling tools: A review. Sustainable Cities and Society. 2020;54:101872. https://doi.org/10.1016/j.scs.2019.101872

29. Tian W, Heo Y, de Wilde P, Li Z, Yan D, Park CS, Feng X, Augenbroe G. A review of uncertainty analysis in building energy assessment. Renewable and Sustainable Energy Reviews. 2018;93:285-301. https://doi.org/10.1016/j.rser.2018.05.029

30. Huebner GM, Hamilton I, Chalabi Z, Shipworth D, Oreszczyn T. Explaining domestic energy consumption - The comparative contribution of building factors, socio-demographics, behaviours and attitudes. Applied Energy. 2015;159:589-600. https://doi.org/10.1016/j.apenergy.2015.09.028

31. Richardson I, Thomson M, Infield D, Clifford C. Domestic electricity use: A high-resolution energy demand model. Energy and Buildings. 2010;42(10):1878-1887. https://doi.org/10.1016/j.enbuild.2010.05.023

32. Widén J, Wäckelgård E. A high-resolution stochastic model of domestic activity patterns and electricity demand. Applied Energy. 2010;87(6):1880-1892. https://doi.org/10.1016/j.apenergy.2009.11.006

33. Besagni G, Borgarello M. The determinants of residential energy expenditure in Italy. Energy. 2018;165:369-386. https://doi.org/10.1016/j.energy.2018.09.108

34. Majcen D, Itard LCM, Visscher H. Theoretical vs. actual energy consumption of labelled dwellings in the Netherlands: Discrepancies and policy implications. Energy Policy. 2013;54:125-136. https://doi.org/10.1016/j.enpol.2012.11.008

35. Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association. 2007;102(477):359-378. https://doi.org/10.1198/016214506000001437

36. Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics. 1979;6(2):65-70. https://www.jstor.org/stable/4615733

37. MacKinnon JG, White H. Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics. 1985;29(3):305-325. https://doi.org/10.1016/0304-4076(85)90158-7

38. Lumley T. Analysis of complex survey samples. Journal of Statistical Software. 2004;9(8):1-19. https://doi.org/10.18637/jss.v009.i08

39. Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A. A survey on concept drift adaptation. ACM Computing Surveys. 2014;46(4):44. https://doi.org/10.1145/2523813

40. Sunikka-Blank M, Galvin R. Introducing the prebound effect: the gap between performance and actual energy consumption. Building Research & Information. 2012;40(3):260-273. https://doi.org/10.1080/09613218.2012.690952
'''
assert source[source.index('## 4 Results'):source.index('## 5 Discussion')]==text[text.index('## 4 Results'):text.index('## 5 Discussion')]
assert re.findall(r'^\|.*$',source,re.M)==re.findall(r'^\|.*$',text,re.M)
body=text.split('## References')[0]
cited=set(int(n) for group in re.findall(r'\[([\d,]+)\]',body) for n in group.split(','))
assert cited==set(range(1,41)), sorted(set(range(1,41))-cited)
(ROOT/'literature_revised_draft.md').write_text(text,encoding='utf-8')
print('40 references, all cited; empirical results and all tables unchanged.')
