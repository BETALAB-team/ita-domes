"""Generate a versioned literature revision; keep all empirical results unchanged."""
from pathlib import Path
import re
ROOT=Path(__file__).parent
original=(ROOT/'first_draft.md').read_text(encoding='utf-8')
text=original
def section(start,end,replacement):
    global text
    assert text.count(start)==1 and text.count(end)==1
    a,b=text.index(start),text.index(end)
    text=text[:a]+replacement.strip()+'\n\n'+text[b:]

section('### 1 3','### 1 4',"""
### 1 3 Occupant representation, temporal change and technology assessment

The relevant literature addresses three connected but distinct questions: how household diversity should enter a building model, how residential demand changes as populations and equipment evolve, and how uncertainty in these inputs affects technology assessment. These questions require different evidence. Survey associations describe populations; monitored schedules describe time-dependent operation; physical models connect operation to energy requirements. Their integration does not remove the need to establish which quantities have actually been observed.

Chen et al. [18] distinguish occupancy, interactions with building systems and behavioral efficiency in their review of occupant behavior. This distinction is useful here because household age and size are not direct observations of system control or energy-saving preferences. Happle et al. [23] distinguish deterministic space-based, stochastic space-based and stochastic person-based approaches in urban building energy modelling. Their review connects model choice to the intended application. Together, these contributions support a representation that preserves the difference between household composition, equipment availability and conditional use, rather than treating all survey variables as interchangeable measures of behavior.

Empirical schedules provide a complementary route. Ferrando et al. [25] derive occupancy and electricity-use schedules from smart-meter information and examine their randomized assignment in an urban model. Their assessment shows that the consequences depend on spatial and temporal aggregation. Chen et al. [8] incorporate stochastic occupant-driven energy use into a bottom-up residential stock model, while Li and Dogan [7] connect survey-derived archetypes and usage patterns to reduced-order simulation. These studies establish that data-informed occupant representation is not itself a new contribution. They also expose two different meanings of time: variation within a day or year, and changes in the population represented by a model between survey waves. The present study addresses the latter without claiming to observe the former at hourly resolution.

Population change has also received explicit attention. Ramos-Martín and Matsumoto [9,10] examine household composition in Spanish residential energy analysis and projections. Their work motivates separating changes in household-type prevalence from conditional demand differences. Latifi et al. [11] address another dimension through probabilistic integration of demographic, building, time-use and weather information. Cross-source integration and cross-wave comparability are related but not identical problems: a plausible synthetic population may still embody relationships estimated for an earlier period. Conversely, a later survey may offer a more current population description while measuring particular appliances less precisely.

Italian technology-assessment studies show why these distinctions matter beyond classification. Teso et al. [19] use CityBES to evaluate envelope and heating-system renovation measures in Venice's Santa Marta social-housing district. This establishes a practical connection between stock representation and locally feasible interventions. Marigo et al. [20] compare survey-based, building-simulation and appliance-stock approaches to residential biomass estimation in the Po Valley. Their triangulation provides a useful model for checking aggregate energy estimates against evidence assembled through different routes. Neither regional specificity nor the combination of surveys with physical calculations is therefore sufficient, alone, to establish the novelty of the present extension.

More recent technology assessment also considers uncertainty explicitly. Bringiotti et al. [21] combine building archetypes, district electrification scenarios and probabilistic techno-economic assessment. Their work is a particularly relevant comparator because it demonstrates that archetype-based technology evaluation under uncertainty is already established. Our proposed contrast concerns a complementary uncertainty: whether household inputs estimated from an earlier population remain adequate for a later one. Aneli et al. [22], examining solar-assisted heat pumps and storage, further illustrate the importance of matching demand and generation in time. An annual appliance indicator can inform an energy balance but cannot, by itself, establish self-consumption or storage performance.

The resulting contribution is deliberately specific. Building on the published Italian household archetypes [1] and their stock-modelling application [2], we evaluate the empirical value of retaining, recalibrating or replacing the household representation using a later independent survey. The framework separates population composition, conditional outcomes and changes in measurement detail, and then connects the alternative input treatments to paired regional simulations. This is a repeated-cross-sectional temporal extension, not a household panel and not a claim to introduce occupant-aware modelling. Its substantive novelty depends on demonstrating where updating changes model performance or an energy-assessment conclusion; the unfinished transfer and simulation experiments are therefore essential to the final article.
""")

section('### 5 1','### 5 2',"""
### 5 1 What an archetype should retain over time

The available 2021 results support a differentiated household representation: demographic composition, cooling ownership and reported improvements vary across archetypes, while dwelling characteristics contain information about assignment. These findings justify testing temporal transfer, but do not establish its success. The central distinction is between heterogeneity within a population and the stability of a representation across populations. Stochastic schedule models address the former [8,25]; the frozen, recalibrated and newly fitted comparisons proposed here address the latter.

Temporal adequacy is application-dependent. A classification can remain demographically recognizable while its prevalence changes, or retain similar shares while equipment and use change within classes. A model that is adequate for an annual regional total may not be adequate for an end-use distribution or a targeted intervention. The aggregation dependence examined by Ferrando et al. [25] reinforces the need to evaluate the output required by the application, rather than select a representation solely from its classification score. Their schedule experiment does not establish the size or direction of a cross-wave effect in Italy.

This leads to a continuity-preserving update strategy. Re-estimating every component can obscure the relationship to the original model; retaining every component can conceal calibration drift. The established 2013 representation supplies the reference vocabulary and empirical foundation for deciding what requires updating. Limited improvement from recalibration would be a substantive finding if demonstrated on common support with appropriate uncertainty. Concentrated deterioration in particular household or dwelling groups would instead justify a targeted update. In either case, the scientific question is the value of additional information for a defined modelling task, not whether the later dataset should automatically supersede the earlier study.
""")

section('### 5 4','### 5 5',"""
### 5 4 Interpreting insulation and heating hours

The heating-group comparisons provide no consistent multiplicity-adjusted association between reported improvement and heating duration. This is a finding about the available indicators, not a test of the engineering effectiveness of insulation. The improvement field combines intervention types and does not quantify achieved thermal performance; heating hours describe operation rather than delivered energy. The distinction between occupancy, system interaction and efficiency in Chen et al. [18] is particularly important when interpreting this contrast.

Measured retrofit evidence illustrates why comfort and energy outcomes need separate treatment. Colclough et al. [24] report a post-occupancy evaluation of 12 retrofitted Irish social dwellings, documenting high satisfaction alongside a discrepancy between predicted and actual energy performance and higher-than-assumed indoor temperatures. This is context-specific evidence, not an effect estimate transferable to Italian households. It nevertheless supports examining temperatures and system operation before interpreting heating duration as an energy-saving measure. Our cross-sectional comparison cannot identify comfort-taking or a rebound effect.

Excluding unknown responses gives the requested known-yes versus known-no comparison, but does not eliminate selection uncertainty. The excluded share and observable characteristics should be reported for each wave. These exclusions remain specific to this analysis: they should not alter unrelated appliance or demographic populations. Nor does a negative response about a reported intervention establish that a dwelling is physically uninsulated.

For regional assessment, documented envelope changes can be simulated under common operating assumptions, with alternative operational responses treated explicitly as scenarios. The Venice renovation assessment [19] provides an Italian example of evaluating physical measures within a defined stock and local context; it does not supply a behavioral response coefficient for our survey. Separating observed responses, assumed operation and simulated performance makes this extension compatible with the earlier archetype study while avoiding causal claims that neither survey wave can support.
""")

section('### 5 5','### 5 6',"""
### 5 5 Appliance electricity and the value of information

Reconstructed electricity occupies a different evidential category from metered demand. Its numerical precision can conceal dependence on reference powers, average capacities and imputed durations. The 2021 sensitivity range therefore matters before any cross-wave interpretation. Reproducing the saved 2013 calculation establishes implementation continuity, but is not an independent validation of household electricity use. The paired coarsening experiment asks a narrower question: how much information is lost when the same households and coefficients are represented using reduced survey detail?

This distinction also limits the technological claims that follow from the indicator. Aneli et al. [22] analyse heat pumps with photovoltaic generation and storage, for which the temporal match between demand and supply matters. The present annual reconstruction does not recover that match. Any later use in self-consumption, storage sizing or flexibility assessment would require time-resolved evidence and validation beyond agreement with an annual total. Higher detail in an appliance questionnaire is not equivalent to observing its operating schedule.

The coarsening test can nevertheless inform data priorities. An input may have little influence on the national mean but matter for a high-consumption subgroup or an appliance-specific intervention. Its value should therefore be evaluated against the intended output, including archetype-level distributions. A fixed-coefficient indicator isolates variation associated with reported ownership and use under a common conversion convention. A historical-efficiency analysis asks a different question about technology and operation together. Keeping these tracks separate preserves the published 2013 reference while making explicit what the new comparison can attribute to changing information.
""")

section('### 5 6','### 5 7',"""
### 5 6 Relevance for regional model maintenance

The regional application should determine whether updating household information changes a useful energy-assessment conclusion. National averages can conceal geographically concentrated changes or differences within the stock eligible for a measure. Regional totals and per-dwelling quantities are consequently both needed: the former represent system scale, the latter help distinguish scale from intensity. The ita-domes framework [2] provides the physical and regional structure for a controlled comparison, while earlier Italian electrification work [4] establishes the relevance of demographic conditions.

Two sources of uncertainty should remain distinguishable. Bringiotti et al. [21] assess district electrification probabilistically with techno-economic inputs, including costs and energy prices. The present extension asks whether the household representation itself should be refreshed before such scenarios are evaluated. These uncertainties are complementary, not competing explanations. A carefully specified financial model can still rely on outdated household inputs, while a current survey cannot resolve uncertainty about future prices. The paired simulations proposed here isolate input-vintage effects by holding other model treatments constant; they do not constitute a comprehensive investment-risk assessment.

Validation should likewise proceed at more than one level. The multiple estimation routes compared by Marigo et al. [20] motivate checking carrier-specific regional totals using evidence with distinct construction methods. Agreement at that level would not verify the archetype assignments or conditional operation that produced the totals. In the present application, survey-distribution checks, predictive calibration and external energy comparisons should therefore be reported separately. Shared source data must be disclosed because two estimates are not fully independent merely because different models produce them.

A selective maintenance policy could update prevalence and equipment ownership while retaining a dwelling-assignment model that remains calibrated. Whether this is sufficient depends on the pending transfer results. Similarly, an apparent retrofit rank reversal is decision-relevant only if its magnitude is meaningful relative to uncertainty and eligibility is held constant. Near-ties should not become categorical recommendations. The article can assess annual energy consequences of updating the household representation; economic optimality, distributional fairness and hourly network impacts require additional objectives and evidence.
""")

text+='''
18. Chen S, Zhang G, Xia X, Chen Y, Setunge S, Shi L. The impacts of occupant behavior on building energy consumption: A review. Sustainable Energy Technologies and Assessments. 2021;45:101212. https://doi.org/10.1016/j.seta.2021.101212

19. Teso L, Carnieletto L, Sun K, Zhang W, Gasparella A, Romagnoni P, Zarrella A, Hong T. Large scale energy analysis and renovation strategies for social housing in the historic city of Venice. Sustainable Energy Technologies and Assessments. 2022;52:102041. https://doi.org/10.1016/j.seta.2022.102041

20. Marigo M, Zulli F, Pillon S, Susanetti L, De Carli M. Heating energy balance and biomass consumption for the residential sector in the Po Valley. Sustainable Energy Technologies and Assessments. 2022;54:102814. https://doi.org/10.1016/j.seta.2022.102814

21. Bringiotti A, Pallonetto F, Bianco V, De Rosa M. Technical and economic strategies for smart electrification of urban districts: A probabilistic modelling framework. Sustainable Energy Technologies and Assessments. 2026;92:105098. https://doi.org/10.1016/j.seta.2026.105098

22. Aneli S, Arena R, Tina GM, Gagliano A. Improvement of energy self-sufficiency in residential buildings by using solar-assisted heat pumps and thermal and electrical storage. Sustainable Energy Technologies and Assessments. 2023;60:103446. https://doi.org/10.1016/j.seta.2023.103446

23. Happle G, Fonseca JA, Schlueter A. A review on occupant behavior in urban building energy models. Energy and Buildings. 2018;174:276-292. https://doi.org/10.1016/j.enbuild.2018.06.030

24. Colclough S, O. Hegarty R, Murray M, Lennon D, Rieux E, Colclough M, Kinnane O. Post occupancy evaluation of 12 retrofit nZEB dwellings: The impact of occupants and high in-use interior temperatures on the predictive accuracy of the nZEB energy standard. Energy and Buildings. 2022;254:111563. https://doi.org/10.1016/j.enbuild.2021.111563

25. Ferrando M, Ferroni S, Pelle M, Tatti A, Erba S, Shi X, Causone F. UBEM's archetypes improvement via data-driven occupant-related schedules randomly distributed and their impact assessment. Sustainable Cities and Society. 2022;87:104164. https://doi.org/10.1016/j.scs.2022.104164
'''
# The revision adds eight references; retain original empirical sections verbatim.
for start,end in [('## 2 Data','## 5 Discussion'),('## 6 Conclusions','## References')]:
    assert text[text.index(start):text.index(end)]==original[original.index(start):original.index(end)]
assert re.findall(r'^!.*$',text,re.M)==re.findall(r'^!.*$',original,re.M)
(ROOT/'literature_revised_draft.md').write_text(text,encoding='utf-8')
print('Created literature_revised_draft.md; empirical sections and figures unchanged.')
