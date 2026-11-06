# Conference evidence: target chapter only

Owner: context_evidence. Access: 20 October 2026. Requested worker model: gpt-6.1-sol; confirmed backend model metadata is not exposed to this worker and remains unknown. This is an evidence record, not manuscript prose.

## Source and access

**CONF-001.** Vivian, J.; Prataviera, E.; Carnieletto, L.; Pasut, W.; Zarrella, A. *A model to Simulate the Energy Consumption of the Italian Residential Building Stock with Regional Resolution*. In CLIMA 2025, C. Zilio et al. (eds.), Lecture Notes in Civil Engineering 788, pp.1302–1311. Published **2026**. DOI: https://doi.org/10.1007/978-3-032-10546-2_123. Bibliographic facts verified directly from chapter first page; full proceedings title/publisher landing-page metadata requires the literature worker's external verification.

Local source: `C:\Users\khajmoh18975\Downloads\Proceedings of the 15th REHVA HVAC World Congress - CLIMA 2025.pdf`. SHA256 `51ce8fb615c15cf34087682eafbc7beb4267dab4e20a78bb49a844ccffa3450b`. Local PDF has 13 pages; only PDF pages **2–11**, printed **1302–1311**, were extracted and read. All ten rendered page images were visually inspected, including Tables 1–3 and Figures 1–2. Adjacent-paper fragments on PDF1 and PDF12–13 were excluded. Artifacts: `conference_complete_text.txt`, `conference_page_02.png` through `conference_page_11.png`, `conference_pdf_metadata.json`. Rendering used bundled pypdfium2 after fitz was unavailable; text extraction used pypdf. No PDF was edited or authored. PDF metadata creation timestamp is a file-production timestamp, not the publication year.

## Page-located evidence ledger

| ID | Locator: printed (PDF) | Established content | Evidence class and permitted inference |
|---|---|---|---|
| CONF-M01 |1302–1303 (2–3), abstract/introduction|ita-domes is a bottom-up Italian residential-stock model using ISTAT household survey data; predecessors MOIRAE and MODENA are acknowledged.|Published method description; survey-informed Italian stock modelling is already established.|
| CONF-D01 |1303 (3), Section2|The paper describes landline households, geographic/demographic stratification, base sample20,000 and reserve100,000, CATI interviews March–July2013, >380 questions in9 sections.|Published survey description, not independently verified survey-method metadata. Interview timing and energy-reference period must be rechecked against official documentation before reuse.|
| CONF-M02 |1304 (4), Section3.1|EUReCA lumped-capacitance calculations, specifically one-capacitance model described as proposed by ISO13790; envelope inputs selected using survey responses;60 construction archetypes linked to reference12.|Published configuration description. No release/commit/environment or executable full run is specified. Does not identify the present local checkout as the published version.|
| CONF-M03 |1304 (4), Table1/text|Dwelling type, vintage, wall type/insulation, frames, exposure, position and area inform geometry/envelope. Area classes use rounded class means;40–59m² becomes50m². Window area=number×1.7m²; square floor plan assumed.|Explicit geometry assumptions. The terminal area category is printed as >120m² after120–149m²: overlapping published label, not a verified code rule. Do not silently repair it.|
| CONF-M04 |1304 (4), final paragraph|Hourly heating setpoints and occupant internal gains are assumed from coarse respondent information. Heating hours are reported for morning/afternoon/evening; season dates use climate zone and cited law.|Survey response plus schedule reconstruction. Coarse hours do not identify the exact hourly placement, occupant presence, thermal demand, or meter series.|
| CONF-M05 |1305 (5), opening|Two cooling calculations: thermal demand at usually26°C/50%RH with outdoor-temperature-related EER; fixed power×daily hours mapped from frequency:6,4,3,0.75,0.5h/day.|Alternative model rules, not observed cooling electricity. Paper does not fully specify which cooling method feeds each reported output or a comparison between methods.|
| CONF-M06 |1305 (5), Section3.1|Emission/distribution/control/generation efficiencies assumed from emitter/layout/age; heat-pump efficiency dynamic via Staffell correlation.|Model assumptions, not measured system performance.|
| CONF-M07 |1305–1306 (5–6), Table2/Section3.2|Appliance average powers/performance from market analysis combined with ownership/use. Lights use15W efficient/40W traditional and3/7/14h daily patterns. Cooking and DHW include electricity/gas/biomass/LPG. DHW uses UNI/TS11300–2.|Derived end-use estimates. Appliance kWh must never be called measured electricity; operating-day and aggregation choices still require source code.|
| CONF-M08 |1306 (6), cooking example|100h×1.40kW=140kWh thermal; /0.40=350kWh gas final energy (~33Smc). Electric:100h×1.5kW=150kWh thermal; /0.74≈202.70kWh final electricity, printed203.|Illustrative arithmetic and an explicit thermal/final-energy distinction. These are examples, not simulation results or appliance meter validation. No numbered equations occur in the chapter.|
| CONF-W01 |1306 (6), Section3.3|Baseline CTI typical meteorological years represent110 provincial stations;2013/2014 weather overwrites cover30 stations/provinces, approximately27% of households.|Partial historical-weather coverage, not nationwide matched historical weather. Province breakdown:Emilia-Romagna8;Friuli-Venezia Giulia4;Lazio5;Milan1;Piemonte4;Trento1;Veneto7.|
| CONF-V01 |1306–1307 (6–7), Sections3.3/4.1 and Table3|National annual final-energy totals compared with National Energy Balance for6 carriers in2013/2014.|Aggregate external comparison; no household, hourly, city, regional-meter or climate-transfer accuracy demonstrated. Reference data provenance is described as NEB/MISE, but no complete extraction/weighting pipeline is published here.|
| CONF-R01 |1307–1308 (7–8), Section4.2/Figure1|Regional annual totals for heating/cooling/appliances/cooking; maps use GWh and separate panel colour scales.|Published modelled regional totals; not per-household intensities, observed regional consumption or uncertainty intervals.|
| CONF-R02 |1308–1309 (8–9), Section4.3/Figure2|Regional annual modelled carrier totals:GWh electricity/natural gas/biomass/LPG.|No gas/LPG/district-heat validation is provided at regional scale. Do not digitize map colour into precise new results.|
| CONF-I01 |1307–1310 (7–10)|Authors associate discrepancies with weather coverage, DHW/heating assumptions and sparse district-heating survey data; maps interpreted using population,HDD,device diffusion and resources.|Plausible published interpretations, not isolated causal effects; no factorial attribution or measured causal test.|
| CONF-A01 |1310 (10), acknowledgements/data availability|GRINS funding acknowledged. Model link: https://github.com/BETALAB-team/ita-domes/ . Results stated available on request or AMELIA:https://ameliadp.grins.it/ .|Repository/platform availability is an author statement; chapter does not specify repository version or result artifact checksums. This does not identify every GRINS-labelled local file as conference output.|
| CONF-F01 |1302 (2), abstract|Updating with2021/2024 surveys stated as future work.|Temporal updating is an anticipated extension; it is not a completed result in this chapter.|

## Table3: published annual final-energy comparison

All values below are **historical published values**, not newly reproduced ita-domes simulations. Units:ktoe. Percentage columns transcribed as printed; signed deviations recalculated as100×(model−NEB)/NEB for checking, with rounding differences expected.

| Carrier |2013NEB|2013model|2013printed deviation|2014NEB|2014model|2014printed deviation|
|---|---:|---:|---:|---:|---:|---:|
|Electricity|5760|5759|0.0%|5525|5752|4.1%|
|Natural gas|18073|18090|+0.1%|15151|17115|13%|
|LPG|1192|1552|+30%|1080|1472|+36%|
|Biomass (wood+pellets)|6633|6265|−5.5%|5676|6036|+6.4%|
|Gasoline (paper's label)|1506|682|−55%|1158|640|−45%|
|District heating|931|86|−91%|818|76|−91%|

Independent arithmetic check:2013 electricity−0.0174%,gas+0.0941%;2014 electricity+4.1086%,gas+12.9628%.NEB gas falls16.1678%;model gas falls5.3897%. These calculations only check printed-table arithmetic. The paper's carrier label “gasoline” should be preserved until code/NEB taxonomy reconciles possible liquid-heating-fuel terminology. It must not be interpreted as road-transport consumption.

## Figures: visual verification and text-reported shares

Figure1, printed1308/PDF8, is a four-panel Italian regional choropleth with distinct colour bars labelled GWh. Heating:Lombardy22%,Piedmont13%,Emilia-Romagna10%,Veneto10%. Cooling:Lombardy/Sicily≈14%each,Veneto12%,Emilia-Romagna11%,Lazio10%. The text reports appliances/cooking shares around17%Lombardy,10%Lazio,9%Veneto,8%Campania/Emilia-Romagna. These are national-total shares, not per-household values or contribution to peaks. DHW is not its own Figure1 panel.

Figure2, printed1309/PDF9, has electricity/natural-gas/biomass/LPG panels, GWh, separate scales. Text gives electricity17%Lombardy,10%Lazio,8.5%Sicily;natural gas28%Lombardy,12%Emilia-Romagna,11%Piedmont;biomass17%Piedmont,11%Veneto,8%Lombardy. The maps contain no numeric regional table, error bars, uncertainty maps, or measured comparator. LPG panel is shown but no exact text percentages are supplied. Geographic distributions cannot alone substantiate causal explanations.

## Boundaries for the extension

The chapter motivates efficiency/renewable policy support but presents **no completed retrofit scenario comparison**, savings/ranking validation, full uncertainty propagation, computational timing benchmark, or empirical household/city/hourly test. Its national comparison cannot support those claims. A meaningful extension can examine how coarse household responses are mapped into hourly simulations and whether representation choices change controlled climate responses, while explicitly separating implementation verification, benchmark checks, aggregate comparisons and empirical validation. Whether that question is novel requires the separate current primary-literature audit. Generic survey-informed EUReCA/Italian climate simulation is not sufficient novelty.

A stronger2013–2021 temporal update remains feasible only after official crosswalk/reference-period/domain verification and paired common-information experiments. Local draft material does not convert its pending experiments into evidence. All sources in the reference list remain leads until independently inspected; reference12(Carnieletto2021) is a physical-prototype source and is distinct from the later household-archetype paper.
