# Independent verification of the selected 2013 heating-hours response

Checked 20 October 2026 by `context_evidence`. Requested model: `gpt-6.1-sol`; confirmed backend metadata remains unknown. Scope is the selected simulation input only. This record does not support a cross-wave article or city-representative inference.

## Record identity and compact-input agreement

The raw public-use release `istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt` contains exactly one `nprog=3523` among 20,000 records. The compact semicolon input `istat_microdata_2013.csv` contains exactly one `id=3523` among 20,000 records. Independently extracted core heating-hour/control values agree after trimming release whitespace. Two ancillary device fields differ: raw routed blanks become compact zeros, as documented below. Exact source hashes, row locators and individual checks are saved in `survey_schedule_verification.json`.

| Field | Raw and compact value | Verified meaning / locator |
|---|---:|---|
|`reg`|5|Veneto; official classification `METADATI/Classificazioni/ECH_Classificazione_Anno 2013_var89.html`, row code5.|
|`reg_new`|50|Region coding with Trento/Bolzano distinguished; this does not add city identification.|
|`q_2_30`|1|Heating system present; Section3 applicability checked in questionnaire p34.|
|`q_3_0`|2|Autonomous heating system for the dwelling, sole or predominant system; questionnaire p34, question3.0.|
|`q_3_10`|1|Winter use every day or almost every day; questionnaire p40, question3.10; classification var175.|
|`q_3_11A`|4|Reported mean on-hours within05:00–13:00; questionnaire p41, question3.11_A.|
|`q_3_11B`|4|Reported mean on-hours within13:00–21:00; questionnaire pp41–42, question3.11_B.|
|`q_3_11C`|1|Reported mean on-hours within21:00–05:00 next morning; questionnaire p42, question3.11_C.|
|`q_3_11tot`|9|Released total equals4+4+1; trace variable order360, HTML line365.|
|`q_3_12`|1|Respondent reports possibility of automatic room-temperature regulation; questionnaire p43, question3.12; classification var179.|
|`q_3_13_1`|0|Thermostat option **not selected**; questionnaire p43, question3.13 option1; classification var180 confirms0=no,1=yes.|
|`q_3_13_2`|1|**Timer (operating-time regulation) selected**; questionnaire p43, question3.13 option2; classification var181 confirms0=no,1=yes.|
|`q_3_13_3`|0|Thermostatic-valve option **not selected**; questionnaire p43, question3.13 option3; classification var182 confirms0=no,1=yes.|
|`q_3_19_6`|0|Relevant additional source-model field agrees; this bounded audit does not independently interpret its item wording.|
|`q_2_31C`|2|No individual fixed heating appliances; parent routing field for2.32. Questionnaire p24 asks2.32only when2.31_C=1.|
|`q_2_32C`,`q_2_32D`|Raw blank;compact0|Traditional biomass stove/fireplace item fields. These are routed non-applicable blanks,not explicit no answers(code2). Compact0is a preprocessing sentinel;do not claim literal raw/compact equality.|
|`coef_red`|470|Released expansion weight; no weighting is applied in the selected illustrative single-record experiment.|

The release trace (`ECH_Tracciato_Anno 2013.html`) locates `reg` at variable order89/HTML line94; heating frequency order175/line180; block-hour fields orders176–178/lines181–183; automatic regulation order179/line184; control options orders180–182/lines185–187. Its abbreviated labels alone do not establish the full question wording; the questionnaire supplies that wording.

## Response units, reference period and routing

Question3.11_A asks about mean hours switched on during a winter day. For autonomous systems, interviewer guidance explicitly asks respondents to average days when somebody is home and days when nobody is home. The answer is not restricted to observed occupied days. Equivalent guidance is repeated for the afternoon and night bands. These are **reported system on-hours**, not measured occupancy, burner duty, indoor comfort, heat demand or energy consumption.

The questionnaire's operational instructions require entering0 if the heating is never on in a band, entering1 for fractions of an hour, and rejecting values above8. Soft ranges are0–5 for A,0–7 for B and0–2 for C; the selected4/4/1 values satisfy all three. A hard consistency check rejects all-zero block totals when frequency is1 or2. Thus the release uses integer, coarsened responses; interpreting them as exact repeated duty hours is a simulation construction, not survey precision.

Section3 is asked of households reporting heating at2.30=1 (questionnaire p34). Q3.10 winter frequency codes3–5 skip the block-hours question (p40); the selected code1 follows the block-hours route. Questions3.12–3.13 apply only to central or autonomous sole/predominant systems (`q_3_0=1,2`, p43); `q_3_12=2` skips3.13. The selected system2/automatic-regulation1 is consistent with its reported timer option. Multiple control options are allowed.

The file description labels the release reference period as2013 (description PDF p1, edition2017). The methodological note p5 dates CATI collection to March–July2013. The field-specific wording is an **average winter day** and winter-month frequency. It does not itself identify one named calendar winter, interview date, exact heating-season dates or an hourly calendar. Do not describe4/4/1 as monitored2013daily operation or as a validated calendar-year schedule.

## Timer versus thermostat and model assumptions

The selected positive control response is `q_3_13_2`, whose questionnaire meaning is timer/schedule regulation. It is **not** the thermostat option (`q_3_13_1`) or thermostatic valves (`q_3_13_3`). The simulation protocol must describe its code branch accordingly. If the source implementation assigns22/19°C when `_2` or `_3` is selected, those temperatures are inherited model assumptions; the official responses do not measure or imply either value. No claim about a respondent's thermostat setpoints is permitted.

Within-block continuity, seeded placement, earliest/latest alternatives, off/on temperatures, night-to-calendar rotation, annual repetition and season boundaries remain modelling choices. The hours constrain total encoded duration inside each band, not the exact sequence or true physical heating operation. A sensitivity comparison can preserve these encoded totals without claiming that any schedule is correct or improved.

## Geographic boundary and source access

The official regional classification confirms `reg=5` as Veneto. Geographic names found in the raw release schema are `rip`,`reg`,`reg_new`,`dom`. The trace describes `dom` as municipality **type**, and the file description p6 lists aggregated size/metropolitan/terrain classes. Neither a type nor a regional code identifies a named municipality. File description p3 states that direct identifiers are removed and indirect identifiers protected. This selected response therefore does **not** establish Padova residence, province, local stock representativeness or a monitored Padova building. Padova can define weather exposure for an explicitly illustrative case.

Read-only access included complete relevant questionnaire pages24,34,38–43, introduction/routing context, trace and classifications, plus file-description pp1,3,6 and methodological-note p5. Rendered questionnaire pages40–43 were visually inspected; printed and PDF page numbers coincide. No original source, existing evidence file, method file or simulation input was edited. Unresolved unknowns remain actual within-band timing, setpoints, measured heating operation/energy, named city and a date-specific winter reference for this response.

This selected-record audit remains local. It does not assume permission or a scientific need to redistribute record-level survey data. All core schedule/control checks pass; the all-selected-field exact-match flag is deliberately false because of the two documented blank-to-zero conversions.
