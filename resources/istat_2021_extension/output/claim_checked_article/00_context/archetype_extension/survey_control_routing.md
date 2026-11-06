# Survey control and eligibility verification for the archetype extension

Checked20 October2026. Scope: archived four-group membership joined to the official2013 public-use file and compact input. No annual simulation or clustering fit was run.

## Verified domain

Questionnaire p34 defines `q_3_0=1` as the central system serving multiple dwellings and `q_3_0=2` as the autonomous system for the dwelling, in either case the sole or predominant heating system. Section3 is routed from heating presence at `q_2_30=1`. Questions3.7–3.11 concern that sole/predominant system (pp38–39).

Question3.10 asks winter-use frequency (p40). Codes1/2 mean daily or almost daily / a few days per week; codes3–5 skip3.11. Questions3.11_A/B/C, pp41–42, ask average winter-day on-hours in05:00–13:00,13:00–21:00 and21:00–05:00. For autonomous systems, guidance explicitly averages days with and without somebody home. Enter0 if never on in a band; enter1 for fractions of an hour. Hard maximum is8 per band. These are coarsened reported operating durations, not temperature, occupancy or thermal-load measurements.

The independently checked extension domain is: archived assignment; raw `q_3_0` in{1,2}; all three raw block-hour fields observed, integer and within0–8; positive sum. It contains **13,624** records: archetypes1/2/3/4 have5,430/3,693/2,991/1,510. Their frequency codes are1(n12,866) or2(n758), consistent with the questionnaire route. Another622 archived central/autonomous records have at least one missing raw band field and do not meet this observed-response rule. Do not recover them by silently turning missing hours into zero.

## Control meaning and routed blanks

| Field | Official locator and meaning | Handling boundary |
|---|---|---|
|`q_3_12`|p43: automatic room-temperature regulation possible;1yes/2no. Asked only for systems1/2.|A no skips3.13. Manual on/off buttons or remotes are explicitly excluded.|
|`q_3_13_1`|p43Q3.13 option1: one or more thermostats.|A reported thermostat is distinct from option2.|
|`q_3_13_2`|p43Q3.13 option2: timer, i.e. operating-time regulation.|Do not relabel it as an observed thermostat temperature.|
|`q_3_13_3`|p43Q3.13 option3: thermostatic valves.|Multiple options allowed.|
|`q_3_18`|p45: money invested in heating-cost reductions during the last five years, or the time resident if shorter;1yes/2no.|A no skips3.19.|
|`q_3_19_6`|pp45–46Q3.19 option6: application of thermostats, thermostatic valves or heat meters/accounting devices.|A past investment response, not measured current control operation. A routed blank is not an explicit no answer.|
|`q_2_31C`|Parent inventory indicator for individual fixed heating devices.|Question2.32 is asked only when this is1.|
|`q_2_32C`,`q_2_32D`|p24: traditional fixed biomass stove / traditional biomass fireplace; respective innovative systems connected to radiators are excluded.|Codes1/2 mean yes/no when asked. Routed blanks become0 in the compact input;0 is a preprocessing sentinel, not an official explicit-no code.|

Among eligible records, raw blank→compact0 occurs3,364 times for each3.13_2/_3,10,248 times for3.19_6, and8,789 times for each2.32C/D. All other selected raw/compact values agree after stripping whitespace. Crucially, applying the source's `==1` predicates to raw versus compact values yields **identical temperature branches for all13,624 records**. This is an implementation equivalence check for the declared branch, not proof that every missing response is substantively “no”. If a future model uses explicit-no versus missing distinctions differently, it must revisit these records.

Current `EndUse.py` maps timer/valve/recent-control-investment selected to22/19°C; otherwise22/11°C. For central/autonomous systems, an affirmative stove/fireplace field adds2°C to both levels, giving24/21 or24/13°C. These four branches are **source-model assumptions**, not measured household setpoints. Preserve joint control flags and within-group branch frequencies in the compression design.

`survey_control_eligibility_audit.json` records all16 archetype×branch counts, domain counts, blank conversions and mismatch lists. It checks the full eligible universe. A later selected-record manifest should still be checked against its exact IDs and frozen hashes before launch; it must not claim a new city-representative sample.
