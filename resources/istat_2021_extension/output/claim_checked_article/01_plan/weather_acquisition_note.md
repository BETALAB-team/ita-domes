# Weather acquisition evidence note

Access date: 2026-10-20. Orchestrator checked current primary JRC documentation while the methods worker was acquiring new inputs. These pages describe an available route, not evidence that any particular download succeeded; actual URLs, responses, hashes and metadata belong to the methods/run records.

- Official API documentation: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en . The documented PVGIS5.3 endpoint supports the TMY tool and EPW output. Access depth: official web documentation.
- Official TMY documentation: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en . It describes weather output and metadata including selected months. Access depth: official web documentation/search-returned technical content.
- The JRC product page distinguishes TMY building-weather output as a PVGIS5 capability while newer PVGIS6 tools are also available. Use explicit endpoint/version and response metadata; do not silently relabel a5.3download as the newest overall API.

Methods worker reported successful authorized new EPW and JSON downloads for Padova, Bolzano, Roma and Palermo coordinates after an initial sandbox-network failure. Acceptance is conditional on inspecting the persisted responses and climate QC. Selection uses computed descriptor contrast, not assumed monotonic temperature or cooling differences from place names. A weather-grid coordinate is not a locally representative household/building sample.
