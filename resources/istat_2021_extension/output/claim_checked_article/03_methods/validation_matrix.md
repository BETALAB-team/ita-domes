# Current validation matrix — v5 candidate

**Current scientific status: v5 — reviewed conditional model illustration.** All 66 earlier annual executions are quarantined. The current campaign has 20 annual cases (12 primary, 4 repeat and 4 longer-warm-up checks), plus one separate genuine seeded replay that is not another scientific sample. Total execution count is 87. Independent technical review 03 supplement passed; fresh full-claim review 04 and final v04 closure passed for the bounded conditional claims (07_reviews/final_closure.md); no empirical validation is claimed. Exact frozen pre-run protocols/configs/source snapshots remain unchanged under each run directory. This canonical index was updated after execution and does not replace those pre-run records.

| Output/claim | Reference and independence | Scale/boundary | Metric and rationale | Current disposition / unvalidated scope |
|---|---|---|---|---|
| Schedule reconstruction | Verified survey fields and hashed GRINS rule; no observed hourly schedule | One response, three 8 h blocks | Exact 4/4/1 high-setpoint durations | Passed; heater runtime is not constrained to those totals |
| Surface physics | Independent synthetic scalar fixtures, windowless roof and standalone contributions | Opaque SW/LW, glazed SW separately | Component error ≤1e-8 W | Passed; not empirical envelope characterization |
| Actual surface forcing | Exported plane/input/area/SHGC components; separate reconstruction | 175,200 site-surface-hours across four sites | Error ≤1e-8 W; max 1.364e-12 W | Passed; retained model radiation approximations remain |
| Surface order | Reverse/rotate actual eight-surface lists | All five cases per site | Node forcing error ≤1e-8 W; max 1.819e-12 W | Passed; correct geometry assumed, not surveyed |
| Solver forcing | Independent component/node sum versus annual exported forcing | All 175,200 campaign hours | Node difference ≤1e-8 W | Passed; no measured truth target |
| Dispatcher and balances | Synthetic transitions plus exported old/current states | Hourly useful sensible heat | Every node residual ≤1e-6 W; max 1.769e-10 W; exact continuity | Passed; bounded sensible-only adapter, not general HVAC validation |
| Integration and peaks | Standard-library summation from raw hourly files | 20 annual files | <1e-9 kWh and <1e-12 kW discrepancies | Passed; useful demand only |
| Repetition/warm-up | Same model/input duplicate and 168/336 h warm-up | Four sites | Exact repeats; annual ≤0.5%, peak ≤1% | Passed; numerical criterion, not empirical tolerance |
| Weather forcing | Primary EPW/PVGIS definitions, exact JSON record matching | Active fields, valid horizontal IR | Sentinel/range checks; IR reconstruction <1e-9 W/m² | Passed active-path gate; raw invalid unused wind explicitly retained |
| Reproduction interface | Separate genuine seeded replay, completed-destination refusal | Fresh run directory | Byte-identical hourly replay; unchanged existing files | Passed; not another scientific sample |
| Whole-building benchmark | Independently specified matched case | Matching input/time/energy boundary required | No acceptance score invented | Missing |
| Padova empirical accuracy | Untouched compatible meters/building metadata | Matching historical weather and meter/end-use boundary | Bias/MAE/peaks as data permit | Not established; ANACI provenance/access plan pending |
| Aggregate accuracy | Independent official totals with aligned scope | Carrier/year/sector/population/weights | Boundary-aligned discrepancy and uncertainty | Not tested by illustrative pilot |
| Climate sensitivity | Same model/input across four coordinate TMYs | Fixed assumed dwelling and season | Paired conditional differences | Candidate result only; no empirical transfer accuracy |

Independent scientific review and final v04 closure are complete for the bounded v5 claims; this does not establish empirical validation. Current result paths are `04_analysis/paired_schedule_contrasts_v5.csv`, `independent_output_checks_v5.json`, `independent_surface_checks_v5.json` and per-site raw/check files. All 66 earlier executions are quarantined.

Post-review status annotation by the orchestrator: the saved final closure supersedes procedural pending-review labels in historical sections. Frozen protocols and run snapshots remain unchanged.
