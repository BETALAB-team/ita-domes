# Archetype extension execution and interpretation

The v1 singleton prototype selection is preserved and unexecuted. Its four group prototypes and pooled prototype have identical complete temperature profiles. The v2 amendment responds to this input-only collapse before thermal results. It implements four demographic archetypes through training group-conditioned probabilities over the same four separately simulated, real survey source-control profiles. Pooled and conditional methods share components; conditional weights add statistical parameters, so the comparison does not establish equal-complexity superiority.

The frozen v2 files are in 05_simulations/archetype_campaign_v2/pre_run. The current governing protocol is mixture_protocol_v2.md, copied there as selection_protocol.md. All 32 holdouts and the seed20261020 split are unchanged. There are 36 distinct scientific profiles per coordinate TMY: 32 detailed respondent references and four pooled branch medoids. Planned execution count is 144 scientific annual cases and eight repeat/warm-up checks, 152 in total. Repeat checks use component22/19, independently of results. Earlier archetype v1 has no annual runs. Earlier v5 and quarantined historical campaigns remain separate.

The physical model is the reviewed v5 study-local path. Source-only readiness confirms the zone construction AST is identical. Its single-commit solver, IR-derived mean longwave approximation and independent-surface preprocessing modules are unchanged. Only traceable respondent heating durations and source-rule assumed temperature/control branches vary. Demographic membership does not create gains, occupancy or physical building differences. These are useful sensible thermal heating results, not final energy or measured validation.

Root authorized the exact frozen launch in 01_plan/archetype_extension/launch_gate.json. The campaign is now technically complete; current candidate results and review status are in simulation_results_v2.md. Exact commands below use an explicit site and fresh campaign ID. Run from C:/Works/GRINS/GRINS-Tool/resources/istat_data. The executable is the existing EUReCA virtual environment; it is read-only. The runtime manifest rejects source/code drift. The runner refuses completed/hourly output destinations. A fresh campaign ID can replay the same frozen inputs without overwriting this campaign.

```powershell
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B output/claim_checked_article/04_analysis/archetype_extension/run_mixture_campaign.py Padova --campaign archetype_campaign_v2 --preflight
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B output/claim_checked_article/04_analysis/archetype_extension/run_mixture_campaign.py Padova --campaign archetype_campaign_v2
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B output/claim_checked_article/04_analysis/check_full_step_balances_v4.py Padova --campaign archetype_campaign_v2
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B output/claim_checked_article/04_analysis/check_surface_forcing_v5.py Padova --campaign archetype_campaign_v2
```

Only after Padova's saved numerical, full-balance and forcing gates pass may the same three annual/check commands be run with Bolzano, Roma and Palermo in place of Padova. The old checker filenames reflect their verified origin, not old scientific output reuse. A separate fresh-destination single-component replay is supported by --replay-only-component; it is not necessary for the planned152-case sample and must be counted separately if executed.

```powershell
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B output/claim_checked_article/04_analysis/archetype_extension/analyze_mixtures.py --campaign archetype_campaign_v2 --out-name results_v2
```

Analysis independently integrates raw hourly CSVs using standard-library arithmetic, binds each site's frozen input copies and source hashes, and then constructs endpoint mixtures. The annual maximum prediction is the expected individual annual maximum, not a coincident aggregate load. Primary metrics are evaluation-stratification-weighted MSE and RMSE plus paired squared-deviation difference; MAE and signed deviation are secondary. No reference denominator is used; zero endpoints are retained. Evaluation weights describe the eligible empirical group×branch domain, while mixture probabilities are learned from training records. Neither are national expansion weights.

Readiness revisions01 and02 remain under pre_run. Padova/failure.txt is explicitly preserved as an input-only preflight source-drift diagnostic, documented beside it; it is not an annual run. No new annual result existed at the stable final pre-run freeze. The completed execution and candidate results are recorded in simulation_results_v2.md and the machine campaign_completion_manifest.json; fresh full-claim review remains pending at this handoff.
