# Non-destructive v5 reproduction

Use the recorded existing EUReCA environment and a **new** campaign ID. Inputs, scientific endpoints and fixed coordinate cases are defined in `surface_correction_protocol_v5.md`, which retains the single-commit correction protocol and explicitly bounds the weather-field exception. A campaign ID is a leaf name beneath this artifact root's `05_simulations`; traversal/path escapes and existing hourly/completed outputs are rejected.

Example PowerShell command, replacing `fresh_reproduction_01` with an unused ID:

```powershell
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B 'C:/Works/GRINS/GRINS-Tool/resources/istat_data/output/claim_checked_article/04_analysis/run_climate_campaign_v5.py' Padova --campaign fresh_reproduction_01
& 'C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe' -B 'C:/Works/GRINS/GRINS-Tool/resources/istat_data/output/claim_checked_article/04_analysis/check_full_step_balances_v4.py' Padova --campaign fresh_reproduction_01
```

This runs the five Padova primary/check configurations and independently recomputes their balances. Run the three other coordinate labels only after Padova's forcing, transition and full-step gates pass. For a single genuine annual replay in another unused destination, add `--replay-only-seeded`; use the same checker afterward. This replay does not become another scientific sample.

The current candidate campaign is `climate_campaign_v5`; the separately executed replay is `climate_campaign_v5_replay`. Its seeded hourly file matches the current Padova seeded file byte-for-byte. `v5_replay_interface_checks.json` records both that replay and a completed-destination refusal test, including unchanged hashes of all existing files. The twenty candidate campaign cases and one replay are listed separately in `run_inventory.csv`.

Each v5 frozen manifest records the exact executable, script, SITE, campaign, optional replay flag, `sys.argv`, launch cwd and all protocol/config/design/source/input hashes. Historical v2/v3 omitted-SITE command records remain unchanged; `recorded_command_addendum.md` explains the correction. The final summarizer is tied to the current named article campaign; reproductions in other destinations can use the per-site checker/summary outputs or an explicit separately named aggregation adapter.

After the state-balance checker, run `04_analysis/check_surface_forcing_v5.py SITE --campaign NEW_ID` for independent component and actual solver-forcing checks. The reused state-balance checker retains its v4 filename and accepts the explicit new campaign path; it does not reuse old run results.
