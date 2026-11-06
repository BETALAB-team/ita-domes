# Historical command recording correction

Reviewer 02 found that v2/v3 `frozen_manifest.json` stored an executable/script command without its required SITE argument. The original manifests are preserved unchanged. The actual tool invocations did include Padova, Bolzano, Roma or Palermo as their positional argument, and generated the corresponding site directory. A normalized replay command for those historical adapters is:

`& "C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe" -B "<artifact-root>/04_analysis/run_climate_campaign_v3.py" Padova`

Substitute the actual site and v2 script where relevant. This is a transparent reconstruction of the omitted argument, not a newly discovered complete original command record. Both historical adapters write fixed paths and should not be rerun over preserved outputs.

The v4 interface requires `SITE --campaign NEW_ID`, records `sys.argv`, executable, launch cwd, command including both arguments and frozen source hashes, and refuses existing completed/hourly outputs. `--replay-only-seeded` provides a genuine separate replay without adding a scientific sample. The replay destination must be fresh and remain under the artifact root.
