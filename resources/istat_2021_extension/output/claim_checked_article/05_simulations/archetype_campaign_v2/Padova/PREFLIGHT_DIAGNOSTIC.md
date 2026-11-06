# Preflight diagnostic, not an annual run

The adjacent failure.txt records the independent design auditor's input-only --preflight attempt during a concurrent metadata/source freeze revision. The runtime hash guard rejected runner drift before engine/weather execution and before any annual thermal timestep. No hourly CSV or thermal summary was produced by that attempt. The failure is preserved unchanged as chronology.

The final source freeze is recorded in ../pre_run/readiness.json and runtime_source_manifest.json. Earlier readiness revisions remain in ../pre_run/readiness_revision_01 and readiness_revision_02. The final runner prints preflight failures without creating or modifying a simulation destination. A subsequent authorized annual campaign is a distinct execution recorded by frozen_manifest.json and completion_manifest.json; the historical preflight failure must not be counted as an annual run or treated as a failed thermal result.
