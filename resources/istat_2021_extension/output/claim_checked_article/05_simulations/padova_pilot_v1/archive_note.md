# Snapshot timing

The protocol, configuration, design and source hashes were saved before the v1 engine loop in `frozen_manifest.json`. Immutable protocol/adapter copies were archived after the v1 run and before v2 expansion. Their exact bytes are checked against those pre-run hashes. This timing is disclosed: the copies were not themselves created before the v1 run. v2 campaigns create their immutable copies before execution.

The first adapter archive copied text through Windows newline conversion. The archive was corrected to an exact byte copy of the unchanged original source and rechecked against the frozen hash. No model input or result changed.
