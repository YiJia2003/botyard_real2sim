# real2sim data bundle `r2s-bridge-check`

- intent: `transport_selftest`
- split: `test`
- joints: MFJ2, MFJ3
- dataset_id: `r2s-bridge-check`
- protocol: `botyard-real2sim-bridge/v1`

This directory was staged, checksum-verified, and atomically renamed by the
real2sim bridge.  Do not edit files in place after `READY.json` exists: create a
new bundle instead.

Remote work must read this directory only.  Write results under
`real2sim/results/.staging/<bundle_id>.partial/`, then publish with
`real2sim/bridge/remote_publish_result.py`.
