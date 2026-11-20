# Remote bridge operator notes

This directory is synced to the simulation host at
`/home/cyj/botyard_real2sim/real2sim/bridge/`.  It contains the two scripts the
remote side must use for the real2sim data bridge.

## 1. Consume a ready bundle

Only directories with `READY.json` are ready.  Do not read or edit
`real2sim/experiment/.staging/` bundles.

```bash
cd /home/cyj/botyard_real2sim
BUNDLE=<bundle_id>
BUNDLE_DIR=real2sim/experiment/$BUNDLE
test -f "$BUNDLE_DIR/READY.json"

./run.sh data --manifest "$BUNDLE_DIR/manifest.json" --strict
```

`READY.json` is produced by `remote_verify.py` only after:

- every SHA-256 in `SHA256SUMS` and `BUNDLE.json/files` matches;
- `manifest.recording.sha256` matches `capture.csv`;
- no symlinks or unknown files are present;
- `./run.sh data --strict` returns `data_ready=true`;
- the staging directory was atomically renamed into `experiment/<bundle_id>/`.

## 2. Do the work described by TASK.json

```bash
python3 -m json.tool "$BUNDLE_DIR/TASK.json"
```

- `intent=transport_selftest`: verify and acknowledge only.  Do not run PhysX,
  curves, sweep, or fit.
- `intent=calibration`: follow the project workflow
  `data -> parameters -> baseline replay -> evaluate -> curves -> fit`.
  Use only the explicit tolerances in `TASK.json`.  Never infer tolerances or
  parameter bounds from presets.
- Never write to `real2sim/experiment/`; it is read-only from the remote side.
- Keep the baseline and evaluation artifacts beside the result.

## 3. Publish the result atomically

Write into the staging directory:

```bash
BUNDLE=<bundle_id>
STAGE=real2sim/results/.staging/$BUNDLE.partial
mkdir -p "$STAGE/baseline" "$STAGE/curves"
# write RESULT.json, report.md, next_data_request.json, baseline/, fit/, curves/
python3 real2sim/bridge/remote_publish_result.py \
  --project-root /home/cyj/botyard_real2sim \
  --bundle-id "$BUNDLE"
```

The publisher validates `RESULT.json`, hashes every artifact, writes
`RESULT_READY.json`, and atomically renames the staging directory to
`real2sim/results/<bundle_id>/`.  If it fails, read
`results/.staging/<bundle>/.partial/PUBLISH_FAILED.json`.

Minimal `RESULT.json`:

```json
{
  "schema_version": 1,
  "protocol": "botyard-real2sim-bridge/v1",
  "bundle_id": "<bundle_id>",
  "status": "completed",
  "claims_scope": "curve_fit_candidate_only",
  "summary": "one-paragraph conclusion",
  "next_data_request": "next_data_request.json"
}
```

`status` must be one of `completed`, `partial`, `blocked`, `failed`,
`acknowledged`.  A fit remains a curve-fit candidate only, even when
`accepted=true`.

## 4. Next data request

Use `experiments` (not `requests`) with one entry per requested excitation:

```json
{
  "schema_version": 1,
  "protocol": "botyard-real2sim-bridge/v1",
  "bundle_id": "<bundle_id>",
  "status": "needs_more_data",
  "reason": "what residual/parameter is not identifiable",
  "experiments": [
    {
      "id": "mfj2_slow_bidirectional",
      "joints": ["MFJ2"],
      "profile": "triangle",
      "center_deg": 20,
      "amplitude_deg": 10,
      "speed_cap_deg_s": 30,
      "duration_s": 8,
      "frequency_hz": 0.1,
      "directions": ["flex", "extend"],
      "dwell_s": 0.5,
      "repeats": 3,
      "purpose": "separate return-spring preload from low-speed routing friction"
    }
  ]
}
```

## 5. Hard limits

- Do not use simulated CSV as hardware evidence.
- Do not substitute motor-side tension for distal tension.
- Do not treat `armature_scale`, `effort_limit_nm`, physics/substep sizes as
  physical parameters.
- Do not continue scoring a replay with `saturated_force_evaluations > 0`.
- Do not edit a ready bundle in place; create a new bundle id for new data.
