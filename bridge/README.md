# Remote bridge operator notes

The simulation-host agent runs the model, replay, comparison and fitting tools
from `PROJECT_ROOT=/home/cyj/botyard_real2sim`. The separate hardware-host agent
controls the real hand, acquires data and uploads inputs. This guide is for the
simulation-host side.

Version-controlled scripts and templates live in `PROJECT_ROOT/bridge/`.
The hardware host's `publish` command deploys this directory to
`PROJECT_ROOT/real2sim/bridge/` by default, unless `--no-sync-scripts` is used;
commands below use that runtime copy. Updating the simulation repository alone
does not refresh the deployed copy, so keep its version consistent.
Full contracts and calibration instructions live at
`PROJECT_ROOT/docs/REAL2SIM_BRIDGE.md` and
`PROJECT_ROOT/docs/AGENT_CALIBRATION.md`. These paths are relative to the
project root, not this guide's source or deployed location. No hardware-host
checkout, ROS installation or probe access is required on the simulation host.

As of 2026-11-20, hardware acquisition uses three-board SWD RAM rings. The
SWD-to-canonical adapter and timing/source-quality qualification are incomplete.
Current SWD diagnostics retain `fit_ready=false`; analyze them only under an
explicit diagnostic task and report missing evidence. READY verifies the bundle
contract and does not override source-run failures or fit restrictions.

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
- `intent=calibration`: follow `data -> parameters -> baseline replay ->
  evaluate -> curves`. Fit only when TASK requests it, the training data pass
  the fit gates, and an independent validation bundle is ready. If validation is
  missing, diagnose and request it. Validation/test recordings are used only for
  evaluation. Use explicit TASK tolerances and evidence-based parameter bounds.
- Never write to `real2sim/experiment/`; it is read-only from the remote side.
- Keep baseline, evaluation, parameter snapshots and logs in the result staging
  directory. Model/source changes follow the repository calibration workflow;
  the read-only rule applies to received input bundles.

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
`real2sim/results/.staging/<bundle>.partial/PUBLISH_FAILED.json`.

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
      "amplitude_deg": 5,
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

The 2026-11-20 capability snapshot is for an unloaded left hand, middle joints
MFJ2/MFJ3. Requests use independent speed caps of 15–360 degrees/s, effective
targets J2=0.5–89.5 and J3=2.5–89.5 degrees, and the original Fallback PID with a
1-degree position deadband. The motor-shaft target deadband is 200 rad/s.
Speed is a position-servo cap; the PID has no 15-degree/s actual-output floor.
Stored module limits and one-sided actuation can prevent actual requested
speeds. Read each recording's archived configuration and quality evidence.
In the separate mode above, `directions` and `dwell_s` describe intent;
single-joint profiles fix their stages internally and have a 0.10 rad amplitude cap.

For a held partner or simultaneous motion, use `execution_mode: "paired"`,
`joints: ["MFJ2", "MFJ3"]` and a complete `pair_plan` with schema
`real2sim-pair-plan-v1`. Copy [the paired request template](next_data_request_pair.example.json).
The hardware host archives the payload and emits one pair-runner dry command
per repeat; it validates and executes the actual acquisition. The simulation
agent only writes the request. A two-joint list in separate mode produces two
independent runs. Paired points encode both targets, independent speed caps,
directions and dwell explicitly. The full schema is in
`PROJECT_ROOT/docs/REAL2SIM_BRIDGE.md`, section 7.

Plans contain 2–6001 points, start at offset 0, use intervals of at least 10 ms
and last at most 60 seconds. Positions and caps use rad and rad/s. Metadata
includes experiment_id, family, split, repeat and conditions. The hardware-host
validator checks bounds and device/controller state before execution. Generated
plans and dry runs do not establish measured coverage. Direct motor velocity is
not supported by the current generic acquisition runners.

When only the observation gap is known, describe it in `purpose` and
`requested_conditions` with `local_preparation_required: true`. The hardware-host
agent must first turn those conditions into a complete plan. Such requests are
not directly executable by `plan-requests`; do not invent a waveform or a claim
of completed acquisition to fill the gap.

With `repeats>1`, each frozen execution plan gets its own experiment ID suffix and repeat number; the source payload remains archived. Use `repeats=1` when preserving one existing campaign experiment ID.

## 5. Hard limits

- Do not use simulated CSV as hardware evidence.
- Do not substitute motor-side tension for distal tension.
- Do not treat `armature_scale`, `effort_limit_nm`, physics/substep sizes as
  physical parameters.
- Do not continue scoring a replay with `saturated_force_evaluations > 0`.
- Do not edit a ready bundle in place; create a new bundle id for new data.
