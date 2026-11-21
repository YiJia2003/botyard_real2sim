# real2sim controller source synchronization

> Execution: native, in the existing checkout; user requested synchronization.

**Goal:** Match the current local embedded real2sim profile's command floors
and motor velocity deadband without changing the physical calibration.

**Architecture:** Retain the H7 Fallback controller output as telemetry. Apply
the G4 velocity-target deadband at the runtime's tendon-speed input, after H7
quantization. Keep motor-side threshold and gear ratio separately configurable
per joint. Existing serialized configurations without the new fields retain
their previous H7-only behavior (zero driver deadband).

**Source specification:** H7 `53d820341688b4b875a6009ac2fd54e57adbe01b`, G4
`d395c9ff50e62faf42c1d39b2c9175ee124f4b56` under
`/home/cyj/botyard_embedded`. H7 trace MFJ3 minimum 2.5 degrees; others 0.5;
maximum 89.5; angle dead zone 1. G4 documented experiment profile threshold
200 rad/s, source gear ratio 800; strict comparison after H7 integer output.
Generic G4 builds default to 250, not 200. Source checkouts are not flashed
firmware evidence.

## Task 1: Synchronize and verify the controller boundary

Files: `config.py`, `runtime.py`, `cli/replay.py`, `parameter_catalog.py`, both
checked-in YAML presets, focused controller/runtime/replay tests, and
`docs/REAL2SIM_CONTROLLER_SYNC.md`.

- [x] Add failing behavior tests for MFJ3's floor, deadband below/at/above the
  signed threshold, per-joint gear ratio, and integration continuing while the
  motor request is suppressed. Check the plant receives the gated speed.
- [x] Add validated controller configuration vectors
  `motor_velocity_deadband_rad_s` (nonnegative, legacy default zero) and
  `motor_gear_ratio` (positive, default 800). Both presets select 200 and 800.
- [x] Add runtime `motor_target_speed_rad_s` in spool units; preserve
  `controller_speed_rad_s`, expose both in CSV, and use the gated value in
  `advance`. Do not impose a minimum nonzero speed or clear H7 PID state.
- [x] Update presets, parameter catalog, and source/compatibility documentation.
- [x] Run focused tests, `./run.sh test`, `./run.sh check`, inspect active
  parameters, and review the final diff. Run a short PhysX smoke replay if the
  Isaac environment supports it; retain CSV/config/metadata.

Review focus: old snapshots; strict signed boundary and integer quantization;
MFJ3's partner-hold target below its floor; no mutation of received bundles;
do not confuse current source/profile with recorded hardware state or fitted
physical parameters.

## Execution evidence

The source audit also confirmed a downstream G4 spool speed cap of 6.25 rad/s.
Added `motor_spool_speed_limit_rad_s`, with legacy default zero (disabled) and
explicit 6.25 in both presets, after the deadband and before the plant input.

Focused regression tests were observed failing before implementation; final
independent review ran 71 focused tests successfully. Full suite: 317 passed,
two existing warnings for deliberately incomplete legacy command fixtures.
`./run.sh check` and `git diff --check` passed. No commit or hardware operation.

Two generated PhysX smoke runs completed: 1500 steps for the MFJ3 target floor
and 300 steps for suppression of low-speed requests. Neither had effort
saturation. All CSV targets followed the configured gate; the latter retained
nonzero H7 requests while the motor target and spool displacement stayed zero.

Artifacts are retained under
`real2sim/results/.staging/controller-sync-20261121.partial/`, including CSVs,
parameter snapshots, metadata, comparison reports, test logs and source/code
reviews. This is standalone repository verification, not a received bridge
task/result packet; no `TASK.json` input or hardware-calibration acceptance.
