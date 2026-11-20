# Agent Calibration Workflow

This document is the operating procedure for agents and engineers using the current real2sim tools. Local-to-remote data handoff uses the staging/READY bundle protocol in [REAL2SIM_BRIDGE.md](REAL2SIM_BRIDGE.md); the remote side must only consume bundles with `READY.json` and must publish results with `real2sim/bridge/remote_publish_result.py`. The repository skill at `.agents/skills/real2sim-calibration/SKILL.md` routes agents here for the detailed workflow. Codex can read the project skill through `AGENTS.md`; DeepSeek Harness can use the same path when its runtime loads repository skills and `AGENTS.md`, though automatic loading is not verified here.

This workflow runs on the simulation host. The hardware host controls and records the hand; its agent prepares and uploads the data and executes requested follow-up experiments. The simulation-host agent validates inputs, diagnoses residuals, improves model structure and parameters, evaluates independent recordings, and returns results and acquisition requests. It does not need `embedding_ws`, ROS or device access. Received experiment bundles remain read-only; evidence-supported model and source changes belong to the simulation work.

## 1. Qualify the recording

Use an untouched copy of the hardware recording. Confirm the exact firmware/control branch, Fallback state, hand-root pose, selected joints, command units, feedback units, sensor positions, and sensor zero/scale. Required comparison traces for each fitted joint are:

- `<joint>/position_rad`: measured joint angle.
- `<joint>/motor_position_rad`: measured continuous winding-spool output-shaft angle, with direction and any gearbox conversion already resolved.
- `<joint>/tension_N`: measured motor-side tendon tension.

The actual sent angle/speed/third-field commands and their event timing must be available for replay. Preserve `command_seq`, event markers, fallback evidence, and the raw timestamps where available. The third command field is not enforced as a tension target by the current Fallback model.

The hardware host prepares the canonical data and manifest using `src/botyard_real2sim/schemas/dataset-manifest.schema.json`. On the simulation host, validate the delivered manifest with `./run.sh data --manifest /data/recording/manifest.json --strict`; do not rebuild or edit an input bundle that is already READY. The validator checks recorded calibration and motor-transform evidence, required channels, timestamps, command events, the canonical CSV SHA-256, split, coverage, and measured sensor skew against the explicit `quality_gates.max_sensor_skew_s` limit and its `max_sensor_skew_source`.

Check the archived source recording ID and SHA-256; exports or time slices from one capture retain the same source identity. The hardware host must document clock mapping, sensor-age limits, angle unwrapping, direction and gearbox conversion. The validator does not perform or independently verify those physical transforms. If evidence is missing, report the blocker and request a corrected new bundle or an additional measurement. Simulation-host diagnostic transforms may be saved separately with their method and source hashes; they do not change the original evidence or remove an unknown quality limit. A plot-time global tension bias is not applied to replay initial tension, so do not rely on it as the sole correction for the initial state.

## 2. Establish a baseline

Before tuning, inspect the active parameter catalog and record the current revision, config, real CSV hash, firmware, experiment conditions, and reason for the run:

The paths under `logs/calibration` below are standalone development examples. For a bridge task, use its actual ready input bundle and put all generated artifacts under the result staging directory specified in `TASK.json`; replace the example paths, joints, tolerances and bounds with that task's values before running.

```bash
./run.sh parameters --config configs/real2sim/s1.yaml \
  > logs/calibration/parameter_catalog.json

./run.sh replay --commands /data/recording/capture.csv --strict-commands \
  --config configs/real2sim/s1.yaml --out logs/calibration/baseline/sim.csv

./run.sh evaluate --real /data/recording/capture.csv \
  --sim logs/calibration/baseline/sim.csv --joints FFJ2 FFJ3 \
  --tolerance angle_deg=2 --tolerance motor_angle_deg=3 --tolerance tension_N=1 \
  --out logs/calibration/baseline/evaluation.json
```

Keep the generated `.params.yaml`, `.metadata.json`, evaluation JSON, and logs beside the run. The evaluator reports `quality_valid` (data/replay can be scored), `tolerance_pass` (every configured channel is in tolerance), and `candidate_pass` (both conditions hold). Per channel it gates both `real_data_coverage_ratio` over the full recording and `coverage_ratio` for simulation support over finite measured intervals. Inspect per-channel metrics and reasons; never treat low RMSE without adequate coverage as a pass.

`./run.sh curves` remains useful for plots and residual diagnosis. For acceptance or fitting, use `evaluate` because it enforces all three required channels, coverage, replay status, clipping, mimic error, and explicitly supplied per-channel tolerances.

## 3. Tune in identifiable groups

Use the documented order in [REAL2SIM_GEOMETRY.md](REAL2SIM_GEOMETRY.md#后续数据和辨识顺序): first validate signs and geometry using slow motion; then fit return-spring/preload and low-speed bidirectional friction; next use motor-to-joint displacement and tension to study compliance/travel; then use acceleration, dwell, and reversals to study equivalent mass, contact history, and motor-speed response.

Change one interpretable group at a time. `./run.sh sweep` is a one-parameter enumeration tool, not an optimizer; its example values are exploratory and must not be described as physical bounds. Check that each parameter affects the selected model before sweeping it:

- Configured tendon geometry overrides constant joint/crossing moment arms for the covered joints.
- Geometric return-spring paths replace the legacy angular spring terms.
- A measured initial motor angle re-estimates the encoder zero during reset.
- Dynamic routing rejects material cable damping and nonzero `crossing_guide_mu` is not implemented.
- `armature_scale`, `effort_limit_nm`, integration steps, and solver controls are numerical settings, not fitted physical parameters.
- The configured distal mimic ratio must agree with the USD's fixed J1=J2 coupling.

Avoid fitting strongly correlated values together. In particular, absolute routing free/max lengths and the motor reference cannot all be inferred independently without an absolute length/encoder reference. Preserve the initial loading history where possible; the current dynamic-routing reset cannot reconstruct unknown friction history from one sample. `parameters` reports activation and tunability but intentionally does not invent numeric search bounds; bounds must come from measurement or an explicit, documented experiment decision.

For a bounded numerical fit, use separate manifests with `split: train` and `split: validation`:

```bash
./run.sh fit --train-manifest /data/train/manifest.json \
  --validation-manifest /data/validation/manifest.json \
  --config configs/real2sim/s1.yaml \
  --parameter tendon.routing_spring_stiffness_n_per_m.FFJ2=500:3000:log \
  --tolerance angle_deg=2 --tolerance motor_angle_deg=3 --tolerance tension_N=1 \
  --max-evaluations 100 --seed 17 --out-dir logs/calibration/fit-001
```

The optimizer can use a scoreable training candidate even while its errors exceed acceptance tolerances; this lets it reduce error progressively. It rejects candidates with invalid data/replay quality. The fit command replays training and validation baselines at each dataset's recorded hand-root pose, searches only the explicitly bounded parameters, and checks the selected candidate on the held-out validation manifest. Validation must use a separate physical recording; the command rejects shared CSV paths/hashes and shared source recording IDs/hashes across splits. Do not split one physical capture into training and validation windows. It marks a fit accepted only when every held-out channel passes its tolerance and the held-out normalized score is strictly lower than the validation baseline. The fit.json result includes both baselines, held-out per-channel error changes, and the evaluation artifact paths. Every trial restarts Isaac Sim; fitting is serial and budget-limited.

## 4. Diagnose residuals before changing structure

Compare residuals across multiple amplitudes, speeds, directions, dwell times, and held-out command sequences. A systematic residual that changes with load, direction, or history can support a new physical mechanism; a single-trace improvement cannot. Check sensor scale/offset, synchronization, controller branch, initialization, effort clipping, geometry, and numerical step sensitivity before adding model freedom.

The motor is currently a commanded speed port with optional first-order lag, not a load-dependent electrical/rotor/gearbox model. The routing model is lumped, and the four fingers share an inferred S1 geometry while the thumb remains provisional. These are hypotheses to test against residuals, not automatic reasons to increase model complexity.

When changing structure, write down the residual pattern and proposed physical mechanism, preserve a baseline, make the smallest model change, refit the affected parameter group, and rerun both fit and held-out recordings. Keep the candidate only when it improves the target curves without degrading other channels or violating physical and numerical checks.

## 5. State the evidence

Report the datasets and split used, channels and units, alignment choice, parameter/config snapshot, code revision and dirty changes, replay quality checks, per-channel metrics, training and validation baselines, held-out score change, and acceptance result. Distinguish simulation-software tests, numerical convergence checks, training-fit performance, and real-hardware validation. Current automation fits one training manifest and evaluates one separate validation manifest; it does not perform multi-dataset joint fitting, convert raw sensor logs, infer model-structure changes, or promote/rollback model versions. A successful fit remains a curve-fit candidate and is not proof of physical calibration until real-hardware evidence supports it.
