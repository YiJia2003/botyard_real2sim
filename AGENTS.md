# Agent Guide

This repository calibrates an unloaded tendon-driven hand against synchronized real-hardware traces. Before repository work, read the [project skill router](.agents/skills/real2sim-agent-router/SKILL.md); it selects the relevant Superpowers workflow and project references. For recorded curves, model parameters, or calibration workflows, follow its route to [the real2sim calibration skill](.agents/skills/real2sim-calibration/SKILL.md). See [shared agent skills](docs/SHARED_AGENT_SKILLS.md) for skill sources and harness notes.

## Deployment and responsibilities

The simulation host runs this repository at `/home/cyj/botyard_real2sim`. Its agent validates received recordings, replays commands, diagnoses residuals, changes model structure or parameters when supported by evidence, evaluates held-out recordings, and publishes results and requests for more data. Hardware control, firmware, ROS, SWD acquisition, clock qualification and dataset preparation run on the separate hardware host.

The simulation checkout is self-contained. It does not require the hardware host's `embedding_ws`, `ros2_ws`, port mappings or device access. Use the in-repository [bridge contract](docs/REAL2SIM_BRIDGE.md) for input, result and acquisition-request schemas. Hardware settings are a dated capability snapshot; the recording's archived firmware/configuration and quality evidence determine what was actually measured.

On the simulation host, read only ready inputs under `real2sim/experiment/<bundle_id>/` and write task artifacts under `real2sim/results/.staging/<bundle_id>.partial/`. Follow `TASK.json` and publish with `real2sim/bridge/remote_publish_result.py`. A request for another excitation is written to `next_data_request.json`; the hardware-host agent validates, plans and executes it. Repository/model edits remain normal simulation-host work; the read-only restriction applies to received experiment bundles.

## Commands

- `./run.sh replay ...` replays recorded commands through Isaac Lab / PhysX.
- `./run.sh curves ...` creates curve plots and per-channel metrics.
- `./run.sh sweep ...` enumerates one parameter; it does not fit or rank candidates.
- `./run.sh data --manifest ... --strict` validates a dataset manifest and its CSV.
- `./run.sh parameters --config ...` prints the active parameter catalog as JSON.
- `./run.sh evaluate ...` scores a replay with explicit channel tolerances and quality gates.
- `./run.sh fit ...` runs bounded training-set fitting and held-out validation; install `.[calibration]` for SciPy.
- `./run.sh test` runs the project tests; `./run.sh check` runs Ruff checks.
- On the simulation host, use `python3 real2sim/bridge/remote_publish_result.py --project-root /home/cyj/botyard_real2sim --bundle-id <id>` to publish the completed result packet. Source scripts live in `bridge/`; `real2sim/bridge/` is their deployed runtime copy.
- On the hardware host only, `python3 scripts/real2sim_bridge.py ...` builds/verifies/uploads bundles and reads remote results. `doctor`, `build-capture`, `publish`, `notify` and `plan-requests` belong to that side. The current exporter accepts legacy host-aligned inputs; the three-board SWD adapter and timing/source-quality qualification remain incomplete. A remote diagnostic task may report those gaps; it must preserve `fit_ready=false` and request the missing evidence.

Use the configured Isaac Lab interpreter. Unit tests and curve comparisons do not prove real-hardware calibration or full PhysX convergence.

For paired or fixed-partner acquisition requests, write `execution_mode=paired` with a full `real2sim-pair-plan-v1` payload, following [the request contract](docs/REAL2SIM_BRIDGE.md#7-补采请求契约) and the in-repository templates. A two-joint list in a separate request produces separate runs. The hardware host owns waveform generation, measured coverage auditing and execution; the simulation host need not run or install those tools. Generated candidates and dry runs are not hardware coverage evidence.

## Calibration Constraints

- Treat `configs/real2sim/s1.yaml` values as provisional unless a recorded experiment and held-out validation support a fitted claim. Generated examples and reference runs are simulation data, not hardware evidence.
- Preserve raw recordings. Record the actual command events, fallback mode, firmware/control revision, timebase, sensor calibration, encoder direction/gear conversion, and hand pose before interpreting a trace.
- The three target observations are joint position, continuous winding-spool output-shaft angle, and motor-side tendon tension. Never substitute simulated channels or distal tension for missing measurements.
- Check replay metadata for failed runs, missing initial state, effort saturation, and mimic error before using a curve for calibration.
- Keep physical parameters separate from numerical solver settings. `armature_scale` and `effort_limit_nm` are numerical controls, not identified motor inertia or tendon tension limits.
- Check that a parameter is active in the selected model before sweeping it. Geometry overrides constant moment arms; geometric return springs replace the legacy angular spring; measured initial motor angle re-estimates the encoder zero.
- Use the manifest schema and validator before fitting. Tolerances and parameter bounds must be explicit and supported by measurement or a stated experiment decision; never infer physical bounds from the preset.
- `quality_valid` means the replay and curves are usable for scoring; `tolerance_pass` means every channel meets its explicit threshold; `candidate_pass` requires both. An over-tolerance but scoreable training candidate may still guide optimization.
- A fit is accepted only when the held-out candidate passes all configured channel gates and its normalized score is strictly lower than the held-out baseline. This still establishes a curve-fit candidate, not a calibrated physical model.
- Project skills live under `.agents/skills/`. Codex can read them through the project instructions; DeepSeek Harness can use the same paths when its runtime loads repository skills and `AGENTS.md`.

## Verification

For model or calibration changes, run focused tests and `./run.sh test`. Run PhysX integration only when the change affects the engine adapter or coupled dynamics and the Isaac environment is available. Report which checks actually ran and retain each run's CSV, parameter snapshot, metadata, and comparison report.
