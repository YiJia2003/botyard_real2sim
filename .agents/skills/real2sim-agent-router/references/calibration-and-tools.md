# Calibration and Tools

## Calibration route

For a new recording, run `./run.sh data --manifest ... --strict` and inspect `./run.sh parameters --config ...`. Establish a baseline, then use `evaluate` with explicit per-channel tolerances and coverage gates. Keep train and validation recordings independent.

For an existing fit, inspect `fit.json` first: compare `baseline_validation`, `heldout_validation`, `heldout_channel_changes`, and the `validation_pass`, `improved`, and `accepted` fields. Then inspect the referenced candidate folders' `evaluation.json`, replay metadata, and logs for quality failures, coverage, missing initial state, effort clipping, and mimic error. Use `./run.sh curves` to locate residual patterns before deciding whether to refit or investigate model structure.

The `fit` command runs bounded, serial SciPy differential evolution. It needs explicit evidence-based bounds and a dimensionless objective, and only accepts active physical parameters from the catalog. It searches parameter values; it does not infer model structure, jointly fit multiple training recordings, or prove physical calibration. The `calibration` package extra provides SciPy. A skill or plugin does not execute the optimization.

Before proposing a new mechanism, compare residuals across amplitude, speed, direction, dwell, and independent held-out sequences. Check synchronization, sensor calibration, control branch, initialization, clipping, geometry, and time-step sensitivity. A structure change needs a recorded residual hypothesis, a preserved baseline, refitting, and held-out checks across all channels.

## Optional tooling

No additional optimization plugin is needed for the local loop: the project CLI already provides dataset validation, parameter discovery, replay, evaluation, and fitting with JSON artifacts. Current software gaps are multi-recording joint fitting, residual-to-model reasoning, experiment design, and run caching/resume; an external plugin would not supply those project-specific behaviors or hardware evidence.

If the harness exposes the local CAD MCP, use it when a STEP geometry measurement is needed. Inspect the artifact first, then find entities and measure with the exact revision-bound IDs. Results are geometry facts, not tendon-path labels or design intent. `docs/S1手指-V3.2.STEP` declares no PMI, so a human or authoritative drawing must map measured entities to physical parameters. CAD measurement can inform geometry priors but cannot fit dynamic parameters or validate curve agreement. DSH access to this MCP is unverified.

A hardware-data connector or long-running experiment service may justify a future adapter once its data and run-state contracts are defined. Add an API or MCP wrapper only when an agent needs access beyond the current local CLI.
