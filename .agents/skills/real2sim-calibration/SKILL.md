---
name: real2sim-calibration
description: Calibrate this project's unloaded tendon hand model against real joint, winding-spool motor-angle, and motor-side tension traces. Use when inspecting datasets, replaying experiments, tuning parameters, diagnosing curve residuals, or deciding whether the model needs a structural change.
---

# Real2Sim Calibration

Follow the project-specific workflow in [AGENT_CALIBRATION.md](../../../docs/AGENT_CALIBRATION.md). This skill is stored in the repository-level `.agents/skills` directory. Codex can read it through the project instructions; DeepSeek Harness can use the same path when its runtime loads repository skills and `AGENTS.md`. Automatic loading has not been verified in DSH.

Start by validating the dataset manifest with `./run.sh data --manifest ... --strict`, then inspect active parameters with `./run.sh parameters --config ...`. Use replay for baseline evidence, `evaluate` for strict quality and tolerance checks, and `fit` only with explicit train/validation manifests and evidence-based parameter bounds. Use `curves` for visual residual diagnosis and `sweep` for one-parameter enumeration; neither replaces the strict evaluator or bounded fitter.

Before calibration, read the CSV contract and model limits in [REAL2SIM.md](../../../docs/REAL2SIM.md), then consult [REAL2SIM_GEOMETRY.md](../../../docs/REAL2SIM_GEOMETRY.md) when selecting physical parameters or interpreting transmission residuals. The fit command optimizes scoreable training candidates even when they exceed tolerances, then requires the held-out candidate to meet every configured tolerance and strictly improve on the held-out baseline.

Keep the conclusion proportional to the evidence. A generated command example, simulation reference, passing unit test, or low metric from sparse overlap does not establish agreement with hardware. Parameter bounds and tolerances must be explicit and evidence-based. The checked-in parameter presets are not fitted to real data, and an accepted fit is still a curve-fit candidate rather than proof that the physical model is calibrated.
