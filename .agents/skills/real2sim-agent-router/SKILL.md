---
name: real2sim-agent-router
description: Use when working in this repository and choosing a workflow for recorded data, curve comparison, calibration, model changes, or agent tooling.
---

# Real2Sim Agent Router

Use this skill to choose only the workflow needed for the task. Follow `AGENTS.md` for project constraints; user instructions and project evidence take precedence over generic process advice.

At task start, read `using-superpowers/SKILL.md`, then load only the matching workflow below. For any hardware trace, replay, curve, parameter, or model-structure task, also read `real2sim-calibration/SKILL.md` and follow its project references.

| Task | Workflow |
|---|---|
| Dataset, replay, residual, parameter fit, or model calibration | `real2sim-calibration`; see `docs/AGENT_CALIBRATION.md` and `docs/REAL2SIM_GEOMETRY.md` as relevant. |
| Reproducible software or physics bug | `systematic-debugging`; add `real2sim-calibration` when curves or model behavior are involved. |
| New model mechanism with unresolved alternatives | `brainstorming` only while requirements or mechanism choices are open. |
| Nontrivial code behavior | `test-driven-development`; add `writing-plans` and `executing-plans` only for multi-stage work. |
| Requested review | `receiving-code-review` for incoming feedback; `requesting-code-review` for completed changes. |
| Independent parallel tasks | `dispatching-parallel-agents` for investigation or `subagent-driven-development` for delegated implementation. |
| Skill authoring | `writing-skills`; preserve vendored Superpowers files and put local routing here. |
| Other bounded work | Use the narrowest direct workflow; do not apply unrelated process skills. |

`diagnosing-superpowers` is for Superpowers behavior, not physical residuals. Branch/worktree skills apply only to branch management. For parameter fitting, model-change evidence, and optional tooling, read [calibration and tools](references/calibration-and-tools.md).
