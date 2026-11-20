# Shared Agent Skills

Project skills are stored in `.agents/skills/` so coding agents working from this checkout can use the same instructions. The repository-level [agent router](../.agents/skills/real2sim-agent-router/SKILL.md) is the entrypoint for choosing the right workflow.

## Installed skills

- `real2sim-calibration` is maintained for this project. Its entrypoint links to the detailed workflow and model references under `docs/`.
- `real2sim-agent-router` is maintained for this project. It maps common task types to project docs and only the relevant general-purpose workflows.
- The 15 general-purpose [Superpowers](https://github.com/obra/superpowers) skills and their supporting files are vendored from version `7.0.2`, commit `22f0b463e7a5e579904dd6211a520321035526f7`.

At task start, read `real2sim-agent-router/SKILL.md`; it directs agents to the Superpowers bootstrap and matching skill only when relevant. For real2sim calibration work, also read `real2sim-calibration/SKILL.md` and the references it identifies. Do not load every skill for unrelated tasks.

## Harness discovery

This repository's `AGENTS.md` directs agents to the router and applicable skills. Codex can read these project skills from `.agents/skills/`. DeepSeek Harness can use the same project path and instructions when its runtime loads repository skills and `AGENTS.md`; this checkout does not include a DeepSeek Harness plugin or a runtime integration test, so automatic startup loading must be confirmed in that harness.

These files are a project-shared skill installation, not a Codex Marketplace plugin installation. The upstream Superpowers license is retained at `third_party/licenses/superpowers-LICENSE.txt`.
