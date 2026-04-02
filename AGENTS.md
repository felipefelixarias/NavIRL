# AI Agent Guidelines

This repository is designed to be primarily developed and maintained by AI
agents. The goal is to keep changes reproducible, reviewable, and aligned with
research best practices.

## Agent responsibilities
- Keep PRs small, focused, and well-scoped.
- Record key assumptions and limitations in the PR description.
- Add tests when behavior changes; explain when not possible.
- Update documentation for any public API or workflow changes.
- Avoid introducing large binary artifacts without discussion.

## Required annotations
Every AI-authored PR should include:
- A concise summary of changes
- Test plan and results
- Any breaking changes and migration notes
- Reproducibility notes (seeds, configs, data sources)

## Auto-merge behavior
AI agents may auto-merge PRs that:
- Pass all CI checks
- Have no unresolved review comments
- Include tests or documented rationale for omission

## Safety and data handling
- Do not include credentials, private datasets, or personally identifiable
  information in commits.
- Prefer synthetic or openly licensed data for tests and examples.

## NavIRL workflow

### Install and test
- Install editable package:
  - `python3 -m venv .venv && ./.venv/bin/python -m pip install -e .[dev]`
- Run tests:
  - `./.venv/bin/pytest -q`

### Run smoke simulation
- Run a canonical scenario:
  - `./.venv/bin/python -m navirl run navirl/scenarios/library/hallway_pass.yaml --out logs/`
- Run thesis floorplan demo:
  - `./.venv/bin/python -m navirl run navirl/scenarios/library/wainscott_main_demo.yaml --out logs/`

### Run verify gate
- Quick suite (required before claiming task completion):
  - `./.venv/bin/python -m navirl verify --suite quick`
- Full suite (required when changing core sim/controllers/planners/metrics):
  - `./.venv/bin/python -m navirl verify --suite full`

### Tune ORCA/social-nav hyperparameters
- Default quick tuning sweep:
  - `./.venv/bin/python -m navirl tune --suite quick --trials 24 --out out/tune/`
- Custom scenario set:
  - `./.venv/bin/python -m navirl tune --scenarios navirl/scenarios/library/hallway_pass.yaml navirl/scenarios/library/doorway_token_yield.yaml --trials 40`
- Notes:
  - tuner jointly optimizes ORCA params, traversability wall-clearance offset,
    and deadlock-retry controls using numeric invariants + visual judge score.

### Plugin conventions
- Register plugins in `navirl/plugins.py`.
- New backends implement `navirl/core/env.py:SceneBackend`.
- Human controllers implement `navirl/humans/base.py:HumanController`.
- Robot controllers implement `navirl/robots/base.py:RobotController`.
- Controllers must be selectable by `controller.type` in ScenarioSpec.
- Behavior-changing plugins should include at least one deterministic test and
  one scenario/library example.

### Map units
- Path maps (`scene.map.source: path`) must define map scale with
  `scene.map.pixels_per_meter` or `scene.map.meters_per_pixel`.
- Optional `scene.map.downsample` is supported; world coordinates remain meters.
