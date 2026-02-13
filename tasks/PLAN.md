# NavIRL Implementation Plan

Date: 2026-02-13

## Milestone Order

1. `M0` Current-state architecture + task planning
2. `M1` Phase 0 foundation (package, CLI, scenario/metrics specs)
3. `M2` Phase 1 thin vertical slice (run/eval/view end-to-end)
4. `M3` Verify gate (quick/full suites + invariants + reports)
5. `M4` Docs/tests hardening and final validation

## Dependency Graph

- `M1` depends on `M0`.
- `M2` depends on `M1`:
  - controllers require core types/registry
  - runner requires scenario validation and backend
  - eval/view require log format from runner
- `M3` depends on `M2`:
  - verify consumes canonical scenarios and run/eval/view artifacts
- `M4` depends on `M3`:
  - documentation must match implemented behavior
  - final tests run against verify gate and CLI

## Detailed Execution Sequence

### Step 1: Build foundation skeleton
- Create `navirl` package and module tree.
- Add CLI with subcommands:
  - `run`
  - `run-batch`
  - `eval`
  - `view`
  - `verify`
  - `validate`
- Add deterministic seed utility and plugin registry.

### Step 2: Scenario + metrics specs
- Implement `ScenarioSpec v1` schema and loader/validator.
- Add curated scenario library (6 canonical verify scenarios).
- Implement standard metrics functions and metric report writer.

### Step 3: Always-works backend and controllers
- Implement `grid2d` backend wrapping INDOORCA environment + ORCA wrapper.
- Implement human controllers:
  - `orca` baseline
  - `orca_plus` with ablation flags
  - `scripted`
  - `replay`
- Implement robot baseline controller:
  - `baseline_astar`

### Step 4: End-to-end run/eval/view
- Implement deterministic runner generating trace bundles:
  - `scenario.yaml`
  - `state.jsonl`
  - `events.jsonl`
  - `summary.json`
  - `frames/*.png` and optional `video.mp4`
- Implement evaluator aggregation over one or many logs.
- Implement viewer replay with overlays and timeline index.

### Step 5: Verify gate
- Implement `verify.runner` with suites:
  - `quick`: tests + canonical scenarios + frame render + judge
  - `full`: all quick checks + stricter thresholds + required video
- Implement numeric validators and strict judge JSON output.
- Generate `out/verify/<suite>/REPORT.md` with reproduction commands.

### Step 6: Documentation and agent workflow
- Update:
  - `README.md` quickstart
  - `AGENTS.md`
  - target architecture/spec docs

## Exit Criteria

- `pytest -q` passes.
- `python -m navirl verify --suite quick` exits `0`.
- `python -m navirl verify --suite full` exits `0` if full suite runs locally.
- Required docs are present and aligned with implemented behavior.

