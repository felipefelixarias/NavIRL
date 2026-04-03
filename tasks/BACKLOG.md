# NavIRL Backlog

Date: 2026-02-13
Scope: Transform INDOORCA into NavIRL toolkit with a working vertical slice.

## Phase 0: Foundation

### T0.1 Architecture recon and current-state doc (1h)
- Description: inventory the existing INDOORCA internals and identify keep vs replace seams.
- Acceptance criteria:
  - `docs/ARCHITECTURE_CURRENT.md` exists.
  - File includes runtime data flow and keep/replace table.
- Verification commands:
  - `sed -n '1,220p' docs/ARCHITECTURE_CURRENT.md`

### T0.2 Create NavIRL package skeleton (2h)
- Description: introduce `navirl/` package with core module layout and CLI entry.
- Acceptance criteria:
  - `navirl/cli.py` exists and registers `run/eval/view/verify/validate`.
  - `navirl/core/{types.py,env.py,registry.py,seeds.py}` exist.
  - `python -m navirl --help` works.
- Verification commands:
  - `python -m navirl --help`
  - `python -m navirl run --help`

### T0.3 ScenarioSpec v1 + validator (2h)
- Description: define JSON schema and YAML loader/validator for deterministic scenarios.
- Acceptance criteria:
  - `navirl/scenarios/schema.json`, `load.py`, `validate.py` exist.
  - `navirl validate <scenario.yaml>` exits 0 for valid scenario and non-zero for invalid.
  - `docs/SCENARIO_SPEC.md` documents required fields and examples.
- Verification commands:
  - `python -m navirl validate navirl/scenarios/library/hallway_pass.yaml`

### T0.4 Metrics foundation + docs (2h)
- Description: implement metric interfaces and standard metrics computation from state logs.
- Acceptance criteria:
  - `navirl/metrics/base.py` and `navirl/metrics/standard.py` exist.
  - `docs/METRICS_SPEC.md` lists stable metric names and formulas.
  - `navirl eval <state.jsonl>` writes a report.
- Verification commands:
  - `python -m navirl eval output/**/*.jsonl --report out_eval`

## Phase 1: Thin Vertical Slice

### T1.1 Grid2D backend wrapper over INDOORCA ORCA (4h)
- Description: implement one always-works backend using occupancy map + ORCA.
- Acceptance criteria:
  - `navirl/backends/grid2d/` can initialize map, obstacles, and ORCA agents.
  - deterministic stepping from seed.
  - no crashes on canonical hallway scenario.
- Verification commands:
  - `python -m navirl run navirl/scenarios/library/hallway_pass.yaml --out out_runs`

### T1.2 Human/Robot controller interfaces + baselines (4h)
- Description: add controller interfaces and baseline implementations.
- Acceptance criteria:
  - Human controllers: `orca`, `orca_plus`, `scripted`, `replay`.
  - Robot controller: `baseline_astar`.
  - Scenario can select controller by type string.
- Verification commands:
  - `python -m navirl run navirl/scenarios/library/doorway_token_yield.yaml --out out_runs`

### T1.3 Logging + trace bundle output (2h)
- Description: output run artifacts in canonical bundle format.
- Acceptance criteria:
  - bundle folder contains `scenario.yaml`, `state.jsonl`, `events.jsonl`, `summary.json`.
  - optional frames/video generation toggles are functional.
- Verification commands:
  - `find out_runs -maxdepth 4 -type f | sort`

### T1.4 Viewer and renderer (3h)
- Description: replay logs with overlays for debugging.
- Acceptance criteria:
  - `navirl view logs/episode_0001.jsonl` produces rendered output.
  - overlays include IDs, velocity arrows, and behavior labels.
- Verification commands:
  - `python -m navirl view out_runs/*/bundle/state.jsonl --out out_view`

## Phase 2: Verify Gate

### T2.1 Numeric invariant validators (3h)
- Description: implement hard checks for physical/log consistency.
- Acceptance criteria:
  - validators for wall penetration, teleport, speed/accel bounds, deadlock bounds.
  - doorway token exclusivity check for doorway scenarios.
- Verification commands:
  - `python -m navirl verify --suite quick`

### T2.2 Verify runner and report (3h)
- Description: implement quick/full suites, canonical scenario execution, artifact report.
- Acceptance criteria:
  - `navirl verify --suite quick|full` exists.
  - exit codes follow contract: `0`, `10`, `20`.
  - report created at `out/verify/<suite>/REPORT.md`.
- Verification commands:
  - `python -m navirl verify --suite quick`
  - `python -m navirl verify --suite full`

### T2.3 Visual judge interface (2h)
- Description: enforce strict JSON judge output using rendered frames + summary table.
- Acceptance criteria:
  - `navirl/verify/judge.py` emits strict JSON with `overall_pass`, `confidence`, `violations`.
  - verify gate uses judge output for pass/fail.
- Verification commands:
  - `python -m navirl verify --suite quick`

## Phase 3: Docs + DX

### T3.1 Architecture/spec docs (2h)
- Description: document final plugin architecture and data flow.
- Acceptance criteria:
  - `docs/ARCHITECTURE_TARGET.md`, `docs/GETTING_STARTED.md`,
    `docs/SCENARIO_SPEC.md`, `docs/METRICS_SPEC.md`, `docs/DATAFORMAT_SPEC.md` exist.
- Verification commands:
  - `ls docs`

### T3.2 README and AGENTS guide refresh (2h)
- Description: update top-level usage and contributor/agent workflows.
- Acceptance criteria:
  - README includes `navirl run/eval/view` and `navirl verify`.
  - `AGENTS.md` includes tests, smoke run, verify commands, plugin conventions.
- Verification commands:
  - `sed -n '1,260p' README.md`
  - `sed -n '1,260p' AGENTS.md`

### T3.3 New tests for NavIRL flows (3h)
- Description: add deterministic smoke/spec/orca_plus tests.
- Acceptance criteria:
  - `tests/test_smoke.py`, `tests/test_scenarios.py`, `tests/test_orca_plus_doorway.py` exist.
  - all tests pass in local environment.
- Verification commands:
  - `pytest -q`
