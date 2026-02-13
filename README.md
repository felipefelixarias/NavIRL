# NavIRL

NavIRL is a standalone, agent-driven research engineering toolkit for indoor
social navigation simulation, evaluation, and debugging.

It is designed for researchers and engineers who need fast iteration,
reproducible experiments, structured scenario definitions, and strict
end-to-end validation before claiming results.

## Research Origin

NavIRL is seeded by Felipe Felix Arias's original indoor social-navigation
research program, including the INDOORCA simulator and master's-thesis work
that initiated this line of development.

- Research home: https://felipefelixarias.github.io/
- Thesis artifact in this repo: `research/ARIAS-THESIS-2023.pdf`
- Legacy provenance snapshot: `docs/ARCHITECTURE_CURRENT.md`

This repository now serves as the standalone NavIRL codebase, while preserving
that lineage as explicit documentation.

## Why This Exists

Most social-navigation projects fail from tooling gaps, not model novelty.
NavIRL prioritizes:

- deterministic simulation runs
- stable scenario and metrics standards
- transparent trace bundles for debugging
- agent-driven development with enforceable verification gates

## What NavIRL Provides

- modular indoor simulator backend (`grid2d` + ORCA/RVO2)
- interchangeable human controllers (`orca`, `orca_plus`, `scripted`, `replay`)
- robot baseline controller (`baseline_astar`)
- `ScenarioSpec` for reproducible scenario definitions
- `MetricsSpec` for stable, comparable evaluation outputs
- replay rendering with debug overlays
- an agentic verification gate with numeric + visual checks

## Units and Measurements

NavIRL uses **meters** for all world coordinates and dynamic parameters
(positions, radii, speeds, ORCA distances).

Map images are converted to meters via `scene.map.pixels_per_meter` (or
`meters_per_pixel`).

- Built-in maps default to `100 px/m`.
- Path maps (`source: path`) require an explicit scale.
- Optional `scene.map.downsample` reduces simulation resolution while preserving
  world dimensions (effective `pixels_per_meter` is scaled automatically).
- ORCA map-distance parameters can be declared in pixels with:
  `scene.orca.units: pixels` (auto-converted to meters).

Builtin map dimensions (default scale):

| Map | Pixels | Pixels/Meter | Size (m) |
|---|---:|---:|---:|
| hallway | 360x220 | 100 | 3.60 x 2.20 |
| doorway | 320x240 | 100 | 3.20 x 2.40 |
| kitchen | 340x300 | 100 | 3.40 x 3.00 |
| group | 360x260 | 100 | 3.60 x 2.60 |
| comfort | 360x260 | 100 | 3.60 x 2.60 |
| apartment_micro | 360x300 | 100 | 3.60 x 3.00 |

## Quickstart

### 1) Install

```bash
python -m pip install -U pip
python -m pip install -e .[dev]
```

### 2) Run a scenario

```bash
python -m navirl run navirl/scenarios/library/hallway_pass.yaml --out logs/
```

Main thesis-map demo:

```bash
python -m navirl run navirl/scenarios/library/wainscott_main_demo.yaml --out logs/
```

### 3) Evaluate logs

```bash
python -m navirl eval logs/**/state.jsonl --report out/eval/
```

### 4) Replay and debug

```bash
python -m navirl view logs/<run_id>/bundle/state.jsonl --out out/view/
```

### 5) Run the victory gate

```bash
python -m navirl verify --suite quick
python -m navirl verify --suite full
```

### 6) Tune ORCA and controller hyperparameters

```bash
python -m navirl tune --suite quick --trials 24 --out out/tune/
```

Use custom scenarios:

```bash
python -m navirl tune --scenarios navirl/scenarios/library/hallway_pass.yaml navirl/scenarios/library/doorway_token_yield.yaml --trials 40
```

### Artifact Retention (TTL)

To avoid unbounded local artifact growth, NavIRL prunes old run directories by
default:

- `run` / `run-batch`: default TTL `168` hours (7 days)
- `tune`: default TTL `168` hours (7 days), keeping the latest 3 tuning runs

You can override TTL per command:

```bash
python -m navirl run ... --retention-hours 48
python -m navirl tune ... --retention-hours 24
```

Or via environment variables:

- `NAVIRL_LOG_TTL_HOURS` for `run` / `run-batch`
- `NAVIRL_TUNE_TTL_HOURS` for `tune`

Exit codes:

- `0`: PASS
- `10`: FAIL (tests, invariants, or judge)
- `20`: NEEDS_HUMAN_REVIEW (for unavailable external judge mode)

## Agent-Driven Research Workflow

NavIRL is intentionally built for AI-agent execution loops:

- agent modifies code and docs in small, reviewable increments
- agent runs deterministic suites and scenario canaries
- agent renders trace artifacts for semantic/visual sanity checks
- agent produces explicit pass/fail reports with repro commands

This is enforced through `navirl verify`, not treated as optional process.

## Hyperparameter Tuning Loop

`navirl tune` runs repeated scenario rollouts with sampled hyperparameters and
scores each trial using:

- numeric invariants
- visual judge confidence/pass status
- stability/comfort metrics (collisions, intrusion, oscillation, jerk, success)
- deadlock/stuck penalties and retry-cost penalties
- traversability offset tuning (`wall_clearance_buffer_m`) and deadlock retry controls

Artifacts are written under `out/tune/<run_id>/`:

- `trials.jsonl` (all trials + per-scenario results)
- `best_params.json` (best override set)
- `REPORT.md` (ranked summary + reproducibility command)

## End-to-End Visual Verification

`navirl verify` executes:

- unit and smoke tests
- scenario feasibility checks (clearance/radius/path viability + bottleneck risk suggestions)
- canonical deterministic scenario runs
- numeric invariant validators
- frame/video rendering for trace bundles
- visual judge output in strict JSON format

Generated report:

- `out/verify/<suite>/REPORT.md`

Per-scenario artifacts:

- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/`

## Canonical Verify Scenarios

- `hallway_pass`
- `doorway_token_yield`
- `kitchen_congestion`
- `group_cohesion`
- `robot_comfort_avoidance`
- `routine_cook_dinner_micro`

## Showcase GIFs

Showcase renders use cinematic overlays, doubled-horizon rollouts, and accelerated playback.

### Main demo (`wainscott_main_demo`, thesis floorplan map)

![Wainscott Main Demo](docs/assets/showcase/wainscott_main_demo.gif)

MP4: [`docs/assets/showcase/wainscott_main_demo.mp4`](docs/assets/showcase/wainscott_main_demo.mp4)

### Hallway passing (`hallway_pass`)

![Hallway Pass](docs/assets/showcase/hallway_pass.gif)

MP4: [`docs/assets/showcase/hallway_pass.mp4`](docs/assets/showcase/hallway_pass.mp4)

### Doorway token etiquette (`doorway_token_yield`)

![Doorway Token Yield](docs/assets/showcase/doorway_token_yield.gif)

MP4: [`docs/assets/showcase/doorway_token_yield.mp4`](docs/assets/showcase/doorway_token_yield.mp4)

### Kitchen congestion (`kitchen_congestion`)

![Kitchen Congestion](docs/assets/showcase/kitchen_congestion.gif)

MP4: [`docs/assets/showcase/kitchen_congestion.mp4`](docs/assets/showcase/kitchen_congestion.mp4)

### Group cohesion (`group_cohesion`)

![Group Cohesion](docs/assets/showcase/group_cohesion.gif)

MP4: [`docs/assets/showcase/group_cohesion.mp4`](docs/assets/showcase/group_cohesion.mp4)

### Routine micro-scene (`routine_cook_dinner_micro`)

![Routine Cook Dinner Micro](docs/assets/showcase/routine_cook_dinner_micro.gif)

MP4: [`docs/assets/showcase/routine_cook_dinner_micro.mp4`](docs/assets/showcase/routine_cook_dinner_micro.mp4)

### Robot comfort avoidance (`robot_comfort_avoidance`)

![Robot Comfort Avoidance](docs/assets/showcase/robot_comfort_avoidance.gif)

MP4: [`docs/assets/showcase/robot_comfort_avoidance.mp4`](docs/assets/showcase/robot_comfort_avoidance.mp4)

## Trace Bundle Format

Each run writes:

- `scenario.yaml`
- `state.jsonl`
- `events.jsonl`
- `summary.json`
- `frames/*.png`
- `frames/video.mp4` (optional or suite-dependent)

See: `docs/DATAFORMAT_SPEC.md`

## Documentation Map

- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `docs/ARCHITECTURE_TARGET.md`
- `docs/SCENARIO_SPEC.md`
- `docs/METRICS_SPEC.md`
- `docs/VERIFY_GATE.md`
- `docs/TUNING.md`
- `docs/RESEARCH_CONTEXT.md`
- `docs/GETTING_STARTED.md`

## Repository Layout

- `navirl/` core toolkit
- `src/` RVO2 C++ core + Cython bindings
- `tests/` NavIRL test suite
- `tasks/` backlog and implementation plan
- `research/` thesis and related artifacts

## Reproducibility Commitments

- scenario-level seeds recorded in run artifacts
- deterministic runner entrypoints and specs
- verification reports with reproduction commands
- explicit thresholds and invariant checks

## License

Apache 2.0. See `LICENSE`.
