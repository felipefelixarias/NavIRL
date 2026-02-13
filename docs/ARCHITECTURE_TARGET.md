# NavIRL Target Architecture

NavIRL is the standalone modular research-engineering toolkit for indoor social
navigation simulation/evaluation.

## Core Packages

- `navirl/core`
  - `env.py`: `SceneBackend` interface
  - `types.py`: `AgentState`, `Action`, `Observation`, `EpisodeLog`
  - `registry.py`: plugin registration for backends/controllers
  - `seeds.py`: deterministic seed controls
- `navirl/backends/grid2d`
  - always-works CPU backend using occupancy grids + RVO2 stepping
- `navirl/humans`
  - `orca`, `orca_plus`, `scripted`, `replay`, `policies` placeholder
- `navirl/robots`
  - robot controller interface + `baseline_astar`
- `navirl/scenarios`
  - schema, loader, validator, scenario library
- `navirl/metrics`
  - standard metric collectors
- `navirl/logging`
  - JSONL trace bundle writer
- `navirl/viz`
  - frame rendering + replay viewing
- `navirl/verify`
  - victory gate runner, numeric validators, visual judge

## Runtime Data Flow

1. `navirl run <scenario.yaml>` loads ScenarioSpec and validates schema.
2. Registry resolves backend and controller plugins by type string.
3. Backend resolves map metadata (`pixels_per_meter`, dimensions, optional downsample).
4. Runner executes deterministic simulation (`seed`, `dt`, `steps`).
5. Logger writes trace bundle:
   - `scenario.yaml`
   - `state.jsonl`
   - `events.jsonl`
   - `summary.json`
   - `frames/*.png` and optional `video.mp4`
6. `navirl eval` computes standardized metrics across one or many logs.
7. `navirl view` re-renders overlay frames/video for debugging.
8. `navirl verify` executes tests + feasibility checks + invariants + judge.

## Plugin Interfaces

### Scene Backend

`SceneBackend` defines:

- `add_agent(...)`
- `set_preferred_velocity(...)`
- `step()`
- `get_position(...)`
- `get_velocity(...)`
- `shortest_path(...)`
- `sample_free_point()`
- `check_obstacle_collision(...)`
- `world_to_map(...)`
- `map_image()`
- `map_metadata()`

### Human Controller

`HumanController` defines:

- `reset(human_ids, starts, goals)`
- `step(step, time_s, dt, states, robot_id, emit_event) -> {agent_id: Action}`

### Robot Controller

`RobotController` defines:

- `reset(robot_id, start, goal, backend)`
- `step(step, time_s, dt, states, emit_event) -> Action`

## Determinism

NavIRL sets deterministic state from scenario seed:

- Python RNG seed
- NumPy RNG seed
- deterministic scenario and controller initialization

Deterministic guarantees target repeatability of trajectory/log artifacts for the
same platform, dependencies, and scenario/config.

## Repository Strategy

- Keep the native ORCA core (`src/*`, `rvo2`) as the low-level engine.
- Expose all user-facing functionality via `navirl` package and CLI.
- Maintain explicit backend/controller/logging interfaces for extensibility.
