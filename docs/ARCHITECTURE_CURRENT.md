# INDOORCA Legacy Architecture Snapshot (2026-02-13)

This document captures the pre-NavIRL structure of the former INDOORCA code.
It is kept as a historical migration record.

## Repository Summary

- Native ORCA core and bindings:
  - `src/*.cpp`, `src/*.h`
  - `src/rvo2.pyx`
  - built as Python extension module `rvo2`
- Python research package:
  - `indoorca/environment/core.py`
  - `indoorca/simulator/orca.py`
  - `indoorca/simulator/core.py`
  - `indoorca/visualization/core.py`
- Utility/demo scripts:
  - `scripts/run_ped_sim.py`
  - `example.py`
- Tests:
  - `tests/test_environment_smoke.py`
  - `tests/test_coordinate_consistency.py`
  - `tests/test_visualization_rendering.py`
  - `tests/test_run_ped_sim_utils.py`
  - `tests/test_imports.py`

## Current Runtime Flow

Typical flow (from `scripts/run_ped_sim.py`):

1. Load/threshold map image into binary occupancy grid.
2. Create `MultiAgentSim(name, map, num_agents)` from
   `indoorca/simulator/core.py`.
3. `MultiAgentSim` constructs:
   - `Environment` (`indoorca/environment/core.py`)
   - `IndoorORCASim` wrapper over `rvo2.PyRVOSimulator`
4. Environment processing:
   - extract obstacle polygons from occupancy map
   - convert obstacle vertices to world coordinates
   - build traversability graph (`networkx`) over free cells
5. Agent setup:
   - ORCA agent IDs added in `IndoorORCASim`
   - starts/goals sampled in world coordinates
   - per-agent waypoint lists from A* shortest path on traversability graph
6. Step loop:
   - compute preferred velocity toward next waypoint
   - `rvo2` advances with collision avoidance
   - pop waypoint on proximity threshold
7. Output:
   - in-memory trajectories list (`IndoorORCASim.trajectories`)
   - visualization via `Visualizer` (trajectory plot or GIF/video)

## Key Components

### Environment Layer (`indoorca/environment/core.py`)

- Responsibilities:
  - occupancy preprocessing (`_trim_map`)
  - obstacle polygon extraction (`skimage.measure`, `shapely`)
  - obstacle rasterization to obstacle map (`cv2.fillPoly`)
  - traversability graph build on free cells (`networkx.Graph`)
  - map/world coordinate transforms
  - shortest path routing via A* (`networkx.astar_path`)
- Observations:
  - strong coupling to globals in `indoorca/__init__.py`
  - several duplicated transform helpers (`_map_to_world`, `map_to_world`)
  - debug prints and mixed conventions in some methods

### ORCA Wrapper (`indoorca/simulator/orca.py`)

- `IndoorORCASimConfig` stores ORCA parameters (dt, neighbor radius, horizons,
  radius, max speed).
- `IndoorORCASim` wraps `rvo2.PyRVOSimulator` with methods for agents,
  obstacles, stepping, and trajectory capture.
- This is the thinnest stable seam to preserve during migration.

### Multi-Agent Orchestration (`indoorca/simulator/core.py`)

- `MultiAgentSim` coordinates environment, ORCA sim, starts/goals, waypoints,
  and run loop.
- Includes basic goal swapping and waypoint-following logic.
- Contains partially implemented or legacy methods (e.g., backoff detection
  helpers) and mixed concerns (sampling, planning, stepping, episode control).

### Visualization (`indoorca/visualization/core.py`)

- `Visualizer` supports static trajectory plots and animated GIF rendering.
- Can overlay map background, draw robot/pedestrians, timestamps, and paths.
- Primarily presentation-focused, not a replay debugger with event/state
  inspection.

### Packaging and Build

- Modern packaging exists via `pyproject.toml`.
- Native extension build uses `setup.py` custom `build_ext` + CMake.
- Install target currently exposes package name/module `indoorca`.

## What To Keep vs Replace

### Keep (directly or with minimal cleanup)

- `src/` ORCA C++ core and `rvo2` binding pipeline.
- `IndoorORCASim` wrapper API shape as a backend primitive.
- Environment occupancy-to-obstacle extraction and traversability graph logic.
- Existing tests around imports, coordinates, smoke simulation, and rendering.

### Wrap / Adapt (high-value, but move behind NavIRL interfaces)

- `Environment` into `navirl.backends.grid2d` backend abstraction.
- `MultiAgentSim` logic split across:
  - backend stepping
  - human controller(s)
  - robot controller(s)
  - episode runner
- `Visualizer` rendering primitives reused in `navirl.viz.render` and
  `navirl.viz.viewer`.

### Replace / Refactor

- Monolithic simulation orchestration with explicit plugin interfaces
  (backend/controller/metrics).
- Ad-hoc data flow with typed scenario spec, metric spec, and JSONL trace
  bundles.
- Debug prints and implicit defaults with structured config + deterministic
  seed handling.
- One-off script entrypoints with stable CLI (`navirl run/eval/view/verify`).

## Current Gaps Relative to NavIRL Target

- No `navirl` package or plugin registry.
- No standardized `ScenarioSpec` or schema validation.
- No standardized metrics library/reporting pipeline.
- No trace bundle layout (`state.jsonl`, `events.jsonl`, `summary.json`).
- No deterministic verification gate with invariant checks and report output.
- No unified CLI commands for run/eval/view/verify.

## Migration Notes (Historical)

- This snapshot informed the standalone NavIRL implementation.
- Current repository runtime surface is now `navirl/*` with no `indoorca`
  package dependency.
