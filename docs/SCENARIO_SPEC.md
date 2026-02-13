# ScenarioSpec v1

Scenario files are YAML documents validated by `navirl/scenarios/schema.json`.

## Required Top-Level Fields

- `id`: stable scenario identifier
- `scene`: backend and map configuration
- `seed`: deterministic seed
- `horizon`: simulation horizon (`steps`, `dt`)
- `humans`: human population and controller config
- `robot`: robot controller and mission config

## Scene

```yaml
scene:
  backend: grid2d
  id: hallway_demo
  map:
    source: builtin   # builtin | path
    id: hallway       # required for builtin
    # path: data/maps/custom.png  # required for source=path
    # pixels_per_meter: 100.0     # required for source=path unless meters_per_pixel is set
    # meters_per_pixel: 0.01
    # downsample: 2.0             # optional; preserves world dimensions
```

Unit conventions:

- all coordinates in scenario files are meters
- controller radii/speeds are meters and meters/second
- map image scale is controlled by `pixels_per_meter` (or `meters_per_pixel`)
- if `scene.orca.units: pixels`, ORCA distance/speed params are converted to meters
- starts/goals are auto-sanitized at run time so each anchor has:
  - no obstacle within one diameter (`2 * radius`)
  - no other start (for starts) or goal (for goals) within one diameter
- if a scenario is geometrically too constrained to satisfy this, NavIRL
  retries with deterministic resampling; if still unresolved, run fails with a
  geometry/clearance error.

## Horizon

```yaml
horizon:
  steps: 120
  dt: 0.1
```

## Humans

```yaml
humans:
  controller:
    type: orca        # orca | orca_plus | scripted | replay | policy
    params: {}
  count: 4
  starts:
    - [-1.2, 0.0]
  goals:
    - [1.2, 0.0]
  radius: 0.16
  max_speed: 0.75
  groups:
    - [1, 2]
```

## Robot

```yaml
robot:
  controller:
    type: baseline_astar
    params:
      replan_interval: 16
  start: [-1.4, 0.0]
  goal: [1.4, 0.0]
  radius: 0.18
  max_speed: 0.9
```

## Evaluation thresholds

```yaml
evaluation:
  intrusion_delta: 0.45
  max_speed: 1.25
  max_accel: 4.5
  teleport_thresh: 1.0
  deadlock_seconds: 4.0
  deadlock_speed_thresh: 0.015
  max_agent_stop_seconds: 8.0
  stop_speed_thresh: 0.02
  near_wall_buffer: 0.02
  max_wall_proximity_fraction: 0.14
  max_heading_flip_rate: 0.82
  jitter_speed_thresh: 0.06
  min_robot_progress: 0.1
  wall_clearance_buffer: 0.0
  enforce_wall_clearance_buffer: false
  resample_on_deadlock: true
  deadlock_resample_attempts: 4
  fail_on_deadlock: true
  auto_tune_traversability_offset: true
  traversability_offset_step: 0.005
  traversability_offset_max: 0.04
  expected_high_interaction: false
```

Render config also supports `playback_speed` and `trail_length` for cinematic output:

```yaml
render:
  enabled: true
  fps: 12
  video: true
  playback_speed: 1.85
  trail_length: 64
```

## Optional routine

```yaml
routine:
  name: cook_dinner
  steps:
    - go_to_kitchen
    - prep_counter
```

## Validation

```bash
python -m navirl validate navirl/scenarios/library/hallway_pass.yaml
```
