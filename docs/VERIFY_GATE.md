# Verify Gate

`navirl verify` is the local victory gate for experiment correctness.

## Commands

Quick suite:

```bash
python -m navirl verify --suite quick
```

Full suite:

```bash
python -m navirl verify --suite full
```

## Exit codes

- `0`: PASS
- `10`: FAIL
- `20`: NEEDS_HUMAN_REVIEW

## What verify runs

1. `pytest -q`
2. canonical deterministic scenarios
3. scenario feasibility analysis (geometry/radius/path viability)
4. numeric invariants
5. rendering artifacts
6. visual judge with strict JSON output

## Numeric invariant checks

- units metadata consistency (`pixels_per_meter`/`meters_per_pixel`)
- start/goal anchor layout (`anchor_layout`)
  - no obstacle within one diameter of any start/goal
  - no other start within one diameter (and similarly for goals)
- scenario feasibility (`scenario_feasibility`) with fix suggestions for:
  - invalid start/goal placement for given radii
  - radius-inflated path blockage
  - bidirectional bottleneck deadlock risk warnings
- no wall penetration (all agents)
- optional wall-clearance buffer threshold
- wall-proximity fraction
- no teleport jumps
- speed and acceleration bounds
- motion jitter bound (`motion_jitter`)
- doorway token exclusivity (doorway scenarios)
- deadlock bounds (always enforced)
- prolonged-stop detection (`agent_stop_duration`)
- robot progress floor (horizon-aware)
- log/render frame sync

Scenario evaluation flags supported by the gate:

- `expected_high_interaction`
- `min_robot_progress`
- `wall_clearance_buffer`
- `enforce_wall_clearance_buffer`
- `near_wall_buffer`
- `max_wall_proximity_fraction`
- `max_heading_flip_rate`
- `jitter_speed_thresh`
- `deadlock_speed_thresh`
- `max_agent_stop_seconds`
- `stop_speed_thresh`
- `resample_on_deadlock`
- `deadlock_resample_attempts`
- `fail_on_deadlock`
- `auto_tune_traversability_offset`
- `traversability_offset_step`
- `traversability_offset_max`

## Visual judge output

Judge outputs strict JSON with:

- `overall_pass`
- `confidence`
- `violations[]`
- `status`
- `judge_type`

The default heuristic judge (`heuristic_rigorous`) also validates:

- directional-arrow coverage
- trajectory-trail density
- overlay text clutter
- frame detail and motion quality

## Generated artifacts

Per suite:

- `out/verify/<suite>/REPORT.md`
  - includes failing-check evidence and suggested fixes when feasibility checks fail

Per scenario:

- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/scenario.yaml`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/state.jsonl`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/events.jsonl`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/summary.json`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/invariants.json`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/judge.json`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/frames/*.png`
- `out/verify/<suite>/verify_<suite>_<scenario>/bundle/frames/video.mp4`

## Reproduction pattern

```bash
pytest -q
python -m navirl verify --suite quick
python -m navirl verify --suite full
```
