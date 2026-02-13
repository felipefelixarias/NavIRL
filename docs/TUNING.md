# Hyperparameter Tuning

`navirl tune` performs hyperparameter search over ORCA/backend/controller
settings and uses the visual judge + numeric invariants to rank candidates.
It jointly tunes:

- ORCA interaction/time-horizon parameters
- traversability wall-clearance offset (`scene.orca.wall_clearance_buffer_m`)
- deadlock retry budget and offset growth controls
- human/robot waypoint following smoothness parameters

## Command

```bash
python -m navirl tune --suite quick --trials 24 --out out/tune/
```

Retention is enabled by default to avoid unbounded artifact growth (default:
`168` hours, while keeping the latest 3 tuning runs). Override as needed:

```bash
python -m navirl tune --suite quick --trials 24 --retention-hours 48
```

Or set environment variable:

```bash
export NAVIRL_TUNE_TTL_HOURS=48
```

## Inputs

- scenario set (`--suite` or `--scenarios ...`)
- search space (built-in default, or `--search-space path/to/space.yaml`)
- tuning budget (`--trials`)

## Outputs

Run directory:

- `out/tune/tune_<suite>_<timestamp>_<id>/`

Artifacts:

- `trials.jsonl`: one JSON record per trial
- `best_params.json`: best-ranked trial
- `REPORT.md`: top-ranked candidates and reproduction command

## Search-space format

YAML/JSON object mapping dotted config paths to candidate value lists:

```yaml
scene.orca.neighbor_dist: [3.5, 4.5, 5.5]
scene.orca.time_horizon: [2.5, 3.5, 4.5]
scene.orca.wall_clearance_buffer_m: [0.0, 0.005, 0.015, 0.03]
evaluation.deadlock_resample_attempts: [4, 6, 8]
evaluation.traversability_offset_step: [0.003, 0.005, 0.008]
humans.controller.params.velocity_smoothing: [0.25, 0.4, 0.55]
robot.controller.params.velocity_smoothing: [0.4, 0.55, 0.7]
```

## Objective function

Trial ranking uses a weighted score with hard penalties for:

- deadlocks / prolonged stops
- wall or obstacle collisions
- high wall-proximity and jitter
- invariant failures and visual judge failures

It also includes a retry-cost penalty so stable parameter sets that work
without repeated resampling/offset bumps rank higher.
