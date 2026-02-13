# MetricsSpec v1

NavIRL emits stable metric keys from trace bundles.

## Collision metrics

- `collisions_agent_agent`
  - count of timesteps where center distance `< r_i + r_j` for any agent pair.
- `collisions_agent_obstacle`
  - count of agent states in obstacle cells or outside map bounds.

## Distance metrics

- `min_dist_robot_human_min`
- `min_dist_robot_human_mean`
- `min_dist_robot_human_p05`
- `min_dist_human_human_min`
- `min_dist_human_human_mean`
- `min_dist_human_human_p05`

## Comfort/safety metric

- `intrusion_rate`
  - fraction of timesteps where robot-human minimum distance `< intrusion_delta`.

## Stagnation metric

- `deadlock_count`
  - number of agents with low-speed streak (`speed < 0.05`) longer than
    `deadlock_seconds` while still away from goal.

## Motion quality metrics

- `oscillation_score`
  - heading sign-flip rate from agent heading-delta sequence.
- `jerk_proxy`
  - mean norm of finite-difference jerk (`d(accel)/dt`).

## Task metrics

- `path_length_robot`
- `time_to_goal_robot`
- `success_rate` (`1.0` if robot reaches goal tolerance, else `0.0`)

## Map metadata metrics

- `map_pixels_per_meter`
- `map_meters_per_pixel`
- `map_width_m`
- `map_height_m`

## Evaluation command

```bash
python -m navirl eval logs/**/state.jsonl --report out/eval/
```
