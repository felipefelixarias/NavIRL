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

VLM-backed overseer mode:

```bash
export NAVIRL_CODEX_CMD='/bin/zsh -lc "codex exec - --output-schema {schema_file} --output-last-message {output_file} {image_flags} < {prompt_file}"'
python -m navirl tune --suite quick --trials 24 --judge-mode vlm --judge-provider codex --no-judge-allow-fallback
```

Wainscott helper script (recommended for reproducible preflight/full runs):

```bash
./scripts/run_wainscott_vlm_tune.sh preflight
./scripts/run_wainscott_vlm_tune.sh full
```

Focused sweep with custom search space:

```bash
NAVIRL_TRIALS=32 \
NAVIRL_OUT_DIR=out/tune_wainscott_vlm_focus \
NAVIRL_SEARCH_SPACE=out/wainscott_stability_search_space.yaml \
./scripts/run_wainscott_vlm_tune.sh full
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

## Cost-efficient iteration policy

- Start with `preflight` (1 trial) to verify provider wiring before expensive sweeps.
- Keep VLM fallback disabled when validating provider reliability.
- Use tighter search spaces after an initial broad run to reduce wasted trials.

## Outputs

Run directory:

- `out/tune/tune_<suite>_<timestamp>_<id>/`

Artifacts:

- `trials.jsonl`: one JSON record per trial
- `best_params.json`: best-ranked trial
- `REPORT.md`: top-ranked candidates and reproduction command
- `AEGIS_RERANK.json`: overseer rerank diagnostics and provider trace

## Overseer reranking

When `--aegis-rerank` is enabled (default), Aegis applies a realism rerank pass:

- computes deterministic realism scores across scenarios per trial
- optionally asks a configured VLM provider to rank top trials (`--aegis-top-k`)
- blends provider ranking with heuristic realism score

Provider controls are shared with verify:

- `--judge-provider`
- `--judge-model`
- `--judge-endpoint`
- `--judge-api-key-env`
- `--judge-native-cmd`
- `--judge-allow-fallback`

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
