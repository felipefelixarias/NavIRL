# Reproducibility

NavIRL is designed for deterministic, inspectable research workflows.

## Environment

- Python: 3.11
- CMake: 3.16+
- Install: `python -m pip install -e .[dev]`

## Required verification before claiming results

```bash
pytest -q
python -m navirl verify --suite quick
```

When core simulator/controllers/planners/metrics change:

```bash
python -m navirl verify --suite full
```

## Seed policy

- Every scenario has an explicit `seed`.
- Runs must preserve the scenario seed in `scenario.yaml` inside bundle outputs.
- Seed changes must be logged in experiment metadata.

## Artifact policy

Each run should emit a trace bundle:

- `scenario.yaml`
- `state.jsonl`
- `events.jsonl`
- `summary.json`
- `frames/*.png`
- `frames/video.mp4` (suite/config dependent)

## Recommended experiment layout

```text
research/
  experiments/
    <study_name>/
      scenarios/
      runs/
      reports/
      notes.md
```

## Required metadata for experiments

- scenario version and file paths
- random seeds
- platform and package versions
- verification suite used and report path
- key thresholds (intrusion, speed, accel, teleport)

## Determinism notes

Cross-platform floating point and rendering differences can appear in marginal
cases. Always record:

- OS
- Python version
- package versions
- CPU architecture
