# Getting Started

## Install

```bash
python -m pip install -U pip
python -m pip install -e .[dev]
```

If you use the repo-managed Python, run commands as:

```bash
.venv311/bin/python -m navirl --help
```

## Three-command workflow

1. Run a scenario:

```bash
python -m navirl run navirl/scenarios/library/hallway_pass.yaml --out logs/
```

`run` / `run-batch` apply log retention by default (7-day TTL). Override with
`--retention-hours` or `NAVIRL_LOG_TTL_HOURS`.

2. Evaluate one or more runs:

```bash
python -m navirl eval logs/**/state.jsonl --report out/eval/
```

3. View/debug a replay:

```bash
python -m navirl view logs/<run_id>/bundle/state.jsonl --out out/view/
```

Main thesis-map demo:

```bash
python -m navirl run navirl/scenarios/library/wainscott_main_demo.yaml --out logs/
```

For path-based maps, always provide map scale in ScenarioSpec:
`scene.map.pixels_per_meter` or `scene.map.meters_per_pixel`.

## Victory gate

Quick suite:

```bash
python -m navirl verify --suite quick
export NAVIRL_CODEX_CMD='/bin/zsh -lc "codex exec - --output-schema {schema_file} --output-last-message {output_file} {image_flags} < {prompt_file}"'
python -m navirl verify --suite quick --judge-mode vlm --judge-provider codex --no-judge-allow-fallback
```

Full suite:

```bash
python -m navirl verify --suite full
```

`verify` applies artifact retention by default (7-day TTL). Override with
`--retention-hours` or `NAVIRL_VERIFY_TTL_HOURS`.

## Hyperparameter tuning

Run ORCA/controller tuning with visual-judge scoring:

```bash
python -m navirl tune --suite quick --trials 24 --out out/tune/
```

Wainscott VLM workflow (strict no-fallback preflight + full run):

```bash
./scripts/run_wainscott_vlm_tune.sh preflight
./scripts/run_wainscott_vlm_tune.sh full
```

Exit codes:

- `0`: PASS
- `10`: FAIL (tests or invariants)
- `20`: NEEDS_HUMAN_REVIEW (e.g., unavailable VLM judge mode)
