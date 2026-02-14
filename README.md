# NavIRL

NavIRL is an agent-driven indoor social-navigation toolkit for reproducible
simulation, rigorous verification, and high-quality demo generation.

## Watch Latest Aegis Demos

### Wainscott Main Demo (Thesis Floorplan, high-density crowd)
![Wainscott Main Demo](docs/assets/showcase/wainscott_main_demo.gif)
[Best Full-Run Video (MP4)](docs/assets/showcase/wainscott_main_demo.mp4) |
[Stability-Focused Comparison (MP4)](docs/assets/showcase/wainscott_main_demo_focused.mp4)

Source artifacts:
`out/tune_wainscott_vlm/tune_quick_20260214_051921_998e5d/trials/trial_022/trial_022_wainscott_main_demo/bundle`
and
`out/tune_wainscott_vlm_focus/tune_quick_20260214_062836_4892bf/trials/trial_022/trial_022_wainscott_main_demo/bundle`

### Group Cohesion
![Group Cohesion](docs/assets/showcase/group_cohesion.gif)

### Kitchen Congestion
![Kitchen Congestion](docs/assets/showcase/kitchen_congestion.gif)

### Robot Comfort Avoidance
![Robot Comfort Avoidance](docs/assets/showcase/robot_comfort_avoidance.gif)

### Hallway Pass
![Hallway Pass](docs/assets/showcase/hallway_pass.gif)

### Doorway Token Yield
![Doorway Token Yield](docs/assets/showcase/doorway_token_yield.gif)

### Routine Cook Dinner Micro
![Routine Cook Dinner Micro](docs/assets/showcase/routine_cook_dinner_micro.gif)

## Why NavIRL

- Deterministic simulation and structured scenario specs
- Numeric invariants + visual quality gate (`verify`) before claiming results
- Aegis overseer pipeline for qualitative realism checks and tuning rerank
- End-to-end artifact traces for debugging and reproducibility

## Quickstart

Install:

```bash
python -m pip install -U pip
python -m pip install -e .[dev]
```

Run a scenario:

```bash
python -m navirl run navirl/scenarios/library/hallway_pass.yaml --out logs/
```

Run the verification gate:

```bash
python -m navirl verify --suite quick
python -m navirl verify --suite full
```

Run Aegis-backed tuning:

```bash
export NAVIRL_CODEX_CMD='/bin/zsh -lc "codex exec - --output-schema {schema_file} --output-last-message {output_file} {image_flags} < {prompt_file}"'
./scripts/run_wainscott_vlm_tune.sh preflight
./scripts/run_wainscott_vlm_tune.sh full
```

Run thesis floorplan demo:

```bash
python -m navirl run navirl/scenarios/library/wainscott_main_demo.yaml --out logs/ --render --video
```

## Wainscott Demo (Updated)

`navirl/scenarios/library/wainscott_main_demo.yaml` now runs a high-density crowd
layout (`16` humans) while preserving the thesis floorplan scale and robot route.

## Detailed Docs

For full specs and workflows:

- `docs/GETTING_STARTED.md`
- `docs/SCENARIO_SPEC.md`
- `docs/VERIFY_GATE.md`
- `docs/TUNING.md`
- `docs/ARCHITECTURE.md`
- `docs/README.md`

## Research Origin

NavIRL is rooted in Felipe Felix Arias's INDOORCA and thesis research lineage.

- Research home: https://felipefelixarias.github.io/
- Thesis artifact: `research/ARIAS-THESIS-2023.pdf`

## License

Apache 2.0 (`LICENSE`).
