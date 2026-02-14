# NavIRL

NavIRL is an agent-driven indoor social-navigation toolkit for reproducible
simulation, rigorous verification, and high-quality demo generation.

## Watch Latest Aegis Demos

Best showcase videos (latest Aegis-generated artifacts linked in this README):

1. **Wainscott Main Demo (Thesis Floorplan, high-density crowd)**  
   [`docs/assets/showcase/wainscott_main_demo.mp4`](docs/assets/showcase/wainscott_main_demo.mp4)
2. **Group Cohesion**  
   [`docs/assets/showcase/group_cohesion.mp4`](docs/assets/showcase/group_cohesion.mp4)
3. **Kitchen Congestion**  
   [`docs/assets/showcase/kitchen_congestion.mp4`](docs/assets/showcase/kitchen_congestion.mp4)
4. **Robot Comfort Avoidance**  
   [`docs/assets/showcase/robot_comfort_avoidance.mp4`](docs/assets/showcase/robot_comfort_avoidance.mp4)
5. **Hallway Pass**  
   [`docs/assets/showcase/hallway_pass.mp4`](docs/assets/showcase/hallway_pass.mp4)
6. **Doorway Token Yield**  
   [`docs/assets/showcase/doorway_token_yield.mp4`](docs/assets/showcase/doorway_token_yield.mp4)
7. **Routine Cook Dinner Micro**  
   [`docs/assets/showcase/routine_cook_dinner_micro.mp4`](docs/assets/showcase/routine_cook_dinner_micro.mp4)

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
python -m navirl tune --suite quick --trials 24 --judge-mode vlm --judge-provider codex
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
