# Architecture Overview

NavIRL has three layers:

1. **Core ORCA library (C++)**
   - `src/` (RVO2-compatible core)
2. **Python bindings (Cython)**
   - `src/rvo2.pyx` exposing `PyRVOSimulator`
3. **Toolkit layer (`navirl/`)**
   - backends, controllers, scenario spec, metrics, logging, viz, verify

For detailed module flow and interfaces, see:

- `docs/ARCHITECTURE_TARGET.md`
- `docs/SCENARIO_SPEC.md`
- `docs/METRICS_SPEC.md`
- `docs/DATAFORMAT_SPEC.md`
