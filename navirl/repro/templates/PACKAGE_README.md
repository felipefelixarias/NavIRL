# Reproducibility Package: {name}

**Version**: {version}
**Created**: {created_at}
**Description**: {description}

## Contents

```
{name}/
  MANIFEST.json          # Package manifest with checksums and environment pins
  README.md              # This file
  scenarios/             # Scenario YAML configurations
  results/               # Run summaries and aggregated metrics
```

## Prerequisites

- Python {python_version}
- NavIRL (install from repository root: `pip install -e .`)

## Replay Instructions

1. Install NavIRL and its dependencies in a clean virtual environment:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e /path/to/NavIRL
   ```

2. Verify package integrity:

   ```bash
   navirl repro verify {package_dir}
   ```

3. Run the publication readiness checklist:

   ```bash
   navirl repro check {package_dir}
   ```

4. Replay each scenario and compare results:

   ```bash
   for scenario in {package_dir}/scenarios/*.yaml; do
     navirl run "$scenario" --out replay_output/
   done
   ```

5. Compare replayed metrics against expected values in `MANIFEST.json`.

## Expected Metrics

{metrics_table}

## Environment

- **Python**: {python_version}
- **Platform**: {platform_system} {platform_machine}
- **Packages**: See `MANIFEST.json` → `environment.packages` for full pin list
