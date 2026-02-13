# Contributing

Thanks for your interest in contributing to IndoorORCA. This project is
maintained with an AI-first workflow and welcomes human and agent-driven
contributions.

## Quick start
1. Fork the repo and create a feature branch.
2. Create a virtual environment (Python 3.11 recommended).
3. Install dev dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install cmake ninja
   python -m pip install -e .[dev]
   ```
4. Run tests:
   ```bash
   python -m pytest
   ```

Optional: enable pre-commit hooks
```bash
pre-commit install
```

## Development workflow
- Keep PRs focused and scoped.
- Update documentation when behavior changes.
- Add or update tests for bug fixes and new features.
- If you change defaults or configuration, document the rationale.

## Pull requests
Include:
- A clear summary of the change
- A test plan or testing notes
- Any breaking changes and migration steps

Auto-merge policy:
- PRs are eligible for auto-merge when CI passes and there are no open review
  objections.
- Maintainers or AI agents may trigger auto-merge per `GOVERNANCE.md`.

## Research contributions
- Place new artifacts in `research/` with a README describing provenance.
- Include experiment configuration and seed details.
- Avoid large binary files unless they are essential; consider external hosting.

## AI agents
AI agents must follow `AGENTS.md`, including explicit assumptions and
reproducibility notes.
