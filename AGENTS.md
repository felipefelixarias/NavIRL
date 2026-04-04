# AI Agent Guidelines

This repository is designed to be primarily developed and maintained by AI
agents. The goal is to keep changes reproducible, reviewable, and aligned with
research best practices.

## Agent responsibilities
- Keep PRs small, focused, and well-scoped.
- Record key assumptions and limitations in the PR description.
- Add tests when behavior changes; explain when not possible.
- Update documentation for any public API or workflow changes.
- Avoid introducing large binary artifacts without discussion.

### Quality Standards Checklist
Before creating a PR, verify:
- [ ] All tests pass (`python -m pytest`)
- [ ] Code style passes (`ruff check .`)
- [ ] No new TODO/FIXME comments without justification
- [ ] No hardcoded paths, credentials, or sensitive data
- [ ] Documentation reflects any public API changes
- [ ] Commit messages are descriptive and follow conventional format

### E2E Verification Requirements
Based on change scope, run appropriate verification:

#### Mandatory for All Changes
- [ ] `python -m pytest` - All unit tests pass
- [ ] `ruff check .` - Code style and linting passes
- [ ] `python -m navirl verify --suite quick` - Basic regression verification

#### Required by Change Scope

**Core Simulation Changes** (backends, physics, collision detection)
- [ ] `python -m navirl verify --suite full` - Full verification suite
- [ ] Manual test with 2+ scenarios from different categories (hallway, doorway, open space)
- [ ] Performance impact assessment (before/after metrics)

**Controller/Planner Changes** (human controllers, robot planners, navigation algorithms)
- [ ] `python -m navirl verify --suite full` - Full verification suite
- [ ] Test with both social navigation scenarios and obstacle avoidance
- [ ] Verify deterministic behavior with same seeds across runs

**Metrics/Evaluation Changes** (aggregation, reporting, benchmark comparisons)
- [ ] Test metric calculations with known ground-truth scenarios
- [ ] Verify backward compatibility with existing result formats
- [ ] Cross-validate against reference implementations where applicable

**New Scenarios/Data** (scenario library additions, new maps, configurations)
- [ ] Test scenario loads and runs without errors
- [ ] Verify scenario meets performance and quality standards
- [ ] Include scenario in appropriate test suite if canonical example

**Plugin/Extension Changes** (new backends, controllers, custom components)
- [ ] Include deterministic unit test for plugin functionality
- [ ] Add example scenario demonstrating plugin usage
- [ ] Document plugin registration and configuration requirements

**Documentation/Infrastructure** (build system, CI, dependencies)
- [ ] Test installation from scratch in clean environment
- [ ] Verify all documented commands and examples still work
- [ ] Check documentation rendering and link validity

**Agent-Driven Development Improvements** (verification, quality gates, automation)
- [ ] Meta-verification: test the verification tools themselves
- [ ] Ensure new quality gates don't break existing workflows
- [ ] Validate automation improvements with historical PR examples

## Regression Prevention and Quality Gates

### Automated CI Requirements
All PRs must pass these automated checks before merge:
- **Unit Tests**: Full test suite must pass (`python -m pytest`)
- **Code Quality**: Style and type checking (`ruff check .`, `ruff format --check .`)
- **Quick Verification**: Basic scenario regression suite (`python -m navirl verify --suite quick`)
- **Build Validation**: Package builds successfully (`python -m build`)

### Behavioral Regression Detection
Changes to core components trigger additional verification:
- **Deterministic Replay**: Scenario results must be reproducible with same seeds
- **Metric Stability**: Key performance indicators within acceptable tolerance
- **Reference Scenario Suite**: Canonical examples continue to pass quality thresholds
- **Cross-Platform Compatibility**: Results consistent across supported environments

### Quality Gate Enforcement
- **Pre-commit Hooks**: Style and basic validations run automatically
- **Branch Protection**: Require passing CI checks and verification evidence
- **Review Requirements**: Human review mandatory for core algorithm changes
- **Merge Criteria**: All verification evidence documented in PR description

## Required annotations
Every AI-authored PR should include:
- A concise summary of changes
- Test plan and results with specific commands run
- Any breaking changes and migration notes
- Reproducibility notes (seeds, configs, data sources)
- Key assumptions and limitations documented
- Evidence of appropriate verification suite completion

## Pre-merge Evidence Requirements
All AI-authored PRs must demonstrate:

### 1. Functional Verification Evidence
- [ ] Appropriate test suite completion (see E2E Verification Requirements above)
- [ ] Test results included in PR description with specific commands run
- [ ] Any test failures explained and justified
- [ ] Manual testing described for UI/interactive components

### 2. Quality Verification Evidence
- [ ] All automated CI checks passing (tests, linting, builds)
- [ ] Code follows project conventions and style guidelines
- [ ] No TODO/FIXME introduced without issue tracking
- [ ] Documentation updated for public API changes

### 3. Regression Verification Evidence
- [ ] No behavior regressions in verification suites
- [ ] Performance impact assessed and documented if applicable
- [ ] Backward compatibility maintained or migration path provided
- [ ] Deterministic behavior verified across multiple test runs

### 4. Documentation Verification Evidence
- [ ] Inline code documentation reflects changes
- [ ] Public API changes documented in appropriate files
- [ ] Examples updated to reflect new functionality
- [ ] CHANGELOG.md updated if user-visible changes

### 5. Reproducibility Verification Evidence
- [ ] No hardcoded paths, credentials, or environment-specific dependencies
- [ ] Seeds and configurations documented for deterministic components
- [ ] Installation tested in clean environment (new virtualenv)
- [ ] Dependencies and versions explicitly specified

### Auto-merge Eligibility
AI agents may auto-merge PRs that meet ALL criteria:
- [ ] Pass all CI checks (tests, linting, builds)
- [ ] Complete required verification suites for change scope
- [ ] Have no unresolved review comments from humans
- [ ] Include tests or documented rationale for omission
- [ ] Follow PR template with all required sections completed
- [ ] Demonstrate reproducibility across environments

## Change-Specific Checklists

### Algorithm/Model Changes
When modifying core algorithms (social forces, ORCA, path planning):
- [ ] Validate against established literature and reference implementations
- [ ] Document mathematical assumptions and parameter sensitivities
- [ ] Test edge cases (zero velocity, boundary conditions, high density)
- [ ] Verify numerical stability and convergence properties
- [ ] Assess computational complexity and performance implications

### Data/Configuration Changes
When adding scenarios, maps, or configuration templates:
- [ ] Validate scenario realism and physical plausibility
- [ ] Ensure map scaling and coordinate system consistency
- [ ] Test configuration parameter ranges and boundary values
- [ ] Verify scenario difficulty spans (easy, medium, hard examples)
- [ ] Document data sources and licensing constraints

### Infrastructure Changes
When modifying build system, CI, or development tools:
- [ ] Test on multiple platforms and Python versions
- [ ] Verify backward compatibility with existing developer workflows
- [ ] Document any new dependencies or installation requirements
- [ ] Ensure CI changes don't break existing branch protection rules
- [ ] Test rollback procedures for infrastructure failures

### Performance/Optimization Changes
When optimizing algorithms or data structures:
- [ ] Benchmark against baseline with statistical significance tests
- [ ] Verify algorithmic correctness is preserved
- [ ] Test performance across different scenario sizes and complexities
- [ ] Document trade-offs between speed, memory, and accuracy
- [ ] Include performance regression tests in verification suite

## Safety and data handling
- Do not include credentials, private datasets, or personally identifiable
  information in commits.
- Prefer synthetic or openly licensed data for tests and examples.

## NavIRL workflow

### Install and test
- Install editable package:
  - `python3 -m venv .venv && ./.venv/bin/python -m pip install -e .[dev]`
- Run tests:
  - `./.venv/bin/pytest -q`

### Run smoke simulation
- Run a canonical scenario:
  - `./.venv/bin/python -m navirl run navirl/scenarios/library/hallway_pass.yaml --out logs/`
- Run thesis floorplan demo:
  - `./.venv/bin/python -m navirl run navirl/scenarios/library/wainscott_main_demo.yaml --out logs/`

### Run verify gate
- Quick suite (required before claiming task completion):
  - `./.venv/bin/python -m navirl verify --suite quick`
- Full suite (required when changing core sim/controllers/planners/metrics):
  - `./.venv/bin/python -m navirl verify --suite full`

### Tune ORCA/social-nav hyperparameters
- Default quick tuning sweep:
  - `./.venv/bin/python -m navirl tune --suite quick --trials 24 --out out/tune/`
- Custom scenario set:
  - `./.venv/bin/python -m navirl tune --scenarios navirl/scenarios/library/hallway_pass.yaml navirl/scenarios/library/doorway_token_yield.yaml --trials 40`
- Notes:
  - tuner jointly optimizes ORCA params, traversability wall-clearance offset,
    and deadlock-retry controls using numeric invariants + visual judge score.

### Plugin conventions
- Register plugins in `navirl/plugins.py`.
- New backends implement `navirl/core/env.py:SceneBackend`.
- Human controllers implement `navirl/humans/base.py:HumanController`.
- Robot controllers implement `navirl/robots/base.py:RobotController`.
- Controllers must be selectable by `controller.type` in ScenarioSpec.
- Behavior-changing plugins should include at least one deterministic test and
  one scenario/library example.

### Map units
- Path maps (`scene.map.source: path`) must define map scale with
  `scene.map.pixels_per_meter` or `scene.map.meters_per_pixel`.
- Optional `scene.map.downsample` is supported; world coordinates remain meters.
