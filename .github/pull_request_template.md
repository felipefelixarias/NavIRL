## Summary
<!-- Provide a clear, concise description of what this PR accomplishes -->
-

## Changes
<!-- List specific changes made to the codebase -->
-

## Assumptions and Limitations
<!-- Document key assumptions made during implementation and known limitations -->
-

## Testing and Verification
- [ ] `python -m pytest` - All unit tests pass
- [ ] `python -m navirl verify --suite quick` - Quick E2E verification passes
- [ ] `python -m navirl verify --suite full` - Full verification (required for core changes)
- [ ] Manual testing performed (describe below)
- [ ] Not run (explain why)

### Test Plan
<!-- Describe your testing approach and any manual verification steps -->
-

### Test Results
<!-- Include relevant test output, performance metrics, or screenshots -->
-

## Reproducibility
<!-- Ensure changes are reproducible across environments -->
- [ ] No hardcoded paths, credentials, or environment-specific values
- [ ] Deterministic behavior (seeds, configs, test data documented)
- [ ] Compatible with existing installation instructions in README.md

## Breaking Changes
<!-- List any breaking changes and migration steps -->
- [ ] No breaking changes
- [ ] Breaking changes documented with migration guide:

## Quality Assurance
- [ ] Code follows project style guidelines (`ruff check .` passes)
- [ ] No new TODO/FIXME comments introduced without justification
- [ ] Documentation updated for public API changes
- [ ] No sensitive data (credentials, PII) included in commits

## AI Agent Pre-Merge Evidence
<!-- Required for AI-authored PRs per AGENTS.md -->
- [ ] I recorded key assumptions and limitations above
- [ ] I added or updated tests for behavior changes
- [ ] I documented any breaking changes and migration steps
- [ ] I verified reproducibility across clean environments
- [ ] I ran appropriate verification suites for the scope of changes
- [ ] I followed incremental commit practices with descriptive messages

## Notes for Reviewers
<!-- Additional context, open questions, or areas needing special attention -->
-
