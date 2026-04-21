# Reproducibility Checklist

Use this checklist to verify that a reproducibility package meets the minimum
requirements for publication alongside a study.

## Package Identity

- [ ] Package has a unique name
- [ ] Package has a semantic version
- [ ] Package includes a human-readable description

## Environment Pins

- [ ] Python version recorded
- [ ] Platform (OS, architecture) recorded
- [ ] All pip-installed packages pinned with exact versions

## Scenario Configs

- [ ] At least one scenario YAML included
- [ ] Scenario files are self-contained (no external references to private data)
- [ ] Scenario seeds are fixed for deterministic replay

## Results and Metrics

- [ ] Result summary JSON files included for each run
- [ ] Expected metrics documented with mean and standard deviation
- [ ] Metrics are reproducible within documented tolerance

## Artifact Integrity

- [ ] All artifacts have SHA-256 checksums in MANIFEST.json
- [ ] Package-level checksum is present
- [ ] `navirl repro verify <package_dir>` passes without errors

## Legal and Data Compliance

- [ ] No private credentials, API keys, or tokens included
- [ ] No personally identifiable information (PII) in any artifact
- [ ] No hardcoded absolute paths referencing private filesystems
- [ ] License file or notice included if distributing third-party data

## Replay Validation

- [ ] Package can be replayed on a clean environment with documented instructions
- [ ] Replayed results match expected metrics within tolerance
- [ ] README includes step-by-step replay instructions
