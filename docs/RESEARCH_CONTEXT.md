# Research Context

NavIRL is an agent-driven research engineering project anchored in a specific
research lineage led by Felipe Felix Arias:

- founder research site: https://felipefelixarias.github.io/
- thesis seed artifact: `research/ARIAS-THESIS-2023.pdf`
- predecessor simulator lineage: INDOORCA (captured in
  `docs/ARCHITECTURE_CURRENT.md`)

## Project intent

NavIRL is not framed as a novelty-only benchmark project.
It is a field-enabling toolkit focused on:

- reproducibility
- modularity
- debugging ergonomics
- standards for scenario and metrics exchange
- reliable agent-assisted development loops

## What changed from legacy roots

- old monolithic simulator workflows are replaced by a spec-first toolkit model
- all public surfaces are now under `navirl/*`
- verification is a hard gate (`navirl verify`) rather than ad-hoc checks

## Agent-driven development stance

NavIRL treats AI agents as first-class contributors.
The process requires:

- explicit docs updates with behavior changes
- deterministic test and verify execution before completion claims
- artifact-rich outputs for human audit (`REPORT.md`, bundles, renders)

This makes AI-assisted development inspectable and defensible in research
workflows.
