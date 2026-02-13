# Governance

IndoorORCA is an open research project maintained by a mix of human
maintainers and AI agents.

## Roles
- **Maintainers**: repository owners who set direction, approve breaking
  changes, and manage releases.
- **AI Maintainers**: automated agents that contribute code, run CI, and
  auto-merge PRs under policy.
- **Contributors**: anyone submitting issues, PRs, or research artifacts.

## Decision making
- Day-to-day changes are decided by consensus in issues and PRs.
- Breaking changes, governance updates, or security-related changes require
  maintainer approval.
- In case of conflict, maintainers are final arbiters.

## Auto-merge policy
PRs are eligible for auto-merge when:
- CI is green
- There are no blocking review comments
- The PR includes tests or a clear rationale for not adding them

AI agents may trigger auto-merge for eligible PRs.

## Transparency
Major decisions and design changes should be recorded in issues or ADR-style
notes (see `docs/` or `research/` as appropriate).
