"""Entry point for running NavIRL as a module via `python -m navirl`."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
