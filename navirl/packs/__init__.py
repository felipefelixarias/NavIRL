"""Standardized experiment packs for cross-lab reproducibility.

An experiment pack bundles a versioned set of scenarios with fixed seeds
and metric selections.  Packs produce checksummed results so that outputs
from different labs can be compared deterministically.
"""

from __future__ import annotations

from navirl.packs.loader import load_pack
from navirl.packs.reporter import write_pack_json, write_pack_markdown
from navirl.packs.runner import run_pack
from navirl.packs.schema import PackManifest, PackResult, PackRunResult, PackScenarioEntry

__all__ = [
    "PackManifest",
    "PackResult",
    "PackRunResult",
    "PackScenarioEntry",
    "load_pack",
    "run_pack",
    "write_pack_json",
    "write_pack_markdown",
]
