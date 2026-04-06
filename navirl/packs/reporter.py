"""Generate JSON and Markdown reports for experiment pack results."""

from __future__ import annotations

import json
from pathlib import Path

from navirl.packs.schema import PackResult


def write_pack_json(result: PackResult, path: str | Path) -> None:
    """Write pack results as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def write_pack_markdown(
    result: PackResult,
    path: str | Path,
    metric_names: list[str] | None = None,
) -> None:
    """Write a Markdown report for pack results.

    Parameters
    ----------
    result:
        The completed pack result.
    path:
        Output file path.
    metric_names:
        Which metrics to include in the report.  If ``None``, all metrics
        found in the first completed run are reported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    completed = [r for r in result.runs if r.status == "completed"]
    failed = [r for r in result.runs if r.status == "failed"]

    if metric_names is None:
        metric_names = sorted({k for r in completed for k in r.metrics})

    agg = result.aggregate(metric_names) if metric_names else {}

    lines: list[str] = []
    lines.append(f"# Experiment Pack Report: {result.manifest_name}")
    lines.append("")
    lines.append(f"**Version:** {result.manifest_version}")
    lines.append(f"**Checksum:** `{result.manifest_checksum[:16]}...`")
    lines.append(f"**Generated:** {result.timestamp}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total runs | {len(result.runs)} |")
    lines.append(f"| Completed | {len(completed)} |")
    lines.append(f"| Failed | {len(failed)} |")
    lines.append("")

    # Aggregated metrics
    if agg:
        lines.append("## Aggregated Metrics")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        for key in metric_names:
            if key in agg:
                s = agg[key]
                lines.append(
                    f"| {key} | {s['mean']:.4f} | {s['std']:.4f} "
                    f"| {s['min']:.4f} | {s['max']:.4f} |"
                )
        lines.append("")

    # Per-scenario breakdown
    entry_ids = []
    seen = set()
    for r in result.runs:
        if r.entry_id not in seen:
            entry_ids.append(r.entry_id)
            seen.add(r.entry_id)

    if entry_ids:
        lines.append("## Per-Scenario Results")
        lines.append("")
        for eid in entry_ids:
            runs_for_entry = [r for r in completed if r.entry_id == eid]
            failed_for_entry = [r for r in failed if r.entry_id == eid]
            lines.append(f"### {eid}")
            lines.append("")
            lines.append(
                f"Runs: {len(runs_for_entry) + len(failed_for_entry)} | "
                f"Completed: {len(runs_for_entry)} | "
                f"Failed: {len(failed_for_entry)}"
            )
            lines.append("")
            if runs_for_entry and metric_names:
                lines.append("| Seed | " + " | ".join(metric_names[:6]) + " |")
                lines.append("|------|" + "|".join(["------"] * min(len(metric_names), 6)) + "|")
                for r in runs_for_entry:
                    vals = []
                    for key in metric_names[:6]:
                        v = r.metrics.get(key)
                        if isinstance(v, (int, float)):
                            vals.append(f"{v:.4f}")
                        else:
                            vals.append("N/A")
                    lines.append(f"| {r.seed} | " + " | ".join(vals) + " |")
                lines.append("")

    # Failure details
    if failed:
        lines.append("## Failures")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.entry_id}** (seed={r.seed}): {r.error}")
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
