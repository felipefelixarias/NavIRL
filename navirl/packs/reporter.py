"""Generate markdown and JSON reports from pack results."""

from __future__ import annotations

import json
from pathlib import Path

from navirl.packs.runner import PackResult


def generate_pack_report(
    result: PackResult,
    out_dir: str | Path,
    metric_names: list[str] | None = None,
) -> Path:
    """Write a JSON results file and a Markdown summary for a pack run.

    Args:
        result: Completed pack execution result.
        out_dir: Directory to write report files into.
        metric_names: Subset of metric names to include. If None, includes all.

    Returns:
        Path to the generated Markdown report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump of full results.
    json_path = out_dir / "pack_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)

    # Aggregate metrics.
    agg = result.aggregate(metric_names)

    # Build markdown report.
    lines = [
        f"# Experiment Pack Report: {result.pack_name}",
        "",
        f"- **Version:** {result.pack_version}",
        f"- **Checksum:** `{result.checksum}`",
        f"- **Total runs:** {len(result.runs)}",
        "",
        "## Aggregate Metrics",
        "",
    ]

    if agg:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in sorted(agg):
            lines.append(f"| {key} | {agg[key]:.4f} |")
    else:
        lines.append("_(no numeric metrics collected)_")

    lines.extend(["", "## Per-Run Results", ""])

    # Group runs by scenario entry.
    entries_seen: dict[str, list[int]] = {}
    for run in result.runs:
        entries_seen.setdefault(run.entry_id, []).append(
            result.runs.index(run)
        )

    for entry_id, indices in entries_seen.items():
        lines.append(f"### {entry_id}")
        lines.append("")
        for idx in indices:
            run = result.runs[idx]
            lines.append(f"- **seed {run.seed}**: bundle=`{run.bundle_dir}`")
            if run.metrics:
                key_metrics = {
                    k: v
                    for k, v in run.metrics.items()
                    if isinstance(v, (int, float))
                    and (metric_names is None or k in metric_names)
                }
                if key_metrics:
                    parts = [f"{k}={v:.3f}" for k, v in sorted(key_metrics.items())]
                    lines.append(f"  - {', '.join(parts)}")
        lines.append("")

    lines.extend([
        "## Artifacts",
        "",
        f"- `{json_path}`",
    ])

    md_path = out_dir / "PACK_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return md_path
