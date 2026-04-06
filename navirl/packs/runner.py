"""Execute experiment packs and collect results."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

from navirl.core.seeds import set_global_seed
from navirl.metrics.standard import StandardMetrics
from navirl.packs.schema import PackManifest, PackResult, PackRunResult
from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario

logger = logging.getLogger(__name__)


def run_pack(
    manifest: PackManifest,
    out_root: str | Path,
    *,
    render: bool = False,
    video: bool = False,
) -> PackResult:
    """Execute all scenarios in a pack and return collected results.

    Parameters
    ----------
    manifest:
        The experiment pack manifest.
    out_root:
        Root output directory for run bundles.
    render:
        Whether to render visuals during simulation.
    video:
        Whether to record video output.

    Returns
    -------
    PackResult
        Results for every (scenario, seed) combination.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metrics_collector = StandardMetrics()
    result = PackResult(
        manifest_name=manifest.name,
        manifest_version=manifest.version,
        manifest_checksum=manifest.checksum(),
        timestamp=datetime.now(UTC).isoformat(),
    )

    task_num = 0
    total = manifest.total_runs
    for entry in manifest.scenarios:
        for seed in entry.seeds:
            task_num += 1
            logger.info(
                "Pack run %d/%d: %s seed=%d",
                task_num, total, entry.id, seed,
            )

            try:
                scenario = load_scenario(entry.path)
                scenario["seed"] = seed
                set_global_seed(seed)

                episode_log = run_scenario_dict(
                    scenario,
                    out_root=str(out_root),
                    render_override=render,
                    video_override=video,
                )

                state_path = Path(episode_log.state_path)
                bundle_dir = state_path.parent
                scenario_yaml = bundle_dir / "scenario.yaml"
                with scenario_yaml.open("r", encoding="utf-8") as f:
                    run_scenario = yaml.safe_load(f)

                run_metrics = metrics_collector.compute(state_path, run_scenario)

                result.runs.append(
                    PackRunResult(
                        entry_id=entry.id,
                        seed=seed,
                        metrics=run_metrics,
                        status="completed",
                    )
                )
            except Exception as exc:
                logger.warning("Pack run %d failed: %s", task_num, exc)
                result.runs.append(
                    PackRunResult(
                        entry_id=entry.id,
                        seed=seed,
                        status="failed",
                        error=str(exc),
                    )
                )

    return result
