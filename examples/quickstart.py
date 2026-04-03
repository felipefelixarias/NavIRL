"""Quick start example demonstrating basic NavIRL simulation execution.

This script runs a simple group cohesion scenario and outputs the results
directory for inspection. It serves as an introduction to the NavIRL pipeline.
"""

from __future__ import annotations

import logging

from navirl.pipeline import run_scenario_file

logger = logging.getLogger(__name__)


def main() -> None:
    """Run a basic NavIRL simulation with group cohesion scenario.

    Executes the group cohesion scenario with rendering enabled and saves
    results to a logs directory. The output directory path is logged for
    easy access to simulation results and visualizations.
    """
    log = run_scenario_file(
        scenario_path="navirl/scenarios/library/group_cohesion.yaml",
        out_root="logs",
        run_id="quickstart",
        render_override=True,
        video_override=False,
    )
    logger.info("Simulation completed. Results saved to: %s", log.bundle_dir)


if __name__ == "__main__":
    main()
