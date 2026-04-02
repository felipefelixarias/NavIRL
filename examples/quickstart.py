from __future__ import annotations

from navirl.pipeline import run_scenario_file


def main() -> None:
    log = run_scenario_file(
        scenario_path="navirl/scenarios/library/group_cohesion.yaml",
        out_root="logs",
        run_id="quickstart",
        render_override=True,
        video_override=False,
    )
    print(log.bundle_dir)


if __name__ == "__main__":
    main()
