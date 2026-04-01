from __future__ import annotations

from navirl import planning


def test_planning_public_exports_match_module() -> None:
    expected_exports = {
        "Path",
        "Planner",
        "PlannerConfig",
        "AStarPlanner",
        "DijkstraPlanner",
        "PRMPlanner",
        "RRTPlanner",
        "RRTStarPlanner",
        "ThetaStarPlanner",
    }

    assert set(planning.__all__) == expected_exports
    for export in expected_exports:
        assert hasattr(planning, export)
