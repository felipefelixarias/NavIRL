from __future__ import annotations

from pathlib import Path

import yaml

from navirl.cli import build_parser


def test_cli_overseer_layout_writes_patched_scenario(tmp_path: Path):
    src = Path("navirl/scenarios/library/hallway_pass.yaml")
    scenario_path = tmp_path / "hallway.yaml"
    scenario_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    out_path = tmp_path / "hallway_layout.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "overseer-layout",
            str(scenario_path),
            "--humans-count",
            "3",
            "--seed",
            "11",
            "--write-scenario",
            str(out_path),
        ]
    )

    rc = int(args.func(args))
    assert rc == 0
    assert out_path.exists()

    patched = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert patched["humans"]["count"] == 3
    assert len(patched["humans"]["starts"]) == 3
    assert len(patched["humans"]["goals"]) == 3
