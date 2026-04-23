"""Tests for navirl.pipeline._run_scenario_worker and run_scenario_file delegation.

The bulk of ``run_scenario_dict`` requires a working backend (rvo2 or grid2d
with all dependencies); these tests target the thin adapter functions that
delegate to it, using mocks to avoid that requirement.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from navirl.pipeline import _run_scenario_worker, run_scenario_file


@patch("navirl.pipeline.run_scenario_dict")
@patch("navirl.pipeline.load_scenario")
def test_run_scenario_worker_loads_and_runs(mock_load, mock_run, tmp_path):
    """The worker must inject the seed and synthesize a deterministic run id."""
    mock_load.return_value = {"id": "hallway"}
    mock_log = MagicMock()
    mock_run.return_value = mock_log

    result = _run_scenario_worker(
        (str(tmp_path / "scn.yaml"), 99, str(tmp_path / "out"), True, False)
    )

    mock_load.assert_called_once_with(str(tmp_path / "scn.yaml"))

    # The scenario passed to run_scenario_dict should have the seed overridden
    call = mock_run.call_args
    assert call.kwargs["scenario"]["seed"] == 99
    assert call.kwargs["run_id"] == "hallway_seed99"
    assert call.kwargs["render_override"] is True
    assert call.kwargs["video_override"] is False
    assert call.kwargs["out_root"] == str(tmp_path / "out")

    assert result is mock_log


@patch("navirl.pipeline.run_scenario_dict")
@patch("navirl.pipeline.load_scenario")
def test_run_scenario_file_delegates(mock_load, mock_run, tmp_path):
    """``run_scenario_file`` is a thin wrapper around ``load_scenario`` + ``run_scenario_dict``."""
    mock_load.return_value = {"id": "kitchen"}
    mock_log = MagicMock()
    mock_run.return_value = mock_log

    result = run_scenario_file(
        scenario_path=tmp_path / "scn.yaml",
        out_root=tmp_path / "out",
        run_id="custom_id",
        render_override=False,
        video_override=True,
    )

    mock_load.assert_called_once_with(tmp_path / "scn.yaml")
    call = mock_run.call_args
    assert call.kwargs["scenario"] == {"id": "kitchen"}
    assert call.kwargs["run_id"] == "custom_id"
    assert call.kwargs["render_override"] is False
    assert call.kwargs["video_override"] is True
    assert result is mock_log
