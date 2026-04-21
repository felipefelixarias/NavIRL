"""Tests for navirl/core/env.py (SceneBackend) and navirl/__init__.py (_resolve_mplconfig_dir)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from navirl.core.env import SceneBackend


# ---------------------------------------------------------------------------
# SceneBackend concrete stub
# ---------------------------------------------------------------------------


class StubBackend(SceneBackend):
    """Minimal concrete implementation for testing the base class."""

    def add_agent(self, agent_id, position, radius, max_speed, kind):
        pass

    def set_preferred_velocity(self, agent_id, velocity):
        pass

    def step(self):
        pass

    def get_position(self, agent_id):
        return (0.0, 0.0)

    def get_velocity(self, agent_id):
        return (0.0, 0.0)

    def shortest_path(self, start, goal):
        return [start, goal]

    def sample_free_point(self):
        return (1.0, 1.0)

    def check_obstacle_collision(self, position, radius):
        return False

    def world_to_map(self, position):
        return (0, 0)

    def map_image(self):
        return np.zeros((10, 10), dtype=np.uint8)


class TestSceneBackendBase:
    def test_nearest_clear_point_returns_input(self):
        b = StubBackend()
        result = b.nearest_clear_point((3.5, -2.1), radius=0.3)
        assert result == (3.5, -2.1)

    def test_nearest_clear_point_converts_to_float(self):
        b = StubBackend()
        result = b.nearest_clear_point((3, -2), radius=0.3)
        assert result == (3.0, -2.0)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_map_metadata_returns_empty_dict(self):
        b = StubBackend()
        assert b.map_metadata() == {}

    def test_abstract_methods_are_callable(self):
        b = StubBackend()
        b.add_agent(1, (0.0, 0.0), 0.3, 1.0, "human")
        b.set_preferred_velocity(1, (0.5, 0.5))
        b.step()
        assert b.get_position(1) == (0.0, 0.0)
        assert b.get_velocity(1) == (0.0, 0.0)
        assert b.shortest_path((0.0, 0.0), (1.0, 1.0)) == [(0.0, 0.0), (1.0, 1.0)]
        assert b.sample_free_point() == (1.0, 1.0)
        assert b.check_obstacle_collision((0.0, 0.0), 0.3) is False
        assert b.world_to_map((0.0, 0.0)) == (0, 0)
        assert b.map_image().shape == (10, 10)


class TestSceneBackendCannotInstantiateDirectly:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SceneBackend()


# ---------------------------------------------------------------------------
# navirl.__init__ — _resolve_mplconfig_dir
# ---------------------------------------------------------------------------


class TestResolveMplconfigDir:
    def test_explicit_env_var(self, tmp_path):
        target = str(tmp_path / "custom_mpl")
        with patch.dict(os.environ, {"NAVIRL_MPLCONFIGDIR": target}, clear=False):
            from navirl import _resolve_mplconfig_dir
            result = _resolve_mplconfig_dir()
            assert result == Path(target)

    def test_xdg_cache_home(self, tmp_path):
        xdg = str(tmp_path / "xdg_cache")
        env = {"NAVIRL_MPLCONFIGDIR": "", "XDG_CACHE_HOME": xdg}
        with patch.dict(os.environ, env, clear=False):
            from navirl import _resolve_mplconfig_dir
            result = _resolve_mplconfig_dir()
            assert "navirl" in str(result)
            assert result.exists()

    def test_fallback_to_home_cache(self, tmp_path):
        env = {"NAVIRL_MPLCONFIGDIR": "", "XDG_CACHE_HOME": ""}
        with patch.dict(os.environ, env, clear=False):
            from navirl import _resolve_mplconfig_dir
            result = _resolve_mplconfig_dir()
            assert result.exists()

    def test_fallback_to_tempdir(self):
        """When home cache fails, falls back to tempdir."""
        env = {"NAVIRL_MPLCONFIGDIR": "", "XDG_CACHE_HOME": ""}
        with patch.dict(os.environ, env, clear=False):
            # Even if default candidates work, the function should still return a valid path
            from navirl import _resolve_mplconfig_dir
            result = _resolve_mplconfig_dir()
            assert isinstance(result, Path)
            assert result.exists()

    def test_all_candidates_fail_uses_mkdtemp(self):
        """When all mkdir attempts fail, mkdtemp is used."""
        env = {"NAVIRL_MPLCONFIGDIR": "", "XDG_CACHE_HOME": ""}
        original_mkdir = Path.mkdir

        call_count = 0
        def failing_mkdir(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError("simulated mkdir failure")

        with patch.dict(os.environ, env, clear=False):
            with patch.object(Path, "mkdir", failing_mkdir):
                from navirl import _resolve_mplconfig_dir
                result = _resolve_mplconfig_dir()
                # Should have fallen through to mkdtemp
                assert isinstance(result, Path)
                assert "navirl-mplconfig" in str(result)
