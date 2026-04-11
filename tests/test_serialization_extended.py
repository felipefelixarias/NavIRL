"""Extended tests for navirl/config/serialization.py.

Covers: TOML save/load, error handling, None config validation,
auto_cast edge cases, flatten_to_args with lists/bools, set_nested,
diff_configs nested, config_to_cli_args edge cases, and CLI round-trips.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from navirl.config.serialization import (
    cli_args_to_config,
    config_to_cli_args,
    diff_configs,
    load_config,
    merge_configs,
    save_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# TOML save/load
# ---------------------------------------------------------------------------


def _has_toml_write():
    try:
        import tomli_w

        return True
    except ImportError:
        pass
    try:
        import toml

        return True
    except ImportError:
        return False


class TestTomlSaveLoad:
    @pytest.mark.skipif(not _has_toml_write(), reason="TOML write library not installed")
    def test_save_load_toml_roundtrip(self, tmp_dir):
        cfg = {"model": {"lr": 0.001, "epochs": 10}, "name": "experiment_1"}
        path = tmp_dir / "config.toml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded == cfg

    @pytest.mark.skipif(not _has_toml_write(), reason="TOML write library not installed")
    def test_save_toml_explicit_format(self, tmp_dir):
        cfg = {"key": "value"}
        path = tmp_dir / "config"
        save_config(cfg, path, file_format="toml")
        loaded = load_config(path.with_suffix(".toml"))
        assert loaded == cfg

    @pytest.mark.skipif(not _has_toml_write(), reason="TOML write library not installed")
    def test_load_toml_nested(self, tmp_dir):
        cfg = {"section": {"subsection": {"value": 42}}}
        path = tmp_dir / "nested.toml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded["section"]["subsection"]["value"] == 42


# ---------------------------------------------------------------------------
# YAML save/load
# ---------------------------------------------------------------------------


class TestYamlSaveLoad:
    def test_save_load_yaml_roundtrip(self, tmp_dir):
        cfg = {"model": {"lr": 0.001}, "data": {"path": "/tmp/data"}}
        path = tmp_dir / "config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded == cfg

    def test_yml_extension(self, tmp_dir):
        cfg = {"key": "value"}
        path = tmp_dir / "config.yml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded == cfg

    def test_empty_yaml_returns_empty_dict(self, tmp_dir):
        path = tmp_dir / "empty.yaml"
        path.write_text("")
        loaded = load_config(path)
        assert loaded == {}


# ---------------------------------------------------------------------------
# JSON error handling
# ---------------------------------------------------------------------------


class TestJsonErrors:
    def test_invalid_json(self, tmp_dir):
        path = tmp_dir / "bad.json"
        path.write_text("{invalid json content}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(path)

    def test_json_non_dict(self, tmp_dir):
        path = tmp_dir / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="dictionary"):
            load_config(path)


# ---------------------------------------------------------------------------
# save_config error handling
# ---------------------------------------------------------------------------


class TestSaveConfigErrors:
    def test_none_config_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="None"):
            save_config(None, tmp_dir / "out.json")

    def test_unsupported_format_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="Unsupported"):
            save_config({"k": "v"}, tmp_dir / "out.xyz", file_format="xyz")


# ---------------------------------------------------------------------------
# load_config error handling
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_dir / "nonexistent.json")

    def test_directory_raises(self, tmp_dir):
        sub = tmp_dir / "subdir"
        sub.mkdir()
        # Create a file matching the resolved path so it passes existence check
        # but give it a known extension so format resolution works
        with pytest.raises((ValueError, FileNotFoundError)):
            load_config(sub)

    def test_unknown_extension(self, tmp_dir):
        path = tmp_dir / "config.xyz"
        path.write_text("data")
        with pytest.raises(ValueError):
            load_config(path)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_resolve_existing_no_extension(self, tmp_dir):
        """Load should find config.json when given 'config' without extension."""
        cfg = {"resolved": True}
        path = tmp_dir / "config.json"
        save_config(cfg, path)
        loaded = load_config(tmp_dir / "config")
        assert loaded == cfg

    def test_resolve_yaml_no_extension(self, tmp_dir):
        cfg = {"resolved": True}
        path = tmp_dir / "settings.yaml"
        save_config(cfg, path)
        loaded = load_config(tmp_dir / "settings")
        assert loaded == cfg

    def test_parent_dir_creation(self, tmp_dir):
        cfg = {"key": "value"}
        path = tmp_dir / "a" / "b" / "c" / "config.json"
        save_config(cfg, path)
        assert path.exists()
        loaded = load_config(path)
        assert loaded == cfg


# ---------------------------------------------------------------------------
# CLI conversion edge cases
# ---------------------------------------------------------------------------


class TestConfigToCliArgs:
    def test_nested_dict(self):
        cfg = {"a": {"b": {"c": 1}}}
        args = config_to_cli_args(cfg)
        assert args == ["--a.b.c", "1"]

    def test_list_values(self):
        cfg = {"items": [1, 2, 3]}
        args = config_to_cli_args(cfg)
        assert args == ["--items", "1", "--items", "2", "--items", "3"]

    def test_bool_true_flag(self):
        cfg = {"verbose": True}
        args = config_to_cli_args(cfg)
        assert args == ["--verbose"]

    def test_bool_false_omitted(self):
        cfg = {"debug": False}
        args = config_to_cli_args(cfg)
        assert args == []

    def test_empty_config(self):
        assert config_to_cli_args({}) == []

    def test_mixed_types(self):
        cfg = {"name": "test", "count": 5, "rate": 0.1}
        args = config_to_cli_args(cfg)
        assert "--name" in args
        assert "test" in args
        assert "--count" in args
        assert "5" in args
        assert "--rate" in args
        assert "0.1" in args


class TestCliArgsToConfig:
    def test_auto_cast_int(self):
        cfg = cli_args_to_config(["--epochs", "10"])
        assert cfg["epochs"] == 10
        assert isinstance(cfg["epochs"], int)

    def test_auto_cast_float(self):
        cfg = cli_args_to_config(["--lr", "0.001"])
        assert cfg["lr"] == pytest.approx(0.001)
        assert isinstance(cfg["lr"], float)

    def test_auto_cast_bool_true(self):
        for val in ["true", "True", "yes", "Yes"]:
            cfg = cli_args_to_config(["--flag", val])
            assert cfg["flag"] is True

    def test_auto_cast_bool_false(self):
        for val in ["false", "False", "no", "No"]:
            cfg = cli_args_to_config(["--flag", val])
            assert cfg["flag"] is False

    def test_auto_cast_none(self):
        for val in ["none", "None", "null", "Null"]:
            cfg = cli_args_to_config(["--val", val])
            assert cfg["val"] is None

    def test_flag_without_value(self):
        cfg = cli_args_to_config(["--verbose"])
        assert cfg["verbose"] is True

    def test_nested_keys(self):
        cfg = cli_args_to_config(["--model.lr", "0.01", "--model.layers", "4"])
        assert cfg == {"model": {"lr": 0.01, "layers": 4}}

    def test_deeply_nested(self):
        cfg = cli_args_to_config(["--a.b.c.d", "deep"])
        assert cfg["a"]["b"]["c"]["d"] == "deep"

    def test_non_flag_tokens_skipped(self):
        cfg = cli_args_to_config(["positional", "--key", "val"])
        assert "positional" not in cfg
        assert cfg["key"] == "val"

    def test_string_preserved(self):
        cfg = cli_args_to_config(["--name", "my_experiment"])
        assert cfg["name"] == "my_experiment"
        assert isinstance(cfg["name"], str)

    def test_empty_args(self):
        assert cli_args_to_config([]) == {}


# ---------------------------------------------------------------------------
# Merge configs edge cases
# ---------------------------------------------------------------------------


class TestMergeConfigsExtended:
    def test_deeply_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        over = {"a": {"b": {"c": 99, "e": 3}}}
        merged = merge_configs(base, over)
        assert merged == {"a": {"b": {"c": 99, "d": 2, "e": 3}}}

    def test_non_destructive(self):
        base = {"key": "original"}
        over = {"key": "modified"}
        merge_configs(base, over)
        assert base["key"] == "original"

    def test_override_dict_with_scalar(self):
        base = {"a": {"nested": True}}
        over = {"a": "flat"}
        merged = merge_configs(base, over)
        assert merged["a"] == "flat"

    def test_empty_override(self):
        base = {"a": 1, "b": 2}
        merged = merge_configs(base, {})
        assert merged == base

    def test_empty_base(self):
        over = {"a": 1}
        merged = merge_configs({}, over)
        assert merged == over


# ---------------------------------------------------------------------------
# Diff configs edge cases
# ---------------------------------------------------------------------------


class TestDiffConfigsExtended:
    def test_nested_diff(self):
        c1 = {"model": {"lr": 0.01, "layers": 3}}
        c2 = {"model": {"lr": 0.001, "layers": 3}}
        diffs = diff_configs(c1, c2)
        assert len(diffs) == 1
        assert diffs[0]["path"] == "model.lr"
        assert diffs[0]["type"] == "changed"
        assert diffs[0]["old"] == 0.01
        assert diffs[0]["new"] == 0.001

    def test_added_nested_key(self):
        c1 = {"a": {}}
        c2 = {"a": {"new_key": 42}}
        diffs = diff_configs(c1, c2)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "added"
        assert diffs[0]["path"] == "a.new_key"

    def test_removed_nested_key(self):
        c1 = {"a": {"gone": True}}
        c2 = {"a": {}}
        diffs = diff_configs(c1, c2)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "removed"

    def test_empty_configs(self):
        assert diff_configs({}, {}) == []

    def test_type_change(self):
        c1 = {"val": "string"}
        c2 = {"val": 42}
        diffs = diff_configs(c1, c2)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "changed"


# ---------------------------------------------------------------------------
# Round-trip CLI conversion
# ---------------------------------------------------------------------------


class TestCliRoundTrip:
    def test_simple_roundtrip(self):
        original = {"lr": 0.01, "epochs": 10, "name": "test"}
        args = config_to_cli_args(original)
        restored = cli_args_to_config(args)
        assert restored == original

    def test_nested_roundtrip(self):
        original = {"model": {"lr": 0.01}, "data": {"batch_size": 32}}
        args = config_to_cli_args(original)
        restored = cli_args_to_config(args)
        assert restored == original
