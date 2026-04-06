"""Tests for navirl/utils/config_utils.py."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from navirl.utils.config_utils import (
    ValidationError,
    argparse_args_to_config,
    config_diff,
    config_to_argparse_args,
    dataclass_to_dict,
    dict_to_dataclass,
    flatten_dict,
    format_config_diff,
    interpolate_env_vars,
    load_json_config,
    load_simple_config,
    save_json_config,
    unflatten_dict,
)

# ---------------------------------------------------------------------------
# flatten_dict / unflatten_dict
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"a": {"b": 1, "c": {"d": 2}}}
        assert flatten_dict(d) == {"a.b": 1, "a.c.d": 2}

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        assert flatten_dict(d, separator="/") == {"a/b": 1}

    def test_parent_key_prefix(self):
        d = {"x": 5}
        assert flatten_dict(d, parent_key="root") == {"root.x": 5}

    def test_mixed_types_in_values(self):
        d = {"a": {"b": [1, 2]}, "c": "hello"}
        result = flatten_dict(d)
        assert result == {"a.b": [1, 2], "c": "hello"}


class TestUnflattenDict:
    def test_simple_keys(self):
        d = {"a": 1, "b": 2}
        assert unflatten_dict(d) == {"a": 1, "b": 2}

    def test_dotted_keys(self):
        d = {"a.b": 1, "a.c.d": 2}
        assert unflatten_dict(d) == {"a": {"b": 1, "c": {"d": 2}}}

    def test_empty_dict(self):
        assert unflatten_dict({}) == {}

    def test_roundtrip(self):
        original = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        assert unflatten_dict(flatten_dict(original)) == original


# ---------------------------------------------------------------------------
# JSON config I/O
# ---------------------------------------------------------------------------


class TestJsonConfig:
    def test_save_and_load(self, tmp_path):
        config = {"learning_rate": 0.01, "layers": [64, 32]}
        path = tmp_path / "config.json"
        save_json_config(config, path)
        loaded = load_json_config(path)
        assert loaded == config

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "config.json"
        save_json_config({"a": 1}, path)
        assert path.exists()

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_json_config(tmp_path / "nonexistent.json")

    def test_load_not_a_file(self, tmp_path):
        with pytest.raises(ValueError, match="not a regular file"):
            load_json_config(tmp_path)

    def test_load_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json_config(bad)

    def test_save_non_serializable_uses_str(self, tmp_path):
        path = tmp_path / "cfg.json"
        save_json_config({"path": Path("/tmp")}, path)
        loaded = load_json_config(path)
        assert loaded["path"] == "/tmp"


# ---------------------------------------------------------------------------
# Simple config (key=value)
# ---------------------------------------------------------------------------


class TestSimpleConfig:
    def test_basic_key_value(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("name = hello\ncount = 42\n")
        result = load_simple_config(cfg_file)
        assert result == {"name": "hello", "count": 42}

    def test_comments_and_blank_lines(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("# comment\n\nkey = value\n")
        result = load_simple_config(cfg_file)
        assert result == {"key": "value"}

    def test_nested_dot_keys(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("a.b = 1\na.c = 2\n")
        result = load_simple_config(cfg_file)
        assert result == {"a": {"b": 1, "c": 2}}

    def test_boolean_parsing(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("a = true\nb = false\nc = yes\nd = no\ne = on\nf = off\n")
        result = load_simple_config(cfg_file)
        assert result["a"] is True
        assert result["b"] is False
        assert result["c"] is True
        assert result["d"] is False
        assert result["e"] is True
        assert result["f"] is False

    def test_none_parsing(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("a = none\nb = null\n")
        result = load_simple_config(cfg_file)
        assert result["a"] is None
        assert result["b"] is None

    def test_float_parsing(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("lr = 0.001\n")
        result = load_simple_config(cfg_file)
        assert result["lr"] == pytest.approx(0.001)

    def test_list_parsing(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("sizes = 64, 32, 16\n")
        result = load_simple_config(cfg_file)
        assert result["sizes"] == [64, 32, 16]

    def test_quoted_string(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text('msg = "hello world"\n')
        result = load_simple_config(cfg_file)
        assert result["msg"] == "hello world"

    def test_inline_comments(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("x = 10 # ten\n")
        result = load_simple_config(cfg_file)
        assert result["x"] == 10

    def test_lines_without_equals_skipped(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("no equals here\nkey = val\n")
        result = load_simple_config(cfg_file)
        assert result == {"key": "val"}

    def test_single_quoted_string(self, tmp_path):
        cfg_file = tmp_path / "test.cfg"
        cfg_file.write_text("msg = 'hello'\n")
        result = load_simple_config(cfg_file)
        assert result["msg"] == "hello"


# ---------------------------------------------------------------------------
# Dataclass conversion
# ---------------------------------------------------------------------------


@dataclass
class Inner:
    x: int = 0
    y: int = 0


@dataclass
class Outer:
    name: str = ""
    inner: Inner = None


class TestDataclassConversion:
    def test_simple_dataclass_to_dict(self):
        obj = Inner(x=1, y=2)
        result = dataclass_to_dict(obj)
        assert result == {"x": 1, "y": 2}

    def test_nested_dataclass_to_dict(self):
        obj = Outer(name="test", inner=Inner(3, 4))
        result = dataclass_to_dict(obj)
        assert result == {"name": "test", "inner": {"x": 3, "y": 4}}

    def test_dataclass_with_list(self):
        @dataclass
        class WithList:
            items: list = None

        obj = WithList(items=[Inner(1, 2), Inner(3, 4)])
        result = dataclass_to_dict(obj)
        assert result["items"] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    def test_non_dataclass_returns_as_is(self):
        assert dataclass_to_dict(42) == 42
        assert dataclass_to_dict("hello") == "hello"

    def test_dict_to_dataclass_simple(self):
        obj = dict_to_dataclass(Inner, {"x": 5, "y": 10})
        assert obj.x == 5
        assert obj.y == 10

    def test_dict_to_dataclass_nested(self):
        """When __future__ annotations is active, field types are strings
        (forward references), so dict_to_dataclass passes them through as
        dicts rather than recursively constructing nested dataclasses."""
        data = {"name": "test", "inner": {"x": 7, "y": 8}}
        obj = dict_to_dataclass(Outer, data)
        assert obj.name == "test"
        # With PEP 563 (from __future__ import annotations), field types
        # are strings, so nested dataclasses aren't auto-constructed.
        assert obj.inner == {"x": 7, "y": 8}

    def test_dict_to_dataclass_missing_fields_use_defaults(self):
        obj = dict_to_dataclass(Inner, {})
        assert obj.x == 0
        assert obj.y == 0

    def test_dict_to_dataclass_non_dataclass_type(self):
        result = dict_to_dataclass(int, 42)
        assert result == 42

    def test_dataclass_with_dict_value(self):
        @dataclass
        class WithDict:
            meta: dict = None

        obj = WithDict(meta={"a": Inner(1, 2)})
        result = dataclass_to_dict(obj)
        assert result["meta"]["a"] == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------


class TestInterpolateEnvVars:
    def test_simple_var(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        result = interpolate_env_vars({"key": "${MY_VAR}"})
        assert result["key"] == "hello"

    def test_default_value(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = interpolate_env_vars({"key": "${MISSING_VAR:-default_val}"})
        assert result["key"] == "default_val"

    def test_strict_mode_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(KeyError, match="MISSING_VAR"):
            interpolate_env_vars({"key": "${MISSING_VAR}"}, strict=True)

    def test_non_strict_leaves_unexpanded(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = interpolate_env_vars({"key": "${MISSING_VAR}"}, strict=False)
        assert result["key"] == "${MISSING_VAR}"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        result = interpolate_env_vars({"db": {"host": "${DB_HOST}"}})
        assert result["db"]["host"] == "localhost"

    def test_list_values(self, monkeypatch):
        monkeypatch.setenv("PORT", "8080")
        result = interpolate_env_vars({"ports": ["${PORT}", 9090]})
        assert result["ports"] == ["8080", 9090]

    def test_non_string_passthrough(self):
        result = interpolate_env_vars({"count": 42, "flag": True})
        assert result == {"count": 42, "flag": True}

    def test_multiple_vars_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        result = interpolate_env_vars({"url": "${HOST}:${PORT}"})
        assert result["url"] == "localhost:5432"


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_fields(self):
        err = ValidationError(path="a.b", message="bad", severity="warning")
        assert err.path == "a.b"
        assert err.message == "bad"
        assert err.severity == "warning"

    def test_default_severity(self):
        err = ValidationError(path="x", message="msg")
        assert err.severity == "error"


# ---------------------------------------------------------------------------
# Argparse conversion
# ---------------------------------------------------------------------------


class TestArgparseConversion:
    def test_config_to_args(self):
        config = {"learning": {"rate": 0.01}, "epochs": 10}
        args = config_to_argparse_args(config)
        assert "--learning-rate" in args
        assert "0.01" in args
        assert "--epochs" in args
        assert "10" in args

    def test_bool_true_flag(self):
        args = config_to_argparse_args({"verbose": True})
        assert args == ["--verbose"]

    def test_bool_false_omitted(self):
        args = config_to_argparse_args({"verbose": False})
        assert args == []

    def test_list_values(self):
        args = config_to_argparse_args({"sizes": [64, 32]})
        assert args == ["--sizes", "64", "32"]

    def test_args_to_config_key_value(self):
        result = argparse_args_to_config(["--learning-rate", "0.01"])
        assert result["learning"]["rate"] == pytest.approx(0.01)

    def test_args_to_config_equals_format(self):
        result = argparse_args_to_config(["--epochs=10"])
        assert result["epochs"] == 10

    def test_args_to_config_bare_flag(self):
        result = argparse_args_to_config(["--verbose"])
        assert result["verbose"] is True

    def test_args_to_config_non_flag_skipped(self):
        result = argparse_args_to_config(["positional", "--key", "val"])
        assert "positional" not in result
        assert result["key"] == "val"

    def test_roundtrip(self):
        config = {"batch": {"size": 32}, "lr": 0.001}
        args = config_to_argparse_args(config)
        restored = argparse_args_to_config(args)
        assert restored["batch"]["size"] == 32
        assert restored["lr"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Config diff
# ---------------------------------------------------------------------------


class TestConfigDiff:
    def test_identical_configs(self):
        c = {"a": 1, "b": {"c": 2}}
        assert config_diff(c, c) == {}

    def test_changed_value(self):
        a = {"x": 1}
        b = {"x": 2}
        diff = config_diff(a, b)
        assert diff == {"x": (1, 2)}

    def test_added_key(self):
        a = {"x": 1}
        b = {"x": 1, "y": 2}
        diff = config_diff(a, b)
        assert diff == {"y": ("<missing>", 2)}

    def test_removed_key(self):
        a = {"x": 1, "y": 2}
        b = {"x": 1}
        diff = config_diff(a, b)
        assert diff == {"y": (2, "<missing>")}

    def test_nested_diff(self):
        a = {"a": {"b": 1}}
        b = {"a": {"b": 2}}
        diff = config_diff(a, b)
        assert diff == {"a.b": (1, 2)}

    def test_format_no_diff(self):
        assert format_config_diff({}) == "No differences."

    def test_format_added(self):
        diff = {"key": ("<missing>", 42)}
        text = format_config_diff(diff)
        assert "+ key: 42" in text

    def test_format_removed(self):
        diff = {"key": (42, "<missing>")}
        text = format_config_diff(diff)
        assert "- key: 42" in text

    def test_format_changed(self):
        diff = {"key": (1, 2)}
        text = format_config_diff(diff)
        assert "~ key: 1 -> 2" in text
