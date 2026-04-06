"""Tests for navirl/utils/config_utils.py."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from navirl.utils.config_utils import (
    ValidationError,
    _parse_value,
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
    def test_flat_already(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_single_nesting(self):
        d = {"a": {"b": 1}}
        assert flatten_dict(d) == {"a.b": 1}

    def test_deep_nesting(self):
        d = {"a": {"b": {"c": {"d": 42}}}}
        assert flatten_dict(d) == {"a.b.c.d": 42}

    def test_mixed_nesting(self):
        d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = flatten_dict(d)
        assert result == {"a": 1, "b.c": 2, "b.d.e": 3}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        assert flatten_dict(d, separator="/") == {"a/b": 1}

    def test_with_parent_key(self):
        d = {"x": 1}
        assert flatten_dict(d, parent_key="root") == {"root.x": 1}

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_preserves_non_dict_values(self):
        d = {"a": [1, 2], "b": {"c": None}}
        result = flatten_dict(d)
        assert result == {"a": [1, 2], "b.c": None}


class TestUnflattenDict:
    def test_single_key(self):
        assert unflatten_dict({"a.b": 1}) == {"a": {"b": 1}}

    def test_multiple_keys_same_parent(self):
        d = {"a.b": 1, "a.c": 2}
        assert unflatten_dict(d) == {"a": {"b": 1, "c": 2}}

    def test_deep_key(self):
        d = {"a.b.c.d": 42}
        assert unflatten_dict(d) == {"a": {"b": {"c": {"d": 42}}}}

    def test_roundtrip(self):
        original = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        assert unflatten_dict(flatten_dict(original)) == original

    def test_custom_separator(self):
        d = {"a/b": 1}
        assert unflatten_dict(d, separator="/") == {"a": {"b": 1}}

    def test_flat_keys_stay_flat(self):
        d = {"x": 10, "y": 20}
        assert unflatten_dict(d) == {"x": 10, "y": 20}

    def test_empty_dict(self):
        assert unflatten_dict({}) == {}


# ---------------------------------------------------------------------------
# JSON config loading / saving
# ---------------------------------------------------------------------------


class TestJsonConfig:
    def test_save_and_load(self, tmp_path):
        cfg = {"model": {"lr": 0.001}, "epochs": 10}
        path = tmp_path / "config.json"
        save_json_config(cfg, path)
        loaded = load_json_config(path)
        assert loaded == cfg

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "config.json"
        save_json_config({"a": 1}, path)
        assert path.exists()
        assert load_json_config(path) == {"a": 1}

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json_config(tmp_path / "missing.json")

    def test_load_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a regular file"):
            load_json_config(tmp_path)

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json_config(path)

    def test_save_handles_non_serializable(self, tmp_path):
        """Non-serializable values should be converted via default=str."""
        path = tmp_path / "cfg.json"
        save_json_config({"p": Path("/foo")}, path)
        loaded = load_json_config(path)
        assert loaded["p"] == "/foo"


# ---------------------------------------------------------------------------
# Simple config (key=value format)
# ---------------------------------------------------------------------------


class TestSimpleConfig:
    def test_basic_key_value(self, tmp_path):
        path = tmp_path / "cfg.ini"
        path.write_text("name = hello\ncount = 42\n")
        result = load_simple_config(path)
        assert result == {"name": "hello", "count": 42}

    def test_comments_and_blanks(self, tmp_path):
        path = tmp_path / "cfg.ini"
        path.write_text("# comment\n\nkey = value\n")
        result = load_simple_config(path)
        assert result == {"key": "value"}

    def test_inline_comment(self, tmp_path):
        path = tmp_path / "cfg.ini"
        path.write_text("key = 10 # inline comment\n")
        result = load_simple_config(path)
        assert result == {"key": 10}

    def test_dotted_keys_unflatten(self, tmp_path):
        path = tmp_path / "cfg.ini"
        path.write_text("model.lr = 0.01\nmodel.batch = 32\n")
        result = load_simple_config(path)
        assert result == {"model": {"lr": 0.01, "batch": 32}}

    def test_lines_without_equals_skipped(self, tmp_path):
        path = tmp_path / "cfg.ini"
        path.write_text("no_equals_here\nkey = val\n")
        result = load_simple_config(path)
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# _parse_value
# ---------------------------------------------------------------------------


class TestParseValue:
    @pytest.mark.parametrize("s", ["true", "True", "yes", "YES", "on", "ON"])
    def test_truthy(self, s):
        assert _parse_value(s) is True

    @pytest.mark.parametrize("s", ["false", "False", "no", "NO", "off", "OFF"])
    def test_falsy(self, s):
        assert _parse_value(s) is False

    @pytest.mark.parametrize("s", ["none", "None", "null", "NULL"])
    def test_none(self, s):
        assert _parse_value(s) is None

    def test_integer(self):
        assert _parse_value("42") == 42

    def test_negative_integer(self):
        assert _parse_value("-7") == -7

    def test_float(self):
        assert _parse_value("3.14") == pytest.approx(3.14)

    def test_comma_list(self):
        assert _parse_value("1,2,3") == [1, 2, 3]

    def test_comma_list_strings(self):
        assert _parse_value("a,b,c") == ["a", "b", "c"]

    def test_quoted_string(self):
        assert _parse_value('"hello world"') == "hello world"

    def test_single_quoted(self):
        assert _parse_value("'hello'") == "hello"

    def test_plain_string(self):
        assert _parse_value("hello") == "hello"


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


@dataclass
class Inner:
    x: int = 0
    y: int = 0


@dataclass
class Outer:
    name: str = "test"
    inner: Inner = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.inner is None:
            self.inner = Inner()


class TestDataclassConversion:
    def test_to_dict(self):
        obj = Outer(name="a", inner=Inner(x=1, y=2))
        d = dataclass_to_dict(obj)
        assert d == {"name": "a", "inner": {"x": 1, "y": 2}}

    def test_from_dict(self):
        data = {"name": "b", "inner": {"x": 3, "y": 4}}
        obj = dict_to_dataclass(Outer, data)
        assert obj.name == "b"
        # With `from __future__ import annotations`, field types are strings
        # so dict_to_dataclass passes through dict values for string annotations
        assert obj.inner == {"x": 3, "y": 4}

    def test_roundtrip_flat(self):
        """Roundtrip works for flat dataclasses without nested types."""
        original = Inner(x=5, y=6)
        d = dataclass_to_dict(original)
        restored = dict_to_dataclass(Inner, d)
        assert restored.x == original.x
        assert restored.y == original.y

    def test_to_dict_with_list(self):
        @dataclass
        class WithList:
            items: list = None  # type: ignore[assignment]

            def __post_init__(self):
                if self.items is None:
                    self.items = []

        obj = WithList(items=[Inner(1, 2), Inner(3, 4)])
        d = dataclass_to_dict(obj)
        assert d == {"items": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}

    def test_non_dataclass_passthrough(self):
        result = dict_to_dataclass(int, {"a": 1})  # type: ignore[arg-type]
        assert result == {"a": 1}

    def test_missing_keys_use_defaults(self):
        obj = dict_to_dataclass(Inner, {})
        assert obj.x == 0
        assert obj.y == 0


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------


class TestEnvVarInterpolation:
    def test_basic_substitution(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        cfg = {"key": "${MY_VAR}"}
        result = interpolate_env_vars(cfg)
        assert result == {"key": "hello"}

    def test_default_value(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        cfg = {"key": "${MISSING_VAR:-fallback}"}
        result = interpolate_env_vars(cfg)
        assert result == {"key": "fallback"}

    def test_missing_non_strict(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        cfg = {"key": "${MISSING_VAR}"}
        result = interpolate_env_vars(cfg)
        assert result == {"key": "${MISSING_VAR}"}

    def test_missing_strict_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        cfg = {"key": "${MISSING_VAR}"}
        with pytest.raises(KeyError):
            interpolate_env_vars(cfg, strict=True)

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        cfg = {"db": {"host": "${DB_HOST}"}}
        result = interpolate_env_vars(cfg, strict=True)
        assert result == {"db": {"host": "localhost"}}

    def test_list_values(self, monkeypatch):
        monkeypatch.setenv("PORT", "8080")
        cfg = {"ports": ["${PORT}", 9090]}
        result = interpolate_env_vars(cfg)
        assert result == {"ports": ["8080", 9090]}

    def test_non_string_passthrough(self):
        cfg = {"count": 42, "flag": True}
        result = interpolate_env_vars(cfg)
        assert result == {"count": 42, "flag": True}

    def test_multiple_vars_in_one_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        cfg = {"url": "${HOST}:${PORT}"}
        result = interpolate_env_vars(cfg, strict=True)
        assert result == {"url": "localhost:5432"}


# ---------------------------------------------------------------------------
# Argparse integration
# ---------------------------------------------------------------------------


class TestArgparseIntegration:
    def test_config_to_args(self):
        cfg = {"learning_rate": 0.01, "epochs": 10}
        args = config_to_argparse_args(cfg)
        # flatten_dict uses dots as separator, which become dashes
        assert "--learning_rate" in args
        assert "0.01" in args
        assert "--epochs" in args
        assert "10" in args

    def test_bool_flag(self):
        cfg = {"verbose": True, "quiet": False}
        args = config_to_argparse_args(cfg)
        assert "--verbose" in args
        assert "--quiet" not in args

    def test_list_value(self):
        cfg = {"tags": ["a", "b"]}
        args = config_to_argparse_args(cfg)
        assert args == ["--tags", "a", "b"]

    def test_nested_config(self):
        cfg = {"model": {"lr": 0.01}}
        args = config_to_argparse_args(cfg)
        assert "--model-lr" in args

    def test_args_to_config_key_value(self):
        args = ["--learning-rate", "0.01", "--epochs", "10"]
        cfg = argparse_args_to_config(args)
        assert cfg["learning"]["rate"] == pytest.approx(0.01)
        assert cfg["epochs"] == 10

    def test_args_to_config_equals_format(self):
        args = ["--epochs=20"]
        cfg = argparse_args_to_config(args)
        assert cfg["epochs"] == 20

    def test_args_to_config_flag(self):
        args = ["--verbose", "--count", "5"]
        cfg = argparse_args_to_config(args)
        assert cfg["verbose"] is True
        assert cfg["count"] == 5

    def test_non_flag_args_skipped(self):
        args = ["positional", "--key", "val"]
        cfg = argparse_args_to_config(args)
        assert "key" in cfg
        assert "positional" not in str(cfg)


# ---------------------------------------------------------------------------
# Config diff
# ---------------------------------------------------------------------------


class TestConfigDiff:
    def test_identical_configs(self):
        cfg = {"a": 1, "b": {"c": 2}}
        assert config_diff(cfg, cfg) == {}

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
        a = {"m": {"lr": 0.01}}
        b = {"m": {"lr": 0.001}}
        diff = config_diff(a, b)
        assert "m.lr" in diff


class TestFormatConfigDiff:
    def test_no_differences(self):
        assert format_config_diff({}) == "No differences."

    def test_addition(self):
        diff = {"x": ("<missing>", 42)}
        output = format_config_diff(diff)
        assert "+ x: 42" in output

    def test_removal(self):
        diff = {"x": (42, "<missing>")}
        output = format_config_diff(diff)
        assert "- x: 42" in output

    def test_change(self):
        diff = {"x": (1, 2)}
        output = format_config_diff(diff)
        assert "~ x: 1 -> 2" in output


# ---------------------------------------------------------------------------
# ValidationError dataclass
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_defaults(self):
        err = ValidationError(path="a.b", message="bad value")
        assert err.severity == "error"

    def test_custom_severity(self):
        err = ValidationError(path="a", message="hmm", severity="warning")
        assert err.severity == "warning"
