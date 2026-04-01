"""Tests for navirl/config/ module: validation, serialization, registry, presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from navirl.config.presets import (
    PRESETS,
    Preset,
    get_preset,
    list_presets,
    merge_presets,
)
from navirl.config.registry import ComponentRegistry
from navirl.config.serialization import (
    cli_args_to_config,
    config_to_cli_args,
    diff_configs,
    load_config,
    merge_configs,
    save_config,
)
from navirl.config.validation import ConfigValidator, SchemaBuilder

# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------


class TestConfigValidator:
    def test_valid_config(self):
        schema = {
            "lr": {"type": float, "required": True, "min": 0},
            "batch_size": {"type": int, "required": True},
        }
        config = {"lr": 0.001, "batch_size": 64}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is True
        assert errors == []

    def test_missing_required_key(self):
        schema = {
            "lr": {"type": float, "required": True},
        }
        config = {}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is False
        assert any("Missing required key" in e for e in errors)

    def test_wrong_type(self):
        schema = {
            "lr": {"type": float, "required": True},
        }
        config = {"lr": "not_a_float"}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is False
        assert any("expected type" in e for e in errors)

    def test_value_below_min(self):
        schema = {
            "lr": {"type": float, "required": False, "min": 0},
        }
        config = {"lr": -0.1}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is False
        assert any("minimum" in e for e in errors)

    def test_value_above_max(self):
        schema = {
            "gamma": {"type": float, "required": False, "max": 1.0},
        }
        config = {"gamma": 1.5}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is False
        assert any("maximum" in e for e in errors)

    def test_choices(self):
        schema = {
            "optimizer": {"type": str, "required": False, "choices": ["adam", "sgd"]},
        }
        config = {"optimizer": "rmsprop"}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is False
        assert any("not in" in e for e in errors)

    def test_valid_choices(self):
        schema = {
            "optimizer": {"type": str, "required": False, "choices": ["adam", "sgd"]},
        }
        config = {"optimizer": "adam"}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is True

    def test_nested_schema(self):
        schema = {
            "agent": {
                "type": dict,
                "required": True,
                "nested": {
                    "lr": {"type": float, "required": True, "min": 0},
                },
            },
        }
        config = {"agent": {"lr": -1.0}}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is False
        assert any("agent" in e for e in errors)

    def test_unknown_keys_allowed(self):
        schema = {"lr": {"type": float, "required": False}}
        config = {"lr": 0.01, "unknown_param": 42}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is True

    def test_empty_config_no_required(self):
        schema = {"lr": {"type": float, "required": False}}
        config = {}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is True

    @pytest.mark.parametrize(
        "value,expected_valid",
        [
            (0, True),
            (0.5, True),
            (1.0, True),
            (-0.1, False),
            (1.1, False),
        ],
    )
    def test_numeric_bounds(self, value, expected_valid):
        schema = {"x": {"type": (int, float), "required": True, "min": 0, "max": 1.0}}
        config = {"x": value}
        is_valid, _ = ConfigValidator.validate(config, schema)
        assert is_valid is expected_valid


# ---------------------------------------------------------------------------
# SchemaBuilder
# ---------------------------------------------------------------------------


class TestSchemaBuilder:
    def test_from_dataclass(self):
        @dataclass
        class MyConfig:
            lr: float = 1e-3
            epochs: int = 10
            name: str = "default"

        schema = SchemaBuilder.from_dataclass(MyConfig)
        assert "lr" in schema
        assert "epochs" in schema
        assert "name" in schema

    def test_not_a_dataclass(self):
        class NotDC:
            pass

        with pytest.raises(TypeError):
            SchemaBuilder.from_dataclass(NotDC)

    def test_generated_schema_validates(self):
        @dataclass
        class Cfg:
            lr: float = 1e-3
            batch_size: int = 64

        schema = SchemaBuilder.from_dataclass(Cfg)
        config = {"lr": 0.01, "batch_size": 128}
        is_valid, errors = ConfigValidator.validate(config, schema)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Serialization: save / load
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_json(self, tmp_path):
        config = {"lr": 0.01, "hidden_sizes": [64, 32], "name": "test"}
        path = tmp_path / "config.json"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_save_load_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        config = {"lr": 0.01, "gamma": 0.99}
        path = tmp_path / "config.yaml"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded["lr"] == pytest.approx(0.01)
        assert loaded["gamma"] == pytest.approx(0.99)

    def test_save_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format|Cannot infer"):
            save_config({}, tmp_path / "config.xyz")

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "config.json"
        save_config({"key": "value"}, path)
        assert path.exists()

    def test_explicit_format(self, tmp_path):
        config = {"a": 1}
        path = tmp_path / "cfg_no_ext"
        save_config(config, path, format="json")
        load_config(Path(str(path)))
        # May need the correct extension for load_config
        # If it fails, that's expected behavior


# ---------------------------------------------------------------------------
# CLI argument conversion
# ---------------------------------------------------------------------------


class TestCLIConversion:
    def test_config_to_cli_args(self):
        config = {"lr": 0.01, "epochs": 100}
        args = config_to_cli_args(config)
        assert "--lr" in args
        assert "0.01" in args

    def test_config_to_cli_nested(self):
        config = {"agent": {"lr": 0.001}}
        args = config_to_cli_args(config)
        assert "--agent.lr" in args

    def test_cli_args_to_config(self):
        args = ["--lr", "0.01", "--epochs", "100"]
        config = cli_args_to_config(args)
        assert config["lr"] == pytest.approx(0.01)
        assert config["epochs"] == 100

    def test_cli_args_nested(self):
        args = ["--agent.lr", "1e-4", "--agent.gamma", "0.99"]
        config = cli_args_to_config(args)
        assert "agent" in config
        assert config["agent"]["lr"] == pytest.approx(1e-4)

    def test_cli_flag(self):
        args = ["--verbose"]
        config = cli_args_to_config(args)
        assert config["verbose"] is True

    def test_round_trip(self):
        original = {"lr": 0.001, "batch_size": 64}
        args = config_to_cli_args(original)
        recovered = cli_args_to_config(args)
        assert recovered["lr"] == pytest.approx(original["lr"])
        assert recovered["batch_size"] == original["batch_size"]


# ---------------------------------------------------------------------------
# Merge / diff
# ---------------------------------------------------------------------------


class TestMergeDiff:
    def test_merge_simple(self):
        base = {"lr": 0.01, "gamma": 0.99}
        overrides = {"lr": 0.001}
        merged = merge_configs(base, overrides)
        assert merged["lr"] == 0.001
        assert merged["gamma"] == 0.99
        # Original should not be modified
        assert base["lr"] == 0.01

    def test_merge_nested(self):
        base = {"agent": {"lr": 0.01, "gamma": 0.99}}
        overrides = {"agent": {"lr": 0.001}}
        merged = merge_configs(base, overrides)
        assert merged["agent"]["lr"] == 0.001
        assert merged["agent"]["gamma"] == 0.99

    def test_merge_new_keys(self):
        base = {"a": 1}
        overrides = {"b": 2}
        merged = merge_configs(base, overrides)
        assert merged["a"] == 1
        assert merged["b"] == 2

    def test_diff_identical(self):
        config = {"lr": 0.01, "gamma": 0.99}
        diffs = diff_configs(config, config)
        assert len(diffs) == 0

    def test_diff_changed(self):
        c1 = {"lr": 0.01}
        c2 = {"lr": 0.001}
        diffs = diff_configs(c1, c2)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "changed"

    def test_diff_added_removed(self):
        c1 = {"a": 1}
        c2 = {"b": 2}
        diffs = diff_configs(c1, c2)
        types = {d["type"] for d in diffs}
        assert "added" in types
        assert "removed" in types


# ---------------------------------------------------------------------------
# ComponentRegistry
# ---------------------------------------------------------------------------


class TestComponentRegistry:
    def test_register_and_get(self):
        reg = ComponentRegistry("test")
        reg.register("my_class", dict)
        assert reg.get("my_class") is dict

    def test_register_duplicate_raises(self):
        reg = ComponentRegistry("test")
        reg.register("x", int)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("x", float)

    def test_get_missing_raises(self):
        reg = ComponentRegistry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_contains(self):
        reg = ComponentRegistry("test")
        reg.register("a", int)
        assert "a" in reg
        assert "b" not in reg

    def test_len(self):
        reg = ComponentRegistry("test")
        assert len(reg) == 0
        reg.register("a", int)
        assert len(reg) == 1

    def test_list_registered(self):
        reg = ComponentRegistry("test")
        reg.register("a", int, metadata={"version": 1})
        reg.register("b", float)
        items = reg.list_registered()
        assert len(items) == 2
        names = [n for n, _ in items]
        assert "a" in names
        assert "b" in names

    def test_from_config(self):
        reg = ComponentRegistry("test")

        class Foo:
            def __init__(self, x=1, y=2):
                self.x = x
                self.y = y

        reg.register("foo", Foo)
        instance = reg.from_config({"name": "foo", "x": 10, "y": 20})
        assert instance.x == 10
        assert instance.y == 20

    def test_from_config_missing_name(self):
        reg = ComponentRegistry("test")
        reg.register("foo", int)
        with pytest.raises(KeyError):
            reg.from_config({"not_name": "foo"})

    def test_metadata(self):
        reg = ComponentRegistry("test")
        reg.register("a", int, metadata={"desc": "integer type"})
        items = reg.list_registered()
        meta = dict(items)
        assert meta["a"]["desc"] == "integer type"


# ---------------------------------------------------------------------------
# Pre-built registries
# ---------------------------------------------------------------------------


class TestPrebuiltRegistries:
    def test_agents_registry_exists(self):
        from navirl.config.registry import agents_registry

        assert isinstance(agents_registry, ComponentRegistry)

    def test_environments_registry_exists(self):
        from navirl.config.registry import environments_registry

        assert isinstance(environments_registry, ComponentRegistry)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_known_presets(self):
        assert "debug" in PRESETS
        assert "fast_train" in PRESETS
        assert "full_train" in PRESETS

    def test_get_preset(self):
        preset = get_preset("debug")
        assert preset.name == "debug"
        assert isinstance(preset.env_config, dict)
        assert isinstance(preset.agent_config, dict)
        assert isinstance(preset.training_config, dict)

    def test_get_preset_returns_copy(self):
        p1 = get_preset("debug")
        p2 = get_preset("debug")
        p1.env_config["custom"] = True
        assert "custom" not in p2.env_config

    def test_get_preset_unknown(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_preset")

    def test_list_presets(self):
        presets = list_presets()
        assert len(presets) >= 3
        names = [name for name, _ in presets]
        assert "debug" in names

    def test_merge_presets(self):
        merged = merge_presets(
            "debug",
            {
                "env_config": {"num_humans": 100},
                "name": "custom_debug",
            },
        )
        assert merged.name == "custom_debug"
        assert merged.env_config["num_humans"] == 100
        # Original values should still be present
        assert "env_size" in merged.env_config or "time_limit" in merged.env_config

    def test_merge_presets_with_preset_object(self):
        base = get_preset("fast_train")
        merged = merge_presets(
            base,
            {
                "agent_config": {"learning_rate": 1e-5},
            },
        )
        assert merged.agent_config["learning_rate"] == 1e-5

    def test_preset_dataclass(self):
        p = Preset(
            name="custom",
            description="A custom preset",
            env_config={"size": 10},
            agent_config={"lr": 0.01},
            training_config={"steps": 1000},
        )
        assert p.name == "custom"
        assert p.description == "A custom preset"

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_all_presets_have_required_fields(self, preset_name):
        preset = get_preset(preset_name)
        assert preset.name == preset_name
        assert isinstance(preset.description, str)
        assert len(preset.description) > 0
        assert isinstance(preset.env_config, dict)
        assert isinstance(preset.agent_config, dict)
        assert isinstance(preset.training_config, dict)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    def test_empty_schema_validates_anything(self):
        is_valid, errors = ConfigValidator.validate({"any": "thing"}, {})
        assert is_valid is True

    def test_empty_config_empty_schema(self):
        is_valid, errors = ConfigValidator.validate({}, {})
        assert is_valid is True

    def test_merge_empty_overrides(self):
        base = {"lr": 0.01}
        merged = merge_configs(base, {})
        assert merged == base

    def test_merge_empty_base(self):
        merged = merge_configs({}, {"lr": 0.01})
        assert merged == {"lr": 0.01}

    def test_diff_empty_configs(self):
        diffs = diff_configs({}, {})
        assert diffs == []

    def test_registry_name(self):
        reg = ComponentRegistry("my_registry")
        assert reg.name == "my_registry"

    def test_cli_empty_args(self):
        config = cli_args_to_config([])
        assert config == {}
