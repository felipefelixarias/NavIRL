"""Tests for domain-specific validators and strict mode in navirl.config.validation."""

from __future__ import annotations

import pytest

from navirl.config.validation import (
    ConfigValidator,
    validate_agent_config,
    validate_env_config,
    validate_training_config,
)


# ---------------------------------------------------------------------------
# ConfigValidator strict mode
# ---------------------------------------------------------------------------


class TestConfigValidatorStrictMode:
    def test_strict_rejects_unknown_keys(self):
        schema = {"lr": {"type": float, "required": False}}
        config = {"lr": 0.01, "unknown_param": 42}
        is_valid, errors = ConfigValidator.validate(config, schema, strict=True)
        assert is_valid is False
        assert any("Unknown key" in e for e in errors)
        assert any("strict" in e.lower() for e in errors)

    def test_strict_allows_known_keys(self):
        schema = {"lr": {"type": float, "required": False}}
        config = {"lr": 0.01}
        is_valid, errors = ConfigValidator.validate(config, schema, strict=True)
        assert is_valid is True

    def test_strict_empty_config(self):
        schema = {"lr": {"type": float, "required": False}}
        is_valid, errors = ConfigValidator.validate({}, schema, strict=True)
        assert is_valid is True

    def test_strict_multiple_unknown_keys(self):
        schema = {"lr": {"type": float, "required": False}}
        config = {"lr": 0.01, "extra1": 1, "extra2": 2}
        is_valid, errors = ConfigValidator.validate(config, schema, strict=True)
        assert is_valid is False
        assert len(errors) == 2

    def test_non_strict_allows_unknown_keys(self):
        schema = {"lr": {"type": float, "required": False}}
        config = {"lr": 0.01, "unknown_param": 42}
        is_valid, errors = ConfigValidator.validate(config, schema, strict=False)
        assert is_valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# validate_agent_config
# ---------------------------------------------------------------------------


class TestValidateAgentConfig:
    def test_valid_agent_config(self):
        config = {
            "hidden_sizes": [128, 64],
            "learning_rate": 1e-4,
            "batch_size": 32,
            "gamma": 0.99,
            "tau": 0.005,
        }
        errors = validate_agent_config(config)
        assert errors == []

    def test_missing_required_hidden_sizes(self):
        config = {"learning_rate": 1e-4}
        errors = validate_agent_config(config)
        assert any("hidden_sizes" in e for e in errors)

    def test_missing_required_learning_rate(self):
        config = {"hidden_sizes": [64]}
        errors = validate_agent_config(config)
        assert any("learning_rate" in e for e in errors)

    def test_negative_learning_rate(self):
        config = {"hidden_sizes": [64], "learning_rate": -0.01}
        errors = validate_agent_config(config)
        assert any("minimum" in e for e in errors)

    def test_gamma_out_of_range(self):
        config = {"hidden_sizes": [64], "learning_rate": 1e-4, "gamma": 1.5}
        errors = validate_agent_config(config)
        assert any("maximum" in e for e in errors)

    def test_gamma_below_zero(self):
        config = {"hidden_sizes": [64], "learning_rate": 1e-4, "gamma": -0.1}
        errors = validate_agent_config(config)
        assert any("minimum" in e for e in errors)

    def test_tau_out_of_range(self):
        config = {"hidden_sizes": [64], "learning_rate": 1e-4, "tau": 2.0}
        errors = validate_agent_config(config)
        assert any("maximum" in e for e in errors)

    def test_batch_size_zero(self):
        config = {"hidden_sizes": [64], "learning_rate": 1e-4, "batch_size": 0}
        errors = validate_agent_config(config)
        assert any("minimum" in e for e in errors)

    def test_wrong_type_learning_rate(self):
        config = {"hidden_sizes": [64], "learning_rate": "fast"}
        errors = validate_agent_config(config)
        assert any("expected type" in e for e in errors)

    def test_minimal_valid_config(self):
        config = {"hidden_sizes": [32], "learning_rate": 0.001}
        errors = validate_agent_config(config)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_env_config
# ---------------------------------------------------------------------------


class TestValidateEnvConfig:
    def test_valid_env_config(self):
        config = {"num_humans": 5, "env_size": 10.0, "time_limit": 100}
        errors = validate_env_config(config)
        assert errors == []

    def test_empty_config_ok(self):
        # All env keys are optional
        errors = validate_env_config({})
        assert errors == []

    def test_negative_num_humans(self):
        config = {"num_humans": -1}
        errors = validate_env_config(config)
        assert any("minimum" in e for e in errors)

    def test_negative_env_size(self):
        config = {"env_size": -5.0}
        errors = validate_env_config(config)
        assert any("minimum" in e for e in errors)

    def test_negative_time_limit(self):
        config = {"time_limit": -10}
        errors = validate_env_config(config)
        assert any("minimum" in e for e in errors)

    def test_zero_values_ok(self):
        config = {"num_humans": 0, "env_size": 0, "time_limit": 0}
        errors = validate_env_config(config)
        assert errors == []

    def test_wrong_type_num_humans(self):
        config = {"num_humans": "five"}
        errors = validate_env_config(config)
        assert any("expected type" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_training_config
# ---------------------------------------------------------------------------


class TestValidateTrainingConfig:
    def test_valid_training_config(self):
        config = {
            "total_steps": 100000,
            "eval_interval": 5000,
            "log_interval": 1000,
            "seed": 42,
        }
        errors = validate_training_config(config)
        assert errors == []

    def test_missing_required_total_steps(self):
        config = {"eval_interval": 5000}
        errors = validate_training_config(config)
        assert any("total_steps" in e for e in errors)

    def test_total_steps_zero(self):
        config = {"total_steps": 0}
        errors = validate_training_config(config)
        assert any("minimum" in e for e in errors)

    def test_eval_interval_zero(self):
        config = {"total_steps": 1000, "eval_interval": 0}
        errors = validate_training_config(config)
        assert any("minimum" in e for e in errors)

    def test_log_interval_zero(self):
        config = {"total_steps": 1000, "log_interval": 0}
        errors = validate_training_config(config)
        assert any("minimum" in e for e in errors)

    def test_seed_is_optional(self):
        config = {"total_steps": 1000}
        errors = validate_training_config(config)
        assert errors == []

    def test_wrong_type_total_steps(self):
        config = {"total_steps": "a lot"}
        errors = validate_training_config(config)
        assert any("expected type" in e for e in errors)

    def test_minimal_valid_config(self):
        config = {"total_steps": 1}
        errors = validate_training_config(config)
        assert errors == []
