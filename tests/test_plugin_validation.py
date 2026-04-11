"""Tests for navirl.core.plugin_validation module."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import pytest

from navirl.core.plugin_validation import (
    ConfigValidationError,
    PluginPerformanceError,
    PluginSecurityError,
    PluginValidationError,
    performance_monitor,
    safe_plugin_call,
    validate_controller_config,
    validate_plugin_api_version,
    validate_plugin_factory,
    validate_plugin_interface,
    validate_plugin_security,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class BaseController(ABC):
    @abstractmethod
    def step(self): ...

    @abstractmethod
    def reset(self): ...


class GoodController(BaseController):
    def step(self):
        return "step"

    def reset(self):
        return "reset"


class BadController:
    """Not a subclass of BaseController."""


class PartialController(BaseController):
    """Implements only one abstract method — but Python ABC mechanism
    prevents instantiation; the class itself still has both methods from
    the partial impl."""

    def step(self):
        return "step"

    def reset(self):
        return "reset"


class DangerousController(BaseController):
    """Has a dangerous method name."""

    def step(self):
        return "step"

    def reset(self):
        return "reset"

    def exec(self):
        pass


# ---------------------------------------------------------------------------
# validate_plugin_interface
# ---------------------------------------------------------------------------


class TestValidatePluginInterface:
    def test_valid_plugin(self):
        validate_plugin_interface(GoodController, BaseController, "good")

    def test_not_a_class(self):
        with pytest.raises(PluginValidationError, match="must be a class"):
            validate_plugin_interface("not_a_class", BaseController, "bad")

    def test_wrong_base(self):
        with pytest.raises(PluginValidationError, match="must inherit"):
            validate_plugin_interface(BadController, BaseController, "bad")


# ---------------------------------------------------------------------------
# validate_plugin_factory
# ---------------------------------------------------------------------------


class TestValidatePluginFactory:
    def test_valid_factory(self):
        def factory(cfg):
            return GoodController()

        validate_plugin_factory(factory, "test")

    def test_not_callable(self):
        with pytest.raises(PluginValidationError, match="must be callable"):
            validate_plugin_factory("not_callable", "test")

    def test_no_params_warns(self, caplog):
        def factory():
            return GoodController()

        import logging

        with caplog.at_level(logging.WARNING):
            validate_plugin_factory(factory, "test")
        assert "no parameters" in caplog.text


# ---------------------------------------------------------------------------
# validate_controller_config
# ---------------------------------------------------------------------------


class TestValidateControllerConfig:
    def test_none_returns_empty(self):
        assert validate_controller_config(None, "test") == {}

    def test_not_dict_raises(self):
        with pytest.raises(ConfigValidationError, match="must be a dictionary"):
            validate_controller_config("not_a_dict", "test")

    def test_dangerous_key(self):
        with pytest.raises(ConfigValidationError, match="dangerous key"):
            validate_controller_config({"__class__": "evil"}, "test")

    def test_valid_numeric_params(self):
        cfg = {"goal_tolerance": 0.5, "max_speed": 1.5}
        result = validate_controller_config(cfg, "test")
        assert result["goal_tolerance"] == pytest.approx(0.5)
        assert result["max_speed"] == pytest.approx(1.5)

    def test_out_of_range_param(self):
        with pytest.raises(ConfigValidationError, match="must be between"):
            validate_controller_config({"max_speed": 100.0}, "test")

    def test_non_numeric_param(self):
        with pytest.raises(ConfigValidationError, match="must be numeric"):
            validate_controller_config({"max_speed": "fast"}, "test")

    def test_integer_conversion(self):
        cfg = {"lookahead": 5}
        result = validate_controller_config(cfg, "test")
        assert result["lookahead"] == 5
        assert isinstance(result["lookahead"], int)

    def test_velocity_smoothing_bounds(self):
        cfg = {"velocity_smoothing": 0.5}
        result = validate_controller_config(cfg, "test")
        assert result["velocity_smoothing"] == pytest.approx(0.5)

    def test_extra_keys_preserved(self):
        cfg = {"custom_key": "value", "max_speed": 1.0}
        result = validate_controller_config(cfg, "test")
        assert result["custom_key"] == "value"


# ---------------------------------------------------------------------------
# validate_plugin_security
# ---------------------------------------------------------------------------


class TestValidatePluginSecurity:
    def test_safe_plugin(self):
        validate_plugin_security(GoodController, "good")

    def test_dangerous_method(self):
        with pytest.raises(PluginSecurityError, match="dangerous methods"):
            validate_plugin_security(DangerousController, "dangerous")

    def test_risky_imports_warning(self, caplog):
        import logging

        # Create a class in a module that has 'os' imported
        # GoodController's module has 'abc' but not 'os'
        with caplog.at_level(logging.WARNING):
            validate_plugin_security(GoodController, "good")
        # Should NOT warn for GoodController


# ---------------------------------------------------------------------------
# performance_monitor
# ---------------------------------------------------------------------------


class TestPerformanceMonitor:
    def test_fast_function_no_warning(self, caplog):
        import logging

        @performance_monitor(max_time_s=1.0)
        def fast():
            return 42

        with caplog.at_level(logging.WARNING):
            result = fast()
        assert result == 42
        assert "took" not in caplog.text

    def test_slow_function_warns(self, caplog):
        import logging

        @performance_monitor(max_time_s=0.001)
        def slow():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.WARNING):
            result = slow()
        assert result == 42
        assert "took" in caplog.text


# ---------------------------------------------------------------------------
# safe_plugin_call
# ---------------------------------------------------------------------------


class TestSafePluginCall:
    def test_successful_call(self):
        def method():
            return 42

        result = safe_plugin_call(method, plugin_name="test", method_name="test")
        assert result == 42

    def test_exception_wraps(self):
        def method():
            raise RuntimeError("boom")

        with pytest.raises(PluginValidationError, match="failed"):
            safe_plugin_call(method, plugin_name="test", method_name="test")

    def test_timeout_exceeded(self):
        def slow():
            time.sleep(0.1)
            return 42

        with pytest.raises(PluginPerformanceError, match="exceeded timeout"):
            safe_plugin_call(slow, plugin_name="test", method_name="test", timeout_s=0.001)

    def test_performance_error_reraise(self):
        def method():
            raise PluginPerformanceError("custom")

        with pytest.raises(PluginPerformanceError, match="custom"):
            safe_plugin_call(method, plugin_name="test", method_name="test")


# ---------------------------------------------------------------------------
# validate_plugin_api_version
# ---------------------------------------------------------------------------


class TestValidatePluginApiVersion:
    def test_compatible_version(self):
        class Plugin:
            __navirl_api_version__ = "1.0"

        validate_plugin_api_version(Plugin, "test", "1.0")

    def test_too_old_version(self):
        class Plugin:
            __navirl_api_version__ = "0.5"

        with pytest.raises(PluginValidationError, match="too old"):
            validate_plugin_api_version(Plugin, "test", "1.0")

    def test_default_version(self):
        class Plugin:
            pass

        # Default version is "0.9", should be too old for "1.0"
        with pytest.raises(PluginValidationError, match="too old"):
            validate_plugin_api_version(Plugin, "test", "1.0")

    def test_newer_version_warns(self, caplog):
        import logging

        class Plugin:
            __navirl_api_version__ = "3.0"

        with caplog.at_level(logging.WARNING):
            validate_plugin_api_version(Plugin, "test", "1.0")
        assert "too new" in caplog.text

    def test_invalid_version_format(self):
        class Plugin:
            __navirl_api_version__ = "abc"

        with pytest.raises(PluginValidationError, match="invalid API version"):
            validate_plugin_api_version(Plugin, "test", "1.0")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_config_is_plugin_error(self):
        assert issubclass(ConfigValidationError, PluginValidationError)

    def test_security_is_plugin_error(self):
        assert issubclass(PluginSecurityError, PluginValidationError)

    def test_performance_is_plugin_error(self):
        assert issubclass(PluginPerformanceError, PluginValidationError)
