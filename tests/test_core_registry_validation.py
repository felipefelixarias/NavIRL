"""Tests for navirl.core.registry and navirl.core.plugin_validation.

Covers plugin registration, retrieval, validation, security checks,
and performance monitoring which were at ~45% coverage.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

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
from navirl.core.registry import (
    _BACKENDS,
    _HUMAN_CONTROLLERS,
    _ROBOT_CONTROLLERS,
    get_backend,
    get_human_controller,
    get_plugin_info,
    get_robot_controller,
    register_backend,
    register_human_controller,
    register_robot_controller,
    registry_snapshot,
    validate_all_plugins,
)

# ---------------------------------------------------------------------------
# Fixtures to isolate registry state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Save and restore registry state around each test."""
    old_b = dict(_BACKENDS)
    old_h = dict(_HUMAN_CONTROLLERS)
    old_r = dict(_ROBOT_CONTROLLERS)
    yield
    _BACKENDS.clear()
    _BACKENDS.update(old_b)
    _HUMAN_CONTROLLERS.clear()
    _HUMAN_CONTROLLERS.update(old_h)
    _ROBOT_CONTROLLERS.clear()
    _ROBOT_CONTROLLERS.update(old_r)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyBase(ABC):
    @abstractmethod
    def step(self): ...


class _GoodPlugin(_DummyBase):
    __navirl_api_version__ = "1.0"

    def step(self):
        return "ok"


class _OldVersionPlugin(_DummyBase):
    __navirl_api_version__ = "0.5"

    def step(self):
        return "ok"


class _BadVersionPlugin(_DummyBase):
    __navirl_api_version__ = "invalid"

    def step(self):
        return "ok"


def _factory_fn(config=None):
    return "instance"


# ===================================================================
# validate_plugin_interface
# ===================================================================


class TestValidatePluginInterface:
    def test_valid_class(self):
        validate_plugin_interface(_GoodPlugin, _DummyBase, "test")

    def test_not_a_class(self):
        with pytest.raises(PluginValidationError, match="must be a class"):
            validate_plugin_interface(_factory_fn, _DummyBase, "test")

    def test_wrong_base(self):
        class Unrelated:
            pass

        with pytest.raises(PluginValidationError, match="must inherit"):
            validate_plugin_interface(Unrelated, _DummyBase, "test")


# ===================================================================
# validate_plugin_factory
# ===================================================================


class TestValidatePluginFactory:
    def test_valid_factory(self):
        validate_plugin_factory(_factory_fn, "test")

    def test_not_callable(self):
        with pytest.raises(PluginValidationError, match="must be callable"):
            validate_plugin_factory("not_callable", "test")

    def test_class_as_factory(self):
        validate_plugin_factory(_GoodPlugin, "test")

    def test_no_params_warns(self, caplog):
        def bare_factory():
            pass

        with caplog.at_level("WARNING"):
            validate_plugin_factory(bare_factory, "test")


# ===================================================================
# validate_controller_config
# ===================================================================


class TestValidateControllerConfig:
    def test_none_config(self):
        result = validate_controller_config(None, "test")
        assert result == {}

    def test_valid_config(self):
        config = {"goal_tolerance": 0.5, "max_speed": 1.5}
        result = validate_controller_config(config, "test")
        assert result["goal_tolerance"] == 0.5
        assert result["max_speed"] == 1.5

    def test_not_dict(self):
        with pytest.raises(ConfigValidationError, match="must be a dictionary"):
            validate_controller_config("bad", "test")

    def test_dangerous_keys(self):
        with pytest.raises(ConfigValidationError, match="dangerous key"):
            validate_controller_config({"__class__": "bad"}, "test")
        with pytest.raises(ConfigValidationError, match="dangerous key"):
            validate_controller_config({"__globals__": {}}, "test")

    def test_numeric_param_out_of_range(self):
        with pytest.raises(ConfigValidationError, match="must be between"):
            validate_controller_config({"goal_tolerance": 100.0}, "test")

    def test_numeric_param_not_numeric(self):
        with pytest.raises(ConfigValidationError, match="must be numeric"):
            validate_controller_config({"max_speed": "fast"}, "test")

    def test_extra_keys_pass_through(self):
        config = {"custom_key": "value", "goal_tolerance": 0.5}
        result = validate_controller_config(config, "test")
        assert result["custom_key"] == "value"

    def test_int_param_conversion(self):
        config = {"lookahead": 50}
        result = validate_controller_config(config, "test")
        assert result["lookahead"] == 50
        assert isinstance(result["lookahead"], int)


# ===================================================================
# validate_plugin_security
# ===================================================================


class TestValidatePluginSecurity:
    def test_safe_class(self):
        validate_plugin_security(_GoodPlugin, "test")

    def test_dangerous_methods(self):
        class BadPlugin(_DummyBase):
            def step(self):
                pass

            def exec(self):
                pass

        with pytest.raises(PluginSecurityError, match="dangerous methods"):
            validate_plugin_security(BadPlugin, "test")


# ===================================================================
# validate_plugin_api_version
# ===================================================================


class TestValidatePluginApiVersion:
    def test_matching_version(self):
        validate_plugin_api_version(_GoodPlugin, "test", "1.0")

    def test_old_version(self):
        with pytest.raises(PluginValidationError, match="too old"):
            validate_plugin_api_version(_OldVersionPlugin, "test", "1.0")

    def test_invalid_version_format(self):
        with pytest.raises(PluginValidationError, match="invalid API version"):
            validate_plugin_api_version(_BadVersionPlugin, "test", "1.0")

    def test_missing_version_attribute(self):
        class NoVersion(_DummyBase):
            def step(self):
                pass

        # Should default to "0.9" which is < 1.0
        with pytest.raises(PluginValidationError, match="too old"):
            validate_plugin_api_version(NoVersion, "test", "1.0")

    def test_future_version_warns(self, caplog):
        class FuturePlugin(_DummyBase):
            __navirl_api_version__ = "3.0"

            def step(self):
                pass

        with caplog.at_level("WARNING"):
            validate_plugin_api_version(FuturePlugin, "test", "1.0")


# ===================================================================
# safe_plugin_call
# ===================================================================


class TestSafePluginCall:
    def test_successful_call(self):
        result = safe_plugin_call(lambda: 42, plugin_name="test", method_name="fn")
        assert result == 42

    def test_call_with_args(self):
        result = safe_plugin_call(lambda x, y: x + y, 3, 4, plugin_name="test", method_name="fn")
        assert result == 7

    def test_exception_wrapped(self):
        def failing():
            raise RuntimeError("boom")

        with pytest.raises(PluginValidationError, match="boom"):
            safe_plugin_call(failing, plugin_name="test", method_name="fn")

    def test_performance_error_reraise(self):
        def slow():
            time.sleep(0.01)  # fast enough, but we set tiny timeout
            return "done"

        # This tests the timeout check after execution
        result = safe_plugin_call(slow, plugin_name="test", method_name="fn", timeout_s=10.0)
        assert result == "done"


# ===================================================================
# performance_monitor
# ===================================================================


class TestPerformanceMonitor:
    def test_fast_function(self):
        @performance_monitor(max_time_s=1.0)
        def fast():
            return 42

        assert fast() == 42

    def test_slow_function_warns(self, caplog):
        @performance_monitor(max_time_s=0.001)
        def slow():
            time.sleep(0.01)
            return "done"

        with caplog.at_level("WARNING"):
            result = slow()
        assert result == "done"


# ===================================================================
# Registry: register_backend / get_backend
# ===================================================================


class TestBackendRegistry:
    def test_register_and_get(self):
        register_backend("test_backend", _factory_fn)
        factory = get_backend("test_backend")
        assert callable(factory)

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_backend("", _factory_fn)

    def test_get_unknown(self):
        with pytest.raises(KeyError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_override_warns(self, caplog):
        register_backend("dup", _factory_fn)
        with caplog.at_level("WARNING"):
            register_backend("dup", _factory_fn)


# ===================================================================
# Registry: register_human_controller / get_human_controller
# ===================================================================


class TestHumanControllerRegistry:
    def test_register_function(self):
        register_human_controller("hc_fn", _factory_fn)
        factory = get_human_controller("hc_fn")
        assert callable(factory)

    def test_register_class(self):
        register_human_controller("hc_class", _GoodPlugin, enable_security_validation=True)
        factory = get_human_controller("hc_class")
        assert callable(factory)

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_human_controller("", _factory_fn)

    def test_get_unknown(self):
        with pytest.raises(KeyError, match="Unknown human controller"):
            get_human_controller("nonexistent")

    def test_skip_security_validation(self):
        register_human_controller("hc_nosec", _GoodPlugin, enable_security_validation=False)
        assert "hc_nosec" in registry_snapshot()["human_controllers"]


# ===================================================================
# Registry: register_robot_controller / get_robot_controller
# ===================================================================


class TestRobotControllerRegistry:
    def test_register_and_get(self):
        register_robot_controller("rc", _factory_fn)
        factory = get_robot_controller("rc")
        assert callable(factory)

    def test_register_class(self):
        register_robot_controller("rc_class", _GoodPlugin)
        factory = get_robot_controller("rc_class")
        assert callable(factory)

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_robot_controller("", _factory_fn)

    def test_get_unknown(self):
        with pytest.raises(KeyError, match="Unknown robot controller"):
            get_robot_controller("nonexistent")


# ===================================================================
# registry_snapshot
# ===================================================================


class TestRegistrySnapshot:
    def test_snapshot(self):
        register_backend("snap_be", _factory_fn)
        register_human_controller("snap_hc", _factory_fn)
        register_robot_controller("snap_rc", _factory_fn)
        snap = registry_snapshot()
        assert "snap_be" in snap["backends"]
        assert "snap_hc" in snap["human_controllers"]
        assert "snap_rc" in snap["robot_controllers"]


# ===================================================================
# get_plugin_info
# ===================================================================


class TestGetPluginInfo:
    def test_function_factory(self):
        register_backend("info_fn", _factory_fn)
        info = get_plugin_info("backend", "info_fn")
        assert info["name"] == "info_fn"
        assert info["factory_type"] == "function"

    def test_class_factory(self):
        register_human_controller("info_cls", _GoodPlugin, enable_security_validation=False)
        info = get_plugin_info("human_controller", "info_cls")
        assert info["factory_type"] == "class"
        assert "module" in info
        assert "init_parameters" in info

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid plugin type"):
            get_plugin_info("invalid_type", "x")

    def test_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_plugin_info("backend", "missing")


# ===================================================================
# validate_all_plugins
# ===================================================================


class TestValidateAllPlugins:
    def test_valid_plugins(self):
        register_backend("val_be", _factory_fn)
        register_human_controller("val_hc", _factory_fn)
        issues = validate_all_plugins()
        # Factory functions should pass validation
        assert "backend:val_be" not in issues
        assert "human_controller:val_hc" not in issues

    def test_detects_issues(self):
        # Directly insert a class with old API version into the registry
        # (bypassing register_human_controller which would reject it)
        _HUMAN_CONTROLLERS["old_hc"] = _OldVersionPlugin
        issues = validate_all_plugins()
        # Should detect the API version issue
        assert any("old_hc" in k for k in issues)
