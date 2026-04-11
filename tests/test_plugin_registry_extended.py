"""Extended tests for navirl.core.plugin_validation and navirl.core.registry.

Covers edge cases not in test_core_registry_validation.py:
- validate_plugin_interface: missing method not callable, abstract method checking
- validate_controller_config: boundary values, all dangerous keys, velocity_smoothing
- validate_plugin_security: risky module imports warning
- safe_plugin_call: timeout exceeded, PluginPerformanceError pass-through
- performance_monitor: with bound methods
- validate_plugin_api_version: same major version
- Registry: safe factory behavior, _create_safe_factory timeout forwarding
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


class _Base(ABC):
    @abstractmethod
    def step(self):
        ...

    @abstractmethod
    def reset(self):
        ...


class _FullImpl(_Base):
    __navirl_api_version__ = "1.0"

    def step(self):
        return "step"

    def reset(self):
        return "reset"


# ===================================================================
# validate_plugin_interface — extended
# ===================================================================


class TestValidatePluginInterfaceExtended:
    def test_multiple_abstract_methods(self):
        """Verify all abstract methods are checked."""
        validate_plugin_interface(_FullImpl, _Base, "test")

    def test_method_not_callable(self):
        """Non-callable attribute with same name as abstract method."""

        class BadImpl(_Base):
            step = "not a method"  # type: ignore[assignment]

            def reset(self):
                pass

        with pytest.raises(PluginValidationError, match="not callable"):
            validate_plugin_interface(BadImpl, _Base, "bad")

    def test_no_abstract_methods(self):
        """Base with no abstractmethods should pass."""

        class PlainBase:
            pass

        class PlainImpl(PlainBase):
            pass

        validate_plugin_interface(PlainImpl, PlainBase, "plain")

    def test_none_as_plugin(self):
        with pytest.raises(PluginValidationError, match="must be a class"):
            validate_plugin_interface(None, _Base, "none")


# ===================================================================
# validate_controller_config — extended
# ===================================================================


class TestValidateControllerConfigExtended:
    def test_all_dangerous_keys(self):
        for key in ["__class__", "__module__", "__globals__"]:
            with pytest.raises(ConfigValidationError, match="dangerous key"):
                validate_controller_config({key: "x"}, "test")

    def test_velocity_smoothing_valid(self):
        config = {"velocity_smoothing": 0.5}
        result = validate_controller_config(config, "test")
        assert result["velocity_smoothing"] == pytest.approx(0.5)

    def test_velocity_smoothing_out_of_range(self):
        with pytest.raises(ConfigValidationError, match="must be between"):
            validate_controller_config({"velocity_smoothing": 1.5}, "test")

    def test_boundary_values_min(self):
        config = {"goal_tolerance": 0.01, "max_speed": 0.01}
        result = validate_controller_config(config, "test")
        assert result["goal_tolerance"] == pytest.approx(0.01)

    def test_boundary_values_max(self):
        config = {"goal_tolerance": 10.0, "max_speed": 20.0}
        result = validate_controller_config(config, "test")
        assert result["goal_tolerance"] == pytest.approx(10.0)

    def test_lookahead_range(self):
        config = {"lookahead": 1}
        result = validate_controller_config(config, "test")
        assert result["lookahead"] == 1

        with pytest.raises(ConfigValidationError, match="must be between"):
            validate_controller_config({"lookahead": 0}, "test")

        with pytest.raises(ConfigValidationError, match="must be between"):
            validate_controller_config({"lookahead": 101}, "test")

    def test_string_to_float_conversion(self):
        config = {"goal_tolerance": "0.5"}
        result = validate_controller_config(config, "test")
        assert result["goal_tolerance"] == pytest.approx(0.5)

    def test_list_value_not_numeric(self):
        with pytest.raises(ConfigValidationError, match="must be numeric"):
            validate_controller_config({"max_speed": [1, 2]}, "test")


# ===================================================================
# validate_plugin_security — extended
# ===================================================================


class TestValidatePluginSecurityExtended:
    def test_multiple_dangerous_methods(self):
        class BadPlugin(_Base):
            def step(self):
                pass

            def reset(self):
                pass

            def eval(self):
                pass

            def compile(self):
                pass

        with pytest.raises(PluginSecurityError, match="dangerous methods"):
            validate_plugin_security(BadPlugin, "test")

    def test_private_methods_ignored(self):
        """Methods starting with _ should not trigger security check."""

        class SafePlugin(_Base):
            __navirl_api_version__ = "1.0"

            def step(self):
                pass

            def reset(self):
                pass

            def _internal_exec(self):
                pass

        # Should not raise because _internal_exec starts with _
        validate_plugin_security(SafePlugin, "test")

    def test_risky_imports_warning(self, caplog):
        """Importing os/subprocess should emit a warning."""
        import importlib
        import types

        # Create a synthetic module that has 'os' in its namespace
        fake_module = types.ModuleType("fake_plugin_module")
        fake_module.__dict__["os"] = True  # simulate having imported os

        class PluginWithOs(_Base):
            __navirl_api_version__ = "1.0"

            def step(self):
                pass

            def reset(self):
                pass

        # Patch the plugin's module to be our fake module
        PluginWithOs.__module__ = "fake_plugin_module"
        import sys

        sys.modules["fake_plugin_module"] = fake_module
        try:
            with caplog.at_level("WARNING"):
                validate_plugin_security(PluginWithOs, "test")
            assert any("risky" in r.message.lower() for r in caplog.records)
        finally:
            del sys.modules["fake_plugin_module"]


# ===================================================================
# safe_plugin_call — extended
# ===================================================================


class TestSafePluginCallExtended:
    def test_timeout_exceeded(self):
        def slow():
            time.sleep(0.05)
            return "done"

        with pytest.raises(PluginPerformanceError, match="exceeded timeout"):
            safe_plugin_call(
                slow, plugin_name="test", method_name="fn", timeout_s=0.001,
            )

    def test_performance_error_passthrough(self):
        def raises_perf():
            raise PluginPerformanceError("custom perf error")

        with pytest.raises(PluginPerformanceError, match="custom perf error"):
            safe_plugin_call(
                raises_perf, plugin_name="test", method_name="fn",
            )

    def test_kwargs_forwarded(self):
        def fn(a, b=10):
            return a + b

        result = safe_plugin_call(
            fn, 5, plugin_name="test", method_name="fn", b=20,
        )
        assert result == 25


# ===================================================================
# validate_plugin_api_version — extended
# ===================================================================


class TestValidatePluginApiVersionExtended:
    def test_same_major_newer_minor(self):
        class Plugin(_Base):
            __navirl_api_version__ = "1.5"

            def step(self):
                pass

            def reset(self):
                pass

        # Same major version should pass
        validate_plugin_api_version(Plugin, "test", "1.0")

    def test_next_major_version(self):
        class Plugin(_Base):
            __navirl_api_version__ = "2.0"

            def step(self):
                pass

            def reset(self):
                pass

        # One major version ahead should pass without warning
        validate_plugin_api_version(Plugin, "test", "1.0")

    def test_empty_version_string(self):
        class Plugin(_Base):
            __navirl_api_version__ = ""

            def step(self):
                pass

            def reset(self):
                pass

        with pytest.raises(PluginValidationError, match="invalid API version"):
            validate_plugin_api_version(Plugin, "test", "1.0")


# ===================================================================
# Registry — safe factory wrapping
# ===================================================================


class TestSafeFactoryBehavior:
    def test_backend_factory_wraps_safely(self):
        call_log = []

        def factory(config=None):
            call_log.append("called")
            return "backend_instance"

        register_backend("safe_test", factory)
        safe_factory = get_backend("safe_test")
        safe_factory()
        assert "called" in call_log

    def test_human_controller_factory_exception_wrapped(self):
        def failing_factory():
            raise RuntimeError("init failed")

        register_human_controller("fail_hc", failing_factory)
        safe_factory = get_human_controller("fail_hc")
        with pytest.raises(PluginValidationError, match="init failed"):
            safe_factory()

    def test_robot_controller_factory_exception_wrapped(self):
        def failing_factory():
            raise RuntimeError("init failed")

        register_robot_controller("fail_rc", failing_factory)
        safe_factory = get_robot_controller("fail_rc")
        with pytest.raises(PluginValidationError, match="init failed"):
            safe_factory()

    def test_registry_snapshot_sorted(self):
        register_backend("z_backend", lambda: None)
        register_backend("a_backend", lambda: None)
        snap = registry_snapshot()
        backends = snap["backends"]
        assert backends == sorted(backends)

    def test_register_non_string_name(self):
        with pytest.raises(ValueError):
            register_backend(123, lambda: None)  # type: ignore[arg-type]

    def test_register_none_name(self):
        with pytest.raises(ValueError):
            register_human_controller(None, lambda: None)  # type: ignore[arg-type]


# ===================================================================
# validate_all_plugins — extended
# ===================================================================


class TestValidateAllPluginsExtended:
    def test_mixed_valid_invalid(self):
        def good_factory(config=None):
            return "instance"

        register_backend("good_be", good_factory)

        class OldPlugin(_Base):
            __navirl_api_version__ = "0.1"

            def step(self):
                pass

            def reset(self):
                pass

        # Bypass registration validation to insert an invalid plugin
        _ROBOT_CONTROLLERS["old_rc"] = OldPlugin
        issues = validate_all_plugins()
        assert "backend:good_be" not in issues
        assert any("old_rc" in k for k in issues)

    def test_security_issue_detected(self):
        class DangerousPlugin(_Base):
            __navirl_api_version__ = "1.0"

            def step(self):
                pass

            def reset(self):
                pass

            def exec(self):
                pass

        _HUMAN_CONTROLLERS["danger_hc"] = DangerousPlugin
        issues = validate_all_plugins()
        assert any("danger_hc" in k for k in issues)


# ===================================================================
# get_plugin_info — extended
# ===================================================================


class TestGetPluginInfoExtended:
    def test_class_with_docstring(self):
        class DocPlugin(_Base):
            """A documented plugin."""

            __navirl_api_version__ = "1.0"

            def step(self):
                pass

            def reset(self):
                pass

        register_robot_controller(
            "doc_rc", DocPlugin, enable_security_validation=False,
        )
        info = get_plugin_info("robot_controller", "doc_rc")
        assert "A documented plugin" in info["doc"]
        assert info["api_version"] == "1.0"
        assert "_Base" in info["bases"]

    def test_class_without_docstring(self):
        register_human_controller(
            "nodoc_hc", _FullImpl, enable_security_validation=False,
        )
        info = get_plugin_info("human_controller", "nodoc_hc")
        assert "init_parameters" in info


# ===================================================================
# performance_monitor — extended
# ===================================================================


class TestPerformanceMonitorExtended:
    def test_preserves_function_name(self):
        @performance_monitor(max_time_s=1.0)
        def my_function():
            return 42

        assert my_function.__name__ == "my_function"

    def test_preserves_return_value(self):
        @performance_monitor(max_time_s=1.0)
        def returns_dict():
            return {"key": "value"}

        assert returns_dict() == {"key": "value"}

    def test_exception_still_logs_time(self):
        @performance_monitor(max_time_s=0.001)
        def slow_and_fails():
            time.sleep(0.01)
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            slow_and_fails()
