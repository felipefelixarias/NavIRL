"""Coverage gaps for navirl/evaluation: attention_visualization torch path,
AgentComparison.statistical_test scipy path, and BenchmarkResults.plot_comparison.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from navirl.evaluation.analysis import attention_visualization
from navirl.evaluation.benchmark import BenchmarkResults
from navirl.evaluation.comparisons import AgentComparison, _try_import_scipy

# ---------------------------------------------------------------------------
# Fake-torch helpers for attention_visualization
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Mimics torch.Tensor surface used by attention_visualization."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def unsqueeze(self, _dim):
        return self


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_torch(monkeypatch, *, capture_call=None):
    """Install a minimal `torch` module into sys.modules for the duration of a test."""

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _FakeTensor
    torch_mod.float32 = "float32"

    def as_tensor(data, dtype=None):
        return _FakeTensor(data)

    torch_mod.as_tensor = as_tensor
    torch_mod.no_grad = _FakeNoGrad
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return torch_mod


class _FakeModule:
    """Stand-in for an nn.Module-like layer that can register a forward hook."""

    def __init__(self, on_call_output):
        self._on_call_output = on_call_output
        self._hooks: list = []

    def register_forward_hook(self, hook):
        self._hooks.append(hook)
        return _FakeHookHandle(self, hook)

    def __call__(self, *args, **kwargs):
        for hook in list(self._hooks):
            hook(self, args, self._on_call_output)
        return self._on_call_output


class _FakeHookHandle:
    def __init__(self, module, hook):
        self._module = module
        self._hook = hook
        self.removed = False

    def remove(self):
        if self._hook in self._module._hooks:
            self._module._hooks.remove(self._hook)
        self.removed = True


class _FakeModel:
    """Wraps a named submodule and dispatches `__call__` through it."""

    def __init__(self, submodules):
        self._submodules = submodules

    def named_modules(self):
        return iter(self._submodules)

    def __call__(self, *args, **kwargs):
        # Invoke every registered attention-layer submodule so hooks fire.
        for _name, mod in self._submodules:
            mod(*args, **kwargs)


# ---------------------------------------------------------------------------
# attention_visualization with fake torch
# ---------------------------------------------------------------------------


class TestAttentionVisualizationWithTorch:
    def test_model_with_attention_returns_captured_tensor(self, monkeypatch):
        _install_fake_torch(monkeypatch)

        weights = _FakeTensor(np.array([[0.1, 0.9]]))
        attn = _FakeModule(on_call_output=weights)
        model = _FakeModel([("feature_extractor", _FakeModule(None)), ("attention_head", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(4))

        np.testing.assert_allclose(out, np.array([[0.1, 0.9]]))

    def test_policy_attr_fallback_when_no_model(self, monkeypatch):
        """When the agent lacks `model` but has `policy`, the policy is used."""
        _install_fake_torch(monkeypatch)

        weights = _FakeTensor(np.array([0.5, 0.5]))
        attn = _FakeModule(on_call_output=weights)
        policy_model = _FakeModel([("block.attention", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.policy = policy_model

        out = attention_visualization(agent, np.zeros(3))
        np.testing.assert_allclose(out, np.array([0.5, 0.5]))

    def test_tuple_output_extracts_weights(self, monkeypatch):
        """Hook output of shape (output, weights) should use index [1]."""
        _install_fake_torch(monkeypatch)

        primary = _FakeTensor(np.array([1.0, 2.0]))
        weights = _FakeTensor(np.array([[0.2, 0.3, 0.5]]))
        attn = _FakeModule(on_call_output=(primary, weights))
        model = _FakeModel([("mha_attention", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(2))
        np.testing.assert_allclose(out, np.array([[0.2, 0.3, 0.5]]))

    def test_single_tuple_output_uses_index_zero(self, monkeypatch):
        """When the hook returns a 1-tuple, weights = output[0]."""
        _install_fake_torch(monkeypatch)

        lone = _FakeTensor(np.array([0.7]))
        attn = _FakeModule(on_call_output=(lone,))
        model = _FakeModel([("attention", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(2))
        np.testing.assert_allclose(out, np.array([0.7]))

    def test_non_tensor_output_converted_via_asarray(self, monkeypatch):
        """When the hook output is not a torch.Tensor, it's wrapped with np.asarray."""
        _install_fake_torch(monkeypatch)

        ndarray_weights = np.array([1.0, 2.0, 3.0])
        attn = _FakeModule(on_call_output=ndarray_weights)
        model = _FakeModel([("self_attention", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(1))
        np.testing.assert_allclose(out, ndarray_weights)

    def test_both_model_and_policy_none_returns_dummy(self, monkeypatch):
        """If `model` and `policy` are both missing, returns a uniform dummy."""
        _install_fake_torch(monkeypatch)

        class Agent:
            model = None
            policy = None

        out = attention_visualization(Agent(), np.zeros(4))
        np.testing.assert_array_equal(out, np.array([1.0]))

    def test_named_layer_missing_returns_dummy(self, monkeypatch):
        """If no submodule name matches *layer_name*, returns a uniform dummy."""
        _install_fake_torch(monkeypatch)

        model = _FakeModel([("encoder", _FakeModule(None)), ("decoder", _FakeModule(None))])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(4), layer_name="self_attn")
        np.testing.assert_array_equal(out, np.array([1.0]))

    def test_custom_layer_name(self, monkeypatch):
        """A non-default *layer_name* should be respected."""
        _install_fake_torch(monkeypatch)

        weights = _FakeTensor(np.array([0.42]))
        target = _FakeModule(on_call_output=weights)
        model = _FakeModel([("block1.self_attn", _FakeModule(None)), ("block2.cross_xyz", target)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(2), layer_name="cross_xyz")
        np.testing.assert_allclose(out, np.array([0.42]))

    def test_first_matching_layer_wins(self, monkeypatch):
        """Only the first submodule whose name contains *layer_name* is hooked."""
        _install_fake_torch(monkeypatch)

        first = _FakeModule(on_call_output=_FakeTensor(np.array([1.0])))
        second = _FakeModule(on_call_output=_FakeTensor(np.array([99.0])))
        model = _FakeModel([("attention_a", first), ("attention_b", second)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(2))
        np.testing.assert_allclose(out, np.array([1.0]))

    def test_hook_handle_is_removed(self, monkeypatch):
        """After capture the registered forward hook should be removed."""
        _install_fake_torch(monkeypatch)

        attn = _FakeModule(on_call_output=_FakeTensor(np.array([0.0])))
        model = _FakeModel([("attention", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        attention_visualization(agent, np.zeros(2))
        # The submodule's hook list should be empty after the hook handle was removed.
        assert attn._hooks == []

    def test_hook_registered_but_never_fires_returns_dummy(self, monkeypatch):
        """Hook is registered but the forward pass doesn't invoke the submodule
        — `captured` stays empty and the final dummy-uniform return fires."""
        _install_fake_torch(monkeypatch)

        attn = _FakeModule(on_call_output=_FakeTensor(np.array([1.0, 2.0])))

        class _SilentModel(_FakeModel):
            def __call__(self, *args, **kwargs):
                # Deliberately skip firing any submodule so no hook captures a tensor.
                return None

        model = _SilentModel([("attention", attn)])

        class Agent:
            pass

        agent = Agent()
        agent.model = model

        out = attention_visualization(agent, np.zeros(3))
        np.testing.assert_array_equal(out, np.array([1.0]))


# ---------------------------------------------------------------------------
# comparisons._try_import_scipy
# ---------------------------------------------------------------------------


class TestTryImportScipy:
    def test_missing_scipy_raises_informative_error(self, monkeypatch):
        """With no scipy available, the helper raises ImportError with install hint."""
        # Pre-empt any real scipy import by mapping scipy -> None.
        monkeypatch.setitem(sys.modules, "scipy", None)

        with pytest.raises(ImportError, match="scipy is required"):
            _try_import_scipy()

    def test_successful_import_returns_stats_module(self, monkeypatch):
        """With scipy.stats present, the helper returns the stats module."""
        fake_scipy = types.ModuleType("scipy")
        fake_stats = types.ModuleType("scipy.stats")
        fake_stats.marker = "fake-stats"
        fake_scipy.stats = fake_stats
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.stats", fake_stats)

        stats = _try_import_scipy()
        assert stats is fake_stats
        assert getattr(stats, "marker", None) == "fake-stats"


# ---------------------------------------------------------------------------
# comparisons.AgentComparison.statistical_test with fake scipy
# ---------------------------------------------------------------------------


def _install_fake_scipy_stats(monkeypatch, *, wilcoxon_p=0.03, mann_p=0.07, call_log=None):
    """Install a minimal `scipy` + `scipy.stats` with wilcoxon / mannwhitneyu."""
    fake_scipy = types.ModuleType("scipy")
    fake_stats = types.ModuleType("scipy.stats")

    def wilcoxon(a, b):
        if call_log is not None:
            call_log.append(("wilcoxon", np.asarray(a).tolist(), np.asarray(b).tolist()))
        return (0.5, wilcoxon_p)

    def mannwhitneyu(a, b, alternative="two-sided"):
        if call_log is not None:
            call_log.append(
                ("mannwhitneyu", np.asarray(a).tolist(), np.asarray(b).tolist(), alternative)
            )
        return (1.5, mann_p)

    fake_stats.wilcoxon = wilcoxon
    fake_stats.mannwhitneyu = mannwhitneyu
    fake_scipy.stats = fake_stats
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.stats", fake_stats)
    return fake_stats


def _mk_results(name, values, scenario_count=None):
    n = scenario_count if scenario_count is not None else len(values)
    return BenchmarkResults(
        suite_name=name,
        scenario_names=[f"s{i}" for i in range(n)],
        metrics={"mean_reward": list(values)},
    )


class TestStatisticalTestWithFakeScipy:
    def test_wilcoxon_branch_invoked(self, monkeypatch):
        call_log: list = []
        _install_fake_scipy_stats(monkeypatch, wilcoxon_p=0.01, call_log=call_log)

        comp = AgentComparison()
        results = {
            "A": _mk_results("A", [1.0, 2.0, 3.0, 4.0]),
            "B": _mk_results("B", [1.5, 2.5, 3.5, 4.5]),
        }
        pvals = comp.statistical_test(results, test="wilcoxon")

        assert pvals[("A", "B")] == pytest.approx(0.01)
        # wilcoxon was invoked exactly once (two agents => one pair)
        assert [c[0] for c in call_log] == ["wilcoxon"]

    def test_mannwhitneyu_branch_invoked_with_alternative(self, monkeypatch):
        call_log: list = []
        _install_fake_scipy_stats(monkeypatch, mann_p=0.12, call_log=call_log)

        comp = AgentComparison()
        results = {
            "A": _mk_results("A", [1.0, 2.0, 3.0]),
            "B": _mk_results("B", [4.0, 5.0, 6.0]),
        }
        pvals = comp.statistical_test(results, test="mannwhitneyu")

        assert pvals[("A", "B")] == pytest.approx(0.12)
        assert call_log[0][0] == "mannwhitneyu"
        # alternative kwarg must be forwarded as two-sided
        assert call_log[0][-1] == "two-sided"

    def test_unknown_test_raises(self, monkeypatch):
        _install_fake_scipy_stats(monkeypatch)

        comp = AgentComparison()
        results = {
            "A": _mk_results("A", [1.0, 2.0]),
            "B": _mk_results("B", [3.0, 4.0]),
        }
        with pytest.raises(ValueError, match="Unknown test"):
            comp.statistical_test(results, test="ttest")

    def test_mixed_length_metrics_use_common_prefix(self, monkeypatch):
        """When agents have different sample counts, both vectors are trimmed to min_len."""
        call_log: list = []
        _install_fake_scipy_stats(monkeypatch, wilcoxon_p=0.5, call_log=call_log)

        comp = AgentComparison()
        # A has 5 values, B has 3. Both should be truncated to 3 when passed to wilcoxon.
        results = {
            "A": _mk_results("A", [1, 2, 3, 4, 5], scenario_count=5),
            "B": _mk_results("B", [10, 20, 30], scenario_count=3),
        }
        comp.statistical_test(results, test="wilcoxon")

        assert call_log, "wilcoxon should have been invoked"
        _, vec_a, vec_b, *_ = call_log[0]
        assert len(vec_a) == len(vec_b) == 3
        assert vec_a == [1, 2, 3]
        assert vec_b == [10, 20, 30]

    def test_three_agents_produce_all_pairs(self, monkeypatch):
        """With three agents, statistical_test should emit p-values for all (a, b) pairs."""
        _install_fake_scipy_stats(monkeypatch, wilcoxon_p=0.5)

        comp = AgentComparison()
        results = {
            "A": _mk_results("A", [1, 2, 3]),
            "B": _mk_results("B", [2, 3, 4]),
            "C": _mk_results("C", [3, 4, 5]),
        }
        pvals = comp.statistical_test(results, test="wilcoxon")

        # Pairs are generated from sorted names => A < B < C.
        assert set(pvals.keys()) == {("A", "B"), ("A", "C"), ("B", "C")}
        for p in pvals.values():
            assert p == pytest.approx(0.5)

    def test_agent_missing_metric_returns_nan(self, monkeypatch):
        """If an agent's metric list is missing/empty, the p-value is NaN (<2 samples)."""
        _install_fake_scipy_stats(monkeypatch)

        comp = AgentComparison()
        results = {
            "A": BenchmarkResults(suite_name="A", scenario_names=["s0"], metrics={}),
            "B": _mk_results("B", [1.0, 2.0, 3.0]),
        }
        pvals = comp.statistical_test(results, test="wilcoxon")

        assert np.isnan(pvals[("A", "B")])


# ---------------------------------------------------------------------------
# BenchmarkResults.plot_comparison
# ---------------------------------------------------------------------------


class TestBenchmarkResultsPlotComparison:
    def test_single_results_set_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = BenchmarkResults(
            suite_name="solo",
            scenario_names=["s1", "s2"],
            metrics={"success_rate": [0.8, 0.9], "reward": [10.0, 12.0]},
        )
        fig = results.plot_comparison()
        try:
            assert fig is not None
            axes = fig.get_axes()
            assert len(axes) == 1
            ax = axes[0]
            # Each metric should have exactly one bar.
            assert len(ax.patches) == 2
            # x-tick labels should match metric names sorted.
            labels = [t.get_text() for t in ax.get_xticklabels()]
            assert labels == sorted(results.metrics.keys())
            # Legend should identify the suite.
            legend = ax.get_legend()
            assert legend is not None
            assert "solo" in {t.get_text() for t in legend.get_texts()}
        finally:
            plt.close(fig)

    def test_with_other_plots_both_series(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = BenchmarkResults(
            suite_name="alpha",
            scenario_names=["s1"],
            metrics={"metric_a": [0.5], "metric_b": [1.0]},
        )
        b = BenchmarkResults(
            suite_name="beta",
            scenario_names=["s1"],
            metrics={"metric_a": [0.6], "metric_b": [1.5]},
        )
        fig = a.plot_comparison(other=b)
        try:
            ax = fig.get_axes()[0]
            # Two metrics × two series = 4 bars.
            assert len(ax.patches) == 4
            legend_texts = {t.get_text() for t in ax.get_legend().get_texts()}
            assert "alpha" in legend_texts
            assert "beta" in legend_texts
        finally:
            plt.close(fig)

    def test_fallback_legend_labels_when_suite_name_empty(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = BenchmarkResults(scenario_names=["s1"], metrics={"m": [1.0]})
        b = BenchmarkResults(scenario_names=["s1"], metrics={"m": [2.0]})
        fig = a.plot_comparison(other=b)
        try:
            legend_texts = {t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()}
            # Empty suite_name should fall back to "A" and "B" labels.
            assert "A" in legend_texts
            assert "B" in legend_texts
        finally:
            plt.close(fig)

    def test_other_metrics_missing_zero_default(self):
        """A metric present in self but missing from other defaults to 0.0 in the other bars."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = BenchmarkResults(
            suite_name="A",
            scenario_names=["s1"],
            metrics={"shared": [2.0], "only_in_a": [5.0]},
        )
        b = BenchmarkResults(
            suite_name="B",
            scenario_names=["s1"],
            metrics={"shared": [3.0]},
        )
        fig = a.plot_comparison(other=b)
        try:
            ax = fig.get_axes()[0]
            # 2 metrics × 2 series = 4 bars.
            assert len(ax.patches) == 4
            # The "other" bars should be the last two patches (drawn after self).
            other_heights = sorted(p.get_height() for p in ax.patches[2:])
            # One should be 0 (only_in_a missing in B), the other the shared value (3.0).
            assert other_heights == pytest.approx([0.0, 3.0])
        finally:
            plt.close(fig)
