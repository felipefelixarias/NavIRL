"""Extended tests for navirl.utils.math_utils — covering uncovered functions.

Targets: log_softmax, gumbel_softmax, weighted_moving_average,
gaussian_kernel_2d, epanechnikov_kernel, cubic_bezier, catmull_rom_spline,
finite_difference (forward/backward/order-2), convolve_1d, low_pass_filter,
savitzky_golay, mahalanobis_distance, dtw_distance, frechet_distance,
js_divergence, and edge cases for cosine_similarity and running_* helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.utils.math_utils import (
    catmull_rom_spline,
    convolve_1d,
    cosine_similarity,
    cubic_bezier,
    dtw_distance,
    epanechnikov_kernel,
    exponential_moving_average,
    finite_difference,
    frechet_distance,
    gaussian_kernel_2d,
    gumbel_softmax,
    js_divergence,
    kl_divergence,
    log_softmax,
    low_pass_filter,
    mahalanobis_distance,
    running_mean,
    running_std,
    savitzky_golay,
    softmax,
    weighted_moving_average,
)

# ===================================================================
# log_softmax
# ===================================================================


class TestLogSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = log_softmax(x)
        # exp(log_softmax) should equal softmax
        np.testing.assert_allclose(np.exp(result), softmax(x), atol=1e-10)

    def test_sum_exp_is_one(self):
        x = np.array([0.0, 0.0, 0.0])
        result = log_softmax(x)
        assert np.sum(np.exp(result)) == pytest.approx(1.0, abs=1e-10)

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = log_softmax(x, axis=-1)
        for row in range(2):
            assert np.sum(np.exp(result[row])) == pytest.approx(1.0, abs=1e-10)


# ===================================================================
# gumbel_softmax
# ===================================================================


class TestGumbelSoftmax:
    def test_output_sums_to_one(self):
        rng = np.random.default_rng(42)
        logits = np.array([1.0, 2.0, 3.0])
        result = gumbel_softmax(logits, temperature=1.0, rng=rng)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_low_temperature_is_peaky(self):
        rng = np.random.default_rng(42)
        logits = np.array([0.0, 0.0, 10.0])
        result = gumbel_softmax(logits, temperature=0.01, rng=rng)
        assert result[2] > 0.9

    def test_default_rng(self):
        # Should work without explicit rng
        result = gumbel_softmax(np.array([1.0, 2.0]))
        assert result.sum() == pytest.approx(1.0, abs=1e-10)


# ===================================================================
# running_mean / running_std edge cases
# ===================================================================


class TestRunningStatsEdgeCases:
    def test_running_mean_empty(self):
        result = running_mean(np.array([]), window=3)
        assert len(result) == 0

    def test_running_std_empty(self):
        result = running_std(np.array([]), window=3)
        assert len(result) == 0

    def test_ema_empty(self):
        result = exponential_moving_average(np.array([]))
        assert len(result) == 0


# ===================================================================
# weighted_moving_average
# ===================================================================


class TestWeightedMovingAverage:
    def test_uniform_weights(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(3)
        result = weighted_moving_average(values, weights=weights)
        # At i=2, chunk=[1,2,3], weights=[1,1,1], mean=2.0
        assert result[2] == pytest.approx(2.0)

    def test_default_weights(self):
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = weighted_moving_average(values)
        np.testing.assert_allclose(result, np.ones(5), atol=1e-10)

    def test_length_preserved(self):
        values = np.arange(10, dtype=float)
        result = weighted_moving_average(values, window=3)
        assert len(result) == 10


# ===================================================================
# gaussian_kernel_2d
# ===================================================================


class TestGaussianKernel2D:
    def test_shape(self):
        k = gaussian_kernel_2d(5)
        assert k.shape == (5, 5)

    def test_normalized(self):
        k = gaussian_kernel_2d(7, sigma=2.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self):
        k = gaussian_kernel_2d(5, sigma=1.0)
        np.testing.assert_allclose(k, k.T, atol=1e-12)

    def test_center_is_max(self):
        k = gaussian_kernel_2d(5)
        assert k[2, 2] == k.max()


# ===================================================================
# epanechnikov_kernel
# ===================================================================


class TestEpanechnikovKernel:
    def test_normalized(self):
        k = epanechnikov_kernel(11)
        assert k.sum() == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self):
        k = epanechnikov_kernel(9)
        assert np.all(k >= 0)

    def test_symmetric(self):
        k = epanechnikov_kernel(7)
        np.testing.assert_allclose(k, k[::-1], atol=1e-12)


# ===================================================================
# cubic_bezier
# ===================================================================


class TestCubicBezier:
    def test_endpoints(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 2.0])
        p3 = np.array([4.0, 0.0])
        np.testing.assert_allclose(cubic_bezier(p0, p1, p2, p3, 0.0), p0, atol=1e-12)
        np.testing.assert_allclose(cubic_bezier(p0, p1, p2, p3, 1.0), p3, atol=1e-12)

    def test_midpoint(self):
        p0 = np.array([0.0, 0.0])
        p3 = np.array([4.0, 0.0])
        # Straight line control points
        p1 = np.array([1.333, 0.0])
        p2 = np.array([2.667, 0.0])
        mid = cubic_bezier(p0, p1, p2, p3, 0.5)
        assert mid[0] == pytest.approx(2.0, abs=0.01)
        assert mid[1] == pytest.approx(0.0, abs=0.01)

    def test_vectorized_t(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 1.0])
        p2 = np.array([2.0, 1.0])
        p3 = np.array([3.0, 0.0])
        t = np.array([0.0, 0.5, 1.0])
        result = cubic_bezier(p0, p1, p2, p3, t)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], p0, atol=1e-12)
        np.testing.assert_allclose(result[2], p3, atol=1e-12)


# ===================================================================
# catmull_rom_spline
# ===================================================================


class TestCatmullRomSpline:
    def test_passes_through_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        result = catmull_rom_spline(pts, num_points=100)
        assert result.shape == (100, 2)
        # Should start near first point and end near last
        np.testing.assert_allclose(result[0], pts[0], atol=0.1)
        np.testing.assert_allclose(result[-1], pts[-1], atol=0.1)

    def test_single_point(self):
        pts = np.array([[1.0, 2.0]])
        result = catmull_rom_spline(pts, num_points=50)
        assert len(result) == 1

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = catmull_rom_spline(pts, num_points=10)
        assert result.shape == (10, 2)


# ===================================================================
# finite_difference — forward, backward, order 2
# ===================================================================


class TestFiniteDifferenceExtended:
    def test_forward_difference(self):
        # f(x) = x^2 at x=0,1,2,3 → derivative = 2x
        values = np.array([0.0, 1.0, 4.0, 9.0])
        result = finite_difference(values, dt=1.0, order=1, method="forward")
        # Forward: [1, 3, 5, 5(repeated)]
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(5.0)

    def test_backward_difference(self):
        values = np.array([0.0, 1.0, 4.0, 9.0])
        result = finite_difference(values, dt=1.0, order=1, method="backward")
        # Backward: [1(repeated), 1, 3, 5]
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(3.0)
        assert result[3] == pytest.approx(5.0)

    def test_second_order(self):
        # f(x) = x^2 → f''(x) = 2
        values = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        result = finite_difference(values, dt=1.0, order=2)
        # Interior points: (4-2*1+0)/1=2, (9-2*4+1)/1=2, (16-2*9+4)/1=2
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(2.0)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            finite_difference(np.array([1.0, 2.0]), order=3)

    def test_two_element_central(self):
        values = np.array([0.0, 2.0])
        result = finite_difference(values, dt=1.0, order=1, method="central")
        np.testing.assert_allclose(result, [2.0, 2.0], atol=1e-10)


# ===================================================================
# convolve_1d
# ===================================================================


class TestConvolve1D:
    def test_identity_kernel(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = np.array([0.0, 1.0, 0.0])
        result = convolve_1d(signal, kernel)
        np.testing.assert_allclose(result, signal, atol=1e-10)

    def test_smoothing(self):
        signal = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        kernel = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        result = convolve_1d(signal, kernel)
        # Peak should be reduced
        assert result[2] < 1.0

    def test_length_preserved(self):
        signal = np.arange(20, dtype=float)
        kernel = np.ones(5) / 5
        result = convolve_1d(signal, kernel)
        assert len(result) == len(signal)


# ===================================================================
# low_pass_filter
# ===================================================================


class TestLowPassFilter:
    def test_preserves_dc(self):
        # Constant signal should be unchanged
        signal = np.ones(100) * 5.0
        result = low_pass_filter(signal, cutoff=10.0, sample_rate=100.0)
        np.testing.assert_allclose(result, signal, atol=0.1)

    def test_negative_cutoff_returns_copy(self):
        signal = np.array([1.0, 2.0, 3.0])
        result = low_pass_filter(signal, cutoff=-1.0, sample_rate=100.0)
        np.testing.assert_allclose(result, signal)

    def test_smooths_noise(self):
        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 2 * np.pi, 200)) + 0.5 * rng.standard_normal(200)
        result = low_pass_filter(signal, cutoff=5.0, sample_rate=200.0, order=3)
        # Filtered signal should have lower variance than noisy input
        assert np.std(result) < np.std(signal)


# ===================================================================
# savitzky_golay
# ===================================================================


class TestSavitzkyGolay:
    def test_preserves_linear(self):
        # Linear signal should be unchanged
        values = np.linspace(0, 10, 50)
        result = savitzky_golay(values, window=7, poly_order=2)
        np.testing.assert_allclose(result, values, atol=0.1)

    def test_short_signal(self):
        values = np.array([1.0, 2.0])
        result = savitzky_golay(values, window=5)
        np.testing.assert_allclose(result, values)

    def test_smoothing_effect(self):
        rng = np.random.default_rng(42)
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.3 * rng.standard_normal(100)
        result = savitzky_golay(values, window=11, poly_order=3)
        # Smoothed should have lower variance of differences
        assert np.std(np.diff(result)) < np.std(np.diff(values))


# ===================================================================
# cosine_similarity edge cases
# ===================================================================


class TestCosineSimilarityEdgeCases:
    def test_zero_vector(self):
        assert cosine_similarity(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0

    def test_both_zero(self):
        assert cosine_similarity(np.zeros(3), np.zeros(3)) == 0.0


# ===================================================================
# mahalanobis_distance
# ===================================================================


class TestMahalanobisDistance:
    def test_identity_covariance(self):
        # With identity cov_inv, should equal Euclidean distance
        x = np.array([3.0, 4.0])
        mean = np.array([0.0, 0.0])
        cov_inv = np.eye(2)
        assert mahalanobis_distance(x, mean, cov_inv) == pytest.approx(5.0)

    def test_zero_distance(self):
        x = np.array([1.0, 2.0])
        assert mahalanobis_distance(x, x, np.eye(2)) == pytest.approx(0.0, abs=1e-10)

    def test_scaled_covariance(self):
        x = np.array([2.0, 0.0])
        mean = np.array([0.0, 0.0])
        cov_inv = np.diag([4.0, 1.0])  # Inverse: higher weight on x-axis
        result = mahalanobis_distance(x, mean, cov_inv)
        assert result == pytest.approx(4.0)


# ===================================================================
# dtw_distance
# ===================================================================


class TestDTWDistance:
    def test_identical_sequences(self):
        seq = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        assert dtw_distance(seq, seq) == pytest.approx(0.0, abs=1e-10)

    def test_shifted_sequences(self):
        seq1 = np.array([[0.0], [1.0], [2.0]])
        seq2 = np.array([[1.0], [2.0], [3.0]])
        # Optimal alignment: 0→1(1), 1→1(0+via diag), 2→3(1) → DTW = 2.0
        assert dtw_distance(seq1, seq2) == pytest.approx(2.0)

    def test_1d_input(self):
        seq1 = np.array([0.0, 1.0, 2.0])
        seq2 = np.array([0.0, 1.0, 2.0])
        assert dtw_distance(seq1, seq2) == pytest.approx(0.0, abs=1e-10)

    def test_different_lengths(self):
        seq1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        seq2 = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        result = dtw_distance(seq1, seq2)
        assert result >= 0.0


# ===================================================================
# frechet_distance
# ===================================================================


class TestFrechetDistance:
    def test_identical_curves(self):
        curve = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        assert frechet_distance(curve, curve) == pytest.approx(0.0, abs=1e-10)

    def test_shifted_curves(self):
        c1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        c2 = np.array([[0.0, 1.0], [1.0, 1.0]])
        # Parallel curves 1 unit apart
        assert frechet_distance(c1, c2) == pytest.approx(1.0, abs=1e-10)

    def test_different_lengths(self):
        c1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        c2 = np.array([[0.0, 0.0], [2.0, 0.0]])
        result = frechet_distance(c1, c2)
        assert result >= 0.0


# ===================================================================
# js_divergence
# ===================================================================


class TestJSDivergence:
    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.2, 0.7])
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-10)

    def test_bounded(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        result = js_divergence(p, q)
        # JS divergence is bounded by ln(2) ≈ 0.693
        assert 0.0 <= result <= np.log(2) + 1e-10

    def test_greater_than_zero_for_different(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert js_divergence(p, q) > 0.0
