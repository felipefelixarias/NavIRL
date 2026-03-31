"""Mathematical utility functions for NavIRL.

Provides common mathematical operations used throughout the framework
including interpolation, smoothing, statistical functions, and
numerical helpers.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Basic scalar operations
# ---------------------------------------------------------------------------


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to the range [min_val, max_val].

    Parameters
    ----------
    value : float
        Input value.
    min_val : float
        Minimum bound.
    max_val : float
        Maximum bound.

    Returns
    -------
    float
        Clamped value.
    """
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b.

    Parameters
    ----------
    a : float
        Start value.
    b : float
        End value.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    float
        Interpolated value.
    """
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, value: float) -> float:
    """Inverse linear interpolation: find t such that lerp(a, b, t) == value.

    Parameters
    ----------
    a : float
        Start value.
    b : float
        End value.
    value : float
        Target value.

    Returns
    -------
    float
        Parameter t (not clamped).
    """
    diff = b - a
    if abs(diff) < 1e-12:
        return 0.0
    return (value - a) / diff


def remap(
    value: float,
    from_min: float,
    from_max: float,
    to_min: float,
    to_max: float,
) -> float:
    """Remap a value from one range to another.

    Parameters
    ----------
    value : float
        Input value.
    from_min, from_max : float
        Source range.
    to_min, to_max : float
        Target range.

    Returns
    -------
    float
        Remapped value.
    """
    t = inverse_lerp(from_min, from_max, value)
    return lerp(to_min, to_max, t)


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation (smooth step) between two edges.

    Parameters
    ----------
    edge0 : float
        Lower edge.
    edge1 : float
        Upper edge.
    x : float
        Input value.

    Returns
    -------
    float
        Smoothly interpolated value in [0, 1].
    """
    t = clamp((x - edge0) / (edge1 - edge0) if abs(edge1 - edge0) > 1e-12 else 0.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoother_step(edge0: float, edge1: float, x: float) -> float:
    """Ken Perlin's improved smooth step (6t^5 - 15t^4 + 10t^3).

    Parameters
    ----------
    edge0 : float
        Lower edge.
    edge1 : float
        Upper edge.
    x : float
        Input value.

    Returns
    -------
    float
        Smoothly interpolated value in [0, 1].
    """
    t = clamp((x - edge0) / (edge1 - edge0) if abs(edge1 - edge0) > 1e-12 else 0.0, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ---------------------------------------------------------------------------
# Activation / probability functions
# ---------------------------------------------------------------------------


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid function.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).

    Returns
    -------
    float or np.ndarray
        Sigmoid output(s) in (0, 1).
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    if result.ndim == 0:
        return float(result)
    return result


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature scaling.

    Parameters
    ----------
    x : np.ndarray
        Input logits.
    axis : int
        Axis along which to compute softmax.
    temperature : float
        Temperature parameter (higher = more uniform).

    Returns
    -------
    np.ndarray
        Softmax probabilities.
    """
    x = np.asarray(x, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    scaled = x / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log-softmax (numerically stable).

    Parameters
    ----------
    x : np.ndarray
        Input logits.
    axis : int
        Axis along which to compute.

    Returns
    -------
    np.ndarray
        Log-softmax values.
    """
    x = np.asarray(x, dtype=np.float64)
    max_val = np.max(x, axis=axis, keepdims=True)
    shifted = x - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def gumbel_softmax(
    logits: np.ndarray,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from the Gumbel-Softmax distribution.

    Parameters
    ----------
    logits : np.ndarray
        Unnormalized log probabilities.
    temperature : float
        Temperature parameter.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Soft one-hot sample.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(size=logits.shape)
    u = np.clip(u, 1e-20, 1.0)
    gumbel_noise = -np.log(-np.log(u))
    return softmax((logits + gumbel_noise) / temperature)


# ---------------------------------------------------------------------------
# Statistical / running statistics
# ---------------------------------------------------------------------------


def running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute running mean with a given window size.

    Parameters
    ----------
    values : np.ndarray
        Input values, shape (N,).
    window : int
        Window size.

    Returns
    -------
    np.ndarray
        Running mean, shape (N,).  Values at the edges use a
        smaller (asymmetric) window.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values.copy()

    result = np.empty(n)
    cumsum = np.concatenate([[0.0], np.cumsum(values)])

    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result


def running_std(values: np.ndarray, window: int) -> np.ndarray:
    """Compute running standard deviation with a given window size.

    Parameters
    ----------
    values : np.ndarray
        Input values, shape (N,).
    window : int
        Window size.

    Returns
    -------
    np.ndarray
        Running standard deviation, shape (N,).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values.copy()

    result = np.empty(n)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.std(values[lo:hi])

    return result


def exponential_moving_average(
    values: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Compute exponential moving average.

    Parameters
    ----------
    values : np.ndarray
        Input values, shape (N,).
    alpha : float
        Smoothing factor in (0, 1).  Higher means less smoothing.

    Returns
    -------
    np.ndarray
        EMA values, shape (N,).
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values.copy()

    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def weighted_moving_average(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    window: int = 5,
) -> np.ndarray:
    """Compute weighted moving average.

    Parameters
    ----------
    values : np.ndarray
        Input values, shape (N,).
    weights : np.ndarray, optional
        Weights for the window.  If None, uses linearly increasing
        weights.
    window : int
        Window size (used only if weights is None).

    Returns
    -------
    np.ndarray
        Weighted moving average, shape (N,).
    """
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.arange(1, window + 1, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    w = len(weights)

    result = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - w + 1)
        chunk = values[lo : i + 1]
        used_weights = weights[-(i - lo + 1) :]
        result[i] = np.sum(chunk * used_weights) / np.sum(used_weights)

    return result


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a 1-D Gaussian kernel.

    Parameters
    ----------
    size : int
        Kernel size (should be odd).
    sigma : float
        Standard deviation.

    Returns
    -------
    np.ndarray
        Normalized Gaussian kernel, shape (size,).
    """
    x = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)


def gaussian_kernel_2d(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a 2-D Gaussian kernel.

    Parameters
    ----------
    size : int
        Kernel size (square, should be odd).
    sigma : float
        Standard deviation.

    Returns
    -------
    np.ndarray
        Normalized 2-D Gaussian kernel, shape (size, size).
    """
    k1d = gaussian_kernel(size, sigma)
    k2d = np.outer(k1d, k1d)
    return k2d / np.sum(k2d)


def epanechnikov_kernel(size: int) -> np.ndarray:
    """Create a 1-D Epanechnikov kernel.

    Parameters
    ----------
    size : int
        Kernel size.

    Returns
    -------
    np.ndarray
        Normalized Epanechnikov kernel, shape (size,).
    """
    x = np.linspace(-1, 1, size)
    kernel = np.maximum(0.0, 1.0 - x**2)
    return kernel / np.sum(kernel)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float | np.ndarray,
) -> np.ndarray:
    """Evaluate a cubic Bezier curve at parameter t.

    Parameters
    ----------
    p0, p1, p2, p3 : np.ndarray
        Control points, shape (D,).
    t : float or np.ndarray
        Parameter(s) in [0, 1].

    Returns
    -------
    np.ndarray
        Point(s) on the curve.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if t.ndim == 0:
        s = 1.0 - t
        return s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3

    t = t.reshape(-1, 1)
    s = 1.0 - t
    return s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3


def catmull_rom_spline(
    points: np.ndarray,
    num_points: int = 100,
    alpha: float = 0.5,
) -> np.ndarray:
    """Compute a Catmull-Rom spline through a set of points.

    Parameters
    ----------
    points : np.ndarray
        Control points, shape (N, D) with N >= 2.
    num_points : int
        Number of output points.
    alpha : float
        Parameterization: 0=uniform, 0.5=centripetal, 1=chordal.

    Returns
    -------
    np.ndarray
        Spline points, shape (num_points, D).
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return points.copy()

    # Pad with repeated endpoints
    padded = np.vstack([points[0:1], points, points[-1:]])

    segments = n - 1
    pts_per_segment = max(1, num_points // segments)
    result = []

    for i in range(segments):
        p0 = padded[i]
        p1 = padded[i + 1]
        p2 = padded[i + 2]
        p3 = padded[i + 3]

        def _knot_interval(pa: np.ndarray, pb: np.ndarray) -> float:
            d = np.linalg.norm(pb - pa)
            return d**alpha if d > 1e-12 else 1e-6

        t0 = 0.0
        t1 = t0 + _knot_interval(p0, p1)
        t2 = t1 + _knot_interval(p1, p2)
        t3 = t2 + _knot_interval(p2, p3)

        n_pts = pts_per_segment if i < segments - 1 else (num_points - len(result))
        ts = np.linspace(t1, t2, n_pts, endpoint=(i == segments - 1))

        for t in ts:
            a1 = (
                (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
                if abs(t1 - t0) > 1e-12
                else p0
            )
            a2 = (
                (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
                if abs(t2 - t1) > 1e-12
                else p1
            )
            a3 = (
                (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
                if abs(t3 - t2) > 1e-12
                else p2
            )

            b1 = (
                (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
                if abs(t2 - t0) > 1e-12
                else a1
            )
            b2 = (
                (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
                if abs(t3 - t1) > 1e-12
                else a2
            )

            c = (
                (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
                if abs(t2 - t1) > 1e-12
                else b1
            )
            result.append(c)

    return np.array(result)


# ---------------------------------------------------------------------------
# Numerical differentiation
# ---------------------------------------------------------------------------


def finite_difference(
    values: np.ndarray,
    dt: float = 1.0,
    order: int = 1,
    method: str = "central",
) -> np.ndarray:
    """Compute finite differences of a signal.

    Parameters
    ----------
    values : np.ndarray
        Input values, shape (N,) or (N, D).
    dt : float
        Time step.
    order : int
        Derivative order (1 or 2).
    method : str
        Difference method: "forward", "backward", or "central".

    Returns
    -------
    np.ndarray
        Derivative values, same shape as input.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if order == 1:
        result = np.zeros_like(values)
        if method == "forward":
            result[:-1] = (values[1:] - values[:-1]) / dt
            result[-1] = result[-2] if n > 1 else 0.0
        elif method == "backward":
            result[1:] = (values[1:] - values[:-1]) / dt
            result[0] = result[1] if n > 1 else 0.0
        else:  # central
            if n >= 3:
                result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
                result[0] = (values[1] - values[0]) / dt if n > 1 else 0.0
                result[-1] = (values[-1] - values[-2]) / dt if n > 1 else 0.0
            elif n == 2:
                result[:] = (values[1] - values[0]) / dt
        return result

    elif order == 2:
        result = np.zeros_like(values)
        if n >= 3:
            result[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (dt * dt)
            result[0] = result[1] if n > 2 else 0.0
            result[-1] = result[-2] if n > 2 else 0.0
        return result

    else:
        raise ValueError(f"Unsupported derivative order: {order}")


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------


def convolve_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1-D convolution with zero padding.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (N,).
    kernel : np.ndarray
        Convolution kernel, shape (K,).

    Returns
    -------
    np.ndarray
        Convolved signal, shape (N,).
    """
    signal = np.asarray(signal, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    k = len(kernel)
    pad = k // 2
    padded = np.pad(signal, pad, mode="edge")
    result = np.convolve(padded, kernel, mode="valid")
    return result[: len(signal)]


def low_pass_filter(
    signal: np.ndarray,
    cutoff: float,
    sample_rate: float,
    order: int = 2,
) -> np.ndarray:
    """Simple low-pass filter using repeated averaging.

    This is a basic implementation that applies a Gaussian kernel
    whose width corresponds to the cutoff frequency.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (N,).
    cutoff : float
        Cutoff frequency in Hz.
    sample_rate : float
        Sample rate in Hz.
    order : int
        Number of filter passes (higher = steeper rolloff).

    Returns
    -------
    np.ndarray
        Filtered signal, shape (N,).
    """
    if cutoff <= 0 or sample_rate <= 0:
        return signal.copy()

    # Kernel size from cutoff frequency
    sigma = sample_rate / (2.0 * math.pi * cutoff)
    kernel_size = max(3, int(6 * sigma) | 1)  # Ensure odd
    kernel = gaussian_kernel(kernel_size, sigma)

    result = signal.copy()
    for _ in range(order):
        result = convolve_1d(result, kernel)
    return result


def savitzky_golay(
    values: np.ndarray,
    window: int = 5,
    poly_order: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing filter.

    Parameters
    ----------
    values : np.ndarray
        Input values, shape (N,).
    window : int
        Window size (must be odd and > poly_order).
    poly_order : int
        Polynomial order for fitting.

    Returns
    -------
    np.ndarray
        Smoothed values, shape (N,).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < window:
        return values.copy()

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    half = window // 2
    result = np.empty_like(values)

    # Build the Vandermonde-like matrix for the window
    x = np.arange(-half, half + 1, dtype=np.float64)
    order_matrix = np.vander(x, N=poly_order + 1, increasing=True)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]

        if len(chunk) < poly_order + 1:
            result[i] = values[i]
            continue

        # Adjust x for edge cases
        actual_x = np.arange(len(chunk), dtype=np.float64) - (i - lo)
        A = np.vander(actual_x, N=poly_order + 1, increasing=True)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, chunk, rcond=None)
            result[i] = coeffs[0]  # Value at x=0
        except np.linalg.LinAlgError:
            result[i] = values[i]

    return result


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : np.ndarray
        Input vectors.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Compute Mahalanobis distance.

    Parameters
    ----------
    x : np.ndarray
        Point, shape (D,).
    mean : np.ndarray
        Mean, shape (D,).
    cov_inv : np.ndarray
        Inverse covariance matrix, shape (D, D).

    Returns
    -------
    float
        Mahalanobis distance.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    diff = x - mean
    return float(np.sqrt(diff @ cov_inv @ diff))


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
) -> float:
    """Compute Dynamic Time Warping distance between two sequences.

    Parameters
    ----------
    seq1 : np.ndarray
        First sequence, shape (N, D).
    seq2 : np.ndarray
        Second sequence, shape (M, D).

    Returns
    -------
    float
        DTW distance.
    """
    seq1 = np.asarray(seq1, dtype=np.float64)
    seq2 = np.asarray(seq2, dtype=np.float64)

    if seq1.ndim == 1:
        seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(-1, 1)

    n = len(seq1)
    m = len(seq2)

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(seq1[i - 1] - seq2[j - 1]))
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return dtw_matrix[n, m]


def frechet_distance(
    curve1: np.ndarray,
    curve2: np.ndarray,
) -> float:
    """Compute the discrete Frechet distance between two curves.

    Parameters
    ----------
    curve1 : np.ndarray
        First curve, shape (N, D).
    curve2 : np.ndarray
        Second curve, shape (M, D).

    Returns
    -------
    float
        Discrete Frechet distance.
    """
    curve1 = np.asarray(curve1, dtype=np.float64)
    curve2 = np.asarray(curve2, dtype=np.float64)

    n = len(curve1)
    m = len(curve2)

    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        if ca[i, j] > -0.5:
            return ca[i, j]

        d = float(np.linalg.norm(curve1[i] - curve2[j]))

        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), d)
        else:
            ca[i, j] = max(
                min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)),
                d,
            )

        return ca[i, j]

    return _c(n - 1, m - 1)


# ---------------------------------------------------------------------------
# Probability / entropy
# ---------------------------------------------------------------------------


def entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution (must sum to ~1).

    Returns
    -------
    float
        Entropy in nats.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D_KL(P || Q).

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions.

    Returns
    -------
    float
        KL divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    q_safe = np.clip(q[mask], 1e-12, None)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence.

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions.

    Returns
    -------
    float
        JS divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
