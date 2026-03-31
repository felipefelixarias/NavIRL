"""Geometric utility functions for 2D navigation computations.

This module provides efficient geometric operations commonly needed
in pedestrian simulation and robot navigation, including angle
manipulation, distance computation, line intersection, and polygon
operations.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi].

    Parameters
    ----------
    angle : float
        Input angle in radians.

    Returns
    -------
    float
        Normalized angle in [-pi, pi].

    Examples
    --------
    >>> normalize_angle(3 * math.pi)
    -3.141592653589793  # approximately -pi
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def wrap_angle(angle: float) -> float:
    """Wrap angle to [0, 2*pi).

    Parameters
    ----------
    angle : float
        Input angle in radians.

    Returns
    -------
    float
        Angle in [0, 2*pi).
    """
    return angle % (2.0 * math.pi)


def angle_diff(a: float, b: float) -> float:
    """Compute the signed difference between two angles.

    Returns the shortest signed angular displacement from angle *b*
    to angle *a*, in the range [-pi, pi].

    Parameters
    ----------
    a : float
        Target angle in radians.
    b : float
        Source angle in radians.

    Returns
    -------
    float
        Signed angular difference (a - b) normalized to [-pi, pi].
    """
    return normalize_angle(a - b)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the unsigned angle between two 2-D vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector, shape (2,).
    v2 : np.ndarray
        Second vector, shape (2,).

    Returns
    -------
    float
        Angle in radians in [0, pi].
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_val))


def heading_from_velocity(vx: float, vy: float) -> float:
    """Compute heading angle from velocity components.

    Parameters
    ----------
    vx : float
        X component of velocity.
    vy : float
        Y component of velocity.

    Returns
    -------
    float
        Heading angle in radians, measured counter-clockwise from the
        positive x-axis.  Returns 0.0 if the velocity magnitude is
        negligible.
    """
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        return 0.0
    return math.atan2(vy, vx)


def angular_velocity(
    heading_prev: float,
    heading_curr: float,
    dt: float,
) -> float:
    """Compute angular velocity from heading change.

    Parameters
    ----------
    heading_prev : float
        Previous heading in radians.
    heading_curr : float
        Current heading in radians.
    dt : float
        Time step (must be > 0).

    Returns
    -------
    float
        Angular velocity in rad/s.
    """
    if dt <= 0:
        return 0.0
    return angle_diff(heading_curr, heading_prev) / dt


# ---------------------------------------------------------------------------
# Basic 2-D operations
# ---------------------------------------------------------------------------


def cross2d(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the 2-D cross product (z-component of 3-D cross product).

    Parameters
    ----------
    u : np.ndarray
        First 2-D vector.
    v : np.ndarray
        Second 2-D vector.

    Returns
    -------
    float
        Scalar cross product u x v.
    """
    return float(u[0] * v[1] - u[1] * v[0])


def dot2d(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the dot product of two 2-D vectors.

    Parameters
    ----------
    u : np.ndarray
        First 2-D vector.
    v : np.ndarray
        Second 2-D vector.

    Returns
    -------
    float
        Dot product.
    """
    return float(u[0] * v[0] + u[1] * v[1])


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points.

    Parameters
    ----------
    p1 : np.ndarray
        First point, shape (2,) or (N, 2).
    p2 : np.ndarray
        Second point, shape (2,) or (N, 2).

    Returns
    -------
    float or np.ndarray
        Euclidean distance(s).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    diff = p1 - p2
    if diff.ndim == 1:
        return float(np.linalg.norm(diff))
    return np.linalg.norm(diff, axis=-1)


def rotate_point(
    point: np.ndarray,
    angle: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate a 2-D point around a center.

    Parameters
    ----------
    point : np.ndarray
        Point to rotate, shape (2,).
    angle : float
        Rotation angle in radians (counter-clockwise positive).
    center : np.ndarray, optional
        Center of rotation, defaults to origin.

    Returns
    -------
    np.ndarray
        Rotated point, shape (2,).
    """
    point = np.asarray(point, dtype=np.float64)
    if center is None:
        center = np.zeros(2)
    else:
        center = np.asarray(center, dtype=np.float64)

    translated = point - center
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = np.array(
        [
            cos_a * translated[0] - sin_a * translated[1],
            sin_a * translated[0] + cos_a * translated[1],
        ]
    )
    return rotated + center


def rotate_points(
    points: np.ndarray,
    angle: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate multiple 2-D points around a center.

    Parameters
    ----------
    points : np.ndarray
        Points to rotate, shape (N, 2).
    angle : float
        Rotation angle in radians.
    center : np.ndarray, optional
        Center of rotation.

    Returns
    -------
    np.ndarray
        Rotated points, shape (N, 2).
    """
    points = np.asarray(points, dtype=np.float64)
    if center is None:
        center = np.zeros(2)
    else:
        center = np.asarray(center, dtype=np.float64)

    translated = points - center
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = translated @ rotation_matrix.T
    return rotated + center


# ---------------------------------------------------------------------------
# Line / segment operations
# ---------------------------------------------------------------------------


def closest_point_on_line(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    clamp_to_segment: bool = True,
) -> np.ndarray:
    """Find the closest point on a line segment to a given point.

    Parameters
    ----------
    point : np.ndarray
        Query point, shape (2,).
    line_start : np.ndarray
        Start of line segment, shape (2,).
    line_end : np.ndarray
        End of line segment, shape (2,).
    clamp_to_segment : bool
        If True, clamp the projection to the segment endpoints.

    Returns
    -------
    np.ndarray
        Closest point on the segment, shape (2,).
    """
    point = np.asarray(point, dtype=np.float64)
    line_start = np.asarray(line_start, dtype=np.float64)
    line_end = np.asarray(line_end, dtype=np.float64)

    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-24:
        return line_start.copy()

    t = np.dot(point - line_start, line_vec) / line_len_sq

    if clamp_to_segment:
        t = np.clip(t, 0.0, 1.0)

    return line_start + t * line_vec


def point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    segment: bool = True,
) -> float:
    """Compute the distance from a point to a line or line segment.

    Parameters
    ----------
    point : np.ndarray
        Query point, shape (2,).
    line_start : np.ndarray
        Start of line, shape (2,).
    line_end : np.ndarray
        End of line, shape (2,).
    segment : bool
        If True, treat as a segment (clamp projection).

    Returns
    -------
    float
        Distance from point to line/segment.
    """
    closest = closest_point_on_line(point, line_start, line_end, clamp_to_segment=segment)
    return float(np.linalg.norm(np.asarray(point) - closest))


def line_segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray | None:
    """Compute the intersection point of two line segments.

    Uses the parametric form of the line segments.  Returns None if
    the segments do not intersect.

    Parameters
    ----------
    p1, p2 : np.ndarray
        Endpoints of the first segment, shape (2,).
    p3, p4 : np.ndarray
        Endpoints of the second segment, shape (2,).

    Returns
    -------
    np.ndarray or None
        Intersection point, shape (2,), or None if no intersection.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)

    d1 = p2 - p1
    d2 = p4 - p3
    cross = cross2d(d1, d2)

    if abs(cross) < 1e-12:
        return None  # Parallel or coincident

    d3 = p3 - p1
    t = cross2d(d3, d2) / cross
    u = cross2d(d3, d1) / cross

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t * d1

    return None


def ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float | None:
    """Compute the parameter t where a ray intersects a line segment.

    Ray: origin + t * direction, t >= 0.

    Parameters
    ----------
    origin : np.ndarray
        Ray origin, shape (2,).
    direction : np.ndarray
        Ray direction (need not be unit), shape (2,).
    seg_start : np.ndarray
        Segment start, shape (2,).
    seg_end : np.ndarray
        Segment end, shape (2,).

    Returns
    -------
    float or None
        Parameter t along the ray, or None if no intersection.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    seg_start = np.asarray(seg_start, dtype=np.float64)
    seg_end = np.asarray(seg_end, dtype=np.float64)

    seg_dir = seg_end - seg_start
    cross = cross2d(direction, seg_dir)
    if abs(cross) < 1e-12:
        return None

    diff = seg_start - origin
    t = cross2d(diff, seg_dir) / cross
    u = cross2d(diff, direction) / cross

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def circle_circle_intersection(
    c1: np.ndarray,
    r1: float,
    c2: np.ndarray,
    r2: float,
) -> list[np.ndarray]:
    """Find intersection points of two circles.

    Parameters
    ----------
    c1 : np.ndarray
        Center of first circle, shape (2,).
    r1 : float
        Radius of first circle.
    c2 : np.ndarray
        Center of second circle, shape (2,).
    r2 : float
        Radius of second circle.

    Returns
    -------
    list of np.ndarray
        List of intersection points (0, 1, or 2 points).
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)

    d = float(np.linalg.norm(c2 - c1))
    if d < 1e-12:
        return []  # Concentric
    if d > r1 + r2 + 1e-12:
        return []  # Too far apart
    if d < abs(r1 - r2) - 1e-12:
        return []  # One inside the other

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < 0:
        h_sq = 0.0
    h = math.sqrt(h_sq)

    mid = c1 + a * (c2 - c1) / d
    offset = h * np.array([-(c2[1] - c1[1]) / d, (c2[0] - c1[0]) / d])

    if h < 1e-12:
        return [mid]
    return [mid + offset, mid - offset]


def circle_line_intersection(
    center: np.ndarray,
    radius: float,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> list[np.ndarray]:
    """Find intersection points of a circle and a line segment.

    Parameters
    ----------
    center : np.ndarray
        Circle center, shape (2,).
    radius : float
        Circle radius.
    line_start : np.ndarray
        Start of line segment, shape (2,).
    line_end : np.ndarray
        End of line segment, shape (2,).

    Returns
    -------
    list of np.ndarray
        Intersection points (0, 1, or 2).
    """
    center = np.asarray(center, dtype=np.float64)
    line_start = np.asarray(line_start, dtype=np.float64)
    line_end = np.asarray(line_end, dtype=np.float64)

    d = line_end - line_start
    f = line_start - center

    a = np.dot(d, d)
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0 or a < 1e-24:
        return []

    sqrt_disc = math.sqrt(discriminant)
    results = []
    for sign in (-1, 1):
        t = (-b + sign * sqrt_disc) / (2.0 * a)
        if 0.0 <= t <= 1.0:
            results.append(line_start + t * d)

    return results


# ---------------------------------------------------------------------------
# Polygon operations
# ---------------------------------------------------------------------------


def point_in_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
) -> bool:
    """Test if a point is inside a polygon using ray casting.

    Parameters
    ----------
    point : np.ndarray
        Query point, shape (2,).
    polygon : np.ndarray
        Polygon vertices, shape (N, 2), in order.

    Returns
    -------
    bool
        True if the point is inside the polygon.
    """
    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    n = len(polygon)
    inside = False
    px, py = point[0], point[1]

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def polygon_area(polygon: np.ndarray) -> float:
    """Compute the signed area of a simple polygon (shoelace formula).

    Parameters
    ----------
    polygon : np.ndarray
        Polygon vertices, shape (N, 2), in order.

    Returns
    -------
    float
        Signed area (positive for CCW winding).
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    n = len(polygon)
    if n < 3:
        return 0.0

    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """Compute the centroid of a simple polygon.

    Parameters
    ----------
    polygon : np.ndarray
        Polygon vertices, shape (N, 2).

    Returns
    -------
    np.ndarray
        Centroid point, shape (2,).
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    n = len(polygon)
    if n == 0:
        return np.zeros(2)
    if n <= 2:
        return np.mean(polygon, axis=0)

    area = polygon_area(polygon)
    if abs(area) < 1e-12:
        return np.mean(polygon, axis=0)

    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i, 0] * polygon[j, 1] - polygon[j, 0] * polygon[i, 1]
        cx += (polygon[i, 0] + polygon[j, 0]) * cross
        cy += (polygon[i, 1] + polygon[j, 1]) * cross

    factor = 1.0 / (6.0 * area)
    return np.array([cx * factor, cy * factor])


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2-D points (Andrew's monotone chain).

    Parameters
    ----------
    points : np.ndarray
        Points, shape (N, 2).

    Returns
    -------
    np.ndarray
        Convex hull vertices in CCW order, shape (M, 2).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points.copy()

    sorted_idx = np.lexsort((points[:, 1], points[:, 0]))
    sorted_pts = points[sorted_idx]

    # Build lower hull
    lower: list[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross2d(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper: list[np.ndarray] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross2d(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def minimum_bounding_rectangle(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Compute the minimum area bounding rectangle for a set of points.

    Uses the rotating calipers algorithm on the convex hull.

    Parameters
    ----------
    points : np.ndarray
        Points, shape (N, 2).

    Returns
    -------
    tuple
        (corners, width, height, angle) where corners is shape (4, 2),
        width and height are the rectangle dimensions, and angle is the
        rotation angle in radians.
    """
    hull = convex_hull(points)
    n = len(hull)
    if n <= 1:
        p = hull[0] if n == 1 else np.zeros(2)
        corners = np.array([p, p, p, p])
        return corners, 0.0, 0.0, 0.0

    best_area = float("inf")
    best_corners = None
    best_width = 0.0
    best_height = 0.0
    best_angle = 0.0

    edges = np.diff(np.vstack([hull, hull[0:1]]), axis=0)

    for i in range(n):
        edge = edges[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue

        angle = math.atan2(edge[1], edge[0])
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)

        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = hull @ rot.T

        min_x, min_y = rotated.min(axis=0)
        max_x, max_y = rotated.max(axis=0)

        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        if area < best_area:
            best_area = area
            best_width = width
            best_height = height
            best_angle = angle

            rect_corners = np.array(
                [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ]
            )
            inv_rot = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            best_corners = rect_corners @ inv_rot.T

    if best_corners is None:
        best_corners = np.zeros((4, 2))

    return best_corners, best_width, best_height, best_angle


# ---------------------------------------------------------------------------
# Transformation utilities
# ---------------------------------------------------------------------------


def transform_2d(
    points: np.ndarray,
    translation: np.ndarray | None = None,
    rotation: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply a 2-D rigid body transformation to points.

    The order is: scale -> rotate -> translate.

    Parameters
    ----------
    points : np.ndarray
        Points, shape (N, 2) or (2,).
    translation : np.ndarray, optional
        Translation vector, shape (2,).
    rotation : float
        Rotation angle in radians.
    scale : float
        Scale factor.

    Returns
    -------
    np.ndarray
        Transformed points.
    """
    points = np.asarray(points, dtype=np.float64)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 2)

    # Scale
    result = points * scale

    # Rotate
    if abs(rotation) > 1e-12:
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        result = result @ rot_mat.T

    # Translate
    if translation is not None:
        result = result + np.asarray(translation, dtype=np.float64)

    return result[0] if single else result


def build_transform_matrix(
    tx: float = 0.0,
    ty: float = 0.0,
    rotation: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Build a 3x3 homogeneous transformation matrix.

    Parameters
    ----------
    tx, ty : float
        Translation components.
    rotation : float
        Rotation angle in radians.
    scale : float
        Uniform scale factor.

    Returns
    -------
    np.ndarray
        3x3 transformation matrix.
    """
    cos_r = math.cos(rotation) * scale
    sin_r = math.sin(rotation) * scale
    return np.array(
        [
            [cos_r, -sin_r, tx],
            [sin_r, cos_r, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def apply_transform_matrix(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to 2-D points.

    Parameters
    ----------
    points : np.ndarray
        Points, shape (N, 2).
    transform : np.ndarray
        3x3 transformation matrix.

    Returns
    -------
    np.ndarray
        Transformed points, shape (N, 2).
    """
    points = np.asarray(points, dtype=np.float64)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 2)

    ones = np.ones((len(points), 1))
    homogeneous = np.hstack([points, ones])
    transformed = homogeneous @ transform.T
    result = transformed[:, :2]
    return result[0] if single else result


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


def compute_curvature(
    positions: np.ndarray,
) -> np.ndarray:
    """Compute discrete curvature along a trajectory.

    Uses the Menger curvature (circumscribed circle) for each triplet
    of consecutive points.

    Parameters
    ----------
    positions : np.ndarray
        Trajectory positions, shape (T, 2).

    Returns
    -------
    np.ndarray
        Curvature values, shape (T,).  First and last values are set
        to zero.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    curvature = np.zeros(n)

    for i in range(1, n - 1):
        a = positions[i - 1]
        b = positions[i]
        c = positions[i + 1]

        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)

        if ab < 1e-12 or bc < 1e-12 or ac < 1e-12:
            continue

        # Menger curvature = 4 * area / (|ab| * |bc| * |ac|)
        area = abs(cross2d(b - a, c - a)) / 2.0
        curvature[i] = 4.0 * area / (ab * bc * ac)

    return curvature


def compute_arc_length(positions: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length along a trajectory.

    Parameters
    ----------
    positions : np.ndarray
        Positions, shape (T, 2).

    Returns
    -------
    np.ndarray
        Cumulative arc length, shape (T,). First element is 0.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return np.zeros(len(positions))

    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def simplify_trajectory(
    positions: np.ndarray,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Simplify a trajectory using the Ramer-Douglas-Peucker algorithm.

    Parameters
    ----------
    positions : np.ndarray
        Trajectory positions, shape (T, 2).
    epsilon : float
        Maximum allowed perpendicular distance.

    Returns
    -------
    np.ndarray
        Simplified trajectory, shape (M, 2) with M <= T.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) <= 2:
        return positions.copy()

    def _rdp(pts: np.ndarray, eps: float) -> list[int]:
        if len(pts) <= 2:
            return list(range(len(pts)))

        # Find point with maximum distance from line
        start = pts[0]
        end = pts[-1]
        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(pts) - 1):
            d = point_to_line_distance(pts[i], start, end, segment=True)
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > eps:
            left = _rdp(pts[: max_idx + 1], eps)
            right = _rdp(pts[max_idx:], eps)
            return left[:-1] + [idx + max_idx for idx in right]
        else:
            return [0, len(pts) - 1]

    indices = _rdp(positions, epsilon)
    return positions[indices]
