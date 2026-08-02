from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from scan_profile import RainbowProfileConfig, make_rainbow_profile, transform_profile


_GEOMETRY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RainbowHalfProfile:
    """One local half of a full rainbow profile, retaining the full-profile anchor."""

    side: Literal["left", "right"]
    polygon: np.ndarray
    full_long_side_anchor_m: tuple[float, float]
    long_side_segment_m: tuple[tuple[float, float], tuple[float, float]]
    symmetry_axis_x_m: float


@dataclass(frozen=True)
class RainbowProfileHalves:
    """The original rainbow and its closed local left/right polygon partition."""

    full_polygon: np.ndarray
    left_half: RainbowHalfProfile
    right_half: RainbowHalfProfile
    long_side_start_m: tuple[float, float]
    long_side_end_m: tuple[float, float]
    long_side_anchor_m: tuple[float, float]
    symmetry_axis_segment_m: tuple[tuple[float, float], tuple[float, float]]


def make_rainbow_profile_halves(
    config: RainbowProfileConfig | None = None,
) -> RainbowProfileHalves:
    """Split the existing local rainbow polygon along its local symmetry axis."""
    full_polygon = make_rainbow_profile(config).copy()
    long_side_start, long_side_end = _long_side_endpoints(full_polygon)
    long_side_anchor = (long_side_start + long_side_end) / 2.0
    symmetry_axis_x = float(long_side_anchor[0])

    left_polygon = _clip_polygon_to_axis(full_polygon, symmetry_axis_x, keep_left=True)
    right_polygon = _clip_polygon_to_axis(full_polygon, symmetry_axis_x, keep_left=False)
    axis_points = np.vstack(
        (
            left_polygon[np.isclose(left_polygon[:, 0], symmetry_axis_x, atol=_GEOMETRY_TOLERANCE)],
            right_polygon[np.isclose(right_polygon[:, 0], symmetry_axis_x, atol=_GEOMETRY_TOLERANCE)],
        )
    )
    axis_start = (symmetry_axis_x, float(np.min(axis_points[:, 1])))
    axis_end = (symmetry_axis_x, float(np.max(axis_points[:, 1])))

    anchor_tuple = _point_tuple(long_side_anchor)
    long_start_tuple = _point_tuple(long_side_start)
    long_end_tuple = _point_tuple(long_side_end)
    left_half = RainbowHalfProfile(
        side="left",
        polygon=left_polygon,
        full_long_side_anchor_m=anchor_tuple,
        long_side_segment_m=(long_start_tuple, anchor_tuple),
        symmetry_axis_x_m=symmetry_axis_x,
    )
    right_half = RainbowHalfProfile(
        side="right",
        polygon=right_polygon,
        full_long_side_anchor_m=anchor_tuple,
        long_side_segment_m=(anchor_tuple, long_end_tuple),
        symmetry_axis_x_m=symmetry_axis_x,
    )
    return RainbowProfileHalves(
        full_polygon=full_polygon,
        left_half=left_half,
        right_half=right_half,
        long_side_start_m=long_start_tuple,
        long_side_end_m=long_end_tuple,
        long_side_anchor_m=anchor_tuple,
        symmetry_axis_segment_m=(axis_start, axis_end),
    )


def transform_rainbow_half(
    half_profile: RainbowHalfProfile,
    x_m: float,
    y_m: float,
    heading_rad: float,
) -> np.ndarray:
    """Apply the parent full profile's unchanged rotation and translation to a half."""
    return transform_profile(half_profile.polygon, x_m, y_m, heading_rad)


def transform_local_point(
    point_m: tuple[float, float],
    x_m: float,
    y_m: float,
    heading_rad: float,
) -> np.ndarray:
    """Apply the same rigid transform convention used by scan_profile.transform_profile."""
    point = np.asarray(point_m, dtype=float)
    cos_heading = math.cos(heading_rad)
    sin_heading = math.sin(heading_rad)
    rotation = np.array(
        [[cos_heading, -sin_heading], [sin_heading, cos_heading]],
        dtype=float,
    )
    return point @ rotation.T + np.array([float(x_m), float(y_m)], dtype=float)


def polygon_area(polygon: np.ndarray) -> float:
    """Return the shoelace area of a closed or open polygon vertex array."""
    points = _without_duplicate_closure(np.asarray(polygon, dtype=float))
    if len(points) < 3:
        return 0.0
    x_values = points[:, 0]
    y_values = points[:, 1]
    return abs(
        float(
            np.dot(x_values, np.roll(y_values, -1))
            - np.dot(y_values, np.roll(x_values, -1))
        )
    ) / 2.0


def reconstruction_area_error(halves: RainbowProfileHalves) -> float:
    """Return full area minus the two non-overlapping half areas in square meters."""
    return abs(
        polygon_area(halves.full_polygon)
        - polygon_area(halves.left_half.polygon)
        - polygon_area(halves.right_half.polygon)
        + half_intersection_area(halves)
    )


def half_intersection_area(halves: RainbowProfileHalves) -> float:
    """Return the exact shared area of halves separated by their common axis."""
    axis_x = halves.left_half.symmetry_axis_x_m
    left_max_x = float(np.max(halves.left_half.polygon[:, 0]))
    right_min_x = float(np.min(halves.right_half.polygon[:, 0]))
    if left_max_x > axis_x + _GEOMETRY_TOLERANCE:
        raise ValueError("Left half crosses to the right of the symmetry axis.")
    if right_min_x < axis_x - _GEOMETRY_TOLERANCE:
        raise ValueError("Right half crosses to the left of the symmetry axis.")
    # The polygons share only their zero-width symmetry-axis boundary.
    return 0.0


def _long_side_endpoints(full_polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_x = float(np.min(full_polygon[:, 0]))
    max_x = float(np.max(full_polygon[:, 0]))
    left_candidates = full_polygon[
        np.isclose(full_polygon[:, 0], min_x, atol=_GEOMETRY_TOLERANCE)
    ]
    right_candidates = full_polygon[
        np.isclose(full_polygon[:, 0], max_x, atol=_GEOMETRY_TOLERANCE)
    ]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError("Rainbow profile does not contain identifiable long-side endpoints.")
    left = left_candidates[np.argmax(left_candidates[:, 1])].copy()
    right = right_candidates[np.argmax(right_candidates[:, 1])].copy()
    return left, right


def _clip_polygon_to_axis(
    polygon: np.ndarray,
    axis_x: float,
    *,
    keep_left: bool,
) -> np.ndarray:
    source = _without_duplicate_closure(np.asarray(polygon, dtype=float))
    if len(source) < 3:
        raise ValueError("Rainbow polygon must contain at least three distinct vertices.")

    def inside(point: np.ndarray) -> bool:
        if keep_left:
            return float(point[0]) <= axis_x + _GEOMETRY_TOLERANCE
        return float(point[0]) >= axis_x - _GEOMETRY_TOLERANCE

    clipped: list[np.ndarray] = []
    previous = source[-1]
    previous_inside = inside(previous)
    for current in source:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                clipped.append(_axis_intersection(previous, current, axis_x))
            clipped.append(_normalize_axis_point(current, axis_x))
        elif previous_inside:
            clipped.append(_axis_intersection(previous, current, axis_x))
        previous = current
        previous_inside = current_inside

    deduplicated: list[np.ndarray] = []
    for point in clipped:
        if not deduplicated or not np.allclose(point, deduplicated[-1], atol=_GEOMETRY_TOLERANCE):
            deduplicated.append(point)
    if len(deduplicated) > 1 and np.allclose(
        deduplicated[0],
        deduplicated[-1],
        atol=_GEOMETRY_TOLERANCE,
    ):
        deduplicated.pop()
    if len(deduplicated) < 3:
        raise ValueError("Symmetry split did not produce a valid half polygon.")
    result = np.asarray(deduplicated, dtype=float)
    return np.vstack((result, result[0]))


def _axis_intersection(start: np.ndarray, end: np.ndarray, axis_x: float) -> np.ndarray:
    delta_x = float(end[0] - start[0])
    if abs(delta_x) <= _GEOMETRY_TOLERANCE:
        return np.array([axis_x, float((start[1] + end[1]) / 2.0)], dtype=float)
    fraction = (axis_x - float(start[0])) / delta_x
    return np.array(
        [axis_x, float(start[1] + fraction * (end[1] - start[1]))],
        dtype=float,
    )


def _normalize_axis_point(point: np.ndarray, axis_x: float) -> np.ndarray:
    result = np.asarray(point, dtype=float).copy()
    if abs(float(result[0]) - axis_x) <= _GEOMETRY_TOLERANCE:
        result[0] = axis_x
    return result


def _without_duplicate_closure(points: np.ndarray) -> np.ndarray:
    if len(points) > 1 and np.allclose(points[0], points[-1], atol=_GEOMETRY_TOLERANCE):
        return points[:-1]
    return points


def _point_tuple(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])
