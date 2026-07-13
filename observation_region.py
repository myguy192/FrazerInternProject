from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scan_profile import RainbowProfileConfig, make_rainbow_profile, transform_profile


CLIP_TOLERANCE_M = 1e-8

try:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
except ImportError:
    LineString = None
    Polygon = None
    unary_union = None


@dataclass
class ScanFootprintRegion:
    polygons: list[np.ndarray]
    polygon_bounds: list[tuple[float, float, float, float]]
    circular_pose_count: int
    profile_config: RainbowProfileConfig
    source: str = "circular_scan_footprints"
    _union_geometry: Any = field(default=None, repr=False, compare=False)


def build_observed_region_from_scan_poses(
    scan_poses: Iterable[Any],
    scan_profile: RainbowProfileConfig | np.ndarray | None = None,
) -> ScanFootprintRegion:
    """Build the union region covered by circular-stage rainbow scan poses."""
    if isinstance(scan_profile, np.ndarray):
        local_profile = np.asarray(scan_profile, dtype=float)
        profile_config = RainbowProfileConfig()
    else:
        profile_config = RainbowProfileConfig() if scan_profile is None else scan_profile
        local_profile = make_rainbow_profile(profile_config)

    polygons: list[np.ndarray] = []
    for pose in scan_poses:
        stage = str(_pose_value(pose, "stage", ""))
        if not stage.startswith("circular"):
            continue
        if str(_pose_value(pose, "status", "kept")) not in {"kept", "accepted", ""}:
            continue
        x_m = float(_pose_value(pose, "x_m"))
        y_m = float(_pose_value(pose, "y_m"))
        heading_rad = float(_pose_value(pose, "heading_rad"))
        if not all(math.isfinite(value) for value in (x_m, y_m, heading_rad)):
            raise ValueError("Circular scan poses must contain finite x, y, and heading values.")
        polygons.append(transform_profile(local_profile, x_m, y_m, heading_rad))

    if not polygons:
        raise ValueError("No accepted circular scan poses were provided.")
    polygon_bounds = [_polygon_bounds(polygon) for polygon in polygons]
    union_geometry = _build_shapely_union(polygons)
    return ScanFootprintRegion(
        polygons=polygons,
        polygon_bounds=polygon_bounds,
        circular_pose_count=len(polygons),
        profile_config=profile_config,
        _union_geometry=union_geometry,
    )


def build_observed_region_from_mission_json(
    filepath: str | Path,
) -> ScanFootprintRegion:
    """Load circular poses and profile dimensions from a saved planner mission."""
    path = Path(filepath)
    if not path.exists():
        raise ValueError(f"Mission JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read mission JSON: {path}") from exc
    poses = data.get("poses")
    if not isinstance(poses, list):
        raise ValueError("Mission JSON must contain a poses list.")
    profile_data = data.get("scan_profile", {})
    profile_config = RainbowProfileConfig(
        width=float(profile_data.get("width_m", 1.6)),
        arc_radius=float(profile_data.get("arc_radius_m", 0.9)),
        side_height=float(profile_data.get("side_height_m", 0.5)),
        arc_samples=int(profile_data.get("arc_samples", 64)),
    )
    region = build_observed_region_from_scan_poses(poses, profile_config)
    region.source = str(path)
    return region


def clip_segment_to_observed_region(
    start: tuple[float, float] | np.ndarray,
    end: tuple[float, float] | np.ndarray,
    observed_region: ScanFootprintRegion,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return only portions of a segment inside the scan-footprint union."""
    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    if start_point.shape != (2,) or end_point.shape != (2,):
        raise ValueError("Segment endpoints must be two-dimensional points.")
    if not np.all(np.isfinite(start_point)) or not np.all(np.isfinite(end_point)):
        raise ValueError("Segment endpoints must be finite.")
    direction = end_point - start_point
    length_squared = float(np.dot(direction, direction))
    if length_squared <= CLIP_TOLERANCE_M**2:
        return []

    if observed_region._union_geometry is not None:
        return _clip_with_shapely(start_point, end_point, observed_region._union_geometry)

    segment_bounds = (
        min(start_point[0], end_point[0]),
        min(start_point[1], end_point[1]),
        max(start_point[0], end_point[0]),
        max(start_point[1], end_point[1]),
    )
    covered_intervals: list[tuple[float, float]] = []
    for polygon, bounds in zip(observed_region.polygons, observed_region.polygon_bounds):
        if not _bounds_intersect(segment_bounds, bounds):
            continue
        covered_intervals.extend(_segment_intervals_inside_polygon(start_point, end_point, polygon))
    if not covered_intervals:
        return []

    parameter_tolerance = CLIP_TOLERANCE_M / max(math.sqrt(length_squared), CLIP_TOLERANCE_M)
    merged = _merge_intervals(covered_intervals, parameter_tolerance)
    return [
        (start_point + direction * first, start_point + direction * second)
        for first, second in merged
        if second - first > parameter_tolerance
    ]


def points_in_observed_region(points: np.ndarray, observed_region: ScanFootprintRegion) -> np.ndarray:
    """Return a mask identifying points inside any completed scan footprint."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be an Nx2 array.")
    inside = np.zeros(len(points), dtype=bool)
    for polygon, bounds in zip(observed_region.polygons, observed_region.polygon_bounds):
        candidates = (
            ~inside
            & (points[:, 0] >= bounds[0] - CLIP_TOLERANCE_M)
            & (points[:, 0] <= bounds[2] + CLIP_TOLERANCE_M)
            & (points[:, 1] >= bounds[1] - CLIP_TOLERANCE_M)
            & (points[:, 1] <= bounds[3] + CLIP_TOLERANCE_M)
        )
        if np.any(candidates):
            indices = np.flatnonzero(candidates)
            inside[indices] = _points_in_or_on_polygon(points[indices], polygon)
    return inside


def _pose_value(pose: Any, name: str, default: Any = None) -> Any:
    if isinstance(pose, dict):
        if name in pose:
            return pose[name]
    elif hasattr(pose, name):
        return getattr(pose, name)
    if default is not None:
        return default
    raise ValueError(f"Scan pose is missing required field: {name}")


def _build_shapely_union(polygons: list[np.ndarray]) -> Any:
    if Polygon is None or unary_union is None:
        return None
    valid_polygons = []
    for points in polygons:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            valid_polygons.append(polygon)
    if not valid_polygons:
        raise ValueError("Circular scan footprints did not produce valid polygons.")
    return unary_union(valid_polygons)


def _clip_with_shapely(start: np.ndarray, end: np.ndarray, union_geometry: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    intersection = LineString([start, end]).intersection(union_geometry)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    def collect(geometry: Any) -> None:
        if geometry.is_empty:
            return
        if geometry.geom_type == "LineString":
            coordinates = np.asarray(geometry.coords, dtype=float)
            if len(coordinates) >= 2:
                segments.append((coordinates[0], coordinates[-1]))
            return
        if hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                collect(child)

    collect(intersection)
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    segments.sort(key=lambda pair: float(np.dot(pair[0] - start, direction) / length_squared))
    return segments


def _segment_intervals_inside_polygon(start: np.ndarray, end: np.ndarray, polygon: np.ndarray) -> list[tuple[float, float]]:
    direction = end - start
    parameters = [0.0, 1.0]
    polygon = _without_duplicate_closure(polygon)
    for edge_start, edge_end in zip(polygon, np.roll(polygon, -1, axis=0)):
        parameters.extend(_segment_edge_parameters(start, direction, edge_start, edge_end - edge_start))
    parameters = _unique_sorted_parameters(parameters)
    intervals: list[tuple[float, float]] = []
    for first, second in zip(parameters, parameters[1:]):
        if second - first <= 1e-12:
            continue
        midpoint = start + direction * ((first + second) / 2.0)
        if _point_in_or_on_polygon(midpoint, polygon):
            intervals.append((first, second))
    return intervals


def _segment_edge_parameters(
    segment_start: np.ndarray,
    segment_direction: np.ndarray,
    edge_start: np.ndarray,
    edge_direction: np.ndarray,
) -> list[float]:
    cross_directions = _cross_2d(segment_direction, edge_direction)
    offset = edge_start - segment_start
    if abs(cross_directions) <= CLIP_TOLERANCE_M:
        if abs(_cross_2d(offset, segment_direction)) > CLIP_TOLERANCE_M:
            return []
        length_squared = float(np.dot(segment_direction, segment_direction))
        values = [
            float(np.dot(edge_start - segment_start, segment_direction) / length_squared),
            float(np.dot(edge_start + edge_direction - segment_start, segment_direction) / length_squared),
        ]
        return [min(1.0, max(0.0, value)) for value in values if -CLIP_TOLERANCE_M <= value <= 1.0 + CLIP_TOLERANCE_M]
    segment_parameter = _cross_2d(offset, edge_direction) / cross_directions
    edge_parameter = _cross_2d(offset, segment_direction) / cross_directions
    if (
        -CLIP_TOLERANCE_M <= segment_parameter <= 1.0 + CLIP_TOLERANCE_M
        and -CLIP_TOLERANCE_M <= edge_parameter <= 1.0 + CLIP_TOLERANCE_M
    ):
        return [min(1.0, max(0.0, float(segment_parameter)))]
    return []


def _points_in_or_on_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    polygon = _without_duplicate_closure(polygon)
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        crosses = (current[1] > points[:, 1]) != (previous[1] > points[:, 1])
        denominator = previous[1] - current[1]
        if abs(float(denominator)) > 1e-15:
            crossing_x = (
                (previous[0] - current[0]) * (points[:, 1] - current[1]) / denominator
                + current[0]
            )
            inside ^= crosses & (points[:, 0] < crossing_x)
        previous = current
    if np.all(inside):
        return inside
    for index in np.flatnonzero(~inside):
        inside[index] = _point_on_polygon_boundary(points[index], polygon)
    return inside


def _point_in_or_on_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    return bool(_points_in_or_on_polygon(np.asarray(point, dtype=float).reshape(1, 2), polygon)[0])


def _point_on_polygon_boundary(point: np.ndarray, polygon: np.ndarray) -> bool:
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= CLIP_TOLERANCE_M**2:
            continue
        projection = float(np.dot(point - start, segment) / length_squared)
        projection = min(1.0, max(0.0, projection))
        closest = start + projection * segment
        if float(np.linalg.norm(point - closest)) <= CLIP_TOLERANCE_M:
            return True
    return False


def _merge_intervals(intervals: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    ordered = sorted((min(first, second), max(first, second)) for first, second in intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + tolerance:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _unique_sorted_parameters(values: list[float]) -> list[float]:
    ordered = sorted(min(1.0, max(0.0, value)) for value in values)
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > 1e-10:
            unique.append(value)
    return unique


def _polygon_bounds(polygon: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(polygon[:, 0])),
        float(np.min(polygon[:, 1])),
        float(np.max(polygon[:, 0])),
        float(np.max(polygon[:, 1])),
    )


def _bounds_intersect(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return not (
        first[2] < second[0] - CLIP_TOLERANCE_M
        or first[0] > second[2] + CLIP_TOLERANCE_M
        or first[3] < second[1] - CLIP_TOLERANCE_M
        or first[1] > second[3] + CLIP_TOLERANCE_M
    )


def _without_duplicate_closure(polygon: np.ndarray) -> np.ndarray:
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
        return polygon[:-1]
    return polygon


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])
