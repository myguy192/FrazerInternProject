"""Shared exact-geometry helpers for the current lawnmower coverage gate."""

from __future__ import annotations

import numpy as np
from shapely.geometry import GeometryCollection, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def as_polygon(points: np.ndarray) -> BaseGeometry:
    """Return a repaired Shapely polygon for an ``Nx2`` point array."""
    geometry: BaseGeometry = Polygon(np.asarray(points, dtype=float))
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def polygon_union(polygons: list[np.ndarray]) -> BaseGeometry:
    """Return the union of nonempty polygons, or an empty geometry."""
    geometries = [as_polygon(polygon) for polygon in polygons]
    geometries = [geometry for geometry in geometries if not geometry.is_empty]
    return unary_union(geometries) if geometries else GeometryCollection()


def build_candidate_valid_region(
    start_m: tuple[float, float],
    end_m: tuple[float, float],
    tank_center_m: tuple[float, float],
    tank_radius_m: float,
    local_full_polygon: np.ndarray,
    local_anchor_m: np.ndarray,
    placement_step_m: float,
) -> BaseGeometry:
    """Build a tank-clipped guide corridor with one lattice-step extension."""
    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-12 or placement_step_m <= 0.0:
        return GeometryCollection()

    direction = delta / length
    normal = np.array([direction[1], -direction[0]], dtype=float)
    relative_progress = local_full_polygon[:, 1] - local_anchor_m[1]
    start_extension = placement_step_m - float(np.min(relative_progress))
    end_extension = placement_step_m + float(np.max(relative_progress))
    half_width = float(np.max(np.abs(local_full_polygon[:, 0] - local_anchor_m[0])))
    corridor_start = start - direction * start_extension
    corridor_end = end + direction * end_extension
    corridor = Polygon(
        [
            corridor_start + normal * half_width,
            corridor_end + normal * half_width,
            corridor_end - normal * half_width,
            corridor_start - normal * half_width,
        ]
    )
    tank = Point(*tank_center_m).buffer(float(tank_radius_m), quad_segs=256)
    return corridor.intersection(tank)
