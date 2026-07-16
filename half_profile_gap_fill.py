from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from shapely.geometry import GeometryCollection, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from partial_scan_profile import make_rainbow_profile_halves, transform_rainbow_half
from scan_profile import RainbowProfileConfig, transform_profile


class FullProfileRejectionReason(str, Enum):
    """Structured reasons why a normal interior full-profile candidate was rejected."""

    SECTION_BOUNDARY_CONFLICT = "section_boundary_conflict"
    OUTSIDE_TANK = "outside_tank"
    EXCESSIVE_OVERLAP = "excessive_overlap"
    ORTHOGONAL_OVERLAP = "orthogonal_overlap"
    DUPLICATE_POSE = "duplicate_pose"
    INVALID_GEOMETRY = "invalid_geometry"
    INVALID_HEADING = "invalid_heading"
    MISSING_GUIDE = "missing_guide"


SALVAGEABLE_REJECTION_REASONS = frozenset(
    {
        FullProfileRejectionReason.SECTION_BOUNDARY_CONFLICT,
        FullProfileRejectionReason.OUTSIDE_TANK,
        FullProfileRejectionReason.EXCESSIVE_OVERLAP,
        FullProfileRejectionReason.ORTHOGONAL_OVERLAP,
    }
)
UNSALVAGEABLE_REJECTION_REASONS = frozenset(FullProfileRejectionReason) - SALVAGEABLE_REJECTION_REASONS


@dataclass(frozen=True)
class RejectedFullProfileCandidate:
    """One unchanged normal-lattice pose retained after full-profile rejection."""

    candidate_id: int
    guide_line_id: int
    guide_order_index: int
    placement_index: int
    projected_order_m: float
    orientation: str
    travel_direction: str
    anchor_m: tuple[float, float]
    profile_origin_m: tuple[float, float]
    heading_rad: float
    heading_deg: float
    full_polygon: np.ndarray
    rejection_reasons: tuple[FullProfileRejectionReason, ...]

    @property
    def is_salvageable(self) -> bool:
        reasons = set(self.rejection_reasons)
        return bool(reasons & SALVAGEABLE_REJECTION_REASONS) and not bool(
            reasons & UNSALVAGEABLE_REJECTION_REASONS
        )


@dataclass(frozen=True)
class ExactProfileMetrics:
    total_area_m2: float
    inside_area_m2: float
    outside_area_m2: float
    overlap_area_m2: float
    new_area_m2: float
    inside_fraction: float
    outside_fraction: float
    overlap_fraction: float


@dataclass(frozen=True)
class HalfProfileSalvageEvaluation:
    candidate: RejectedFullProfileCandidate
    full: ExactProfileMetrics
    left_half: ExactProfileMetrics
    right_half: ExactProfileMetrics
    selected_variant: str | None
    reason: str


def as_polygon(points: np.ndarray) -> BaseGeometry:
    geometry: BaseGeometry = Polygon(np.asarray(points, dtype=float))
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def polygon_union(polygons: list[np.ndarray]) -> BaseGeometry:
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
    """Build a tank-clipped row/column corridor allowing one lattice-step extension."""
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


def exact_profile_metrics(
    candidate_polygon: np.ndarray,
    valid_region: BaseGeometry,
    existing_coverage: BaseGeometry,
) -> ExactProfileMetrics:
    candidate = as_polygon(candidate_polygon)
    total_area = float(candidate.area)
    if total_area <= 0.0:
        return ExactProfileMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    inside_polygon = candidate.intersection(valid_region)
    outside_polygon = candidate.difference(valid_region)
    overlap_polygon = inside_polygon.intersection(existing_coverage)
    new_polygon = inside_polygon.difference(existing_coverage)
    inside_area = float(inside_polygon.area)
    outside_area = float(outside_polygon.area)
    overlap_area = float(overlap_polygon.area)
    new_area = float(new_polygon.area)
    return ExactProfileMetrics(
        total_area_m2=total_area,
        inside_area_m2=inside_area,
        outside_area_m2=outside_area,
        overlap_area_m2=overlap_area,
        new_area_m2=new_area,
        inside_fraction=inside_area / total_area,
        outside_fraction=outside_area / total_area,
        overlap_fraction=overlap_area / total_area,
    )


def add_geometric_rejection_reasons(
    candidate: RejectedFullProfileCandidate,
    valid_region: BaseGeometry,
    accepted_full_coverage: BaseGeometry,
    orthogonal_full_coverage: BaseGeometry,
    *,
    max_outside_fraction: float = 0.20,
    max_overlap_fraction: float = 0.35,
    orthogonal_overlap_fraction: float = 0.20,
) -> RejectedFullProfileCandidate:
    reasons = list(candidate.rejection_reasons)
    if not math.isfinite(candidate.heading_rad):
        reasons.append(FullProfileRejectionReason.INVALID_HEADING)
    full_geometry = as_polygon(candidate.full_polygon)
    if full_geometry.is_empty or not full_geometry.is_valid or full_geometry.area <= 0.0:
        reasons.append(FullProfileRejectionReason.INVALID_GEOMETRY)
    else:
        metrics = exact_profile_metrics(candidate.full_polygon, valid_region, accepted_full_coverage)
        if metrics.outside_fraction > max_outside_fraction:
            reasons.append(FullProfileRejectionReason.OUTSIDE_TANK)
        if metrics.overlap_fraction > max_overlap_fraction:
            reasons.append(FullProfileRejectionReason.EXCESSIVE_OVERLAP)
        orthogonal_area = float(full_geometry.intersection(orthogonal_full_coverage).area)
        if orthogonal_area / metrics.total_area_m2 > orthogonal_overlap_fraction:
            reasons.append(FullProfileRejectionReason.ORTHOGONAL_OVERLAP)
    return replace(candidate, rejection_reasons=tuple(dict.fromkeys(reasons)))


def evaluate_half_profile_salvage(
    candidate: RejectedFullProfileCandidate,
    valid_region: BaseGeometry,
    existing_coverage: BaseGeometry,
    *,
    profile_config: RainbowProfileConfig | None = None,
    min_inside_fraction: float = 0.80,
    max_outside_fraction: float = 0.20,
    max_overlap_fraction: float = 0.35,
    min_new_area_m2: float = 0.08,
    min_burden_improvement_fraction: float = 0.10,
    min_one_sided_advantage_fraction: float = 0.10,
) -> HalfProfileSalvageEvaluation:
    """Select one unchanged local half only for a strongly one-sided full rejection."""
    halves = make_rainbow_profile_halves(profile_config)
    x_m, y_m = candidate.profile_origin_m
    polygons = {
        "full": transform_profile(halves.full_polygon, x_m, y_m, candidate.heading_rad),
        "left_half": transform_rainbow_half(
            halves.left_half, x_m, y_m, candidate.heading_rad
        ),
        "right_half": transform_rainbow_half(
            halves.right_half, x_m, y_m, candidate.heading_rad
        ),
    }
    metrics = {
        variant: exact_profile_metrics(polygon, valid_region, existing_coverage)
        for variant, polygon in polygons.items()
    }
    result_args = (candidate, metrics["full"], metrics["left_half"], metrics["right_half"])
    if not candidate.is_salvageable:
        return HalfProfileSalvageEvaluation(*result_args, None, "rejection reason is not salvageable")
    viable: list[tuple[str, ExactProfileMetrics]] = []
    full_burden = metrics["full"].outside_fraction + metrics["full"].overlap_fraction
    for variant in ("left_half", "right_half"):
        half_metrics = metrics[variant]
        if not _metrics_acceptable(
            half_metrics,
            min_inside_fraction,
            max_outside_fraction,
            max_overlap_fraction,
            min_new_area_m2,
        ):
            continue
        half_burden = half_metrics.outside_fraction + half_metrics.overlap_fraction
        if full_burden - half_burden < min_burden_improvement_fraction:
            continue
        viable.append((variant, half_metrics))
    if not viable:
        return HalfProfileSalvageEvaluation(*result_args, None, "neither half passes exact thresholds")

    selected_variant, selected_metrics = max(
        viable,
        key=lambda item: (
            item[1].new_area_m2,
            -(item[1].outside_fraction + item[1].overlap_fraction),
            item[0] == "right_half",
        ),
    )
    other_variant = "right_half" if selected_variant == "left_half" else "left_half"
    other_metrics = metrics[other_variant]
    selected_burden = selected_metrics.outside_fraction + selected_metrics.overlap_fraction
    other_burden = other_metrics.outside_fraction + other_metrics.overlap_fraction
    new_advantage_fraction = (
        selected_metrics.new_area_m2 - other_metrics.new_area_m2
    ) / selected_metrics.total_area_m2
    burden_advantage = other_burden - selected_burden
    if max(new_advantage_fraction, burden_advantage) < min_one_sided_advantage_fraction:
        return HalfProfileSalvageEvaluation(*result_args, None, "failure is not strongly one-sided")
    return HalfProfileSalvageEvaluation(
        *result_args,
        selected_variant,
        f"{selected_variant} salvages a strongly one-sided full-profile rejection",
    )


def _metrics_acceptable(
    metrics: ExactProfileMetrics,
    min_inside_fraction: float,
    max_outside_fraction: float,
    max_overlap_fraction: float,
    min_new_area_m2: float,
) -> bool:
    return (
        metrics.inside_fraction >= min_inside_fraction
        and metrics.outside_fraction <= max_outside_fraction
        and metrics.overlap_fraction <= max_overlap_fraction
        and metrics.new_area_m2 >= min_new_area_m2
    )
