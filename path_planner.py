from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from dxf_importer import DxfImportError, GeometryModel, import_dxf
from half_profile_gap_fill import (
    FullProfileRejectionReason,
    HalfProfileSalvageEvaluation,
    RejectedFullProfileCandidate,
    add_geometric_rejection_reasons,
    as_polygon,
    build_candidate_valid_region,
    evaluate_half_profile_salvage,
    polygon_union,
)
from partial_scan_profile import make_rainbow_profile_halves, transform_rainbow_half
from scan_profile import RainbowProfileConfig, make_rainbow_profile, transform_profile


DEFAULT_EDGE_OFFSET_FROM_WALL = 0.0
DEFAULT_ROW_SPACING = 1.60
DEFAULT_SPACING_FACTOR = 1.0
# Retained for strategy-search API compatibility; profile density is now chosen
# directly from measured neighboring-polygon overlap.
DEFAULT_CIRCULAR_SPACING_CANDIDATES = (1.0,)
DEFAULT_MAX_CIRCULAR_ROWS = 2
DEFAULT_MAX_CIRCULAR_GAP_FRACTION = 0.05
DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS = 3
DEFAULT_CIRCULAR_NEIGHBOR_OVERLAP_FRACTION = 0.04
MIN_CIRCULAR_OVERLAP_FRACTION = 0.01
MAX_CIRCULAR_OVERLAP_FRACTION = 0.04
DEFAULT_VERTICAL_SPACING_FACTOR = 0.95
DEFAULT_OVERLAP_DISCARD_THRESHOLD = 0.50
DEFAULT_VERTICAL_COLUMN_EDGE_OVERLAP_LIMIT = 1.0 / 3.0
DEFAULT_LAWNMOWER_NEIGHBOR_OVERLAP_FRACTION = 0.02
GEOMETRY_EPSILON = 1e-8
MAX_EXPECTED_NEIGHBOR_OVERLAP_FRACTION = 0.04
MIN_FULL_INSIDE_FRACTION = 0.80
MAX_FULL_OUTSIDE_FRACTION = 0.20
MIN_FULL_NEW_AREA_M2 = 0.08
MIN_FULL_NEW_FRACTION = 0.30
MAX_FULL_HARMFUL_OVERLAP_FRACTION = 0.55
MIN_HALF_INSIDE_FRACTION = 0.80
MAX_HALF_OUTSIDE_FRACTION = 0.20
MIN_HALF_NEW_AREA_M2 = 0.05
MIN_HALF_NEW_FRACTION = 0.25
MAX_HALF_HARMFUL_OVERLAP_FRACTION = 0.50
MIN_BURDEN_DOMINANCE = 0.75
OUTSIDE_BURDEN_WEIGHT = 2.0
MIN_HALF_SCORE_ADVANTAGE_M2 = 0.05
MIN_GUIDE_INCREMENTAL_AREA_M2 = 0.08
MIN_GUIDE_NEW_FRACTION = 0.10
SIDE_GUARD_HEADING_RAD = math.pi / 2.0
DEFAULT_JSON_PATH = "mission_plan.json"
DEFAULT_PLOT_PATH = "mission_preview.png"
VERTICAL_SLOPE_TOLERANCE = 0.02
VERTICAL_X_GROUP_TOLERANCE_M = 0.02
VERTICAL_LINE_COVERAGE_RATIO = 0.60
HORIZONTAL_Y_GROUP_TOLERANCE_M = VERTICAL_X_GROUP_TOLERANCE_M
HORIZONTAL_FRAGMENT_MERGE_GAP_M = 0.05
HORIZONTAL_PLATE_SPACING_TOLERANCE_FRACTION = 0.15
HORIZONTAL_PLATE_SPACING_MIN_TOLERANCE_M = 0.08
PERPENDICULAR_SECTION_CLEARANCE_M = 0.02
SECTION_BOUNDARY_MATCH_TOLERANCE_M = 0.03
SECTION_MIN_LENGTH_M = 1e-6
INTERIOR_POSE_POSITION_TOLERANCE_M = 0.01
INTERIOR_POSE_ANGLE_TOLERANCE_RAD = math.radians(1.0)
OVERLAP_SAMPLE_GRID_SIZE = 28
CIRCULAR_GAP_TOLERANCE_M = 0.002
CIRCULAR_TEST_BAND_HALF_WIDTH_M = 0.005
CIRCULAR_TEST_ANGULAR_SAMPLES = 17
CIRCULAR_PROFILE_COUNT_SEARCH_RADIUS = 32
CIRCULAR_OVERLAP_TIE_FRACTION = 0.0025
CIRCULAR_OVERLAP_SAMPLE_GRID_SIZE = 48
CIRCULAR_MAX_PROFILE_COUNT = 10000
_TOUCHING_ANGULAR_STEP_CACHE: dict[tuple[float, float, float, float, int], float] = {}


@dataclass(frozen=True)
class TankCircleEstimate:
    center_x: float
    center_y: float
    radius: float
    method: str


@dataclass(frozen=True)
class SweepPose:
    scan_id: int
    stage: str
    x_m: float
    y_m: float
    heading_rad: float
    heading_deg: float
    profile_type: str = "rainbow"
    row_id: int | None = None
    row_name: str | None = None
    column_id: int | None = None
    theta_rad: float | None = None
    sweep_radius_m: float | None = None
    status: str = "kept"
    orientation: str | None = None
    group_id: int | None = None
    section_id: int | None = None
    travel_direction: str | None = None
    profile_variant: str = "full"
    anchor_x_m: float | None = None
    anchor_y_m: float | None = None
    parent_full_x_m: float | None = None
    parent_full_y_m: float | None = None


@dataclass(frozen=True)
class VerticalPlateColumn:
    column_id: int
    center_x_m: float
    left_weld_x_m: float
    right_weld_x_m: float
    min_y_m: float
    max_y_m: float


@dataclass(frozen=True)
class InteriorSection:
    section_id: int
    group_id: int
    orientation: str
    center_cross_m: float
    lower_weld_coordinate_m: float
    upper_weld_coordinate_m: float
    progress_min_m: float
    progress_max_m: float
    source_segment_count: int = 0


@dataclass(frozen=True)
class LawnmowerSectionLine:
    """One deterministically ordered straight interior section."""

    line_id: int
    orientation: str
    order_index: int
    source_section_id: int
    start_m: tuple[float, float]
    end_m: tuple[float, float]
    is_outer_extension: bool = False


@dataclass(frozen=True)
class LawnmowerScanPlacement:
    """One rainbow profile anchored by its long-side midpoint on a guide line."""

    placement_id: int
    guide_line_id: int
    guide_order_index: int
    placement_index: int
    orientation: str
    travel_direction: str
    anchor_m: tuple[float, float]
    profile_origin_m: tuple[float, float]
    heading_rad: float
    heading_deg: float
    profile_variant: str = "full"
    parent_profile_origin_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class LawnmowerCandidateMetrics:
    total_area_m2: float
    inside_area_m2: float
    outside_area_m2: float
    inside_fraction: float
    outside_fraction: float
    total_overlap_area_m2: float
    expected_neighbor_overlap_area_m2: float
    harmful_overlap_area_m2: float
    harmful_overlap_fraction: float
    new_area_m2: float
    new_fraction: float


@dataclass(frozen=True)
class LawnmowerCandidateRecord:
    candidate_id: int
    guide_line_id: int
    guide_order_index: int
    placement_index: int
    orientation: str
    travel_direction: str
    projected_position_m: float
    anchor_m: tuple[float, float]
    profile_origin_m: tuple[float, float]
    heading_rad: float
    heading_deg: float
    full_polygon: np.ndarray
    left_half_polygon: np.ndarray
    right_half_polygon: np.ndarray
    candidate_source: str
    candidate_stage: str
    acceptance_result: str = "pending"
    rejection_reasons: tuple[str, ...] = ()
    full_metrics: LawnmowerCandidateMetrics | None = None
    left_half_metrics: LawnmowerCandidateMetrics | None = None
    right_half_metrics: LawnmowerCandidateMetrics | None = None


@dataclass(frozen=True)
class AcceptedLawnmowerCoverage:
    guide_line_id: int
    orientation: str
    projected_position_m: float
    profile_variant: str
    polygon: np.ndarray
    inside_area_m2: float


@dataclass(frozen=True)
class LawnmowerGatingPass:
    placements: list[LawnmowerScanPlacement]
    candidates: list[LawnmowerCandidateRecord]
    retained_guide_ids: list[int]
    discarded_guide_ids: list[int]


@dataclass(frozen=True)
class LawnmowerPlacementPass:
    """Normal full-profile results before the separate half-salvage pass."""

    accepted: list[LawnmowerScanPlacement]
    rejected: list[RejectedFullProfileCandidate]


@dataclass(frozen=True)
class LawnmowerHalfSalvagePass:
    placements: list[LawnmowerScanPlacement]
    evaluations: list[HalfProfileSalvageEvaluation]


@dataclass(frozen=True)
class CircularSweepRow:
    row_id: int
    sweep_radius_m: float
    scan_count: int
    angular_spacing_rad: float
    arc_spacing_m: float
    estimated_gap_area_m2: float
    estimated_gap_fraction: float
    max_neighbor_gap_distance_m: float
    minimum_neighbor_contact_passed: bool
    wraparound_passed: bool
    continuous_coverage_verified: bool
    spacing_adjustments: int
    target_neighbor_overlap_fraction: float
    minimum_neighbor_overlap_fraction: float
    average_neighbor_overlap_fraction: float
    maximum_neighbor_overlap_fraction: float
    wraparound_overlap_fraction: float
    no_neighbor_gaps_verified: bool


@dataclass(frozen=True)
class CircularSweepPlan:
    rows: list[CircularSweepRow]
    poses: list[SweepPose]
    rejected_radius_m: float | None
    stop_reason: str


@dataclass(frozen=True)
class CoverageEstimate:
    total_profiles: int
    covered_percent: float
    multi_covered_percent: float


@dataclass(frozen=True)
class CircularSpacingCandidate:
    circular_rows: int
    spacing_factor: float | None
    coverage_percent: float
    scan_count: int
    travel_distance_m: float
    selected: bool = False


@dataclass(frozen=True)
class OuterEdgeSweepMission:
    units: str
    tank_center_m: dict[str, float]
    tank_radius_m: float
    outer_edge_offset_m: float
    row_spacing_m: float
    profile_radial_half_width_m: float
    profile_tangential_span_m: float
    profile_wall_contact_span_m: float
    spacing_factor: float
    selected_circular_spacing_factor: float | None
    circular_spacing_candidates: list[CircularSpacingCandidate]
    circular_spacing_selection_reason: str | None
    max_circular_gap_fraction: float
    circular_stop_backoff_rows: int
    circular_rows: list[CircularSweepRow]
    rejected_circular_radius_m: float | None
    circular_stop_reason: str
    direction: str
    scan_profile: dict[str, float | int | str]
    interior_enabled: bool
    vertical_spacing_factor: float
    overlap_discard_threshold: float
    vertical_columns: list[VerticalPlateColumn]
    edge_sweep_scan_count: int
    interior_kept_count: int
    interior_discarded_count: int
    total_scan_count: int
    poses: list[SweepPose]
    source_dxf: str | None = None
    tank_estimate_method: str = "bounds"
    horizontal_sections: list[InteriorSection] = field(default_factory=list)
    vertical_group_count: int = 0
    horizontal_group_count: int = 0
    vertical_scan_count: int = 0
    horizontal_scan_count: int = 0
    duplicate_poses_removed: int = 0
    invalid_horizontal_segments_skipped: int = 0
    lawnmower_section_lines: list[LawnmowerSectionLine] = field(default_factory=list)
    rejected_full_profile_candidate_count: int = 0
    salvaged_half_profile_count: int = 0


def estimate_tank_circle_from_geometry(model: GeometryModel) -> TankCircleEstimate:
    """Estimate a circular tank boundary from imported geometry bounds."""
    if model.bounds is None:
        raise ValueError("Geometry bounds are empty; cannot estimate tank circle.")
    width = float(model.bounds.width)
    height = float(model.bounds.height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Geometry bounds have zero area; cannot estimate tank circle.")

    center_x = (model.bounds.min_x + model.bounds.max_x) / 2.0
    center_y = (model.bounds.min_y + model.bounds.max_y) / 2.0
    radius = max(width, height) / 2.0
    return TankCircleEstimate(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        method="bounding_box_circle",
    )


def predict_tank_layout_from_geometry(
    model: GeometryModel,
    config: Any | None = None,
    *,
    circular_scan_poses: Any | None = None,
    scan_profile: Any | None = None,
) -> Any:
    """Explicit opt-in bridge from imported geometry to the standalone layout predictor."""
    from tank_layout_predictor import (
        observed_geometry_from_circular_sweeps,
        observed_geometry_from_dxf,
        predict_tank_layout,
    )

    observations = observed_geometry_from_dxf(model)
    if circular_scan_poses is not None:
        observations, _ = observed_geometry_from_circular_sweeps(
            observations,
            circular_scan_poses,
            scan_profile,
        )
    return predict_tank_layout(observations, config=config)


def generate_outer_edge_sweep_poses(
    tank_center: tuple[float, float],
    tank_radius: float,
    *,
    outer_edge_offset_m: float = DEFAULT_EDGE_OFFSET_FROM_WALL,
    row_spacing_m: float = DEFAULT_ROW_SPACING,
    spacing_factor: float = DEFAULT_SPACING_FACTOR,
    max_circular_gap_fraction: float = DEFAULT_MAX_CIRCULAR_GAP_FRACTION,
    circular_stop_backoff_rows: int = DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS,
    profile_config: RainbowProfileConfig | None = None,
    clockwise: bool = False,
) -> list[SweepPose]:
    """Compatibility wrapper returning accepted circular edge poses only."""
    return _generate_circular_sweep_plan(
        tank_center,
        tank_radius,
        outer_edge_offset_m=outer_edge_offset_m,
        row_spacing_m=row_spacing_m,
        spacing_factor=spacing_factor,
        max_circular_gap_fraction=max_circular_gap_fraction,
        circular_stop_backoff_rows=circular_stop_backoff_rows,
        profile_config=profile_config,
        clockwise=clockwise,
    ).poses


def _generate_sweep_row_poses(
    tank_center: tuple[float, float],
    sweep_radius: float,
    *,
    profile_config: RainbowProfileConfig,
    spacing_factor: float,
    row_id: int,
    row_name: str,
    scan_id_start: int,
    start_theta: float,
    clockwise: bool,
    target_overlap_fraction: float = DEFAULT_CIRCULAR_NEIGHBOR_OVERLAP_FRACTION,
) -> tuple[list[SweepPose], CircularSweepRow] | None:
    if not math.isfinite(sweep_radius) or sweep_radius <= CIRCULAR_GAP_TOLERANCE_M:
        return None
    if not MIN_CIRCULAR_OVERLAP_FRACTION <= target_overlap_fraction <= MAX_CIRCULAR_OVERLAP_FRACTION:
        raise ValueError(
            "Circular neighbor overlap target must be between "
            f"{MIN_CIRCULAR_OVERLAP_FRACTION:.3f} and {MAX_CIRCULAR_OVERLAP_FRACTION:.3f}."
        )

    local_profile = make_rainbow_profile(profile_config)
    if not _is_usable_profile_polygon(local_profile):
        return None

    touching_angle = _touching_angular_step(sweep_radius, profile_config)
    if not math.isfinite(touching_angle) or touching_angle <= 0.0:
        return None
    initial_pose_count = max(3, int(math.ceil((2.0 * math.pi) / touching_angle)))
    if initial_pose_count > CIRCULAR_MAX_PROFILE_COUNT:
        return None

    # spacing_factor is retained in the public API for compatibility, but it
    # no longer controls circular density. Actual polygon overlap does.
    _ = spacing_factor
    direction_sign = -1.0 if clockwise else 1.0
    center_x, center_y = tank_center
    candidate_counts = _ranked_circular_profile_counts(
        sweep_radius,
        local_profile,
        initial_pose_count,
        clockwise=clockwise,
        target_overlap_fraction=target_overlap_fraction,
    )
    if not candidate_counts:
        return None

    selected: tuple[list[SweepPose], int, dict[str, float | int | bool]] | None = None
    for pose_count in candidate_counts:
        angular_spacing = 2.0 * math.pi / pose_count
        poses: list[SweepPose] = []
        for local_index in range(pose_count):
            theta = start_theta + direction_sign * angular_spacing * local_index
            normalized_theta = _normalize_angle(theta)
            x_m = center_x + sweep_radius * math.cos(theta)
            y_m = center_y + sweep_radius * math.sin(theta)
            poses.append(
                SweepPose(
                    scan_id=scan_id_start + local_index,
                    stage="circular_edge",
                    x_m=float(x_m),
                    y_m=float(y_m),
                    heading_rad=normalized_theta,
                    heading_deg=math.degrees(normalized_theta),
                    row_id=row_id,
                    row_name=row_name,
                    theta_rad=normalized_theta,
                    sweep_radius_m=sweep_radius,
                )
            )

        world_profiles = [
            transform_profile(local_profile, pose.x_m, pose.y_m, pose.heading_rad)
            for pose in poses
        ]
        verification = _verify_circular_row_coverage(
            tank_center,
            sweep_radius,
            poses,
            world_profiles,
            local_profile=local_profile,
            clockwise=clockwise,
            minimum_overlap_fraction=MIN_CIRCULAR_OVERLAP_FRACTION,
        )
        if bool(verification["continuous_coverage_verified"]):
            selected = (poses, pose_count, verification)
            break
    if selected is None:
        return None

    poses, pose_count, verification = selected
    angular_spacing = 2.0 * math.pi / pose_count
    return poses, CircularSweepRow(
        row_id=row_id,
        sweep_radius_m=sweep_radius,
        scan_count=pose_count,
        angular_spacing_rad=angular_spacing,
        arc_spacing_m=sweep_radius * angular_spacing,
        estimated_gap_area_m2=0.0,
        estimated_gap_fraction=0.0,
        max_neighbor_gap_distance_m=float(verification["max_neighbor_gap_distance_m"]),
        minimum_neighbor_contact_passed=bool(verification["minimum_neighbor_contact_passed"]),
        wraparound_passed=bool(verification["wraparound_passed"]),
        continuous_coverage_verified=bool(verification["continuous_coverage_verified"]),
        spacing_adjustments=abs(pose_count - initial_pose_count),
        target_neighbor_overlap_fraction=target_overlap_fraction,
        minimum_neighbor_overlap_fraction=float(verification["minimum_neighbor_overlap_fraction"]),
        average_neighbor_overlap_fraction=float(verification["average_neighbor_overlap_fraction"]),
        maximum_neighbor_overlap_fraction=float(verification["maximum_neighbor_overlap_fraction"]),
        wraparound_overlap_fraction=float(verification["wraparound_overlap_fraction"]),
        no_neighbor_gaps_verified=bool(verification["no_neighbor_gaps_verified"]),
    )


def _ranked_circular_profile_counts(
    sweep_radius: float,
    local_profile: np.ndarray,
    initial_pose_count: int,
    *,
    clockwise: bool,
    target_overlap_fraction: float,
) -> list[int]:
    """Rank a small set of integer counts by measured neighboring overlap."""
    lower = max(3, initial_pose_count - CIRCULAR_PROFILE_COUNT_SEARCH_RADIUS)
    upper = min(CIRCULAR_MAX_PROFILE_COUNT, initial_pose_count + CIRCULAR_PROFILE_COUNT_SEARCH_RADIUS)
    direction_sign = -1.0 if clockwise else 1.0
    first_profile = transform_profile(local_profile, sweep_radius, 0.0, 0.0)
    local_samples = _profile_interior_sample_points(local_profile)
    first_samples = transform_profile(local_samples, sweep_radius, 0.0, 0.0)
    candidates: list[tuple[int, float]] = []

    for pose_count in range(lower, upper + 1):
        angular_spacing = direction_sign * 2.0 * math.pi / pose_count
        second_profile = transform_profile(
            local_profile,
            sweep_radius * math.cos(angular_spacing),
            sweep_radius * math.sin(angular_spacing),
            _normalize_angle(angular_spacing),
        )
        second_samples = transform_profile(
            local_samples,
            sweep_radius * math.cos(angular_spacing),
            sweep_radius * math.sin(angular_spacing),
            _normalize_angle(angular_spacing),
        )
        if not _is_usable_profile_polygon(second_profile):
            continue
        overlap_fraction = _estimate_neighbor_overlap_fraction(
            first_profile,
            second_profile,
            first_sample_points=first_samples,
            second_sample_points=second_samples,
        )
        if overlap_fraction >= MIN_CIRCULAR_OVERLAP_FRACTION:
            candidates.append((pose_count, overlap_fraction))

    if not candidates:
        return []
    best_delta = min(abs(overlap - target_overlap_fraction) for _, overlap in candidates)
    effectively_tied = [
        item
        for item in candidates
        if abs(item[1] - target_overlap_fraction) <= best_delta + CIRCULAR_OVERLAP_TIE_FRACTION
    ]
    effectively_tied.sort(key=lambda item: item[0])
    remaining = [item for item in candidates if item not in effectively_tied]
    remaining.sort(key=lambda item: (abs(item[1] - target_overlap_fraction), item[0]))
    return [pose_count for pose_count, _overlap in effectively_tied + remaining]


def _verify_circular_row_coverage(
    tank_center: tuple[float, float],
    sweep_radius: float,
    poses: list[SweepPose],
    world_profiles: list[np.ndarray],
    *,
    local_profile: np.ndarray,
    clockwise: bool,
    minimum_overlap_fraction: float = MIN_CIRCULAR_OVERLAP_FRACTION,
) -> dict[str, float | int | bool]:
    """Verify measured polygon overlap and annular coverage for every neighbor pair."""
    if len(poses) < 3 or len(poses) != len(world_profiles):
        return _failed_circular_row_verification()
    if any(not _is_usable_profile_polygon(profile) for profile in world_profiles):
        return _failed_circular_row_verification()

    direction_sign = -1.0 if clockwise else 1.0
    angular_spacing = 2.0 * math.pi / len(poses)
    max_neighbor_gap = 0.0
    wraparound_passed = False
    overlap_fractions: list[float] = []
    pair_gap_results: list[bool] = []
    local_samples = _profile_interior_sample_points(local_profile)
    world_sample_sets = [
        transform_profile(local_samples, pose.x_m, pose.y_m, pose.heading_rad)
        for pose in poses
    ]

    for index, first_profile in enumerate(world_profiles):
        next_index = (index + 1) % len(world_profiles)
        second_profile = world_profiles[next_index]
        overlap_fraction = _estimate_neighbor_overlap_fraction(
            first_profile,
            second_profile,
            first_sample_points=world_sample_sets[index],
            second_sample_points=world_sample_sets[next_index],
        )
        overlap_fractions.append(overlap_fraction)
        neighbor_gap = 0.0 if overlap_fraction > 0.0 else _polygon_distance(first_profile, second_profile)
        max_neighbor_gap = max(max_neighbor_gap, neighbor_gap)
        overlap_passed = overlap_fraction >= minimum_overlap_fraction

        theta_start = float(poses[index].theta_rad or 0.0)
        band_passed = _circular_pair_covers_test_band(
            tank_center,
            sweep_radius,
            theta_start,
            direction_sign * angular_spacing,
            first_profile,
            second_profile,
        )
        no_gap = overlap_fraction > 0.0 and band_passed
        pair_gap_results.append(no_gap)
        pair_passed = overlap_passed and no_gap
        if next_index == 0:
            wraparound_passed = pair_passed

    minimum_overlap = min(overlap_fractions)
    average_overlap = float(np.mean(overlap_fractions))
    maximum_overlap = max(overlap_fractions)
    no_neighbor_gaps = bool(all(pair_gap_results))
    minimum_overlap_passed = minimum_overlap >= minimum_overlap_fraction

    return {
        "max_neighbor_gap_distance_m": max_neighbor_gap,
        "minimum_neighbor_contact_passed": minimum_overlap_passed,
        "wraparound_passed": wraparound_passed,
        "continuous_coverage_verified": no_neighbor_gaps and minimum_overlap_passed and wraparound_passed,
        "minimum_neighbor_overlap_fraction": minimum_overlap,
        "average_neighbor_overlap_fraction": average_overlap,
        "maximum_neighbor_overlap_fraction": maximum_overlap,
        "wraparound_overlap_fraction": overlap_fractions[-1],
        "no_neighbor_gaps_verified": no_neighbor_gaps,
    }


def _failed_circular_row_verification() -> dict[str, float | int | bool]:
    return {
        "max_neighbor_gap_distance_m": math.inf,
        "minimum_neighbor_contact_passed": False,
        "wraparound_passed": False,
        "continuous_coverage_verified": False,
        "minimum_neighbor_overlap_fraction": 0.0,
        "average_neighbor_overlap_fraction": 0.0,
        "maximum_neighbor_overlap_fraction": 0.0,
        "wraparound_overlap_fraction": 0.0,
        "no_neighbor_gaps_verified": False,
    }


def _estimate_neighbor_overlap_fraction(
    first_profile: np.ndarray,
    second_profile: np.ndarray,
    *,
    first_sample_points: np.ndarray | None = None,
    second_sample_points: np.ndarray | None = None,
) -> float:
    """Estimate intersection area as a fraction of the first transformed profile."""
    if not _is_usable_profile_polygon(first_profile) or not _is_usable_profile_polygon(second_profile):
        return 0.0
    if first_sample_points is not None and second_sample_points is not None:
        if (
            first_sample_points.ndim != 2
            or first_sample_points.shape[1] != 2
            or len(first_sample_points) == 0
            or second_sample_points.ndim != 2
            or second_sample_points.shape[1] != 2
            or len(second_sample_points) == 0
        ):
            return 0.0
        first_fraction = np.count_nonzero(_points_in_polygon(first_sample_points, second_profile)) / len(
            first_sample_points
        )
        second_fraction = np.count_nonzero(_points_in_polygon(second_sample_points, first_profile)) / len(
            second_sample_points
        )
        return float((first_fraction + second_fraction) / 2.0)
    return _estimate_union_overlap_ratio(
        first_profile,
        [(second_profile, _polygon_bounds(second_profile))],
    )


def _profile_interior_sample_points(local_profile: np.ndarray) -> np.ndarray:
    """Return a rotation-stable uniform sample of the local profile interior."""
    min_x, min_y, max_x, max_y = _polygon_bounds(local_profile)
    x_step = (max_x - min_x) / CIRCULAR_OVERLAP_SAMPLE_GRID_SIZE
    y_step = (max_y - min_y) / CIRCULAR_OVERLAP_SAMPLE_GRID_SIZE
    if x_step <= 0.0 or y_step <= 0.0:
        return np.empty((0, 2), dtype=float)
    xs = min_x + (np.arange(CIRCULAR_OVERLAP_SAMPLE_GRID_SIZE) + 0.5) * x_step
    ys = min_y + (np.arange(CIRCULAR_OVERLAP_SAMPLE_GRID_SIZE) + 0.5) * y_step
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return points[_points_in_polygon(points, local_profile)]


def _circular_pair_covers_test_band(
    tank_center: tuple[float, float],
    sweep_radius: float,
    theta_start: float,
    theta_step: float,
    first_profile: np.ndarray,
    second_profile: np.ndarray,
) -> bool:
    """Sample the narrow annular strip between two consecutive scan centers."""
    angle_fractions = np.linspace(0.0, 1.0, CIRCULAR_TEST_ANGULAR_SAMPLES)
    angles = theta_start + theta_step * angle_fractions
    radial_offsets = np.array(
        [-CIRCULAR_TEST_BAND_HALF_WIDTH_M, 0.0, CIRCULAR_TEST_BAND_HALF_WIDTH_M]
    )
    radii = sweep_radius + radial_offsets
    if np.any(radii <= 0.0):
        return False
    grid_radii, grid_angles = np.meshgrid(radii, angles)
    center_x, center_y = tank_center
    points = np.column_stack(
        (
            center_x + (grid_radii * np.cos(grid_angles)).ravel(),
            center_y + (grid_radii * np.sin(grid_angles)).ravel(),
        )
    )
    covered = _points_in_or_near_polygon(points, first_profile, CIRCULAR_GAP_TOLERANCE_M)
    covered |= _points_in_or_near_polygon(points, second_profile, CIRCULAR_GAP_TOLERANCE_M)
    return bool(np.all(covered))


def _is_usable_profile_polygon(polygon: np.ndarray) -> bool:
    return (
        polygon.ndim == 2
        and polygon.shape[1] == 2
        and len(polygon) >= 3
        and bool(np.all(np.isfinite(polygon)))
        and _polygon_area(polygon) > 1e-9
    )


def determine_max_feasible_circular_rows(
    tank_radius: float,
    *,
    outer_edge_offset_m: float = DEFAULT_EDGE_OFFSET_FROM_WALL,
    row_spacing_m: float = DEFAULT_ROW_SPACING,
    profile_config: RainbowProfileConfig | None = None,
    minimum_turning_radius_m: float | None = None,
    preferred_max_rows: int = DEFAULT_MAX_CIRCULAR_ROWS,
) -> int:
    """Return the geometry-supported circular-row limit, capped by preference.

    In the absence of a separate robot turning-radius constant, the full scan
    footprint width is the conservative minimum usable path radius. This keeps
    a circular robot path from becoming tighter than the footprint it carries.
    Every generated row must still pass the existing polygon continuity,
    overlap, no-gap, and wraparound checks before mission acceptance.
    """
    if tank_radius <= 0.0:
        raise ValueError("Tank radius must be positive.")
    if outer_edge_offset_m < 0.0:
        raise ValueError("Outer edge offset must be non-negative.")
    if row_spacing_m <= 0.0:
        raise ValueError("Row spacing must be positive.")
    if preferred_max_rows < 0:
        raise ValueError("Preferred circular row count must be zero or greater.")
    if minimum_turning_radius_m is not None and minimum_turning_radius_m <= 0.0:
        raise ValueError("Minimum turning radius must be positive when provided.")

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    minimum_usable_radius = max(
        profile_config.width,
        minimum_turning_radius_m or 0.0,
    )
    sweep_radius = _profile_constrained_sweep_radius(
        tank_radius,
        profile_config,
        outer_edge_offset_m,
    )
    feasible_rows = 0
    while feasible_rows < preferred_max_rows and sweep_radius >= minimum_usable_radius:
        feasible_rows += 1
        sweep_radius -= row_spacing_m
    return feasible_rows


def _generate_circular_sweep_plan(
    tank_center: tuple[float, float],
    tank_radius: float,
    *,
    outer_edge_offset_m: float,
    row_spacing_m: float,
    spacing_factor: float,
    max_circular_gap_fraction: float,
    circular_stop_backoff_rows: int,
    profile_config: RainbowProfileConfig | None,
    clockwise: bool,
) -> CircularSweepPlan:
    _validate_sweep_inputs(
        tank_radius,
        outer_edge_offset_m,
        row_spacing_m,
        spacing_factor,
        max_circular_gap_fraction,
        circular_stop_backoff_rows,
    )
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    current_radius = _profile_constrained_sweep_radius(tank_radius, profile_config, outer_edge_offset_m)
    max_feasible_rows = determine_max_feasible_circular_rows(
        tank_radius,
        outer_edge_offset_m=outer_edge_offset_m,
        row_spacing_m=row_spacing_m,
        profile_config=profile_config,
        preferred_max_rows=DEFAULT_MAX_CIRCULAR_ROWS,
    )
    rows: list[CircularSweepRow] = []
    poses: list[SweepPose] = []
    start_theta = 0.0

    while len(rows) < max_feasible_rows:
        generated = _generate_sweep_row_poses(
            tank_center,
            current_radius,
            profile_config=profile_config,
            spacing_factor=spacing_factor,
            row_id=len(rows),
            row_name=f"circular_{len(rows)}",
            scan_id_start=len(poses),
            start_theta=start_theta,
            clockwise=clockwise,
        )
        if generated is None:
            return _apply_circular_stop_backoff(
                rows,
                poses,
                current_radius,
                "continuous_coverage_failed",
                circular_stop_backoff_rows,
            )

        row_poses, row = generated
        if row.estimated_gap_fraction > max_circular_gap_fraction:
            return _apply_circular_stop_backoff(rows, poses, current_radius, "gap", circular_stop_backoff_rows)

        rows.append(row)
        poses.extend(row_poses)
        start_theta = row_poses[-1].theta_rad
        current_radius -= row_spacing_m

    if len(rows) >= max_feasible_rows:
        return CircularSweepPlan(rows, poses, current_radius, "geometry_row_limit")

    return _apply_circular_stop_backoff(rows, poses, current_radius, "radius", circular_stop_backoff_rows)


def _generate_fixed_circular_sweep_plan(
    tank_center: tuple[float, float],
    tank_radius: float,
    *,
    outer_edge_offset_m: float,
    row_spacing_m: float,
    spacing_factor: float,
    fixed_circular_rows: int,
    max_circular_gap_fraction: float,
    profile_config: RainbowProfileConfig | None,
    clockwise: bool,
) -> CircularSweepPlan:
    """Generate exactly the requested number of circular rows, capped by the planner max."""
    if fixed_circular_rows < 0:
        raise ValueError("Fixed circular row count must be zero or greater.")
    if fixed_circular_rows > DEFAULT_MAX_CIRCULAR_ROWS:
        raise ValueError(f"Fixed circular row count cannot exceed {DEFAULT_MAX_CIRCULAR_ROWS}.")
    if fixed_circular_rows == 0:
        return CircularSweepPlan([], [], None, "fixed_row_count")

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    current_radius = _profile_constrained_sweep_radius(tank_radius, profile_config, outer_edge_offset_m)
    max_feasible_rows = determine_max_feasible_circular_rows(
        tank_radius,
        outer_edge_offset_m=outer_edge_offset_m,
        row_spacing_m=row_spacing_m,
        profile_config=profile_config,
        preferred_max_rows=fixed_circular_rows,
    )
    rows: list[CircularSweepRow] = []
    poses: list[SweepPose] = []
    start_theta = 0.0

    for row_id in range(max_feasible_rows):
        generated = _generate_sweep_row_poses(
            tank_center,
            current_radius,
            profile_config=profile_config,
            spacing_factor=spacing_factor,
            row_id=row_id,
            row_name=f"circular_{row_id}",
            scan_id_start=len(poses),
            start_theta=start_theta,
            clockwise=clockwise,
        )
        if generated is None:
            return CircularSweepPlan(rows, poses, current_radius, "continuous_coverage_failed")

        row_poses, row = generated
        if row.estimated_gap_fraction > max_circular_gap_fraction:
            return CircularSweepPlan(rows, poses, current_radius, "gap")

        rows.append(row)
        poses.extend(row_poses)
        start_theta = row_poses[-1].theta_rad
        current_radius -= row_spacing_m

    stop_reason = "fixed_row_count" if max_feasible_rows == fixed_circular_rows else "geometry_row_limit"
    return CircularSweepPlan(rows, poses, current_radius, stop_reason)


def _apply_circular_stop_backoff(
    rows: list[CircularSweepRow],
    poses: list[SweepPose],
    rejected_radius_m: float | None,
    stop_reason: str,
    backoff_rows: int,
) -> CircularSweepPlan:
    """Drop the last accepted circular rows so tight-radius wraps stop earlier."""
    if backoff_rows <= 0 or not rows:
        return CircularSweepPlan(rows, poses, rejected_radius_m, stop_reason)

    keep_count = max(1, len(rows) - backoff_rows)
    if keep_count == len(rows):
        return CircularSweepPlan(rows, poses, rejected_radius_m, stop_reason)

    first_removed_radius = rows[keep_count].sweep_radius_m
    kept_rows = rows[:keep_count]
    kept_poses = [pose for pose in poses if pose.row_id is not None and pose.row_id < keep_count]
    return CircularSweepPlan(
        kept_rows,
        kept_poses,
        first_removed_radius,
        f"backoff_{backoff_rows}_rows_after_{stop_reason}",
    )


def build_outer_edge_sweep_mission(
    model: GeometryModel,
    *,
    outer_edge_offset_m: float = DEFAULT_EDGE_OFFSET_FROM_WALL,
    row_spacing_m: float = DEFAULT_ROW_SPACING,
    spacing_factor: float = DEFAULT_SPACING_FACTOR,
    fixed_circular_rows: int | None = None,
    max_circular_gap_fraction: float = DEFAULT_MAX_CIRCULAR_GAP_FRACTION,
    circular_stop_backoff_rows: int = DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS,
    profile_config: RainbowProfileConfig | None = None,
    clockwise: bool = False,
) -> OuterEdgeSweepMission:
    """Build metadata and accepted circular edge rows from imported geometry."""
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    tank = estimate_tank_circle_from_geometry(model)
    if fixed_circular_rows is None:
        circular_plan = _generate_circular_sweep_plan(
            (tank.center_x, tank.center_y),
            tank.radius,
            outer_edge_offset_m=outer_edge_offset_m,
            row_spacing_m=row_spacing_m,
            spacing_factor=spacing_factor,
            max_circular_gap_fraction=max_circular_gap_fraction,
            circular_stop_backoff_rows=circular_stop_backoff_rows,
            profile_config=profile_config,
            clockwise=clockwise,
        )
    else:
        circular_plan = _generate_fixed_circular_sweep_plan(
            (tank.center_x, tank.center_y),
            tank.radius,
            outer_edge_offset_m=outer_edge_offset_m,
            row_spacing_m=row_spacing_m,
            spacing_factor=spacing_factor,
            fixed_circular_rows=fixed_circular_rows,
            max_circular_gap_fraction=max_circular_gap_fraction,
            profile_config=profile_config,
            clockwise=clockwise,
        )
    profile_radial_half_width = profile_config.width / 2.0
    profile_tangential_span = _profile_tangential_span(profile_config)
    profile_wall_contact_span = profile_config.side_height
    return OuterEdgeSweepMission(
        units="meters",
        tank_center_m={"x": tank.center_x, "y": tank.center_y},
        tank_radius_m=tank.radius,
        outer_edge_offset_m=outer_edge_offset_m,
        row_spacing_m=row_spacing_m,
        profile_radial_half_width_m=profile_radial_half_width,
        profile_tangential_span_m=profile_tangential_span,
        profile_wall_contact_span_m=profile_wall_contact_span,
        spacing_factor=spacing_factor,
        selected_circular_spacing_factor=None,
        circular_spacing_candidates=[],
        circular_spacing_selection_reason=None,
        max_circular_gap_fraction=max_circular_gap_fraction,
        circular_stop_backoff_rows=circular_stop_backoff_rows,
        circular_rows=circular_plan.rows,
        rejected_circular_radius_m=circular_plan.rejected_radius_m,
        circular_stop_reason=circular_plan.stop_reason,
        direction="clockwise" if clockwise else "counterclockwise",
        scan_profile={
            "profile_type": "rainbow",
            "width_m": profile_config.width,
            "arc_radius_m": profile_config.arc_radius,
            "side_height_m": profile_config.side_height,
            "tangential_span_m": profile_tangential_span,
            "wall_contact_span_m": profile_wall_contact_span,
            "arc_samples": profile_config.arc_samples,
        },
        interior_enabled=False,
        vertical_spacing_factor=DEFAULT_VERTICAL_SPACING_FACTOR,
        overlap_discard_threshold=DEFAULT_OVERLAP_DISCARD_THRESHOLD,
        vertical_columns=[],
        horizontal_sections=[],
        vertical_group_count=0,
        horizontal_group_count=0,
        vertical_scan_count=0,
        horizontal_scan_count=0,
        duplicate_poses_removed=0,
        invalid_horizontal_segments_skipped=0,
        edge_sweep_scan_count=len(circular_plan.poses),
        interior_kept_count=0,
        interior_discarded_count=0,
        total_scan_count=len(circular_plan.poses),
        poses=circular_plan.poses,
        source_dxf=str(model.source_path) if model.source_path is not None else None,
        tank_estimate_method=tank.method,
    )


def _build_mission_plan_once(
    model: GeometryModel,
    *,
    outer_edge_offset_m: float = DEFAULT_EDGE_OFFSET_FROM_WALL,
    row_spacing_m: float = DEFAULT_ROW_SPACING,
    spacing_factor: float = DEFAULT_SPACING_FACTOR,
    fixed_circular_rows: int | None = None,
    max_circular_gap_fraction: float = DEFAULT_MAX_CIRCULAR_GAP_FRACTION,
    circular_stop_backoff_rows: int = DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS,
    vertical_spacing_factor: float = DEFAULT_VERTICAL_SPACING_FACTOR,
    overlap_discard_threshold: float = DEFAULT_OVERLAP_DISCARD_THRESHOLD,
    interior_enabled: bool = True,
    profile_config: RainbowProfileConfig | None = None,
    clockwise: bool = False,
) -> OuterEdgeSweepMission:
    """Build one mission using one circular spacing factor."""
    if vertical_spacing_factor <= 0.0:
        raise ValueError("Vertical spacing factor must be positive.")
    if not 0.0 <= overlap_discard_threshold <= 1.0:
        raise ValueError("Overlap discard threshold must be between 0 and 1.")

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    edge_mission = build_outer_edge_sweep_mission(
        model,
        outer_edge_offset_m=outer_edge_offset_m,
        row_spacing_m=row_spacing_m,
        spacing_factor=spacing_factor,
        fixed_circular_rows=fixed_circular_rows,
        max_circular_gap_fraction=max_circular_gap_fraction,
        circular_stop_backoff_rows=circular_stop_backoff_rows,
        profile_config=profile_config,
        clockwise=clockwise,
    )
    if not interior_enabled:
        return replace(
            edge_mission,
            interior_enabled=False,
            vertical_spacing_factor=vertical_spacing_factor,
            overlap_discard_threshold=overlap_discard_threshold,
        )

    guide_lines = generate_lawnmower_section_lines(model)
    gating_pass = gate_lawnmower_candidates(
        guide_lines,
        TankCircleEstimate(
            center_x=edge_mission.tank_center_m["x"],
            center_y=edge_mission.tank_center_m["y"],
            radius=edge_mission.tank_radius_m,
            method=edge_mission.tank_estimate_method,
        ),
        edge_mission.poses,
        profile_config=profile_config,
    )
    lawnmower_poses = _lawnmower_placements_to_sweep_poses(
        guide_lines,
        gating_pass.placements,
        scan_id_start=len(edge_mission.poses),
    )
    combined_interior, duplicate_count = _deduplicate_interior_poses(
        lawnmower_poses,
        scan_id_start=len(edge_mission.poses),
    )
    vertical_group_ids = {
        pose.group_id
        for pose in combined_interior
        if pose.stage == "interior_vertical" and pose.group_id is not None
    }
    horizontal_group_ids = {
        pose.group_id
        for pose in combined_interior
        if pose.stage == "interior_horizontal" and pose.group_id is not None
    }
    vertical_scan_count = sum(pose.stage == "interior_vertical" for pose in combined_interior)
    horizontal_scan_count = sum(pose.stage == "interior_horizontal" for pose in combined_interior)
    return replace(
        edge_mission,
        interior_enabled=True,
        vertical_spacing_factor=vertical_spacing_factor,
        overlap_discard_threshold=overlap_discard_threshold,
        vertical_columns=[],
        horizontal_sections=[],
        vertical_group_count=len(vertical_group_ids),
        horizontal_group_count=len(horizontal_group_ids),
        vertical_scan_count=vertical_scan_count,
        horizontal_scan_count=horizontal_scan_count,
        duplicate_poses_removed=duplicate_count,
        invalid_horizontal_segments_skipped=0,
        interior_kept_count=len(combined_interior),
        interior_discarded_count=(
            len(gating_pass.candidates) - len(gating_pass.placements)
        ),
        total_scan_count=len(edge_mission.poses) + len(combined_interior),
        poses=edge_mission.poses + combined_interior,
        lawnmower_section_lines=guide_lines,
        rejected_full_profile_candidate_count=sum(
            candidate.acceptance_result != "accepted_full"
            for candidate in gating_pass.candidates
        ),
        salvaged_half_profile_count=sum(
            placement.profile_variant in {"left_half", "right_half"}
            for placement in gating_pass.placements
        ),
    )


def build_mission_plan(
    model: GeometryModel,
    *,
    outer_edge_offset_m: float = DEFAULT_EDGE_OFFSET_FROM_WALL,
    row_spacing_m: float = DEFAULT_ROW_SPACING,
    spacing_factor: float = DEFAULT_SPACING_FACTOR,
    circular_spacing_candidates: tuple[float, ...] | None = DEFAULT_CIRCULAR_SPACING_CANDIDATES,
    max_circular_gap_fraction: float = DEFAULT_MAX_CIRCULAR_GAP_FRACTION,
    circular_stop_backoff_rows: int = DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS,
    vertical_spacing_factor: float = DEFAULT_VERTICAL_SPACING_FACTOR,
    overlap_discard_threshold: float = DEFAULT_OVERLAP_DISCARD_THRESHOLD,
    interior_enabled: bool = True,
    profile_config: RainbowProfileConfig | None = None,
    clockwise: bool = False,
) -> OuterEdgeSweepMission:
    """Build one mission using the authoritative geometry-derived row limit."""
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    mission = _build_mission_plan_once(
        model,
        outer_edge_offset_m=outer_edge_offset_m,
        row_spacing_m=row_spacing_m,
        spacing_factor=spacing_factor,
        max_circular_gap_fraction=max_circular_gap_fraction,
        circular_stop_backoff_rows=circular_stop_backoff_rows,
        vertical_spacing_factor=vertical_spacing_factor,
        overlap_discard_threshold=overlap_discard_threshold,
        interior_enabled=interior_enabled,
        profile_config=profile_config,
        clockwise=clockwise,
    )
    if not circular_spacing_candidates:
        return mission

    coverage = _estimate_tank_coverage(mission, profile_config, grid_resolution=80)
    candidate_record = CircularSpacingCandidate(
        circular_rows=len(mission.circular_rows),
        spacing_factor=None,
        coverage_percent=coverage.covered_percent,
        scan_count=len(mission.poses),
        travel_distance_m=_estimate_mission_travel_distance(mission.poses),
        selected=True,
    )
    row_count = len(mission.circular_rows)
    return replace(
        mission,
        spacing_factor=spacing_factor,
        selected_circular_spacing_factor=None,
        circular_spacing_candidates=[candidate_record],
        circular_spacing_selection_reason=(
            f"geometry supports {row_count} usable circular row"
            f"{'s' if row_count != 1 else ''} (preferred maximum {DEFAULT_MAX_CIRCULAR_ROWS})"
        ),
    )


def _estimate_mission_travel_distance(poses: list[SweepPose]) -> float:
    if len(poses) < 2:
        return 0.0
    distance = 0.0
    for first, second in zip(poses, poses[1:]):
        distance += math.hypot(second.x_m - first.x_m, second.y_m - first.y_m)
    return float(distance)


def detect_vertical_plate_columns(
    model: GeometryModel,
    tank: TankCircleEstimate | None = None,
) -> list[VerticalPlateColumn]:
    """Compatibility wrapper for the existing vertical-column representation."""
    sections, _invalid_count = detect_interior_sections(model, "vertical", tank)
    return [
        VerticalPlateColumn(
            column_id=section.section_id,
            center_x_m=section.center_cross_m,
            left_weld_x_m=section.lower_weld_coordinate_m,
            right_weld_x_m=section.upper_weld_coordinate_m,
            min_y_m=section.progress_min_m,
            max_y_m=section.progress_max_m,
        )
        for section in sections
    ]


def detect_horizontal_plate_rows(
    model: GeometryModel,
    tank: TankCircleEstimate | None = None,
) -> tuple[list[InteriorSection], int]:
    """Detect horizontal plate sections between long near-horizontal weld runs."""
    return detect_interior_sections(model, "horizontal", tank)


def detect_interior_sections(
    model: GeometryModel,
    orientation: str,
    tank: TankCircleEstimate | None = None,
) -> tuple[list[InteriorSection], int]:
    """Detect plate sections between adjacent structural welds along either principal axis."""
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("Interior section orientation must be 'vertical' or 'horizontal'.")
    tank = estimate_tank_circle_from_geometry(model) if tank is None else tank
    grouped_segments = _group_axis_segments(_iter_geometry_segments(model), orientation)
    qualifying_lines: list[tuple[float, float, float, int]] = []
    invalid_count = 0

    for cross_m, intervals, source_count in grouped_segments:
        merge_tolerance = HORIZONTAL_FRAGMENT_MERGE_GAP_M if orientation == "horizontal" else 1e-6
        merged = _merge_intervals(intervals, tolerance=merge_tolerance)
        center_cross = tank.center_x if orientation == "vertical" else tank.center_y
        radial_offset = cross_m - center_cross
        chord_length = 2.0 * math.sqrt(max(0.0, tank.radius**2 - radial_offset**2))
        if chord_length <= 1e-9 or not merged:
            invalid_count += max(1, source_count)
            continue

        if orientation == "vertical":
            union_length = sum(end - start for start, end in merged)
            if union_length / chord_length < VERTICAL_LINE_COVERAGE_RATIO:
                invalid_count += source_count
                continue
            qualifying_lines.append(
                (
                    cross_m,
                    min(start for start, _end in merged),
                    max(end for _start, end in merged),
                    source_count,
                )
            )
            continue

        for start, end in merged:
            if (end - start) / chord_length < VERTICAL_LINE_COVERAGE_RATIO:
                invalid_count += 1
                continue
            qualifying_lines.append((cross_m, start, end, source_count))

    qualifying_lines.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(qualifying_lines) < 2:
        return [], invalid_count

    unique_cross_values = sorted({line[0] for line in qualifying_lines})
    positive_gaps = np.diff(unique_cross_values)
    positive_gaps = positive_gaps[positive_gaps > 1e-9]
    if len(positive_gaps) == 0:
        return [], invalid_count + len(qualifying_lines)
    typical_gap = float(np.median(positive_gaps))
    max_regular_gap = typical_gap * 1.5
    sections: list[InteriorSection] = []

    for lower_index, lower in enumerate(qualifying_lines):
        for upper in qualifying_lines[lower_index + 1 :]:
            gap = upper[0] - lower[0]
            if gap <= 1e-9:
                continue
            if gap > max_regular_gap:
                break
            if any(lower[0] < other_cross < upper[0] for other_cross in unique_cross_values):
                continue
            progress_min = max(lower[1], upper[1])
            progress_max = min(lower[2], upper[2])
            if progress_max <= progress_min:
                invalid_count += 1
                continue
            section_id = len(sections)
            sections.append(
                InteriorSection(
                    section_id=section_id,
                    group_id=section_id,
                    orientation=orientation,
                    center_cross_m=(lower[0] + upper[0]) / 2.0,
                    lower_weld_coordinate_m=lower[0],
                    upper_weld_coordinate_m=upper[0],
                    progress_min_m=progress_min,
                    progress_max_m=progress_max,
                    source_segment_count=lower[3] + upper[3],
                )
            )
    return sections, invalid_count


def generate_lawnmower_section_lines(
    model: GeometryModel,
    tank: TankCircleEstimate | None = None,
) -> list[LawnmowerSectionLine]:
    """Return direction-neutral vertical and horizontal centerlines without poses or paths."""
    tank = estimate_tank_circle_from_geometry(model) if tank is None else tank
    vertical_specs = _vertical_lawnmower_section_specs(model, tank)
    horizontal_specs = _horizontal_lawnmower_section_specs(model, tank, vertical_specs)

    lines: list[LawnmowerSectionLine] = []
    vertical_order_index = 0
    for center_x_m, min_y_m, max_y_m, source_section_id in vertical_specs:
        clipped = _clip_axis_section_to_tank(
            "vertical",
            center_x_m,
            min_y_m,
            max_y_m,
            tank,
        )
        if clipped is None:
            continue
        order_index = vertical_order_index
        vertical_order_index += 1
        lower, upper = clipped
        start = (center_x_m, lower)
        end = (center_x_m, upper)
        lines.append(
            LawnmowerSectionLine(
                line_id=len(lines),
                orientation="vertical",
                order_index=order_index,
                source_section_id=source_section_id,
                start_m=start,
                end_m=end,
            )
        )

    horizontal_order_index = 0
    for center_y_m, min_x_m, max_x_m, source_section_id in horizontal_specs:
        clipped = _clip_axis_section_to_tank(
            "horizontal",
            center_y_m,
            min_x_m,
            max_x_m,
            tank,
        )
        if clipped is None:
            continue
        order_index = horizontal_order_index
        horizontal_order_index += 1
        lower, upper = clipped
        start = (lower, center_y_m)
        end = (upper, center_y_m)
        lines.append(
            LawnmowerSectionLine(
                line_id=len(lines),
                orientation="horizontal",
                order_index=order_index,
                source_section_id=source_section_id,
                start_m=start,
                end_m=end,
            )
        )
    # The normal planner uses only guide sections derived from the detected
    # weld geometry.  The experimental outer extensions remain diagnostic-only
    # and must not participate in normal mission generation.
    return lines


def _add_outer_lawnmower_sections(
    lines: list[LawnmowerSectionLine],
    tank: TankCircleEstimate,
) -> list[LawnmowerSectionLine]:
    """Append the next natural guide on each populated outer grid edge.

    Existing structural guides remain intact.  For each orientation with a
    stable grid (at least three distinct cross-axis coordinates), this adds at
    most one low-side and one high-side guide at the median neighboring-grid
    spacing.  The candidate is clipped by the same circular tank-interior
    rule used by detected guides.
    """
    result = list(lines)
    next_line_id = max((line.line_id for line in result), default=-1) + 1
    tolerance = SECTION_BOUNDARY_MATCH_TOLERANCE_M
    profile_config = RainbowProfileConfig()
    transverse_half_width = profile_config.width / 2.0
    # The regular placement lattice can overhang a guide endpoint by less than
    # one profile-depth step.  Reserving the configured arc radius covers that
    # existing behavior without constructing a profile during guide detection.
    endpoint_margin = profile_config.arc_radius

    for orientation in ("vertical", "horizontal"):
        existing = [line for line in result if line.orientation == orientation]
        if orientation == "vertical":
            cross_coordinates = sorted({round(line.start_m[0], 12) for line in existing})
        else:
            cross_coordinates = sorted({round(line.start_m[1], 12) for line in existing})
        if len(cross_coordinates) < 3:
            continue
        gaps = np.diff(cross_coordinates)
        gaps = gaps[gaps > tolerance]
        if len(gaps) == 0:
            continue
        spacing = float(np.median(gaps))
        if spacing <= tolerance:
            continue
        spacing_tolerance = max(
            HORIZONTAL_PLATE_SPACING_MIN_TOLERANCE_M,
            spacing * HORIZONTAL_PLATE_SPACING_TOLERANCE_FRACTION,
        )
        if any(abs(float(gap) - spacing) > spacing_tolerance for gap in gaps):
            continue

        source_ids = [line.source_section_id for line in existing]
        candidates = (
            (cross_coordinates[0] - spacing, min(source_ids) - 1),
            (cross_coordinates[-1] + spacing, max(source_ids) + 1),
        )
        for cross_m, source_section_id in candidates:
            if any(abs(cross_m - coordinate) <= tolerance for coordinate in cross_coordinates):
                continue
            center_cross = tank.center_x if orientation == "vertical" else tank.center_y
            progress_center = tank.center_y if orientation == "vertical" else tank.center_x
            outer_cross_offset = abs(cross_m - center_cross) + transverse_half_width
            if outer_cross_offset >= tank.radius:
                continue
            half_progress = math.sqrt(tank.radius**2 - outer_cross_offset**2) - endpoint_margin
            if half_progress <= SECTION_MIN_LENGTH_M:
                continue
            lower = progress_center - half_progress
            upper = progress_center + half_progress
            if orientation == "vertical":
                start_m = (cross_m, lower)
                end_m = (cross_m, upper)
            else:
                start_m = (lower, cross_m)
                end_m = (upper, cross_m)
            result.append(
                LawnmowerSectionLine(
                    line_id=next_line_id,
                    orientation=orientation,
                    order_index=0,
                    source_section_id=source_section_id,
                    start_m=start_m,
                    end_m=end_m,
                    is_outer_extension=True,
                )
            )
            next_line_id += 1

    ordered: list[LawnmowerSectionLine] = []
    for orientation in ("vertical", "horizontal"):
        orientation_lines = [line for line in result if line.orientation == orientation]
        if orientation == "vertical":
            orientation_lines.sort(key=lambda line: (line.start_m[0], line.start_m[1], line.end_m[1], line.line_id))
        else:
            orientation_lines.sort(key=lambda line: (line.start_m[1], line.start_m[0], line.end_m[0], line.line_id))
        ordered.extend(
            replace(line, order_index=index)
            for index, line in enumerate(orientation_lines)
        )
    return ordered


def generate_lawnmower_scan_placements(
    lines: list[LawnmowerSectionLine],
    *,
    profile_config: RainbowProfileConfig | None = None,
    target_overlap_fraction: float = DEFAULT_LAWNMOWER_NEIGHBOR_OVERLAP_FRACTION,
) -> list[LawnmowerScanPlacement]:
    """Return the unchanged accepted full-profile placements."""
    return _generate_lawnmower_placement_pass(
        lines,
        profile_config=profile_config,
        target_overlap_fraction=target_overlap_fraction,
    ).accepted


def _generate_lawnmower_placement_pass(
    lines: list[LawnmowerSectionLine],
    *,
    profile_config: RainbowProfileConfig | None = None,
    target_overlap_fraction: float = DEFAULT_LAWNMOWER_NEIGHBOR_OVERLAP_FRACTION,
) -> LawnmowerPlacementPass:
    """Place rainbow profiles along existing guide lines without altering them.

    Each placement is translated from the midpoint of the rainbow's outer
    1.600 m chord.  The chord remains perpendicular to the ordered guide
    line while the profile's local positive-y direction follows travel.  The
    accepted list is identical to the public full-profile placer; rejected
    normal-lattice candidates are retained separately with structured reasons.
    """
    if not 0.0 < target_overlap_fraction < 1.0:
        raise ValueError("Lawnmower neighbor overlap fraction must be between 0 and 1.")

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    local_profile = make_rainbow_profile(profile_config)
    long_side_start, long_side_end = _rainbow_long_side_endpoints(local_profile)
    local_anchor = (long_side_start + long_side_end) / 2.0
    target_step = _lawnmower_spacing_for_overlap(local_profile, target_overlap_fraction)

    placements: list[LawnmowerScanPlacement] = []
    rejected: list[RejectedFullProfileCandidate] = []
    candidate_id = 0
    for line in lines:
        start = np.asarray(line.start_m, dtype=float)
        end = np.asarray(line.end_m, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= SECTION_MIN_LENGTH_M:
            continue
        direction = delta / length
        heading_rad = _normalize_angle(math.atan2(direction[1], direction[0]) - math.pi / 2.0)
        canonical_start, canonical_direction = _canonical_lawnmower_line_reference(start, end)
        anchor_distances = _lawnmower_anchor_distances(
            length,
            local_profile,
            local_anchor,
            target_step,
        )
        anchor_points = [canonical_start + canonical_direction * distance_m for distance_m in anchor_distances]
        if float(np.dot(direction, canonical_direction)) < 0.0:
            anchor_points.reverse()
        travel_direction = _line_travel_direction(direction)
        rotation = np.array(
            [
                [math.cos(heading_rad), -math.sin(heading_rad)],
                [math.sin(heading_rad), math.cos(heading_rad)],
            ],
            dtype=float,
        )
        candidate_specs: list[tuple[int, np.ndarray, FullProfileRejectionReason | None]] = []
        if anchor_points:
            candidate_specs.append(
                (-1, anchor_points[0] - direction * target_step, FullProfileRejectionReason.SECTION_BOUNDARY_CONFLICT)
            )
        candidate_specs.extend(
            (placement_index, anchor, None)
            for placement_index, anchor in enumerate(anchor_points)
        )
        if anchor_points:
            candidate_specs.append(
                (
                    len(anchor_points),
                    anchor_points[-1] + direction * target_step,
                    FullProfileRejectionReason.SECTION_BOUNDARY_CONFLICT,
                )
            )

        for placement_index, anchor, preliminary_rejection in candidate_specs:
            profile_origin = anchor - local_anchor @ rotation.T
            candidate = LawnmowerScanPlacement(
                placement_id=len(placements),
                guide_line_id=line.line_id,
                guide_order_index=line.order_index,
                placement_index=placement_index,
                orientation=line.orientation,
                travel_direction=travel_direction,
                anchor_m=(float(anchor[0]), float(anchor[1])),
                profile_origin_m=(float(profile_origin[0]), float(profile_origin[1])),
                heading_rad=heading_rad,
                heading_deg=math.degrees(heading_rad),
            )
            rejection_reason = preliminary_rejection
            if rejection_reason is None and _is_effectively_duplicate_lawnmower_placement(candidate, placements):
                rejection_reason = FullProfileRejectionReason.DUPLICATE_POSE
            if rejection_reason is not None:
                full_polygon = transform_profile(
                    local_profile,
                    candidate.profile_origin_m[0],
                    candidate.profile_origin_m[1],
                    candidate.heading_rad,
                )
                rejected.append(
                    RejectedFullProfileCandidate(
                        candidate_id=candidate_id,
                        guide_line_id=candidate.guide_line_id,
                        guide_order_index=candidate.guide_order_index,
                        placement_index=candidate.placement_index,
                        projected_order_m=float(np.dot(anchor - start, direction)),
                        orientation=candidate.orientation,
                        travel_direction=candidate.travel_direction,
                        anchor_m=candidate.anchor_m,
                        profile_origin_m=candidate.profile_origin_m,
                        heading_rad=candidate.heading_rad,
                        heading_deg=candidate.heading_deg,
                        full_polygon=full_polygon,
                        rejection_reasons=(rejection_reason,),
                    )
                )
                candidate_id += 1
                continue
            placements.append(candidate)
            candidate_id += 1
    return LawnmowerPlacementPass(accepted=placements, rejected=rejected)


def _generate_lawnmower_candidate_lattice(
    lines: list[LawnmowerSectionLine],
    *,
    profile_config: RainbowProfileConfig | None = None,
    target_overlap_fraction: float = DEFAULT_LAWNMOWER_NEIGHBOR_OVERLAP_FRACTION,
) -> list[LawnmowerCandidateRecord]:
    """Generate every unchanged normal-lattice candidate without gating it."""
    if not 0.0 < target_overlap_fraction < 1.0:
        raise ValueError("Lawnmower neighbor overlap fraction must be between 0 and 1.")

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    halves = make_rainbow_profile_halves(profile_config)
    local_profile = halves.full_polygon
    local_anchor = np.asarray(halves.long_side_anchor_m, dtype=float)
    target_step = _lawnmower_spacing_for_overlap(local_profile, target_overlap_fraction)
    ordered_lines = sorted(
        lines,
        key=lambda line: (
            0 if line.orientation == "vertical" else 1,
            line.order_index,
            line.line_id,
        ),
    )
    records: list[LawnmowerCandidateRecord] = []
    for line in ordered_lines:
        start = np.asarray(line.start_m, dtype=float)
        end = np.asarray(line.end_m, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= SECTION_MIN_LENGTH_M:
            continue
        direction = delta / length
        heading_rad = _normalize_angle(math.atan2(direction[1], direction[0]) - math.pi / 2.0)
        canonical_start, canonical_direction = _canonical_lawnmower_line_reference(start, end)
        anchor_distances = _lawnmower_anchor_distances(
            length,
            local_profile,
            local_anchor,
            target_step,
        )
        anchor_points = [
            canonical_start + canonical_direction * distance_m
            for distance_m in anchor_distances
        ]
        if float(np.dot(direction, canonical_direction)) < 0.0:
            anchor_points.reverse()
        travel_direction = _line_travel_direction(direction)
        rotation = np.array(
            [
                [math.cos(heading_rad), -math.sin(heading_rad)],
                [math.sin(heading_rad), math.cos(heading_rad)],
            ],
            dtype=float,
        )
        for placement_index, anchor in enumerate(anchor_points):
            profile_origin = anchor - local_anchor @ rotation.T
            x_m = float(profile_origin[0])
            y_m = float(profile_origin[1])
            records.append(
                LawnmowerCandidateRecord(
                    candidate_id=len(records),
                    guide_line_id=line.line_id,
                    guide_order_index=line.order_index,
                    placement_index=placement_index,
                    orientation=line.orientation,
                    travel_direction=travel_direction,
                    projected_position_m=float(np.dot(anchor - start, direction)),
                    anchor_m=(float(anchor[0]), float(anchor[1])),
                    profile_origin_m=(x_m, y_m),
                    heading_rad=heading_rad,
                    heading_deg=math.degrees(heading_rad),
                    full_polygon=transform_profile(local_profile, x_m, y_m, heading_rad),
                    left_half_polygon=transform_rainbow_half(
                        halves.left_half,
                        x_m,
                        y_m,
                        heading_rad,
                    ),
                    right_half_polygon=transform_rainbow_half(
                        halves.right_half,
                        x_m,
                        y_m,
                        heading_rad,
                    ),
                    candidate_source="normal_lattice",
                    candidate_stage=(
                        "interior_vertical"
                        if line.orientation == "vertical"
                        else "interior_horizontal"
                    ),
                )
            )
    return records


def _exact_lawnmower_candidate_metrics(
    candidate_polygon: np.ndarray,
    valid_region: BaseGeometry,
    existing_coverage: BaseGeometry,
    immediate_neighbor: AcceptedLawnmowerCoverage | None,
) -> LawnmowerCandidateMetrics:
    candidate = as_polygon(candidate_polygon)
    total_area = float(candidate.area)
    if total_area <= GEOMETRY_EPSILON:
        return LawnmowerCandidateMetrics(
            total_area_m2=total_area,
            inside_area_m2=0.0,
            outside_area_m2=0.0,
            inside_fraction=0.0,
            outside_fraction=0.0,
            total_overlap_area_m2=0.0,
            expected_neighbor_overlap_area_m2=0.0,
            harmful_overlap_area_m2=0.0,
            harmful_overlap_fraction=0.0,
            new_area_m2=0.0,
            new_fraction=0.0,
        )
    inside_polygon = candidate.intersection(valid_region)
    outside_polygon = candidate.difference(valid_region)
    total_overlap_polygon = inside_polygon.intersection(existing_coverage)
    new_polygon = inside_polygon.difference(existing_coverage)
    inside_area = float(inside_polygon.area)
    outside_area = float(outside_polygon.area)
    total_overlap_area = float(total_overlap_polygon.area)
    new_area = float(new_polygon.area)
    allowed_neighbor_overlap = 0.0
    if immediate_neighbor is not None:
        actual_neighbor_overlap = float(
            inside_polygon.intersection(as_polygon(immediate_neighbor.polygon)).area
        )
        allowed_neighbor_overlap = min(
            actual_neighbor_overlap,
            MAX_EXPECTED_NEIGHBOR_OVERLAP_FRACTION
            * min(inside_area, immediate_neighbor.inside_area_m2),
        )
    harmful_overlap_area = max(0.0, total_overlap_area - allowed_neighbor_overlap)
    return LawnmowerCandidateMetrics(
        total_area_m2=total_area,
        inside_area_m2=inside_area,
        outside_area_m2=outside_area,
        inside_fraction=inside_area / max(total_area, GEOMETRY_EPSILON),
        outside_fraction=outside_area / max(total_area, GEOMETRY_EPSILON),
        total_overlap_area_m2=total_overlap_area,
        expected_neighbor_overlap_area_m2=allowed_neighbor_overlap,
        harmful_overlap_area_m2=harmful_overlap_area,
        harmful_overlap_fraction=harmful_overlap_area / max(inside_area, GEOMETRY_EPSILON),
        new_area_m2=new_area,
        new_fraction=new_area / max(inside_area, GEOMETRY_EPSILON),
    )


def _full_candidate_rejection_reasons(
    candidate: LawnmowerCandidateRecord,
    metrics: LawnmowerCandidateMetrics,
    accepted_placements: list[LawnmowerScanPlacement],
) -> tuple[str, ...]:
    reasons: list[str] = []
    geometry = as_polygon(candidate.full_polygon)
    if (
        not math.isfinite(candidate.heading_rad)
        or not bool(np.all(np.isfinite(candidate.full_polygon)))
        or geometry.is_empty
        or not geometry.is_valid
        or metrics.total_area_m2 <= GEOMETRY_EPSILON
    ):
        reasons.append("invalid_geometry")
    if (
        metrics.inside_fraction < MIN_FULL_INSIDE_FRACTION
        or metrics.outside_fraction > MAX_FULL_OUTSIDE_FRACTION
    ):
        reasons.append("outside_valid_region")
    if metrics.new_area_m2 < MIN_FULL_NEW_AREA_M2:
        reasons.append("insufficient_new_area")
    if metrics.new_fraction < MIN_FULL_NEW_FRACTION:
        reasons.append("insufficient_new_fraction")
    if metrics.harmful_overlap_fraction > MAX_FULL_HARMFUL_OVERLAP_FRACTION:
        reasons.append("excessive_harmful_overlap")
    hypothetical = _placement_from_lawnmower_candidate(
        candidate,
        profile_variant="full",
        placement_id=-1,
    )
    if _is_effectively_duplicate_lawnmower_placement(hypothetical, accepted_placements):
        reasons.append("duplicate_pose")
    return tuple(dict.fromkeys(reasons))


def _half_metrics_pass(metrics: LawnmowerCandidateMetrics) -> bool:
    return (
        metrics.inside_fraction >= MIN_HALF_INSIDE_FRACTION
        and metrics.outside_fraction <= MAX_HALF_OUTSIDE_FRACTION
        and metrics.new_area_m2 >= MIN_HALF_NEW_AREA_M2
        and metrics.new_fraction >= MIN_HALF_NEW_FRACTION
        and metrics.harmful_overlap_fraction <= MAX_HALF_HARMFUL_OVERLAP_FRACTION
    )


def _select_salvaged_half(
    rejection_reasons: tuple[str, ...],
    left_metrics: LawnmowerCandidateMetrics,
    right_metrics: LawnmowerCandidateMetrics,
) -> tuple[str | None, str]:
    reasons = set(rejection_reasons)
    if reasons & {"invalid_geometry", "duplicate_pose"}:
        return None, "non_salvageable_rejection"
    salvageable = {
        "outside_valid_region",
        "excessive_harmful_overlap",
        "insufficient_new_area",
        "insufficient_new_fraction",
    }
    if not reasons or not reasons.issubset(salvageable):
        return None, "non_salvageable_rejection"

    insufficient_reasons = {"insufficient_new_area", "insufficient_new_fraction"}
    if reasons & insufficient_reasons:
        combined_new_area = left_metrics.new_area_m2 + right_metrics.new_area_m2
        useful_concentration = max(
            left_metrics.new_area_m2,
            right_metrics.new_area_m2,
        ) / max(combined_new_area, GEOMETRY_EPSILON)
        if useful_concentration < MIN_BURDEN_DOMINANCE:
            return None, "useful_new_area_not_one_sided"

    left_burden = (
        left_metrics.harmful_overlap_area_m2
        + OUTSIDE_BURDEN_WEIGHT * left_metrics.outside_area_m2
    )
    right_burden = (
        right_metrics.harmful_overlap_area_m2
        + OUTSIDE_BURDEN_WEIGHT * right_metrics.outside_area_m2
    )
    combined_burden = left_burden + right_burden
    dominant_burden_share = max(left_burden, right_burden) / max(
        combined_burden,
        GEOMETRY_EPSILON,
    )
    if dominant_burden_share < MIN_BURDEN_DOMINANCE:
        return None, "burden_not_one_sided"

    passing = [
        variant
        for variant, metrics in (
            ("left_half", left_metrics),
            ("right_half", right_metrics),
        )
        if _half_metrics_pass(metrics)
    ]
    if len(passing) == 0:
        return None, "neither_half_passes"
    if len(passing) == 1:
        return passing[0], f"accepted_{passing[0]}"

    scores = {
        "left_half": (
            left_metrics.new_area_m2
            - left_metrics.harmful_overlap_area_m2
            - OUTSIDE_BURDEN_WEIGHT * left_metrics.outside_area_m2
        ),
        "right_half": (
            right_metrics.new_area_m2
            - right_metrics.harmful_overlap_area_m2
            - OUTSIDE_BURDEN_WEIGHT * right_metrics.outside_area_m2
        ),
    }
    higher = max(scores, key=scores.get)
    lower = "right_half" if higher == "left_half" else "left_half"
    if scores[higher] - scores[lower] >= MIN_HALF_SCORE_ADVANTAGE_M2:
        return higher, f"accepted_{higher}"
    return None, "ambiguous_halves"


def _placement_from_lawnmower_candidate(
    candidate: LawnmowerCandidateRecord,
    *,
    profile_variant: str,
    placement_id: int,
) -> LawnmowerScanPlacement:
    return LawnmowerScanPlacement(
        placement_id=placement_id,
        guide_line_id=candidate.guide_line_id,
        guide_order_index=candidate.guide_order_index,
        placement_index=candidate.placement_index,
        orientation=candidate.orientation,
        travel_direction=candidate.travel_direction,
        anchor_m=candidate.anchor_m,
        profile_origin_m=candidate.profile_origin_m,
        heading_rad=candidate.heading_rad,
        heading_deg=candidate.heading_deg,
        profile_variant=profile_variant,
        parent_profile_origin_m=(
            candidate.profile_origin_m if profile_variant != "full" else None
        ),
    )


def gate_lawnmower_candidates(
    lines: list[LawnmowerSectionLine],
    tank: TankCircleEstimate,
    edge_poses: list[SweepPose],
    *,
    profile_config: RainbowProfileConfig | None = None,
) -> LawnmowerGatingPass:
    """Apply the single authoritative full/half exact-geometry gate."""
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    halves = make_rainbow_profile_halves(profile_config)
    local_profile = halves.full_polygon
    local_anchor = np.asarray(halves.long_side_anchor_m, dtype=float)
    target_step = _lawnmower_spacing_for_overlap(
        local_profile,
        DEFAULT_LAWNMOWER_NEIGHBOR_OVERLAP_FRACTION,
    )
    candidates = _generate_lawnmower_candidate_lattice(
        lines,
        profile_config=profile_config,
    )
    candidates_by_guide: dict[int, list[LawnmowerCandidateRecord]] = {}
    for candidate in candidates:
        candidates_by_guide.setdefault(candidate.guide_line_id, []).append(candidate)

    circular_polygons = [
        polygon
        for polygon, _bounds in _circular_edge_polygon_records(local_profile, edge_poses)
    ]
    coverage = polygon_union(circular_polygons)
    accepted_placements: list[LawnmowerScanPlacement] = []
    evaluated_candidates: list[LawnmowerCandidateRecord] = []
    retained_guide_ids: list[int] = []
    discarded_guide_ids: list[int] = []
    ordered_lines = sorted(
        lines,
        key=lambda line: (
            0 if line.orientation == "vertical" else 1,
            line.order_index,
            line.line_id,
        ),
    )
    for line in ordered_lines:
        coverage_before_guide = coverage
        guide_placements: list[LawnmowerScanPlacement] = []
        guide_polygons: list[np.ndarray] = []
        guide_candidate_indices: list[int] = []
        immediate_neighbor: AcceptedLawnmowerCoverage | None = None
        valid_region = build_candidate_valid_region(
            line.start_m,
            line.end_m,
            (tank.center_x, tank.center_y),
            tank.radius,
            local_profile,
            local_anchor,
            target_step,
        )
        for candidate in candidates_by_guide.get(line.line_id, []):
            full_metrics = _exact_lawnmower_candidate_metrics(
                candidate.full_polygon,
                valid_region,
                coverage,
                immediate_neighbor,
            )
            left_metrics = _exact_lawnmower_candidate_metrics(
                candidate.left_half_polygon,
                valid_region,
                coverage,
                immediate_neighbor,
            )
            right_metrics = _exact_lawnmower_candidate_metrics(
                candidate.right_half_polygon,
                valid_region,
                coverage,
                immediate_neighbor,
            )
            reasons = _full_candidate_rejection_reasons(
                candidate,
                full_metrics,
                accepted_placements + guide_placements,
            )
            selected_variant: str | None = None
            result = "rejected_full"
            if not reasons:
                selected_variant = "full"
                result = "accepted_full"
            else:
                selected_variant, result = _select_salvaged_half(
                    reasons,
                    left_metrics,
                    right_metrics,
                )

            evaluated = replace(
                candidate,
                acceptance_result=result,
                rejection_reasons=reasons,
                full_metrics=full_metrics,
                left_half_metrics=left_metrics,
                right_half_metrics=right_metrics,
            )
            evaluated_candidates.append(evaluated)
            guide_candidate_indices.append(len(evaluated_candidates) - 1)
            if selected_variant is None:
                continue

            placement = _placement_from_lawnmower_candidate(
                candidate,
                profile_variant=selected_variant,
                placement_id=len(accepted_placements) + len(guide_placements),
            )
            selected_polygon = {
                "full": candidate.full_polygon,
                "left_half": candidate.left_half_polygon,
                "right_half": candidate.right_half_polygon,
            }[selected_variant]
            selected_metrics = {
                "full": full_metrics,
                "left_half": left_metrics,
                "right_half": right_metrics,
            }[selected_variant]
            guide_placements.append(placement)
            guide_polygons.append(selected_polygon)
            coverage = coverage.union(as_polygon(selected_polygon))
            immediate_neighbor = AcceptedLawnmowerCoverage(
                guide_line_id=line.line_id,
                orientation=line.orientation,
                projected_position_m=candidate.projected_position_m,
                profile_variant=selected_variant,
                polygon=selected_polygon,
                inside_area_m2=selected_metrics.inside_area_m2,
            )

        guide_union = polygon_union(guide_polygons)
        guide_incremental_area = float(guide_union.difference(coverage_before_guide).area)
        guide_new_fraction = guide_incremental_area / max(
            float(guide_union.area),
            GEOMETRY_EPSILON,
        )
        retain_guide = (
            bool(guide_placements)
            and guide_incremental_area >= MIN_GUIDE_INCREMENTAL_AREA_M2
            and guide_new_fraction >= MIN_GUIDE_NEW_FRACTION
        )
        if retain_guide:
            retained_guide_ids.append(line.line_id)
            accepted_placements.extend(guide_placements)
        else:
            discarded_guide_ids.append(line.line_id)
            coverage = coverage_before_guide
            for candidate_index in guide_candidate_indices:
                evaluated = evaluated_candidates[candidate_index]
                if evaluated.acceptance_result.startswith("accepted_"):
                    evaluated_candidates[candidate_index] = replace(
                        evaluated,
                        acceptance_result="guide_discarded",
                        rejection_reasons=evaluated.rejection_reasons
                        + ("guide_insufficient_incremental_coverage",),
                    )

    accepted_placements = [
        replace(placement, placement_id=index)
        for index, placement in enumerate(accepted_placements)
    ]
    return LawnmowerGatingPass(
        placements=accepted_placements,
        candidates=evaluated_candidates,
        retained_guide_ids=retained_guide_ids,
        discarded_guide_ids=discarded_guide_ids,
    )


def salvage_rejected_lawnmower_candidates(
    lines: list[LawnmowerSectionLine],
    placement_pass: LawnmowerPlacementPass,
    tank: TankCircleEstimate,
    edge_poses: list[SweepPose],
    *,
    profile_config: RainbowProfileConfig | None = None,
) -> LawnmowerHalfSalvagePass:
    """Run the deterministic second pass over rejected interior full candidates."""
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    local_profile = make_rainbow_profile(profile_config)
    long_side_start, long_side_end = _rainbow_long_side_endpoints(local_profile)
    local_anchor = (long_side_start + long_side_end) / 2.0
    target_step = _lawnmower_spacing_for_overlap(
        local_profile,
        DEFAULT_LAWNMOWER_NEIGHBOR_OVERLAP_FRACTION,
    )
    lines_by_id = {line.line_id: line for line in lines}
    full_polygons = [
        lawnmower_scan_profile_polygon(placement, profile_config)
        for placement in placement_pass.accepted
    ]
    circular_polygons = [
        polygon
        for polygon, _bounds in _circular_edge_polygon_records(local_profile, edge_poses)
    ]
    base_coverage = polygon_union(circular_polygons + full_polygons)
    current_coverage = base_coverage
    orientation_coverage = {
        orientation: polygon_union(
            [
                polygon
                for placement, polygon in zip(placement_pass.accepted, full_polygons)
                if placement.orientation == orientation
            ]
        )
        for orientation in ("vertical", "horizontal")
    }

    accepted_halves: list[LawnmowerScanPlacement] = []
    evaluations: list[HalfProfileSalvageEvaluation] = []
    next_placement_id = max(
        (placement.placement_id for placement in placement_pass.accepted),
        default=-1,
    ) + 1
    ordered_rejected = sorted(
        placement_pass.rejected,
        key=lambda candidate: (
            0 if candidate.orientation == "vertical" else 1,
            candidate.guide_order_index,
            candidate.projected_order_m,
            candidate.candidate_id,
        ),
    )
    for rejected in ordered_rejected:
        line = lines_by_id.get(rejected.guide_line_id)
        if line is None:
            classified = replace(
                rejected,
                rejection_reasons=tuple(
                    dict.fromkeys(
                        rejected.rejection_reasons
                        + (FullProfileRejectionReason.MISSING_GUIDE,)
                    )
                ),
            )
            continue
        valid_region = build_candidate_valid_region(
            line.start_m,
            line.end_m,
            (tank.center_x, tank.center_y),
            tank.radius,
            local_profile,
            local_anchor,
            target_step,
        )
        opposite_orientation = "horizontal" if rejected.orientation == "vertical" else "vertical"
        classified = add_geometric_rejection_reasons(
            rejected,
            valid_region,
            base_coverage,
            orientation_coverage[opposite_orientation],
        )
        if not classified.is_salvageable:
            continue
        evaluation = evaluate_half_profile_salvage(
            classified,
            valid_region,
            current_coverage,
            profile_config=profile_config,
        )
        evaluations.append(evaluation)
        if evaluation.selected_variant is None:
            continue
        selected = LawnmowerScanPlacement(
            placement_id=next_placement_id,
            guide_line_id=classified.guide_line_id,
            guide_order_index=classified.guide_order_index,
            placement_index=classified.placement_index,
            orientation=classified.orientation,
            travel_direction=classified.travel_direction,
            anchor_m=classified.anchor_m,
            profile_origin_m=classified.profile_origin_m,
            heading_rad=classified.heading_rad,
            heading_deg=classified.heading_deg,
            profile_variant=evaluation.selected_variant,
            parent_profile_origin_m=classified.profile_origin_m,
        )
        next_placement_id += 1
        accepted_halves.append(selected)
        current_coverage = current_coverage.union(
            as_polygon(lawnmower_scan_profile_polygon(selected, profile_config))
        )
    return LawnmowerHalfSalvagePass(placements=accepted_halves, evaluations=evaluations)


def _lawnmower_placements_to_sweep_poses(
    lines: list[LawnmowerSectionLine],
    placements: list[LawnmowerScanPlacement],
    *,
    scan_id_start: int,
) -> list[SweepPose]:
    """Convert approved anchored lawnmower placements into mission-compatible poses."""
    lines_by_id = {line.line_id: line for line in lines}
    ordered = sorted(
        placements,
        key=lambda placement: (
            0 if placement.orientation == "vertical" else 1,
            placement.guide_order_index,
            placement.placement_index,
        ),
    )
    poses: list[SweepPose] = []
    for placement in ordered:
        line = lines_by_id[placement.guide_line_id]
        stage = "interior_vertical" if placement.orientation == "vertical" else "interior_horizontal"
        poses.append(
            SweepPose(
                scan_id=scan_id_start + len(poses),
                stage=stage,
                x_m=placement.profile_origin_m[0],
                y_m=placement.profile_origin_m[1],
                heading_rad=placement.heading_rad,
                heading_deg=placement.heading_deg,
                row_id=placement.guide_line_id,
                row_name=f"lawnmower_{placement.orientation}_guide",
                column_id=line.source_section_id if placement.orientation == "vertical" else None,
                orientation=placement.orientation,
                group_id=placement.guide_order_index,
                section_id=placement.guide_line_id,
                travel_direction=placement.travel_direction,
                profile_variant=placement.profile_variant,
                anchor_x_m=placement.anchor_m[0],
                anchor_y_m=placement.anchor_m[1],
                parent_full_x_m=(
                    placement.parent_profile_origin_m[0]
                    if placement.parent_profile_origin_m is not None
                    else None
                ),
                parent_full_y_m=(
                    placement.parent_profile_origin_m[1]
                    if placement.parent_profile_origin_m is not None
                    else None
                ),
            )
        )
    return poses


def lawnmower_scan_profile_polygon(
    placement: LawnmowerScanPlacement,
    profile_config: RainbowProfileConfig | None = None,
) -> np.ndarray:
    """Return the transformed full or local-half lawnmower footprint."""
    if placement.profile_variant == "full":
        local_profile = make_rainbow_profile(profile_config)
    else:
        halves = make_rainbow_profile_halves(profile_config)
        if placement.profile_variant == "left_half":
            local_profile = halves.left_half.polygon
        elif placement.profile_variant == "right_half":
            local_profile = halves.right_half.polygon
        else:
            raise ValueError(f"Unsupported lawnmower profile variant: {placement.profile_variant}")
    return transform_profile(
        local_profile,
        placement.profile_origin_m[0],
        placement.profile_origin_m[1],
        placement.heading_rad,
    )


def scan_pose_profile_polygon(
    pose: SweepPose,
    profile_config: RainbowProfileConfig | None = None,
) -> np.ndarray:
    """Return the serialized pose's actual full or local-half footprint."""
    if pose.stage == "circular_edge" and pose.profile_variant != "full":
        raise ValueError("Circular sweep poses cannot use half-profile variants.")
    if pose.profile_variant == "full":
        local_profile = make_rainbow_profile(profile_config)
    else:
        halves = make_rainbow_profile_halves(profile_config)
        if pose.profile_variant == "left_half":
            local_profile = halves.left_half.polygon
        elif pose.profile_variant == "right_half":
            local_profile = halves.right_half.polygon
        else:
            raise ValueError(f"Unsupported scan profile variant: {pose.profile_variant}")
    return transform_profile(local_profile, pose.x_m, pose.y_m, pose.heading_rad)


def lawnmower_long_side_endpoints(
    placement: LawnmowerScanPlacement,
    profile_config: RainbowProfileConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the transformed endpoints of the profile's 1.600 m outer chord."""
    local_profile = make_rainbow_profile(profile_config)
    start, end = _rainbow_long_side_endpoints(local_profile)
    rotation = np.array(
        [
            [math.cos(placement.heading_rad), -math.sin(placement.heading_rad)],
            [math.sin(placement.heading_rad), math.cos(placement.heading_rad)],
        ],
        dtype=float,
    )
    origin = np.asarray(placement.profile_origin_m, dtype=float)
    return start @ rotation.T + origin, end @ rotation.T + origin


def estimate_lawnmower_neighbor_overlap_fraction(
    first: LawnmowerScanPlacement,
    second: LawnmowerScanPlacement,
    *,
    profile_config: RainbowProfileConfig | None = None,
) -> float:
    """Estimate the actual polygon intersection area fraction for neighboring scans."""
    local_profile = make_rainbow_profile(profile_config)
    sample_points = _profile_interior_sample_points(local_profile)
    first_polygon = transform_profile(
        local_profile,
        first.profile_origin_m[0],
        first.profile_origin_m[1],
        first.heading_rad,
    )
    second_polygon = transform_profile(
        local_profile,
        second.profile_origin_m[0],
        second.profile_origin_m[1],
        second.heading_rad,
    )
    first_samples = transform_profile(
        sample_points,
        first.profile_origin_m[0],
        first.profile_origin_m[1],
        first.heading_rad,
    )
    second_samples = transform_profile(
        sample_points,
        second.profile_origin_m[0],
        second.profile_origin_m[1],
        second.heading_rad,
    )
    return _estimate_neighbor_overlap_fraction(
        first_polygon,
        second_polygon,
        first_sample_points=first_samples,
        second_sample_points=second_samples,
    )


def _rainbow_long_side_endpoints(local_profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the outer 1.600 m chord from the upper endpoints of the side drops."""
    min_x = float(np.min(local_profile[:, 0]))
    max_x = float(np.max(local_profile[:, 0]))
    tolerance = 1e-9
    left_candidates = local_profile[np.isclose(local_profile[:, 0], min_x, atol=tolerance)]
    right_candidates = local_profile[np.isclose(local_profile[:, 0], max_x, atol=tolerance)]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError("Rainbow profile does not contain identifiable long-side endpoints.")
    left = left_candidates[np.argmax(left_candidates[:, 1])]
    right = right_candidates[np.argmax(right_candidates[:, 1])]
    expected_width = max_x - min_x
    if not math.isclose(float(np.linalg.norm(right - left)), expected_width, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Rainbow profile does not preserve its long-side chord width.")
    return left.copy(), right.copy()


def _lawnmower_spacing_for_overlap(
    local_profile: np.ndarray,
    target_overlap_fraction: float,
) -> float:
    """Find local-y translation with the requested sampled polygon-area overlap."""
    depth = float(np.max(local_profile[:, 1]) - np.min(local_profile[:, 1]))
    if depth <= SECTION_MIN_LENGTH_M:
        raise ValueError("Rainbow profile must have positive depth along a lawnmower guide line.")
    sample_points = _profile_interior_sample_points(local_profile)
    lower = 0.0
    upper = depth
    for _ in range(45):
        midpoint = (lower + upper) / 2.0
        overlap = _estimate_neighbor_overlap_fraction(
            local_profile,
            local_profile + np.array([0.0, midpoint]),
            first_sample_points=sample_points,
            second_sample_points=sample_points + np.array([0.0, midpoint]),
        )
        if overlap > target_overlap_fraction:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0)


def _lawnmower_anchor_distances(
    line_length_m: float,
    local_profile: np.ndarray,
    local_anchor: np.ndarray,
    target_step_m: float,
) -> np.ndarray:
    relative_progress = local_profile[:, 1] - local_anchor[1]
    minimum_progress = float(np.min(relative_progress))
    maximum_progress = float(np.max(relative_progress))
    coverage_depth = maximum_progress - minimum_progress
    if coverage_depth <= SECTION_MIN_LENGTH_M:
        return np.array([line_length_m / 2.0])

    first_anchor = max(0.0, -minimum_progress)
    last_anchor = min(line_length_m, line_length_m - maximum_progress)
    if line_length_m <= coverage_depth:
        if last_anchor < first_anchor:
            return np.array([line_length_m / 2.0])
        return np.array([(first_anchor + last_anchor) / 2.0])

    anchor_span = last_anchor - first_anchor
    interval_count = max(1, int(math.ceil(anchor_span / target_step_m)))
    total_coverage = coverage_depth + interval_count * target_step_m
    endpoint_overhang = (total_coverage - line_length_m) / 2.0
    first_anchor -= endpoint_overhang
    return first_anchor + np.arange(interval_count + 1, dtype=float) * target_step_m


def _line_travel_direction(direction: np.ndarray) -> str:
    if abs(direction[0]) >= abs(direction[1]):
        return "right" if direction[0] >= 0.0 else "left"
    return "up" if direction[1] >= 0.0 else "down"


def _canonical_lawnmower_line_reference(
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable endpoint order so reversing a guide reverses only traversal."""
    if (float(start[0]), float(start[1])) <= (float(end[0]), float(end[1])):
        canonical_start = start
        canonical_end = end
    else:
        canonical_start = end
        canonical_end = start
    canonical_direction = canonical_end - canonical_start
    canonical_direction /= np.linalg.norm(canonical_direction)
    return canonical_start, canonical_direction


def _is_effectively_duplicate_lawnmower_placement(
    candidate: LawnmowerScanPlacement,
    placements: list[LawnmowerScanPlacement],
) -> bool:
    return any(
        math.hypot(
            candidate.profile_origin_m[0] - existing.profile_origin_m[0],
            candidate.profile_origin_m[1] - existing.profile_origin_m[1],
        )
        <= INTERIOR_POSE_POSITION_TOLERANCE_M
        and _angle_difference(candidate.heading_rad, existing.heading_rad)
        <= INTERIOR_POSE_ANGLE_TOLERANCE_RAD
        for existing in placements
    )


def _vertical_lawnmower_section_specs(
    model: GeometryModel,
    tank: TankCircleEstimate,
) -> list[tuple[float, float, float, int]]:
    """Preserve existing column centers while retaining real vertical fragment gaps."""
    columns = detect_vertical_plate_columns(model, tank)
    grouped = _group_axis_segments(_iter_geometry_segments(model), "vertical")
    specs: list[tuple[float, float, float, int]] = []
    for column in columns:
        left_intervals = _axis_group_intervals_at_coordinate(grouped, column.left_weld_x_m)
        right_intervals = _axis_group_intervals_at_coordinate(grouped, column.right_weld_x_m)
        if not left_intervals or not right_intervals:
            continue
        common_intervals = _intersect_interval_sets(
            _merge_intervals(left_intervals),
            _merge_intervals(right_intervals),
        )
        for start, end in common_intervals:
            start = max(start, column.min_y_m)
            end = min(end, column.max_y_m)
            if end - start > SECTION_MIN_LENGTH_M:
                specs.append((column.center_x_m, start, end, column.column_id))
    specs = _add_short_vertical_fragment_columns(specs, grouped, columns)
    return sorted(specs, key=lambda item: (item[0], item[1], item[2], item[3]))


def _add_short_vertical_fragment_columns(
    specs: list[tuple[float, float, float, int]],
    groups: list[tuple[float, list[tuple[float, float]], int]],
    columns: list[VerticalPlateColumn],
) -> list[tuple[float, float, float, int]]:
    """Add plate centers bounded by short weld fragments in the dominant row bands."""
    widths = [column.right_weld_x_m - column.left_weld_x_m for column in columns]
    if not specs or not widths:
        return specs
    target_spacing = float(np.median(widths))
    spacing_tolerance = max(
        HORIZONTAL_PLATE_SPACING_MIN_TOLERANCE_M,
        target_spacing * HORIZONTAL_PLATE_SPACING_TOLERANCE_FRACTION,
    )

    interval_bands: list[list[float]] = []
    for _center_x_m, start_m, end_m, _source_section_id in specs:
        matched_band = next(
            (
                band
                for band in interval_bands
                if abs(band[0] - start_m) <= SECTION_BOUNDARY_MATCH_TOLERANCE_M
                and abs(band[1] - end_m) <= SECTION_BOUNDARY_MATCH_TOLERANCE_M
            ),
            None,
        )
        if matched_band is None:
            interval_bands.append([start_m, end_m, 1.0])
        else:
            count = matched_band[2]
            matched_band[0] = (matched_band[0] * count + start_m) / (count + 1.0)
            matched_band[1] = (matched_band[1] * count + end_m) / (count + 1.0)
            matched_band[2] = count + 1.0
    maximum_support = max(band[2] for band in interval_bands)
    dominant_bands = [
        (band[0], band[1])
        for band in interval_bands
        if band[2] >= maximum_support - 0.5
    ]
    if len(dominant_bands) < 3:
        return specs

    merged_groups = [
        (cross_m, _merge_intervals(intervals))
        for cross_m, intervals, _source_count in groups
    ]
    paired_centers: list[float] = []
    for left_index, (left_x_m, left_intervals) in enumerate(merged_groups):
        for right_x_m, right_intervals in merged_groups[left_index + 1 :]:
            separation = right_x_m - left_x_m
            if separation > target_spacing + spacing_tolerance:
                break
            if abs(separation - target_spacing) > spacing_tolerance:
                continue
            common_intervals = _intersect_interval_sets(left_intervals, right_intervals)
            if any(
                min(common_end, band_end) - max(common_start, band_start) > SECTION_MIN_LENGTH_M
                for common_start, common_end in common_intervals
                for band_start, band_end in dominant_bands
            ):
                paired_centers.append((left_x_m + right_x_m) / 2.0)

    result = list(specs)
    next_source_id = max(spec[3] for spec in specs) + 1
    for center_x_m in sorted(paired_centers):
        for band_start, band_end in sorted(dominant_bands):
            duplicate = any(
                abs(existing_x_m - center_x_m) <= SECTION_BOUNDARY_MATCH_TOLERANCE_M
                and abs(existing_start_m - band_start) <= SECTION_BOUNDARY_MATCH_TOLERANCE_M
                and abs(existing_end_m - band_end) <= SECTION_BOUNDARY_MATCH_TOLERANCE_M
                for existing_x_m, existing_start_m, existing_end_m, _source_id in result
            )
            if not duplicate:
                result.append((center_x_m, band_start, band_end, next_source_id))
                next_source_id += 1
    return result


def _horizontal_lawnmower_section_specs(
    model: GeometryModel,
    tank: TankCircleEstimate,
    vertical_specs: list[tuple[float, float, float, int]],
) -> list[tuple[float, float, float, int]]:
    """Pair horizontal weld fragments at the detected plate short-side spacing."""
    groups = _group_axis_segments(_iter_geometry_segments(model), "horizontal")
    if len(groups) < 2:
        return []
    merged_groups = [
        (cross_m, _merge_intervals(intervals, HORIZONTAL_FRAGMENT_MERGE_GAP_M))
        for cross_m, intervals, _source_count in groups
    ]
    target_spacing = _detected_plate_short_side_spacing(model, tank, merged_groups)
    if target_spacing is None:
        return []
    spacing_tolerance = max(
        HORIZONTAL_PLATE_SPACING_MIN_TOLERANCE_M,
        target_spacing * HORIZONTAL_PLATE_SPACING_TOLERANCE_FRACTION,
    )

    raw_specs: list[tuple[float, float, float]] = []
    for lower_index, (lower_y, lower_intervals) in enumerate(merged_groups):
        for upper_y, upper_intervals in merged_groups[lower_index + 1 :]:
            separation = upper_y - lower_y
            if separation > target_spacing + spacing_tolerance:
                break
            if abs(separation - target_spacing) > spacing_tolerance:
                continue
            for progress_min, progress_max in _intersect_interval_sets(lower_intervals, upper_intervals):
                clipped = _clip_axis_section_to_tank(
                    "horizontal",
                    (lower_y + upper_y) / 2.0,
                    progress_min,
                    progress_max,
                    tank,
                )
                if clipped is not None:
                    raw_specs.append(((lower_y + upper_y) / 2.0, clipped[0], clipped[1]))

    grouped_specs: list[tuple[float, list[tuple[float, float]]]] = []
    for center_y_m, progress_min_m, progress_max_m in sorted(raw_specs):
        if grouped_specs and abs(center_y_m - grouped_specs[-1][0]) <= HORIZONTAL_Y_GROUP_TOLERANCE_M:
            grouped_specs[-1][1].append((progress_min_m, progress_max_m))
        else:
            grouped_specs.append((center_y_m, [(progress_min_m, progress_max_m)]))

    specs: list[tuple[float, float, float, int]] = []
    for center_y_m, intervals in grouped_specs:
        for progress_min_m, progress_max_m in _merge_intervals(intervals, HORIZONTAL_FRAGMENT_MERGE_GAP_M):
            for clear_min_m, clear_max_m in _subtract_active_vertical_contacts(
                center_y_m,
                progress_min_m,
                progress_max_m,
                vertical_specs,
            ):
                if clear_max_m - clear_min_m > SECTION_MIN_LENGTH_M:
                    specs.append((center_y_m, clear_min_m, clear_max_m, len(specs)))
    return sorted(specs, key=lambda item: (item[0], item[1], item[2], item[3]))


def _detected_plate_short_side_spacing(
    model: GeometryModel,
    tank: TankCircleEstimate,
    horizontal_groups: list[tuple[float, list[tuple[float, float]]]],
) -> float | None:
    columns = detect_vertical_plate_columns(model, tank)
    widths = [column.right_weld_x_m - column.left_weld_x_m for column in columns]
    if widths:
        return float(np.median(widths))
    cross_values = sorted(cross_m for cross_m, intervals in horizontal_groups if intervals)
    gaps = np.diff(cross_values)
    gaps = gaps[gaps > SECTION_MIN_LENGTH_M]
    if len(gaps) == 0:
        return None
    return float(np.median(gaps))


def _subtract_active_vertical_contacts(
    center_y_m: float,
    progress_min_m: float,
    progress_max_m: float,
    vertical_specs: list[tuple[float, float, float, int]],
) -> list[tuple[float, float]]:
    intervals = [(progress_min_m, progress_max_m)]
    blockers = sorted(
        center_x_m
        for center_x_m, min_y_m, max_y_m, _source_section_id in vertical_specs
        if min_y_m - SECTION_MIN_LENGTH_M <= center_y_m <= max_y_m + SECTION_MIN_LENGTH_M
        and progress_min_m - PERPENDICULAR_SECTION_CLEARANCE_M
        <= center_x_m
        <= progress_max_m + PERPENDICULAR_SECTION_CLEARANCE_M
    )
    for blocker_x_m in blockers:
        blocked_min = blocker_x_m - PERPENDICULAR_SECTION_CLEARANCE_M
        blocked_max = blocker_x_m + PERPENDICULAR_SECTION_CLEARANCE_M
        next_intervals: list[tuple[float, float]] = []
        for start, end in intervals:
            if blocked_max <= start or blocked_min >= end:
                next_intervals.append((start, end))
                continue
            if blocked_min - start > SECTION_MIN_LENGTH_M:
                next_intervals.append((start, min(end, blocked_min)))
            if end - blocked_max > SECTION_MIN_LENGTH_M:
                next_intervals.append((max(start, blocked_max), end))
        intervals = next_intervals
    return intervals


def _axis_group_intervals_at_coordinate(
    groups: list[tuple[float, list[tuple[float, float]], int]],
    coordinate_m: float,
) -> list[tuple[float, float]]:
    closest = min(groups, key=lambda item: abs(item[0] - coordinate_m), default=None)
    if closest is None or abs(closest[0] - coordinate_m) > SECTION_BOUNDARY_MATCH_TOLERANCE_M:
        return []
    return closest[1]


def _intersect_interval_sets(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    first_index = 0
    second_index = 0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        start = max(first_start, second_start)
        end = min(first_end, second_end)
        if end - start > SECTION_MIN_LENGTH_M:
            intersections.append((start, end))
        if first_end < second_end:
            first_index += 1
        else:
            second_index += 1
    return intersections


def _clip_axis_section_to_tank(
    orientation: str,
    cross_m: float,
    progress_min_m: float,
    progress_max_m: float,
    tank: TankCircleEstimate,
) -> tuple[float, float] | None:
    if orientation == "vertical":
        cross_offset = cross_m - tank.center_x
        progress_center = tank.center_y
    elif orientation == "horizontal":
        cross_offset = cross_m - tank.center_y
        progress_center = tank.center_x
    else:
        raise ValueError("Section orientation must be 'vertical' or 'horizontal'.")
    if abs(cross_offset) >= tank.radius:
        return None
    half_chord = math.sqrt(max(0.0, tank.radius**2 - cross_offset**2))
    lower = max(progress_min_m, progress_center - half_chord)
    upper = min(progress_max_m, progress_center + half_chord)
    if upper - lower <= SECTION_MIN_LENGTH_M:
        return None
    return float(lower), float(upper)


def generate_interior_vertical_poses(
    columns: list[VerticalPlateColumn],
    tank: TankCircleEstimate,
    edge_poses: list[SweepPose],
    *,
    scan_id_start: int,
    profile_config: RainbowProfileConfig,
    vertical_spacing_factor: float,
    overlap_discard_threshold: float,
) -> tuple[list[SweepPose], int]:
    """Generate alternating up/down interior profile stacks and reject edge overlap."""
    local_profile = make_rainbow_profile(profile_config)
    touching_step = _touching_vertical_step(local_profile)
    target_step = touching_step * vertical_spacing_factor
    edge_polygon_records = _circular_edge_polygon_records(local_profile, edge_poses)
    regular_columns = _expand_vertical_columns_to_edge(
        columns,
        tank,
        local_profile,
        edge_polygon_records,
        vertical_spacing_factor=vertical_spacing_factor,
        overlap_limit=DEFAULT_VERTICAL_COLUMN_EDGE_OVERLAP_LIMIT,
    )

    kept: list[SweepPose] = []
    discarded_count = 0
    for column_index, column in enumerate(regular_columns):
        column_poses, column_discarded = _generate_regular_vertical_column_poses(
            column,
            tank,
            local_profile,
            edge_polygon_records,
            scan_id_start=scan_id_start + len(kept),
            column_index=column_index,
            target_step=target_step,
            overlap_discard_threshold=overlap_discard_threshold,
        )
        kept.extend(column_poses)
        discarded_count += column_discarded

    side_guard_poses, side_guard_discarded = _generate_side_guard_poses(
        regular_columns,
        tank,
        edge_polygon_records,
        scan_id_start=scan_id_start + len(kept),
        profile_config=profile_config,
        vertical_spacing_factor=vertical_spacing_factor,
        overlap_discard_threshold=overlap_discard_threshold,
    )
    kept.extend(side_guard_poses)
    discarded_count += side_guard_discarded
    return kept, discarded_count


def generate_interior_horizontal_poses(
    sections: list[InteriorSection],
    tank: TankCircleEstimate,
    edge_poses: list[SweepPose],
    *,
    scan_id_start: int,
    profile_config: RainbowProfileConfig,
    vertical_spacing_factor: float,
    overlap_discard_threshold: float,
    previous_pose: SweepPose | None = None,
) -> tuple[list[SweepPose], int, int]:
    """Generate alternating left/right scans along detected horizontal sections."""
    local_profile = make_rainbow_profile(profile_config)
    heading_rad = math.pi / 2.0
    rotated_profile = transform_profile(local_profile, 0.0, 0.0, heading_rad)
    target_step = _touching_vertical_step(local_profile) * vertical_spacing_factor
    edge_polygon_records = _circular_edge_polygon_records(local_profile, edge_poses)
    ordered_sections = sorted(sections, key=lambda section: (section.center_cross_m, section.progress_min_m))

    valid_rows: list[tuple[InteriorSection, np.ndarray]] = []
    invalid_count = 0
    for section in ordered_sections:
        x_positions = _horizontal_section_x_positions(section, tank, rotated_profile, target_step)
        if x_positions is None or len(x_positions) == 0:
            invalid_count += 1
            continue
        valid_rows.append((section, x_positions))
    if not valid_rows:
        return [], 0, invalid_count

    first_positions = valid_rows[0][1]
    first_moves_right = True
    if previous_pose is not None:
        first_section = valid_rows[0][0]
        left_distance = math.hypot(
            previous_pose.x_m - float(first_positions[0]),
            previous_pose.y_m - first_section.center_cross_m,
        )
        right_distance = math.hypot(
            previous_pose.x_m - float(first_positions[-1]),
            previous_pose.y_m - first_section.center_cross_m,
        )
        first_moves_right = left_distance <= right_distance

    kept: list[SweepPose] = []
    discarded_count = 0
    for row_index, (section, x_positions) in enumerate(valid_rows):
        moves_right = first_moves_right if row_index % 2 == 0 else not first_moves_right
        ordered_x = x_positions if moves_right else x_positions[::-1]
        travel_direction = "right" if moves_right else "left"
        for x_m in ordered_x:
            world_profile = transform_profile(
                local_profile,
                float(x_m),
                section.center_cross_m,
                heading_rad,
            )
            overlap_ratio = _estimate_union_overlap_ratio(world_profile, edge_polygon_records)
            if overlap_ratio > overlap_discard_threshold:
                discarded_count += 1
                continue
            kept.append(
                SweepPose(
                    scan_id=scan_id_start + len(kept),
                    stage="interior_horizontal",
                    x_m=float(x_m),
                    y_m=section.center_cross_m,
                    heading_rad=heading_rad,
                    heading_deg=math.degrees(heading_rad),
                    row_id=section.section_id,
                    row_name="horizontal_section",
                    orientation="horizontal",
                    group_id=row_index,
                    section_id=section.section_id,
                    travel_direction=travel_direction,
                )
            )
    return kept, discarded_count, invalid_count


def _horizontal_section_x_positions(
    section: InteriorSection,
    tank: TankCircleEstimate,
    rotated_profile: np.ndarray,
    target_step: float,
) -> np.ndarray | None:
    x_bounds = _profile_center_x_bounds_at_y(
        section.center_cross_m,
        tank,
        rotated_profile,
        section.progress_min_m,
        section.progress_max_m,
    )
    if x_bounds is None:
        return None
    x_min, x_max = x_bounds
    span = x_max - x_min
    interval_count = max(1, int(math.ceil(span / target_step)))
    return np.linspace(x_min, x_max, interval_count + 1)


def _circular_edge_polygon_records(
    local_profile: np.ndarray,
    poses: list[SweepPose],
) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    polygons = [
        transform_profile(local_profile, pose.x_m, pose.y_m, pose.heading_rad)
        for pose in poses
        if pose.stage == "circular_edge"
    ]
    return [(polygon, _polygon_bounds(polygon)) for polygon in polygons]


def _deduplicate_interior_poses(
    poses: list[SweepPose],
    *,
    scan_id_start: int,
) -> tuple[list[SweepPose], int]:
    kept: list[SweepPose] = []
    removed = 0
    for pose in poses:
        duplicate = any(
            math.hypot(pose.x_m - existing.x_m, pose.y_m - existing.y_m)
            <= INTERIOR_POSE_POSITION_TOLERANCE_M
            and _angle_difference(pose.heading_rad, existing.heading_rad)
            <= INTERIOR_POSE_ANGLE_TOLERANCE_RAD
            for existing in kept
        )
        if duplicate:
            removed += 1
            continue
        kept.append(replace(pose, scan_id=scan_id_start + len(kept)))
    return kept, removed


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _expand_vertical_columns_to_edge(
    columns: list[VerticalPlateColumn],
    tank: TankCircleEstimate,
    local_profile: np.ndarray,
    edge_polygon_records: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    *,
    vertical_spacing_factor: float,
    overlap_limit: float,
) -> list[VerticalPlateColumn]:
    """Extend regular vertical columns outward until the next full column hits the edge sweeps too much."""
    if not columns:
        return []

    ordered = sorted(columns, key=lambda column: column.center_x_m)
    spacing = _detected_vertical_column_spacing(ordered, local_profile)
    target_step = _touching_vertical_step(local_profile) * vertical_spacing_factor
    y_min_hint = min(column.min_y_m for column in ordered)
    y_max_hint = max(column.max_y_m for column in ordered)

    left_expansions: list[VerticalPlateColumn] = []
    candidate_x = ordered[0].center_x_m - spacing
    candidate_id = -1
    while True:
        candidate = VerticalPlateColumn(
            column_id=candidate_id,
            center_x_m=candidate_x,
            left_weld_x_m=candidate_x - spacing / 2.0,
            right_weld_x_m=candidate_x + spacing / 2.0,
            min_y_m=y_min_hint,
            max_y_m=y_max_hint,
        )
        if not _regular_vertical_column_is_acceptable(
            candidate,
            tank,
            local_profile,
            edge_polygon_records,
            target_step=target_step,
            overlap_limit=overlap_limit,
        ):
            break
        left_expansions.append(candidate)
        candidate_x -= spacing
        candidate_id -= 1

    right_expansions: list[VerticalPlateColumn] = []
    candidate_x = ordered[-1].center_x_m + spacing
    candidate_id = 1000
    while True:
        candidate = VerticalPlateColumn(
            column_id=candidate_id,
            center_x_m=candidate_x,
            left_weld_x_m=candidate_x - spacing / 2.0,
            right_weld_x_m=candidate_x + spacing / 2.0,
            min_y_m=y_min_hint,
            max_y_m=y_max_hint,
        )
        if not _regular_vertical_column_is_acceptable(
            candidate,
            tank,
            local_profile,
            edge_polygon_records,
            target_step=target_step,
            overlap_limit=overlap_limit,
        ):
            break
        right_expansions.append(candidate)
        candidate_x += spacing
        candidate_id += 1

    return list(reversed(left_expansions)) + ordered + right_expansions


def _detected_vertical_column_spacing(columns: list[VerticalPlateColumn], local_profile: np.ndarray) -> float:
    if len(columns) < 2:
        return float(np.max(local_profile[:, 0]) - np.min(local_profile[:, 0]))
    gaps = np.diff([column.center_x_m for column in columns])
    positive_gaps = gaps[gaps > 1e-9]
    if len(positive_gaps) == 0:
        return float(np.max(local_profile[:, 0]) - np.min(local_profile[:, 0]))
    return float(np.median(positive_gaps))


def _regular_vertical_column_is_acceptable(
    column: VerticalPlateColumn,
    tank: TankCircleEstimate,
    local_profile: np.ndarray,
    edge_polygon_records: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    *,
    target_step: float,
    overlap_limit: float,
) -> bool:
    y_positions = _regular_vertical_column_y_positions(column, tank, local_profile, target_step)
    if y_positions is None:
        return False
    if not edge_polygon_records:
        return True
    middle_index = len(y_positions) // 2
    middle_y = float(y_positions[middle_index])
    world_profile = transform_profile(local_profile, column.center_x_m, middle_y, 0.0)
    middle_overlap = _estimate_union_overlap_ratio(world_profile, edge_polygon_records)
    return middle_overlap <= overlap_limit


def _generate_regular_vertical_column_poses(
    column: VerticalPlateColumn,
    tank: TankCircleEstimate,
    local_profile: np.ndarray,
    edge_polygon_records: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    *,
    scan_id_start: int,
    column_index: int,
    target_step: float,
    overlap_discard_threshold: float,
) -> tuple[list[SweepPose], int]:
    y_positions = _regular_vertical_column_y_positions(column, tank, local_profile, target_step)
    if y_positions is None:
        return [], 0
    travel_direction = "down" if column_index % 2 == 1 else "up"
    if travel_direction == "down":
        y_positions = y_positions[::-1]

    kept: list[SweepPose] = []
    discarded_count = 0
    for y_m in y_positions:
        world_profile = transform_profile(local_profile, column.center_x_m, float(y_m), 0.0)
        overlap_ratio = _estimate_union_overlap_ratio(world_profile, edge_polygon_records)
        if overlap_ratio > overlap_discard_threshold:
            discarded_count += 1
            continue
        kept.append(
            SweepPose(
                scan_id=scan_id_start + len(kept),
                stage="interior_vertical",
                x_m=column.center_x_m,
                y_m=float(y_m),
                heading_rad=0.0,
                heading_deg=0.0,
                column_id=column.column_id,
                row_name="expanded_vertical" if column.column_id < 0 or column.column_id >= 1000 else None,
                orientation="vertical",
                group_id=column_index,
                section_id=column.column_id,
                travel_direction=travel_direction,
            )
        )
    return kept, discarded_count


def _regular_vertical_column_y_positions(
    column: VerticalPlateColumn,
    tank: TankCircleEstimate,
    local_profile: np.ndarray,
    target_step: float,
) -> np.ndarray | None:
    y_bounds = _interior_profile_center_y_bounds(column, tank, local_profile)
    if y_bounds is None:
        return None
    y_min, y_max = y_bounds
    span = y_max - y_min
    interval_count = max(1, int(math.ceil(span / target_step)))
    return np.linspace(y_min, y_max, interval_count + 1)


def _generate_side_guard_poses(
    columns: list[VerticalPlateColumn],
    tank: TankCircleEstimate,
    edge_polygon_records: list[tuple[np.ndarray, tuple[float, float, float, float]]],
    *,
    scan_id_start: int,
    profile_config: RainbowProfileConfig,
    vertical_spacing_factor: float,
    overlap_discard_threshold: float,
) -> tuple[list[SweepPose], int]:
    """Add one rotated scan column on each side of the detected vertical coverage block."""
    if not columns:
        return [], 0

    local_profile = make_rainbow_profile(profile_config)
    rotated_profile = transform_profile(local_profile, 0.0, 0.0, SIDE_GUARD_HEADING_RAD)
    main_min_x = min(column.center_x_m + float(np.min(local_profile[:, 0])) for column in columns)
    main_max_x = max(column.center_x_m + float(np.max(local_profile[:, 0])) for column in columns)
    rotated_min_x = float(np.min(rotated_profile[:, 0]))
    rotated_max_x = float(np.max(rotated_profile[:, 0]))
    left_x = main_min_x - rotated_max_x
    right_x = main_max_x - rotated_min_x

    y_min_hint = min(column.min_y_m for column in columns)
    y_max_hint = max(column.max_y_m for column in columns)
    touching_step = _touching_vertical_step(rotated_profile)
    target_step = touching_step * vertical_spacing_factor

    kept: list[SweepPose] = []
    discarded_count = 0
    side_specs = [
        ("left", left_x, -1001),
        ("right", right_x, -1002),
    ]
    for side_index, (side_name, x_m, column_id) in enumerate(side_specs):
        y_bounds = _profile_center_y_bounds_at_x(float(x_m), tank, rotated_profile, y_min_hint, y_max_hint)
        if y_bounds is None:
            continue
        y_min, y_max = y_bounds
        span = y_max - y_min
        interval_count = max(1, int(math.ceil(span / target_step)))
        y_positions = np.linspace(y_min, y_max, interval_count + 1)
        travel_direction = "down" if side_index % 2 == 1 else "up"
        if travel_direction == "down":
            y_positions = y_positions[::-1]

        for y_m in y_positions:
            world_profile = transform_profile(local_profile, float(x_m), float(y_m), SIDE_GUARD_HEADING_RAD)
            overlap_ratio = _estimate_union_overlap_ratio(world_profile, edge_polygon_records)
            if overlap_ratio > overlap_discard_threshold:
                discarded_count += 1
                continue
            kept.append(
                SweepPose(
                    scan_id=scan_id_start + len(kept),
                    stage="interior_side_guard",
                    x_m=float(x_m),
                    y_m=float(y_m),
                    heading_rad=SIDE_GUARD_HEADING_RAD,
                    heading_deg=math.degrees(SIDE_GUARD_HEADING_RAD),
                    column_id=column_id,
                    row_name=f"{side_name}_rotated_side_guard",
                    orientation="vertical",
                    group_id=len(columns) + side_index,
                    section_id=column_id,
                    travel_direction=travel_direction,
                )
            )

    return kept, discarded_count


def save_lawnmower_section_lines_json(
    lines: list[LawnmowerSectionLine],
    filepath: str | Path,
) -> Path:
    """Save the section-only geometry without mission poses or circular data."""
    path = Path(filepath)
    data = {
        "units": "meters",
        "section_count": len(lines),
        "sections": [asdict(line) for line in lines],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def plot_lawnmower_section_lines(
    model: GeometryModel,
    lines: list[LawnmowerSectionLine],
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot imported geometry plus straight interior section lines only."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 11))
    _plot_imported_geometry_background(ax, model)
    colors = {"vertical": "#16a34a", "horizontal": "#dc2626"}
    labels_drawn: set[str] = set()
    for line in lines:
        color = colors[line.orientation]
        label = f"Generated {line.orientation} sections" if line.orientation not in labels_drawn else "_nolegend_"
        labels_drawn.add(line.orientation)
        start = np.asarray(line.start_m, dtype=float)
        end = np.asarray(line.end_m, dtype=float)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=2.2,
            alpha=0.92,
            label=label,
            zorder=4,
        )

    if model.bounds is not None:
        span = max(model.bounds.width, model.bounds.height, 1e-9)
        margin = span * 0.04
        ax.set_xlim(model.bounds.min_x - margin, model.bounds.max_x + margin)
        ax.set_ylim(model.bounds.min_y - margin, model.bounds.max_y + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Interior lawnmower section lines only")
    ax.grid(True, alpha=0.18)
    if lines:
        ax.legend(loc="upper right")
    else:
        ax.legend(
            handles=[Line2D([], [], color="#6b7280", label="Imported geometry")],
            loc="upper right",
        )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig, ax


def plot_lawnmower_scan_placements(
    model: GeometryModel,
    lines: list[LawnmowerSectionLine],
    placements: list[LawnmowerScanPlacement],
    *,
    profile_config: RainbowProfileConfig | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot existing guide lines and their anchored interior rainbow footprints only."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    fig, ax = plt.subplots(figsize=(11, 11))
    _plot_imported_geometry_background(ax, model)

    guide_colors = {"vertical": "#16a34a", "horizontal": "#dc2626"}
    guide_labels_drawn: set[str] = set()
    for line in lines:
        start = np.asarray(line.start_m, dtype=float)
        end = np.asarray(line.end_m, dtype=float)
        label = (
            f"Existing {line.orientation} guide lines"
            if line.orientation not in guide_labels_drawn
            else "_nolegend_"
        )
        guide_labels_drawn.add(line.orientation)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=guide_colors[line.orientation],
            linewidth=1.8,
            alpha=0.9,
            label=label,
            zorder=5,
        )

    for placement_index, placement in enumerate(placements):
        polygon = lawnmower_scan_profile_polygon(placement, profile_config)
        closed = _closed_points(polygon)
        ax.fill(
            closed[:, 0],
            closed[:, 1],
            color="#7c3aed",
            alpha=0.055,
            linewidth=0.0,
            zorder=3,
        )
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#7c3aed",
            linewidth=0.45,
            alpha=0.65,
            label="Interior rainbow profiles" if placement_index == 0 else "_nolegend_",
            zorder=4,
        )

    if placements:
        anchors = np.asarray([placement.anchor_m for placement in placements], dtype=float)
        ax.scatter(
            anchors[:, 0],
            anchors[:, 1],
            color="#f59e0b",
            s=6,
            alpha=0.9,
            label="Long-side midpoint anchors",
            zorder=6,
        )

    if model.bounds is not None:
        span = max(model.bounds.width, model.bounds.height, 1e-9)
        margin = span * 0.04
        ax.set_xlim(model.bounds.min_x - margin, model.bounds.max_x + margin)
        ax.set_ylim(model.bounds.min_y - margin, model.bounds.max_y + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Interior rainbow profiles anchored to existing lawnmower guide lines")
    ax.grid(True, alpha=0.18)
    if lines or placements:
        ax.legend(loc="upper right")
    else:
        ax.legend(
            handles=[Line2D([], [], color="#6b7280", label="Imported geometry")],
            loc="upper right",
        )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig, ax


def save_mission_json(mission: OuterEdgeSweepMission, filepath: str | Path) -> Path:
    """Save the outer edge sweep mission as JSON."""
    path = Path(filepath)
    data = asdict(mission)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_mission_csv(mission: OuterEdgeSweepMission, filepath: str | Path) -> Path:
    """Save the ordered scan pose list as CSV."""
    path = Path(filepath)
    fieldnames = [
        "scan_id",
        "stage",
        "row_id",
        "row_name",
        "column_id",
        "x_m",
        "y_m",
        "heading_rad",
        "heading_deg",
        "theta_rad",
        "sweep_radius_m",
        "profile_type",
        "status",
        "orientation",
        "group_id",
        "section_id",
        "travel_direction",
        "profile_variant",
        "anchor_x_m",
        "anchor_y_m",
        "parent_full_x_m",
        "parent_full_y_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for pose in mission.poses:
            writer.writerow(asdict(pose))
    return path


def plot_outer_edge_sweep(
    model: GeometryModel,
    mission: OuterEdgeSweepMission,
    *,
    show: bool = True,
    save_path: str | Path | None = DEFAULT_PLOT_PATH,
    max_profile_draw_count: int | None = None,
    highlight_outer_lawnmower_sections: bool = False,
) -> Any:
    """Plot imported geometry, estimated tank circle, sweep circle, and scan placements."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(11, 11))
    _plot_imported_geometry_background(ax, model)

    cx = mission.tank_center_m["x"]
    cy = mission.tank_center_m["y"]
    ax.add_patch(
        Circle((cx, cy), mission.tank_radius_m, fill=False, color="#111827", linewidth=2.0, label="Estimated tank boundary")
    )
    circular_palette = ["#2563eb", "#059669", "#d97706", "#be185d", "#0891b2"]
    for row in mission.circular_rows:
        color = circular_palette[row.row_id % len(circular_palette)]
        ax.add_patch(
            Circle(
                (cx, cy),
                row.sweep_radius_m,
                fill=False,
                color=color,
                linestyle="--",
                linewidth=1.5,
                label="Accepted circular rows" if row.row_id == 0 else "_nolegend_",
            )
        )
    if mission.rejected_circular_radius_m is not None and mission.rejected_circular_radius_m > 0.0:
        ax.add_patch(
            Circle(
                (cx, cy),
                mission.rejected_circular_radius_m,
                fill=False,
                color="#6b7280",
                linestyle=":",
                linewidth=1.4,
                label=f"Circular stop: {mission.circular_stop_reason}",
            )
        )

    profile_config = RainbowProfileConfig(
        width=float(mission.scan_profile["width_m"]),
        arc_radius=float(mission.scan_profile["arc_radius_m"]),
        side_height=float(mission.scan_profile["side_height_m"]),
        arc_samples=int(mission.scan_profile["arc_samples"]),
    )
    local_profile = make_rainbow_profile(profile_config)
    if max_profile_draw_count is None:
        profile_stride = 1
    else:
        profile_stride = max(1, int(math.ceil(len(mission.poses) / max(1, max_profile_draw_count))))
    for pose in mission.poses[::profile_stride]:
        world_profile = scan_pose_profile_polygon(pose, profile_config)
        closed_profile = _closed_points(world_profile)
        if pose.stage == "circular_edge":
            color = circular_palette[int(pose.row_id or 0) % len(circular_palette)]
            style = {"color": color, "fill": color}
        elif pose.stage == "interior_horizontal":
            style = {"color": "#dc2626", "fill": "#fca5a5"}
        elif pose.stage == "interior_side_guard":
            style = {"color": "#0f766e", "fill": "#5eead4"}
        else:
            style = {"color": "#7c3aed", "fill": "#a78bfa"}
        is_half = pose.profile_variant != "full"
        if is_half:
            style = {"color": "#0891b2", "fill": "#67e8f9"}
        ax.fill(
            closed_profile[:, 0],
            closed_profile[:, 1],
            color=style["fill"],
            alpha=0.12 if is_half else 0.07,
            zorder=2,
        )
    if highlight_outer_lawnmower_sections:
        outer_guides = [
            line
            for line in mission.lawnmower_section_lines
            if line.is_outer_extension
        ]
        for index, line in enumerate(outer_guides):
            ax.plot(
                [line.start_m[0], line.end_m[0]],
                [line.start_m[1], line.end_m[1]],
                color="#f59e0b",
                linewidth=3.0,
                linestyle=":",
                label="New outer lawnmower guides" if index == 0 else "_nolegend_",
                zorder=8,
            )
        ax.plot(
            closed_profile[:, 0],
            closed_profile[:, 1],
            color=style["color"],
            linewidth=1.3 if is_half else 0.7,
            linestyle="--" if is_half else "-",
            alpha=0.9 if is_half else 0.38,
            zorder=3,
        )

    circular_poses = [pose for pose in mission.poses if pose.stage == "circular_edge"]
    interior_poses = [pose for pose in mission.poses if pose.stage == "interior_vertical"]
    horizontal_poses = [pose for pose in mission.poses if pose.stage == "interior_horizontal"]
    side_guard_poses = [pose for pose in mission.poses if pose.stage == "interior_side_guard"]
    for row in mission.circular_rows:
        row_poses = [pose for pose in circular_poses if pose.row_id == row.row_id]
        color = circular_palette[row.row_id % len(circular_palette)]
        ax.scatter(
            [pose.x_m for pose in row_poses],
            [pose.y_m for pose in row_poses],
            s=14,
            color=color,
            label="_nolegend_",
            zorder=5,
        )
        _plot_direction_arrows(ax, row_poses, label=None)
    if interior_poses:
        ax.scatter(
            [pose.x_m for pose in interior_poses],
            [pose.y_m for pose in interior_poses],
            s=17,
            color="#7c3aed",
            marker="s",
            label="Interior vertical centers",
            zorder=5,
        )
        _plot_group_paths(ax, interior_poses, "#7c3aed", "Vertical travel path")
    if horizontal_poses:
        ax.scatter(
            [pose.x_m for pose in horizontal_poses],
            [pose.y_m for pose in horizontal_poses],
            s=18,
            color="#dc2626",
            marker="^",
            label="Interior horizontal centers",
            zorder=5,
        )
        _plot_group_paths(ax, horizontal_poses, "#dc2626", "Horizontal travel path")
    if side_guard_poses:
        ax.scatter(
            [pose.x_m for pose in side_guard_poses],
            [pose.y_m for pose in side_guard_poses],
            s=18,
            color="#0f766e",
            marker="D",
            label="_nolegend_",
            zorder=5,
        )
    half_poses = [pose for pose in mission.poses if pose.profile_variant != "full"]
    if half_poses:
        ax.scatter(
            [pose.anchor_x_m for pose in half_poses],
            [pose.anchor_y_m for pose in half_poses],
            s=28,
            color="#0e7490",
            marker="x",
            label="Salvaged half-profile anchors",
            zorder=7,
        )
    path_x = [pose.x_m for pose in mission.poses]
    path_y = [pose.y_m for pose in mission.poses]
    ax.plot(path_x, path_y, color="#f97316", linewidth=0.9, alpha=0.42, label="Ordered mission path", zorder=4)
    ax.scatter([cx], [cy], s=35, color="#111827", label="_nolegend_", zorder=6)
    _apply_plot_bounds(ax, model, mission)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.22)
    geometry_legend = ax.legend(loc="upper left", fontsize=8)
    ax.add_artist(geometry_legend)
    coverage = _estimate_tank_coverage(mission, profile_config)
    if mission.selected_circular_spacing_factor is None:
        ax.set_title(f"Mission preview - coverage {coverage.covered_percent:.1f}%")
    else:
        ax.set_title(
            "Mission preview - "
            f"coverage {coverage.covered_percent:.1f}%, "
            f"circular spacing {mission.selected_circular_spacing_factor:.2f}"
        )
    coverage_handles = [
        Line2D([], [], color="none", label=f"Scan profiles: {coverage.total_profiles}"),
        Line2D([], [], color="none", label=f"Tank covered: {coverage.covered_percent:.1f}%"),
        Line2D([], [], color="none", label=f"Covered 2+ times: {coverage.multi_covered_percent:.1f}%"),
    ]
    ax.legend(
        handles=coverage_handles,
        title="Coverage estimate",
        loc="upper right",
        handlelength=0,
        handletextpad=0,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig, ax


def print_mission_summary(
    mission: OuterEdgeSweepMission,
    *,
    json_path: Path | None,
    csv_path: Path | None,
    plot_path: Path | None,
) -> None:
    """Print a compact run summary."""
    print("Mission plan summary")
    print(f"Tank center: ({mission.tank_center_m['x']:.6g}, {mission.tank_center_m['y']:.6g}) m")
    print(f"Estimated tank radius: {mission.tank_radius_m:.6g} m")
    print(f"Row spacing: {mission.row_spacing_m:.6g} m")
    print(f"Flat-side wall clearance: {mission.outer_edge_offset_m:.6g} m")
    print(f"Circular stop backoff rows: {mission.circular_stop_backoff_rows}")
    print(f"Accepted circular rows: {len(mission.circular_rows)}")
    for row in mission.circular_rows:
        print(
            f"  Row {row.row_id + 1}: radius={row.sweep_radius_m:.6g} m, scans={row.scan_count}, "
            f"angular_spacing={math.degrees(row.angular_spacing_rad):.3f} deg, "
            f"arc_spacing={row.arc_spacing_m:.4f} m, adjustments={row.spacing_adjustments}"
        )
        print(
            f"    max_neighbor_gap={row.max_neighbor_gap_distance_m:.6g} m, "
            f"neighbor_contact={'pass' if row.minimum_neighbor_contact_passed else 'fail'}, "
            f"wraparound={'pass' if row.wraparound_passed else 'fail'}, "
            f"continuous_coverage={'verified' if row.continuous_coverage_verified else 'failed'}"
        )
        print(
            f"    neighbor_overlap: min={100.0 * row.minimum_neighbor_overlap_fraction:.2f}%, "
            f"avg={100.0 * row.average_neighbor_overlap_fraction:.2f}%, "
            f"max={100.0 * row.maximum_neighbor_overlap_fraction:.2f}%, "
            f"wraparound={100.0 * row.wraparound_overlap_fraction:.2f}%, "
            f"no_gaps={'verified' if row.no_neighbor_gaps_verified else 'failed'}"
        )
    if mission.rejected_circular_radius_m is not None:
        print(f"Circular stop radius: {mission.rejected_circular_radius_m:.6g} m")
    print(f"Circular wrapping stop reason: {mission.circular_stop_reason}")
    print(f"Max circular gap fraction: {mission.max_circular_gap_fraction:.6g}")
    print(f"Total circular scans: {mission.edge_sweep_scan_count}")
    vertical_guides = sum(line.orientation == "vertical" for line in mission.lawnmower_section_lines)
    horizontal_guides = sum(line.orientation == "horizontal" for line in mission.lawnmower_section_lines)
    print(f"Lawnmower vertical guide lines: {vertical_guides}")
    print(f"Lawnmower horizontal guide lines: {horizontal_guides}")
    side_guard_count = sum(1 for pose in mission.poses if pose.stage == "interior_side_guard")
    print(f"Vertical groups: {mission.vertical_group_count}")
    print(f"Horizontal groups: {mission.horizontal_group_count}")
    print(f"Interior vertical scans kept: {mission.vertical_scan_count}")
    print(f"Interior horizontal scans kept: {mission.horizontal_scan_count}")
    print(f"Rejected interior full-profile candidates retained: {mission.rejected_full_profile_candidate_count}")
    print(f"Salvaged left-half profiles: {sum(pose.profile_variant == 'left_half' for pose in mission.poses)}")
    print(f"Salvaged right-half profiles: {sum(pose.profile_variant == 'right_half' for pose in mission.poses)}")
    print(f"Rotated side-guard scans kept: {side_guard_count}")
    print(f"Interior scans discarded by circular-overlap filter: {mission.interior_discarded_count}")
    print(f"Duplicate interior poses removed: {mission.duplicate_poses_removed}")
    print(f"Invalid horizontal segments/sections skipped: {mission.invalid_horizontal_segments_skipped}")
    print(f"Total mission scans: {len(mission.poses)}")
    if mission.circular_spacing_candidates:
        print("Circular spacing comparison")
        print("rows | circular_spacing | coverage_% | scans | travel_m")
        for candidate in mission.circular_spacing_candidates:
            selected_marker = "  <-- selected" if candidate.selected else ""
            spacing_text = "n/a" if candidate.spacing_factor is None else f"{candidate.spacing_factor:.2f}"
            print(
                f"{candidate.circular_rows:<4d} | "
                f"{spacing_text:<16} | "
                f"{candidate.coverage_percent:<10.1f} | "
                f"{candidate.scan_count:<5d} | "
                f"{candidate.travel_distance_m:<8.1f}"
                f"{selected_marker}"
            )
        if mission.selected_circular_spacing_factor is not None:
            print(f"Selected circular spacing factor: {mission.selected_circular_spacing_factor:.2f}")
        if mission.circular_spacing_selection_reason:
            print(f"Reason: {mission.circular_spacing_selection_reason}")
    print(f"Direction: {mission.direction}")
    if json_path is not None:
        print(f"JSON output: {json_path}")
    if csv_path is not None:
        print(f"CSV output: {csv_path}")
    if plot_path is not None:
        print(f"Plot output: {plot_path}")


def _validate_sweep_inputs(
    tank_radius: float,
    outer_edge_offset_m: float,
    row_spacing_m: float,
    spacing_factor: float,
    max_circular_gap_fraction: float,
    circular_stop_backoff_rows: int,
) -> None:
    if tank_radius <= 0.0:
        raise ValueError("Tank radius must be positive.")
    if outer_edge_offset_m < 0.0:
        raise ValueError("Outer edge offset must be non-negative.")
    if row_spacing_m <= 0.0:
        raise ValueError("Row spacing must be positive.")
    if spacing_factor <= 0.0:
        raise ValueError("Spacing factor must be positive.")
    if not 0.0 <= max_circular_gap_fraction <= 1.0:
        raise ValueError("Maximum circular gap fraction must be between 0 and 1.")
    if circular_stop_backoff_rows < 0:
        raise ValueError("Circular stop backoff rows must be zero or greater.")


def _profile_constrained_sweep_radius(
    tank_radius: float,
    profile_config: RainbowProfileConfig,
    edge_offset_from_wall: float,
) -> float:
    local_profile = make_rainbow_profile(profile_config)
    max_tangent_offset = float(np.max(np.abs(local_profile[:, 1])))
    if max_tangent_offset >= tank_radius:
        raise ValueError("Tank radius is too small for the scan profile tangent extent.")

    allowed_center_radii: list[float] = []
    for radial_offset, tangent_offset in local_profile:
        tangent_limit = tank_radius**2 - float(tangent_offset) ** 2
        if tangent_limit < 0.0:
            raise ValueError("Tank radius is too small for the scan profile tangent extent.")
        allowed_center_radii.append(math.sqrt(tangent_limit) - float(radial_offset))
    return min(allowed_center_radii) - edge_offset_from_wall


def _profile_tangential_span(profile_config: RainbowProfileConfig) -> float:
    local_profile = make_rainbow_profile(profile_config)
    return float(np.max(local_profile[:, 1]) - np.min(local_profile[:, 1]))


def _touching_row_spacing(outer_sweep_radius: float, profile_config: RainbowProfileConfig) -> float:
    """Return the radial center spacing where two profile envelopes just meet."""
    local_profile = make_rainbow_profile(profile_config)
    outer_inner_radius = float(
        np.min(np.hypot(outer_sweep_radius + local_profile[:, 0], local_profile[:, 1]))
    )

    def inner_outer_radius(inner_sweep_radius: float) -> float:
        return float(np.max(np.hypot(inner_sweep_radius + local_profile[:, 0], local_profile[:, 1])))

    if inner_outer_radius(0.0) > outer_inner_radius:
        raise ValueError("Tank is too small to place two non-overlapping rainbow scan rows.")

    low = 0.0
    high = outer_sweep_radius
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if inner_outer_radius(midpoint) <= outer_inner_radius:
            low = midpoint
        else:
            high = midpoint
    return outer_sweep_radius - low


def _touching_angular_step(sweep_radius: float, profile_config: RainbowProfileConfig) -> float:
    """Find the angular separation where neighboring profile polygons just meet."""
    cache_key = (
        round(float(sweep_radius), 6),
        round(float(profile_config.width), 6),
        round(float(profile_config.arc_radius), 6),
        round(float(profile_config.side_height), 6),
        int(profile_config.arc_samples),
    )
    cached = _TOUCHING_ANGULAR_STEP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    local_profile = make_rainbow_profile(profile_config)
    first_profile = transform_profile(local_profile, sweep_radius, 0.0, 0.0)

    low = 0.0
    high = min(math.pi, max(0.05, 2.0 * _profile_tangential_span(profile_config) / sweep_radius))
    while high < math.pi:
        second_profile = transform_profile(
            local_profile,
            sweep_radius * math.cos(high),
            sweep_radius * math.sin(high),
            high,
        )
        if not _polygons_overlap_or_touch(first_profile, second_profile):
            break
        high = min(math.pi, high * 1.5)

    second_profile = transform_profile(
        local_profile,
        sweep_radius * math.cos(high),
        sweep_radius * math.sin(high),
        high,
    )
    if _polygons_overlap_or_touch(first_profile, second_profile):
        raise ValueError("Could not find non-overlapping angular spacing for the scan profile.")

    for _ in range(60):
        midpoint = (low + high) / 2.0
        second_profile = transform_profile(
            local_profile,
            sweep_radius * math.cos(midpoint),
            sweep_radius * math.sin(midpoint),
            midpoint,
        )
        if _polygons_overlap_or_touch(first_profile, second_profile):
            low = midpoint
        else:
            high = midpoint
    touching_step = (low + high) / 2.0
    _TOUCHING_ANGULAR_STEP_CACHE[cache_key] = touching_step
    return touching_step


def _touching_vertical_step(local_profile: np.ndarray) -> float:
    """Find the vertical center spacing where stacked profile polygons just meet."""
    low = 0.0
    high = float(np.max(local_profile[:, 1]) - np.min(local_profile[:, 1])) * 1.5
    while _polygons_overlap_or_touch(local_profile, local_profile + np.array([0.0, high])):
        high *= 1.5

    for _ in range(60):
        midpoint = (low + high) / 2.0
        shifted = local_profile + np.array([0.0, midpoint])
        if _polygons_overlap_or_touch(local_profile, shifted):
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _polygons_overlap_or_touch(first: np.ndarray, second: np.ndarray, tolerance: float = 1e-10) -> bool:
    first_polygon = _without_duplicate_closure(first)
    second_polygon = _without_duplicate_closure(second)

    for first_index in range(len(first_polygon)):
        first_start = first_polygon[first_index]
        first_end = first_polygon[(first_index + 1) % len(first_polygon)]
        for second_index in range(len(second_polygon)):
            second_start = second_polygon[second_index]
            second_end = second_polygon[(second_index + 1) % len(second_polygon)]
            if (
                max(first_start[0], first_end[0]) < min(second_start[0], second_end[0]) - tolerance
                or min(first_start[0], first_end[0]) > max(second_start[0], second_end[0]) + tolerance
                or max(first_start[1], first_end[1]) < min(second_start[1], second_end[1]) - tolerance
                or min(first_start[1], first_end[1]) > max(second_start[1], second_end[1]) + tolerance
            ):
                continue
            if _segments_intersect(first_start, first_end, second_start, second_end, tolerance):
                return True

    return _point_in_polygon(first_polygon[0], second_polygon) or _point_in_polygon(second_polygon[0], first_polygon)


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return zero for intersecting polygons, otherwise their nearest boundary distance."""
    if _polygons_overlap_or_touch(first, second, tolerance=CIRCULAR_GAP_TOLERANCE_M):
        return 0.0

    first_polygon = _without_duplicate_closure(first)
    second_polygon = _without_duplicate_closure(second)
    minimum_distance = math.inf
    for point in first_polygon:
        minimum_distance = min(minimum_distance, _point_to_polygon_boundary_distance(point, second_polygon))
    for point in second_polygon:
        minimum_distance = min(minimum_distance, _point_to_polygon_boundary_distance(point, first_polygon))
    return float(minimum_distance)


def _point_to_polygon_boundary_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    projections = np.zeros(len(segments), dtype=float)
    usable = lengths_squared > 1e-20
    projections[usable] = np.sum((point - starts[usable]) * segments[usable], axis=1) / lengths_squared[usable]
    projections = np.clip(projections, 0.0, 1.0)
    closest = starts + projections[:, None] * segments
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


def _points_in_or_near_polygon(points: np.ndarray, polygon: np.ndarray, tolerance: float) -> np.ndarray:
    """Include points inside a polygon or within tolerance of its boundary."""
    covered = _points_in_polygon(points, polygon)
    if np.all(covered):
        return covered

    polygon = _without_duplicate_closure(polygon)
    uncovered_indices = np.flatnonzero(~covered)
    uncovered_points = points[uncovered_indices]
    minimum_distances = np.full(len(uncovered_points), math.inf, dtype=float)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-20:
            distances = np.linalg.norm(uncovered_points - start, axis=1)
        else:
            projections = np.sum((uncovered_points - start) * segment, axis=1) / length_squared
            projections = np.clip(projections, 0.0, 1.0)
            closest = start + projections[:, None] * segment
            distances = np.linalg.norm(uncovered_points - closest, axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    covered[uncovered_indices] = minimum_distances <= tolerance
    return covered


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = _cross_2d(first_direction, second_direction)
    offset = second_start - first_start

    if abs(denominator) <= tolerance:
        if abs(_cross_2d(offset, first_direction)) > tolerance:
            return False
        first_length_squared = float(np.dot(first_direction, first_direction))
        if first_length_squared <= tolerance:
            return float(np.linalg.norm(first_start - second_start)) <= tolerance
        start_projection = float(np.dot(second_start - first_start, first_direction) / first_length_squared)
        end_projection = float(np.dot(second_end - first_start, first_direction) / first_length_squared)
        return max(start_projection, end_projection) >= -tolerance and min(start_projection, end_projection) <= 1.0 + tolerance

    first_parameter = _cross_2d(offset, second_direction) / denominator
    second_parameter = _cross_2d(offset, first_direction) / denominator
    return (
        -tolerance <= first_parameter <= 1.0 + tolerance
        and -tolerance <= second_parameter <= 1.0 + tolerance
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    previous_index = len(polygon) - 1
    for current_index in range(len(polygon)):
        current = polygon[current_index]
        previous = polygon[previous_index]
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing_x = (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if point[0] < crossing_x:
                inside = not inside
        previous_index = current_index
    return inside


def _without_duplicate_closure(points: np.ndarray) -> np.ndarray:
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        return points[:-1]
    return points


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_area(polygon: np.ndarray) -> float:
    polygon = _without_duplicate_closure(polygon)
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    return abs(float(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))) / 2.0


def _iter_geometry_segments(model: GeometryModel):
    for line in model.line_segments:
        yield np.array([line.start.x, line.start.y]), np.array([line.end.x, line.end.y])
    for polyline in model.polylines:
        points = [np.array([vertex.point.x, vertex.point.y]) for vertex in polyline.vertices]
        for start, end in zip(points, points[1:]):
            yield start, end
        if polyline.closed and len(points) > 2:
            yield points[-1], points[0]


def _group_vertical_segments(segments) -> list[tuple[float, list[tuple[float, float]]]]:
    return [
        (cross_m, intervals)
        for cross_m, intervals, _source_count in _group_axis_segments(segments, "vertical")
    ]


def _group_axis_segments(
    segments,
    orientation: str,
) -> list[tuple[float, list[tuple[float, float]], int]]:
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("Axis grouping orientation must be 'vertical' or 'horizontal'.")
    classified: list[tuple[float, float, float, float]] = []
    for start, end in segments:
        delta = end - start
        length = float(np.linalg.norm(delta))
        off_axis_delta = float(delta[0] if orientation == "vertical" else delta[1])
        if length <= 1e-9 or abs(off_axis_delta) > VERTICAL_SLOPE_TOLERANCE * length:
            continue
        cross_start = float(start[0] if orientation == "vertical" else start[1])
        cross_end = float(end[0] if orientation == "vertical" else end[1])
        progress_start = float(start[1] if orientation == "vertical" else start[0])
        progress_end = float(end[1] if orientation == "vertical" else end[0])
        classified.append(
            (
                (cross_start + cross_end) / 2.0,
                min(progress_start, progress_end),
                max(progress_start, progress_end),
                length,
            )
        )
    classified.sort(key=lambda item: item[0])

    groups: list[list[tuple[float, float, float, float]]] = []
    cross_tolerance = (
        VERTICAL_X_GROUP_TOLERANCE_M
        if orientation == "vertical"
        else HORIZONTAL_Y_GROUP_TOLERANCE_M
    )
    for segment in classified:
        if not groups:
            groups.append([segment])
            continue
        group_x = sum(item[0] * item[3] for item in groups[-1]) / sum(item[3] for item in groups[-1])
        if abs(segment[0] - group_x) <= cross_tolerance:
            groups[-1].append(segment)
        else:
            groups.append([segment])

    grouped: list[tuple[float, list[tuple[float, float]], int]] = []
    for group in groups:
        total_length = sum(item[3] for item in group)
        cross_m = sum(item[0] * item[3] for item in group) / total_length
        grouped.append((cross_m, [(item[1], item[2]) for item in group], len(group)))
    return grouped


def _merge_intervals(intervals: list[tuple[float, float]], tolerance: float = 1e-6) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interior_profile_center_y_bounds(
    column: VerticalPlateColumn,
    tank: TankCircleEstimate,
    local_profile: np.ndarray,
) -> tuple[float, float] | None:
    return _profile_center_y_bounds_at_x(
        column.center_x_m,
        tank,
        local_profile,
        column.min_y_m,
        column.max_y_m,
    )


def _profile_center_y_bounds_at_x(
    center_x_m: float,
    tank: TankCircleEstimate,
    profile_points: np.ndarray,
    min_y_hint: float,
    max_y_hint: float,
) -> tuple[float, float] | None:
    min_local_y = float(np.min(profile_points[:, 1]))
    max_local_y = float(np.max(profile_points[:, 1]))
    lower = min_y_hint - min_local_y
    upper = max_y_hint - max_local_y

    for local_x, local_y in profile_points:
        radial_x = center_x_m + float(local_x) - tank.center_x
        if abs(radial_x) >= tank.radius:
            return None
        vertical_limit = math.sqrt(max(0.0, tank.radius**2 - radial_x**2))
        lower = max(lower, tank.center_y - vertical_limit - float(local_y))
        upper = min(upper, tank.center_y + vertical_limit - float(local_y))
    if upper < lower:
        return None
    return lower, upper


def _profile_center_x_bounds_at_y(
    center_y_m: float,
    tank: TankCircleEstimate,
    profile_points: np.ndarray,
    min_x_hint: float,
    max_x_hint: float,
) -> tuple[float, float] | None:
    """Transpose the vertical bounds check for a profile progressing horizontally."""
    min_local_x = float(np.min(profile_points[:, 0]))
    max_local_x = float(np.max(profile_points[:, 0]))
    lower = min_x_hint - min_local_x
    upper = max_x_hint - max_local_x

    for local_x, local_y in profile_points:
        radial_y = center_y_m + float(local_y) - tank.center_y
        if abs(radial_y) >= tank.radius:
            return None
        horizontal_limit = math.sqrt(max(0.0, tank.radius**2 - radial_y**2))
        lower = max(lower, tank.center_x - horizontal_limit - float(local_x))
        upper = min(upper, tank.center_x + horizontal_limit - float(local_x))
    if upper < lower:
        return None
    return lower, upper


def _polygon_bounds(polygon: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(polygon[:, 0])),
        float(np.min(polygon[:, 1])),
        float(np.max(polygon[:, 0])),
        float(np.max(polygon[:, 1])),
    )


def _estimate_union_overlap_ratio(
    candidate_polygon: np.ndarray,
    comparison_polygons: list[tuple[np.ndarray, tuple[float, float, float, float]]],
) -> float:
    min_x, min_y, max_x, max_y = _polygon_bounds(candidate_polygon)
    x_step = (max_x - min_x) / OVERLAP_SAMPLE_GRID_SIZE
    y_step = (max_y - min_y) / OVERLAP_SAMPLE_GRID_SIZE
    if x_step <= 0.0 or y_step <= 0.0:
        return 0.0

    xs = min_x + (np.arange(OVERLAP_SAMPLE_GRID_SIZE) + 0.5) * x_step
    ys = min_y + (np.arange(OVERLAP_SAMPLE_GRID_SIZE) + 0.5) * y_step
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidate_mask = _points_in_polygon(points, candidate_polygon)
    candidate_points = points[candidate_mask]
    if len(candidate_points) == 0:
        return 0.0

    overlap_mask = np.zeros(len(candidate_points), dtype=bool)
    for polygon, bounds in comparison_polygons:
        other_min_x, other_min_y, other_max_x, other_max_y = bounds
        if other_max_x < min_x or other_min_x > max_x or other_max_y < min_y or other_min_y > max_y:
            continue
        relevant = (
            (candidate_points[:, 0] >= other_min_x)
            & (candidate_points[:, 0] <= other_max_x)
            & (candidate_points[:, 1] >= other_min_y)
            & (candidate_points[:, 1] <= other_max_y)
            & ~overlap_mask
        )
        if not np.any(relevant):
            continue
        relevant_indices = np.flatnonzero(relevant)
        overlap_mask[relevant_indices] = _points_in_polygon(candidate_points[relevant], polygon)
        if np.all(overlap_mask):
            break
    return float(np.count_nonzero(overlap_mask) / len(candidate_points))


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    polygon = _without_duplicate_closure(polygon)
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        crosses = (current[1] > points[:, 1]) != (previous[1] > points[:, 1])
        denominator = previous[1] - current[1]
        if abs(float(denominator)) > 1e-15:
            crossing_x = (
                (previous[0] - current[0])
                * (points[:, 1] - current[1])
                / denominator
                + current[0]
            )
            inside ^= crosses & (points[:, 0] < crossing_x)
        previous = current
    return inside


def _estimate_tank_coverage(
    mission: OuterEdgeSweepMission,
    profile_config: RainbowProfileConfig,
    *,
    grid_resolution: int = 220,
) -> CoverageEstimate:
    """Estimate one-pass and multi-pass coverage over the circular tank area."""
    center_x = mission.tank_center_m["x"]
    center_y = mission.tank_center_m["y"]
    radius = mission.tank_radius_m
    cell_size = 2.0 * radius / grid_resolution
    xs = center_x - radius + (np.arange(grid_resolution) + 0.5) * cell_size
    ys = center_y - radius + (np.arange(grid_resolution) + 0.5) * cell_size
    grid_x, grid_y = np.meshgrid(xs, ys)
    tank_mask = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2 <= radius**2
    coverage_count = np.zeros((grid_resolution, grid_resolution), dtype=np.uint16)
    for pose in mission.poses:
        polygon = scan_pose_profile_polygon(pose, profile_config)
        min_x, min_y, max_x, max_y = _polygon_bounds(polygon)
        x_start = max(0, int(math.floor((min_x - (center_x - radius)) / cell_size)))
        x_end = min(grid_resolution, int(math.ceil((max_x - (center_x - radius)) / cell_size)))
        y_start = max(0, int(math.floor((min_y - (center_y - radius)) / cell_size)))
        y_end = min(grid_resolution, int(math.ceil((max_y - (center_y - radius)) / cell_size)))
        if x_start >= x_end or y_start >= y_end:
            continue

        candidate_mask = tank_mask[y_start:y_end, x_start:x_end]
        if not np.any(candidate_mask):
            continue
        points = np.column_stack(
            (
                grid_x[y_start:y_end, x_start:x_end][candidate_mask],
                grid_y[y_start:y_end, x_start:x_end][candidate_mask],
            )
        )
        inside_profile = _points_in_polygon(points, polygon)
        if not np.any(inside_profile):
            continue
        local_counts = coverage_count[y_start:y_end, x_start:x_end]
        flat_indices = np.flatnonzero(candidate_mask)
        local_counts.flat[flat_indices[inside_profile]] += 1

    tank_sample_count = int(np.count_nonzero(tank_mask))
    if tank_sample_count == 0:
        return CoverageEstimate(total_profiles=len(mission.poses), covered_percent=0.0, multi_covered_percent=0.0)
    covered_percent = 100.0 * np.count_nonzero(tank_mask & (coverage_count >= 1)) / tank_sample_count
    multi_covered_percent = 100.0 * np.count_nonzero(tank_mask & (coverage_count >= 2)) / tank_sample_count
    return CoverageEstimate(
        total_profiles=len(mission.poses),
        covered_percent=float(covered_percent),
        multi_covered_percent=float(multi_covered_percent),
    )


def _warn_for_spacing_factor(spacing_factor: float) -> None:
    if not math.isclose(spacing_factor, DEFAULT_SPACING_FACTOR):
        print(
            "Warning: --spacing-factor is retained for compatibility but circular profile density "
            "is now selected from measured neighboring-polygon overlap."
        )


def _warn_for_row_spacing(row_spacing_m: float, profile_config: RainbowProfileConfig) -> None:
    if row_spacing_m < profile_config.width * 0.9:
        print("Warning: row spacing is smaller than the profile width; expect overlap between rows.")
    if row_spacing_m > profile_config.width:
        print("Warning: row spacing exceeds the profile width; an uncovered radial band may remain.")


def _warn_for_vertical_spacing_factor(vertical_spacing_factor: float) -> None:
    if vertical_spacing_factor < 0.5 or vertical_spacing_factor > 1.1:
        print(
            "Warning: vertical spacing factor is outside the typical v1 range of 0.5 to 1.1; "
            "expect heavy overlap or vertical gaps."
        )


def _plot_imported_geometry_background(ax: Any, model: GeometryModel) -> None:
    from matplotlib.patches import Circle

    for line in model.line_segments:
        ax.plot([line.start.x, line.end.x], [line.start.y, line.end.y], color="#6b7280", linewidth=0.8, alpha=0.28)
    for polyline in model.polylines:
        if not polyline.vertices:
            continue
        xs = [vertex.point.x for vertex in polyline.vertices]
        ys = [vertex.point.y for vertex in polyline.vertices]
        if polyline.closed and len(polyline.vertices) > 1:
            xs.append(polyline.vertices[0].point.x)
            ys.append(polyline.vertices[0].point.y)
        ax.plot(xs, ys, color="#6b7280", linewidth=0.8, alpha=0.28)
    for arc in model.arcs:
        points = _sample_arc_points(arc.center.x, arc.center.y, arc.radius, arc.start_angle, arc.end_angle)
        ax.plot(points[:, 0], points[:, 1], color="#6b7280", linewidth=0.8, alpha=0.28)
    for circle in model.circles:
        ax.add_patch(
            Circle((circle.center.x, circle.center.y), circle.radius, fill=False, color="#6b7280", linewidth=0.8, alpha=0.28)
        )


def _sample_arc_points(
    center_x: float,
    center_y: float,
    radius: float,
    start_angle_deg: float,
    end_angle_deg: float,
    samples: int = 72,
) -> np.ndarray:
    sweep = (end_angle_deg - start_angle_deg) % 360.0
    if math.isclose(sweep, 0.0):
        sweep = 360.0
    angles = np.radians(np.linspace(start_angle_deg, start_angle_deg + sweep, samples))
    return np.column_stack((center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)))


def _plot_group_paths(
    ax: Any,
    poses: list[SweepPose],
    color: str,
    label: str,
) -> None:
    group_ids = []
    for pose in poses:
        if pose.group_id is not None and pose.group_id not in group_ids:
            group_ids.append(pose.group_id)
    for index, group_id in enumerate(group_ids):
        group = [pose for pose in poses if pose.group_id == group_id]
        if len(group) < 2:
            continue
        ax.plot(
            [pose.x_m for pose in group],
            [pose.y_m for pose in group],
            color=color,
            linewidth=1.2,
            alpha=0.8,
            label=label if index == 0 else "_nolegend_",
            zorder=4,
        )


def _plot_direction_arrows(ax: Any, poses: list[SweepPose], *, label: str | None) -> None:
    if len(poses) < 2:
        return
    stride = max(1, len(poses) // 24)
    indices = list(range(0, len(poses), stride))
    x_values: list[float] = []
    y_values: list[float] = []
    u_values: list[float] = []
    v_values: list[float] = []
    for idx in indices:
        current_pose = poses[idx]
        next_pose = poses[(idx + 1) % len(poses)]
        x_values.append(current_pose.x_m)
        y_values.append(current_pose.y_m)
        u_values.append(next_pose.x_m - current_pose.x_m)
        v_values.append(next_pose.y_m - current_pose.y_m)
    ax.quiver(
        x_values,
        y_values,
        u_values,
        v_values,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color="#f97316",
        label=label,
        zorder=7,
    )


def _apply_plot_bounds(ax: Any, model: GeometryModel, mission: OuterEdgeSweepMission) -> None:
    cx = mission.tank_center_m["x"]
    cy = mission.tank_center_m["y"]
    radius = mission.tank_radius_m + float(mission.scan_profile["arc_radius_m"])
    if model.bounds is not None:
        min_x = min(model.bounds.min_x, cx - radius)
        max_x = max(model.bounds.max_x, cx + radius)
        min_y = min(model.bounds.min_y, cy - radius)
        max_y = max(model.bounds.max_y, cy + radius)
    else:
        min_x, max_x = cx - radius, cx + radius
        min_y, max_y = cy - radius, cy + radius
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    margin = max(width, height) * 0.04
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)


def _closed_points(points: np.ndarray) -> np.ndarray:
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack((points, points[0]))


def _normalize_angle(angle_rad: float) -> float:
    return float(angle_rad % (2.0 * math.pi))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan edge sweeps and axis-aligned interior coverage from a tank DXF."
    )
    parser.add_argument("dxf_path", help="Path to the tank floor DXF file.")
    parser.add_argument(
        "--outer-edge-offset",
        "--edge-offset",
        dest="outer_edge_offset",
        type=float,
        default=DEFAULT_EDGE_OFFSET_FROM_WALL,
        help=(
            "Inward clearance from the estimated tank wall to the outward flat side of each scan profile. "
            f"Default: {DEFAULT_EDGE_OFFSET_FROM_WALL} m."
        ),
    )
    parser.add_argument(
        "--row-spacing",
        type=float,
        default=DEFAULT_ROW_SPACING,
        help=(
            "Radial spacing between accepted circular rows. "
            f"Default: {DEFAULT_ROW_SPACING} m."
        ),
    )
    parser.add_argument(
        "--spacing-factor",
        type=float,
        default=DEFAULT_SPACING_FACTOR,
        help=(
            "Legacy compatibility option. Circular profile density is now selected from measured "
            f"neighbor overlap. Default: {DEFAULT_SPACING_FACTOR}."
        ),
    )
    parser.add_argument(
        "--max-circular-gap-fraction",
        type=float,
        default=DEFAULT_MAX_CIRCULAR_GAP_FRACTION,
        help=(
            "Reject a circular row when its estimated neighbor gap area exceeds this fraction of one profile area. "
            f"Default: {DEFAULT_MAX_CIRCULAR_GAP_FRACTION}."
        ),
    )
    parser.add_argument(
        "--circular-stop-backoff-rows",
        type=int,
        default=DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS,
        help=(
            "After circular wrapping reaches its normal stop condition, remove this many of the final "
            "accepted circular rows so the planner stops before tight-radius rows with visible gaps. "
            f"Default: {DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS}."
        ),
    )
    parser.add_argument(
        "--vertical-spacing-factor",
        type=float,
        default=DEFAULT_VERTICAL_SPACING_FACTOR,
        help=(
            "Multiplier on geometry-derived vertical contact spacing. Values below 1 make neighboring "
            "interior profiles overlap slightly; values above 1 leave gaps. "
            f"Default: {DEFAULT_VERTICAL_SPACING_FACTOR}."
        ),
    )
    parser.add_argument(
        "--overlap-discard-threshold",
        type=float,
        default=DEFAULT_OVERLAP_DISCARD_THRESHOLD,
        help=(
            "Discard an interior scan when its estimated edge-sweep overlap exceeds this fraction. "
            f"Default: {DEFAULT_OVERLAP_DISCARD_THRESHOLD}."
        ),
    )
    parser.add_argument("--disable-interior", action="store_true", help="Generate only accepted circular edge rows.")
    parser.add_argument(
        "--section-lines-only",
        action="store_true",
        help="Generate and plot ordered interior straight section lines without profiles or circular sweeps.",
    )
    parser.add_argument(
        "--lawnmower-profiles-only",
        action="store_true",
        help="Plot rainbow profiles anchored to existing interior guide lines without circular sweeps.",
    )
    parser.add_argument(
        "--save-section-lines-json",
        help="Optional JSON output for --section-lines-only geometry.",
    )
    parser.add_argument(
        "--highlight-outer-sections",
        action="store_true",
        help="Show the four generated outer lawnmower guide sections in diagnostic plots.",
    )
    parser.add_argument("--clockwise", action="store_true", help="Generate poses in clockwise order.")
    parser.add_argument("--no-plot", action="store_true", help="Save plot without opening a matplotlib window.")
    parser.add_argument("--save-json", default=DEFAULT_JSON_PATH, help="Path for mission JSON output.")
    parser.add_argument("--save-csv", help="Optional path for mission CSV output.")
    parser.add_argument("--save-plot", default=DEFAULT_PLOT_PATH, help="Path for preview plot output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.lawnmower_profiles_only:
        try:
            model = import_dxf(args.dxf_path)
            lines = generate_lawnmower_section_lines(model)
            placements = generate_lawnmower_scan_placements(lines)
        except (DxfImportError, ValueError) as exc:
            print(f"Lawnmower profile placement failed: {exc}")
            return 1
        plot_path = Path(args.save_plot) if args.save_plot else None
        if args.save_plot or not args.no_plot:
            plot_lawnmower_scan_placements(
                model,
                lines,
                placements,
                show=not args.no_plot,
                save_path=plot_path,
            )
        vertical_count = sum(placement.orientation == "vertical" for placement in placements)
        horizontal_count = sum(placement.orientation == "horizontal" for placement in placements)
        print("Lawnmower rainbow profile summary")
        print(f"Vertical guide-line profiles: {vertical_count}")
        print(f"Horizontal guide-line profiles: {horizontal_count}")
        print(f"Total interior profiles: {len(placements)}")
        if plot_path is not None:
            print(f"Plot output: {plot_path}")
        return 0

    if args.section_lines_only:
        try:
            model = import_dxf(args.dxf_path)
            lines = generate_lawnmower_section_lines(model)
        except (DxfImportError, ValueError) as exc:
            print(f"Section-line generation failed: {exc}")
            return 1
        section_json_path = (
            save_lawnmower_section_lines_json(lines, args.save_section_lines_json)
            if args.save_section_lines_json
            else None
        )
        plot_path = Path(args.save_plot) if args.save_plot else None
        if args.save_plot or not args.no_plot:
            plot_lawnmower_section_lines(model, lines, show=not args.no_plot, save_path=plot_path)
        vertical_count = sum(line.orientation == "vertical" for line in lines)
        horizontal_count = sum(line.orientation == "horizontal" for line in lines)
        print("Lawnmower section-line summary")
        print(f"Vertical section lines: {vertical_count}")
        print(f"Horizontal section lines: {horizontal_count}")
        print(f"Total section lines: {len(lines)}")
        if section_json_path is not None:
            print(f"Section JSON output: {section_json_path}")
        if plot_path is not None:
            print(f"Plot output: {plot_path}")
        return 0

    _warn_for_spacing_factor(args.spacing_factor)
    _warn_for_row_spacing(args.row_spacing, RainbowProfileConfig())
    _warn_for_vertical_spacing_factor(args.vertical_spacing_factor)

    try:
        model = import_dxf(args.dxf_path)
        mission = build_mission_plan(
            model,
            outer_edge_offset_m=args.outer_edge_offset,
            row_spacing_m=args.row_spacing,
            spacing_factor=args.spacing_factor,
            max_circular_gap_fraction=args.max_circular_gap_fraction,
            circular_stop_backoff_rows=args.circular_stop_backoff_rows,
            vertical_spacing_factor=args.vertical_spacing_factor,
            overlap_discard_threshold=args.overlap_discard_threshold,
            interior_enabled=not args.disable_interior,
            clockwise=args.clockwise,
        )
    except (DxfImportError, ValueError) as exc:
        print(f"Outer edge sweep planning failed: {exc}")
        return 1

    json_path = save_mission_json(mission, args.save_json) if args.save_json else None
    csv_path = save_mission_csv(mission, args.save_csv) if args.save_csv else None
    plot_path = Path(args.save_plot) if args.save_plot else None
    if args.save_plot or not args.no_plot:
        plot_outer_edge_sweep(
            model,
            mission,
            show=not args.no_plot,
            save_path=plot_path,
            highlight_outer_lawnmower_sections=args.highlight_outer_sections,
        )
    print_mission_summary(mission, json_path=json_path, csv_path=csv_path, plot_path=plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
