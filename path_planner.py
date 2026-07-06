from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from dxf_importer import DxfImportError, GeometryModel, import_dxf
from scan_profile import RainbowProfileConfig, make_rainbow_profile, transform_profile


DEFAULT_EDGE_OFFSET_FROM_WALL = 0.0
DEFAULT_ROW_SPACING = 1.60
DEFAULT_SPACING_FACTOR = 1.0
DEFAULT_CIRCULAR_SPACING_CANDIDATES = (1.00, 0.95, 0.90, 0.85, 0.80)
TARGET_COVERAGE_PERCENT = 95.0
DEFAULT_MAX_CIRCULAR_GAP_FRACTION = 0.05
DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS = 3
DEFAULT_VERTICAL_SPACING_FACTOR = 0.95
DEFAULT_OVERLAP_DISCARD_THRESHOLD = 0.50
DEFAULT_VERTICAL_COLUMN_EDGE_OVERLAP_LIMIT = 1.0 / 3.0
SIDE_GUARD_HEADING_RAD = math.pi / 2.0
DEFAULT_JSON_PATH = "mission_plan.json"
DEFAULT_PLOT_PATH = "mission_preview.png"
VERTICAL_SLOPE_TOLERANCE = 0.02
VERTICAL_X_GROUP_TOLERANCE_M = 0.02
VERTICAL_LINE_COVERAGE_RATIO = 0.60
OVERLAP_SAMPLE_GRID_SIZE = 28
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


@dataclass(frozen=True)
class VerticalPlateColumn:
    column_id: int
    center_x_m: float
    left_weld_x_m: float
    right_weld_x_m: float
    min_y_m: float
    max_y_m: float


@dataclass(frozen=True)
class CircularSweepRow:
    row_id: int
    sweep_radius_m: float
    scan_count: int
    angular_spacing_rad: float
    arc_spacing_m: float
    estimated_gap_area_m2: float
    estimated_gap_fraction: float


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
    spacing_factor: float
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
) -> tuple[list[SweepPose], CircularSweepRow] | None:
    touching_angle = _touching_angular_step(sweep_radius, profile_config)
    target_arc_step = touching_angle * sweep_radius * spacing_factor
    if target_arc_step <= 0.0:
        return None
    pose_count = max(3, int(math.ceil((2.0 * math.pi * sweep_radius) / target_arc_step)))
    angular_spacing = 2.0 * math.pi / pose_count

    profile_area = _polygon_area(make_rainbow_profile(profile_config))
    gap_arc_length = max(0.0, angular_spacing - touching_angle) * sweep_radius
    gap_area = gap_arc_length * profile_config.width
    gap_fraction = gap_area / max(profile_area, 1e-9)
    direction_sign = -1.0 if clockwise else 1.0
    center_x, center_y = tank_center
    poses: list[SweepPose] = []

    for local_index in range(pose_count):
        theta = start_theta + direction_sign * 2.0 * math.pi * local_index / pose_count
        normalized_theta = _normalize_angle(theta)
        x_m = center_x + sweep_radius * math.cos(theta)
        y_m = center_y + sweep_radius * math.sin(theta)
        heading_rad = normalized_theta
        poses.append(
            SweepPose(
                scan_id=scan_id_start + local_index,
                stage="circular_edge",
                x_m=float(x_m),
                y_m=float(y_m),
                heading_rad=heading_rad,
                heading_deg=math.degrees(heading_rad),
                row_id=row_id,
                row_name=row_name,
                theta_rad=normalized_theta,
                sweep_radius_m=sweep_radius,
            )
        )
    return poses, CircularSweepRow(
        row_id=row_id,
        sweep_radius_m=sweep_radius,
        scan_count=pose_count,
        angular_spacing_rad=angular_spacing,
        arc_spacing_m=sweep_radius * angular_spacing,
        estimated_gap_area_m2=gap_area,
        estimated_gap_fraction=gap_fraction,
    )


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
    rows: list[CircularSweepRow] = []
    poses: list[SweepPose] = []
    start_theta = 0.0

    while current_radius > profile_config.width / 2.0:
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
            return _apply_circular_stop_backoff(rows, poses, current_radius, "overlap", circular_stop_backoff_rows)

        row_poses, row = generated
        if row.estimated_gap_fraction > max_circular_gap_fraction:
            return _apply_circular_stop_backoff(rows, poses, current_radius, "gap", circular_stop_backoff_rows)

        rows.append(row)
        poses.extend(row_poses)
        start_theta = row_poses[-1].theta_rad
        current_radius -= row_spacing_m

    return _apply_circular_stop_backoff(rows, poses, current_radius, "radius", circular_stop_backoff_rows)


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
    max_circular_gap_fraction: float = DEFAULT_MAX_CIRCULAR_GAP_FRACTION,
    circular_stop_backoff_rows: int = DEFAULT_CIRCULAR_STOP_BACKOFF_ROWS,
    profile_config: RainbowProfileConfig | None = None,
    clockwise: bool = False,
) -> OuterEdgeSweepMission:
    """Build metadata and accepted circular edge rows from imported geometry."""
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    tank = estimate_tank_circle_from_geometry(model)
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

    tank = estimate_tank_circle_from_geometry(model)
    columns = detect_vertical_plate_columns(model, tank)
    interior_poses, discarded_count = generate_interior_vertical_poses(
        columns,
        tank,
        edge_mission.poses,
        scan_id_start=len(edge_mission.poses),
        profile_config=profile_config,
        vertical_spacing_factor=vertical_spacing_factor,
        overlap_discard_threshold=overlap_discard_threshold,
    )
    return replace(
        edge_mission,
        interior_enabled=True,
        vertical_spacing_factor=vertical_spacing_factor,
        overlap_discard_threshold=overlap_discard_threshold,
        vertical_columns=columns,
        interior_kept_count=len(interior_poses),
        interior_discarded_count=discarded_count,
        total_scan_count=len(edge_mission.poses) + len(interior_poses),
        poses=edge_mission.poses + interior_poses,
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
    """Build a mission, selecting circular spacing by coverage when candidates are provided."""
    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    if not circular_spacing_candidates:
        return _build_mission_plan_once(
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

    candidate_missions: list[tuple[float, OuterEdgeSweepMission, CoverageEstimate, float]] = []
    for candidate_factor in circular_spacing_candidates:
        if candidate_factor < 0.80:
            continue
        mission = _build_mission_plan_once(
            model,
            outer_edge_offset_m=outer_edge_offset_m,
            row_spacing_m=row_spacing_m,
            spacing_factor=candidate_factor,
            max_circular_gap_fraction=max_circular_gap_fraction,
            circular_stop_backoff_rows=circular_stop_backoff_rows,
            vertical_spacing_factor=vertical_spacing_factor,
            overlap_discard_threshold=overlap_discard_threshold,
            interior_enabled=interior_enabled,
            profile_config=profile_config,
            clockwise=clockwise,
        )
        coverage = _estimate_tank_coverage(mission, profile_config, grid_resolution=80)
        travel_distance = _estimate_mission_travel_distance(mission.poses)
        candidate_missions.append((candidate_factor, mission, coverage, travel_distance))

    if not candidate_missions:
        raise ValueError("No valid circular spacing candidates were provided.")

    selected = _select_circular_spacing_candidate(candidate_missions)

    selected_factor, selected_mission, _selected_coverage, _selected_travel = selected
    candidate_records = [
        CircularSpacingCandidate(
            spacing_factor=factor,
            coverage_percent=coverage.covered_percent,
            scan_count=len(mission.poses),
            travel_distance_m=travel_distance,
            selected=math.isclose(factor, selected_factor),
        )
        for factor, mission, coverage, travel_distance in candidate_missions
    ]
    return replace(
        selected_mission,
        spacing_factor=selected_factor,
        selected_circular_spacing_factor=selected_factor,
        circular_spacing_candidates=candidate_records,
        circular_spacing_selection_reason=_circular_spacing_selection_reason(candidate_records),
    )


def _select_circular_spacing_candidate(
    candidates: list[tuple[float, OuterEdgeSweepMission, CoverageEstimate, float]],
) -> tuple[float, OuterEdgeSweepMission, CoverageEstimate, float]:
    target_candidates = [
        candidate for candidate in candidates if candidate[2].covered_percent >= TARGET_COVERAGE_PERCENT
    ]
    if target_candidates:
        return min(
            target_candidates,
            key=lambda candidate: (
                len(candidate[1].poses),
                candidate[3],
                -candidate[0],
            ),
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate[2].covered_percent,
            -len(candidate[1].poses),
            -candidate[3],
            candidate[0],
        ),
    )


def _circular_spacing_selection_reason(candidates: list[CircularSpacingCandidate]) -> str | None:
    selected = next((candidate for candidate in candidates if candidate.selected), None)
    if selected is None:
        return None
    if selected.coverage_percent >= TARGET_COVERAGE_PERCENT:
        return (
            f"reached {TARGET_COVERAGE_PERCENT:.0f}% coverage; chose the simpler qualifying spacing"
        )
    return (
        f"target {TARGET_COVERAGE_PERCENT:.0f}% was not reached; selected the best available coverage "
        "without going below the allowed overlap limit"
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
    """Detect regular plate columns between long near-vertical DXF weld lines."""
    tank = estimate_tank_circle_from_geometry(model) if tank is None else tank
    grouped_segments = _group_vertical_segments(_iter_geometry_segments(model))
    qualifying_lines: list[tuple[float, float, float]] = []

    for x_m, intervals in grouped_segments:
        merged = _merge_intervals(intervals)
        union_length = sum(end - start for start, end in merged)
        radial_x = x_m - tank.center_x
        chord_height = 2.0 * math.sqrt(max(0.0, tank.radius**2 - radial_x**2))
        if chord_height <= 1e-9:
            continue
        if union_length / chord_height < VERTICAL_LINE_COVERAGE_RATIO:
            continue
        qualifying_lines.append(
            (
                x_m,
                min(start for start, _end in merged),
                max(end for _start, end in merged),
            )
        )

    qualifying_lines.sort(key=lambda item: item[0])
    if len(qualifying_lines) < 2:
        return []

    gaps = np.diff([line[0] for line in qualifying_lines])
    typical_gap = float(np.median(gaps))
    max_regular_gap = typical_gap * 1.5
    columns: list[VerticalPlateColumn] = []
    for left, right in zip(qualifying_lines, qualifying_lines[1:]):
        gap = right[0] - left[0]
        if gap <= 1e-9 or gap > max_regular_gap:
            continue
        min_y = max(left[1], right[1])
        max_y = min(left[2], right[2])
        if max_y <= min_y:
            continue
        columns.append(
            VerticalPlateColumn(
                column_id=len(columns),
                center_x_m=(left[0] + right[0]) / 2.0,
                left_weld_x_m=left[0],
                right_weld_x_m=right[0],
                min_y_m=min_y,
                max_y_m=max_y,
            )
        )
    return columns


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
    edge_polygons = [
        transform_profile(local_profile, pose.x_m, pose.y_m, pose.heading_rad)
        for pose in edge_poses
        if pose.stage == "circular_edge"
    ]
    edge_polygon_records = [(polygon, _polygon_bounds(polygon)) for polygon in edge_polygons]
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
    if column_index % 2 == 1:
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
        if side_index % 2 == 1:
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
                )
            )

    return kept, discarded_count


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
        world_profile = transform_profile(local_profile, pose.x_m, pose.y_m, pose.heading_rad)
        closed_profile = _closed_points(world_profile)
        if pose.stage == "circular_edge":
            color = circular_palette[int(pose.row_id or 0) % len(circular_palette)]
            style = {"color": color, "fill": color}
        elif pose.stage == "interior_side_guard":
            style = {"color": "#0f766e", "fill": "#5eead4"}
        else:
            style = {"color": "#7c3aed", "fill": "#a78bfa"}
        ax.fill(closed_profile[:, 0], closed_profile[:, 1], color=style["fill"], alpha=0.07, zorder=2)
        ax.plot(closed_profile[:, 0], closed_profile[:, 1], color=style["color"], linewidth=0.7, alpha=0.38, zorder=3)

    circular_poses = [pose for pose in mission.poses if pose.stage == "circular_edge"]
    interior_poses = [pose for pose in mission.poses if pose.stage == "interior_vertical"]
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
    path_x = [pose.x_m for pose in mission.poses]
    path_y = [pose.y_m for pose in mission.poses]
    ax.plot(path_x, path_y, color="#f97316", linewidth=1.0, alpha=0.65, label="Ordered mission path", zorder=4)
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
            f"gap_fraction={row.estimated_gap_fraction:.6g}"
        )
    if mission.rejected_circular_radius_m is not None:
        print(f"Circular stop radius: {mission.rejected_circular_radius_m:.6g} m")
    print(f"Circular wrapping stop reason: {mission.circular_stop_reason}")
    print(f"Max circular gap fraction: {mission.max_circular_gap_fraction:.6g}")
    print(f"Total circular scans: {mission.edge_sweep_scan_count}")
    print(f"Detected vertical plate columns: {len(mission.vertical_columns)}")
    vertical_count = sum(1 for pose in mission.poses if pose.stage == "interior_vertical")
    side_guard_count = sum(1 for pose in mission.poses if pose.stage == "interior_side_guard")
    print(f"Interior vertical scans kept: {vertical_count}")
    print(f"Rotated side-guard scans kept: {side_guard_count}")
    print(f"Interior vertical scans discarded by overlap: {mission.interior_discarded_count}")
    print(f"Total mission scans: {len(mission.poses)}")
    if mission.circular_spacing_candidates:
        print("Circular spacing comparison")
        print("circular_spacing | coverage_% | scans | travel_m")
        for candidate in mission.circular_spacing_candidates:
            selected_marker = "  <-- selected" if candidate.selected else ""
            print(
                f"{candidate.spacing_factor:<16.2f} | "
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
    vertical: list[tuple[float, float, float, float]] = []
    for start, end in segments:
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9 or abs(float(delta[0])) > VERTICAL_SLOPE_TOLERANCE * length:
            continue
        vertical.append(
            (
                float((start[0] + end[0]) / 2.0),
                float(min(start[1], end[1])),
                float(max(start[1], end[1])),
                length,
            )
        )
    vertical.sort(key=lambda item: item[0])

    groups: list[list[tuple[float, float, float, float]]] = []
    for segment in vertical:
        if not groups:
            groups.append([segment])
            continue
        group_x = sum(item[0] * item[3] for item in groups[-1]) / sum(item[3] for item in groups[-1])
        if abs(segment[0] - group_x) <= VERTICAL_X_GROUP_TOLERANCE_M:
            groups[-1].append(segment)
        else:
            groups.append([segment])

    grouped: list[tuple[float, list[tuple[float, float]]]] = []
    for group in groups:
        total_length = sum(item[3] for item in group)
        x_m = sum(item[0] * item[3] for item in group) / total_length
        grouped.append((x_m, [(item[1], item[2]) for item in group]))
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
    local_profile = make_rainbow_profile(profile_config)

    for pose in mission.poses:
        polygon = transform_profile(local_profile, pose.x_m, pose.y_m, pose.heading_rad)
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
    if spacing_factor < 0.25 or spacing_factor > 1.25:
        print(
            "Warning: spacing factor is outside the typical v1 range of 0.25 to 1.25; "
            "expect unusual overlap or gaps."
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
    parser = argparse.ArgumentParser(description="Plan edge sweeps and vertical interior coverage from a tank DXF.")
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
            "Multiplier on the maximum non-overlapping circular pose count. Values above 1 reduce density; "
            "values below 1 are capped to prevent overlap. "
            f"Default: {DEFAULT_SPACING_FACTOR}."
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
    parser.add_argument("--clockwise", action="store_true", help="Generate poses in clockwise order.")
    parser.add_argument("--no-plot", action="store_true", help="Save plot without opening a matplotlib window.")
    parser.add_argument("--save-json", default=DEFAULT_JSON_PATH, help="Path for mission JSON output.")
    parser.add_argument("--save-csv", help="Optional path for mission CSV output.")
    parser.add_argument("--save-plot", default=DEFAULT_PLOT_PATH, help="Path for preview plot output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
        plot_outer_edge_sweep(model, mission, show=not args.no_plot, save_path=plot_path)
    print_mission_summary(mission, json_path=json_path, csv_path=csv_path, plot_path=plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
