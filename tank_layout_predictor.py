from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ORIENTATIONS = ("vertical", "horizontal", "diagonal")


@dataclass(frozen=True)
class LayoutPoint:
    x: float
    y: float


@dataclass(frozen=True)
class TankGeometry:
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class ObservationRegion:
    kind: str = "unknown"
    inner_radius_fraction: float | None = None
    outer_radius_fraction: float = 1.0


@dataclass(frozen=True)
class ObservedSegment:
    start: LayoutPoint
    end: LayoutPoint
    source: str = "observed"
    confidence: float = 1.0
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedArc:
    center: LayoutPoint
    radius: float
    start_angle_deg: float
    end_angle_deg: float
    source: str = "observed"
    confidence: float = 1.0
    source_id: str | None = None


@dataclass(frozen=True)
class ObservedTankGeometry:
    tank: TankGeometry
    segments: list[ObservedSegment]
    arcs: list[ObservedArc] = field(default_factory=list)
    observation_region: ObservationRegion | None = None
    units: str = "m"
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictorConfig:
    angle_tolerance_deg: float = 5.0
    position_tolerance_norm: float = 0.008
    merge_gap_norm: float = 0.025
    minimum_segment_length_norm: float = 0.008
    minimum_spacing_norm: float = 0.025
    maximum_spacing_norm: float = 0.85
    symmetry_tolerance_norm: float = 0.018
    boundary_margin_norm: float = 0.002
    minimum_pattern_support: int = 2


@dataclass(frozen=True)
class LayoutSegment:
    start: LayoutPoint
    end: LayoutPoint
    source: str
    confidence: float
    confidence_level: str
    orientation: str
    prediction_method: str
    supporting_observations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PatternParameters:
    vertical_positions_normalized: list[float]
    horizontal_positions_normalized: list[float]
    vertical_spacing_normalized: float | None
    horizontal_spacing_normalized: float | None
    staggered_horizontal_joints: bool
    symmetry_score: float


@dataclass(frozen=True)
class CompletedTankLayout:
    units: str
    tank_boundary: TankGeometry
    observed_weld_segments: list[LayoutSegment]
    predicted_weld_segments: list[LayoutSegment]
    selected_pattern_family: str
    overall_confidence: float
    evidence: list[str]
    warnings: list[str]
    normalized_pattern_parameters: PatternParameters
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrientationValidation:
    orientation: str
    matched_lines: int
    false_predictions: int
    missed_lines: int
    mean_normalized_position_error: float | None
    mean_world_position_error: float | None


@dataclass(frozen=True)
class PredictionValidationReport:
    orientations: list[OrientationValidation]
    confidence_distribution: dict[str, int]


@dataclass(frozen=True)
class _NormalizedSegment:
    start: LayoutPoint
    end: LayoutPoint
    orientation: str
    confidence: float
    source_id: str

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


@dataclass(frozen=True)
class _SpacingEstimate:
    spacing: float
    support_fraction: float
    direct_support: int


def observed_geometry_from_dxf(
    model: Any,
    *,
    tank_geometry: TankGeometry | None = None,
    observation_region: ObservationRegion | None = None,
    units: str = "m",
) -> ObservedTankGeometry:
    """Adapt the importer model to the predictor's source-independent input."""
    if tank_geometry is None:
        if model.bounds is None:
            raise ValueError("DXF geometry has no bounds; tank geometry is required.")
        center_x = (float(model.bounds.min_x) + float(model.bounds.max_x)) / 2.0
        center_y = (float(model.bounds.min_y) + float(model.bounds.max_y)) / 2.0
        radius = max(float(model.bounds.width), float(model.bounds.height)) / 2.0
        if model.circles:
            boundary = max(model.circles, key=lambda circle: float(circle.radius))
            center_x = float(boundary.center.x)
            center_y = float(boundary.center.y)
            radius = float(boundary.radius)
        tank_geometry = TankGeometry(center_x, center_y, radius)

    segments: list[ObservedSegment] = []
    for index, line in enumerate(model.line_segments):
        segments.append(
            ObservedSegment(
                LayoutPoint(float(line.start.x), float(line.start.y)),
                LayoutPoint(float(line.end.x), float(line.end.y)),
                source_id=f"line:{line.handle or index}",
                metadata={"layer": line.layer, "entity_type": "LINE"},
            )
        )
    for polyline_index, polyline in enumerate(model.polylines):
        points = [vertex.point for vertex in polyline.vertices]
        pairs = list(zip(points, points[1:]))
        if polyline.closed and len(points) > 1:
            pairs.append((points[-1], points[0]))
        for edge_index, (start, end) in enumerate(pairs):
            segments.append(
                ObservedSegment(
                    LayoutPoint(float(start.x), float(start.y)),
                    LayoutPoint(float(end.x), float(end.y)),
                    source_id=f"polyline:{polyline.handle or polyline_index}:{edge_index}",
                    metadata={"layer": polyline.layer, "entity_type": polyline.entity_type},
                )
            )

    arcs = [
        ObservedArc(
            center=LayoutPoint(float(arc.center.x), float(arc.center.y)),
            radius=float(arc.radius),
            start_angle_deg=float(arc.start_angle),
            end_angle_deg=float(arc.end_angle),
            source_id=f"arc:{arc.handle or index}",
        )
        for index, arc in enumerate(model.arcs)
    ]
    source_path = str(model.source_path) if model.source_path is not None else None
    return ObservedTankGeometry(
        tank=tank_geometry,
        segments=segments,
        arcs=arcs,
        observation_region=observation_region,
        units=units,
        source_metadata={"source_type": "dxf", "source_path": source_path},
    )


def predict_tank_layout(
    observed_geometry: ObservedTankGeometry,
    tank_geometry: TankGeometry | None = None,
    config: PredictorConfig | None = None,
) -> CompletedTankLayout:
    """Infer a scale-independent weld layout from partial observed geometry."""
    config = PredictorConfig() if config is None else config
    tank = observed_geometry.tank if tank_geometry is None else tank_geometry
    _validate_inputs(observed_geometry, tank, config)

    normalized = _normalize_and_classify(observed_geometry.segments, tank, config)
    cleaned = _merge_collinear_segments(normalized, config)
    observed_layout = [_observed_layout_segment(segment, tank) for segment in normalized]
    warnings: list[str] = []
    evidence: list[str] = []

    vertical = [segment for segment in cleaned if segment.orientation == "vertical"]
    horizontal = [segment for segment in cleaned if segment.orientation == "horizontal"]
    diagonal = [segment for segment in cleaned if segment.orientation == "diagonal"]
    if len(cleaned) < config.minimum_pattern_support:
        warnings.append("Too few usable observations to infer a repeated layout.")
    if observed_geometry.segments and len(normalized) < len(observed_geometry.segments) * 0.75:
        warnings.append("Many observations were too short or noisy for pattern detection.")

    vertical_positions = _cluster_axis_positions(vertical, "vertical", config.position_tolerance_norm)
    horizontal_positions = _cluster_axis_positions(horizontal, "horizontal", config.position_tolerance_norm)
    vertical_spacing = _infer_spacing([vertical_positions], config)
    horizontal_groups = _horizontal_position_groups(horizontal, vertical_positions, config)
    horizontal_spacing = _infer_spacing(list(horizontal_groups.values()), config)
    if horizontal_spacing is None:
        horizontal_spacing = _infer_spacing([horizontal_positions], config)

    symmetry_score = _symmetry_score(vertical_positions + horizontal_positions, config.symmetry_tolerance_norm)
    staggered = _detect_staggered_groups(horizontal_groups, horizontal_spacing, config)
    family = _select_pattern_family(vertical, horizontal, diagonal, staggered)

    if vertical_spacing is not None:
        evidence.append(
            f"Vertical normalized spacing {vertical_spacing.spacing:.5f} with "
            f"{vertical_spacing.direct_support} direct repeated intervals."
        )
        if vertical_spacing.support_fraction < 0.60:
            warnings.append("Vertical spacing is inconsistent; vertical completion confidence is limited.")
    else:
        warnings.append("Vertical spacing could not be inferred reliably.")
    if horizontal_spacing is not None:
        evidence.append(
            f"Horizontal normalized spacing {horizontal_spacing.spacing:.5f} with "
            f"{horizontal_spacing.direct_support} direct repeated intervals."
        )
        if horizontal_spacing.support_fraction < 0.60:
            warnings.append("Horizontal spacing is inconsistent; horizontal completion confidence is limited.")
    else:
        warnings.append("Horizontal spacing could not be inferred reliably.")
    if diagonal:
        evidence.append(f"Observed {len(diagonal)} diagonal transition segments.")
    evidence.append(f"Normalized axis symmetry score: {symmetry_score:.2f}.")
    if family == "partial_structural_pattern" and diagonal and not (vertical or horizontal):
        warnings.append("Irregular diagonal-only geometry is not supported for detailed completion in v1.")
    if vertical and horizontal and symmetry_score < 0.45:
        warnings.append("Multiple asymmetric pattern interpretations remain plausible.")

    proposed: list[_NormalizedSegment] = []
    proposed.extend(_predict_vertical_grid(vertical, vertical_positions, vertical_spacing, config))
    proposed.extend(
        _predict_horizontal_grid(
            horizontal,
            vertical_positions,
            horizontal_groups,
            horizontal_spacing,
            staggered,
            config,
        )
    )
    proposed.extend(_predict_diagonal_symmetry(diagonal, config))
    proposed = _deduplicate_predictions(proposed, normalized, config)

    predicted_layout: list[LayoutSegment] = []
    spacing_confidences = [
        estimate.support_fraction
        for estimate in (vertical_spacing, horizontal_spacing)
        if estimate is not None
    ]
    base_confidence = float(np.mean(spacing_confidences)) if spacing_confidences else 0.25
    for segment in proposed:
        confidence = min(segment.confidence, max(0.2, base_confidence))
        predicted_layout.append(
            _predicted_layout_segment(
                segment,
                tank,
                confidence,
                _prediction_method_for(segment.orientation, staggered),
            )
        )

    if not predicted_layout:
        warnings.append("No missing geometry had enough support to predict.")
    if base_confidence < 0.45:
        warnings.append("Pattern evidence is weak; treat predictions as provisional.")

    parameters = PatternParameters(
        vertical_positions_normalized=[round(value, 9) for value in vertical_positions],
        horizontal_positions_normalized=[round(value, 9) for value in horizontal_positions],
        vertical_spacing_normalized=None if vertical_spacing is None else vertical_spacing.spacing,
        horizontal_spacing_normalized=None if horizontal_spacing is None else horizontal_spacing.spacing,
        staggered_horizontal_joints=staggered,
        symmetry_score=symmetry_score,
    )
    return CompletedTankLayout(
        units=observed_geometry.units,
        tank_boundary=tank,
        observed_weld_segments=observed_layout,
        predicted_weld_segments=predicted_layout,
        selected_pattern_family=family,
        overall_confidence=min(0.90, max(0.0, base_confidence * (0.70 + 0.20 * symmetry_score))),
        evidence=evidence,
        warnings=warnings,
        normalized_pattern_parameters=parameters,
        source_metadata=dict(observed_geometry.source_metadata),
    )


def mask_observations_to_annulus(
    observed_geometry: ObservedTankGeometry,
    inner_radius_fraction: float,
) -> ObservedTankGeometry:
    """Simulate geometry visible only in an outer scanned annulus."""
    if not 0.0 <= inner_radius_fraction < 1.0:
        raise ValueError("Annulus inner radius fraction must be in [0, 1).")
    tank = observed_geometry.tank
    inner_radius = tank.radius * inner_radius_fraction
    masked: list[ObservedSegment] = []
    for segment in observed_geometry.segments:
        pieces = _clip_segment_to_annulus(segment, tank, inner_radius, tank.radius)
        masked.extend(pieces)
    return ObservedTankGeometry(
        tank=tank,
        segments=masked,
        arcs=list(observed_geometry.arcs),
        observation_region=ObservationRegion("annulus", inner_radius_fraction, 1.0),
        units=observed_geometry.units,
        source_metadata={**observed_geometry.source_metadata, "masked_for_validation": True},
    )


def scale_observed_geometry(
    observed_geometry: ObservedTankGeometry,
    scale_factor: float,
    *,
    new_center: tuple[float, float] | None = None,
) -> ObservedTankGeometry:
    """Uniformly scale source geometry while preserving its normalized pattern."""
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("Scale factor must be positive and finite.")
    original_tank = observed_geometry.tank
    center_x, center_y = new_center or (original_tank.center_x, original_tank.center_y)
    scaled_tank = TankGeometry(center_x, center_y, original_tank.radius * scale_factor)

    def scale_point(point: LayoutPoint) -> LayoutPoint:
        normalized = _normalize_point(point, original_tank)
        return _world_point(normalized, scaled_tank)

    segments = [
        ObservedSegment(
            start=scale_point(segment.start),
            end=scale_point(segment.end),
            source=segment.source,
            confidence=segment.confidence,
            source_id=segment.source_id,
            metadata=dict(segment.metadata),
        )
        for segment in observed_geometry.segments
    ]
    arcs = [
        ObservedArc(
            center=scale_point(arc.center),
            radius=arc.radius * scale_factor,
            start_angle_deg=arc.start_angle_deg,
            end_angle_deg=arc.end_angle_deg,
            source=arc.source,
            confidence=arc.confidence,
            source_id=arc.source_id,
        )
        for arc in observed_geometry.arcs
    ]
    return ObservedTankGeometry(
        tank=scaled_tank,
        segments=segments,
        arcs=arcs,
        observation_region=observed_geometry.observation_region,
        units=observed_geometry.units,
        source_metadata={
            **observed_geometry.source_metadata,
            "uniform_scale_factor": scale_factor,
        },
    )


def validate_prediction(
    layout: CompletedTankLayout,
    ground_truth: ObservedTankGeometry,
    *,
    position_tolerance_norm: float = 0.02,
) -> PredictionValidationReport:
    """Compare predicted and ground-truth axis-line positions."""
    completed_normalized = [
        _layout_to_normalized(segment, layout.tank_boundary)
        for segment in layout.observed_weld_segments + layout.predicted_weld_segments
    ]
    truth_normalized = _normalize_and_classify(
        ground_truth.segments,
        ground_truth.tank,
        PredictorConfig(position_tolerance_norm=position_tolerance_norm),
    )
    reports: list[OrientationValidation] = []
    for orientation in ("vertical", "horizontal"):
        predicted_positions = _cluster_axis_positions(
            [segment for segment in completed_normalized if segment.orientation == orientation],
            orientation,
            position_tolerance_norm / 2.0,
        )
        truth_positions = _cluster_axis_positions(
            [segment for segment in truth_normalized if segment.orientation == orientation],
            orientation,
            position_tolerance_norm / 2.0,
        )
        matched, errors = _match_positions(predicted_positions, truth_positions, position_tolerance_norm)
        reports.append(
            OrientationValidation(
                orientation=orientation,
                matched_lines=matched,
                false_predictions=max(0, len(predicted_positions) - matched),
                missed_lines=max(0, len(truth_positions) - matched),
                mean_normalized_position_error=float(np.mean(errors)) if errors else None,
                mean_world_position_error=(
                    float(np.mean(errors)) * ground_truth.tank.radius if errors else None
                ),
            )
        )
    return PredictionValidationReport(
        orientations=reports,
        confidence_distribution=dict(Counter(segment.confidence_level for segment in layout.predicted_weld_segments)),
    )


def save_layout_json(layout: CompletedTankLayout, filepath: str | Path) -> Path:
    path = Path(filepath)
    path.write_text(json.dumps(asdict(layout), indent=2), encoding="utf-8")
    return path


def export_layout_dxf(layout: CompletedTankLayout, filepath: str | Path) -> Path:
    """Write a new DXF with observed and predicted confidence layers."""
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError("DXF export requires ezdxf.") from exc

    path = Path(filepath)
    doc = ezdxf.new("R2010")
    for layer_name, color in (
        ("TANK_BOUNDARY", 7),
        ("OBSERVED_WELDS", 3),
        ("PREDICTED_WELDS_HIGH", 5),
        ("PREDICTED_WELDS_MEDIUM", 2),
        ("PREDICTED_WELDS_LOW", 1),
    ):
        doc.layers.add(layer_name, color=color)
    modelspace = doc.modelspace()
    tank = layout.tank_boundary
    modelspace.add_circle((tank.center_x, tank.center_y), tank.radius, dxfattribs={"layer": "TANK_BOUNDARY"})
    for segment in layout.observed_weld_segments:
        modelspace.add_line(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
            dxfattribs={"layer": "OBSERVED_WELDS"},
        )
    for segment in layout.predicted_weld_segments:
        layer = f"PREDICTED_WELDS_{segment.confidence_level.upper()}"
        modelspace.add_line(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
            dxfattribs={"layer": layer},
        )
    doc.saveas(path)
    return path


def plot_predicted_layout(
    layout: CompletedTankLayout,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(9, 9))
    tank = layout.tank_boundary
    ax.add_patch(Circle((tank.center_x, tank.center_y), tank.radius, fill=False, color="#111827", linewidth=2.0))
    for segment in layout.observed_weld_segments:
        ax.plot(
            [segment.start.x, segment.end.x],
            [segment.start.y, segment.end.y],
            color="#374151",
            linewidth=1.2,
            alpha=0.75,
        )
    colors = {"high": "#059669", "medium": "#d97706", "low": "#dc2626"}
    for segment in layout.predicted_weld_segments:
        ax.plot(
            [segment.start.x, segment.end.x],
            [segment.start.y, segment.end.y],
            color=colors[segment.confidence_level],
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )
    ax.set_aspect("equal", adjustable="box")
    margin = tank.radius * 0.05
    ax.set_xlim(tank.center_x - tank.radius - margin, tank.center_x + tank.radius + margin)
    ax.set_ylim(tank.center_y - tank.radius - margin, tank.center_y + tank.radius + margin)
    ax.set_xlabel(f"x ({layout.units})")
    ax.set_ylabel(f"y ({layout.units})")
    ax.set_title(
        f"Predicted tank layout - {layout.selected_pattern_family}\n"
        f"confidence {layout.overall_confidence:.0%}"
    )
    ax.grid(True, alpha=0.2)
    handles = [
        Line2D([], [], color="#374151", label="Observed welds"),
        Line2D([], [], color="#059669", linestyle="--", label="High confidence"),
        Line2D([], [], color="#d97706", linestyle="--", label="Medium confidence"),
        Line2D([], [], color="#dc2626", linestyle="--", label="Low confidence"),
    ]
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    return fig, ax


def _validate_inputs(
    observed: ObservedTankGeometry,
    tank: TankGeometry,
    config: PredictorConfig,
) -> None:
    if not all(math.isfinite(value) for value in (tank.center_x, tank.center_y, tank.radius)):
        raise ValueError("Tank center and radius must be finite.")
    if tank.radius <= 0.0:
        raise ValueError("Tank radius must be positive.")
    if not 0.0 < config.angle_tolerance_deg < 45.0:
        raise ValueError("Angle tolerance must be between 0 and 45 degrees.")
    if config.position_tolerance_norm <= 0.0 or config.merge_gap_norm < 0.0:
        raise ValueError("Normalized tolerances must be positive.")
    for segment in observed.segments:
        values = (segment.start.x, segment.start.y, segment.end.x, segment.end.y, segment.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Observed segments must contain finite coordinates and confidence.")


def _normalize_and_classify(
    segments: Iterable[ObservedSegment],
    tank: TankGeometry,
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    normalized: list[_NormalizedSegment] = []
    for index, segment in enumerate(segments):
        start = _normalize_point(segment.start, tank)
        end = _normalize_point(segment.end, tank)
        length = math.hypot(end.x - start.x, end.y - start.y)
        if length < config.minimum_segment_length_norm:
            continue
        normalized.append(
            _NormalizedSegment(
                start=start,
                end=end,
                orientation=_classify_orientation(start, end, config.angle_tolerance_deg),
                confidence=min(1.0, max(0.0, float(segment.confidence))),
                source_id=segment.source_id or f"segment:{index}",
            )
        )
    return normalized


def _classify_orientation(start: LayoutPoint, end: LayoutPoint, tolerance_deg: float) -> str:
    dx = end.x - start.x
    dy = end.y - start.y
    angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
    if abs(angle - 90.0) <= tolerance_deg:
        return "vertical"
    if min(angle, abs(180.0 - angle)) <= tolerance_deg:
        return "horizontal"
    return "diagonal"


def _merge_collinear_segments(
    segments: list[_NormalizedSegment],
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    merged: list[_NormalizedSegment] = []
    for orientation in ("vertical", "horizontal"):
        axis_segments = [segment for segment in segments if segment.orientation == orientation]
        clusters: list[list[_NormalizedSegment]] = []
        for segment in sorted(axis_segments, key=lambda item: _axis_position(item, orientation)):
            position = _axis_position(segment, orientation)
            if not clusters or abs(position - np.mean([_axis_position(item, orientation) for item in clusters[-1]])) > config.position_tolerance_norm:
                clusters.append([segment])
            else:
                clusters[-1].append(segment)
        for cluster in clusters:
            position = float(np.average(
                [_axis_position(segment, orientation) for segment in cluster],
                weights=[max(segment.length, 1e-9) for segment in cluster],
            ))
            intervals = sorted(
                ((_primary_interval(segment, orientation), segment) for segment in cluster),
                key=lambda item: item[0],
            )
            current_start, current_end = intervals[0][0]
            support = [intervals[0][1]]
            for (start, end), segment in intervals[1:]:
                if start <= current_end + config.merge_gap_norm:
                    current_end = max(current_end, end)
                    support.append(segment)
                else:
                    merged.append(_axis_segment(position, current_start, current_end, orientation, support))
                    current_start, current_end = start, end
                    support = [segment]
            merged.append(_axis_segment(position, current_start, current_end, orientation, support))
    merged.extend(segment for segment in segments if segment.orientation == "diagonal")
    return merged


def _axis_segment(
    position: float,
    start: float,
    end: float,
    orientation: str,
    support: list[_NormalizedSegment],
) -> _NormalizedSegment:
    if orientation == "vertical":
        first, second = LayoutPoint(position, start), LayoutPoint(position, end)
    else:
        first, second = LayoutPoint(start, position), LayoutPoint(end, position)
    return _NormalizedSegment(
        first,
        second,
        orientation,
        float(np.mean([segment.confidence for segment in support])),
        ",".join(segment.source_id for segment in support[:12]),
    )


def _cluster_axis_positions(
    segments: list[_NormalizedSegment],
    orientation: str,
    tolerance: float,
) -> list[float]:
    values = sorted(_axis_position(segment, orientation) for segment in segments)
    if not values:
        return []
    clusters = [[values[0]]]
    for value in values[1:]:
        if abs(value - float(np.mean(clusters[-1]))) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def _infer_spacing(groups: list[list[float]], config: PredictorConfig) -> _SpacingEstimate | None:
    adjacent_differences: list[float] = []
    all_differences: list[float] = []
    for values in groups:
        ordered = sorted(values)
        adjacent_differences.extend(
            difference
            for difference in np.diff(ordered)
            if config.minimum_spacing_norm <= difference <= config.maximum_spacing_norm
        )
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                difference = second - first
                if difference >= config.minimum_spacing_norm:
                    all_differences.append(difference)
    if not adjacent_differences:
        return None

    clusters: list[list[float]] = []
    for difference in sorted(adjacent_differences):
        if not clusters or abs(difference - float(np.mean(clusters[-1]))) > config.position_tolerance_norm * 1.5:
            clusters.append([difference])
        else:
            clusters[-1].append(difference)

    best: _SpacingEstimate | None = None
    best_key: tuple[float, int, float] | None = None
    for cluster in clusters:
        spacing = float(np.median(cluster))
        explained = 0
        for difference in all_differences:
            multiple = max(1, round(difference / spacing))
            if abs(difference - multiple * spacing) <= config.position_tolerance_norm * 2.0:
                explained += 1
        support_fraction = explained / max(1, len(all_differences))
        direct_support = len(cluster)
        key = (support_fraction, direct_support, spacing)
        if best_key is None or key > best_key:
            best_key = key
            best = _SpacingEstimate(spacing, support_fraction, direct_support)
    return best


def _horizontal_position_groups(
    horizontal: list[_NormalizedSegment],
    vertical_positions: list[float],
    config: PredictorConfig,
) -> dict[int, list[float]]:
    boundaries = [-1.0, *vertical_positions, 1.0]
    groups: dict[int, list[float]] = defaultdict(list)
    for segment in horizontal:
        midpoint = (segment.start.x + segment.end.x) / 2.0
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            if left - config.position_tolerance_norm <= midpoint <= right + config.position_tolerance_norm:
                groups[index].append(_axis_position(segment, "horizontal"))
                break
    return {index: _cluster_values(values, config.position_tolerance_norm) for index, values in groups.items()}


def _detect_staggered_groups(
    groups: dict[int, list[float]],
    spacing: _SpacingEstimate | None,
    config: PredictorConfig,
) -> bool:
    if spacing is None or len(groups) < 2:
        return False
    phases: list[float] = []
    for values in groups.values():
        if values:
            phases.append(values[0] % spacing.spacing)
    return len(_cluster_values(phases, config.position_tolerance_norm)) > 1


def _select_pattern_family(
    vertical: list[_NormalizedSegment],
    horizontal: list[_NormalizedSegment],
    diagonal: list[_NormalizedSegment],
    staggered: bool,
) -> str:
    if diagonal and vertical and horizontal:
        return "grid_with_symmetric_transitions"
    if staggered and vertical and horizontal:
        return "staggered_plate_grid"
    if vertical and horizontal:
        return "orthogonal_plate_grid"
    if vertical or horizontal or diagonal:
        return "partial_structural_pattern"
    return "insufficient_observations"


def _predict_vertical_grid(
    observed: list[_NormalizedSegment],
    positions: list[float],
    spacing: _SpacingEstimate | None,
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    if spacing is None or len(positions) < config.minimum_pattern_support:
        return []
    predicted: list[_NormalizedSegment] = []
    for x in positions:
        y_limit = math.sqrt(max(0.0, 1.0 - x * x)) - config.boundary_margin_norm
        if y_limit <= 0.0:
            continue
        candidate = _NormalizedSegment(
            LayoutPoint(x, -y_limit),
            LayoutPoint(x, y_limit),
            "vertical",
            spacing.support_fraction,
            _support_ids(observed),
        )
        predicted.extend(_subtract_axis_observations(candidate, observed, config))
    return predicted


def _predict_horizontal_grid(
    observed: list[_NormalizedSegment],
    vertical_positions: list[float],
    groups: dict[int, list[float]],
    spacing: _SpacingEstimate | None,
    staggered: bool,
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    if spacing is None or not observed:
        return []
    boundaries = [-1.0, *vertical_positions, 1.0]
    median_length = float(np.median([segment.length for segment in observed]))
    observed_y_positions = _cluster_axis_positions(observed, "horizontal", config.position_tolerance_norm)
    minimum_y = min(observed_y_positions) - config.position_tolerance_norm
    maximum_y = max(observed_y_positions) + config.position_tolerance_norm
    predicted: list[_NormalizedSegment] = []

    if median_length >= 0.65 or not vertical_positions:
        for y in observed_y_positions:
            x_limit = math.sqrt(max(0.0, 1.0 - y * y)) - config.boundary_margin_norm
            if x_limit <= 0.0:
                continue
            candidate = _NormalizedSegment(
                LayoutPoint(-x_limit, y),
                LayoutPoint(x_limit, y),
                "horizontal",
                spacing.support_fraction,
                _support_ids(observed),
            )
            predicted.extend(_subtract_axis_observations(candidate, observed, config))
        return predicted

    for group_index, values in groups.items():
        if not values or group_index >= len(boundaries) - 1:
            continue
        local_spacing = _infer_spacing([values], config)
        if local_spacing is None:
            continue
        spacing_difference = abs(local_spacing.spacing - spacing.spacing)
        if spacing_difference > max(config.position_tolerance_norm * 2.0, spacing.spacing * 0.20):
            continue
        left, right = boundaries[group_index], boundaries[group_index + 1]
        phase = _lattice_phase(values, local_spacing.spacing)
        for y in _lattice_positions(phase, local_spacing.spacing, minimum_y, maximum_y):
            nearest_observed_y = min(observed_y_positions, key=lambda observed_y: abs(y - observed_y))
            if abs(y - nearest_observed_y) > config.position_tolerance_norm * 2.0:
                continue
            y = nearest_observed_y
            x_limit = math.sqrt(max(0.0, 1.0 - y * y)) - config.boundary_margin_norm
            start_x = max(left, -x_limit)
            end_x = min(right, x_limit)
            if end_x - start_x < config.minimum_segment_length_norm:
                continue
            confidence = min(spacing.support_fraction, local_spacing.support_fraction) * (0.9 if staggered else 0.75)
            candidate = _NormalizedSegment(
                LayoutPoint(start_x, y),
                LayoutPoint(end_x, y),
                "horizontal",
                confidence,
                _support_ids([segment for segment in observed if left <= (segment.start.x + segment.end.x) / 2.0 <= right]),
            )
            predicted.extend(_subtract_axis_observations(candidate, observed, config))
    return predicted


def _predict_diagonal_symmetry(
    observed: list[_NormalizedSegment],
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    if len(observed) < 2:
        return []
    predicted: list[_NormalizedSegment] = []
    for segment in observed:
        for mirror_x, mirror_y in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
            mirrored = _NormalizedSegment(
                LayoutPoint(segment.start.x * mirror_x, segment.start.y * mirror_y),
                LayoutPoint(segment.end.x * mirror_x, segment.end.y * mirror_y),
                "diagonal",
                min(0.75, segment.confidence),
                segment.source_id,
            )
            if not _segment_matches_any(mirrored, observed + predicted, config.symmetry_tolerance_norm):
                predicted.append(mirrored)
    return predicted


def _subtract_axis_observations(
    candidate: _NormalizedSegment,
    observed: list[_NormalizedSegment],
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    orientation = candidate.orientation
    intervals = [_primary_interval(candidate, orientation)]
    position = _axis_position(candidate, orientation)
    for segment in observed:
        if segment.orientation != orientation:
            continue
        if abs(_axis_position(segment, orientation) - position) > config.position_tolerance_norm:
            continue
        cut_start, cut_end = _primary_interval(segment, orientation)
        remaining: list[tuple[float, float]] = []
        for start, end in intervals:
            if cut_end <= start + config.position_tolerance_norm or cut_start >= end - config.position_tolerance_norm:
                remaining.append((start, end))
                continue
            if cut_start - start >= config.minimum_segment_length_norm:
                remaining.append((start, min(cut_start, end)))
            if end - cut_end >= config.minimum_segment_length_norm:
                remaining.append((max(cut_end, start), end))
        intervals = remaining
    return [
        _axis_segment_from_candidate(candidate, start, end)
        for start, end in intervals
        if end - start >= config.minimum_segment_length_norm
    ]


def _deduplicate_predictions(
    predictions: list[_NormalizedSegment],
    observed: list[_NormalizedSegment],
    config: PredictorConfig,
) -> list[_NormalizedSegment]:
    kept: list[_NormalizedSegment] = []
    for prediction in predictions:
        if _segment_matches_any(prediction, observed, config.position_tolerance_norm):
            continue
        if _segment_matches_any(prediction, kept, config.position_tolerance_norm):
            continue
        kept.append(prediction)
    return kept


def _segment_matches_any(
    candidate: _NormalizedSegment,
    segments: list[_NormalizedSegment],
    tolerance: float,
) -> bool:
    candidate_points = (candidate.start, candidate.end)
    for segment in segments:
        if segment.orientation != candidate.orientation:
            continue
        direct = max(_point_distance(candidate_points[0], segment.start), _point_distance(candidate_points[1], segment.end))
        reverse = max(_point_distance(candidate_points[0], segment.end), _point_distance(candidate_points[1], segment.start))
        if min(direct, reverse) <= tolerance:
            return True
    return False


def _clip_segment_to_annulus(
    segment: ObservedSegment,
    tank: TankGeometry,
    inner_radius: float,
    outer_radius: float,
) -> list[ObservedSegment]:
    start = np.array([segment.start.x - tank.center_x, segment.start.y - tank.center_y], dtype=float)
    end = np.array([segment.end.x - tank.center_x, segment.end.y - tank.center_y], dtype=float)
    direction = end - start
    breakpoints = [0.0, 1.0]
    for radius in (inner_radius, outer_radius):
        a = float(np.dot(direction, direction))
        if a <= 1e-20:
            continue
        b = 2.0 * float(np.dot(start, direction))
        c = float(np.dot(start, start) - radius * radius)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if 0.0 < value < 1.0:
                breakpoints.append(value)
    breakpoints = sorted(set(round(value, 12) for value in breakpoints))
    pieces: list[ObservedSegment] = []
    for piece_index, (first, second) in enumerate(zip(breakpoints, breakpoints[1:])):
        midpoint = start + direction * ((first + second) / 2.0)
        radius = float(np.linalg.norm(midpoint))
        if inner_radius - 1e-9 <= radius <= outer_radius + 1e-9:
            piece_start = start + direction * first
            piece_end = start + direction * second
            pieces.append(
                ObservedSegment(
                    LayoutPoint(piece_start[0] + tank.center_x, piece_start[1] + tank.center_y),
                    LayoutPoint(piece_end[0] + tank.center_x, piece_end[1] + tank.center_y),
                    source=segment.source,
                    confidence=segment.confidence,
                    source_id=f"{segment.source_id or 'segment'}:annulus:{piece_index}",
                    metadata=dict(segment.metadata),
                )
            )
    return pieces


def _observed_layout_segment(segment: _NormalizedSegment, tank: TankGeometry) -> LayoutSegment:
    return LayoutSegment(
        _world_point(segment.start, tank),
        _world_point(segment.end, tank),
        "observed",
        segment.confidence,
        _confidence_level(segment.confidence),
        segment.orientation,
        "direct_observation",
        [segment.source_id],
    )


def _predicted_layout_segment(
    segment: _NormalizedSegment,
    tank: TankGeometry,
    confidence: float,
    method: str,
) -> LayoutSegment:
    return LayoutSegment(
        _world_point(segment.start, tank),
        _world_point(segment.end, tank),
        "predicted",
        confidence,
        _confidence_level(confidence),
        segment.orientation,
        method,
        [item for item in segment.source_id.split(",") if item][:20],
    )


def _prediction_method_for(orientation: str, staggered: bool) -> str:
    if orientation == "diagonal":
        return "normalized_symmetry_completion"
    if orientation == "horizontal" and staggered:
        return "normalized_staggered_lattice_extrapolation"
    return "normalized_repeated_spacing_extrapolation"


def _layout_to_normalized(segment: LayoutSegment, tank: TankGeometry) -> _NormalizedSegment:
    return _NormalizedSegment(
        _normalize_point(segment.start, tank),
        _normalize_point(segment.end, tank),
        segment.orientation,
        segment.confidence,
        ",".join(segment.supporting_observations),
    )


def _normalize_point(point: LayoutPoint, tank: TankGeometry) -> LayoutPoint:
    return LayoutPoint((point.x - tank.center_x) / tank.radius, (point.y - tank.center_y) / tank.radius)


def _world_point(point: LayoutPoint, tank: TankGeometry) -> LayoutPoint:
    return LayoutPoint(tank.center_x + point.x * tank.radius, tank.center_y + point.y * tank.radius)


def _axis_position(segment: _NormalizedSegment, orientation: str) -> float:
    if orientation == "vertical":
        return (segment.start.x + segment.end.x) / 2.0
    return (segment.start.y + segment.end.y) / 2.0


def _primary_interval(segment: _NormalizedSegment, orientation: str) -> tuple[float, float]:
    values = (segment.start.y, segment.end.y) if orientation == "vertical" else (segment.start.x, segment.end.x)
    return min(values), max(values)


def _axis_segment_from_candidate(
    candidate: _NormalizedSegment,
    start: float,
    end: float,
) -> _NormalizedSegment:
    position = _axis_position(candidate, candidate.orientation)
    if candidate.orientation == "vertical":
        first, second = LayoutPoint(position, start), LayoutPoint(position, end)
    else:
        first, second = LayoutPoint(start, position), LayoutPoint(end, position)
    return _NormalizedSegment(first, second, candidate.orientation, candidate.confidence, candidate.source_id)


def _lattice_phase(values: list[float], spacing: float) -> float:
    phases = np.mod(np.asarray(values, dtype=float), spacing)
    angles = phases / spacing * 2.0 * math.pi
    mean_angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    return float((mean_angle % (2.0 * math.pi)) / (2.0 * math.pi) * spacing)


def _lattice_positions(
    phase: float,
    spacing: float,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> list[float]:
    start_index = math.ceil((minimum - phase) / spacing)
    end_index = math.floor((maximum - phase) / spacing)
    return [phase + index * spacing for index in range(start_index, end_index + 1)]


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - float(np.mean(clusters[-1]))) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def _symmetry_score(values: list[float], tolerance: float) -> float:
    if not values:
        return 0.0
    matches = sum(any(abs(other + value) <= tolerance for other in values) for value in values)
    return matches / len(values)


def _support_ids(segments: list[_NormalizedSegment]) -> str:
    return ",".join(segment.source_id for segment in segments[:20])


def _point_distance(first: LayoutPoint, second: LayoutPoint) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _match_positions(
    predicted: list[float],
    truth: list[float],
    tolerance: float,
) -> tuple[int, list[float]]:
    remaining = list(truth)
    errors: list[float] = []
    for value in predicted:
        if not remaining:
            break
        index = min(range(len(remaining)), key=lambda item: abs(remaining[item] - value))
        error = abs(remaining[index] - value)
        if error <= tolerance:
            errors.append(error)
            remaining.pop(index)
    return len(errors), errors


def print_prediction_summary(layout: CompletedTankLayout) -> None:
    print("Tank layout prediction summary")
    print(f"Pattern family: {layout.selected_pattern_family}")
    print(f"Overall confidence: {layout.overall_confidence:.1%}")
    print(f"Observed weld segments: {len(layout.observed_weld_segments)}")
    print(f"Predicted weld segments: {len(layout.predicted_weld_segments)}")
    parameters = layout.normalized_pattern_parameters
    print(f"Vertical spacing (normalized): {parameters.vertical_spacing_normalized}")
    print(f"Horizontal spacing (normalized): {parameters.horizontal_spacing_normalized}")
    for warning in layout.warnings:
        print(f"Warning: {warning}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict missing tank plate and weld geometry from partial observations.")
    parser.add_argument("dxf_path", help="DXF used as an observation source.")
    parser.add_argument(
        "--observation-inner-fraction",
        type=float,
        help="Mask input to an outer annulus before prediction, for validation or simulation.",
    )
    parser.add_argument("--save-json", default="predicted_tank_layout.json")
    parser.add_argument("--save-dxf", default="predicted_tank_layout.dxf")
    parser.add_argument("--save-plot", default="predicted_tank_layout_preview.png")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from dxf_importer import DxfImportError, import_dxf

    try:
        model = import_dxf(args.dxf_path)
        observations = observed_geometry_from_dxf(model)
        if args.observation_inner_fraction is not None:
            observations = mask_observations_to_annulus(observations, args.observation_inner_fraction)
        layout = predict_tank_layout(observations)
        json_path = save_layout_json(layout, args.save_json)
        dxf_path = export_layout_dxf(layout, args.save_dxf)
        plot_predicted_layout(layout, show=not args.no_plot, save_path=args.save_plot)
    except (DxfImportError, RuntimeError, ValueError) as exc:
        print(f"Tank layout prediction failed: {exc}")
        return 1
    print_prediction_summary(layout)
    print(f"JSON output: {json_path}")
    print(f"DXF output: {dxf_path}")
    print(f"Plot output: {args.save_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
