from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dxf_importer import DxfImportError, GeometryModel, import_dxf
from observation_region import ScanFootprintRegion, clip_segment_to_observed_region
from scan_profile import RainbowProfileConfig
from grid_predictor import (
    LayoutPoint,
    ObservedSegment,
    ObservedTankGeometry,
    TankGeometry,
    predict_tank_layout,
)


DEFAULT_MINIMUM_FAMILY_SCORE = 0.58
DEFAULT_MINIMUM_SCORE_MARGIN = 0.08
POSITION_TOLERANCE_NORMALIZED = 0.022
ANGLE_TOLERANCE_RAD = math.radians(8.0)


@dataclass(frozen=True)
class StructuralSegment:
    start: LayoutPoint
    end: LayoutPoint
    orientation: str
    source_id: str

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


@dataclass(frozen=True)
class PatternFeatures:
    vertical_positions: list[float]
    horizontal_positions: list[float]
    vertical_spacing: float | None
    horizontal_spacing: float | None
    diagonal_length_fraction: float
    symmetry_score: float


@dataclass(frozen=True)
class FamilyTemplate:
    name: str
    source_path: str
    segments: list[StructuralSegment]
    features: PatternFeatures


@dataclass(frozen=True)
class FamilyScore:
    family: str
    score: float
    observed_coverage: float
    expected_visible_coverage: float
    position_orientation_score: float
    feature_score: float
    rotation_deg: float
    reflected: bool
    spacing_refinement: float
    phase_offset_normalized_x: float
    phase_offset_normalized_y: float
    expected_visible_segments: int
    fitting_error_normalized: float | None


@dataclass(frozen=True)
class PredictedSegment:
    start: LayoutPoint
    end: LayoutPoint
    confidence: float
    confidence_level: str
    family: str
    template_source_id: str

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


@dataclass
class PredictionResult:
    status: str
    target: ObservedTankGeometry
    selected_family: str | None
    family_scores: list[FamilyScore]
    score_margin: float
    rotation_deg: float | None
    reflected: bool | None
    observed_segments: list[ObservedSegment]
    predicted_segments: list[PredictedSegment]
    warnings: list[str] = field(default_factory=list)
    completion_method: str | None = None
    completion_confidence: float | None = None


def load_reference_families(reference_dir: str | Path) -> list[FamilyTemplate]:
    """Construct normalized templates from complete reference DXFs."""
    directory = Path(reference_dir)
    if not directory.is_dir():
        raise ValueError(f"Reference directory not found: {directory}")
    paths = sorted(directory.glob("*.dxf"))
    if not paths:
        raise ValueError(f"No reference DXFs found in: {directory}")
    families: list[FamilyTemplate] = []
    for path in paths:
        model = import_dxf(path)
        tank = _tank_from_model(model)
        segments = _normalized_model_segments(model, tank)
        if not segments:
            raise ValueError(f"Reference DXF contains no usable structural lines: {path}")
        families.append(
            FamilyTemplate(
                name=f"{path.stem}_family",
                source_path=str(path),
                segments=segments,
                features=_pattern_features(segments),
            )
        )
    return families


def load_partial_dxf(filepath: str | Path) -> tuple[ObservedTankGeometry, ScanFootprintRegion | None]:
    """Load only sanitized observation layers from a partial DXF."""
    model = import_dxf(filepath)
    tank = _tank_from_model(model)
    observed_layers = {"OBSERVED_CIRCULAR_SCAN", "OBSERVED_GEOMETRY"}
    segments: list[ObservedSegment] = []
    for index, line in enumerate(model.line_segments):
        if line.layer.upper() not in observed_layers:
            continue
        segments.append(
            ObservedSegment(
                start=LayoutPoint(float(line.start.x), float(line.start.y)),
                end=LayoutPoint(float(line.end.x), float(line.end.y)),
                source="observed_circular_scan",
                source_id=f"partial:{line.handle or index}",
                metadata={"layer": line.layer},
            )
        )
    footprint_polygons = []
    for polyline in model.polylines:
        if polyline.layer.upper() != "SCAN_FOOTPRINTS" or len(polyline.vertices) < 3:
            continue
        footprint_polygons.append(
            np.asarray([(vertex.point.x, vertex.point.y) for vertex in polyline.vertices], dtype=float)
        )
    region = _region_from_polygons(footprint_polygons) if footprint_polygons else None
    geometry = ObservedTankGeometry(
        tank=tank,
        segments=segments,
        arcs=[],
        observation_region=region,
        units="m",
        source_metadata={"source_type": "partial_dxf", "source_path": str(filepath)},
    )
    if not geometry.segments:
        raise ValueError("Partial DXF contains no OBSERVED_CIRCULAR_SCAN or OBSERVED_GEOMETRY lines.")
    return geometry, region


def load_runtime_observations(filepath: str | Path) -> tuple[ObservedTankGeometry, ScanFootprintRegion | None]:
    path = Path(filepath)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read runtime observations JSON: {path}") from exc
    tank_data = data.get("tank", data)
    center = tank_data.get("center", {})
    tank = TankGeometry(
        center_x=float(tank_data.get("center_x", center.get("x"))),
        center_y=float(tank_data.get("center_y", center.get("y"))),
        radius=float(tank_data.get("radius", tank_data.get("radius_m"))),
    )
    raw_segments = data.get("segments", data.get("observed_segments", []))
    segments = []
    for index, segment in enumerate(raw_segments):
        start = segment["start"]
        end = segment["end"]
        segments.append(
            ObservedSegment(
                start=LayoutPoint(float(start[0] if isinstance(start, list) else start["x"]), float(start[1] if isinstance(start, list) else start["y"])),
                end=LayoutPoint(float(end[0] if isinstance(end, list) else end["x"]), float(end[1] if isinstance(end, list) else end["y"])),
                source=str(segment.get("source", "runtime_observation")),
                confidence=float(segment.get("confidence", 1.0)),
                source_id=str(segment.get("source_id", f"runtime:{index}")),
                metadata=dict(segment.get("metadata", {})),
            )
        )
    polygons = [np.asarray(polygon, dtype=float) for polygon in data.get("footprint_polygons", [])]
    region = _region_from_polygons(polygons) if polygons else None
    return (
        ObservedTankGeometry(
            tank=tank,
            segments=segments,
            observation_region=region,
            units=str(data.get("units", "m")),
            source_metadata={"source_type": "runtime_json", "source_path": str(path)},
        ),
        region,
    )


def predict_from_observations(
    target: ObservedTankGeometry,
    families: list[FamilyTemplate],
    observed_region: ScanFootprintRegion | None,
    *,
    minimum_score: float = DEFAULT_MINIMUM_FAMILY_SCORE,
    minimum_margin: float = DEFAULT_MINIMUM_SCORE_MARGIN,
    allow_reflection: bool = False,
) -> PredictionResult:
    """Classify a normalized family, fit it, and complete missing geometry."""
    if not target.segments:
        return PredictionResult(
            status="ambiguous",
            target=target,
            selected_family=None,
            family_scores=[],
            score_margin=0.0,
            rotation_deg=None,
            reflected=None,
            observed_segments=[],
            predicted_segments=[],
            warnings=["No observed weld segments were supplied."],
        )
    target_segments = _normalized_observed_segments(target)
    insufficient_evidence = len(target_segments) < 3 or sum(segment.length for segment in target_segments) < 0.12
    base_rotation = _dominant_rotation(target_segments)
    scores: list[FamilyScore] = []
    alignments: dict[str, tuple[float, bool, float, float, float]] = {}
    for family in families:
        best_score: FamilyScore | None = None
        for reflected in ((False, True) if allow_reflection else (False,)):
            score = _score_family(target, target_segments, observed_region, family, base_rotation, reflected)
            if best_score is None or score.score > best_score.score:
                best_score = score
        if best_score is not None:
            scores.append(best_score)
            alignments[family.name] = (
                math.radians(best_score.rotation_deg),
                best_score.reflected,
                best_score.spacing_refinement,
                best_score.phase_offset_normalized_x,
                best_score.phase_offset_normalized_y,
            )
    scores.sort(key=lambda item: item.score, reverse=True)
    if not scores:
        raise ValueError("No reference families could be scored.")
    margin = scores[0].score - scores[1].score if len(scores) > 1 else scores[0].score
    warnings: list[str] = []
    if observed_region is None:
        warnings.append("No scan footprints were supplied; hidden expected geometry was not penalized.")
    if insufficient_evidence:
        warnings.append("Too few observed fragments or too little normalized weld length for family completion.")
    if scores[0].score < minimum_score:
        warnings.append(
            f"Best family score {scores[0].score:.3f} is below minimum {minimum_score:.3f}."
        )
    if margin < minimum_margin:
        warnings.append(f"Family score margin {margin:.3f} is below minimum {minimum_margin:.3f}.")
    if insufficient_evidence or scores[0].score < minimum_score or margin < minimum_margin:
        return _predict_structural_grid_fallback(target, scores, margin, warnings)

    selected = next(family for family in families if family.name == scores[0].family)
    rotation, reflected, spacing_refinement, phase_x, phase_y = alignments[selected.name]
    completed_template = _template_to_world(
        selected.segments,
        target.tank,
        rotation,
        reflected,
        spacing_refinement=spacing_refinement,
        phase_offset=(phase_x, phase_y),
    )
    predicted = _subtract_observed_geometry(
        completed_template,
        target.segments,
        target.tank,
        selected.name,
        scores[0].score,
    )
    return PredictionResult(
        status="completed",
        target=target,
        selected_family=selected.name,
        family_scores=scores,
        score_margin=margin,
        rotation_deg=math.degrees(rotation),
        reflected=reflected,
        observed_segments=list(target.segments),
        predicted_segments=predicted,
        warnings=warnings,
        completion_method="family_template",
        completion_confidence=scores[0].score,
    )


def _predict_structural_grid_fallback(
    target: ObservedTankGeometry,
    scores: list[FamilyScore],
    margin: float,
    warnings: list[str],
) -> PredictionResult:
    """Complete an ambiguous family match from repeated local grid evidence.

    The structural predictor is deliberately reached only after template-family
    selection declines to choose a confident match.  Its output is adapted into
    this module's normal result type so callers, reports, and DXF writers have
    one completion interface.
    """
    layout = predict_tank_layout(target)
    fallback_warnings = [
        *warnings,
        "Family matching was ambiguous; completed from repeated structural-grid evidence.",
        *layout.warnings,
    ]
    if not layout.predicted_weld_segments:
        return PredictionResult(
            status="ambiguous",
            target=target,
            selected_family=None,
            family_scores=scores,
            score_margin=margin,
            rotation_deg=None,
            reflected=None,
            observed_segments=list(target.segments),
            predicted_segments=[],
            warnings=fallback_warnings,
        )

    predicted = [
        PredictedSegment(
            start=segment.start,
            end=segment.end,
            confidence=segment.confidence,
            confidence_level=segment.confidence_level,
            family="structural_grid_fallback",
            template_source_id=f"{segment.prediction_method}:{index}",
        )
        for index, segment in enumerate(layout.predicted_weld_segments)
    ]
    return PredictionResult(
        status="completed",
        target=target,
        selected_family=None,
        family_scores=scores,
        score_margin=margin,
        rotation_deg=None,
        reflected=None,
        observed_segments=list(target.segments),
        predicted_segments=predicted,
        warnings=fallback_warnings,
        completion_method="structural_grid_fallback",
        completion_confidence=layout.overall_confidence,
    )


def save_completed_dxf(result: PredictionResult, filepath: str | Path) -> Path:
    if result.status != "completed":
        raise ValueError("A completed DXF is not written for an ambiguous prediction.")
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError("DXF output requires ezdxf.") from exc
    path = Path(filepath)
    doc = ezdxf.new("R2010")
    doc.units = 6
    for layer_name, color in (
        ("TANK_BOUNDARY", 7),
        ("OBSERVED_GEOMETRY", 3),
        ("PREDICTED_GEOMETRY_HIGH", 5),
        ("PREDICTED_GEOMETRY_MEDIUM", 2),
        ("PREDICTED_GEOMETRY_LOW", 1),
    ):
        doc.layers.add(layer_name, color=color)
    modelspace = doc.modelspace()
    tank = result.target.tank
    modelspace.add_circle((tank.center_x, tank.center_y), tank.radius, dxfattribs={"layer": "TANK_BOUNDARY"})
    for segment in result.observed_segments:
        modelspace.add_line(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
            dxfattribs={"layer": "OBSERVED_GEOMETRY"},
        )
    for segment in result.predicted_segments:
        layer = f"PREDICTED_GEOMETRY_{segment.confidence_level.upper()}"
        entity = modelspace.add_line(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
            dxfattribs={"layer": layer},
        )
        entity.set_xdata(
            "DXF_PREDICTOR",
            [(1000, segment.family), (1040, segment.confidence), (1000, segment.template_source_id)],
        ) if _ensure_appid(doc, "DXF_PREDICTOR") else None
    doc.saveas(path)
    return path


def save_prediction_report(
    result: PredictionResult,
    filepath: str | Path,
    *,
    output_paths: dict[str, str | None] | None = None,
) -> Path:
    observed_length = sum(_observed_length(segment) for segment in result.observed_segments)
    predicted_length = sum(segment.length for segment in result.predicted_segments)
    selected_score = next(
        (score for score in result.family_scores if score.family == result.selected_family),
        None,
    )
    data = {
        "status": result.status,
        "selected_family": result.selected_family,
        "family_scores": [asdict(score) for score in result.family_scores],
        "confidence": (
            result.completion_confidence
            if result.completion_confidence is not None
            else (result.family_scores[0].score if result.family_scores else 0.0)
        ),
        "completion_method": result.completion_method,
        "score_margin": result.score_margin,
        "target_center_m": {"x": result.target.tank.center_x, "y": result.target.tank.center_y},
        "target_radius_m": result.target.tank.radius,
        "scale_fit": result.target.tank.radius,
        "rotation_fit_deg": result.rotation_deg,
        "reflected": result.reflected,
        "phase_offset_normalized": {
            "x": selected_score.phase_offset_normalized_x if selected_score is not None else None,
            "y": selected_score.phase_offset_normalized_y if selected_score is not None else None,
        },
        "spacing_refinement": selected_score.spacing_refinement if selected_score is not None else None,
        "observed_line_count": len(result.observed_segments),
        "observed_length_m": observed_length,
        "predicted_line_count": len(result.predicted_segments),
        "predicted_length_m": predicted_length,
        "warnings": result.warnings,
        "output_paths": output_paths or {},
    }
    path = Path(filepath)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def plot_prediction(
    result: PredictionResult,
    filepath: str | Path,
    *,
    observed_region: ScanFootprintRegion | None = None,
    hidden_ground_truth: ObservedTankGeometry | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(10, 10))
    tank = result.target.tank
    ax.add_patch(Circle((tank.center_x, tank.center_y), tank.radius, fill=False, color="#111827", linewidth=2.0))
    if observed_region is not None:
        for polygon in observed_region.polygons:
            closed = np.vstack((polygon, polygon[0])) if not np.allclose(polygon[0], polygon[-1]) else polygon
            ax.fill(closed[:, 0], closed[:, 1], color="#60a5fa", alpha=0.018)
            ax.plot(closed[:, 0], closed[:, 1], color="#2563eb", linewidth=0.3, alpha=0.16)
    if hidden_ground_truth is not None:
        for segment in hidden_ground_truth.segments:
            ax.plot(
                [segment.start.x, segment.end.x],
                [segment.start.y, segment.end.y],
                color="#9ca3af",
                linestyle=":",
                linewidth=0.6,
                alpha=0.4,
            )
    for segment in result.observed_segments:
        ax.plot(
            [segment.start.x, segment.end.x],
            [segment.start.y, segment.end.y],
            color="#047857",
            linewidth=1.4,
        )
    colors = {"high": "#7c3aed", "medium": "#d97706", "low": "#dc2626"}
    for segment in result.predicted_segments:
        ax.plot(
            [segment.start.x, segment.end.x],
            [segment.start.y, segment.end.y],
            color=colors[segment.confidence_level],
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
        )
    margin = tank.radius * 0.05
    ax.set_xlim(tank.center_x - tank.radius - margin, tank.center_x + tank.radius + margin)
    ax.set_ylim(tank.center_y - tank.radius - margin, tank.center_y + tank.radius + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(f"x ({result.target.units})")
    ax.set_ylabel(f"y ({result.target.units})")
    score_text = ", ".join(f"{score.family}: {score.score:.2f}" for score in result.family_scores)
    ax.set_title(f"DXF family prediction: {result.status}\n{score_text}")
    handles = [
        Line2D([], [], color="#047857", label="Observed geometry"),
        Line2D([], [], color="#7c3aed", linestyle="--", label="Predicted geometry"),
    ]
    if observed_region is not None:
        handles.insert(0, Line2D([], [], color="#2563eb", alpha=0.4, label="Scan footprints"))
    if hidden_ground_truth is not None:
        handles.append(Line2D([], [], color="#9ca3af", linestyle=":", label="Hidden ground truth"))
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    path = Path(filepath)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def reconstruction_metrics(
    result: PredictionResult,
    ground_truth: ObservedTankGeometry,
) -> dict[str, Any]:
    """Geometry-only orientation precision/recall for validation reporting."""
    completed = [
        StructuralSegment(segment.start, segment.end, _orientation(segment.start, segment.end), segment.source_id or "observed")
        for segment in result.observed_segments
    ] + [
        StructuralSegment(segment.start, segment.end, _orientation(segment.start, segment.end), segment.template_source_id)
        for segment in result.predicted_segments
    ]
    truth = [
        StructuralSegment(segment.start, segment.end, _orientation(segment.start, segment.end), segment.source_id or "truth")
        for segment in ground_truth.segments
    ]
    metrics: dict[str, Any] = {}
    for orientation in ("vertical", "horizontal", "diagonal"):
        predicted_positions = _orientation_signatures(completed, ground_truth.tank, orientation)
        truth_positions = _orientation_signatures(truth, ground_truth.tank, orientation)
        matched, errors = _match_scalar_positions(predicted_positions, truth_positions, POSITION_TOLERANCE_NORMALIZED)
        precision = matched / len(predicted_positions) if predicted_positions else (1.0 if not truth_positions else 0.0)
        recall = matched / len(truth_positions) if truth_positions else 1.0
        metrics[orientation] = {
            "precision": precision,
            "recall": recall,
            "matched": matched,
            "false_lines": max(0, len(predicted_positions) - matched),
            "missed_lines": max(0, len(truth_positions) - matched),
            "mean_normalized_position_error": float(np.mean(errors)) if errors else None,
            "mean_world_position_error": float(np.mean(errors)) * ground_truth.tank.radius if errors else None,
        }
    return metrics


def _score_family(
    target: ObservedTankGeometry,
    target_normalized: list[StructuralSegment],
    region: ScanFootprintRegion | None,
    family: FamilyTemplate,
    rotation: float,
    reflected: bool,
) -> FamilyScore:
    spacing_refinement = _estimate_spacing_refinement(
        target_normalized,
        family.segments,
        rotation,
        reflected,
    )
    phase_x, phase_y = _estimate_phase_offset(
        target_normalized,
        family.segments,
        rotation,
        reflected,
        spacing_refinement,
    )
    expected_world = _template_to_world(
        family.segments,
        target.tank,
        rotation,
        reflected,
        spacing_refinement=spacing_refinement,
        phase_offset=(phase_x, phase_y),
    )
    if region is not None:
        expected_visible = _sample_visible_structural_segments(expected_world, region, target.tank)
    else:
        expected_visible = expected_world
    expected_normalized = _world_to_normalized_segments(expected_visible, target.tank)

    observed_scores, observed_errors = _best_match_scores(target_normalized, expected_normalized)
    expected_scores, _ = _best_match_scores(expected_normalized, target_normalized)
    observed_lengths = np.asarray([segment.length for segment in target_normalized], dtype=float)
    expected_lengths = np.asarray([segment.length for segment in expected_normalized], dtype=float)
    observed_coverage = _weighted_mean(observed_scores, observed_lengths)
    if region is None:
        expected_coverage = observed_coverage
    else:
        expected_coverage = _weighted_mean(expected_scores, expected_lengths)
    position_score = float(np.mean(observed_scores)) if observed_scores else 0.0
    visible_features = _pattern_features(expected_normalized)
    target_features = _pattern_features(target_normalized)
    feature_score = _feature_similarity(target_features, visible_features)
    score = 0.45 * observed_coverage + 0.30 * expected_coverage + 0.15 * position_score + 0.10 * feature_score
    fitting_error = float(np.mean(observed_errors)) if observed_errors else None
    return FamilyScore(
        family=family.name,
        score=float(score),
        observed_coverage=observed_coverage,
        expected_visible_coverage=expected_coverage,
        position_orientation_score=position_score,
        feature_score=feature_score,
        rotation_deg=math.degrees(rotation),
        reflected=reflected,
        spacing_refinement=spacing_refinement,
        phase_offset_normalized_x=phase_x,
        phase_offset_normalized_y=phase_y,
        expected_visible_segments=len(expected_visible),
        fitting_error_normalized=fitting_error,
    )


def _best_match_scores(
    sources: list[StructuralSegment],
    candidates: list[StructuralSegment],
) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    errors: list[float] = []
    for source in sources:
        best_score = 0.0
        best_error = 1.0
        for candidate in candidates:
            score, position_error = _segment_match_score(source, candidate)
            if score > best_score:
                best_score = score
                best_error = position_error
        scores.append(best_score)
        if best_score > 0.0:
            errors.append(best_error)
    return scores, errors


def _segment_match_score(first: StructuralSegment, second: StructuralSegment) -> tuple[float, float]:
    first_angle = _segment_angle(first.start, first.end)
    second_angle = _segment_angle(second.start, second.end)
    angle_error = _undirected_angle_difference(first_angle, second_angle)
    if angle_error > ANGLE_TOLERANCE_RAD:
        return 0.0, 1.0
    direction = np.array([second.end.x - second.start.x, second.end.y - second.start.y], dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return 0.0, 1.0
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    first_midpoint = np.array([(first.start.x + first.end.x) / 2.0, (first.start.y + first.end.y) / 2.0])
    second_start = np.array([second.start.x, second.start.y])
    position_error = abs(float(np.dot(first_midpoint - second_start, normal)))
    if position_error > POSITION_TOLERANCE_NORMALIZED:
        return 0.0, position_error
    projections = [
        float(np.dot(np.array([point.x, point.y]) - second_start, unit))
        for point in (first.start, first.end)
    ]
    overlap = max(0.0, min(max(projections), length) - max(min(projections), 0.0))
    if overlap <= 1e-5:
        endpoint_gap = min(abs(value) for value in projections) if max(projections) < 0.0 else min(abs(value - length) for value in projections)
        if endpoint_gap > POSITION_TOLERANCE_NORMALIZED:
            return 0.0, position_error
    position_component = max(0.0, 1.0 - position_error / POSITION_TOLERANCE_NORMALIZED)
    angle_component = max(0.0, 1.0 - angle_error / ANGLE_TOLERANCE_RAD)
    return 0.75 * position_component + 0.25 * angle_component, position_error


def _template_to_world(
    segments: list[StructuralSegment],
    tank: TankGeometry,
    rotation: float,
    reflected: bool,
    spacing_refinement: float = 1.0,
    phase_offset: tuple[float, float] = (0.0, 0.0),
) -> list[StructuralSegment]:
    return [
        StructuralSegment(
            start=_normalized_to_world(
                segment.start,
                tank,
                rotation,
                reflected,
                spacing_refinement,
                phase_offset,
            ),
            end=_normalized_to_world(
                segment.end,
                tank,
                rotation,
                reflected,
                spacing_refinement,
                phase_offset,
            ),
            orientation=segment.orientation,
            source_id=segment.source_id,
        )
        for segment in segments
    ]


def _normalized_to_world(
    point: LayoutPoint,
    tank: TankGeometry,
    rotation: float,
    reflected: bool,
    spacing_refinement: float = 1.0,
    phase_offset: tuple[float, float] = (0.0, 0.0),
) -> LayoutPoint:
    scaled_x = point.x * spacing_refinement
    scaled_y = point.y * spacing_refinement
    x = -scaled_x if reflected else scaled_x
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    rotated_x = x * cos_r - scaled_y * sin_r
    rotated_y = x * sin_r + scaled_y * cos_r
    return LayoutPoint(
        tank.center_x + tank.radius * (rotated_x + phase_offset[0]),
        tank.center_y + tank.radius * (rotated_y + phase_offset[1]),
    )


def _clip_structural_segments(
    segments: list[StructuralSegment],
    region: ScanFootprintRegion,
) -> list[StructuralSegment]:
    clipped: list[StructuralSegment] = []
    for segment in segments:
        pieces = clip_segment_to_observed_region(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
            region,
        )
        for index, (start, end) in enumerate(pieces):
            first = LayoutPoint(float(start[0]), float(start[1]))
            second = LayoutPoint(float(end[0]), float(end[1]))
            clipped.append(
                StructuralSegment(first, second, _orientation(first, second), f"{segment.source_id}:visible:{index}")
            )
    return clipped


def _sample_visible_structural_segments(
    segments: list[StructuralSegment],
    region: ScanFootprintRegion,
    tank: TankGeometry,
) -> list[StructuralSegment]:
    """Batch-sample candidate visibility more finely than the matching tolerance."""
    sample_spacing = max(tank.radius * 0.006, 0.002)
    point_batches: list[np.ndarray] = []
    batch_sizes: list[int] = []
    for segment in segments:
        count = max(3, int(math.ceil(segment.length / sample_spacing)) + 1)
        fractions = np.linspace(0.0, 1.0, count)
        start = np.array([segment.start.x, segment.start.y], dtype=float)
        end = np.array([segment.end.x, segment.end.y], dtype=float)
        point_batches.append(start + fractions[:, None] * (end - start))
        batch_sizes.append(count)
    if not point_batches:
        return []
    all_points = np.vstack(point_batches)
    visible_mask = _points_in_region_fast(all_points, region)
    visible_segments: list[StructuralSegment] = []
    offset = 0
    for segment, points, count in zip(segments, point_batches, batch_sizes):
        mask = visible_mask[offset:offset + count]
        offset += count
        run_start: int | None = None
        for index, visible in enumerate(np.append(mask, False)):
            if visible and run_start is None:
                run_start = index
            elif not visible and run_start is not None:
                run_end = index - 1
                if run_end > run_start:
                    first = LayoutPoint(float(points[run_start, 0]), float(points[run_start, 1]))
                    second = LayoutPoint(float(points[run_end, 0]), float(points[run_end, 1]))
                    visible_segments.append(
                        StructuralSegment(
                            first,
                            second,
                            _orientation(first, second),
                            f"{segment.source_id}:visible:{len(visible_segments)}",
                        )
                    )
                run_start = None
    return visible_segments


def _points_in_region_fast(points: np.ndarray, region: ScanFootprintRegion) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for polygon, bounds in zip(region.polygons, region.polygon_bounds):
        candidates = (
            ~inside
            & (points[:, 0] >= bounds[0])
            & (points[:, 0] <= bounds[2])
            & (points[:, 1] >= bounds[1])
            & (points[:, 1] <= bounds[3])
        )
        if not np.any(candidates):
            continue
        indices = np.flatnonzero(candidates)
        inside[indices] = _points_in_polygon_fast(points[indices], polygon)
    return inside


def _points_in_polygon_fast(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
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
    return inside


def _subtract_observed_geometry(
    template_segments: list[StructuralSegment],
    observed: list[ObservedSegment],
    tank: TankGeometry,
    family: str,
    confidence: float,
) -> list[PredictedSegment]:
    predicted: list[PredictedSegment] = []
    distance_tolerance = tank.radius * POSITION_TOLERANCE_NORMALIZED
    level = "high" if confidence >= 0.82 else "medium" if confidence >= 0.68 else "low"
    for template in template_segments:
        start = np.array([template.start.x, template.start.y], dtype=float)
        end = np.array([template.end.x, template.end.y], dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        covered: list[tuple[float, float]] = []
        template_angle = math.atan2(direction[1], direction[0])
        for segment in observed:
            observed_start = np.array([segment.start.x, segment.start.y], dtype=float)
            observed_end = np.array([segment.end.x, segment.end.y], dtype=float)
            observed_angle = math.atan2(observed_end[1] - observed_start[1], observed_end[0] - observed_start[0])
            if _undirected_angle_difference(template_angle, observed_angle) > ANGLE_TOLERANCE_RAD:
                continue
            midpoint = (observed_start + observed_end) / 2.0
            if abs(float(np.dot(midpoint - start, normal))) > distance_tolerance:
                continue
            first = float(np.dot(observed_start - start, unit))
            second = float(np.dot(observed_end - start, unit))
            interval = (max(0.0, min(first, second)), min(length, max(first, second)))
            if interval[1] > interval[0]:
                covered.append(interval)
        uncovered = _subtract_intervals((0.0, length), covered, distance_tolerance * 0.25)
        for first, second in uncovered:
            if second - first <= distance_tolerance * 0.2:
                continue
            predicted.append(
                PredictedSegment(
                    start=LayoutPoint(*(start + unit * first)),
                    end=LayoutPoint(*(start + unit * second)),
                    confidence=confidence,
                    confidence_level=level,
                    family=family,
                    template_source_id=template.source_id,
                )
            )
    return predicted


def _subtract_intervals(
    whole: tuple[float, float],
    covered: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    if not covered:
        return [whole]
    ordered = sorted(covered)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    result = []
    cursor = whole[0]
    for start, end in merged:
        if start > cursor + tolerance:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < whole[1] - tolerance:
        result.append((cursor, whole[1]))
    return result


def _tank_from_model(model: GeometryModel) -> TankGeometry:
    boundary_circles = [circle for circle in model.circles if circle.layer.upper() == "TANK_BOUNDARY"]
    circles = boundary_circles or model.circles
    if circles:
        circle = max(circles, key=lambda item: item.radius)
        return TankGeometry(float(circle.center.x), float(circle.center.y), float(circle.radius))
    if model.bounds is None:
        raise ValueError("Tank boundary or geometry bounds are required.")
    return TankGeometry(
        (model.bounds.min_x + model.bounds.max_x) / 2.0,
        (model.bounds.min_y + model.bounds.max_y) / 2.0,
        max(model.bounds.width, model.bounds.height) / 2.0,
    )


def _normalized_model_segments(model: GeometryModel, tank: TankGeometry) -> list[StructuralSegment]:
    segments = []
    for index, line in enumerate(model.line_segments):
        start = _normalize(LayoutPoint(line.start.x, line.start.y), tank)
        end = _normalize(LayoutPoint(line.end.x, line.end.y), tank)
        if _point_distance(start, end) > 1e-6:
            segments.append(StructuralSegment(start, end, _orientation(start, end), f"line:{line.handle or index}"))
    for polyline_index, polyline in enumerate(model.polylines):
        if polyline.layer.upper() == "SCAN_FOOTPRINTS":
            continue
        points = [vertex.point for vertex in polyline.vertices]
        pairs = list(zip(points, points[1:]))
        if polyline.closed and len(points) > 1:
            pairs.append((points[-1], points[0]))
        for edge_index, (first, second) in enumerate(pairs):
            start = _normalize(LayoutPoint(first.x, first.y), tank)
            end = _normalize(LayoutPoint(second.x, second.y), tank)
            if _point_distance(start, end) > 1e-6:
                segments.append(
                    StructuralSegment(start, end, _orientation(start, end), f"poly:{polyline_index}:{edge_index}")
                )
    return segments


def _normalized_observed_segments(target: ObservedTankGeometry) -> list[StructuralSegment]:
    return [
        StructuralSegment(
            _normalize(segment.start, target.tank),
            _normalize(segment.end, target.tank),
            _orientation(segment.start, segment.end),
            segment.source_id or f"observed:{index}",
        )
        for index, segment in enumerate(target.segments)
        if _observed_length(segment) / target.tank.radius > 1e-6
    ]


def _world_to_normalized_segments(
    segments: list[StructuralSegment],
    tank: TankGeometry,
) -> list[StructuralSegment]:
    return [
        StructuralSegment(
            _normalize(segment.start, tank),
            _normalize(segment.end, tank),
            segment.orientation,
            segment.source_id,
        )
        for segment in segments
    ]


def _pattern_features(segments: list[StructuralSegment]) -> PatternFeatures:
    vertical = [_line_position(segment, "vertical") for segment in segments if segment.orientation == "vertical"]
    horizontal = [_line_position(segment, "horizontal") for segment in segments if segment.orientation == "horizontal"]
    vertical_positions = _cluster_values(vertical, 0.012)
    horizontal_positions = _cluster_values(horizontal, 0.012)
    total_length = sum(segment.length for segment in segments)
    diagonal_length = sum(segment.length for segment in segments if segment.orientation == "diagonal")
    all_positions = vertical_positions + horizontal_positions
    symmetry = (
        sum(any(abs(other + value) <= 0.025 for other in all_positions) for value in all_positions) / len(all_positions)
        if all_positions else 0.0
    )
    return PatternFeatures(
        vertical_positions=vertical_positions,
        horizontal_positions=horizontal_positions,
        vertical_spacing=_dominant_spacing(vertical_positions),
        horizontal_spacing=_dominant_spacing(horizontal_positions),
        diagonal_length_fraction=diagonal_length / total_length if total_length else 0.0,
        symmetry_score=symmetry,
    )


def _feature_similarity(first: PatternFeatures, second: PatternFeatures) -> float:
    components = [
        _optional_relative_similarity(first.vertical_spacing, second.vertical_spacing),
        _optional_relative_similarity(first.horizontal_spacing, second.horizontal_spacing),
        max(0.0, 1.0 - abs(first.diagonal_length_fraction - second.diagonal_length_fraction) / 0.35),
        max(0.0, 1.0 - abs(first.symmetry_score - second.symmetry_score)),
    ]
    return float(np.mean(components))


def _dominant_rotation(segments: list[StructuralSegment]) -> float:
    axis_segments = [segment for segment in segments if segment.orientation in {"vertical", "horizontal"}]
    samples = axis_segments or segments
    if not samples:
        return 0.0
    vector = sum(
        segment.length * np.exp(1j * 4.0 * _segment_angle(segment.start, segment.end))
        for segment in samples
    )
    rotation = float(np.angle(vector) / 4.0)
    while rotation > math.pi / 4.0:
        rotation -= math.pi / 2.0
    while rotation < -math.pi / 4.0:
        rotation += math.pi / 2.0
    return rotation


def _estimate_phase_offset(
    observed: list[StructuralSegment],
    template: list[StructuralSegment],
    rotation: float,
    reflected: bool,
    spacing_refinement: float,
) -> tuple[float, float]:
    local_observed = [_inverse_align_segment(segment, rotation, reflected) for segment in observed]
    observed_vertical = [_line_position(segment, "vertical") for segment in local_observed if segment.orientation == "vertical"]
    observed_horizontal = [_line_position(segment, "horizontal") for segment in local_observed if segment.orientation == "horizontal"]
    template_vertical = [
        _line_position(segment, "vertical") * spacing_refinement
        for segment in template
        if segment.orientation == "vertical"
    ]
    template_horizontal = [
        _line_position(segment, "horizontal") * spacing_refinement
        for segment in template
        if segment.orientation == "horizontal"
    ]

    def median_residual(values: list[float], candidates: list[float]) -> float:
        if len(values) < 2 or not candidates:
            return 0.0
        residuals = [value - min(candidates, key=lambda candidate: abs(value - candidate)) for value in values]
        return float(np.clip(np.median(residuals), -0.025, 0.025))

    local_x = median_residual(observed_vertical, template_vertical)
    local_y = median_residual(observed_horizontal, template_horizontal)
    reflected_x = -local_x if reflected else local_x
    return (
        reflected_x * math.cos(rotation) - local_y * math.sin(rotation),
        reflected_x * math.sin(rotation) + local_y * math.cos(rotation),
    )


def _estimate_spacing_refinement(
    observed: list[StructuralSegment],
    template: list[StructuralSegment],
    rotation: float,
    reflected: bool,
) -> float:
    """Fit a deliberately small common row/column spacing adjustment."""
    local_observed = [_inverse_align_segment(segment, rotation, reflected) for segment in observed]
    observed_positions = {
        orientation: _cluster_values(
            [
                _line_position(segment, orientation)
                for segment in local_observed
                if segment.orientation == orientation
            ],
            0.012,
        )
        for orientation in ("vertical", "horizontal")
    }
    template_positions = {
        orientation: _cluster_values(
            [
                _line_position(segment, orientation)
                for segment in template
                if segment.orientation == orientation
            ],
            0.012,
        )
        for orientation in ("vertical", "horizontal")
    }

    best_factor = 1.0
    best_error = math.inf
    for factor in np.linspace(0.98, 1.02, 17):
        residuals: list[float] = []
        for orientation in ("vertical", "horizontal"):
            values = observed_positions[orientation]
            candidates = [position * float(factor) for position in template_positions[orientation]]
            if len(values) < 2 or len(candidates) < 2:
                continue
            initial = [value - min(candidates, key=lambda candidate: abs(value - candidate)) for value in values]
            offset = float(np.clip(np.median(initial), -0.025, 0.025))
            residuals.extend(
                min(abs(value - (candidate + offset)) for candidate in candidates)
                for value in values
            )
        if not residuals:
            continue
        error = float(np.mean(residuals))
        if error < best_error - 1e-9 or (
            abs(error - best_error) <= 1e-9 and abs(float(factor) - 1.0) < abs(best_factor - 1.0)
        ):
            best_factor = float(factor)
            best_error = error
    return best_factor


def _inverse_align_segment(
    segment: StructuralSegment,
    rotation: float,
    reflected: bool,
) -> StructuralSegment:
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    def transform(point: LayoutPoint) -> LayoutPoint:
        x = point.x * cos_r + point.y * sin_r
        y = -point.x * sin_r + point.y * cos_r
        return LayoutPoint(-x if reflected else x, y)

    start = transform(segment.start)
    end = transform(segment.end)
    return StructuralSegment(start, end, _orientation(start, end), segment.source_id)


def _region_from_polygons(polygons: list[np.ndarray]) -> ScanFootprintRegion:
    bounds = [
        (
            float(np.min(polygon[:, 0])),
            float(np.min(polygon[:, 1])),
            float(np.max(polygon[:, 0])),
            float(np.max(polygon[:, 1])),
        )
        for polygon in polygons
    ]
    return ScanFootprintRegion(
        polygons=polygons,
        polygon_bounds=bounds,
        circular_pose_count=len(polygons),
        profile_config=RainbowProfileConfig(),
        source="partial_dxf_footprint_layer",
    )


def _orientation_signatures(
    segments: list[StructuralSegment],
    tank: TankGeometry,
    orientation: str,
) -> list[float]:
    normalized = [
        StructuralSegment(_normalize(segment.start, tank), _normalize(segment.end, tank), segment.orientation, segment.source_id)
        for segment in segments
        if segment.orientation == orientation
    ]
    if orientation == "vertical":
        return _cluster_values([_line_position(segment, orientation) for segment in normalized], 0.012)
    if orientation == "horizontal":
        return _cluster_values([_line_position(segment, orientation) for segment in normalized], 0.012)
    return _cluster_values([round(_segment_angle(segment.start, segment.end), 2) for segment in normalized], 0.03)


def _match_scalar_positions(first: list[float], second: list[float], tolerance: float) -> tuple[int, list[float]]:
    remaining = list(second)
    errors = []
    for value in first:
        if not remaining:
            break
        index = min(range(len(remaining)), key=lambda item: abs(remaining[item] - value))
        error = abs(remaining[index] - value)
        if error <= tolerance:
            errors.append(error)
            remaining.pop(index)
    return len(errors), errors


def _dominant_spacing(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    differences = [difference for difference in np.diff(sorted(values)) if difference > 0.015]
    if not differences:
        return None
    clusters = _cluster_values(differences, 0.012)
    return min(clusters, key=lambda candidate: sum(abs(value - candidate) for value in differences))


def _cluster_values(values: Iterable[float], tolerance: float) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - float(np.mean(clusters[-1]))) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def _line_position(segment: StructuralSegment, orientation: str) -> float:
    if orientation == "vertical":
        return (segment.start.x + segment.end.x) / 2.0
    return (segment.start.y + segment.end.y) / 2.0


def _orientation(start: LayoutPoint, end: LayoutPoint) -> str:
    angle = abs(math.degrees(_segment_angle(start, end))) % 180.0
    if abs(angle - 90.0) <= 6.0:
        return "vertical"
    if min(angle, abs(180.0 - angle)) <= 6.0:
        return "horizontal"
    return "diagonal"


def _segment_angle(start: LayoutPoint, end: LayoutPoint) -> float:
    return math.atan2(end.y - start.y, end.x - start.x)


def _undirected_angle_difference(first: float, second: float) -> float:
    difference = abs((first - second) % math.pi)
    return min(difference, math.pi - difference)


def _normalize(point: LayoutPoint, tank: TankGeometry) -> LayoutPoint:
    return LayoutPoint((point.x - tank.center_x) / tank.radius, (point.y - tank.center_y) / tank.radius)


def _point_distance(first: LayoutPoint, second: LayoutPoint) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _observed_length(segment: ObservedSegment) -> float:
    return math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)


def _weighted_mean(values: list[float], weights: np.ndarray) -> float:
    if not values or not np.any(weights > 0.0):
        return 0.0
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def _optional_relative_similarity(first: float | None, second: float | None) -> float:
    if first is None and second is None:
        return 1.0
    if first is None or second is None:
        return 0.25
    return max(0.0, 1.0 - abs(first - second) / max(abs(first), abs(second), 1e-9))


def _ensure_appid(doc: Any, name: str) -> bool:
    if name not in doc.appids:
        doc.appids.new(name)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and complete a partial tank-layout observation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("partial_dxf", nargs="?", help="Partial DXF generated from circular scan footprints.")
    source.add_argument("--observations-json", help="Source-independent runtime observations JSON.")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--save-dxf", default="completed_predicted_layout.dxf")
    parser.add_argument("--save-plot", default="completed_predicted_layout.png")
    parser.add_argument("--save-report", default="prediction_report.json")
    parser.add_argument("--minimum-score", type=float, default=DEFAULT_MINIMUM_FAMILY_SCORE)
    parser.add_argument("--minimum-margin", type=float, default=DEFAULT_MINIMUM_SCORE_MARGIN)
    parser.add_argument("--allow-reflection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.observations_json:
            target, region = load_runtime_observations(args.observations_json)
        else:
            target, region = load_partial_dxf(args.partial_dxf)
        families = load_reference_families(args.reference_dir)
        result = predict_from_observations(
            target,
            families,
            region,
            minimum_score=args.minimum_score,
            minimum_margin=args.minimum_margin,
            allow_reflection=args.allow_reflection,
        )
        plot_path = plot_prediction(result, args.save_plot, observed_region=region)
        dxf_path = None
        if result.status == "completed":
            dxf_path = save_completed_dxf(result, args.save_dxf)
        report_path = save_prediction_report(
            result,
            args.save_report,
            output_paths={
                "dxf": str(dxf_path) if dxf_path is not None else None,
                "plot": str(plot_path),
                "report": str(args.save_report),
            },
        )
    except (DxfImportError, RuntimeError, ValueError) as exc:
        print(f"DXF prediction failed: {exc}")
        return 1

    print("DXF prediction summary")
    for score in result.family_scores:
        marker = " <-- selected" if score.family == result.selected_family else ""
        print(f"{score.family}: {score.score:.3f}{marker}")
    print(f"Status: {result.status}")
    print(f"Score margin: {result.score_margin:.3f}")
    print(f"Observed segments: {len(result.observed_segments)}")
    print(f"Predicted segments: {len(result.predicted_segments)}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if dxf_path is not None:
        print(f"Completed DXF: {dxf_path}")
    print(f"Preview: {plot_path}")
    print(f"Report: {report_path}")
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
