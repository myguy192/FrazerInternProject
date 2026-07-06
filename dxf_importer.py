from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_ENTITY_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE"}


class DxfImportError(RuntimeError):
    """Raised for expected DXF loading/import failures."""


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class RawEntity:
    entity_type: str
    layer: str
    handle: str | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LineSegment:
    start: Point2D
    end: Point2D
    layer: str
    handle: str | None = None


@dataclass(frozen=True)
class PolylineVertex:
    point: Point2D
    bulge: float = 0.0


@dataclass(frozen=True)
class PolylineGeometry:
    vertices: list[PolylineVertex]
    closed: bool
    layer: str
    handle: str | None = None
    entity_type: str = "POLYLINE"


@dataclass(frozen=True)
class ArcGeometry:
    center: Point2D
    radius: float
    start_angle: float
    end_angle: float
    layer: str
    handle: str | None = None


@dataclass(frozen=True)
class CircleGeometry:
    center: Point2D
    radius: float
    layer: str
    handle: str | None = None


@dataclass(frozen=True)
class UnsupportedEntity:
    entity_type: str
    layer: str
    handle: str | None = None


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass
class GeometryModel:
    source_path: Path | None
    raw_entities: list[RawEntity] = field(default_factory=list)
    line_segments: list[LineSegment] = field(default_factory=list)
    polylines: list[PolylineGeometry] = field(default_factory=list)
    arcs: list[ArcGeometry] = field(default_factory=list)
    circles: list[CircleGeometry] = field(default_factory=list)
    unsupported_entities: list[UnsupportedEntity] = field(default_factory=list)
    layer_names: set[str] = field(default_factory=set)
    entity_counts: Counter[str] = field(default_factory=Counter)
    layer_counts: Counter[str] = field(default_factory=Counter)
    bounds: Bounds | None = None

    @property
    def total_entities(self) -> int:
        return len(self.raw_entities)


def load_dxf(filepath: str | Path) -> Any:
    """Load a DXF document from disk with clear, expected error messages."""
    path = Path(filepath)
    if not path.exists():
        raise DxfImportError(f"DXF file not found: {path}")
    if not path.is_file():
        raise DxfImportError(f"DXF path is not a file: {path}")

    try:
        import ezdxf
        from ezdxf import recover
        from ezdxf.lldxf.const import DXFStructureError
    except ImportError as exc:
        raise DxfImportError("Missing dependency 'ezdxf'. Install it with: pip install ezdxf") from exc

    try:
        doc = ezdxf.readfile(path)
    except (OSError, IOError) as exc:
        raise DxfImportError(f"Could not read DXF file: {path}") from exc
    except DXFStructureError:
        try:
            doc, auditor = recover.readfile(path)
        except Exception as exc:
            raise DxfImportError(f"Invalid or corrupt DXF file: {path}") from exc
        if auditor.has_errors:
            raise DxfImportError(f"DXF recovery found unrecoverable errors: {path}")

    doc.import_source_path = path
    return doc


def extract_entities(doc: Any) -> list[Any]:
    """Return all modelspace entities from a loaded ezdxf document."""
    try:
        modelspace = doc.modelspace()
    except Exception as exc:
        raise DxfImportError("Could not access DXF modelspace.") from exc

    entities = list(modelspace)
    if not entities:
        raise DxfImportError("DXF modelspace is empty.")
    return entities


def build_geometry_model(entities: list[Any], source_path: str | Path | None = None) -> GeometryModel:
    """Convert ezdxf entities into the v1 internal geometry representation."""
    model = GeometryModel(source_path=Path(source_path) if source_path else None)
    bounds_builder = _BoundsBuilder()

    for entity in entities:
        entity_type = entity.dxftype()
        layer = _entity_layer(entity)
        handle = _entity_handle(entity)

        model.raw_entities.append(_raw_entity(entity, entity_type, layer, handle))
        model.entity_counts[entity_type] += 1
        model.layer_counts[layer] += 1
        model.layer_names.add(layer)

        if entity_type == "LINE":
            line = LineSegment(
                start=_point_from_vec(entity.dxf.start),
                end=_point_from_vec(entity.dxf.end),
                layer=layer,
                handle=handle,
            )
            model.line_segments.append(line)
            bounds_builder.add_point(line.start)
            bounds_builder.add_point(line.end)
        elif entity_type == "LWPOLYLINE":
            polyline = _build_lwpolyline(entity, layer, handle)
            model.polylines.append(polyline)
            bounds_builder.add_polyline(polyline)
        elif entity_type == "POLYLINE":
            polyline = _build_polyline(entity, layer, handle)
            model.polylines.append(polyline)
            bounds_builder.add_polyline(polyline)
        elif entity_type == "ARC":
            arc = ArcGeometry(
                center=_point_from_vec(entity.dxf.center),
                radius=float(entity.dxf.radius),
                start_angle=float(entity.dxf.start_angle),
                end_angle=float(entity.dxf.end_angle),
                layer=layer,
                handle=handle,
            )
            model.arcs.append(arc)
            bounds_builder.add_arc(arc)
        elif entity_type == "CIRCLE":
            circle = CircleGeometry(
                center=_point_from_vec(entity.dxf.center),
                radius=float(entity.dxf.radius),
                layer=layer,
                handle=handle,
            )
            model.circles.append(circle)
            bounds_builder.add_circle(circle)
        else:
            model.unsupported_entities.append(UnsupportedEntity(entity_type=entity_type, layer=layer, handle=handle))

    model.bounds = bounds_builder.bounds
    return model


def plot_geometry_model(
    model: GeometryModel,
    *,
    color_by_layer: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot supported imported geometry for visual verification."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(10, 10))
    layer_colors = _layer_color_map(model.layer_names) if color_by_layer else {}

    for line in model.line_segments:
        color = layer_colors.get(line.layer, "#111827")
        ax.plot([line.start.x, line.end.x], [line.start.y, line.end.y], color=color, linewidth=1.4)

    for polyline in model.polylines:
        if not polyline.vertices:
            continue
        color = layer_colors.get(polyline.layer, "#2563eb")
        xs = [vertex.point.x for vertex in polyline.vertices]
        ys = [vertex.point.y for vertex in polyline.vertices]
        if polyline.closed and len(polyline.vertices) > 1:
            xs.append(polyline.vertices[0].point.x)
            ys.append(polyline.vertices[0].point.y)
        ax.plot(xs, ys, color=color, linewidth=1.2)

    for arc in model.arcs:
        color = layer_colors.get(arc.layer, "#dc2626")
        points = _sample_arc_points(arc)
        ax.plot([point.x for point in points], [point.y for point in points], color=color, linewidth=1.3)

    for circle in model.circles:
        color = layer_colors.get(circle.layer, "#059669")
        ax.add_patch(Circle((circle.center.x, circle.center.y), circle.radius, fill=False, color=color, linewidth=1.3))

    _apply_plot_bounds(ax, model.bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Imported DXF geometry")
    ax.grid(True, alpha=0.22)

    if color_by_layer and model.layer_names:
        handles = [
            Line2D([0], [0], color=layer_colors[layer], lw=2, label=f"{layer} ({model.layer_counts[layer]})")
            for layer in sorted(model.layer_names)
        ]
        if len(handles) <= 20:
            ax.legend(handles=handles, title="Layers", loc="upper right")
        else:
            ax.text(
                0.01,
                0.99,
                f"{len(handles)} layers imported; legend hidden",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
            )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig, ax


def print_geometry_summary(model: GeometryModel) -> None:
    """Print a compact debug summary of imported geometry."""
    print("DXF import summary")
    if model.source_path:
        print(f"Source: {model.source_path}")
    print(f"Total entities: {model.total_entities}")
    print("Entity counts:")
    for entity_type, count in sorted(model.entity_counts.items()):
        supported = "" if entity_type in SUPPORTED_ENTITY_TYPES else " (unsupported)"
        print(f"  {entity_type}: {count}{supported}")
    print("Layer counts:")
    for layer, count in sorted(model.layer_counts.items()):
        print(f"  {layer}: {count}")
    if model.bounds is None:
        print("Bounds: none")
    else:
        bounds = model.bounds
        print(
            "Bounds: "
            f"min=({bounds.min_x:.6g}, {bounds.min_y:.6g}), "
            f"max=({bounds.max_x:.6g}, {bounds.max_y:.6g}), "
            f"size=({bounds.width:.6g}, {bounds.height:.6g})"
        )
    if model.unsupported_entities:
        unsupported_counts = Counter(entity.entity_type for entity in model.unsupported_entities)
        print("Unsupported entities:")
        for entity_type, count in sorted(unsupported_counts.items()):
            print(f"  {entity_type}: {count}")
    else:
        print("Unsupported entities: none")


def import_dxf(filepath: str | Path) -> GeometryModel:
    """Convenience wrapper for the full v1 import pipeline."""
    doc = load_dxf(filepath)
    entities = extract_entities(doc)
    source_path = getattr(doc, "import_source_path", filepath)
    return build_geometry_model(entities, source_path=source_path)


def _entity_layer(entity: Any) -> str:
    return str(getattr(entity.dxf, "layer", "0") or "0")


def _entity_handle(entity: Any) -> str | None:
    handle = getattr(entity.dxf, "handle", None)
    return str(handle) if handle is not None else None


def _raw_entity(entity: Any, entity_type: str, layer: str, handle: str | None) -> RawEntity:
    try:
        attributes = dict(entity.dxf.all_existing_dxf_attribs())
    except Exception:
        attributes = {}
    clean_attributes = {key: _serializable_value(value) for key, value in attributes.items()}
    return RawEntity(entity_type=entity_type, layer=layer, handle=handle, attributes=clean_attributes)


def _serializable_value(value: Any) -> Any:
    if hasattr(value, "xyz"):
        return tuple(float(coord) for coord in value.xyz)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _point_from_vec(value: Any) -> Point2D:
    if all(hasattr(value, attr) for attr in ("x", "y")):
        return Point2D(float(value.x), float(value.y), float(getattr(value, "z", 0.0) or 0.0))
    try:
        z = float(value[2])
    except (IndexError, TypeError):
        z = 0.0
    return Point2D(float(value[0]), float(value[1]), z)


def _build_lwpolyline(entity: Any, layer: str, handle: str | None) -> PolylineGeometry:
    vertices = [
        PolylineVertex(point=Point2D(float(x), float(y)), bulge=float(bulge))
        for x, y, _start_width, _end_width, bulge in entity.get_points("xyseb")
    ]
    return PolylineGeometry(
        vertices=vertices,
        closed=bool(entity.closed),
        layer=layer,
        handle=handle,
        entity_type="LWPOLYLINE",
    )


def _build_polyline(entity: Any, layer: str, handle: str | None) -> PolylineGeometry:
    vertices: list[PolylineVertex] = []
    entity_vertices = entity.vertices() if callable(getattr(entity, "vertices", None)) else entity.vertices
    for vertex in entity_vertices:
        point = _point_from_vec(vertex.dxf.location)
        bulge = float(getattr(vertex.dxf, "bulge", 0.0) or 0.0)
        vertices.append(PolylineVertex(point=point, bulge=bulge))
    return PolylineGeometry(
        vertices=vertices,
        closed=bool(entity.is_closed),
        layer=layer,
        handle=handle,
        entity_type="POLYLINE",
    )


def _sample_arc_points(arc: ArcGeometry, samples: int = 96) -> list[Point2D]:
    sweep = _ccw_sweep_degrees(arc.start_angle, arc.end_angle)
    steps = max(8, int(samples * max(sweep, 1.0) / 360.0))
    points: list[Point2D] = []
    for index in range(steps + 1):
        angle = math.radians(arc.start_angle + sweep * index / steps)
        points.append(
            Point2D(
                arc.center.x + arc.radius * math.cos(angle),
                arc.center.y + arc.radius * math.sin(angle),
                arc.center.z,
            )
        )
    return points


def _ccw_sweep_degrees(start_angle: float, end_angle: float) -> float:
    sweep = (end_angle - start_angle) % 360.0
    return 360.0 if math.isclose(sweep, 0.0) else sweep


def _angle_on_ccw_arc(angle: float, start_angle: float, end_angle: float) -> bool:
    sweep = _ccw_sweep_degrees(start_angle, end_angle)
    relative = (angle - start_angle) % 360.0
    return relative <= sweep or math.isclose(relative, sweep)


def _layer_color_map(layer_names: set[str]) -> dict[str, str]:
    palette = [
        "#111827",
        "#2563eb",
        "#059669",
        "#dc2626",
        "#7c3aed",
        "#d97706",
        "#0891b2",
        "#be185d",
        "#4b5563",
        "#65a30d",
    ]
    return {layer: palette[index % len(palette)] for index, layer in enumerate(sorted(layer_names))}


def _apply_plot_bounds(ax: Any, bounds: Bounds | None) -> None:
    if bounds is None:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return

    width = max(bounds.width, 1e-9)
    height = max(bounds.height, 1e-9)
    margin = max(width, height) * 0.05
    if margin <= 0.0:
        margin = 1.0
    ax.set_xlim(bounds.min_x - margin, bounds.max_x + margin)
    ax.set_ylim(bounds.min_y - margin, bounds.max_y + margin)


class _BoundsBuilder:
    def __init__(self) -> None:
        self._min_x = math.inf
        self._min_y = math.inf
        self._max_x = -math.inf
        self._max_y = -math.inf

    @property
    def bounds(self) -> Bounds | None:
        if not math.isfinite(self._min_x):
            return None
        return Bounds(self._min_x, self._min_y, self._max_x, self._max_y)

    def add_point(self, point: Point2D) -> None:
        self._min_x = min(self._min_x, point.x)
        self._min_y = min(self._min_y, point.y)
        self._max_x = max(self._max_x, point.x)
        self._max_y = max(self._max_y, point.y)

    def add_polyline(self, polyline: PolylineGeometry) -> None:
        for vertex in polyline.vertices:
            self.add_point(vertex.point)

    def add_circle(self, circle: CircleGeometry) -> None:
        self.add_point(Point2D(circle.center.x - circle.radius, circle.center.y - circle.radius))
        self.add_point(Point2D(circle.center.x + circle.radius, circle.center.y + circle.radius))

    def add_arc(self, arc: ArcGeometry) -> None:
        for point in _sample_arc_points(arc, samples=48):
            self.add_point(point)
        for angle in (0.0, 90.0, 180.0, 270.0):
            if _angle_on_ccw_arc(angle, arc.start_angle, arc.end_angle):
                radians = math.radians(angle)
                self.add_point(
                    Point2D(
                        arc.center.x + arc.radius * math.cos(radians),
                        arc.center.y + arc.radius * math.sin(radians),
                    )
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and visualize v1 DXF tank geometry.")
    parser.add_argument("filepath", help="Path to a DXF file.")
    parser.add_argument("--no-plot", action="store_true", help="Import and summarize without opening a plot window.")
    parser.add_argument("--save-plot", help="Optional path for saving the plot image.")
    parser.add_argument("--single-color", action="store_true", help="Draw all layers in one color.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        model = import_dxf(args.filepath)
    except DxfImportError as exc:
        print(f"DXF import failed: {exc}")
        return 1

    print_geometry_summary(model)
    if not args.no_plot or args.save_plot:
        plot_geometry_model(
            model,
            color_by_layer=not args.single_color,
            show=not args.no_plot,
            save_path=args.save_plot,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
