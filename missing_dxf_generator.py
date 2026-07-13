from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dxf_importer import DxfImportError, GeometryModel, import_dxf
from observation_region import ScanFootprintRegion, build_observed_region_from_scan_poses
from scan_profile import RainbowProfileConfig
from tank_layout_predictor import (
    ObservedTankGeometry,
    clip_geometry_to_observed_region,
    observed_geometry_from_dxf,
)


@dataclass(frozen=True)
class SelectedSweep:
    poses: list[dict[str, Any] | Any]
    row_ids: list[str]
    profile_config: RainbowProfileConfig
    source: str


def load_selected_sweep_from_mission(
    mission_path: str | Path,
    circular_rows: int,
) -> SelectedSweep:
    """Load exactly the requested first circular rows from mission order."""
    _validate_row_count(circular_rows)
    path = Path(mission_path)
    if not path.is_file():
        raise ValueError(f"Mission JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read mission JSON: {path}") from exc

    poses = data.get("poses")
    if not isinstance(poses, list):
        raise ValueError("Mission JSON must contain a poses list.")
    circular_poses = [pose for pose in poses if str(pose.get("stage", "")).startswith("circular")]
    if not circular_poses:
        raise ValueError("Mission JSON contains no circular sweep poses.")

    ordered_rows: list[tuple[str, list[dict[str, Any]]]] = []
    row_lookup: dict[str, list[dict[str, Any]]] = {}
    for pose in circular_poses:
        row_key = _mission_pose_row_key(pose)
        if row_key not in row_lookup:
            row_lookup[row_key] = []
            ordered_rows.append((row_key, row_lookup[row_key]))
        row_lookup[row_key].append(pose)
    if len(ordered_rows) < circular_rows:
        raise ValueError(
            f"Requested {circular_rows} circular rows, but mission identifies only {len(ordered_rows)}."
        )

    selected_rows = ordered_rows[:circular_rows]
    selected_poses = [pose for _, row_poses in selected_rows for pose in row_poses]
    profile_data = data.get("scan_profile", {})
    profile_config = RainbowProfileConfig(
        width=float(profile_data.get("width_m", 1.6)),
        arc_radius=float(profile_data.get("arc_radius_m", 0.9)),
        side_height=float(profile_data.get("side_height_m", 0.5)),
        arc_samples=int(profile_data.get("arc_samples", 64)),
    )
    return SelectedSweep(
        poses=selected_poses,
        row_ids=[key for key, _ in selected_rows],
        profile_config=profile_config,
        source=str(path),
    )


def generate_selected_sweep_from_planner(
    model: GeometryModel,
    circular_rows: int,
    profile_config: RainbowProfileConfig | None = None,
) -> SelectedSweep:
    """Use the project's fixed-row circular placement logic without interior planning."""
    _validate_row_count(circular_rows)
    from path_planner import (
        DEFAULT_ROW_SPACING,
        _profile_constrained_sweep_radius,
        _touching_angular_step,
        estimate_tank_circle_from_geometry,
    )

    profile_config = RainbowProfileConfig() if profile_config is None else profile_config
    tank = estimate_tank_circle_from_geometry(model)
    sweep_radius = _profile_constrained_sweep_radius(tank.radius, profile_config, 0.0)
    poses: list[dict[str, Any]] = []
    row_ids: list[str] = []
    for row_id in range(circular_rows):
        if sweep_radius <= profile_config.width / 2.0:
            raise ValueError(f"Tank cannot fit requested circular row {row_id + 1}.")
        touching_angle = _touching_angular_step(sweep_radius, profile_config)
        pose_count = max(3, int(math.ceil(2.0 * math.pi / (touching_angle * 0.80))))
        for index in range(pose_count):
            theta = 2.0 * math.pi * index / pose_count
            poses.append(
                {
                    "stage": "circular_edge",
                    "status": "kept",
                    "row_id": row_id,
                    "x_m": tank.center_x + sweep_radius * math.cos(theta),
                    "y_m": tank.center_y + sweep_radius * math.sin(theta),
                    "heading_rad": theta,
                    "sweep_radius_m": sweep_radius,
                }
            )
        row_ids.append(str(row_id))
        sweep_radius -= DEFAULT_ROW_SPACING
    return SelectedSweep(
        poses=poses,
        row_ids=row_ids,
        profile_config=profile_config,
        source="planner_geometry_fixed_row_fallback",
    )


def generate_partial_observation(
    model: GeometryModel,
    selected_sweep: SelectedSweep,
) -> tuple[ObservedTankGeometry, ObservedTankGeometry, ScanFootprintRegion]:
    full_geometry = observed_geometry_from_dxf(model)
    region = build_observed_region_from_scan_poses(selected_sweep.poses, selected_sweep.profile_config)
    observed_geometry = clip_geometry_to_observed_region(full_geometry, region)
    return full_geometry, observed_geometry, region


def save_partial_dxf(
    observed_geometry: ObservedTankGeometry,
    observed_region: ScanFootprintRegion,
    filepath: str | Path,
    *,
    include_footprints: bool = True,
    source_path: str | Path | None = None,
) -> Path:
    try:
        import ezdxf
    except ImportError as exc:
        raise RuntimeError("DXF output requires ezdxf.") from exc

    path = Path(filepath)
    if source_path is not None and path.resolve() == Path(source_path).resolve():
        raise ValueError("Partial DXF output must not overwrite the source DXF.")
    doc = ezdxf.new("R2010")
    doc.units = 6  # meters
    for layer_name, color in (
        ("TANK_BOUNDARY", 7),
        ("OBSERVED_CIRCULAR_SCAN", 3),
        ("SCAN_FOOTPRINTS", 5),
    ):
        doc.layers.add(layer_name, color=color)
    modelspace = doc.modelspace()
    tank = observed_geometry.tank
    modelspace.add_circle(
        (tank.center_x, tank.center_y),
        tank.radius,
        dxfattribs={"layer": "TANK_BOUNDARY"},
    )
    for segment in observed_geometry.segments:
        modelspace.add_line(
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
            dxfattribs={"layer": "OBSERVED_CIRCULAR_SCAN"},
        )
    if include_footprints:
        for polygon in observed_region.polygons:
            modelspace.add_lwpolyline(
                [(float(x), float(y)) for x, y in polygon],
                close=True,
                dxfattribs={"layer": "SCAN_FOOTPRINTS"},
            )
    doc.saveas(path)
    return path


def plot_partial_observation(
    observed_geometry: ObservedTankGeometry,
    observed_region: ScanFootprintRegion,
    filepath: str | Path,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(9, 9))
    tank = observed_geometry.tank
    ax.add_patch(Circle((tank.center_x, tank.center_y), tank.radius, fill=False, color="#111827", linewidth=2.0))
    for polygon in observed_region.polygons:
        closed = list(polygon) + [polygon[0]]
        xs = [point[0] for point in closed]
        ys = [point[1] for point in closed]
        ax.fill(xs, ys, color="#60a5fa", alpha=0.025)
        ax.plot(xs, ys, color="#2563eb", linewidth=0.35, alpha=0.22)
    for segment in observed_geometry.segments:
        ax.plot(
            [segment.start.x, segment.end.x],
            [segment.start.y, segment.end.y],
            color="#047857",
            linewidth=1.4,
        )
    margin = tank.radius * 0.05
    ax.set_xlim(tank.center_x - tank.radius - margin, tank.center_x + tank.radius + margin)
    ax.set_ylim(tank.center_y - tank.radius - margin, tank.center_y + tank.radius + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({observed_geometry.units})")
    ax.set_ylabel(f"y ({observed_geometry.units})")
    ax.set_title(
        f"Partial circular-sweep observation\n"
        f"{observed_region.circular_pose_count} scan footprints"
    )
    handles = [
        Line2D([], [], color="#2563eb", alpha=0.5, label="Circular scan footprints"),
        Line2D([], [], color="#047857", label="Observed weld fragments"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = Path(filepath)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _mission_pose_row_key(pose: dict[str, Any]) -> str:
    row_id = pose.get("row_id")
    if row_id is not None:
        return f"row:{row_id}"
    radius = pose.get("sweep_radius_m")
    if radius is not None:
        return f"radius:{round(float(radius), 6)}"
    raise ValueError("A circular mission pose is missing both row_id and sweep_radius_m.")


def _validate_row_count(circular_rows: int) -> None:
    if circular_rows not in (1, 2):
        raise ValueError("Circular row count must be explicitly 1 or 2.")


def _geometry_length(geometry: ObservedTankGeometry) -> float:
    return sum(
        math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)
        for segment in geometry.segments
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a partial DXF from exact circular rainbow scan footprints.")
    parser.add_argument("source_dxf", help="Complete reference DXF used only for validation simulation.")
    parser.add_argument("--mission-json", help="Saved mission containing actual circular scan poses.")
    parser.add_argument("--circular-rows", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", required=True, help="New partial DXF output path.")
    parser.add_argument("--save-preview", help="Optional partial-observation preview PNG.")
    parser.add_argument("--no-footprint-layer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        model = import_dxf(args.source_dxf)
        if args.mission_json:
            selected = load_selected_sweep_from_mission(args.mission_json, args.circular_rows)
        else:
            selected = generate_selected_sweep_from_planner(model, args.circular_rows)
        full, observed, region = generate_partial_observation(model, selected)
        output_path = save_partial_dxf(
            observed,
            region,
            args.output,
            include_footprints=not args.no_footprint_layer,
            source_path=args.source_dxf,
        )
        preview_path = None
        if args.save_preview:
            preview_path = plot_partial_observation(
                observed,
                region,
                args.save_preview,
            )
    except (DxfImportError, RuntimeError, ValueError) as exc:
        print(f"Partial DXF generation failed: {exc}")
        return 1

    full_length = _geometry_length(full)
    observed_length = _geometry_length(observed)
    print("Partial DXF generation summary")
    print(f"Circular rows used: {len(selected.row_ids)} ({', '.join(selected.row_ids)})")
    print(f"Circular poses used: {region.circular_pose_count}")
    print(f"Observed weld fragments: {len(observed.segments)}")
    print(f"Observed weld length: {observed_length:.3f} m / {full_length:.3f} m")
    print(f"Sweep source: {selected.source}")
    print(f"Partial DXF: {output_path}")
    if preview_path is not None:
        print(f"Preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
