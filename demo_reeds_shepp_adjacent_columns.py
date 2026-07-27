"""One isolated Reeds-Shepp test between adjacent 24ft vertical scan columns.

This script reads the current 24ft mission only to confirm the selected scan
poses and to obtain the tank center.  It does not modify the mission or the
planner.  The connector itself always uses the explicit stored pose anchors
specified for this demonstration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from reeds_shepp_connector import (
    Pose2D,
    plan_reeds_shepp,
    sample_segments_for_visualization,
    validate_path_in_circle,
)
from scan_profile import RainbowProfileConfig, make_rainbow_profile, transform_profile


MISSION_PATH = Path("24ft_final_mission.json")
OUTPUT_PATH = Path("reeds_shepp_adjacent_columns_24ft.png")
EXPECTED_ANCHORS = ((-0.6096, 1.72549), (0.6096, 1.23425))
TURN_RADIUS_M = 1.8288
SAMPLE_STEP_M = 0.05
TANK_RADIUS_M = 3.6576
SELECTED_GUIDE_IDS = (1, 2)


def _profile_axes_and_robot_pose(pose: dict) -> tuple[Pose2D, tuple[float, float], tuple[float, float]]:
    """Derive the robot heading from the transformed footprint and row traversal."""
    local_profile = make_rainbow_profile()
    world_profile = transform_profile(
        local_profile, pose["x_m"], pose["y_m"], pose["heading_rad"]
    )
    # The outer 1.600 m chord is the long side: its endpoints are actual
    # polygon vertices before and after the outer arc.
    outer_right_index = RainbowProfileConfig().arc_samples - 1
    long_vector = world_profile[outer_right_index] - world_profile[0]
    long_length = float(math.hypot(*long_vector))
    if not math.isclose(long_length, RainbowProfileConfig().width, abs_tol=1e-9):
        raise RuntimeError("Could not identify the transformed 1.600 m long side.")
    long_unit = (float(long_vector[0] / long_length), float(long_vector[1] / long_length))
    short_unit = (-long_unit[1], long_unit[0])
    traversal = pose.get("travel_direction")
    if traversal == "up":
        if short_unit[1] < 0.0:
            short_unit = (-short_unit[0], -short_unit[1])
    elif traversal == "down":
        if short_unit[1] > 0.0:
            short_unit = (-short_unit[0], -short_unit[1])
    else:
        raise RuntimeError(f"Unsupported vertical traversal direction: {traversal!r}")

    heading = math.atan2(short_unit[1], short_unit[0])
    heading_vector = (math.cos(heading), math.sin(heading))
    dot_short = sum(a * b for a, b in zip(heading_vector, short_unit))
    dot_long = sum(a * b for a, b in zip(heading_vector, long_unit))
    if not (abs(dot_short) >= 1.0 - 1e-6 and abs(dot_long) <= 1e-6):
        raise RuntimeError("Derived robot heading is not aligned to the profile short side.")
    return Pose2D(pose["anchor_x_m"], pose["anchor_y_m"], heading), long_unit, short_unit


def _load_selected_pose_data() -> tuple[dict, dict, dict, Pose2D, Pose2D]:
    mission = json.loads(MISSION_PATH.read_text(encoding="utf-8"))
    poses = mission["poses"]
    selected: list[dict] = []
    for guide_id, expected_anchor in zip(SELECTED_GUIDE_IDS, EXPECTED_ANCHORS):
        candidates = [
            pose
            for pose in poses
            if pose.get("stage") == "interior_vertical"
            and pose.get("section_id") == guide_id
            and pose.get("profile_variant") == "full"
        ]
        if not candidates:
            raise RuntimeError(f"No full vertical scan poses found for guide {guide_id}.")
        top = max(candidates, key=lambda pose: pose["anchor_y_m"])
        if not (
            math.isclose(top["anchor_x_m"], expected_anchor[0], abs_tol=1e-4)
            and math.isclose(top["anchor_y_m"], expected_anchor[1], abs_tol=1e-4)
        ):
            raise RuntimeError(f"Guide {guide_id} no longer matches the required stored waypoint.")
        selected.append(top)
    start, _start_long, _start_short = _profile_axes_and_robot_pose(selected[0])
    goal, _goal_long, _goal_short = _profile_axes_and_robot_pose(selected[1])
    return mission, selected[0], selected[1], start, goal


def _plot_profile(ax, pose: dict, color: str, label: str) -> None:
    polygon = transform_profile(
        make_rainbow_profile(), pose["x_m"], pose["y_m"], pose["heading_rad"]
    )
    closed = polygon.tolist() + [polygon[0].tolist()]
    xs, ys = zip(*closed)
    ax.fill(xs, ys, color=color, alpha=0.15, zorder=3)
    ax.plot(xs, ys, color=color, linewidth=3.1, label=label, zorder=4)


def main() -> int:
    mission, start_scan, goal_scan, start, goal = _load_selected_pose_data()
    tank_center = mission["tank_center_m"]
    if not math.isclose(mission["tank_radius_m"], TANK_RADIUS_M, abs_tol=1e-9):
        raise RuntimeError("The mission tank radius no longer matches this fixed demo.")

    path = plan_reeds_shepp(
        start, goal, turn_radius_m=TURN_RADIUS_M, sample_step_m=SAMPLE_STEP_M
    )
    validate_path_in_circle(
        path, tank_center["x"], tank_center["y"], mission["tank_radius_m"]
    )
    segment_groups = sample_segments_for_visualization(path, SAMPLE_STEP_M)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_patch(
        Circle(
            (tank_center["x"], tank_center["y"]),
            mission["tank_radius_m"],
            fill=False,
            color="#111827",
            linewidth=2.2,
            label="24ft tank boundary",
            zorder=1,
        )
    )
    for guide_id, selected in zip(SELECTED_GUIDE_IDS, (start_scan, goal_scan)):
        guide_poses = [
            pose
            for pose in mission["poses"]
            if pose.get("stage") == "interior_vertical" and pose.get("section_id") == guide_id
        ]
        ys = [pose["anchor_y_m"] for pose in guide_poses]
        ax.plot(
            [selected["anchor_x_m"]] * 2,
            [min(ys), max(ys)],
            color="#94a3b8",
            linewidth=1.5,
            alpha=0.65,
            label="Selected vertical columns" if guide_id == SELECTED_GUIDE_IDS[0] else "_nolegend_",
            zorder=2,
        )

    _plot_profile(ax, start_scan, "#059669", "Start scan profile")
    _plot_profile(ax, goal_scan, "#7c3aed", "Goal scan profile")

    seen_directions: set[str] = set()
    cusp_points: list[Pose2D] = []
    for index, (segment, samples) in enumerate(segment_groups):
        color = "#2563eb" if segment.direction == "forward" else "#dc2626"
        direction_label = "Forward connector" if segment.direction == "forward" else "Reverse connector"
        ax.plot(
            [sample.x for sample in samples],
            [sample.y for sample in samples],
            color=color,
            linewidth=2.8,
            linestyle="-" if segment.direction == "forward" else "--",
            label=direction_label if segment.direction not in seen_directions else "_nolegend_",
            zorder=5,
        )
        seen_directions.add(segment.direction)
        if index and segment_groups[index - 1][0].direction != segment.direction:
            cusp_points.append(samples[0])
    if cusp_points:
        ax.scatter(
            [pose.x for pose in cusp_points],
            [pose.y for pose in cusp_points],
            s=70,
            marker="D",
            color="#f59e0b",
            edgecolor="#78350f",
            label="Cusp",
            zorder=7,
        )
    if path.first_invalid_sample_index is not None:
        invalid = path.samples[path.first_invalid_sample_index]
        ax.scatter(
            [invalid.x], [invalid.y], s=125, marker="X", color="#ef4444",
            edgecolor="#7f1d1d", label="First invalid boundary sample", zorder=9,
        )

    for pose, color, label in ((start, "#059669", "Start anchor"), (goal, "#7c3aed", "Goal anchor")):
        ax.scatter([pose.x], [pose.y], color=color, s=72, label=label, zorder=8)
        ax.arrow(
            pose.x, pose.y, 0.42 * math.cos(pose.heading), 0.42 * math.sin(pose.heading),
            width=0.012, head_width=0.105, length_includes_head=True, color=color, zorder=8,
        )

    status = "VALID" if path.boundary_valid else "INVALID: leaves tank boundary"
    ax.set_title(
        f"24ft adjacent-column Reeds-Shepp connector — {status}\n"
        f"{path.word}; {path.total_length_m:.3f} m; turn radius {TURN_RADIUS_M:.4f} m"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    print(f"guide_ids={SELECTED_GUIDE_IDS}")
    print(f"start={start}")
    print(f"goal={goal}")
    print(f"anchor_distance_m={math.dist((start.x, start.y), (goal.x, goal.y)):.9f}")
    print(f"word={path.word}")
    print("segments=" + "; ".join(
        f"{segment.segment_type}{'+' if segment.direction == 'forward' else '-'} {segment.signed_length_m:.9f} m"
        for segment in path.segments
    ))
    print(f"total_length_m={path.total_length_m:.9f}")
    print(f"endpoint_position_error_m={path.endpoint_position_error_m:.12g}")
    print(f"endpoint_heading_error_rad={path.endpoint_heading_error_rad:.12g}")
    print(f"boundary_valid={path.boundary_valid}")
    print(f"invalid_sample_count={path.invalid_sample_count}")
    if path.first_invalid_sample_index is not None:
        print(f"first_invalid_sample={path.samples[path.first_invalid_sample_index]}")
    print(f"output={OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
