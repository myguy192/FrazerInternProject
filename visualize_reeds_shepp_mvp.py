"""Focused CLI visualization for the isolated Reeds-Shepp connector MVP."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from reeds_shepp_connector import (
    Pose2D,
    plan_reeds_shepp,
    sample_segments_for_visualization,
    validate_path_in_circle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "HEADING_DEG"), required=True)
    parser.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "HEADING_DEG"), required=True)
    parser.add_argument("--turn-radius", type=float, required=True)
    parser.add_argument("--tank-center", nargs=2, type=float, metavar=("X", "Y"), required=True)
    parser.add_argument("--tank-radius", type=float, required=True)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def plot_connector(
    start: Pose2D,
    goal: Pose2D,
    turn_radius_m: float,
    tank_center_x: float,
    tank_center_y: float,
    tank_radius_m: float,
    sample_step_m: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    path = plan_reeds_shepp(start, goal, turn_radius_m, sample_step_m)
    validate_path_in_circle(path, tank_center_x, tank_center_y, tank_radius_m)
    groups = sample_segments_for_visualization(path, sample_step_m)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(
        Circle(
            (tank_center_x, tank_center_y),
            tank_radius_m,
            fill=False,
            color="#111827",
            linewidth=2.0,
            label="Tank boundary",
        )
    )
    labels_drawn: set[str] = set()
    cusp_points: list[Pose2D] = []
    for index, (segment, poses) in enumerate(groups):
        label = "Forward" if segment.direction == "forward" else "Reverse"
        color = "#2563eb" if segment.direction == "forward" else "#dc2626"
        ax.plot(
            [pose.x for pose in poses],
            [pose.y for pose in poses],
            color=color,
            linewidth=2.6,
            linestyle="-" if segment.direction == "forward" else "--",
            label=label if label not in labels_drawn else "_nolegend_",
        )
        labels_drawn.add(label)
        if index and groups[index - 1][0].direction != segment.direction:
            cusp_points.append(poses[0])
    if cusp_points:
        ax.scatter(
            [pose.x for pose in cusp_points],
            [pose.y for pose in cusp_points],
            s=55,
            color="#f59e0b",
            marker="D",
            label="Cusp",
            zorder=5,
        )

    arrow_length = max(0.3, min(turn_radius_m * 0.6, tank_radius_m * 0.15))
    for pose, color, label in (
        (path.start, "#059669", "Start"),
        (path.goal, "#7c3aed", "Goal"),
    ):
        ax.scatter([pose.x], [pose.y], s=65, color=color, label=label, zorder=6)
        ax.arrow(
            pose.x,
            pose.y,
            arrow_length * math.cos(pose.heading),
            arrow_length * math.sin(pose.heading),
            width=0.015,
            head_width=0.12,
            length_includes_head=True,
            color=color,
            zorder=6,
        )
    if path.first_invalid_sample_index is not None:
        invalid = path.samples[path.first_invalid_sample_index]
        ax.scatter(
            [invalid.x],
            [invalid.y],
            s=105,
            color="#ef4444",
            edgecolor="#7f1d1d",
            marker="X",
            label="First invalid sample",
            zorder=8,
        )

    validity = "VALID" if path.boundary_valid else "INVALID: outside tank boundary"
    ax.set_title(
        f"Reeds-Shepp connector — {validity}\n"
        f"{path.word}, length {path.total_length_m:.3f} m"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    start = Pose2D(args.start[0], args.start[1], math.radians(args.start[2]))
    goal = Pose2D(args.goal[0], args.goal[1], math.radians(args.goal[2]))
    plot_connector(
        start,
        goal,
        args.turn_radius,
        args.tank_center[0],
        args.tank_center[1],
        args.tank_radius,
        args.sample_step,
        args.output,
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
