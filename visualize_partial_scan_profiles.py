from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from partial_scan_profile import (
    half_intersection_area,
    make_rainbow_profile_halves,
    polygon_area,
    reconstruction_area_error,
    transform_local_point,
    transform_rainbow_half,
)
from scan_profile import transform_profile


def _draw_partition(
    ax,
    full_polygon: np.ndarray,
    left_polygon: np.ndarray,
    right_polygon: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    anchor: np.ndarray,
    *,
    title: str,
) -> None:
    ax.fill(left_polygon[:, 0], left_polygon[:, 1], color="#2563eb", alpha=0.32, label="Local left half")
    ax.fill(right_polygon[:, 0], right_polygon[:, 1], color="#f97316", alpha=0.32, label="Local right half")
    ax.plot(full_polygon[:, 0], full_polygon[:, 1], color="#111827", linewidth=2.0, label="Original full outline")
    ax.plot(
        [axis_start[0], axis_end[0]],
        [axis_start[1], axis_end[1]],
        color="#16a34a",
        linestyle="--",
        linewidth=1.8,
        label="Local symmetry axis",
    )
    ax.scatter([anchor[0]], [anchor[1]], color="#dc2626", s=48, marker="x", linewidth=2.2, label="Long-side midpoint anchor")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def create_visualization(*, show: bool = True, save_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    halves = make_rainbow_profile_halves()
    axis_start = np.asarray(halves.symmetry_axis_segment_m[0], dtype=float)
    axis_end = np.asarray(halves.symmetry_axis_segment_m[1], dtype=float)
    anchor = np.asarray(halves.long_side_anchor_m, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_partition(
        axes[0],
        halves.full_polygon,
        halves.left_half.polygon,
        halves.right_half.polygon,
        axis_start,
        axis_end,
        anchor,
        title="Local rainbow split at x = 0",
    )

    heading_rad = math.pi / 2.0
    translation_x = 0.0
    translation_y = 0.0
    full_rotated = transform_profile(halves.full_polygon, translation_x, translation_y, heading_rad)
    left_rotated = transform_rainbow_half(halves.left_half, translation_x, translation_y, heading_rad)
    right_rotated = transform_rainbow_half(halves.right_half, translation_x, translation_y, heading_rad)
    axis_start_rotated = transform_local_point(
        halves.symmetry_axis_segment_m[0], translation_x, translation_y, heading_rad
    )
    axis_end_rotated = transform_local_point(
        halves.symmetry_axis_segment_m[1], translation_x, translation_y, heading_rad
    )
    anchor_rotated = transform_local_point(
        halves.long_side_anchor_m, translation_x, translation_y, heading_rad
    )
    _draw_partition(
        axes[1],
        full_rotated,
        left_rotated,
        right_rotated,
        axis_start_rotated,
        axis_end_rotated,
        anchor_rotated,
        title="Same halves rotated 90° (local right points up)",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Reusable left/right rainbow scan-profile geometry", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig, axes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize local and rotated left/right rainbow profile halves.")
    parser.add_argument("--save-path", default="partial_scan_profiles_preview.png")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    halves = make_rainbow_profile_halves()
    create_visualization(show=not args.no_show, save_path=args.save_path)
    print("Partial rainbow profile summary")
    print(f"Full area: {polygon_area(halves.full_polygon):.12g} m^2")
    print(f"Left area: {polygon_area(halves.left_half.polygon):.12g} m^2")
    print(f"Right area: {polygon_area(halves.right_half.polygon):.12g} m^2")
    print(f"Intersection area: {half_intersection_area(halves):.12g} m^2")
    print(f"Union reconstruction error: {reconstruction_area_error(halves):.12g} m^2")
    print(f"Plot output: {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
