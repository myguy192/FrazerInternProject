from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RainbowProfileConfig:
    """Approximate scan profile dimensions in meters."""

    width: float = 1.600
    arc_radius: float = 0.900
    side_height: float = 0.500
    arc_samples: int = 64


@dataclass(frozen=True)
class ProfileBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: float
    height: float


def make_rainbow_profile(config: RainbowProfileConfig | None = None) -> np.ndarray:
    """Return a closeable Nx2 polygon for the rainbow scan footprint.

    Local coordinates:
    - x axis: left/right across the scan profile
    - y axis: forward/back direction of the footprint
    - origin: approximate center of the profile bounds

    The v1 approximation uses two circular arcs with the same radius and
    chord width, joined by vertical side edges. This matches the drawing's
    curved lower/inner edge instead of using a flat bottom.
    """
    config = RainbowProfileConfig() if config is None else config
    _validate_config(config)

    half_width = config.width / 2.0
    chord_to_center = math.sqrt(config.arc_radius**2 - half_width**2)
    inner_arc = _arc_points_for_chord(
        half_width=half_width,
        radius=config.arc_radius,
        center_y=-chord_to_center,
        samples=config.arc_samples,
    )
    outer_arc = _arc_points_for_chord(
        half_width=half_width,
        radius=config.arc_radius,
        center_y=config.side_height - chord_to_center,
        samples=config.arc_samples,
    )

    points = np.vstack(
        (
            outer_arc,
            np.array([[half_width, 0.0]]),
            inner_arc[::-1],
            np.array([[-half_width, config.side_height]]),
        )
    )

    bounds = profile_bounds(points)
    center = np.array(
        [
            (bounds.min_x + bounds.max_x) / 2.0,
            (bounds.min_y + bounds.max_y) / 2.0,
        ]
    )
    return points - center


def _arc_points_for_chord(half_width: float, radius: float, center_y: float, samples: int) -> np.ndarray:
    chord_to_center = math.sqrt(radius**2 - half_width**2)
    start_angle = math.atan2(chord_to_center, -half_width)
    end_angle = math.atan2(chord_to_center, half_width)
    crown_angle = math.pi / 2.0
    left_count = samples // 2 + 1
    right_count = samples - left_count + 1
    angles = np.concatenate(
        (
            np.linspace(start_angle, crown_angle, left_count),
            np.linspace(crown_angle, end_angle, right_count)[1:],
        )
    )
    return np.column_stack(
        (
            radius * np.cos(angles),
            center_y + radius * np.sin(angles),
        )
    )


def transform_profile(profile_points: np.ndarray, x: float, y: float, heading_rad: float) -> np.ndarray:
    """Rotate local profile points by heading_rad and translate them to x, y."""
    points = _as_profile_array(profile_points)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
    translation = np.array([float(x), float(y)], dtype=float)
    return points @ rotation.T + translation


def profile_bounds(profile_points: np.ndarray) -> ProfileBounds:
    """Compute axis-aligned bounds and dimensions for profile points."""
    points = _as_profile_array(profile_points)
    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    return ProfileBounds(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        width=max_x - min_x,
        height=max_y - min_y,
    )


def plot_profile(
    ax,
    profile_points: np.ndarray,
    label: str = "Rainbow scan profile",
    *,
    color: str = "#2563eb",
    fill_alpha: float = 0.16,
    linewidth: float = 2.0,
):
    """Draw the scan profile outline and a light fill on a matplotlib axis."""
    points = _closed_points(_as_profile_array(profile_points))
    ax.fill(points[:, 0], points[:, 1], color=color, alpha=fill_alpha, label=label)
    (line,) = ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth)
    return line


def annotate_profile_measurements(
    ax,
    profile_points: np.ndarray,
    config: RainbowProfileConfig | None = None,
    *,
    text_color: str = "#111827",
) -> None:
    """Add dimension labels for the local scan profile on a matplotlib axis."""
    points = _as_profile_array(profile_points)
    bounds = profile_bounds(points)
    x_pad = max(bounds.width * 0.08, 0.08)
    y_pad = max(bounds.height * 0.08, 0.08)

    width_y = bounds.min_y - y_pad
    ax.annotate(
        "",
        xy=(bounds.min_x, width_y),
        xytext=(bounds.max_x, width_y),
        arrowprops={"arrowstyle": "<->", "color": text_color, "linewidth": 1.0},
    )
    ax.text(
        (bounds.min_x + bounds.max_x) / 2.0,
        width_y - y_pad * 0.25,
        f"width {bounds.width:.3f} m",
        ha="center",
        va="top",
        color=text_color,
    )

    height_x = bounds.max_x + x_pad
    ax.annotate(
        "",
        xy=(height_x, bounds.min_y),
        xytext=(height_x, bounds.max_y),
        arrowprops={"arrowstyle": "<->", "color": text_color, "linewidth": 1.0},
    )
    ax.text(
        height_x + x_pad * 0.2,
        (bounds.min_y + bounds.max_y) / 2.0,
        f"height {bounds.height:.3f} m",
        ha="left",
        va="center",
        rotation=90,
        color=text_color,
    )

    if config is None:
        return

    side_x = bounds.min_x - x_pad
    side_top_y = bounds.max_y
    side_bottom_y = side_top_y - config.side_height
    ax.annotate(
        "",
        xy=(side_x, side_bottom_y),
        xytext=(side_x, side_top_y),
        arrowprops={"arrowstyle": "<->", "color": text_color, "linewidth": 1.0},
    )
    ax.text(
        side_x - x_pad * 0.2,
        (side_bottom_y + side_top_y) / 2.0,
        f"side {config.side_height:.3f} m",
        ha="right",
        va="center",
        rotation=90,
        color=text_color,
    )
    ax.text(
        bounds.min_x + bounds.width * 0.05,
        bounds.max_y + y_pad * 0.45,
        f"arc R {config.arc_radius:.3f} m",
        ha="left",
        va="bottom",
        color=text_color,
    )


def _validate_config(config: RainbowProfileConfig) -> None:
    if config.width <= 0.0:
        raise ValueError("Scan profile width must be positive.")
    if config.arc_radius <= 0.0:
        raise ValueError("Scan profile arc radius must be positive.")
    if config.side_height <= 0.0:
        raise ValueError("Scan profile side height must be positive.")
    if config.arc_radius <= config.width / 2.0:
        raise ValueError("Arc radius must be greater than half the scan width.")
    if not 4 <= config.arc_samples <= 1000:
        raise ValueError("Arc sample count must be between 4 and 1000.")


def _as_profile_array(profile_points: np.ndarray) -> np.ndarray:
    points = np.asarray(profile_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Profile points must be an Nx2 array.")
    if len(points) < 3:
        raise ValueError("Profile must contain at least three points.")
    return points


def _closed_points(points: np.ndarray) -> np.ndarray:
    if np.allclose(points[0], points[-1]):
        return points
    return np.vstack((points, points[0]))


def _print_demo_summary(profile: np.ndarray) -> None:
    bounds = profile_bounds(profile)
    print("Rainbow scan profile demo")
    print(f"Point count: {len(profile)}")
    print(
        "Bounds: "
        f"min=({bounds.min_x:.6g}, {bounds.min_y:.6g}), "
        f"max=({bounds.max_x:.6g}, {bounds.max_y:.6g})"
    )
    print(f"Dimensions: width={bounds.width:.6g} m, height={bounds.height:.6g} m")


def _run_demo() -> None:
    import matplotlib.pyplot as plt

    config = RainbowProfileConfig()
    profile = make_rainbow_profile(config)
    _print_demo_summary(profile)

    examples = [
        (profile, "Local profile", "#111827"),
        (transform_profile(profile, 2.0, 0.4, math.radians(25.0)), "Placed +25 deg", "#2563eb"),
        (transform_profile(profile, -1.6, -0.9, math.radians(-40.0)), "Placed -40 deg", "#059669"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    for points, label, color in examples:
        plot_profile(ax, points, label=label, color=color)
    annotate_profile_measurements(ax, profile, config)

    ax.scatter([0.0], [0.0], color="#dc2626", s=24, label="Origin")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Rainbow scan profile geometry")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _run_demo()
