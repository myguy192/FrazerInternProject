"""Isolated Reeds-Shepp connector MVP.

Path-family equations are adapted from the complete MIT-licensed
PythonRobotics implementation by Atsushi Sakai and contributors:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/
ReedsSheppPath/reeds_shepp_path_planning.py

Copyright (c) 2016-now Atsushi Sakai and PythonRobotics contributors.
Adaptation copyright (c) 2026 AIR Internship contributors.
SPDX-License-Identifier: MIT

Coordinates are meters; headings are counterclockwise-positive radians.
Boundary validation checks only the sampled robot centerline, not the robot's
physical footprint and not obstacles inside the tank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


MAX_ENDPOINT_POSITION_ERROR_M = 1e-3
MAX_ENDPOINT_HEADING_ERROR_RAD = 1e-3
BOUNDARY_NUMERICAL_TOLERANCE_M = 1e-9
_NUMERICAL_EPSILON = 1e-12


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    heading: float  # radians


@dataclass(frozen=True)
class ReedsSheppSegment:
    segment_type: str       # "L", "R", or "S"
    direction: str          # "forward" or "reverse"
    signed_length_m: float
    length_m: float


@dataclass
class ReedsSheppPath:
    start: Pose2D
    goal: Pose2D
    turn_radius_m: float
    word: str
    segments: list[ReedsSheppSegment]
    samples: list[Pose2D]
    total_length_m: float
    endpoint_position_error_m: float
    endpoint_heading_error_rad: float
    endpoint_valid: bool
    boundary_valid: bool | None = None
    violation_reason: str | None = None
    first_invalid_sample_index: int | None = None
    invalid_sample_count: int = 0


@dataclass(frozen=True)
class _NormalizedCandidate:
    segment_types: tuple[str, ...]
    signed_lengths: tuple[float, ...]

    @property
    def normalized_total_length(self) -> float:
        return sum(abs(length) for length in self.signed_lengths)


_FamilyFunction = Callable[
    [float, float, float],
    tuple[bool, list[float], list[str]],
]


def plan_reeds_shepp(
    start: Pose2D,
    goal: Pose2D,
    turn_radius_m: float,
    sample_step_m: float = 0.05,
) -> ReedsSheppPath:
    """Return the shortest complete Reeds-Shepp candidate that reconstructs the goal."""
    _validate_pose(start, "start")
    _validate_pose(goal, "goal")
    _validate_positive_finite(turn_radius_m, "turn_radius_m")
    _validate_positive_finite(sample_step_m, "sample_step_m")
    normalized_start = Pose2D(start.x, start.y, _wrap_angle(start.heading))
    normalized_goal = Pose2D(goal.x, goal.y, _wrap_angle(goal.heading))

    position_delta = math.hypot(
        normalized_goal.x - normalized_start.x,
        normalized_goal.y - normalized_start.y,
    )
    heading_delta = abs(_angle_difference(normalized_goal.heading, normalized_start.heading))
    if position_delta <= _NUMERICAL_EPSILON and heading_delta <= _NUMERICAL_EPSILON:
        return ReedsSheppPath(
            start=normalized_start,
            goal=normalized_goal,
            turn_radius_m=turn_radius_m,
            word="",
            segments=[],
            samples=[normalized_start],
            total_length_m=0.0,
            endpoint_position_error_m=position_delta,
            endpoint_heading_error_rad=heading_delta,
            endpoint_valid=True,
        )

    dx = normalized_goal.x - normalized_start.x
    dy = normalized_goal.y - normalized_start.y
    cos_start = math.cos(normalized_start.heading)
    sin_start = math.sin(normalized_start.heading)
    local_x = (cos_start * dx + sin_start * dy) / turn_radius_m
    local_y = (-sin_start * dx + cos_start * dy) / turn_radius_m
    local_heading = _wrap_angle(normalized_goal.heading - normalized_start.heading)
    candidates = _generate_complete_candidates(local_x, local_y, local_heading)
    if not candidates:
        raise RuntimeError("Reeds-Shepp solver generated no mathematical candidates.")

    valid_paths: list[ReedsSheppPath] = []
    diagnostics: list[tuple[str, float, float]] = []
    for candidate in candidates:
        segments = _metric_segments(candidate, turn_radius_m)
        samples, _segment_samples = _sample_segments(
            normalized_start,
            segments,
            turn_radius_m,
            sample_step_m,
        )
        final_pose = samples[-1]
        position_error = math.hypot(
            final_pose.x - normalized_goal.x,
            final_pose.y - normalized_goal.y,
        )
        heading_error = abs(_angle_difference(final_pose.heading, normalized_goal.heading))
        word = _path_word(segments)
        diagnostics.append((word, position_error, heading_error))
        endpoint_valid = (
            position_error <= MAX_ENDPOINT_POSITION_ERROR_M
            and heading_error <= MAX_ENDPOINT_HEADING_ERROR_RAD
        )
        if not endpoint_valid:
            continue
        valid_paths.append(
            ReedsSheppPath(
                start=normalized_start,
                goal=normalized_goal,
                turn_radius_m=turn_radius_m,
                word=word,
                segments=segments,
                samples=samples,
                total_length_m=sum(segment.length_m for segment in segments),
                endpoint_position_error_m=position_error,
                endpoint_heading_error_rad=heading_error,
                endpoint_valid=True,
            )
        )
    if not valid_paths:
        best_word, best_position_error, best_heading_error = min(
            diagnostics,
            key=lambda item: (item[1], item[2]),
        )
        raise RuntimeError(
            "No Reeds-Shepp candidate reconstructed the goal within tolerance; "
            f"best={best_word}, position_error_m={best_position_error:.6g}, "
            f"heading_error_rad={best_heading_error:.6g}."
        )
    return min(
        valid_paths,
        key=lambda path: (path.total_length_m, len(path.segments), path.word),
    )


def validate_path_in_circle(
    path: ReedsSheppPath,
    tank_center_x: float,
    tank_center_y: float,
    tank_radius_m: float,
) -> ReedsSheppPath:
    """Mark whether all sampled robot-center poses remain inside a circular tank."""
    _validate_finite(tank_center_x, "tank_center_x")
    _validate_finite(tank_center_y, "tank_center_y")
    _validate_positive_finite(tank_radius_m, "tank_radius_m")
    invalid_indices = [
        index
        for index, sample in enumerate(path.samples)
        if math.hypot(sample.x - tank_center_x, sample.y - tank_center_y)
        > tank_radius_m + BOUNDARY_NUMERICAL_TOLERANCE_M
    ]
    path.boundary_valid = not invalid_indices
    path.invalid_sample_count = len(invalid_indices)
    path.first_invalid_sample_index = invalid_indices[0] if invalid_indices else None
    path.violation_reason = "outside_tank_boundary" if invalid_indices else None
    return path


def sample_segments_for_visualization(
    path: ReedsSheppPath,
    sample_step_m: float | None = None,
) -> list[tuple[ReedsSheppSegment, list[Pose2D]]]:
    """Return per-segment samples for the standalone visualizer."""
    step = sample_step_m if sample_step_m is not None else max(path.turn_radius_m / 30.0, 0.02)
    _validate_positive_finite(step, "sample_step_m")
    _samples, segment_samples = _sample_segments(
        path.start,
        path.segments,
        path.turn_radius_m,
        step,
    )
    return list(zip(path.segments, segment_samples))


def _validate_pose(pose: Pose2D, name: str) -> None:
    _validate_finite(pose.x, f"{name}.x")
    _validate_finite(pose.y, f"{name}.y")
    _validate_finite(pose.heading, f"{name}.heading")


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_positive_finite(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _wrap_angle(angle: float) -> float:
    if -math.pi <= angle <= math.pi:
        return angle
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi) and angle > 0.0 else wrapped


def _angle_difference(first: float, second: float) -> float:
    return _wrap_angle(first - second)


def _mod2pi(angle: float) -> float:
    divisor = math.copysign(2.0 * math.pi, angle if angle != 0.0 else 1.0)
    value = angle % divisor
    if value < -math.pi:
        value += 2.0 * math.pi
    elif value > math.pi:
        value -= 2.0 * math.pi
    return value


def _polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _left_straight_left(x: float, y: float, phi: float):
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = _mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]
    return False, [], []


def _left_straight_right(x: float, y: float, phi: float):
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    squared = u1 * u1
    if squared >= 4.0:
        u = math.sqrt(max(0.0, squared - 4.0))
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, v], ["L", "S", "R"]
    return False, [], []


def _left_x_right_x_left(x: float, y: float, phi: float):
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        a = math.acos(max(-1.0, min(1.0, 0.25 * u1)))
        t = _mod2pi(a + theta + math.pi / 2.0)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]
    return False, [], []


def _left_x_right_left(x: float, y: float, phi: float):
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        a = math.acos(max(-1.0, min(1.0, 0.25 * u1)))
        t = _mod2pi(a + theta + math.pi / 2.0)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]
    return False, [], []


def _left_right_x_left(x: float, y: float, phi: float):
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if _NUMERICAL_EPSILON < u1 <= 4.0:
        u = math.acos(max(-1.0, min(1.0, 1.0 - u1 * u1 * 0.125)))
        a = math.asin(max(-1.0, min(1.0, 2.0 * math.sin(u) / u1)))
        t = _mod2pi(-a + theta + math.pi / 2.0)
        v = _mod2pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]
    return False, [], []


def _left_right_x_left_right(x: float, y: float, phi: float):
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 <= 2.0:
        a = math.acos(max(-1.0, min(1.0, (u1 + 2.0) * 0.25)))
        t = _mod2pi(theta + a + math.pi / 2.0)
        u = _mod2pi(a)
        v = _mod2pi(phi - t + 2.0 * u)
        if t >= 0.0 and u >= 0.0 and v >= 0.0:
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]
    return False, [], []


def _left_x_right_left_x_right(x: float, y: float, phi: float):
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 <= _NUMERICAL_EPSILON:
        return False, [], []
    u2 = (20.0 - u1 * u1) / 16.0
    if 0.0 <= u2 <= 1.0:
        u = math.acos(u2)
        a = math.asin(max(-1.0, min(1.0, 2.0 * math.sin(u) / u1)))
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]
    return False, [], []


def _left_x_right90_straight_left(x: float, y: float, phi: float):
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 >= 2.0:
        root = math.sqrt(max(0.0, u1 * u1 - 4.0))
        u = root - 2.0
        a = math.atan2(2.0, root)
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi + math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -v], ["L", "R", "S", "L"]
    return False, [], []


def _left_straight_right90_x_left(x: float, y: float, phi: float):
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 >= 2.0:
        root = math.sqrt(max(0.0, u1 * u1 - 4.0))
        u = root - 2.0
        a = math.atan2(root, 2.0)
        t = _mod2pi(theta - a + math.pi / 2.0)
        v = _mod2pi(t - phi - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, math.pi / 2.0, -v], ["L", "S", "R", "L"]
    return False, [], []


def _left_x_right90_straight_right(x: float, y: float, phi: float):
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 2.0:
        t = _mod2pi(theta + math.pi / 2.0)
        u = u1 - 2.0
        v = _mod2pi(phi - t - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -v], ["L", "R", "S", "R"]
    return False, [], []


def _left_straight_left90_x_right(x: float, y: float, phi: float):
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 2.0:
        t = _mod2pi(theta)
        u = u1 - 2.0
        v = _mod2pi(phi - t - math.pi / 2.0)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, math.pi / 2.0, -v], ["L", "S", "L", "R"]
    return False, [], []


def _left_x_right90_straight_left90_x_right(x: float, y: float, phi: float):
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 4.0:
        root = math.sqrt(max(0.0, u1 * u1 - 4.0))
        u = root - 4.0
        a = math.atan2(2.0, root)
        t = _mod2pi(theta + a + math.pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, -math.pi / 2.0, -u, -math.pi / 2.0, v], ["L", "R", "S", "L", "R"]
    return False, [], []


def _generate_complete_candidates(x: float, y: float, phi: float) -> list[_NormalizedCandidate]:
    families: tuple[_FamilyFunction, ...] = (
        _left_straight_left,
        _left_straight_right,
        _left_x_right_x_left,
        _left_x_right_left,
        _left_right_x_left,
        _left_right_x_left_right,
        _left_x_right_left_x_right,
        _left_x_right90_straight_left,
        _left_x_right90_straight_right,
        _left_straight_right90_x_left,
        _left_straight_left90_x_right,
        _left_x_right90_straight_left90_x_right,
    )
    candidates: list[_NormalizedCandidate] = []
    seen: set[tuple[tuple[str, ...], tuple[float, ...]]] = set()

    def add(flag: bool, lengths: list[float], types: list[str]) -> None:
        if not flag:
            return
        key = (tuple(types), tuple(round(length, 12) for length in lengths))
        if key in seen:
            return
        seen.add(key)
        candidates.append(_NormalizedCandidate(tuple(types), tuple(lengths)))

    for family in families:
        flag, lengths, types = family(x, y, phi)
        add(flag, lengths, types)
        flag, lengths, types = family(-x, y, -phi)
        add(flag, [-length for length in lengths], types)
        flag, lengths, types = family(x, -y, -phi)
        add(flag, lengths, [_reflect_type(segment_type) for segment_type in types])
        flag, lengths, types = family(-x, -y, phi)
        add(
            flag,
            [-length for length in lengths],
            [_reflect_type(segment_type) for segment_type in types],
        )
    return candidates


def _reflect_type(segment_type: str) -> str:
    if segment_type == "L":
        return "R"
    if segment_type == "R":
        return "L"
    return "S"


def _metric_segments(
    candidate: _NormalizedCandidate,
    turn_radius_m: float,
) -> list[ReedsSheppSegment]:
    return [
        ReedsSheppSegment(
            segment_type=segment_type,
            direction="forward" if signed_length >= 0.0 else "reverse",
            signed_length_m=signed_length * turn_radius_m,
            length_m=abs(signed_length) * turn_radius_m,
        )
        for segment_type, signed_length in zip(
            candidate.segment_types,
            candidate.signed_lengths,
        )
    ]


def _path_word(segments: list[ReedsSheppSegment]) -> str:
    return "".join(
        f"{segment.segment_type}{'+' if segment.signed_length_m >= 0.0 else '-'}"
        for segment in segments
    )


def _sample_segments(
    start: Pose2D,
    segments: list[ReedsSheppSegment],
    turn_radius_m: float,
    sample_step_m: float,
) -> tuple[list[Pose2D], list[list[Pose2D]]]:
    samples = [start]
    segment_samples: list[list[Pose2D]] = []
    current = start
    for segment in segments:
        count = max(1, int(math.ceil(segment.length_m / sample_step_m)))
        start_of_segment = current
        current_segment_samples = [start_of_segment]
        if segment.length_m <= _NUMERICAL_EPSILON:
            segment_samples.append(current_segment_samples)
            continue
        for index in range(1, count + 1):
            signed_distance = segment.signed_length_m * index / count
            pose = _propagate_segment(
                start_of_segment,
                segment.segment_type,
                signed_distance,
                turn_radius_m,
            )
            current_segment_samples.append(pose)
            samples.append(pose)
        current = current_segment_samples[-1]
        segment_samples.append(current_segment_samples)
    return samples, segment_samples


def _propagate_segment(
    start: Pose2D,
    segment_type: str,
    signed_distance_m: float,
    turn_radius_m: float,
) -> Pose2D:
    if segment_type == "S":
        return Pose2D(
            start.x + signed_distance_m * math.cos(start.heading),
            start.y + signed_distance_m * math.sin(start.heading),
            _wrap_angle(start.heading),
        )
    curvature = (1.0 if segment_type == "L" else -1.0) / turn_radius_m
    heading = start.heading + curvature * signed_distance_m
    x = start.x + (math.sin(heading) - math.sin(start.heading)) / curvature
    y = start.y - (math.cos(heading) - math.cos(start.heading)) / curvature
    return Pose2D(x, y, _wrap_angle(heading))
