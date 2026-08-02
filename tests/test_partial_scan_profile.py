import math
import unittest

import numpy as np

from partial_scan_profile import (
    half_intersection_area,
    make_rainbow_profile_halves,
    polygon_area,
    reconstruction_area_error,
    transform_local_point,
    transform_rainbow_half,
)
from scan_profile import make_rainbow_profile, transform_profile


TOLERANCE = 1e-9


def _point_on_segment(point, start, end, tolerance=TOLERANCE):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    offset = point - start
    if abs(segment[0] * offset[1] - segment[1] * offset[0]) > tolerance:
        return False
    return float(np.dot(offset, segment)) >= -tolerance and float(np.dot(point - end, segment)) <= tolerance


def _point_in_or_on_polygon(point, polygon):
    points = polygon[:-1] if np.allclose(polygon[0], polygon[-1]) else polygon
    if any(_point_on_segment(point, start, end) for start, end in zip(points, np.roll(points, -1, axis=0))):
        return True
    x, y = point
    inside = False
    previous = points[-1]
    for current in points:
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (
                (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1])
                + current[0]
            )
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _proper_segments_intersect(first_start, first_end, second_start, second_end):
    def cross(first, second):
        return float(first[0] * second[1] - first[1] * second[0])

    first_start = np.asarray(first_start, dtype=float)
    first_end = np.asarray(first_end, dtype=float)
    second_start = np.asarray(second_start, dtype=float)
    second_end = np.asarray(second_end, dtype=float)
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = cross(first_direction, second_direction)
    if abs(denominator) <= TOLERANCE:
        return False
    offset = second_start - first_start
    first_parameter = cross(offset, second_direction) / denominator
    second_parameter = cross(offset, first_direction) / denominator
    return TOLERANCE < first_parameter < 1.0 - TOLERANCE and TOLERANCE < second_parameter < 1.0 - TOLERANCE


def _is_simple_closed_polygon(polygon):
    if len(polygon) < 4 or not np.allclose(polygon[0], polygon[-1], atol=TOLERANCE):
        return False
    edges = list(zip(polygon[:-1], polygon[1:]))
    for first_index, first_edge in enumerate(edges):
        for second_index, second_edge in enumerate(edges[first_index + 1 :], start=first_index + 1):
            if second_index in {first_index, first_index + 1}:
                continue
            if first_index == 0 and second_index == len(edges) - 1:
                continue
            if _proper_segments_intersect(*first_edge, *second_edge):
                return False
    return polygon_area(polygon) > TOLERANCE


class PartialScanProfileTests(unittest.TestCase):
    def setUp(self):
        self.halves = make_rainbow_profile_halves()

    def test_union_reconstructs_full_polygon_and_halves_do_not_overlap(self):
        full_area = polygon_area(self.halves.full_polygon)
        left_area = polygon_area(self.halves.left_half.polygon)
        right_area = polygon_area(self.halves.right_half.polygon)

        self.assertLessEqual(reconstruction_area_error(self.halves), 1e-12)
        self.assertEqual(half_intersection_area(self.halves), 0.0)
        self.assertAlmostEqual(left_area, full_area / 2.0, places=12)
        self.assertAlmostEqual(right_area, full_area / 2.0, places=12)

        for point in self.halves.full_polygon:
            target = self.halves.left_half.polygon if point[0] <= TOLERANCE else self.halves.right_half.polygon
            self.assertTrue(any(np.allclose(point, candidate, atol=TOLERANCE) for candidate in target))

    def test_both_halves_are_valid_closed_polygons(self):
        self.assertTrue(_is_simple_closed_polygon(self.halves.left_half.polygon))
        self.assertTrue(_is_simple_closed_polygon(self.halves.right_half.polygon))
        self.assertLessEqual(float(np.max(self.halves.left_half.polygon[:, 0])), TOLERANCE)
        self.assertGreaterEqual(float(np.min(self.halves.right_half.polygon[:, 0])), -TOLERANCE)

    def test_halves_contain_the_correct_long_side_segments(self):
        left_start, left_end = self.halves.left_half.long_side_segment_m
        right_start, right_end = self.halves.right_half.long_side_segment_m

        self.assertEqual(left_start, self.halves.long_side_start_m)
        self.assertEqual(left_end, self.halves.long_side_anchor_m)
        self.assertEqual(right_start, self.halves.long_side_anchor_m)
        self.assertEqual(right_end, self.halves.long_side_end_m)
        for fraction in np.linspace(0.0, 1.0, 41):
            left_point = np.asarray(left_start) + fraction * (np.asarray(left_end) - np.asarray(left_start))
            right_point = np.asarray(right_start) + fraction * (np.asarray(right_end) - np.asarray(right_start))
            self.assertTrue(_point_in_or_on_polygon(left_point, self.halves.left_half.polygon))
            self.assertTrue(_point_in_or_on_polygon(right_point, self.halves.right_half.polygon))

    def test_same_rigid_transform_preserves_reconstruction_and_anchor(self):
        x_m = 2.4
        y_m = -1.7
        heading_rad = math.radians(37.0)
        full_world = transform_profile(self.halves.full_polygon, x_m, y_m, heading_rad)
        left_world = transform_rainbow_half(self.halves.left_half, x_m, y_m, heading_rad)
        right_world = transform_rainbow_half(self.halves.right_half, x_m, y_m, heading_rad)
        anchor_world = transform_local_point(self.halves.long_side_anchor_m, x_m, y_m, heading_rad)

        self.assertAlmostEqual(
            polygon_area(full_world),
            polygon_area(left_world) + polygon_area(right_world),
            places=12,
        )
        self.assertTrue(
            np.allclose(
                anchor_world,
                transform_local_point(self.halves.left_half.full_long_side_anchor_m, x_m, y_m, heading_rad),
                atol=TOLERANCE,
            )
        )
        self.assertTrue(
            np.allclose(
                anchor_world,
                transform_local_point(self.halves.right_half.full_long_side_anchor_m, x_m, y_m, heading_rad),
                atol=TOLERANCE,
            )
        )

    def test_ninety_degree_rotation_keeps_left_right_as_local_labels(self):
        x_m = 3.0
        y_m = -2.0
        heading_rad = math.pi / 2.0
        left_world = transform_rainbow_half(self.halves.left_half, x_m, y_m, heading_rad)
        right_world = transform_rainbow_half(self.halves.right_half, x_m, y_m, heading_rad)

        self.assertLessEqual(float(np.max(left_world[:, 1])), y_m + TOLERANCE)
        self.assertGreaterEqual(float(np.min(right_world[:, 1])), y_m - TOLERANCE)

    def test_halves_are_never_recentered_independently(self):
        x_m = -4.0
        y_m = 2.5
        heading_rad = math.radians(-28.0)
        for half in (self.halves.left_half, self.halves.right_half):
            expected = transform_profile(half.polygon, x_m, y_m, heading_rad)
            actual = transform_rainbow_half(half, x_m, y_m, heading_rad)
            self.assertTrue(np.array_equal(actual, expected))
            local_bounds_center = np.array(
                [
                    (float(np.min(half.polygon[:, 0])) + float(np.max(half.polygon[:, 0]))) / 2.0,
                    (float(np.min(half.polygon[:, 1])) + float(np.max(half.polygon[:, 1]))) / 2.0,
                ]
            )
            self.assertFalse(np.allclose(local_bounds_center, half.full_long_side_anchor_m, atol=1e-3))

    def test_original_full_profile_geometry_is_unchanged(self):
        before = make_rainbow_profile()
        make_rainbow_profile_halves()
        after = make_rainbow_profile()

        self.assertTrue(np.array_equal(before, after))
        self.assertTrue(np.array_equal(before, self.halves.full_polygon))


if __name__ == "__main__":
    unittest.main()
