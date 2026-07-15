import math
import unittest
from unittest.mock import patch

import numpy as np

from path_planner import (
    LawnmowerSectionLine,
    estimate_lawnmower_neighbor_overlap_fraction,
    generate_lawnmower_scan_placements,
    lawnmower_long_side_endpoints,
    lawnmower_scan_profile_polygon,
)


def _guide(line_id, orientation, start, end):
    return LawnmowerSectionLine(
        line_id=line_id,
        orientation=orientation,
        order_index=line_id,
        source_section_id=line_id,
        start_m=start,
        end_m=end,
    )


def _direction(line):
    direction = np.asarray(line.end_m, dtype=float) - np.asarray(line.start_m, dtype=float)
    return direction / np.linalg.norm(direction)


def _polygon_centroid(points):
    closed = np.vstack((points, points[0]))
    cross = closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1]
    area_twice = np.sum(cross)
    return np.array(
        [
            np.sum((closed[:-1, 0] + closed[1:, 0]) * cross) / (3.0 * area_twice),
            np.sum((closed[:-1, 1] + closed[1:, 1]) * cross) / (3.0 * area_twice),
        ]
    )


class LawnmowerScanPlacementTests(unittest.TestCase):
    def assertAnchorAndLongSide(self, guide_line, placement):
        long_start, long_end = lawnmower_long_side_endpoints(placement)
        midpoint = (long_start + long_end) / 2.0
        guide_direction = _direction(guide_line)
        long_direction = (long_end - long_start) / np.linalg.norm(long_end - long_start)
        line_normal = np.array([-guide_direction[1], guide_direction[0]])

        self.assertAlmostEqual(float(np.dot(midpoint - np.asarray(guide_line.start_m), line_normal)), 0.0, places=9)
        self.assertTrue(np.allclose(midpoint, placement.anchor_m, atol=1e-9))
        self.assertAlmostEqual(float(np.dot(long_direction, guide_direction)), 0.0, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(long_end - long_start)), 1.6, places=9)

    def test_vertical_profiles_are_ordered_anchored_and_horizontal_across_the_line(self):
        line = _guide(0, "vertical", (2.0, -3.0), (2.0, 3.0))
        placements = generate_lawnmower_scan_placements([line])

        self.assertGreater(len(placements), 2)
        self.assertEqual([placement.travel_direction for placement in placements], ["up"] * len(placements))
        self.assertEqual([placement.anchor_m[1] for placement in placements], sorted(placement.anchor_m[1] for placement in placements))
        for placement in placements:
            self.assertAnchorAndLongSide(line, placement)
            start, end = lawnmower_long_side_endpoints(placement)
            self.assertAlmostEqual(start[1], end[1], places=9)
            self.assertAlmostEqual(placement.anchor_m[0], 2.0, places=9)

    def test_horizontal_profiles_are_ordered_anchored_and_vertical_across_the_line(self):
        line = _guide(1, "horizontal", (-3.0, 1.5), (3.0, 1.5))
        placements = generate_lawnmower_scan_placements([line])

        self.assertGreater(len(placements), 2)
        self.assertEqual([placement.travel_direction for placement in placements], ["right"] * len(placements))
        self.assertEqual([placement.anchor_m[0] for placement in placements], sorted(placement.anchor_m[0] for placement in placements))
        for placement in placements:
            self.assertAnchorAndLongSide(line, placement)
            start, end = lawnmower_long_side_endpoints(placement)
            self.assertAlmostEqual(start[0], end[0], places=9)
            self.assertAlmostEqual(placement.anchor_m[1], 1.5, places=9)

    def test_reversing_guide_reverses_order_and_rotates_profiles(self):
        forward = _guide(2, "vertical", (0.0, -3.0), (0.0, 3.0))
        reverse = _guide(2, "vertical", (0.0, 3.0), (0.0, -3.0))
        forward_placements = generate_lawnmower_scan_placements([forward])
        reverse_placements = generate_lawnmower_scan_placements([reverse])

        self.assertEqual([placement.travel_direction for placement in reverse_placements], ["down"] * len(reverse_placements))
        self.assertTrue(
            np.allclose(
                [placement.anchor_m[1] for placement in reverse_placements],
                [placement.anchor_m[1] for placement in forward_placements[::-1]],
                atol=1e-9,
            )
        )
        for forward_placement, reverse_placement in zip(forward_placements[::-1], reverse_placements):
            angle_delta = (reverse_placement.heading_rad - forward_placement.heading_rad) % (2.0 * math.pi)
            self.assertAlmostEqual(angle_delta, math.pi, places=9)

    def test_neighbors_have_small_polygon_area_overlap_without_a_guide_gap(self):
        line = _guide(3, "horizontal", (-4.0, 0.0), (4.0, 0.0))
        placements = generate_lawnmower_scan_placements([line])
        direction = _direction(line)

        for first, second in zip(placements, placements[1:]):
            overlap = estimate_lawnmower_neighbor_overlap_fraction(first, second)
            self.assertGreaterEqual(overlap, 0.01)
            self.assertLessEqual(overlap, 0.04)
            first_projection = np.dot(lawnmower_scan_profile_polygon(first), direction)
            second_projection = np.dot(lawnmower_scan_profile_polygon(second), direction)
            self.assertGreaterEqual(float(np.max(first_projection)), float(np.min(second_projection)))

    def test_long_side_anchor_is_not_polygon_centroid(self):
        line = _guide(4, "vertical", (1.0, -2.0), (1.0, 2.0))
        placement = generate_lawnmower_scan_placements([line])[0]
        centroid = _polygon_centroid(lawnmower_scan_profile_polygon(placement))

        self.assertGreater(float(np.linalg.norm(centroid - np.asarray(placement.anchor_m))), 0.01)

    def test_crossing_orientations_are_not_deduplicated(self):
        vertical = _guide(5, "vertical", (0.0, -3.0), (0.0, 3.0))
        horizontal = _guide(6, "horizontal", (-3.0, 0.0), (3.0, 0.0))
        combined = generate_lawnmower_scan_placements([vertical, horizontal])
        separate_count = len(generate_lawnmower_scan_placements([vertical])) + len(
            generate_lawnmower_scan_placements([horizontal])
        )

        self.assertEqual(len(combined), separate_count)
        self.assertEqual({placement.orientation for placement in combined}, {"vertical", "horizontal"})

    def test_guide_lines_are_not_changed_and_circular_code_is_not_called(self):
        lines = [
            _guide(7, "vertical", (-1.0, -3.0), (-1.0, 3.0)),
            _guide(8, "horizontal", (-3.0, 1.0), (3.0, 1.0)),
        ]
        original = list(lines)
        with patch("path_planner._generate_circular_sweep_plan", side_effect=AssertionError("circular generation called")):
            placements = generate_lawnmower_scan_placements(lines)

        self.assertEqual(lines, original)
        self.assertTrue(placements)


if __name__ == "__main__":
    unittest.main()
