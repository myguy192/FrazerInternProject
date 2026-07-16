import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Point

from dxf_importer import Bounds, GeometryModel, LineSegment, Point2D
from half_profile_gap_fill import FullProfileRejectionReason, as_polygon
from path_planner import (
    LawnmowerSectionLine,
    TankCircleEstimate,
    _add_outer_lawnmower_sections,
    _build_mission_plan_once,
    _generate_lawnmower_placement_pass,
    generate_lawnmower_scan_placements,
    lawnmower_scan_profile_polygon,
)


TANK = TankCircleEstimate(center_x=0.0, center_y=0.0, radius=5.0, method="test")


def _base_grid_lines():
    lines = []
    for index, x_m in enumerate((-2.0, 0.0, 2.0)):
        lines.append(
            LawnmowerSectionLine(
                line_id=len(lines),
                orientation="vertical",
                order_index=index,
                source_section_id=index,
                start_m=(x_m, -3.0),
                end_m=(x_m, 3.0),
            )
        )
    for index, y_m in enumerate((-2.0, 0.0, 2.0)):
        lines.append(
            LawnmowerSectionLine(
                line_id=len(lines),
                orientation="horizontal",
                order_index=index,
                source_section_id=index,
                start_m=(-3.0, y_m),
                end_m=(3.0, y_m),
            )
        )
    return lines


def _cross_coordinate(line):
    return line.start_m[0] if line.orientation == "vertical" else line.start_m[1]


def _segment(x1, y1, x2, y2):
    return LineSegment(Point2D(x1, y1), Point2D(x2, y2), "TEST")


def _stable_grid_model():
    coordinates = (-3.0, -1.0, 1.0, 3.0)
    return GeometryModel(
        source_path=None,
        line_segments=[
            *[_segment(x_m, -4.0, x_m, 4.0) for x_m in coordinates],
            *[_segment(-4.0, y_m, 4.0, y_m) for y_m in coordinates],
        ],
        arcs=[],
        circles=[],
        bounds=Bounds(-6.0, -6.0, 6.0, 6.0),
    )


class OuterLawnmowerSectionTests(unittest.TestCase):
    def setUp(self):
        self.base = _base_grid_lines()
        self.result = _add_outer_lawnmower_sections(self.base, TANK)
        self.added = [line for line in self.result if line.is_outer_extension]

    def test_adds_exactly_one_section_at_each_outer_grid_edge(self):
        self.assertEqual(len(self.added), 4)
        self.assertEqual(
            [(line.orientation, _cross_coordinate(line)) for line in self.added],
            [("vertical", -4.0), ("vertical", 4.0), ("horizontal", -4.0), ("horizontal", 4.0)],
        )

    def test_new_sections_use_the_normal_neighbor_spacing(self):
        for orientation in ("vertical", "horizontal"):
            base_cross = sorted(_cross_coordinate(line) for line in self.base if line.orientation == orientation)
            added_cross = sorted(_cross_coordinate(line) for line in self.added if line.orientation == orientation)
            spacing = float(np.median(np.diff(base_cross)))
            self.assertAlmostEqual(base_cross[0] - added_cross[0], spacing)
            self.assertAlmostEqual(added_cross[-1] - base_cross[-1], spacing)

    def test_new_sections_are_tank_clipped_and_not_tiny(self):
        for line in self.added:
            self.assertGreater(np.linalg.norm(np.subtract(line.end_m, line.start_m)), 1e-6)
            for point in (line.start_m, line.end_m):
                self.assertLessEqual(np.linalg.norm(point), TANK.radius + 1e-9)

    def test_existing_section_geometry_is_unchanged(self):
        by_id = {line.line_id: line for line in self.result if not line.is_outer_extension}
        self.assertEqual(set(by_id), {line.line_id for line in self.base})
        for line in self.base:
            self.assertEqual(by_id[line.line_id].start_m, line.start_m)
            self.assertEqual(by_id[line.line_id].end_m, line.end_m)
            self.assertEqual(by_id[line.line_id].orientation, line.orientation)

    def test_ordering_is_deterministic_and_geometric(self):
        again = _add_outer_lawnmower_sections(self.base, TANK)
        self.assertEqual(self.result, again)
        for orientation in ("vertical", "horizontal"):
            lines = [line for line in self.result if line.orientation == orientation]
            self.assertEqual([line.order_index for line in lines], list(range(len(lines))))
            self.assertEqual(
                [_cross_coordinate(line) for line in lines],
                sorted(_cross_coordinate(line) for line in lines),
            )

    def test_new_sections_use_the_existing_full_and_half_candidate_pipeline(self):
        placement_pass = _generate_lawnmower_placement_pass(self.result)
        new_ids = {line.line_id for line in self.added}
        self.assertTrue(
            any(placement.guide_line_id in new_ids for placement in placement_pass.accepted)
        )
        rejected_on_new_lines = [
            candidate
            for candidate in placement_pass.rejected
            if candidate.guide_line_id in new_ids
        ]
        self.assertTrue(rejected_on_new_lines)
        self.assertTrue(
            all(
                FullProfileRejectionReason.SECTION_BOUNDARY_CONFLICT
                in candidate.rejection_reasons
                for candidate in rejected_on_new_lines
            )
        )
        self.assertEqual(
            generate_lawnmower_scan_placements(self.result),
            placement_pass.accepted,
        )

        tank_polygon = Point(TANK.center_x, TANK.center_y).buffer(TANK.radius, quad_segs=256)
        for placement in placement_pass.accepted:
            if placement.guide_line_id in new_ids:
                footprint = as_polygon(lawnmower_scan_profile_polygon(placement))
                self.assertLessEqual(footprint.difference(tank_polygon).area, 1e-8)

    def test_no_outer_extensions_are_created_without_a_stable_grid(self):
        irregular = [
            LawnmowerSectionLine(
                line_id=index,
                orientation="vertical",
                order_index=index,
                source_section_id=index,
                start_m=(x_m, -2.0),
                end_m=(x_m, 2.0),
            )
            for index, x_m in enumerate((-2.0, 0.0, 3.0))
        ]
        self.assertEqual(_add_outer_lawnmower_sections(irregular, TANK), irregular)

    def test_circular_poses_are_unchanged_by_outer_sections(self):
        model = _stable_grid_model()
        with patch(
            "path_planner._add_outer_lawnmower_sections",
            side_effect=lambda lines, _tank: lines,
        ):
            baseline = _build_mission_plan_once(model, fixed_circular_rows=1)
        expanded = _build_mission_plan_once(model, fixed_circular_rows=1)

        baseline_circular = [pose for pose in baseline.poses if pose.stage == "circular_edge"]
        expanded_circular = [pose for pose in expanded.poses if pose.stage == "circular_edge"]
        self.assertEqual(expanded_circular, baseline_circular)
        self.assertEqual(len(expanded.lawnmower_section_lines), len(baseline.lawnmower_section_lines) + 4)


if __name__ == "__main__":
    unittest.main()
