import unittest
from unittest.mock import patch

from dxf_importer import ArcGeometry, Bounds, CircleGeometry, GeometryModel, LineSegment, Point2D
from path_planner import LawnmowerSectionLine, TankCircleEstimate, generate_lawnmower_section_lines


TANK = TankCircleEstimate(center_x=0.0, center_y=0.0, radius=5.0, method="test")


def _line(x1, y1, x2, y2):
    return LineSegment(Point2D(x1, y1), Point2D(x2, y2), "TEST")


def _model(*lines, bounds=(-5.0, -5.0, 5.0, 5.0), arcs=None, circles=None):
    return GeometryModel(
        source_path=None,
        line_segments=list(lines),
        arcs=[] if arcs is None else arcs,
        circles=[] if circles is None else circles,
        bounds=Bounds(*bounds),
    )


def _orientation(lines, name):
    return [line for line in lines if line.orientation == name]


class LawnmowerSectionLineTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_one_vertical_section_has_correct_bounds(self):
        model = _model(_line(-1, -4, -1, 4), _line(1, -4, 1, 4))
        vertical = _orientation(generate_lawnmower_section_lines(model, TANK), "vertical")
        self.assertEqual(len(vertical), 1)
        self.assertPointAlmostEqual(vertical[0].start_m, (0.0, -4.0))
        self.assertPointAlmostEqual(vertical[0].end_m, (0.0, 4.0))

    def test_vertical_sections_all_run_bottom_to_top(self):
        model = _model(_line(-2, -4, -2, 4), _line(0, -4, 0, 4), _line(2, -4, 2, 4))
        vertical = _orientation(generate_lawnmower_section_lines(model, TANK), "vertical")
        self.assertEqual([line.start_m[0] for line in vertical], [-1.0, 1.0])
        self.assertLess(vertical[0].start_m[1], vertical[0].end_m[1])
        self.assertLess(vertical[1].start_m[1], vertical[1].end_m[1])

    def test_one_horizontal_section_has_correct_bounds(self):
        model = _model(_line(-4, -1, 4, -1), _line(-4, 1, 4, 1))
        horizontal = _orientation(generate_lawnmower_section_lines(model, TANK), "horizontal")
        self.assertEqual(len(horizontal), 1)
        self.assertPointAlmostEqual(horizontal[0].start_m, (-4.0, 0.0))
        self.assertPointAlmostEqual(horizontal[0].end_m, (4.0, 0.0))

    def test_horizontal_sections_all_run_left_to_right(self):
        model = _model(_line(-4, -2, 4, -2), _line(-4, 0, 4, 0), _line(-4, 2, 4, 2))
        horizontal = _orientation(generate_lawnmower_section_lines(model, TANK), "horizontal")
        self.assertEqual([line.start_m[1] for line in horizontal], [-1.0, 1.0])
        self.assertLess(horizontal[0].start_m[0], horizontal[0].end_m[0])
        self.assertLess(horizontal[1].start_m[0], horizontal[1].end_m[0])

    def test_nearly_connected_horizontal_fragments_join_within_tolerance(self):
        model = _model(
            _line(-4, -1, 0, -1), _line(0.04, -1, 4, -1),
            _line(-4, 1, 0, 1), _line(0.04, 1, 4, 1),
        )
        horizontal = _orientation(generate_lawnmower_section_lines(model, TANK), "horizontal")
        self.assertEqual(len(horizontal), 1)
        self.assertPointAlmostEqual(horizontal[0].start_m, (-4.0, 0.0))
        self.assertPointAlmostEqual(horizontal[0].end_m, (4.0, 0.0))

    def test_clearly_separated_horizontal_fragments_remain_separate(self):
        model = _model(
            _line(-7, -1, -0.5, -1), _line(0.5, -1, 7, -1),
            _line(-7, 1, -0.5, 1), _line(0.5, 1, 7, 1),
            bounds=(-7, -5, 7, 5),
        )
        horizontal = _orientation(generate_lawnmower_section_lines(model, TANK), "horizontal")
        self.assertEqual(len(horizontal), 2)
        ranges = sorted((min(line.start_m[0], line.end_m[0]), max(line.start_m[0], line.end_m[0])) for line in horizontal)
        self.assertAlmostEqual(ranges[0][1], -0.5)
        self.assertAlmostEqual(ranges[1][0], 0.5)

    def test_similar_y_coordinates_alone_do_not_create_full_width_section(self):
        model = _model(_line(-4, 0.0, -1, 0.0), _line(1, 0.01, 4, 0.01))
        lines = generate_lawnmower_section_lines(model, TANK)
        self.assertEqual(_orientation(lines, "horizontal"), [])

    def test_horizontal_sections_are_split_away_from_vertical_contacts(self):
        model = _model(
            _line(-1, -4, -1, 4), _line(1, -4, 1, 4),
            _line(-4, -1, 4, -1), _line(-4, 1, 4, 1),
        )
        lines = generate_lawnmower_section_lines(model, TANK)
        vertical = _orientation(lines, "vertical")
        horizontal = _orientation(lines, "horizontal")
        self.assertEqual(len(vertical), 1)
        self.assertEqual(len(horizontal), 2)
        self.assertEqual(vertical[0].start_m[0], 0.0)
        for line in horizontal:
            self.assertEqual(line.start_m[1], 0.0)
            self.assertFalse(min(line.start_m[0], line.end_m[0]) <= 0.0 <= max(line.start_m[0], line.end_m[0]))

    def test_boundary_circle_and_arc_are_not_section_lines(self):
        model = _model(
            arcs=[ArcGeometry(Point2D(0, 0), 5, 0, 180, "BOUNDARY")],
            circles=[CircleGeometry(Point2D(0, 0), 5, "BOUNDARY")],
        )
        self.assertEqual(generate_lawnmower_section_lines(model, TANK), [])

    def test_non_plate_width_horizontal_gap_is_not_generated(self):
        model = _model(
            _line(-4, -3, 4, -3),
            _line(-4, -1, 4, -1),
            _line(-4, 2, 4, 2),
            _line(-4, 4, 4, 4),
        )
        horizontal = _orientation(generate_lawnmower_section_lines(model, TANK), "horizontal")
        self.assertEqual(len(horizontal), 2)
        self.assertEqual([line.start_m[1] for line in horizontal], [-2.0, 3.0])

    def test_section_generation_never_builds_profiles_or_circular_sweeps(self):
        model = _model(_line(-1, -4, -1, 4), _line(1, -4, 1, 4))
        with patch("path_planner.make_rainbow_profile", side_effect=AssertionError("profile generation called")), patch(
            "path_planner._generate_circular_sweep_plan", side_effect=AssertionError("circular generation called")
        ):
            lines = generate_lawnmower_section_lines(model, TANK)
        self.assertTrue(lines)
        self.assertTrue(all(isinstance(line, LawnmowerSectionLine) for line in lines))
        self.assertTrue(all(not hasattr(line, "stage") for line in lines))

    def test_vertical_fragment_gap_is_not_bridged(self):
        model = _model(
            _line(-1, -4, -1, -1), _line(-1, 1, -1, 4),
            _line(1, -4, 1, -1), _line(1, 1, 1, 4),
        )
        vertical = _orientation(generate_lawnmower_section_lines(model, TANK), "vertical")
        self.assertEqual(len(vertical), 2)
        ranges = sorted((min(line.start_m[1], line.end_m[1]), max(line.start_m[1], line.end_m[1])) for line in vertical)
        self.assertEqual(ranges, [(-4.0, -1.0), (1.0, 4.0)])

    def test_short_vertical_weld_pair_extends_across_supported_fragment_bands(self):
        bands = [(-12, -7), (-6, -2), (-1, 3), (4, 8)]
        long_boundaries = [
            _line(x_m, start_m, x_m, end_m)
            for x_m in (-1, 1, 3)
            for start_m, end_m in bands
        ]
        short_pair = [_line(6, -10, 6, -7), _line(8, -10, 8, -7)]
        model = _model(*long_boundaries, *short_pair, bounds=(-14, -14, 14, 14))
        tank = TankCircleEstimate(center_x=0.0, center_y=0.0, radius=14.0, method="test")

        vertical = _orientation(generate_lawnmower_section_lines(model, tank), "vertical")
        added = [line for line in vertical if abs(line.start_m[0] - 7.0) < 1e-9]

        self.assertEqual(len(added), 4)
        self.assertEqual([(line.start_m[1], line.end_m[1]) for line in added], bands)

    def test_offset_tank_center_is_respected(self):
        tank = TankCircleEstimate(center_x=10.0, center_y=20.0, radius=5.0, method="test")
        model = _model(_line(9, 16, 9, 24), _line(11, 16, 11, 24), bounds=(5, 15, 15, 25))
        vertical = _orientation(generate_lawnmower_section_lines(model, tank), "vertical")
        self.assertEqual(len(vertical), 1)
        self.assertPointAlmostEqual(vertical[0].start_m, (10.0, 16.0))
        self.assertPointAlmostEqual(vertical[0].end_m, (10.0, 24.0))


if __name__ == "__main__":
    unittest.main()
