import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from dxf_importer import Bounds, GeometryModel, LineSegment, Point2D
from path_planner import (
    _build_mission_plan_once,
    _lawnmower_placements_to_sweep_poses,
    build_outer_edge_sweep_mission,
    determine_max_feasible_circular_rows,
    generate_lawnmower_scan_placements,
    generate_lawnmower_section_lines,
    plot_outer_edge_sweep,
    save_mission_json,
)


def _line(x1, y1, x2, y2):
    return LineSegment(Point2D(x1, y1), Point2D(x2, y2), "TEST")


def _model():
    return GeometryModel(
        source_path=None,
        line_segments=[
            _line(-1, -4, -1, 4),
            _line(1, -4, 1, 4),
            _line(-4, -1, 4, -1),
            _line(-4, 1, 4, 1),
        ],
        arcs=[],
        circles=[],
        bounds=Bounds(-5, -5, 5, 5),
    )


class LawnmowerMissionIntegrationTests(unittest.TestCase):
    def test_geometry_limit_transitions_naturally_from_one_to_two_rows(self):
        self.assertEqual(determine_max_feasible_circular_rows(3.7), 1)
        self.assertEqual(determine_max_feasible_circular_rows(4.5), 2)

    def test_robot_turning_radius_can_reduce_the_geometry_limit(self):
        self.assertEqual(
            determine_max_feasible_circular_rows(
                10.0,
                minimum_turning_radius_m=9.0,
            ),
            1,
        )

    def test_fixed_two_row_request_is_capped_by_small_tank_geometry(self):
        small_model = GeometryModel(
            source_path=None,
            line_segments=[],
            arcs=[],
            circles=[],
            bounds=Bounds(-3.7, -3.7, 3.7, 3.7),
        )

        mission = build_outer_edge_sweep_mission(
            small_model,
            fixed_circular_rows=2,
        )

        self.assertEqual(len(mission.circular_rows), 1)
        self.assertEqual(mission.circular_stop_reason, "geometry_row_limit")

    def test_normal_mission_uses_only_approved_lawnmower_poses_after_circular_stage(self):
        model = _model()
        lines = generate_lawnmower_section_lines(model)
        placements = generate_lawnmower_scan_placements(lines)
        edge_mission = build_outer_edge_sweep_mission(model, fixed_circular_rows=1)
        expected_interior = _lawnmower_placements_to_sweep_poses(
            lines,
            placements,
            scan_id_start=len(edge_mission.poses),
        )

        with patch(
            "path_planner.predict_tank_layout_from_geometry",
            side_effect=AssertionError("predictor code was invoked"),
        ):
            mission = _build_mission_plan_once(model, fixed_circular_rows=1)

        circular_count = len(edge_mission.poses)
        self.assertEqual(mission.poses[:circular_count], edge_mission.poses)
        self.assertEqual(mission.circular_rows, edge_mission.circular_rows)
        self.assertEqual(mission.poses[circular_count:], expected_interior)
        self.assertEqual(mission.lawnmower_section_lines, lines)
        self.assertEqual(mission.edge_sweep_scan_count, circular_count)
        self.assertEqual(mission.interior_discarded_count, 0)
        self.assertEqual(mission.duplicate_poses_removed, 0)
        self.assertFalse(any(pose.stage == "interior_side_guard" for pose in mission.poses))

        stages = [pose.stage for pose in mission.poses]
        self.assertEqual(stages[:circular_count], ["circular_edge"] * circular_count)
        vertical_count = sum(placement.orientation == "vertical" for placement in placements)
        horizontal_count = sum(placement.orientation == "horizontal" for placement in placements)
        self.assertEqual(stages[circular_count : circular_count + vertical_count], ["interior_vertical"] * vertical_count)
        self.assertEqual(stages[circular_count + vertical_count :], ["interior_horizontal"] * horizontal_count)
        self.assertEqual(mission.vertical_scan_count, vertical_count)
        self.assertEqual(mission.horizontal_scan_count, horizontal_count)

        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "mission.json"
            save_mission_json(mission, json_path)
            data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["total_scan_count"], len(mission.poses))
        self.assertEqual(len(data["lawnmower_section_lines"]), len(lines))

        figure, _axes = plot_outer_edge_sweep(
            model,
            mission,
            show=False,
            save_path=None,
            max_profile_draw_count=24,
        )
        self.assertIsNotNone(figure)
        figure.clf()

    def test_normal_mission_does_not_call_experimental_outer_guide_generation(self):
        with patch(
            "path_planner._add_outer_lawnmower_sections",
            side_effect=AssertionError("experimental outer guides were invoked"),
        ):
            mission = _build_mission_plan_once(_model(), fixed_circular_rows=1)

        self.assertFalse(
            any(line.is_outer_extension for line in mission.lawnmower_section_lines)
        )


if __name__ == "__main__":
    unittest.main()
