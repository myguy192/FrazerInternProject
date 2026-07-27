import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
from matplotlib.quiver import Quiver
from matplotlib.patches import Circle
from shapely.geometry import Polygon

matplotlib.use("Agg")

from dxf_importer import Bounds, GeometryModel, LineSegment, Point2D, import_dxf
from path_planner import (
    MIN_CIRCULAR_OVERLAP_FRACTION,
    _build_mission_plan_once,
    _exact_circular_neighbor_overlap_fraction,
    _lawnmower_placements_to_sweep_poses,
    build_outer_edge_sweep_mission,
    determine_max_feasible_circular_rows,
    estimate_tank_circle_from_geometry,
    gate_lawnmower_candidates,
    generate_lawnmower_section_lines,
    plot_outer_edge_sweep,
    scan_pose_profile_polygon,
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
    def test_24ft_uses_one_stronger_circular_row_and_four_symmetric_vertical_columns(self):
        model = import_dxf(Path("3 Tank examples") / "24ft.dxf")

        mission = _build_mission_plan_once(model)
        vertical_lines = [
            line
            for line in mission.lawnmower_section_lines
            if line.orientation == "vertical"
        ]
        horizontal_lines = [
            line
            for line in mission.lawnmower_section_lines
            if line.orientation == "horizontal"
        ]

        self.assertEqual(len(mission.circular_rows), 1)
        self.assertEqual(mission.circular_rows[0].scan_count, 35)
        self.assertGreater(mission.circular_rows[0].minimum_neighbor_overlap_fraction, 0.055)
        self.assertEqual(horizontal_lines, [])
        self.assertEqual(mission.horizontal_scan_count, 0)
        self.assertEqual(len(vertical_lines), 4)
        self.assertFalse(any(line.is_outer_extension for line in vertical_lines))

        ordered_lines = sorted(vertical_lines, key=lambda line: line.start_m[0])
        x_values = [line.start_m[0] for line in ordered_lines]
        self.assertAlmostEqual(x_values[1] - x_values[0], x_values[2] - x_values[1], places=9)
        self.assertAlmostEqual(x_values[2] - x_values[1], x_values[3] - x_values[2], places=9)
        poses_by_guide = {
            line.line_id: [
                pose
                for pose in mission.poses
                if pose.stage == "interior_vertical" and pose.section_id == line.line_id
            ]
            for line in ordered_lines
        }
        for line in (ordered_lines[0], ordered_lines[3]):
            poses = poses_by_guide[line.line_id]
            self.assertTrue(poses)
            self.assertTrue(all(pose.profile_variant != "full" for pose in poses))
            for pose in poses:
                centroid_x = float(Polygon(scan_pose_profile_polygon(pose)).centroid.x)
                self.assertLess(
                    abs(centroid_x - mission.tank_center_m["x"]),
                    abs(pose.anchor_x_m - mission.tank_center_m["x"]),
                )
                self.assertAlmostEqual(pose.parent_full_x_m, pose.x_m, places=9)
                self.assertAlmostEqual(pose.parent_full_y_m, pose.y_m, places=9)
        inner_left = poses_by_guide[ordered_lines[1].line_id]
        inner_right = poses_by_guide[ordered_lines[2].line_id]
        self.assertTrue(inner_left)
        self.assertEqual(len(inner_left), len(inner_right))
        self.assertTrue(all(pose.profile_variant == "full" for pose in inner_left + inner_right))
        ordered_anchor_lists = [
            [
                pose.anchor_y_m
                for pose in sorted(
                    poses_by_guide[line.line_id],
                    key=lambda pose: pose.anchor_y_m,
                )
            ]
            for line in ordered_lines
        ]
        self.assertTrue(ordered_anchor_lists[0])
        self.assertTrue(
            all(anchors == ordered_anchor_lists[0] for anchors in ordered_anchor_lists)
        )
        self.assertEqual(
            [len(poses_by_guide[line.line_id]) for line in ordered_lines],
            [6, 6, 6, 6],
        )
        self.assertAlmostEqual(
            x_values[0] + x_values[3],
            2.0 * mission.tank_center_m["x"],
            places=9,
        )
        self.assertAlmostEqual(
            x_values[1] + x_values[2],
            2.0 * mission.tank_center_m["x"],
            places=9,
        )

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

    def test_every_circular_neighbor_has_exact_overlap_including_wraparound(self):
        mission = build_outer_edge_sweep_mission(_model())

        self.assertTrue(mission.circular_rows)
        for row in mission.circular_rows:
            poses = [pose for pose in mission.poses if pose.row_id == row.row_id]
            polygons = [scan_pose_profile_polygon(pose) for pose in poses]
            overlaps = [
                _exact_circular_neighbor_overlap_fraction(
                    polygons[index],
                    polygons[(index + 1) % len(polygons)],
                )
                for index in range(len(polygons))
            ]
            self.assertTrue(
                all(overlap >= MIN_CIRCULAR_OVERLAP_FRACTION for overlap in overlaps)
            )
            self.assertAlmostEqual(row.wraparound_overlap_fraction, overlaps[-1], places=10)
            self.assertTrue(row.no_neighbor_gaps_verified)
            self.assertTrue(row.wraparound_passed)

    def test_normal_mission_uses_only_approved_lawnmower_poses_after_circular_stage(self):
        model = _model()
        lines = generate_lawnmower_section_lines(model)
        edge_mission = build_outer_edge_sweep_mission(model, fixed_circular_rows=1)
        gating_pass = gate_lawnmower_candidates(
            lines,
            estimate_tank_circle_from_geometry(model),
            edge_mission.poses,
        )
        placements = gating_pass.placements
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
        self.assertEqual(
            mission.interior_discarded_count,
            len(gating_pass.candidates) - len(gating_pass.placements),
        )
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
        labels = {line.get_label() for line in _axes.lines}
        self.assertNotIn("Ordered mission path", labels)
        self.assertNotIn("Vertical travel path", labels)
        self.assertNotIn("Horizontal travel path", labels)
        self.assertFalse(any(isinstance(collection, Quiver) for collection in _axes.collections))
        self.assertFalse(
            any(
                isinstance(patch, Circle) and patch.get_linestyle() in {"--", ":"}
                for patch in _axes.patches
            )
        )
        profile_outlines = [
            line for line in _axes.lines if line.get_label() == "_scan_profile_outline"
        ]
        self.assertTrue(profile_outlines)
        self.assertTrue(all(line.get_linewidth() >= 1.25 for line in profile_outlines))
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
