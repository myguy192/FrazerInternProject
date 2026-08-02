import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dxf_importer import import_dxf
from observation_region import (
    build_observed_region_from_scan_poses,
    points_in_observed_region,
)
from path_planner import build_mission_plan
from scan_profile import RainbowProfileConfig
from grid_predictor import (
    LayoutPoint,
    ObservedSegment,
    ObservedTankGeometry,
    TankGeometry,
    clip_geometry_to_observed_region,
    observed_geometry_from_circular_sweeps,
    observed_geometry_from_dxf,
    predict_tank_layout,
    scale_observed_geometry,
    validate_prediction,
)


def _synthetic_staggered_layout(radius: float, center_x: float = 0.0, center_y: float = 0.0):
    tank = TankGeometry(center_x, center_y, radius)
    segments = []
    vertical_positions = [-0.6, -0.2, 0.2, 0.6]
    for index, x_norm in enumerate(vertical_positions):
        y_limit = math.sqrt(1.0 - x_norm * x_norm)
        segments.append(
            _world_segment(
                tank,
                (x_norm, -y_limit),
                (x_norm, y_limit),
                f"runtime-v-{index}",
            )
        )

    boundaries = [-1.0, *vertical_positions, 1.0]
    for column, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        phase = -0.8 if column % 2 == 0 else -0.6
        level = phase
        while level < 1.0:
            x_limit = math.sqrt(max(0.0, 1.0 - level * level))
            start_x = max(left, -x_limit)
            end_x = min(right, x_limit)
            if end_x - start_x > 0.03:
                segments.append(
                    _world_segment(
                        tank,
                        (start_x, level),
                        (end_x, level),
                        f"runtime-h-{column}-{level:.2f}",
                    )
                )
            level += 0.4
    return ObservedTankGeometry(
        tank=tank,
        segments=segments,
        source_metadata={"source_type": "runtime_test"},
    )


def _world_segment(tank, start, end, source_id):
    return ObservedSegment(
        LayoutPoint(tank.center_x + start[0] * tank.radius, tank.center_y + start[1] * tank.radius),
        LayoutPoint(tank.center_x + end[0] * tank.radius, tank.center_y + end[1] * tank.radius),
        source_id=source_id,
    )


def _circular_observation(full, row_fractions=(0.82, 0.62)):
    tank = full.tank
    profile = RainbowProfileConfig(
        width=tank.radius * 0.24,
        arc_radius=tank.radius * 0.14,
        side_height=tank.radius * 0.08,
        arc_samples=32,
    )
    poses = []
    for row_id, radius_fraction in enumerate(row_fractions):
        pose_count = 32 if row_id == 0 else 24
        for index in range(pose_count):
            theta = 2.0 * math.pi * index / pose_count
            sweep_radius = tank.radius * radius_fraction
            poses.append(
                SimpleNamespace(
                    stage="circular_edge",
                    status="kept",
                    x_m=tank.center_x + sweep_radius * math.cos(theta),
                    y_m=tank.center_y + sweep_radius * math.sin(theta),
                    heading_rad=theta,
                )
            )
    poses.append(SimpleNamespace(stage="interior_vertical", status="kept", x_m=0.0, y_m=0.0, heading_rad=0.0))
    return observed_geometry_from_circular_sweeps(full, poses, profile)


class GridPredictorTests(unittest.TestCase):
    def test_no_dxf_runtime_observations_predict_layout(self):
        full = _synthetic_staggered_layout(10.0, 3.0, -2.0)
        observed, region = _circular_observation(full)

        layout = predict_tank_layout(observed)

        self.assertEqual(region.circular_pose_count, 56)
        self.assertIn(layout.selected_pattern_family, {"staggered_plate_grid", "orthogonal_plate_grid"})
        self.assertGreater(len(layout.observed_weld_segments), 0)
        self.assertGreater(len(layout.predicted_weld_segments), 0)
        self.assertTrue(
            all(segment.source == "observed_circular_scan" for segment in layout.observed_weld_segments)
        )
        self.assertTrue(all(segment.source == "predicted" for segment in layout.predicted_weld_segments))
        self.assertTrue(all(segment.prediction_method for segment in layout.predicted_weld_segments))

    def test_uniform_scale_preserves_normalized_pattern(self):
        small_observed, _ = _circular_observation(_synthetic_staggered_layout(5.0))
        large_observed, _ = _circular_observation(_synthetic_staggered_layout(17.0))
        small = predict_tank_layout(small_observed)
        large = predict_tank_layout(large_observed)

        small_parameters = small.normalized_pattern_parameters
        large_parameters = large.normalized_pattern_parameters
        self.assertAlmostEqual(
            small_parameters.vertical_spacing_normalized,
            large_parameters.vertical_spacing_normalized,
            places=8,
        )
        self.assertAlmostEqual(
            small_parameters.horizontal_spacing_normalized,
            large_parameters.horizontal_spacing_normalized,
            places=8,
        )
        self.assertEqual(
            len(small.predicted_weld_segments),
            len(large.predicted_weld_segments),
        )

    def test_validation_report_has_required_metrics(self):
        full = _synthetic_staggered_layout(8.0)
        observed, _ = _circular_observation(full)
        layout = predict_tank_layout(observed)

        report = validate_prediction(layout, full)

        self.assertEqual({item.orientation for item in report.orientations}, {"vertical", "horizontal"})
        self.assertTrue(all(item.false_predictions >= 0 for item in report.orientations))
        self.assertIsInstance(report.confidence_distribution, dict)

    def test_too_few_observations_returns_partial_result(self):
        observed = ObservedTankGeometry(
            tank=TankGeometry(0.0, 0.0, 5.0),
            segments=[
                ObservedSegment(LayoutPoint(-1.0, 0.0), LayoutPoint(1.0, 0.0), source_id="single")
            ],
        )

        layout = predict_tank_layout(observed)

        self.assertEqual(layout.selected_pattern_family, "partial_structural_pattern")
        self.assertEqual(layout.predicted_weld_segments, [])
        self.assertTrue(any("Too few" in warning for warning in layout.warnings))

    def test_known_dxf_masking_and_scaled_variants(self):
        examples = Path(__file__).resolve().parents[1] / "examples" / "inputs"
        for filename in ("24ft.dxf", "65ft.dxf", "150ft.dxf"):
            with self.subTest(filename=filename):
                full = observed_geometry_from_dxf(import_dxf(examples / filename))
                observed, _ = _circular_observation(full)
                original_layout = predict_tank_layout(observed)
                report = validate_prediction(original_layout, full)

                self.assertTrue(all(metric.false_predictions == 0 for metric in report.orientations))

                scaled_full = scale_observed_geometry(full, 0.67, new_center=(4.0, -3.0))
                scaled_observed, _ = _circular_observation(scaled_full)
                scaled_layout = predict_tank_layout(scaled_observed)
                original_parameters = original_layout.normalized_pattern_parameters
                scaled_parameters = scaled_layout.normalized_pattern_parameters

                self.assertEqual(original_layout.selected_pattern_family, scaled_layout.selected_pattern_family)
                self.assertAlmostEqual(
                    original_parameters.vertical_spacing_normalized,
                    scaled_parameters.vertical_spacing_normalized,
                    places=7,
                )
                self.assertAlmostEqual(
                    original_parameters.horizontal_spacing_normalized,
                    scaled_parameters.horizontal_spacing_normalized,
                    places=7,
                )

    def test_24ft_mission_footprints_clip_hidden_geometry(self):
        root = Path(__file__).resolve().parents[1]
        model = import_dxf(root / "examples" / "inputs" / "24ft.dxf")
        full = observed_geometry_from_dxf(model)
        mission = build_mission_plan(model)
        profile = RainbowProfileConfig(
            width=float(mission.scan_profile["width_m"]),
            arc_radius=float(mission.scan_profile["arc_radius_m"]),
            side_height=float(mission.scan_profile["side_height_m"]),
            arc_samples=int(mission.scan_profile["arc_samples"]),
        )
        region = build_observed_region_from_scan_poses(mission.poses, profile)
        observed = clip_geometry_to_observed_region(full, region)
        layout = predict_tank_layout(observed)

        self.assertGreater(region.circular_pose_count, 0)
        self.assertTrue(observed.segments)
        self.assertTrue(all(segment.source == "observed_circular_scan" for segment in observed.segments))
        self.assertTrue(
            all(segment.source == "observed_circular_scan" for segment in layout.observed_weld_segments)
        )
        full_length = sum(
            math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)
            for segment in full.segments
        )
        observed_length = sum(
            math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)
            for segment in observed.segments
        )
        self.assertLess(observed_length, full_length)
        midpoints = np.array(
            [
                [(segment.start.x + segment.end.x) / 2.0, (segment.start.y + segment.end.y) / 2.0]
                for segment in observed.segments
            ]
        )
        self.assertTrue(np.all(points_in_observed_region(midpoints, region)))

    def test_interior_scan_poses_do_not_expand_observed_region(self):
        profile = RainbowProfileConfig()
        circular = SimpleNamespace(stage="circular_edge", x_m=3.0, y_m=0.0, heading_rad=0.0)
        interior = SimpleNamespace(stage="interior_vertical", x_m=0.0, y_m=0.0, heading_rad=0.0)
        region = build_observed_region_from_scan_poses([circular, interior], profile)

        self.assertEqual(region.circular_pose_count, 1)
        self.assertFalse(points_in_observed_region(np.array([[0.0, 0.0]]), region)[0])


if __name__ == "__main__":
    unittest.main()
