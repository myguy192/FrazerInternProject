import math
import unittest
from pathlib import Path

from dxf_importer import import_dxf
from tank_layout_predictor import (
    LayoutPoint,
    ObservedSegment,
    ObservedTankGeometry,
    TankGeometry,
    mask_observations_to_annulus,
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


class TankLayoutPredictorTests(unittest.TestCase):
    def test_no_dxf_runtime_observations_predict_layout(self):
        full = _synthetic_staggered_layout(10.0, 3.0, -2.0)
        observed = mask_observations_to_annulus(full, 0.65)

        layout = predict_tank_layout(observed)

        self.assertIn(layout.selected_pattern_family, {"staggered_plate_grid", "orthogonal_plate_grid"})
        self.assertGreater(len(layout.observed_weld_segments), 0)
        self.assertGreater(len(layout.predicted_weld_segments), 0)
        self.assertTrue(all(segment.source == "observed" for segment in layout.observed_weld_segments))
        self.assertTrue(all(segment.source == "predicted" for segment in layout.predicted_weld_segments))
        self.assertTrue(all(segment.prediction_method for segment in layout.predicted_weld_segments))

    def test_uniform_scale_preserves_normalized_pattern(self):
        small = predict_tank_layout(mask_observations_to_annulus(_synthetic_staggered_layout(5.0), 0.65))
        large = predict_tank_layout(mask_observations_to_annulus(_synthetic_staggered_layout(17.0), 0.65))

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
        layout = predict_tank_layout(mask_observations_to_annulus(full, 0.65))

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
        examples = Path(__file__).parent / "3 Tank examples"
        for filename in ("24ft.dxf", "65ft.dxf", "150ft.dxf"):
            with self.subTest(filename=filename):
                full = observed_geometry_from_dxf(import_dxf(examples / filename))
                inner_fraction = max(0.1, 1.0 - 3.2 / full.tank.radius)
                masked = mask_observations_to_annulus(full, inner_fraction)
                original_layout = predict_tank_layout(masked)
                report = validate_prediction(original_layout, full)

                self.assertTrue(all(metric.false_predictions == 0 for metric in report.orientations))

                scaled_full = scale_observed_geometry(full, 0.67, new_center=(4.0, -3.0))
                scaled_masked = mask_observations_to_annulus(scaled_full, inner_fraction)
                scaled_layout = predict_tank_layout(scaled_masked)
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


if __name__ == "__main__":
    unittest.main()
