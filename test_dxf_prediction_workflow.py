import json
import math
import tempfile
import unittest
from pathlib import Path

from dxf_importer import import_dxf
from dxf_predictor import (
    LayoutPoint,
    ObservedSegment,
    ObservedTankGeometry,
    TankGeometry,
    load_reference_families,
    predict_from_observations,
    reconstruction_metrics,
)
from missing_dxf_generator import (
    SelectedSweep,
    generate_partial_observation,
    generate_selected_sweep_from_planner,
    load_selected_sweep_from_mission,
    save_partial_dxf,
)
from observation_region import build_observed_region_from_scan_poses
from scan_profile import RainbowProfileConfig
from tank_layout_predictor import clip_geometry_to_observed_region, observed_geometry_from_dxf


ROOT = Path(__file__).parent
EXAMPLES = ROOT / "3 Tank examples"


class DxfPredictionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.families = load_reference_families(EXAMPLES)
        cls.models = {name: import_dxf(EXAMPLES / f"{name}.dxf") for name in ("24ft", "65ft", "150ft")}
        cls.base_cases = {}

    def _base_case(self, name, rows):
        key = (name, rows)
        if key not in self.base_cases:
            selected = generate_selected_sweep_from_planner(self.models[name], rows)
            full, observed, region = generate_partial_observation(self.models[name], selected)
            result = predict_from_observations(observed, self.families, region)
            self.base_cases[key] = (full, observed, region, result)
        return self.base_cases[key]

    def test_exact_family_reconstruction_with_one_and_two_rows(self):
        for name in ("24ft", "65ft", "150ft"):
            for rows in (1, 2):
                with self.subTest(name=name, rows=rows):
                    full, _, _, result = self._base_case(name, rows)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.selected_family, f"{name}_family")
                    metrics = reconstruction_metrics(result, full)
                    self.assertGreaterEqual(metrics["vertical"]["recall"], 0.90)
                    self.assertGreaterEqual(metrics["horizontal"]["recall"], 0.80)

    def test_scaled_variants_keep_original_family_for_one_and_two_rows(self):
        scale_factors = {"24ft": 1.25, "65ft": 0.78, "150ft": 2.0 / 3.0}
        for name, scale_factor in scale_factors.items():
            original = observed_geometry_from_dxf(self.models[name])
            scaled = _transform_geometry(original, scale_factor, 0.0, (4.0, -3.0))
            for rows in (1, 2):
                with self.subTest(name=name, rows=rows):
                    selected = _sweep_for_tank(scaled.tank, rows)
                    region = build_observed_region_from_scan_poses(selected.poses, selected.profile_config)
                    observed = clip_geometry_to_observed_region(scaled, region)
                    result = predict_from_observations(observed, self.families, region)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.selected_family, f"{name}_family")

    def test_translated_and_rotated_target(self):
        original = observed_geometry_from_dxf(self.models["65ft"])
        transformed = _transform_geometry(original, 0.91, math.radians(4.0), (12.0, -7.0))
        selected = _sweep_for_tank(transformed.tank, 2, start_angle=math.radians(4.0))
        region = build_observed_region_from_scan_poses(selected.poses, selected.profile_config)
        observed = clip_geometry_to_observed_region(transformed, region)
        result = predict_from_observations(observed, self.families, region)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.selected_family, "65ft_family")
        self.assertAlmostEqual(result.rotation_deg, 4.0, delta=0.5)

    def test_ambiguous_observation_does_not_complete(self):
        full = observed_geometry_from_dxf(self.models["24ft"])
        single = full.segments[0]
        direction_x = single.end.x - single.start.x
        direction_y = single.end.y - single.start.y
        ambiguous = ObservedTankGeometry(
            tank=full.tank,
            segments=[
                ObservedSegment(
                    start=single.start,
                    end=LayoutPoint(single.start.x + direction_x * 0.08, single.start.y + direction_y * 0.08),
                    source="runtime_observation",
                    source_id="ambiguous:0",
                )
            ],
        )
        result = predict_from_observations(
            ambiguous,
            self.families,
            None,
            minimum_score=0.65,
            minimum_margin=0.12,
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.predicted_segments, [])

    def test_mission_selection_uses_exact_requested_rows(self):
        poses = []
        for row_id in range(3):
            for index in range(2):
                poses.append(
                    {
                        "stage": "circular_edge",
                        "row_id": row_id,
                        "x_m": row_id + index + 1.0,
                        "y_m": 0.0,
                        "heading_rad": 0.0,
                        "sweep_radius_m": 5.0 - row_id,
                    }
                )
        poses.append({"stage": "interior_vertical", "x_m": 0.0, "y_m": 0.0, "heading_rad": 0.0})
        with tempfile.TemporaryDirectory() as folder:
            mission_path = Path(folder) / "mission.json"
            mission_path.write_text(json.dumps({"poses": poses}), encoding="utf-8")
            selected = load_selected_sweep_from_mission(mission_path, 2)

        self.assertEqual(selected.row_ids, ["row:0", "row:1"])
        self.assertEqual(len(selected.poses), 4)
        self.assertTrue(all(pose["row_id"] in (0, 1) for pose in selected.poses))

    def test_partial_dxf_contains_only_sanitized_layers(self):
        _, observed, region, _ = self._base_case("24ft", 1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "partial.dxf"
            save_partial_dxf(observed, region, output)
            imported = import_dxf(output)

        self.assertEqual(
            imported.layer_names,
            {"TANK_BOUNDARY", "OBSERVED_CIRCULAR_SCAN", "SCAN_FOOTPRINTS"},
        )


def _sweep_for_tank(tank, rows, start_angle=0.0):
    from path_planner import DEFAULT_ROW_SPACING, _touching_angular_step

    profile = RainbowProfileConfig()
    sweep_radius = tank.radius - profile.width / 2.0
    poses = []
    for row_id in range(rows):
        touching_angle = _touching_angular_step(sweep_radius, profile)
        pose_count = max(3, math.ceil(2.0 * math.pi / (touching_angle * 0.80)))
        for index in range(pose_count):
            theta = start_angle + 2.0 * math.pi * index / pose_count
            poses.append(
                {
                    "stage": "circular_edge",
                    "status": "kept",
                    "row_id": row_id,
                    "x_m": tank.center_x + sweep_radius * math.cos(theta),
                    "y_m": tank.center_y + sweep_radius * math.sin(theta),
                    "heading_rad": theta,
                    "sweep_radius_m": sweep_radius,
                }
            )
        sweep_radius -= DEFAULT_ROW_SPACING
    return SelectedSweep(poses, [str(index) for index in range(rows)], profile, "test_fixed_rows")


def _transform_geometry(geometry, scale, rotation, center):
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    target_tank = TankGeometry(center[0], center[1], geometry.tank.radius * scale)

    def transform(point):
        nx = (point.x - geometry.tank.center_x) / geometry.tank.radius
        ny = (point.y - geometry.tank.center_y) / geometry.tank.radius
        rx = nx * cos_r - ny * sin_r
        ry = nx * sin_r + ny * cos_r
        return LayoutPoint(center[0] + rx * target_tank.radius, center[1] + ry * target_tank.radius)

    return ObservedTankGeometry(
        tank=target_tank,
        segments=[
            ObservedSegment(
                start=transform(segment.start),
                end=transform(segment.end),
                source=segment.source,
                confidence=segment.confidence,
                source_id=segment.source_id,
                metadata=dict(segment.metadata),
            )
            for segment in geometry.segments
        ],
        units=geometry.units,
        source_metadata={"synthetic_transform": {"scale": scale, "rotation_rad": rotation}},
    )


if __name__ == "__main__":
    unittest.main()
