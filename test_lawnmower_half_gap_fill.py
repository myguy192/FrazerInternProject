import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import GeometryCollection, Polygon

from dxf_importer import import_dxf
from half_profile_gap_fill import (
    FullProfileRejectionReason,
    RejectedFullProfileCandidate,
    build_candidate_valid_region,
    evaluate_half_profile_salvage,
    exact_profile_metrics,
)
from path_planner import (
    LawnmowerSectionLine,
    SweepPose,
    TankCircleEstimate,
    _generate_lawnmower_placement_pass,
    _lawnmower_placements_to_sweep_poses,
    _lawnmower_spacing_for_overlap,
    _rainbow_long_side_endpoints,
    build_outer_edge_sweep_mission,
    estimate_tank_circle_from_geometry,
    generate_lawnmower_scan_placements,
    generate_lawnmower_section_lines,
    lawnmower_scan_profile_polygon,
    salvage_rejected_lawnmower_candidates,
    save_mission_json,
)
from scan_profile import make_rainbow_profile


def _guide(line_id, orientation, start, end):
    return LawnmowerSectionLine(
        line_id=line_id,
        orientation=orientation,
        order_index=line_id,
        source_section_id=line_id,
        start_m=start,
        end_m=end,
    )


@dataclass
class _SerializableMission:
    poses: list[SweepPose]


class RejectedFullCandidateHalfSalvageTests(unittest.TestCase):
    def test_public_full_placement_output_is_unchanged_by_rejection_capture(self):
        lines = [
            _guide(0, "vertical", (-1.0, -3.0), (-1.0, 3.0)),
            _guide(1, "horizontal", (-3.0, 1.0), (3.0, 1.0)),
        ]

        public = generate_lawnmower_scan_placements(lines)
        placement_pass = _generate_lawnmower_placement_pass(lines)

        self.assertEqual(public, placement_pass.accepted)
        self.assertEqual(len(placement_pass.rejected), 4)
        self.assertTrue(
            all(
                candidate.rejection_reasons
                == (FullProfileRejectionReason.SECTION_BOUNDARY_CONFLICT,)
                for candidate in placement_pass.rejected
            )
        )

    def test_duplicate_internal_candidates_are_retained_but_not_salvageable(self):
        lines = [
            _guide(0, "vertical", (0.0, -3.0), (0.0, 3.0)),
            _guide(1, "vertical", (0.0, -3.0), (0.0, 3.0)),
        ]

        placement_pass = _generate_lawnmower_placement_pass(lines)
        duplicate_rejections = [
            candidate
            for candidate in placement_pass.rejected
            if FullProfileRejectionReason.DUPLICATE_POSE in candidate.rejection_reasons
        ]

        self.assertTrue(duplicate_rejections)
        self.assertTrue(all(not candidate.is_salvageable for candidate in duplicate_rejections))
        self.assertTrue(all(candidate.full_polygon.shape[1] == 2 for candidate in duplicate_rejections))

    def test_exact_metrics_match_direct_shapely_operations(self):
        candidate_points = np.array(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]
        )
        valid_region = Polygon([(-0.5, -2.0), (2.0, -2.0), (2.0, 2.0), (-0.5, 2.0)])
        existing = Polygon([(0.0, -2.0), (2.0, -2.0), (2.0, 2.0), (0.0, 2.0)])

        metrics = exact_profile_metrics(candidate_points, valid_region, existing)
        candidate = Polygon(candidate_points)
        inside = candidate.intersection(valid_region)
        outside = candidate.difference(valid_region)
        overlap = inside.intersection(existing)
        new = inside.difference(existing)

        self.assertAlmostEqual(metrics.inside_area_m2, inside.area, places=12)
        self.assertAlmostEqual(metrics.outside_area_m2, outside.area, places=12)
        self.assertAlmostEqual(metrics.overlap_area_m2, overlap.area, places=12)
        self.assertAlmostEqual(metrics.new_area_m2, new.area, places=12)

    def test_selected_half_uses_exact_rejected_parent_transform(self):
        line = _guide(0, "vertical", (0.0, -3.0), (0.0, 3.0))
        placement = generate_lawnmower_scan_placements([line])[3]
        full_polygon = lawnmower_scan_profile_polygon(placement)
        rejected = RejectedFullProfileCandidate(
            candidate_id=0,
            guide_line_id=placement.guide_line_id,
            guide_order_index=placement.guide_order_index,
            placement_index=placement.placement_index,
            projected_order_m=3.0,
            orientation=placement.orientation,
            travel_direction=placement.travel_direction,
            anchor_m=placement.anchor_m,
            profile_origin_m=placement.profile_origin_m,
            heading_rad=placement.heading_rad,
            heading_deg=placement.heading_deg,
            full_polygon=full_polygon,
            rejection_reasons=(FullProfileRejectionReason.OUTSIDE_TANK,),
        )
        local = make_rainbow_profile()
        long_start, long_end = _rainbow_long_side_endpoints(local)
        anchor = (long_start + long_end) / 2.0
        valid_region = build_candidate_valid_region(
            line.start_m,
            line.end_m,
            (0.4, placement.anchor_m[1]),
            0.85,
            local,
            anchor,
            _lawnmower_spacing_for_overlap(local, 0.02),
        )

        evaluation = evaluate_half_profile_salvage(
            rejected,
            valid_region,
            GeometryCollection(),
        )

        self.assertEqual(evaluation.selected_variant, "right_half")
        self.assertEqual(evaluation.candidate.anchor_m, placement.anchor_m)
        self.assertEqual(evaluation.candidate.profile_origin_m, placement.profile_origin_m)
        self.assertEqual(evaluation.candidate.heading_rad, placement.heading_rad)

    def test_real_150ft_salvage_is_deterministic_and_parent_derived(self):
        model = import_dxf(Path("3 Tank examples") / "150ft.dxf")
        tank = estimate_tank_circle_from_geometry(model)
        lines = generate_lawnmower_section_lines(model, tank)
        placement_pass = _generate_lawnmower_placement_pass(lines)
        original_full = list(placement_pass.accepted)
        edge = build_outer_edge_sweep_mission(model, fixed_circular_rows=1)

        first = salvage_rejected_lawnmower_candidates(lines, placement_pass, tank, edge.poses)
        second = salvage_rejected_lawnmower_candidates(lines, placement_pass, tank, edge.poses)

        self.assertEqual(placement_pass.accepted, original_full)
        self.assertGreaterEqual(len(first.placements), 20)
        self.assertEqual(first.placements, second.placements)
        self.assertEqual(
            {placement.profile_variant for placement in first.placements},
            {"left_half", "right_half"},
        )
        self.assertEqual(
            {placement.orientation for placement in first.placements},
            {"vertical", "horizontal"},
        )
        rejected_parent_poses = {
            (
                candidate.guide_line_id,
                candidate.anchor_m,
                candidate.profile_origin_m,
                candidate.heading_rad,
            )
            for candidate in placement_pass.rejected
        }
        for placement in first.placements:
            self.assertIn(
                (
                    placement.guide_line_id,
                    placement.anchor_m,
                    placement.parent_profile_origin_m,
                    placement.heading_rad,
                ),
                rejected_parent_poses,
            )
        self.assertTrue(all(pose.profile_variant == "full" for pose in edge.poses))

    def test_pose_order_and_serialization_record_half_parent_metadata(self):
        line = _guide(0, "vertical", (0.0, -3.0), (0.0, 3.0))
        placement_pass = _generate_lawnmower_placement_pass([line])
        candidate = placement_pass.rejected[0]
        half = placement_pass.accepted[0].__class__(
            placement_id=100,
            guide_line_id=candidate.guide_line_id,
            guide_order_index=candidate.guide_order_index,
            placement_index=candidate.placement_index,
            orientation=candidate.orientation,
            travel_direction=candidate.travel_direction,
            anchor_m=candidate.anchor_m,
            profile_origin_m=candidate.profile_origin_m,
            heading_rad=candidate.heading_rad,
            heading_deg=candidate.heading_deg,
            profile_variant="left_half",
            parent_profile_origin_m=candidate.profile_origin_m,
        )
        poses = _lawnmower_placements_to_sweep_poses(
            [line],
            placement_pass.accepted + [half],
            scan_id_start=10,
        )

        self.assertEqual(poses[0].profile_variant, "left_half")
        self.assertEqual([pose.scan_id for pose in poses], list(range(10, 10 + len(poses))))
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "mission.json"
            save_mission_json(_SerializableMission(poses), output)
            data = json.loads(output.read_text(encoding="utf-8"))
        serialized = data["poses"][0]
        self.assertEqual(serialized["profile_variant"], "left_half")
        self.assertEqual(serialized["parent_full_x_m"], candidate.profile_origin_m[0])
        self.assertEqual(serialized["parent_full_y_m"], candidate.profile_origin_m[1])


if __name__ == "__main__":
    unittest.main()
