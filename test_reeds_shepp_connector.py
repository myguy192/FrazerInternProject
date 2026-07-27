import math
import unittest

from reeds_shepp_connector import (
    MAX_ENDPOINT_HEADING_ERROR_RAD,
    MAX_ENDPOINT_POSITION_ERROR_M,
    Pose2D,
    plan_reeds_shepp,
    sample_segments_for_visualization,
    validate_path_in_circle,
)


def _angle_error(first, second):
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


class ReedsSheppConnectorTests(unittest.TestCase):
    def test_identical_start_and_goal(self):
        pose = Pose2D(1.0, -2.0, 0.4)
        path = plan_reeds_shepp(pose, pose, 1.2)
        self.assertEqual(path.segments, [])
        self.assertEqual(path.samples, [pose])
        self.assertEqual(path.total_length_m, 0.0)
        self.assertTrue(path.endpoint_valid)

    def test_straight_forward_path(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 0.0, 0.0), 1.0)
        nonzero = [segment for segment in path.segments if segment.length_m > 1e-10]
        self.assertEqual(len(nonzero), 1)
        self.assertEqual(nonzero[0].segment_type, "S")
        self.assertEqual(nonzero[0].direction, "forward")
        self.assertAlmostEqual(nonzero[0].length_m, 3.0, places=10)

    def test_straight_reverse_path_and_propagation(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(-3.0, 0.0, 0.0), 1.0)
        nonzero = [segment for segment in path.segments if segment.length_m > 1e-10]
        self.assertEqual(len(nonzero), 1)
        self.assertEqual(nonzero[0].segment_type, "S")
        self.assertEqual(nonzero[0].direction, "reverse")
        self.assertTrue(all(a.x >= b.x for a, b in zip(path.samples, path.samples[1:])))
        self.assertTrue(all(abs(sample.y) < 1e-10 for sample in path.samples))
        self.assertTrue(all(_angle_error(sample.heading, 0.0) < 1e-10 for sample in path.samples))

    def test_same_position_with_different_heading(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, math.pi), 1.0)
        self.assertTrue(path.endpoint_valid)
        self.assertTrue(any(segment.direction == "reverse" for segment in path.segments))
        self.assertGreater(path.total_length_m, 0.0)

    def test_turning_path_uses_forward_and_reverse(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 1.0, math.pi), 1.0)
        self.assertTrue(any(segment.segment_type in {"L", "R"} for segment in path.segments))
        self.assertEqual({segment.direction for segment in path.segments}, {"forward", "reverse"})

    def test_translation_invariance(self):
        base = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.2), Pose2D(2.0, 1.0, 2.1), 0.8)
        shifted = plan_reeds_shepp(Pose2D(5.0, -3.0, 0.2), Pose2D(7.0, -2.0, 2.1), 0.8)
        self.assertEqual(base.word, shifted.word)
        self.assertEqual(len(base.segments), len(shifted.segments))
        for first, second in zip(base.segments, shifted.segments):
            self.assertAlmostEqual(first.signed_length_m, second.signed_length_m, places=9)

    def test_rotation_invariance(self):
        angle = 0.7
        start = Pose2D(0.0, 0.0, 0.0)
        goal = Pose2D(2.0, 1.0, 2.2)
        base = plan_reeds_shepp(start, goal, 0.9)
        rotated_goal = Pose2D(
            math.cos(angle) * goal.x - math.sin(angle) * goal.y,
            math.sin(angle) * goal.x + math.cos(angle) * goal.y,
            goal.heading + angle,
        )
        rotated = plan_reeds_shepp(Pose2D(0.0, 0.0, angle), rotated_goal, 0.9)
        self.assertEqual(base.word, rotated.word)
        for first, second in zip(base.segments, rotated.segments):
            self.assertAlmostEqual(first.signed_length_m, second.signed_length_m, places=9)

    def test_requested_turning_radius_is_used_for_every_curve(self):
        radius = 1.3
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(1.2, 1.5, math.pi), radius)
        curved_count = 0
        for segment, poses in sample_segments_for_visualization(path, 0.025):
            if segment.segment_type == "S" or segment.length_m < 1e-10:
                continue
            curved_count += 1
            start = poses[0]
            side = 1.0 if segment.segment_type == "L" else -1.0
            center_x = start.x - side * radius * math.sin(start.heading)
            center_y = start.y + side * radius * math.cos(start.heading)
            for pose in poses:
                self.assertAlmostEqual(
                    math.hypot(pose.x - center_x, pose.y - center_y),
                    radius,
                    places=8,
                )
        self.assertGreater(curved_count, 0)

    def test_exact_start_sample_and_endpoint_reconstruction(self):
        start = Pose2D(-1.0, 0.5, -0.3)
        goal = Pose2D(2.0, -1.0, 1.4)
        path = plan_reeds_shepp(start, goal, 0.75)
        self.assertEqual(path.samples[0], start)
        self.assertLessEqual(path.endpoint_position_error_m, MAX_ENDPOINT_POSITION_ERROR_M)
        self.assertLessEqual(path.endpoint_heading_error_rad, MAX_ENDPOINT_HEADING_ERROR_RAD)
        self.assertAlmostEqual(path.samples[-1].x, goal.x, places=9)
        self.assertAlmostEqual(path.samples[-1].y, goal.y, places=9)

    def test_wrapped_heading_error(self):
        goal = Pose2D(1.0, 0.5, 2.0 * math.pi + 0.2)
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.2), goal, 1.0)
        self.assertTrue(path.endpoint_valid)
        self.assertLess(path.endpoint_heading_error_rad, 1e-9)
        self.assertLess(_angle_error(path.samples[-1].heading, 0.2), 1e-9)

    def test_reverse_curves_update_heading_and_cusps_are_continuous(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, math.pi), 1.0)
        groups = sample_segments_for_visualization(path, 0.03)
        reverse_curve_seen = False
        for index, (segment, poses) in enumerate(groups):
            if index:
                previous_end = groups[index - 1][1][-1]
                self.assertAlmostEqual(previous_end.x, poses[0].x, places=12)
                self.assertAlmostEqual(previous_end.y, poses[0].y, places=12)
            if segment.segment_type in {"L", "R"} and segment.direction == "reverse":
                reverse_curve_seen = True
                heading_change = (poses[-1].heading - poses[0].heading + math.pi) % (2 * math.pi) - math.pi
                expected_sign = -1.0 if segment.segment_type == "L" else 1.0
                self.assertGreater(expected_sign * heading_change, 0.0)
        self.assertTrue(reverse_curve_seen)

    def test_segment_lengths_sum_to_total(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 3.0, math.pi), 1.0)
        self.assertAlmostEqual(
            sum(segment.length_m for segment in path.segments),
            path.total_length_m,
            places=12,
        )
        self.assertTrue(all(segment.length_m >= 0.0 for segment in path.segments))

    def test_valid_path_inside_circle(self):
        path = plan_reeds_shepp(Pose2D(-1.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0), 0.5)
        result = validate_path_in_circle(path, 0.0, 0.0, 2.0)
        self.assertTrue(result.boundary_valid)
        self.assertEqual(result.invalid_sample_count, 0)
        self.assertIsNone(result.first_invalid_sample_index)
        self.assertIsNone(result.violation_reason)

    def test_path_leaving_circle_is_marked_invalid_with_first_index_and_count(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 0.0, 0.0), 0.5, 0.1)
        result = validate_path_in_circle(path, 0.0, 0.0, 1.0)
        expected_invalid = [
            index
            for index, sample in enumerate(result.samples)
            if math.hypot(sample.x, sample.y) > 1.0 + 1e-9
        ]
        self.assertFalse(result.boundary_valid)
        self.assertEqual(result.violation_reason, "outside_tank_boundary")
        self.assertEqual(result.first_invalid_sample_index, expected_invalid[0])
        self.assertEqual(result.invalid_sample_count, len(expected_invalid))

    def test_smaller_step_is_denser_and_geometrically_consistent(self):
        start = Pose2D(0.0, 0.0, 0.0)
        goal = Pose2D(3.0, 3.0, math.pi)
        coarse = plan_reeds_shepp(start, goal, 1.0, 0.2)
        fine = plan_reeds_shepp(start, goal, 1.0, 0.04)
        self.assertEqual(coarse.word, fine.word)
        self.assertGreater(len(fine.samples), len(coarse.samples))
        self.assertAlmostEqual(coarse.total_length_m, fine.total_length_m, places=12)
        self.assertAlmostEqual(coarse.samples[-1].x, fine.samples[-1].x, places=10)
        self.assertAlmostEqual(coarse.samples[-1].y, fine.samples[-1].y, places=10)
        self.assertLess(_angle_error(coarse.samples[-1].heading, fine.samples[-1].heading), 1e-10)

    def test_complete_solver_returns_a_four_or_five_segment_solution(self):
        path = plan_reeds_shepp(Pose2D(0.0, 0.0, 0.0), Pose2D(3.0, 3.0, math.pi), 1.0)
        self.assertIn(len(path.segments), {4, 5})
        self.assertTrue(all(segment.length_m > 1e-8 for segment in path.segments))

    def test_invalid_inputs_raise_clear_errors(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "turn_radius_m must be positive"):
            plan_reeds_shepp(pose, pose, 0.0)
        with self.assertRaisesRegex(ValueError, "sample_step_m must be positive"):
            plan_reeds_shepp(pose, pose, 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "start.x must be finite"):
            plan_reeds_shepp(Pose2D(math.nan, 0.0, 0.0), pose, 1.0)
        path = plan_reeds_shepp(pose, pose, 1.0)
        with self.assertRaisesRegex(ValueError, "tank_radius_m must be positive"):
            validate_path_in_circle(path, 0.0, 0.0, -1.0)


if __name__ == "__main__":
    unittest.main()
