import unittest

import numpy as np
from shapely.geometry import box

import path_planner as planner


def _metrics(*, total_area, new_area, inside_area=None, outside_area=0.0):
    inside_area = total_area if inside_area is None else inside_area
    return planner.LawnmowerCandidateMetrics(
        total_area_m2=total_area,
        inside_area_m2=inside_area,
        outside_area_m2=outside_area,
        inside_fraction=inside_area / total_area,
        outside_fraction=outside_area / total_area,
        total_overlap_area_m2=inside_area - new_area,
        expected_neighbor_overlap_area_m2=0.0,
        harmful_overlap_area_m2=0.0,
        harmful_overlap_fraction=0.0,
        new_area_m2=new_area,
        new_fraction=new_area / inside_area,
    )


def _select(full, left=None, right=None, *, full_safety=None, left_safety=None, right_safety=None):
    left = _metrics(total_area=1.0, new_area=0.0) if left is None else left
    right = _metrics(total_area=1.0, new_area=0.0) if right is None else right
    return planner._select_simple_new_coverage_variant(
        full,
        left,
        right,
        full_safety_reason=full_safety,
        left_safety_reason=left_safety,
        right_safety_reason=right_safety,
    )


class SimpleNewCoverageGateTests(unittest.TestCase):
    def test_full_profile_exactly_at_threshold_is_accepted(self):
        self.assertEqual(
            _select(_metrics(total_area=10.0, new_area=4.0)),
            ("full", "accepted_full_new_coverage"),
        )

    def test_full_profile_below_threshold_is_not_accepted_as_full(self):
        self.assertEqual(
            _select(_metrics(total_area=10.0, new_area=3.999999)),
            (None, "rejected_below_new_coverage_threshold"),
        )

    def test_full_profile_above_threshold_is_accepted(self):
        self.assertEqual(
            _select(_metrics(total_area=10.0, new_area=4.1)),
            ("full", "accepted_full_new_coverage"),
        )

    def test_left_half_uses_its_own_area_denominator(self):
        selected = _select(
            _metrics(total_area=10.0, new_area=3.9),
            _metrics(total_area=2.0, new_area=0.8),
            _metrics(total_area=2.0, new_area=0.79),
        )
        self.assertEqual(selected, ("left_half", "accepted_left_half_new_coverage"))

    def test_right_half_can_be_selected(self):
        selected = _select(
            _metrics(total_area=10.0, new_area=3.0),
            _metrics(total_area=2.0, new_area=0.79),
            _metrics(total_area=2.0, new_area=0.8),
        )
        self.assertEqual(selected, ("right_half", "accepted_right_half_new_coverage"))

    def test_both_passing_halves_choose_larger_new_area_then_left_on_tie(self):
        self.assertEqual(
            _select(
                _metrics(total_area=10.0, new_area=3.0),
                _metrics(total_area=2.0, new_area=0.8),
                _metrics(total_area=2.0, new_area=0.9),
            ),
            ("right_half", "accepted_right_half_new_coverage"),
        )
        self.assertEqual(
            _select(
                _metrics(total_area=10.0, new_area=3.0),
                _metrics(total_area=2.0, new_area=0.8),
                _metrics(total_area=2.0, new_area=0.8),
            ),
            ("left_half", "accepted_left_half_new_coverage"),
        )

    def test_neither_half_passing_rejects_candidate(self):
        self.assertEqual(
            _select(
                _metrics(total_area=10.0, new_area=3.0),
                _metrics(total_area=2.0, new_area=0.79),
                _metrics(total_area=2.0, new_area=0.79),
            ),
            (None, "rejected_below_new_coverage_threshold"),
        )

    def test_outside_tank_area_is_not_counted_in_new_coverage(self):
        candidate = np.asarray(box(-1.0, 0.0, 1.0, 1.0).exterior.coords)
        valid_tank = box(0.0, 0.0, 1.0, 1.0)
        metrics = planner._exact_lawnmower_candidate_metrics(
            candidate,
            valid_tank,
            box(3.0, 3.0, 4.0, 4.0),
            None,
        )
        self.assertAlmostEqual(metrics.total_area_m2, 2.0)
        self.assertAlmostEqual(metrics.new_area_m2, 1.0)
        self.assertAlmostEqual(planner._new_coverage_fraction(metrics), 0.5)

    def test_safety_reasons_override_coverage_and_do_not_create_two_halves(self):
        full = _metrics(total_area=10.0, new_area=10.0)
        self.assertEqual(
            _select(full, full_safety="rejected_duplicate"),
            (None, "rejected_duplicate"),
        )
        variant, _reason = _select(
            _metrics(total_area=10.0, new_area=3.0),
            _metrics(total_area=2.0, new_area=0.9),
            _metrics(total_area=2.0, new_area=0.8),
        )
        self.assertIn(variant, {"left_half", "right_half"})


if __name__ == "__main__":
    unittest.main()
