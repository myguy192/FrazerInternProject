"""Isolated full/half harmful-overlap coverage experiment for three tank sizes.

The production planner is imported unchanged.  For each run this script
temporarily sets the active full/half harmful-overlap limits in memory, then
restores both values before continuing or exiting.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import path_planner as planner
from scan_profile import RainbowProfileConfig


ROOT = Path(__file__).parent
TANKS = (
    ("24ft", ROOT / "3 Tank examples" / "24ft.dxf"),
    ("65ft", ROOT / "3 Tank examples" / "65ft.dxf"),
    ("150ft", ROOT / "3 Tank examples" / "150ft.dxf"),
)
CSV_PATH = ROOT / "overlap_coverage_results.csv"
JSON_PATH = ROOT / "overlap_coverage_results.json"
POINTS_PNG_PATH = ROOT / "overlap_vs_coverage_points.png"
FIT_PNG_PATH = ROOT / "overlap_vs_coverage_best_fit.png"
GRID_RESOLUTION = 220
BASELINE_FULL_PERCENT = 55.0
HALF_THRESHOLD_OFFSET_PERCENT = 5.0
SWEEP_PERCENTAGES = tuple(float(value) for value in range(0, 101, 5))
DEGREE_THREE_MINIMUM_CV_IMPROVEMENT = 0.05
PLOT_STYLES = {
    "24ft": {"coverage": "#2563eb", "double": "#dc2626", "marker": "o"},
    "65ft": {"coverage": "#0f766e", "double": "#d97706", "marker": "s"},
    "150ft": {"coverage": "#7c3aed", "double": "#be185d", "marker": "D"},
}


def _reset_planner_state() -> None:
    """Clear the only mutable planner cache used by the normal mission path."""
    planner._TOUCHING_ANGULAR_STEP_CACHE.clear()
    gc.collect()


@contextmanager
def _temporary_overlap_thresholds(full_percent: float):
    original_full = planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION
    original_half = planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION
    half_percent = max(0.0, full_percent - HALF_THRESHOLD_OFFSET_PERCENT)
    try:
        planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION = full_percent / 100.0
        planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION = half_percent / 100.0
        yield half_percent
    finally:
        planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION = original_full
        planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION = original_half


def _profile_config(mission: planner.OuterEdgeSweepMission) -> RainbowProfileConfig:
    return RainbowProfileConfig(
        width=float(mission.scan_profile["width_m"]),
        arc_radius=float(mission.scan_profile["arc_radius_m"]),
        side_height=float(mission.scan_profile["side_height_m"]),
        arc_samples=int(mission.scan_profile["arc_samples"]),
    )


def _pose_counts(mission: planner.OuterEdgeSweepMission) -> dict[str, int]:
    poses = mission.poses
    return {
        "circular_profile_count": sum(pose.stage == "circular_edge" for pose in poses),
        "vertical_full_profile_count": sum(
            pose.stage == "interior_vertical" and pose.profile_variant == "full"
            for pose in poses
        ),
        "vertical_half_profile_count": sum(
            pose.stage == "interior_vertical" and pose.profile_variant != "full"
            for pose in poses
        ),
        "horizontal_full_profile_count": sum(
            pose.stage == "interior_horizontal" and pose.profile_variant == "full"
            for pose in poses
        ),
        "horizontal_half_profile_count": sum(
            pose.stage == "interior_horizontal" and pose.profile_variant != "full"
            for pose in poses
        ),
        "total_scan_profile_count": len(poses),
    }


def _variant_stage_counts(mission: planner.OuterEdgeSweepMission) -> Counter[tuple[str, str]]:
    return Counter((pose.stage, pose.profile_variant) for pose in mission.poses)


def _run_experiment_setting(
    model,
    tank_name: str,
    dxf_path: Path,
    full_percent: float,
) -> tuple[planner.OuterEdgeSweepMission, dict[str, Any]]:
    _reset_planner_state()
    started = time.perf_counter()
    with _temporary_overlap_thresholds(full_percent) as half_percent:
        mission = planner.build_mission_plan(model)
        coverage = planner._estimate_tank_coverage(
            mission,
            _profile_config(mission),
            grid_resolution=GRID_RESOLUTION,
        )
    runtime_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "tank": tank_name,
        "dxf": str(dxf_path.relative_to(ROOT)),
        "full_allowed_harmful_overlap_percent": full_percent,
        "half_allowed_harmful_overlap_percent": half_percent,
        "coverage_percent": coverage.covered_percent,
        "double_coverage_percent": coverage.multi_covered_percent,
        **_pose_counts(mission),
        "rejected_interior_candidate_count": mission.interior_discarded_count,
        "runtime_seconds": runtime_seconds,
    }
    return mission, result


def _validate_baseline(
    model,
    tank_name: str,
    dxf_path: Path,
) -> tuple[planner.OuterEdgeSweepMission, dict[str, Any]]:
    """Require x=55% / 50% to reproduce the current, untouched planner."""
    original_full = planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION
    original_half = planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION
    if not (math.isclose(original_full, 0.55) and math.isclose(original_half, 0.50)):
        raise RuntimeError(
            "The production defaults are not the expected 55% full / 50% half baseline."
        )
    _reset_planner_state()
    normal_mission = planner.build_mission_plan(model)
    normal_coverage = planner._estimate_tank_coverage(
        normal_mission,
        _profile_config(normal_mission),
        grid_resolution=GRID_RESOLUTION,
    )
    experiment_mission, experiment_result = _run_experiment_setting(
        model,
        tank_name,
        dxf_path,
        BASELINE_FULL_PERCENT,
    )
    differences: list[str] = []
    if normal_mission.poses != experiment_mission.poses:
        differences.append("ordered poses")
    if _pose_counts(normal_mission) != _pose_counts(experiment_mission):
        differences.append("stage/variant counts")
    if _variant_stage_counts(normal_mission) != _variant_stage_counts(experiment_mission):
        differences.append("stage/variant breakdown")
    if not math.isclose(normal_coverage.covered_percent, experiment_result["coverage_percent"], abs_tol=0.0):
        differences.append("coverage percentage")
    if not math.isclose(normal_coverage.multi_covered_percent, experiment_result["double_coverage_percent"], abs_tol=0.0):
        differences.append("two-or-more coverage percentage")
    if differences:
        raise RuntimeError("Baseline experiment differs from normal planner: " + ", ".join(differences))
    return experiment_mission, experiment_result


def _cross_validated_rmse(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    errors: list[float] = []
    for held_out in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[held_out] = False
        coefficients = np.polyfit(x[mask], y[mask], degree)
        errors.append(float(np.polyval(coefficients, x[held_out]) - y[held_out]))
    return float(math.sqrt(float(np.mean(np.square(errors)))))


def _choose_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    candidates = {
        degree: _cross_validated_rmse(x, y, degree)
        for degree in (2, 3)
    }
    selected_degree = 2
    if candidates[3] <= candidates[2] * (1.0 - DEGREE_THREE_MINIMUM_CV_IMPROVEMENT):
        selected_degree = 3
    coefficients = np.polyfit(x, y, selected_degree)
    predicted = np.polyval(coefficients, x)
    residuals = y - predicted
    rmse = float(math.sqrt(float(np.mean(np.square(residuals)))))
    total_sum_squares = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 if total_sum_squares == 0.0 else float(1.0 - np.sum(np.square(residuals)) / total_sum_squares)
    return {
        "degree": selected_degree,
        "coefficients": [float(value) for value in coefficients],
        "rmse": rmse,
        "r_squared": r_squared,
        "cross_validated_rmse_by_degree": {str(degree): value for degree, value in candidates.items()},
    }


def _pareto_efficient(results: list[dict[str, Any]]) -> list[float]:
    efficient: list[float] = []
    for candidate in results:
        dominated = any(
            other is not candidate
            and other["coverage_percent"] >= candidate["coverage_percent"]
            and other["double_coverage_percent"] <= candidate["double_coverage_percent"]
            and (
                other["coverage_percent"] > candidate["coverage_percent"]
                or other["double_coverage_percent"] < candidate["double_coverage_percent"]
            )
            for other in results
        )
        if not dominated:
            efficient.append(candidate["full_allowed_harmful_overlap_percent"])
    return efficient


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Full-profile allowed harmful overlap (%)")
    ax.set_ylabel("Tank area (%)")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.30)
    ax.axvline(BASELINE_FULL_PERCENT, color="#475569", linestyle="--", linewidth=1.4, label="Current full threshold (55%)")


def _tank_rows(results: list[dict[str, Any]], tank_name: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in results if row["tank"] == tank_name),
        key=lambda row: row["full_allowed_harmful_overlap_percent"],
    )


def _plot_points(results: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for tank_name, _dxf_path in TANKS:
        rows = _tank_rows(results, tank_name)
        x = np.array([row["full_allowed_harmful_overlap_percent"] for row in rows], dtype=float)
        coverage = np.array([row["coverage_percent"] for row in rows], dtype=float)
        double_coverage = np.array([row["double_coverage_percent"] for row in rows], dtype=float)
        style = PLOT_STYLES[tank_name]
        baseline_index = int(np.where(x == BASELINE_FULL_PERCENT)[0][0])
        ax.scatter(x, coverage, color=style["coverage"], marker=style["marker"], s=38, label=f"{tank_name}: covered at least once", zorder=3)
        ax.scatter(x, double_coverage, color=style["double"], marker=style["marker"], s=38, label=f"{tank_name}: covered two or more times", zorder=3)
        ax.scatter([x[baseline_index]], [coverage[baseline_index]], s=110, facecolors="none", edgecolors="#111827", linewidths=1.5, zorder=4)
        ax.scatter([x[baseline_index]], [double_coverage[baseline_index]], s=110, facecolors="none", edgecolors="#111827", linewidths=1.5, zorder=4, label="Baseline data points" if tank_name == "24ft" else "_nolegend_")
    _style_axes(ax, "Raw experiment points\nHalf-profile allowance is max(0, x - 5 percentage points)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(POINTS_PNG_PATH, dpi=200)
    plt.close(fig)


def _plot_fits(results: list[dict[str, Any]], fits: dict[str, dict[str, dict[str, Any]]]) -> None:
    trend_x = np.linspace(0.0, 100.0, 500)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for tank_name, _dxf_path in TANKS:
        rows = _tank_rows(results, tank_name)
        x = np.array([row["full_allowed_harmful_overlap_percent"] for row in rows], dtype=float)
        coverage = np.array([row["coverage_percent"] for row in rows], dtype=float)
        double_coverage = np.array([row["double_coverage_percent"] for row in rows], dtype=float)
        style = PLOT_STYLES[tank_name]
        coverage_fit = fits[tank_name]["coverage"]
        double_fit = fits[tank_name]["double_coverage"]
        coverage_trend = np.clip(np.polyval(coverage_fit["coefficients"], trend_x), 0.0, 100.0)
        double_trend = np.clip(np.polyval(double_fit["coefficients"], trend_x), 0.0, 100.0)
        ax.scatter(x, coverage, color=style["coverage"], marker=style["marker"], s=24, alpha=0.28, label=f"{tank_name}: coverage raw")
        ax.scatter(x, double_coverage, color=style["double"], marker=style["marker"], s=24, alpha=0.28, label=f"{tank_name}: two-or-more raw")
        ax.plot(trend_x, coverage_trend, color=style["coverage"], linewidth=2.4, label=f"{tank_name}: coverage trend (degree {coverage_fit['degree']})")
        ax.plot(trend_x, double_trend, color=style["double"], linewidth=2.4, label=f"{tank_name}: two-or-more trend (degree {double_fit['degree']})")
    _style_axes(ax, "Trend fits, not exact planner outputs\nHalf-profile allowance is max(0, x - 5 percentage points)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIT_PNG_PATH, dpi=200)
    plt.close(fig)


def _write_results(results: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    fieldnames = list(results[0])
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    JSON_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    original_full = planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION
    original_half = planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION
    try:
        results: list[dict[str, Any]] = []
        fits: dict[str, dict[str, dict[str, Any]]] = {}
        pareto: dict[str, list[float]] = {}
        baseline_validation: dict[str, str] = {}
        for tank_name, dxf_path in TANKS:
            model = planner.import_dxf(dxf_path)
            baseline_mission, baseline_result = _validate_baseline(model, tank_name, dxf_path)
            if tank_name == "65ft" and not (baseline_mission.vertical_scan_count and baseline_mission.horizontal_scan_count):
                raise RuntimeError("65ft mission does not contain both vertical and horizontal interior stages.")
            if tank_name == "24ft" and baseline_mission.horizontal_scan_count:
                raise RuntimeError("24ft mission unexpectedly contains horizontal interior scans.")
            if tank_name == "150ft" and not (baseline_mission.vertical_scan_count and baseline_mission.horizontal_scan_count):
                raise RuntimeError("150ft mission does not contain both vertical and horizontal interior stages.")
            baseline_validation[tank_name] = "passed: poses, stage/variant counts, and coverage metrics exactly match normal planner"
            tank_results: list[dict[str, Any]] = []
            for index, full_percent in enumerate(SWEEP_PERCENTAGES, start=1):
                print(f"{tank_name} [{index}/{len(SWEEP_PERCENTAGES)}] allowed overlap = {full_percent:.0f}%")
                if math.isclose(full_percent, BASELINE_FULL_PERCENT):
                    result = baseline_result
                else:
                    _mission, result = _run_experiment_setting(model, tank_name, dxf_path, full_percent)
                tank_results.append(result)
            tank_results.sort(key=lambda row: row["full_allowed_harmful_overlap_percent"])
            results.extend(tank_results)
            x = np.array([row["full_allowed_harmful_overlap_percent"] for row in tank_results], dtype=float)
            coverage = np.array([row["coverage_percent"] for row in tank_results], dtype=float)
            double_coverage = np.array([row["double_coverage_percent"] for row in tank_results], dtype=float)
            fits[tank_name] = {
                "coverage": _choose_fit(x, coverage),
                "double_coverage": _choose_fit(x, double_coverage),
            }
            pareto[tank_name] = _pareto_efficient(tank_results)
        results.sort(key=lambda row: (row["tank"], row["full_allowed_harmful_overlap_percent"]))
        metadata = {
            "tank_dxfs": {tank_name: str(dxf_path.relative_to(ROOT)) for tank_name, dxf_path in TANKS},
            "grid_resolution": GRID_RESOLUTION,
            "threshold_relationship": "half = max(0, full - 5 percentage points)",
            "baseline_full_allowed_harmful_overlap_percent": BASELINE_FULL_PERCENT,
            "baseline_half_allowed_harmful_overlap_percent": BASELINE_FULL_PERCENT - HALF_THRESHOLD_OFFSET_PERCENT,
            "baseline_validation": baseline_validation,
            "fits": fits,
            "pareto_efficient_full_overlap_percentages": pareto,
            "results": results,
        }
        _write_results(results, metadata)
        _plot_points(results)
        _plot_fits(results, fits)
        print(f"Saved: {CSV_PATH.name}, {JSON_PATH.name}, {POINTS_PNG_PATH.name}, {FIT_PNG_PATH.name}")
        return 0
    finally:
        planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION = original_full
        planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION = original_half
        assert planner.MAX_FULL_HARMFUL_OVERLAP_FRACTION == original_full
        assert planner.MAX_HALF_HARMFUL_OVERLAP_FRACTION == original_half


if __name__ == "__main__":
    raise SystemExit(main())
