"""Generate the canonical raw-points overlap-versus-coverage graph.

The current planner accepts an interior profile when enough of that profile
covers new area.  The displayed overlap allowance is therefore
``100 * (1 - MIN_NEW_COVERAGE_FRACTION)``.  Each experiment changes that
threshold only in memory and restores the production value before exit.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import path_planner as planner
from scan_profile import RainbowProfileConfig


ROOT = Path(__file__).resolve().parent
TANKS = (
    ("24ft", ROOT / "examples" / "inputs" / "24ft.dxf"),
    ("65ft", ROOT / "examples" / "inputs" / "65ft.dxf"),
    ("150ft", ROOT / "examples" / "inputs" / "150ft.dxf"),
)
OUTPUT_PATH = ROOT / "examples" / "outputs" / "overlap_vs_coverage.png"
GRID_RESOLUTION = 220
SWEEP_PERCENTAGES = tuple(float(value) for value in range(0, 101, 20))
PLOT_STYLES = {
    "24ft": {"coverage": "#2563eb", "double": "#dc2626", "marker": "o"},
    "65ft": {"coverage": "#0f766e", "double": "#d97706", "marker": "s"},
    "150ft": {"coverage": "#7c3aed", "double": "#be185d", "marker": "D"},
}


def _reset_planner_state() -> None:
    """Clear the mutable cache used by the normal mission-planning path."""
    planner._TOUCHING_ANGULAR_STEP_CACHE.clear()
    gc.collect()


@contextmanager
def _temporary_overlap_allowance(overlap_percent: float):
    original = planner.MIN_NEW_COVERAGE_FRACTION
    try:
        planner.MIN_NEW_COVERAGE_FRACTION = 1.0 - overlap_percent / 100.0
        yield
    finally:
        planner.MIN_NEW_COVERAGE_FRACTION = original


def _profile_config(mission: planner.OuterEdgeSweepMission) -> RainbowProfileConfig:
    return RainbowProfileConfig(
        width=float(mission.scan_profile["width_m"]),
        arc_radius=float(mission.scan_profile["arc_radius_m"]),
        side_height=float(mission.scan_profile["side_height_m"]),
        arc_samples=int(mission.scan_profile["arc_samples"]),
    )


def _run_setting(
    model,
    tank_name: str,
    overlap_percent: float,
) -> dict[str, Any]:
    _reset_planner_state()
    started = time.perf_counter()
    with _temporary_overlap_allowance(overlap_percent):
        mission = planner.build_mission_plan(model)
        coverage = planner._estimate_tank_coverage(
            mission,
            _profile_config(mission),
            grid_resolution=GRID_RESOLUTION,
        )
    return {
        "tank": tank_name,
        "overlap_percent": overlap_percent,
        "coverage_percent": coverage.covered_percent,
        "double_coverage_percent": coverage.multi_covered_percent,
        "runtime_seconds": time.perf_counter() - started,
    }


def _plot_raw_points(results: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for tank_name, _dxf_path in TANKS:
        rows = sorted(
            (row for row in results if row["tank"] == tank_name),
            key=lambda row: row["overlap_percent"],
        )
        style = PLOT_STYLES[tank_name]
        x = [row["overlap_percent"] for row in rows]
        ax.scatter(
            x,
            [row["coverage_percent"] for row in rows],
            color=style["coverage"],
            marker=style["marker"],
            s=38,
            label=f"{tank_name}: covered at least once",
        )
        ax.scatter(
            x,
            [row["double_coverage_percent"] for row in rows],
            color=style["double"],
            marker=style["marker"],
            s=38,
            label=f"{tank_name}: covered two or more times",
        )

    ax.set_title("Overlap vs coverage — current planner raw points")
    ax.set_xlabel("Allowed existing coverage per candidate (%)")
    ax.set_ylabel("Tank area (%)")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    return OUTPUT_PATH


def main() -> int:
    original_minimum_new_coverage = planner.MIN_NEW_COVERAGE_FRACTION
    results: list[dict[str, Any]] = []
    try:
        for tank_name, dxf_path in TANKS:
            model = planner.import_dxf(dxf_path)
            for completed_count, overlap_percent in enumerate(
                SWEEP_PERCENTAGES,
                start=1,
            ):
                print(
                    f"{tank_name} [{completed_count}/{len(SWEEP_PERCENTAGES)}]: "
                    f"allowed existing coverage = {overlap_percent:.0f}%"
                )
                results.append(_run_setting(model, tank_name, overlap_percent))
        output_path = _plot_raw_points(results)
    finally:
        planner.MIN_NEW_COVERAGE_FRACTION = original_minimum_new_coverage
        assert planner.MIN_NEW_COVERAGE_FRACTION == original_minimum_new_coverage

    print(f"Saved: {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
