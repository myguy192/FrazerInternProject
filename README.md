# Tank Inspection Coverage Planner

This repository turns circular-tank DXF geometry into inspection scan plans,
simulates incomplete circular-scan observations, and completes the missing
structural grid from known tank families or repeated-grid evidence.

The current planner uses exact geometry and one authoritative interior gate:
a full or half scan profile must contribute at least 40% new coverage. The
Reeds–Shepp connector is an isolated MVP and is not yet integrated into the
mission path.

## Setup

Python 3.12 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -v
```

## Main files

- `path_planner.py` — production circular-edge and interior lawnmower planner.
- `dxf_importer.py` — supported DXF parsing and geometry normalization.
- `missing_dxf_generator.py` — creates partial observations from circular scans.
- `dxf_family_matcher.py` — classifies and completes partial tank layouts.
- `grid_predictor.py` — structural-grid fallback used when family matching is ambiguous.
- `reeds_shepp_connector.py` — standalone curvature-constrained connector MVP.
- `overlap_coverage_experiment.py` — canonical raw-points coverage experiment.
- `demo_reeds_shepp_adjacent_columns.py` — reproducible 24ft connector demo.
- `HANDOFF.md` — architecture, workflows, limitations, and every retained file.

## Run the planner

```powershell
python path_planner.py "3 Tank examples/24ft.dxf" `
  --no-plot `
  --save-json= `
  --save-plot "output examples/path_planner_24ft.png"
```

Replace `24ft` with `65ft` or `150ft` for the other reference tanks.

## Run DXF completion

Generate a partial observation:

```powershell
python missing_dxf_generator.py "3 Tank examples/24ft.dxf" `
  --circular-rows 1 `
  --output "output examples/uncompleted_dxf_24.dxf" `
  --save-preview "output examples/uncompleted_dxf_24.png"
```

Complete it:

```powershell
python dxf_family_matcher.py "output examples/uncompleted_dxf_24.dxf" `
  --reference-dir "3 Tank examples" `
  --save-dxf "output examples/dxf_completed_24ft.dxf" `
  --save-plot "output examples/dxf_completed_24ft.png" `
  --save-report "output examples/dxf_completed_24ft_report.json"
```

The 24ft example uses one circular row; the 65ft and 150ft examples use two.

## Run maintained analysis tools

```powershell
python overlap_coverage_experiment.py
python demo_reeds_shepp_adjacent_columns.py
```

Both commands update their single canonical PNG in `output examples`.

## Current limitations

- Reeds–Shepp paths validate the robot centerline against the tank boundary;
  robot footprint and internal-obstacle collision checks are future work.
- The connector is not yet inserted between mission scan poses.
- DXF family completion is trained only on the three supplied reference tanks.
- The structural-grid fallback requires enough repeated line evidence.
- The overlap graph runs exact planning repeatedly and is intentionally slow.
- This is an offline Python planning/validation base, not a ROS2 runtime node.
