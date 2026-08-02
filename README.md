# Oil Tank Inspection Path Planning

A geometry-based planning and layout-reconstruction system for robotic inspection of circular oil-storage tank floors.

The project generates scan-coverage plans from tank DXF files, simulates partially observed layouts, reconstructs missing structural geometry, and provides curvature-constrained Reeds-Shepp connections between oriented robot poses.

This repository was developed as an internship project and is intended to provide a clear foundation for continued engineering work.

## Quick Visual Overview

For the fastest overview of the project, open:

[`examples/outputs/`](examples/outputs/)

That directory contains representative:

- 24-foot, 65-foot, and 150-foot path-planning previews
- incomplete and completed DXF previews
- generated DXF completion results
- overlap-versus-coverage analysis
- Reeds-Shepp connector visualization
- completion reports

These outputs are intentionally retained so a technical reviewer can understand the project's current capabilities without first running the complete workflow.

---

## What This Repository Does

The repository contains four main systems.

### 1. Tank Geometry Import

The DXF importer:

- reads supported DXF entities
- converts source geometry into normalized internal models
- extracts structural geometry used by downstream planning
- provides the geometry from which the planner estimates the circular tank boundary

### 2. Inspection Path Planning

The path planner:

- generates circular scan rows near the tank wall
- generates vertical and horizontal interior lawnmower scans
- evaluates complete and partial scan footprints
- retains scans that add sufficient new tank coverage
- exports mission data and preview images

### 3. Partial Observation and DXF Completion

The completion workflow:

- simulates what tank geometry would be visible from the generated scans
- writes an incomplete observation DXF
- compares the incomplete layout against known tank families
- reconstructs missing structural geometry
- uses grid prediction as a fallback
- writes completed DXFs, previews, and reports

### 4. Curvature-Constrained Connections

The Reeds-Shepp connector:

- connects oriented robot poses
- supports forward and reverse motion
- respects a minimum turning radius
- samples candidate paths for geometric validation
- validates connector centerlines against the circular tank boundary

The connector currently operates independently from full mission generation. Automatic connection of every ordered scan pose remains future work.

---

## Current Status

| Component | Status |
|---|---|
| DXF importing | Working |
| Circular boundary scans | Working |
| Vertical interior scans | Working |
| Horizontal interior scans | Working |
| Full scan-profile placement | Working |
| Left/right half-profile placement | Working |
| 40% new-coverage acceptance gate | Working |
| Mission preview generation | Working |
| Incomplete-DXF simulation | Working |
| DXF family matching | Working |
| Structural-grid fallback | Working |
| Completed-DXF generation | Working |
| Overlap-versus-coverage analysis | Working |
| Reeds-Shepp connector | Standalone MVP |
| Full-mission Reeds-Shepp integration | Future work |
| Full robot-footprint validation | Future work |
| Internal obstacle handling | Future work |
| Stay-out and exclusion zones | Future work |
| Tether-aware path planning | Future work |
| Tether routing and entanglement prevention | Future work |

---

## System Flow

```text
Source tank DXF
      |
      v
DXF import and normalization
      |
      v
Tank-circle estimation
      |
      v
Circular boundary scans
      |
      v
Vertical interior scans
      |
      v
Horizontal interior scans
      |
      v
Mission validation and preview
```

The DXF-completion workflow continues from the generated scan coverage:

```text
Generated scan coverage
      |
      v
Observed-region construction
      |
      v
Incomplete DXF generation
      |
      v
Reference-family matching
      |
      +-- Successful family completion
      |
      +-- Structural-grid fallback
      |
      v
Completed DXF, preview, and report
```

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── path_planner.py
├── dxf_importer.py
├── scan_profile.py
├── partial_scan_profile.py
├── coverage_geometry.py
├── observation_region.py
│
├── missing_dxf_generator.py
├── dxf_family_matcher.py
├── grid_predictor.py
│
├── reeds_shepp_connector.py
├── demo_reeds_shepp_adjacent_columns.py
├── overlap_coverage_experiment.py
│
├── tests/
│   ├── test_dxf_prediction_workflow.py
│   ├── test_lawnmower_mission_integration.py
│   ├── test_lawnmower_scan_placement.py
│   ├── test_lawnmower_section_lines.py
│   ├── test_partial_scan_profile.py
│   ├── test_reeds_shepp_connector.py
│   ├── test_simple_new_coverage_gate.py
│   └── test_grid_predictor.py
│
└── examples/
    ├── inputs/
    │   ├── 24ft.dxf
    │   ├── 24ft.png
    │   ├── 65ft.dxf
    │   ├── 65ft.png
    │   ├── 150ft.dxf
    │   └── 150ft.png
    │
    └── outputs/
        ├── path_planner_24ft.png
        ├── path_planner_65ft.png
        ├── path_planner_150ft.png
        ├── overlap_vs_coverage.png
        ├── reeds_shepp_output_example.png
        ├── incomplete DXF examples
        ├── completed DXF examples
        ├── before-and-after previews
        └── completion reports
```

The exact contents of `examples/outputs/` may include additional canonical artifacts retained for technical review.

---

## Main Modules

### `path_planner.py`

The primary entry point for inspection-mission generation.

Responsibilities include:

- importing tank geometry
- estimating the usable tank circle
- generating circular scan rows
- extracting structural section lines
- generating interior scan candidates
- applying the current full/half coverage gate
- ordering accepted scan profiles
- exporting mission data
- generating preview images

### `dxf_importer.py`

Loads supported DXF entities and converts them into normalized geometry used by the planner and completion tools.

### `scan_profile.py`

Defines the complete sensor scan footprint and its geometric transformations.

The scan profile is used as the main coverage primitive throughout planning and observation.

### `partial_scan_profile.py`

Splits a transformed full scan footprint into exact local left and right halves.

The halves preserve the parent profile's transform and are not independently recentered.

### `coverage_geometry.py`

Contains exact polygon construction, polygon-union, and candidate-valid-region helpers used by scan placement.

### `observation_region.py`

Builds unions of observed scan footprints and clips tank geometry to the region that the simulated robot has scanned.

### `missing_dxf_generator.py`

Creates incomplete observation DXFs by clipping source geometry to the scan coverage generated by the planner.

### `dxf_family_matcher.py`

Compares an incomplete tank layout against known reference families, estimates the required transformation, and reconstructs missing geometry.

It can write:

- a completed DXF
- a preview image
- a machine-readable completion report

### `grid_predictor.py`

Provides structural-grid inference when direct family matching is incomplete or ambiguous.

### `reeds_shepp_connector.py`

Computes curvature-constrained paths between oriented robot poses.

The implementation supports forward and reverse motion and includes multiple Reeds-Shepp path families.

### `demo_reeds_shepp_adjacent_columns.py`

Builds a representative mission and visualizes a Reeds-Shepp connection between poses from adjacent interior scan columns.

### `overlap_coverage_experiment.py`

Runs the current planner across overlap settings and produces the raw overlap-versus-coverage graph.

The canonical experiment output is a single PNG containing raw points. It does not retain fitted curves or JSON/CSV sidecar data.

---

## Installation

### Requirements

- Python 3.12 or newer
- NumPy
- Shapely
- Matplotlib
- ezdxf
- any additional packages listed in `requirements.txt`

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Quick Start

From the repository root, generate the 24-foot path-planning preview:

```bash
python path_planner.py "examples/inputs/24ft.dxf" --no-plot --save-json= --save-plot "examples/outputs/path_planner_24ft.png"
```

The resulting preview is written to:

```text
examples/outputs/path_planner_24ft.png
```

Equivalent input files are included for the 65-foot and 150-foot tanks.

For a quick review without running anything, browse:

[`examples/outputs/`](examples/outputs/)

---

## Path-Planning Logic

The planner executes three coverage stages in order:

1. circular boundary scans
2. vertical interior scans
3. horizontal interior scans

Previously accepted coverage contributes to the evaluation of later candidates.

### Authoritative Interior Coverage Gate

The current interior acceptance threshold is:

```python
MIN_NEW_COVERAGE_FRACTION = 0.40
```

For each interior candidate:

1. Construct the exact transformed full scan-profile polygon.
2. Measure the candidate's previously uncovered in-tank area.
3. Divide the new area by the full profile's area.
4. Accept the full profile when the fraction is at least `0.40`.
5. If the full profile fails, evaluate the exact local left and right halves.
6. Divide each half's new area by that half's own area.
7. Accept at most one passing half.
8. Reject the candidate when neither the full profile nor either half passes.

A passing full profile is preferred over either half.

Coverage outside the tank does not count toward the new-coverage fraction.

The accepted coverage union can include:

- circular scans
- previously accepted vertical scans
- previously accepted horizontal scans
- full profiles
- half profiles

---

## Generate Path-Planning Examples

### 24-foot tank

```bash
python path_planner.py "examples/inputs/24ft.dxf" --no-plot --save-json= --save-plot "examples/outputs/path_planner_24ft.png"
```

### 65-foot tank

```bash
python path_planner.py "examples/inputs/65ft.dxf" --no-plot --save-json= --save-plot "examples/outputs/path_planner_65ft.png"
```

### 150-foot tank

```bash
python path_planner.py "examples/inputs/150ft.dxf" --no-plot --save-json= --save-plot "examples/outputs/path_planner_150ft.png"
```

---

## Generate an Incomplete DXF

The incomplete-DXF generator simulates the structural geometry that would be visible from the robot's scan coverage.

Example for the 24-foot tank:

```bash
python missing_dxf_generator.py "examples/inputs/24ft.dxf" --circular-rows 1 --output "examples/outputs/uncompleted_dxf_24.dxf" --save-preview "examples/outputs/uncompleted_dxf_24.png"
```

This produces:

```text
examples/outputs/uncompleted_dxf_24.dxf
examples/outputs/uncompleted_dxf_24.png
```

The incomplete DXF can then be passed into the completion workflow.

---

## Complete a Partial DXF

Run the family matcher against an incomplete observation:

```bash
python dxf_family_matcher.py "examples/outputs/uncompleted_dxf_24.dxf" --reference-dir "examples/inputs" --save-dxf "examples/outputs/dxf_completed_24ft.dxf" --save-plot "examples/outputs/dxf_completed_24ft.png" --save-report "examples/outputs/dxf_completed_24ft_report.json"
```

The completion workflow:

1. analyzes observed structural geometry
2. compares it against known tank-layout families
3. estimates alignment and transformation
4. reconstructs missing structural members
5. uses structural-grid inference when needed
6. writes the completed layout and validation artifacts

Canonical examples for the 24-foot, 65-foot, and 150-foot tanks are retained in `examples/outputs/`.

---

## Overlap Versus Coverage

Generate the current overlap-versus-coverage graph:

```bash
python overlap_coverage_experiment.py
```

Canonical output:

```text
examples/outputs/overlap_vs_coverage.png
```

The graph contains raw points produced using the current path-planner implementation.

Only one canonical graph is retained. The experiment does not keep:

- fitted-curve variants
- alternate graph versions
- CSV sidecar files
- JSON sidecar files

---

## Reeds-Shepp Connector Demo

Run the adjacent-column connector demonstration:

```bash
python demo_reeds_shepp_adjacent_columns.py
```

Canonical output:

```text
examples/outputs/reeds_shepp_output_example.png
```

The connector supports curvature-constrained forward and reverse motion between oriented robot poses.

Current validation focuses on the path centerline and tank boundary. It does not yet represent the complete physical footprint of the robot, scanner, or tether.

---

## Running Tests

Run the complete test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Compile-check the repository:

```bash
python -m compileall .
```

The tests cover:

- DXF import and normalization
- structural section-line extraction
- circular scan-row selection
- interior lawnmower placement
- scan-profile transformations
- full and half scan partitioning
- the 40% new-coverage gate
- mission ordering
- incomplete-DXF generation
- family-based layout completion
- grid-prediction fallback
- Reeds-Shepp path families
- path sampling and cusp behavior
- tank-boundary validation
- end-to-end workflow behavior

---

## Example Inputs

The `examples/inputs/` directory contains complete reference tanks in three sizes:

```text
examples/inputs/
├── 24ft.dxf
├── 24ft.png
├── 65ft.dxf
├── 65ft.png
├── 150ft.dxf
└── 150ft.png
```

The DXF files are the machine-readable source inputs.

The PNG files provide quick visual references for reviewers who do not have a DXF viewer available.

---

## Example Outputs

Visit [`examples/outputs/`](examples/outputs/) for the quickest look at the project.

The directory contains representative:

- planner previews
- incomplete tank observations
- completed tank layouts
- before-and-after DXF previews
- generated DXF files
- completion reports
- raw overlap-versus-coverage results
- Reeds-Shepp connector output

These files are intended as canonical examples and should make it possible to understand the project's major workflows before reading the implementation.

---

## Geometry Conventions

The current project uses the following conventions:

- distances are represented in meters
- headings are represented in radians
- positive rotation is counterclockwise
- the scan-profile anchor is the midpoint of the profile's long side
- the robot's scan heading is parallel to the profile's short side
- full and half profiles preserve the same parent transform
- lawnmower scan direction alternates between neighboring placements
- the current Reeds-Shepp demo uses a minimum turning radius of `1.8288 m`

---

## Known Limitations

- Reeds-Shepp connectors are not yet integrated between every ordered mission pose.
- Reeds-Shepp validation currently focuses on centerline containment rather than the complete robot footprint.
- Internal obstacles are not fully modeled.
- Configurable stay-out zones and exclusion regions are not yet supported.
- The planner does not currently model a physical tether.
- Tether length, drag, routing, winding, snagging, and entanglement risks are not considered during path generation.
- The planner does not prevent the tether from wrapping around tank structures or crossing previous tether routes.
- Safe tether unwinding and return-path behavior are not yet planned.
- DXF completion has been validated using a limited number of reference tank families.
- The planner assumes supported DXF geometry and a primarily circular tank boundary.
- Very dense parameter sweeps may require additional performance optimization.
- Real-world robot execution and hardware-level validation remain outside the current repository scope.

---

## Future Mission-Planning Work

A production deployment would need to extend the current coverage planner with additional operational constraints.

### Reeds-Shepp Integration

- connect all ordered scan poses
- choose connector directions consistently
- account for travel distance and reversing cost
- validate connectors against all relevant geometry

### Robot-Footprint Validation

- validate the complete robot body
- validate the complete scanner footprint
- include turn clearance
- include safety margins around structural members

### Stay-Out Zones

- support configurable exclusion polygons
- represent unsafe floor areas
- exclude internal equipment and structural hazards
- prevent scan profiles and connectors from entering restricted regions

### Tether Management

- model available tether length
- track the tether's route through the mission
- model winding and unwinding
- avoid wrapping around structural members
- avoid unsafe tether crossings
- detect possible snag and entanglement conditions
- preserve a safe route back to the mission start
- optimize the route while respecting tether constraints

---

## Recommended Next Steps

1. Integrate Reeds-Shepp connectors between ordered mission poses.
2. Validate the complete robot and scanner footprint during connectors.
3. Add configurable stay-out and exclusion zones.
4. Add internal-obstacle handling.
5. Add tether-aware path planning.
6. Track tether length, route, winding direction, and possible entanglement.
7. Prevent unsafe tether crossings and structural wrapping.
8. Add safe return-path and tether-unwinding behavior.
9. Optimize mission ordering and total travel distance.
10. Add more independent tank families for DXF-completion validation.
11. Expand integration and regression testing.
12. Validate the planner using real robot and tank data.

---

## License

This project is licensed under the MIT License.
