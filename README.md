# ROS2 Coverage Path Planner for Circular Tank Inspection

A ROS2 Jazzy node that generates and publishes an Archimedean spiral coverage path for robotic inspection of circular above-ground crude oil storage tanks. Built for Applied Impact Robotics.

## What It Does

The planner generates a continuous spiral path starting from the tank manway, covering the entire tank floor while navigating around internal structural columns. The path is published as a `nav_msgs/Path` message over ROS2, making it directly compatible with downstream navigation and control nodes.

The path is computed once on startup and republished every second, ensuring any node that comes online late still receives the full coverage path.

## Algorithm

- Builds a base Archimedean spiral parameterized by tank radius, pass spacing, and sample arc length
- Projects each waypoint into free space using an iterative obstacle avoidance pass that pushes points away from column exclusion zones and the tank wall
- Smooths the projected path using a moving average filter to reduce sharp direction changes
- Computes tangent and inward normal vectors at each waypoint for downstream sensor orientation control

For a 15m radius tank with 0.31m pass spacing, the planner generates approximately 35,000 waypoints at 0.06m arc length resolution.

## Stack

- ROS2 Jazzy on Ubuntu 24.04
- Python 3.12
- numpy

## Usage
```bash
cd ~/ros2_ws
colcon build --packages-select my_first_pkg
source install/setup.bash
ros2 launch my_first_pkg my_launch.py
```

In a second terminal, visualize the path in RViz2:
```bash
rviz2
```

Add a Path display, set the topic to `/coverage_path`, and set the fixed frame to `map`.

## Published Topics

| Topic | Type | Description |
| `/coverage_path` | `nav_msgs/Path` | Spiral coverage waypoints in the map frame |

## Parameters (coverage_planner.py)

| Parameter | Value | Description |

| `TANK_RADIUS` | 15.0 m | Tank inner radius |
| `PASS_SPACING` | 0.31 m | Distance between spiral passes |
| `SCAN_WIDTH` | 0.364 m | Sensor scan width |
| `COLUMN_RADIUS` | 0.3 m | Structural column radius |
| `COLUMN_CLEARANCE` | 0.3 m | Clearance margin around columns |
| `SAMPLE_ARC_LENGTH` | 0.06 m | Distance between waypoints |

## DXF Importer v1

`dxf_importer.py` imports and visualizes tank floor DXF geometry before any planning step.

Install the DXF parser dependency:
```bash
pip install ezdxf
```

Run the importer with plotting:
```bash
python dxf_importer.py path/to/tank_floor.dxf
```

Run only the import summary:
```bash
python dxf_importer.py path/to/tank_floor.dxf --no-plot
```

Save a plot image without opening a window:
```bash
python dxf_importer.py path/to/tank_floor.dxf --no-plot --save-plot imported_geometry.png
```

Supported v1 modelspace entities:

- `LINE`
- `LWPOLYLINE`
- `POLYLINE`
- `ARC`
- `CIRCLE`

Unsupported entities are reported in the summary instead of stopping the import. The importer preserves raw entity metadata, layer names, supported geometry objects, and simple drawing bounds so later tools can add tank boundary detection, plate detection, weld extraction, node generation, and scan profile placement.
