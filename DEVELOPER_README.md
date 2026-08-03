# Developer Guide

**Project:** Autonomous Indoor Navigation using ROS2 on Quanser QCar

---

# Table of Contents

1. Introduction
2. System Architecture
3. Repository Structure
4. Software Stack
5. ROS2 Node Architecture
6. Launch Architecture
7. TF Tree
8. Topics and Services
9. Mapping Pipeline
10. Localization Pipeline
11. Trajectory Recording
12. Path Processing
13. Model Predictive Controller
14. Obstacle Avoidance
15. Emergency Stop
16. Configuration Files
17. Debugging
18. Extending the Project
19. Future Improvements

---

# 1. Introduction

This document describes the internal software architecture of the autonomous navigation stack developed for the Quanser QCar.

Unlike the user guide, which explains how to operate the system, this document is intended for developers who wish to understand, modify or extend the software.

The project follows a modular ROS2 architecture in which each subsystem performs one specific task while communicating through ROS2 topics, services and the TF tree.

The primary objectives of the software are:

- Autonomous indoor navigation
- Accurate localization
- Smooth path following
- Robust obstacle avoidance
- Reliable emergency stopping
- Easy extensibility

---

# 2. System Architecture

The complete navigation pipeline is shown below.

```
                    +----------------+
                    |   RPLidar      |
                    +--------+-------+
                             |
                             v
                    Obstacle Detection
                             |
                             v
                      Lane Detection
                             |
                             |
                             v
      +---------------- Model Predictive Controller ----------------+
      |                                                             |
      |                                                             |
      +-------------------------------------------------------------+
                             ^
                             |
                       Vehicle Pose
                             ^
                             |
                  +----------+----------+
                  |                     |
                  |                     |
               EKF Fusion            AMCL
                  ^                     ^
                  |                     |
             Wheel Encoder          Occupancy Map
                  |                     ^
                  |                     |
                 IMU              Cartographer
```

Every major subsystem has a clearly defined responsibility.

| Module | Responsibility |
|---------|---------------|
| Cartographer | Generate occupancy grid maps |
| AMCL | Estimate global vehicle pose |
| EKF | Fuse wheel encoder and IMU measurements |
| MPC | Follow reference trajectory |
| LiDAR Analyzer | Detect obstacles |
| Trajectory Recorder | Record reference paths |
| Path Processor | Generate smooth MPC trajectory |
| Emergency Stop | Stop vehicle when required |

---

# 3. Repository Structure

```
ros2_ws/

├── README.md
├── DEVELOPER_README.md
│
├── src/
│   │
│   ├── qcar_science_night_pkg/
│   │
│   ├── launch/
│   │
│   ├── config/
│   │
│   │
│   ├── resource/
│   │
│   │
│   │
│   └── qcar_science_night_pkg/
│       │
│       ├── controllers/
│       ├── localization/
│       ├── perception/
│       ├── planning/
│       ├── trajectory/
│       ├── audio/
│       └── common/
│
├── utils/
├── build/
├── install/
└── log/
```

---

# 4. Software Stack

The project is built on top of ROS2 Humble.

## Middleware

- ROS2 Humble
- DDS

## Localization

- Cartographer
- AMCL
- robot_localization

## Motion Control

- Model Predictive Controller
- NumPy
- SciPy

## Perception

- RPLidar A2M12
- Intel RealSense D435i

## Visualization

- RViz2
- Matplotlib

---

# 5. ROS2 Node Architecture

The system is composed of independent ROS2 nodes.

## Hardware Layer

```
LiDAR Driver
Camera Driver
Wheel Encoder
IMU
```

These nodes provide sensor data.

---

## Localization Layer

```
EKF
```

Publishes

```
odom → base_link
```

---

```
AMCL
```

Publishes

```
map → odom
```

---

## Navigation Layer

```
Trajectory Recorder
```

Records

```
CSV trajectory
```

---

```
Trajectory Processor
```

Converts

```
CSV

↓

Smoothed NPY
```

---

```
MPC Controller
```

Reads

```
AMCL pose

Reference trajectory

Obstacle status
```

Produces

```
Throttle

Steering
```

---

## Perception Layer

```
LiDAR Sector Analyzer
```

Computes

- obstacle ahead
- lane availability
- emergency detection

---

```
Depth Emergency Node
```

Provides an independent emergency stop mechanism using the Intel RealSense D435i.

---

# 6. Launch Architecture

The system is launched in stages.

```
Hardware Drivers
        │
        ▼
Robot State Publisher
        │
        ▼
EKF
        │
        ▼
Map Server
        │
        ▼
AMCL
        │
        ▼
Localization Check
        │
        ▼
MPC Controller
```

**Important**

The MPC controller **must not** be launched until localization has converged.

Localization only estimates the robot pose.

The vehicle begins moving **only after** the MPC node has been started.

---

# Coding Guidelines

Each ROS2 node has a single responsibility.

Avoid placing multiple algorithms inside one node.

Prefer composition through ROS topics instead of tightly coupling modules.

Configuration values should always be stored in YAML or launch files rather than hard-coded whenever possible.

Use descriptive logging:

```python
self.get_logger().info(...)
self.get_logger().warn(...)
self.get_logger().error(...)
```

This greatly simplifies debugging during autonomous operation.

---
# 7. TF Tree

The navigation stack follows the standard ROS2 TF hierarchy.

```
                 map
                  │
                  │ (AMCL)
                  │
                 odom
                  │
                  │ (EKF)
                  │
              base_link
             /    |     \
            /     |      \
     base_scan camera_link imu_link
```

## Frame Description

| Frame | Publisher | Description |
|---------|-----------|-------------|
| map | AMCL | Global fixed frame |
| odom | EKF | Continuous odometry frame |
| base_link | EKF | Vehicle body frame |
| base_scan | Robot State Publisher | LiDAR frame |
| camera_link | Robot State Publisher | RealSense frame |

---

# 8. ROS Topics

## Sensor Topics

| Topic | Type | Publisher |
|---------|------|-----------|
| /scan | sensor_msgs/LaserScan | LiDAR |
| /camera/depth/image_rect_raw | sensor_msgs/Image | RealSense |
| /imu | sensor_msgs/Imu | IMU |
| /ekf_odom | nav_msgs/Odometry | EKF |

---

## Localization Topics

| Topic | Description |
|---------|-------------|
| /map | Occupancy grid |
| /amcl_pose | Estimated vehicle pose |
| /particlecloud | AMCL particles |
| /tf | Transform tree |

---

## Navigation Topics

| Topic | Description |
|---------|-------------|
| /reference_path | MPC reference trajectory |
| /cmd_vel | Velocity commands |
| /vehicle_command | Steering and throttle |

---

## Obstacle Topics

| Topic | Description |
|---------|-------------|
| /depth_emergency_stop | Emergency stop signal |
| /obstacle_status | Obstacle information |

---

# 9. Mapping Pipeline

The system uses Cartographer to create occupancy grid maps.

```
LaserScan

↓

Cartographer

↓

Pose Graph

↓

Occupancy Grid

↓

Map Server
```

### Mapping Procedure

1. Launch Cartographer.

2. Drive throughout the environment.

3. Return close to the starting location.

4. Save the pose graph.

```
demo_map.pbstream
```

5. Export the occupancy grid.

```
demo_map.pgm
demo_map.yaml
```

These files are later used by AMCL.

---

# 10. Localization Pipeline

Localization combines two independent systems.

## EKF

The Extended Kalman Filter fuses

- Wheel encoder
- IMU

and estimates

```
odom → base_link
```

---

## AMCL

AMCL estimates

```
map → odom
```

using

- Occupancy map
- LaserScan

The final pose becomes

```
map

↓

odom

↓

base_link
```

---

## Initialization

The localization procedure is

```
Load map

↓

Start AMCL

↓

Set Initial Pose

↓

Drive slightly

↓

Localization converges

↓

Launch MPC
```

The MPC controller should only be started after localization has converged.

---

# 11. Trajectory Recording

The trajectory recorder stores the vehicle pose while manually driving.

Recorded information includes

```
timestamp
x
y
theta
record reason
```

Example

```
t,tf_t,x,y,theta,yaw_deg,reason
...
```

---

## Recording Conditions

A waypoint is stored when

- Distance threshold exceeded

or

- Time threshold exceeded

This ensures

- Dense sampling in corners
- Sparse sampling on straight roads

---

# 12. Path Processing

The recorded trajectory is processed before it can be used by MPC.

Pipeline

```
Recorded CSV

↓

Remove stationary points

↓

Remove duplicate points

↓

Fit cubic B-Spline

↓

Uniform resampling

↓

Yaw computation

↓

Curvature computation

↓

Reference Path (.npy)
```

Output format

```
Nx4

x
y
yaw
curvature
```

---

## Why B-Splines?

The recorded trajectory contains

- Localization noise
- Small steering oscillations
- Uneven waypoint spacing

The spline

- Smooths the trajectory
- Produces continuous curvature
- Produces continuous heading
- Creates evenly spaced waypoints

This significantly improves MPC performance.

---

## Curvature Computation

Curvature is computed analytically from the spline.

```
κ =
(x'y'' − y'x'')

--------------------
(x'² + y'²)^(3/2)
```

Advantages

- Smooth steering
- Better prediction
- Stable MPC optimization

---

## Uniform Waypoint Spacing

After smoothing, the trajectory is resampled.

Typical spacing

```
3 cm
```

Advantages

- Constant prediction horizon
- Uniform MPC timing
- Consistent steering behaviour

---

# 13. Reference Path

The processed trajectory is stored as

```
recorded_path_amcl_final.npy
```

Structure

```
Column 0

x

Column 1

y

Column 2

yaw

Column 3

curvature
```

The MPC loads this file once during initialization.

No additional preprocessing is required during runtime.

---
# 14. Model Predictive Controller (MPC)

The MPC controller is responsible for autonomous vehicle motion. Unlike classical controllers such as Pure Pursuit or Stanley, MPC predicts the future vehicle motion over a finite prediction horizon and computes an optimal sequence of control commands.

The controller minimizes a cost function that considers:

- Cross-track error
- Heading error
- Steering effort
- Steering rate
- Velocity tracking
- Terminal error

The optimization is solved at every control cycle using the current vehicle state obtained from AMCL.

## Inputs

The MPC controller subscribes to:

- Vehicle pose (AMCL)
- Vehicle velocity
- Reference trajectory
- Obstacle status

## Outputs

The controller publishes:

- Steering angle
- Vehicle speed

These commands are sent directly to the QCar low-level controller.

---

# MPC Workflow

```
Reference Path

        │
        ▼

Current Vehicle Pose

        │
        ▼

Nearest Waypoint Search

        │
        ▼

Prediction Model

        │
        ▼

Optimization

        │
        ▼

Steering + Speed

        │
        ▼

Vehicle
```

The optimization is repeated continuously until the final waypoint is reached.

---

# Reference Path Tracking

The controller continuously searches for the closest waypoint.

```
Current Pose

↓

Nearest Waypoint

↓

Prediction Horizon

↓

Optimization
```

Using a moving prediction horizon allows the vehicle to anticipate upcoming turns rather than reacting after entering them.

---

# End-of-Path Detection

The controller monitors the remaining distance to the final waypoint.

If

```
distance_to_goal < threshold
```

the controller:

- Commands zero speed
- Centers steering
- Stops the vehicle safely
- Announces mission completion

---

# 15. Obstacle Avoidance

Obstacle avoidance is implemented independently from the MPC controller.

The perception module continuously analyses the LiDAR scan and classifies the surrounding environment into predefined regions.

```
LaserScan

↓

Cartesian Conversion

↓

Region Segmentation

↓

Obstacle Classification

↓

Lane Decision
```

---

## Detection Regions

The scan is divided into multiple regions.

### Front Lane

Used for detecting obstacles directly ahead.

```
      FRONT

 ┌─────────────┐
 │             │
 │             │
 └─────────────┘
```

---

### Left Lane

Checks whether the adjacent lane is available.

```
 ┌─────────────┐

 LEFT LANE

 └─────────────┘
```

---

### Right Lane

Equivalent region on the right side.

---

### Emergency Zone

A small region directly in front of the vehicle.

Objects entering this region trigger an immediate stop.

---

# Obstacle Avoidance State Machine

The navigation logic follows the state machine below.

```
Normal Driving

      │

Obstacle Detected

      │

      ▼

Within Overtake Distance?

      │

  No ─────────► Continue Driving

      │

     Yes

      │

      ▼

Check Adjacent Lane

      │

Lane Free?

      │

 No ─────────► Stop

      │

 Yes

      ▼

Lane Change

      │

Pass Obstacle

      │

Return To Original Lane

      │

Continue Path Following
```

This approach ensures the vehicle never attempts an unsafe lane change.

---

# Side Sweep Verification

Before changing lanes, the vehicle verifies that the adjacent lane is clear over an extended region rather than only checking a single point.

This prevents situations where:

- the front of the lane appears clear,
- but another obstacle exists alongside the vehicle.

Only after the entire side region is free is a lane change initiated.

---

# Lane Return

After successfully passing an obstacle:

```
Passed Obstacle

↓

Original Lane Clear

↓

Return

↓

Resume Path Following
```

The controller then switches back to the nominal reference trajectory.

---

# 16. Emergency Stop

The emergency stop layer is completely independent from obstacle avoidance.

Its purpose is to stop the vehicle whenever an unexpected object appears too close to react safely.

Sources include:

- LiDAR emergency region
- Intel RealSense D435i depth camera (optional)

If an emergency condition is detected:

```
Emergency

↓

Cancel Motion

↓

Speed = 0

↓

Steering = 0

↓

Wait
```

The emergency layer has higher priority than all other navigation modules.

---

# Safety Priority

The controller follows the priority order below.

```
Emergency Stop

        ▲

Obstacle Avoidance

        ▲

MPC

        ▲

Trajectory Following
```

Higher-priority modules override lower-priority ones whenever necessary.

---

# 17. Launch Architecture

The recommended launch sequence is:

```
Hardware Drivers

↓

Robot State Publisher

↓

EKF

↓

Map Server

↓

AMCL

↓

Set Initial Pose

↓

Localization Convergence

↓

MPC Controller

↓

Autonomous Navigation
```

The MPC controller **must not** be launched until localization has converged.

---

# 18. Configuration Files

The project uses configuration files extensively.

```
config/

├── ekf.yaml

├── amcl.yaml

├── cartographer.lua


└── trajectory parameters
```

Keeping parameters external allows the behaviour of the vehicle to be modified without recompiling the software.

---

# 19. Debugging

Useful commands during development.

## TF

```
ros2 run tf2_ros tf2_echo map base_link
```

---

## AMCL

```
ros2 topic echo /amcl_pose
```

---

## LiDAR

```
ros2 topic echo /scan
```

---

## Vehicle Commands

```
ros2 topic echo /cmd_vel
```

---

## Topic Frequency

```
ros2 topic hz /scan

ros2 topic hz /amcl_pose
```

---

## Node List

```
ros2 node list
```

---

## Topic List

```
ros2 topic list
```

---

## TF Tree

```
ros2 run tf2_tools view_frames
```

---

# Common Issues

## Vehicle does not move

Verify:

- MPC node is running.
- AMCL has converged.
- Initial pose has been set.
- Emergency stop is inactive.
- Reference trajectory exists.

---

## Poor Localization

Check:

- LiDAR visibility
- Map quality
- Initial pose
- AMCL covariance

---

## Oscillating Steering

Possible causes:

- Excessive curvature
- Poor waypoint spacing
- Incorrect MPC tuning
- Incorrect vehicle parameters

---

# 20. Extending the Project

The modular architecture allows new functionality to be added without modifying existing components.

Possible extensions include:

- Dynamic obstacle tracking
- Multi-obstacle overtaking
- Automatic parking
- Traffic sign recognition
- Traffic light detection
- Semantic mapping
- Global path planning
- Mission scheduler
- Fleet management
- Cloud connectivity

---

# Coding Guidelines

Developers should follow the principles below.

- Keep each ROS2 node focused on a single responsibility.
- Avoid hard-coded parameters.
- Use ROS parameters whenever possible.
- Prefer composition over large monolithic nodes.
- Use descriptive log messages.
- Document new topics and parameters.
- Keep launch files modular.
- Test each module independently before integration.

---

# Future Improvements

Several enhancements are planned for future versions of the project.

- Adaptive MPC tuning
- Dynamic speed planning
- Online trajectory generation
- Dynamic obstacle prediction
- Multi-lane navigation
- Automatic localization routine
- Human-aware navigation
- Multi-robot coordination
- Web-based monitoring dashboard

---

# Conclusion

This project demonstrates a complete autonomous navigation stack for the Quanser QCar built on ROS2 Humble. The system combines modern localization, mapping, trajectory generation, Model Predictive Control, and perception algorithms to enable robust indoor autonomous navigation.

The modular architecture allows each subsystem to be developed, tested, and extended independently while maintaining clear interfaces between localization, planning, perception, and control.