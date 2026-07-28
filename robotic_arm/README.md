# Robotic Arm Vision and Control

This project provides a complete pipeline for tabletop pick-and-place tasks,
including camera calibration, object detection, coordinate conversion, inverse
kinematics, motion planning, and a local control console.

The default target is a red soda can. YOLO-World detects the object, the red
contour is used to estimate the centre of the can base, and the detected pixel
is converted into the robot's J1-centred coordinate frame. The controller then
generates commands for `J1,J2,J3,J4,J6`.

## Robot and 3D Files

This project uses
[Robotic Arm with Servo & Arduino by Emre Kalem](https://makerworld.com/en/models/1134925-robotic-arm-with-servo-arduino?from=search#profileId-1135927).

Download the printable STL files, assembly information, and bill of materials
from the MakerWorld page above. This repository already includes the
project-authored URDF at:

```text
robotic_arm_console/public/robot/robotic_arm_v0_6.urdf
```

To enable the console's optional 3D preview, place the STL files referenced by
the URDF in the same `robotic_arm_console/public/robot/` directory. Camera
calibration, detection, planning, and physical control remain available
without the STL files.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Vision, coordinate conversion, motion planning, and control entry point |
| `calibration/` | Camera intrinsics, workspace-plane, and joint calibration |
| `vision/` | YOLO-World detection and pixel-to-robot conversion |
| `control/` | Inverse kinematics, workspace analysis, and pick-and-place trajectories |
| `robotic_arm_console/` | Local web console and Python gateway |

## 1. Install the Python Environment

Python 3.11 is recommended. The project uses the `codex` Conda environment by
default:

```bash
conda run -n codex python -m pip install \
  numpy scipy opencv-python ultralytics matplotlib
```

YOLO-World and CLIP weights are downloaded automatically on the first
detection request. An existing `yolov8s-world.pt` checkpoint can instead be
placed at:

```text
vision/models/yolov8s-world.pt
```

## 2. Calibrate the Camera

Generate the printable checkerboards:

```bash
conda run -n codex python calibration/generate_checkerboards.py
```

Print a checkerboard from `calibration/checkerboards/` at actual size and
measure the printed square size. Fix the camera in its operating position, then
capture images and calculate the camera intrinsics:

```bash
conda run -n codex python calibration/calibrate_camera.py \
  --camera-index 0 \
  --square-mm 25.0
```

The default output is:

```text
calibration/output/camera_intrinsics.json
```

Repeat calibration after changing the camera, resolution, crop, or camera
position.

## 3. Run the Vision and Control Pipeline

Coordinates are measured in millimetres in the J1-centred robot frame:

```bash
conda run -n codex python main.py \
  --place 245 120 0 \
  --grasp-z-mm 80 \
  --current-control 90 60 90 90 150
```

Camera-window controls:

- `c`: capture the checkerboard and establish workspace coordinates
- `s`: capture the current frame, detect the target, and plan a trajectory
- `i`: confirm the planned trajectory; physical commands are sent only when
  the program was started with `--send`
- `q` or `Esc`: quit or cancel

The default mode plans motion without sending commands. Add `--send` only after
checking the calibration, target coordinates, and complete trajectory. Set the
robot controller address with the `ROBOT_IP` environment variable or the
`--robot-ip` option.

## 4. Start the Local Console

Start the Python gateway:

```bash
cd robotic_arm_console
npm install
npm run bridge
```

Open another terminal and start the web interface:

```bash
cd robotic_arm_console
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The console connects to
`http://127.0.0.1:8765` by default and displays the camera stream, detections,
joint angles, and task stage. Planning and physical sending are separate
actions, and sending requires an additional confirmation.

## Current Operating Boundaries

- Only the red soda can currently has grasp-point geometry and can enter the
  control pipeline. Other text categories return detection boxes only.
- J5 remains fixed at `q5=0` and is not included in the control payload.
- Camera calibration is valid only for the calibrated lens, resolution, crop,
  and camera position.
- The first physical run should use two nearby, simple positions at low speed
  without grasping an object.
