# Stereo Calibration, Multi-Camera Detection, and 3D Triangulation

A real-time multi-camera object detection system using ROS2 and YOLOv8 with probabilistic fusion.

This project performs synchronized multi-camera detection and fuses results across views to improve confidence, robustness, and reliability.

---

## 📑 Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Fusion Logic](#fusion-logic)
4. [System Architecture](#system-architecture)
5. [Installation](#installation)
6. [Running the System](#running-the-system)
7. [Verification](#verification)
8. [Expected Behavior](#expected-behavior)
9. [Stop All Nodes](#stop-all-nodes)
10. [Camera Calibration & Triangulation](#camera-calibration--triangulation)
    - [Stereo Calibration](#stereo-calibration)
    - [Extract Synchronized Frame Pair](#extract-synchronized-frame-pair)
    - [Manual Point Selection](#manual-point-selection)
    - [Triangulate Clicked Points](#triangulate-clicked-points)
    - [Automatic Stereo YOLO Triangulation](#automatic-stereo-yolo-triangulation)
11. [Understanding 3D Output](#understanding-3d-output)
12. [Recommended Workflow](#recommended-workflow)
13. [Practical Tips](#practical-tips)
14. [Common Commands Reference](#common-commands-reference)
15. [Author](#author)
16. [Acknowledgements](#acknowledgements)

---

## Project Structure

```text
/home/shahid/multicam_3d/
├── data/
│   ├── left_calib.mp4
│   ├── right_calib.mp4
│   ├── video_cam0.mp4
│   ├── video_cam1.mp4
│   ├── left_frame.png
│   └── right_frame.png
├── scripts/
│   ├── stereo_calibrate_from_videos.py
│   ├── extract_same_frame.py
│   ├── click_points_stereo.py
│   ├── triangulate_points.py
│   └── stereo_yolo_triangulation.py
├── src/
│   ├── multicam_fusion_detector/
└── calib_out/
    ├── stereo_calib.npz
    └── summary.json
```

---

## 🚀 Features

- Multi-camera input (USB / RTSP / video files)
- YOLOv8-based object detection
- Optional image enhancement module
- Real-time ROS2 pipeline
- Probabilistic multi-view fusion:
  - Object matching across cameras
  - Confidence fusion
  - Bounding box fusion
- Works on:
  - Desktop GPU
  - Jetson / embedded systems

---

## 🧠 Fusion Logic

Instead of simple concatenation, this system performs **true multi-view fusion**:

- Objects are matched across cameras using spatial proximity and class
- Confidence is fused using: `score = 1 - (1 - s0)*(1 - s1)`
- Bounding boxes are merged: `bbox = average(bbox_cam0, bbox_cam1)`

### Result

| Scenario | Behavior |
|----------|----------|
| Both cameras see object | Confidence increases |
| One camera blocked | Confidence decreases |
| Both disagree | No fusion |

---

## 🧩 System Architecture

This project implements a **multi-camera, synchronized, detection-level fusion pipeline** in ROS2. The system is designed so that standard single-image object detectors such as YOLOv8 can be reused in a **multi-view perception framework** without modifying the detector architecture itself.

### 1. High-Level Overview

The system has three main stages:

1. **Image acquisition**
2. **Per-camera object detection**
3. **Cross-camera fusion**

At runtime, each camera publishes its own image stream. The fusion node synchronizes frames from all cameras, runs detection on each image independently, and then combines the detections into a single fused result.

### 2. Processing Pipeline

```text
Camera 0 (/pc_usb0/image_raw) ----\
                                    \
                                     --> Approximate Time Synchronizer
                                    /
Camera 1 (/pc_usb1/image_raw) ----/

          synchronized image set
                    |
                    v
      +-----------------------------------+
      |     MultiCamFusionDetector Node   |
      +-----------------------------------+
          |                        |
          |                        |
          v                        v
   Per-camera detection      Optional enhancement
   using YOLOv8              (pre-processing)
          |                        |
          +-----------+------------+
                      |
                      v
           Per-camera detection lists
              cam0_dets, cam1_dets
                      |
                      v
              Cross-camera fusion
       - object matching across views
       - confidence fusion
       - bounding box fusion
                      |
                      v
         /fusion/detections (final output)
```

**Additional outputs:**
- `/fusion/cam0/detections`
- `/fusion/cam1/detections`
- `/fusion/cam0/annotated`
- `/fusion/cam1/annotated`

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/multicam-fusion-detector.git
cd multicam-fusion-detector

# Build ROS2 workspace
cd ~/bluerov/bluerov_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select multicam_fusion_detector --symlink-install
source install/setup.bash
```

---

## ▶️ Running the System

### OPTION 1 — Live Cameras

**Terminal 1 (Camera 0)**
```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -r image_raw:=/pc_usb0/image_raw \
  -p video_device:=/dev/video4
```

**Terminal 2 (Camera 1)**
```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -r image_raw:=/pc_usb1/image_raw \
  -p video_device:=/dev/video6
```

**Terminal 3 (Fusion + Detection)**
```bash
source ~/venvs/multicam_det_cpu/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/bluerov/bluerov_ws/install/setup.bash

ros2 launch multicam_fusion_detector run.launch.py \
  venv_python:=/home/shahid/venvs/multicam_det_cpu/bin/python
```

### OPTION 2 — Using Recorded Videos

**Terminal 1 (Video Cam 0)**
```bash
ros2 run multicam_fusion_detector video_publisher --ros-args \
  -p video_path:=/home/shahid/videos/video_cam0.mp4 \
  -p topic_name:=/pc_usb0/image_raw \
  -p fps:=30
```

**Terminal 2 (Video Cam 1)**
```bash
ros2 run multicam_fusion_detector video_publisher --ros-args \
  -p video_path:=/home/shahid/videos/video_cam1.mp4 \
  -p topic_name:=/pc_usb1/image_raw \
  -p fps:=30
```

**Terminal 3 (Fusion + Detection)**
```bash
ros2 launch multicam_fusion_detector run.launch.py \
  venv_python:=/home/shahid/venvs/multicam_det_cpu/bin/python
```

---

## 🧪 Verification

Check fused output:

```bash
ros2 topic echo /fusion/detections
```

You should see:

```text
class_id: person
score: 0.92
id: fused_cam0_cam1
```

---

## 📊 Expected Behavior

| Test | Result |
|------|--------|
| Both cameras see object | High confidence |
| One camera blocked | Lower confidence |
| No object | No detection |

---

## 🛑 Stop All Nodes

```bash
pkill -f video_publisher
pkill -f multicam_fusion_detector
pkill -f v4l2_camera
```

---

# Camera Calibration & Triangulation Pipeline

## Stereo Calibration

### Step 1: Stereo Calibration from Checkerboard Videos

Run stereo calibration using the direct script:

```bash
python3 /home/shahid/multicam_3d/scripts/stereo_calibrate_from_videos.py \
  --left_video /home/shahid/multicam_3d/data/left_calib.mp4 \
  --right_video /home/shahid/multicam_3d/data/right_calib.mp4 \
  --board_cols 9 \
  --board_rows 5 \
  --square_size 0.025 \
  --show
```

### Requirements for Calibration

- One left calibration video
- One right calibration video
- Both synchronized
- Both showing the same checkerboard pattern

### Checkerboard Parameters

| Parameter | Description |
|-----------|-------------|
| `board_cols` | Number of inner corners across width |
| `board_rows` | Number of inner corners across height |
| `square_size` | Checker square size in meters |

**Example:** 9 inner corners across, 5 inner corners down, square size = 0.025 m

### Expected Output

This creates:
- `calib_out/stereo_calib.npz`
- `calib_out/summary.json`

Check the summary:

```bash
cat calib_out/summary.json
```

**A good result usually has:**
- Low left/right reprojection error
- Stereo reprojection error around 0.5 to 1.5 px is usually usable
- A reasonable baseline length

---

## Extract Synchronized Frame Pair

Extract the same frame from synchronized videos for manual point triangulation:

```bash
python3 /home/shahid/multicam_3d/scripts/extract_same_frame.py \
  --left_video /home/shahid/multicam_3d/data/video_cam0.mp4 \
  --right_video /home/shahid/multicam_3d/data/video_cam1.mp4 \
  --frame_idx 100 \
  --left_out /home/shahid/multicam_3d/data/left_frame.png \
  --right_out /home/shahid/multicam_3d/data/right_frame.png
```

This saves `left_frame.png` and `right_frame.png`.

---

## Manual Point Selection

Run the interactive point selection tool:

```bash
python3 /home/shahid/multicam_3d/scripts/click_points_stereo.py \
  --left_image /home/shahid/multicam_3d/data/left_frame.png \
  --right_image /home/shahid/multicam_3d/data/right_frame.png
```

### How to Use

| Action | Key |
|--------|-----|
| Click a point in the left image | Mouse click |
| Click the same physical point in the right image | Mouse click |
| Save points | `s` |
| Undo last point | `u` |
| Quit | `q` or `Esc` |

**Note:** Keep the order the same for corresponding points.

This saves `left_points.json` and `right_points.json`.

---

## Triangulate Clicked Points

```bash
python3 /home/shahid/multicam_3d/scripts/triangulate_points.py \
  --calib calib_out/stereo_calib.npz \
  --left_points_json left_points.json \
  --right_points_json right_points.json
```

### Output Example

```
point_0: X=0.298470, Y=-0.032005, Z=0.717078
```

This means the point is reconstructed in 3D, in meters, in the stereo rectified coordinate frame. It also saves `triangulated_points.json`.

---

## Automatic Stereo YOLO Triangulation

This is the next step after manual clicking. It performs:

- Rectify left and right video frames
- Run YOLO on both
- Match detections by class and center distance
- Triangulate matched detection centers
- Display class name, confidence, and estimated depth

### Run

```bash
python3 /home/shahid/multicam_3d/scripts/stereo_yolo_triangulation.py \
  --left_video /home/shahid/multicam_3d/data/video_cam0.mp4 \
  --right_video /home/shahid/multicam_3d/data/video_cam1.mp4 \
  --calib /home/shahid/Downloads/stereo_calibration-main/calib_out/stereo_calib.npz \
  --yolo_model yolov8n.pt \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.45 \
  --device cuda \
  --match_dist_thresh 80 \
  --show
```

### Notes

| Parameter | Recommendation |
|-----------|----------------|
| `--device` | Use `cuda` if CUDA works, otherwise `cpu` |
| `--imgsz` | Lower to 416 or 320 if performance is slow |
| `--match_dist_thresh` | Increase if matching is too strict, decrease if wrong detections are being matched |

### Output

This saves `triangulated_detections.json`. Each matched object will contain:

- Class ID
- Class name
- Left confidence
- Right confidence
- Fused confidence
- Left center
- Right center
- 3D XYZ position in meters

---

## Understanding 3D Output

A result like:
```
X = 0.12
Y = -0.03
Z = 0.74
```

Means:

| Axis | Meaning |
|------|---------|
| **X** | Horizontal offset |
| **Y** | Vertical offset |
| **Z** | Depth from the stereo camera rig |

**Note:** Larger Z means farther away. Z is the most important value for distance estimation.

---

## Recommended Workflow

### Calibration Workflow
1. Record synchronized checkerboard videos
2. Run stereo calibration
3. Inspect errors
4. Keep the best calibration file

### Triangulation Workflow
1. Extract same frame from left/right videos
2. Manually click matching points
3. Triangulate
4. Verify depths look reasonable

### Automatic Detection Workflow
1. Load stereo calibration
2. Rectify both videos
3. Detect with YOLO on both
4. Match detections
5. Triangulate bbox centers
6. Save 3D detection results

---

## Practical Tips

### For Better Calibration
- Move checkerboard to different depths
- Tilt and rotate it
- Cover the full image area
- Avoid blurry frames
- Do not use too many nearly identical frames

### For Better Triangulation
- Click the exact same physical point in both images
- Use synchronized frames
- Use clear, sharp images

### For Better Detection Matching
- Start with a simple scene
- One object at a time is easiest
- Matched classes should be the same
- Lower the distance threshold if mismatches happen
- Raise it if true matches are missed

---

## Common Commands Reference

### Activate Environment
```bash
source ~/venvs/multicam_det_cpu/bin/activate
```

### Run Stereo Calibration
```bash
python3 /home/shahid/multicam_3d/scripts/stereo_calibrate_from_videos.py \
  --left_video /home/shahid/multicam_3d/data/left_calib.mp4 \
  --right_video /home/shahid/multicam_3d/data/right_calib.mp4 \
  --board_cols 9 \
  --board_rows 5 \
  --square_size 0.025 \
  --show
```

### Extract Synchronized Frame Pair
```bash
python3 /home/shahid/multicam_3d/scripts/extract_same_frame.py \
  --left_video /home/shahid/multicam_3d/data/video_cam0.mp4 \
  --right_video /home/shahid/multicam_3d/data/video_cam1.mp4 \
  --frame_idx 100 \
  --left_out /home/shahid/multicam_3d/data/left_frame.png \
  --right_out /home/shahid/multicam_3d/data/right_frame.png
```

### Click Points
```bash
python3 /home/shahid/multicam_3d/scripts/click_points_stereo.py \
  --left_image /home/shahid/multicam_3d/data/left_frame.png \
  --right_image /home/shahid/multicam_3d/data/right_frame.png
```

### Triangulate Clicked Points
```bash
python3 /home/shahid/multicam_3d/scripts/triangulate_points.py \
  --calib calib_out/stereo_calib.npz \
  --left_points_json left_points.json \
  --right_points_json right_points.json
```

### Automatic YOLO + Triangulation
```bash
python3 /home/shahid/multicam_3d/scripts/stereo_yolo_triangulation.py \
  --left_video /home/shahid/multicam_3d/data/video_cam0.mp4 \
  --right_video /home/shahid/multicam_3d/data/video_cam1.mp4 \
  --calib /home/shahid/Downloads/stereo_calibration-main/calib_out/stereo_calib.npz \
  --yolo_model yolov8n.pt \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.45 \
  --device cuda \
  --match_dist_thresh 80 \
  --show
```

---

## What This Pipeline Provides

After completing this pipeline, you can perform:

-  Stereo calibration
-  Image rectification
-  3D point triangulation
-  3D object center triangulation from detections
-  Class-aware stereo matching
-  Depth estimation from two cameras

This is the foundation for:

-  3D object localization
-  Stereo perception for robotics
-  Obstacle distance estimation
-  Integration into ROS 2 for real-time 3D detections

---

## 👨‍💻 Author

**Shahid Ahamed Hasib**  
MSc Marine and Maritime Intelligent Robotics

---

## ⭐ Acknowledgements

- Ultralytics YOLOv8
- ROS2
- OpenCV

