# DS685-851 Assignment 2: ROS2 Webcam RTSP & Gazebo Maze

## Submission Summary

This repo contains all source code, models, world files, and detailed screenshots demonstrating completion of Assignment 2 Tasks 1 & 2. Screenshots for both tasks are in `assignment-2/screenshots/`, clearly named.

---

## Task 1: MacBook Webcam RTSP Publisher to ROS2

- Developed a custom ROS2 Python node: `webcam_rtsp_publisher.py` that publishes an image stream from a MacBook’s built-in camera to `/camera/image_raw` using OpenCV and cv_bridge.
- Mac camera streamed using `ffmpeg` to a local RTSP server (mediamtx). Docker accessed the stream using `rtsp://host.docker.internal:8554/mystream`.
- Required significant install/config, including resolving:
  - NumPy and OpenCV version conflicts inside ROS2 Docker (numpy 2.x vs cv_bridge built against numpy 1.x).
  - Created a new, clean official ROS2 container for compatibility.
  - Accidentally deleted the first container before copying files—had to recreate publisher code (whoops, lesson learned!).
- Node and topic tested and confirmed working with `ros2 topic echo /camera/image_raw`.
- Screenshots documented: publisher running, `ros2 topic list`, `ros2 topic echo`, and VLC live stream.

---

## Task 2: 3D Assets and Maze in Gazebo

- Enhanced the turtlebot-maze world to include:
  - Four benches touching maze walls, each with different COCO objects (bottle, cup, book).
  - Used custom model folders for each asset (`models/bench/`, etc.).
  - Edited `assignment2_world.sdf.xacro` to place new objects on benches.
- All design/model code, and supplementary screenshots, are included in repo.
- Verified world consistent with assignment requirements via multiple overview/close-up/entity-tree/code screenshots.
- All Task 2 screenshots moved from Desktop to screenshots directory, clearly named for grading.

---

## Directory Guide

- `assignment-2/webcam_rtsp_publisher.py`    : RTSP-to-ROS2 camera publisher code
- `assignment-2/models/` and `worlds/`        : Custom model assets and world configs for Gazebo
- `assignment-2/screenshots/`                 : Proof of work/steps for Tasks 1 and 2

---

## Special Notes

- Container loss (deletion) required partial Task 1 code re-write.
- All steps followed for reproducible rig, all solutions/decisions and fixes included (see commit history and screenshot names).

---

## Full Completion:

- Task 1 (Webcam RTSP publisher): COMPLETE
- Task 2 (3D assets in maze): COMPLETE
