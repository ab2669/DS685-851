# Assignment 1 DS685-851

**Name:** Andrew Boksz  
**Date:** September 21st 2025  


## How to run

1. Build overlay workspace:
   # in bash terminal
   cd ~/overlay_ws
   source /opt/ros/$ROS_DISTRO/setup.bash
   colcon build --symlink-install
   source install/setup.bash

2. Run demo
    ros2 launch tb_worlds tb_demo_world.launch.py

Screenshots included