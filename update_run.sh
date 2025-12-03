#!/bin/bash
# Uncomment if using a virtual environment
# echo "Entering virtual env \"one_env\""
# source one_env/local/bin/activate
# echo "Entered one_env"

# Build the ROS 2 package
colcon build --symlink-install --packages-select reactive_racing 

# Source ROS 2 and workspace setup files
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch ZED camera and reactive racing node in the background
#ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 & 
ros2 launch reactive_racing race.launch.py &

# Wait a bit to ensure initial nodes start correctly
sleep 3

# ONLY publish the static transform for laser sensor placement
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link laser &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &


# Launch SLAM Toolbox with specific parameters to use its pose
ros2 launch slam_toolbox online_async_launch.py &

# Wait for SLAM to initialize
sleep 2
#ros2 launch robot_localization ekf.launch.py & 

# Launch VESC-to-Odometry Node with parameters and ENABLE TF publishing
ros2 run vesc_ackermann vesc_to_odom_node --ros-args \
  -p odom_frame:=odom \
  -p base_frame:=base_link \
  -p speed_to_erpm_gain:=-5000.0 \
  -p speed_to_erpm_offset:=0.0 \
  -p steering_angle_to_servo_gain:=-0.7 \
  -p steering_angle_to_servo_offset:=0.5 \
  -p wheelbase:=0.33 \
  -p publish_tf:=true &  # Changed to true to ensure TF is published
  
# Launch Navigation2 stack 
#ros2 launch nav2_bringup navigation_launch.py &

# Keep the script running to avoid terminating background processes
wait
