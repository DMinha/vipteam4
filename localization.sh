#!/bin/bash
# Clean up environment. 
echo "Stopping and restarting ROS 2 daemon..."
ros2 daemon stop 
sleep 2
ros2 daemon start 
sleep 2
ros2 daemon status 

# Build your ROS 2 package
colcon build --symlink-install --packages-select reactive_racing 

# Source ROS 2 and workspace setup
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch ZED camera and reactive racing logic
#ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 & 
ros2 launch reactive_racing race.launch.py &
sleep 3

# Static transforms
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link laser &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &

# Launch RViz
rviz2 -d /home/nvidia/.rviz2/rviz_slam_config_vip.rviz &
sleep 15

# Launch map_server to publish your static map
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/home/nvidia/ISTHISIT.yaml &
sleep 2
ros2 run nav2_util lifecycle_bringup map_server & 
sleep 2

# Replace SLAM Toolbox with AMCLz
echo "Starting AMCL localization..."
ros2 run nav2_amcl amcl --ros-args -p use_sim_time:=false -p publish_pose:=true -p base_frame_id:=base_link &
sleep 2
ros2 run nav2_util lifecycle_bringup amcl & 
sleep 2

# Set initial pose - adjust coordinates as needed for your map
echo "Setting initial pose..."
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {stamp: {sec: 1764475147, nanosec: 150273775}, frame_id: 'map'}, pose: {pose: {position: {x: 1.0310522317886353, y: 0.29995158314704895, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: -0.07761408850660331, w: 0.9969834769269194}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]}}" &
# 3. VESC TO ODOM (Odometry Calculation)
# echo "  → Launching vesc_to_odom_node"
# ros2 run vesc_ackermann vesc_to_odom_node --ros-args \
#   -p odom_frame:=odom \
#   -p base_frame:=base_link \
#   -p speed_to_erpm_gain:=5000.0 \
#   -p speed_to_erpm_offset:=0.0 \
#   -p steering_angle_to_servo_gain:=-0.7 \
#   -p steering_angle_to_servo_offset:=0.5 \
#   -p wheelbase:=0.33 \
#   -p publish_tf:=true &

# Launch Nav2 (but only the components you need)
#ros2 launch nav2_bringup navigation_launch.py &

# Keep the script alive
wait
