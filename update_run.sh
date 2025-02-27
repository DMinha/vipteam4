#echo "Entering virtual env \"one_env\""
#source one_env/local/bin/activate
#echo "Entered one_env"
#colcon build --symlink-install --packages-select reactive_racing 
#source /opt/ros/humble/setup.bash
#source install/setup.bash
#ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 & ros2 launch reactive_racing race.launch.py 
#ros2 launch reactive_racing race.launch.py
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
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 & 
ros2 launch reactive_racing race.launch.py &

# Wait a bit to ensure initial nodes start correctly
sleep 3

# Publish static transforms
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link laser &

# Launch Extended Kalman Filter (EKF) for localization
ros2 launch robot_localization ekf.launch.py &

# Launch SLAM Toolbox
ros2 launch slam_toolbox online_async_launch.py &

# Launch Navigation2 stack
ros2 launch nav2_bringup navigation_launch.py &

# Launch VESC-to-Odometry Node with parameters

ros2 run vesc_ackermann vesc_to_odom_node --ros-args \
  -p odom_frame:=odom \
  -p base_frame:=base_link \
  -p speed_to_erpm_gain:=2000.0 \
  -p speed_to_erpm_offset:=0.0 \
  -p steering_angle_to_servo_gain:=-1.0 \
  -p steering_angle_to_servo_offset:=0.5 \
  -p wheelbase:=0.25 \
  -p publish_tf:=true &

# Keep the script running to avoid terminating background processes
wait

