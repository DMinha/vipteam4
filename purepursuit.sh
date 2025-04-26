#!/bin/bash

# --- Configuration ---
# Set the path to your SLAM parameters file (optional, but recommended)
 params file
# Set the name of your track file (expected to be in the launch directory)
TRACK_FILENAME="map.p" # <<< ADJUST FILENAME IF NEEDED

# --- Script Start ---
echo "Starting F1TENTH Pure Pursuit Run..."

# Optional: Clean up environment
# echo "Restarting ROS 2 daemon..."
# ros2 daemon stop && sleep 1 && ros2 daemon start && sleep 1 && ros2 daemon status

# Build the ROS 2 package
echo "Building reactive_racing package..."
colcon build --symlink-install --packages-select reactive_racing
BUILD_RESULT=$? # Check build result
if [ $BUILD_RESULT -ne 0 ]; then
    echo "ERROR: Build failed, exiting."
    exit 1
fi
echo "Build successful."

# Source ROS 2 and workspace setup files
echo "Sourcing ROS environment..."
source /opt/ros/humble/setup.bash
source install/setup.bash

# --- Cleanup Function ---
# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "Caught Ctrl+C or SIGTERM. Shutting down launched nodes..."
    # Kill processes using their PIDs
    # Use kill -TERM first for graceful shutdown, then -KILL if needed
    kill -TERM $RACE_LAUNCH_PID $SLAM_PID $VESC_ODOM_PID $TF_LASER_PID $ZED_PID 2>/dev/null
    sleep 2 # Give time for graceful shutdown
    kill -KILL $RACE_LAUNCH_PID $SLAM_PID $VESC_ODOM_PID $TF_LASER_PID $ZED_PID 2>/dev/null
    wait # Wait for kills to complete
    echo "Cleanup finished."
    # Optional: Stop daemon if started by this script
    # ros2 daemon stop
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# --- Launch Nodes ---

# Launch ZED camera (if needed)
echo "Launching ZED camera..."
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2 &
ZED_PID=$!
sleep 2 # Allow camera to initialize

# Launch SLAM Toolbox (provides map -> odom transform)
echo "Launching SLAM Toolbox..."
# Construct the launch command with optional parameters file
SLAM_CMD="ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false"
if [ -f "$SLAM_PARAMS_FILE" ]; then
    SLAM_CMD="$SLAM_CMD params_file:=$SLAM_PARAMS_FILE"
    echo "Using SLAM params: $SLAM_PARAMS_FILE"
else
    echo "WARN: SLAM params file not found or not set: $SLAM_PARAMS_FILE. Using defaults."
fi
$SLAM_CMD &
SLAM_PID=$!
echo "Waiting for SLAM Toolbox to initialize (publishing map -> odom TF)..."
sleep 5 # Give SLAM time to start up and publish the transform

# Launch VESC-to-Odometry Node (provides odom -> base_link transform and /odom topic)
echo "Launching VESC to Odometry node..."
ros2 run vesc_ackermann vesc_to_odom_node --ros-args \
  -p odom_frame:=odom \
  -p base_frame:=base_link \
  -p speed_to_erpm_gain:=-4614.0 \
  -p speed_to_erpm_offset:=0.0 \
  -p steering_angle_to_servo_gain:=-1.2135 \
  -p steering_angle_to_servo_offset:=0.5304 \
  -p wheelbase:=0.33 \
  -p publish_tf:=true & # Ensure TF is published
VESC_ODOM_PID=$!
sleep 2

# Static transform for the laser sensor relative to the base link
echo "Publishing static TF for laser..."
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link laser &
TF_LASER_PID=$!
sleep 1

# Launch the main racing logic (includes VESC Driver, Mux, Pure Pursuit Controller)
echo "Launching reactive_racing stack..."
# Construct the path to the track file within the installed launch directory
TRACK_PATH_ARG="track_path:=$(ros2 pkg prefix reactive_racing)/share/reactive_racing/launch/$TRACK_FILENAME"
echo "Using track path: $TRACK_PATH_ARG"
ros2 launch reactive_racing race.launch.py $TRACK_PATH_ARG &
RACE_LAUNCH_PID=$!
sleep 3 # Allow controller and other nodes in launch file to initialize

# --- Keep Alive ---
echo ""
echo "-----------------------------------------------------"
echo "System running. Pure Pursuit controller is active."
echo "Ensure SLAM has localized correctly."
echo "Press Ctrl+C to stop all processes cleanly."
echo "-----------------------------------------------------"

# Wait for the main launch process to exit (or be killed by Ctrl+C)
wait $RACE_LAUNCH_PID

# Explicitly call cleanup in case wait exits normally (less likely for launch files)
cleanup

echo "Script finished."
exit 0
