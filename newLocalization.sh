ros2 bag record /odom /map /tf /tf_static /amcl_pose /pose

ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/home/nvidia/my_map.yaml

ros2 run nav2_util lifecycle_bringup map_server

ros2 launch spring2025 race.launch.py

ros2 run nav2_amcl amcl --ros-args -p use_sim_time:=false -p publish_pose:=true -p base_frame_id:=base_link

ros2 run nav2_util lifecycle_bringup amcl

ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {pose: {position: {x: 2.8760793209075928, y: -0.15586026012897491, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.7157604755895094, w: 0.6983458610057622}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]}}"

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p max_linear_speed:=0.5 -p max_angular_speed:=0.5

