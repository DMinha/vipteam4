from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vesc_driver',
            executable='vesc_driver_node',
            name='vesc_driver',
            parameters=[{
                'port': '/dev/ttyACM0',
                # Changed to integers from doubles
                'brake_max': 32000,
                'brake_min': -32000,
                'speed_max': 32000,
                'speed_min': -32000,
                'position_max': 32000,
                'position_min': -32000,
                'servo_max': 0.85,
                'servo_min': 0.15
            }]
        ),
        Node(
            package='vesc_ackermann',
            executable='vesc_to_odom_node',
            name='vesc_to_odom',
            parameters=[{
                'odom_frame': 'odom',
                'base_frame': 'base_link',
                'publish_tf': True,
                'use_servo_cmd_to_calc_angular_velocity': True,
                'servo_max': 0.85,
                'servo_min': 0.15,
                'wheelbase': 0.25,  # Adjust this to your robot's wheelbase
                # Adding required parameters
                'speed_to_erpm_gain': 4614.0,  # Adjust this for your VESC
                'speed_to_erpm_offset': 0.0,
                'steering_angle_to_servo_gain': -1.0,
                'steering_angle_to_servo_offset': 0.5,
                'rolling_average_size': 10
            }]
        )
    ])
