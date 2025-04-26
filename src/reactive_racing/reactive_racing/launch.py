#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Launch the joystick driver node from the 'joy' package.
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    # Launch the custom node that reads joystick commands and publishes Ackermann commands.
    ackermann_teleop_node = Node(
        package='joy_teleop',                # Replace with your package name if different
        executable='joystick_ackermann_teleop',  # Replace with the installed executable name of your node
        name='joystick_ackermann_teleop',
        output='screen'
    )

    return LaunchDescription([
        joy_node,
        ackermann_teleop_node
    ])

