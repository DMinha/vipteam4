#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped

class JoystickAckermannTeleop(Node):
    def __init__(self):
        super().__init__('joystick_ackermann_teleop')

        # Declare parameters for joystick-to-ackermann mapping.
        # You can adjust these values as needed.
        self.declare_parameter('axis_speed', 1)       # e.g., left stick vertical axis for speed
        self.declare_parameter('axis_steering', 0)      # e.g., left stick horizontal axis for steering
        self.declare_parameter('scale_speed', 1.0)      # Speed multiplier
        self.declare_parameter('scale_steering', 0.5)   # Steering multiplier (radians)
        self.declare_parameter('deadzone', 0.1)           # Threshold to ignore small inputs
        self.declare_parameter('ackermann_topic', '/ackermann_cmd')

        # Retrieve parameters.
        self.axis_speed = self.get_parameter('axis_speed').value
        self.axis_steering = self.get_parameter('axis_steering').value
        self.scale_speed = self.get_parameter('scale_speed').value
        self.scale_steering = self.get_parameter('scale_steering').value
        self.deadzone = self.get_parameter('deadzone').value
        self.ackermann_topic = self.get_parameter('ackermann_topic').value

        # Publisher for Ackermann drive commands.
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.ackermann_topic, 10)
        # Subscriber for joystick messages.
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)

        self.get_logger().info("Joystick Ackermann Teleop Node has started.")

    def joy_callback(self, msg: Joy):
        # Log the raw joystick inputs.
        self.get_logger().info(f"Joystick Input => Axes: {msg.axes}, Buttons: {msg.buttons}")

        # Ensure the message has enough axes.
        if len(msg.axes) > max(self.axis_speed, self.axis_steering):
            # Retrieve and filter the joystick values.
            speed_val = msg.axes[self.axis_speed]
            steering_val = msg.axes[self.axis_steering]

            # Apply deadzone filtering.
            if abs(speed_val) < self.deadzone:
                speed_val = 0.0
            if abs(steering_val) < self.deadzone:
                steering_val = 0.0

            # Scale the inputs.
            speed = self.scale_speed * speed_val
            steering_angle = self.scale_steering * steering_val

            # Create and populate the Ackermann command message.
            ackermann_cmd = AckermannDriveStamped()
            ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
            ackermann_cmd.header.frame_id = "base_link"  # Change as needed
            ackermann_cmd.drive.speed = speed
            ackermann_cmd.drive.steering_angle = steering_angle

            # Publish the Ackermann command.
            self.drive_pub.publish(ackermann_cmd)
            self.get_logger().info(f"Published Ackermann: speed={speed:.2f}, steering_angle={steering_angle:.2f}")
        else:
            self.get_logger().warning("Received Joy message with insufficient axes.")

def main(args=None):
    rclpy.init(args=args)
    node = JoystickAckermannTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

