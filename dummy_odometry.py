#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist

class DummyOdometry(Node):
    def __init__(self):
        super().__init__('dummy_odometry')
        self.publisher = self.create_publisher(Odometry, '/odom', 10)
        self.timer = self.create_timer(0.1, self.publish_odometry)

    def publish_odometry(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose = Pose()  # Set position and orientation here
        msg.twist.twist = Twist()  # Set velocity here
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyOdometry()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
