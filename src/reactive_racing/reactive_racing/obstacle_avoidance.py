#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import numpy as np

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        
        self.laser_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )
        
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, 
            '/ackermann_cmd',
            10
        )
        
        self.safety_threshold = 1.0 
        self.default_speed = 1.0
        self.get_logger().info('Obstacle Avoidance Node initialized')

    def lidar_callback(self, msg: LaserScan):
        front_angle = 15
        front_indices = len(msg.ranges) // 2 - front_angle, len(msg.ranges) // 2 + front_angle
        front_ranges = np.array(msg.ranges[front_indices[0]:front_indices[1]])
        
        front_ranges = front_ranges[np.isfinite(front_ranges)]
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        
        if len(front_ranges) > 0:
            min_distance = np.min(front_ranges)
        
            if min_distance < self.safety_threshold:
                drive_msg.drive.speed = 0.0
                drive_msg.drive.steering_angle = 0.0
                self.get_logger().warn(f'Object detected at {min_distance:.2f}m! Stopping!')
            else:
                drive_msg.drive.speed = self.default_speed
                drive_msg.drive.steering_angle = 0.0
                self.get_logger().info(f'Path clear. Minimum distance: {min_distance:.2f}m')
        else:
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.get_logger().error('No valid LiDAR measurements! Stopping!')
        
        self.drive_publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoider = ObstacleAvoidance()
    
    try:
        rclpy.spin(obstacle_avoider)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_avoider.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()