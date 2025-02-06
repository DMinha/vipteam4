import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import math

class LidarProcessing(Node):
    def __init__(self):
        super().__init__('lidar_processing_node')

        # Constants
        self.MAX_SPEED = 0.5  # Maximum speed in m/s
        self.TURN_SPEED = 0.3  # Speed while turning
        self.STOP_SPEED = 0.0
        self.MAX_STEERING_ANGLE = 0.4  # About 23 degrees in radians

        # Increase obstacle detection threshold
        self.OBSTACLE_THRESHOLD = 1  # Detect obstacles earlier

        # Subscriber to '/scan' topic for LiDAR data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )

        # Publisher for Ackermann drive commands
        self.publisher = self.create_publisher(
            AckermannDriveStamped,
            '/ackermann_cmd',
            10
        )

        # Expand sector angles for wider detection
        self.sector_angles = {
            'left': (-1.2, -0.5),   # -69° to -30°
            'front': (-0.6, 0.6),   # -34° to 34° (wider front detection)
            'right': (0.5, 1.2)     # 30° to 69°
        }

    def lidar_callback(self, msg):
        self.get_logger().info('Processing LaserScan message')

        # Extract LiDAR data
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = msg.ranges

        # Calculate minimum distance in each sector
        sector_distances = {}
        for sector, (start_angle, end_angle) in self.sector_angles.items():
            start_idx = int((start_angle - angle_min) / angle_increment)
            end_idx = int((end_angle - angle_min) / angle_increment)
            sector_ranges = ranges[start_idx:end_idx]
            sector_distances[sector] = min(sector_ranges) if sector_ranges else float('inf')

        # Decision-making based on LiDAR readings
        front_distance = sector_distances['front']
        left_distance = sector_distances['left']
        right_distance = sector_distances['right']

        # Create Ackermann message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()

        # Adjust steering based on obstacle distance
        if front_distance < self.OBSTACLE_THRESHOLD:
            if left_distance > right_distance:
                self.get_logger().info("Obstacle detected: Early turn left!")
                drive_msg.drive.speed = self.TURN_SPEED
                drive_msg.drive.steering_angle = -self.MAX_STEERING_ANGLE * (0.5 + (0.5 * (self.OBSTACLE_THRESHOLD - front_distance) / self.OBSTACLE_THRESHOLD))  # Gradual turn
            else:
                self.get_logger().info("Obstacle detected: Early turn right!")
                drive_msg.drive.speed = self.TURN_SPEED
                drive_msg.drive.steering_angle = self.MAX_STEERING_ANGLE * (0.5 + (0.5 * (self.OBSTACLE_THRESHOLD - front_distance) / self.OBSTACLE_THRESHOLD))  # Gradual turn
        else:
            self.get_logger().info("No obstacle: Continuing forward")
            drive_msg.drive.speed = self.MAX_SPEED
            drive_msg.drive.steering_angle = 0.0

        # Publish the Ackermann command
        self.publisher.publish(drive_msg)
        self.get_logger().info(f"Published command - Speed: {drive_msg.drive.speed:.2f} m/s, Steering: {drive_msg.drive.steering_angle:.2f} rad")


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
