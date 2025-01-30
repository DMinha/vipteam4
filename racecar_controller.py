import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math

class RacecarController(Node):
    def __init__(self):
        super().__init__('racecar_controller')
        
        # Declare parameters with defaults
        self.declare_parameter('max_speed', 0.2)  # m/s
        self.declare_parameter('slow_down_distance', 0.6096)  # 2ft in meters
        self.declare_parameter('stop_distance', 0.3048)  # 1ft in meters
        self.declare_parameter('max_steering_angle', 0.4)  # radians
        
        # Get parameters
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.slow_down_distance = self.get_parameter('slow_down_distance').get_parameter_value().double_value
        self.stop_distance = self.get_parameter('stop_distance').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        
        # Subscriber and Publisher
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        
    def scan_callback(self, scan_msg):
        min_distance = float('inf')
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        
        # Check scans within ±30 degrees (front of the car)
        for i in range(len(scan_msg.ranges)):
            angle = angle_min + i * angle_increment
            if -math.pi/6 <= angle <= math.pi/6:  # -30° to +30°
                distance = scan_msg.ranges[i]
                # Validate distance reading
                if scan_msg.range_min <= distance <= scan_msg.range_max:
                    if distance < min_distance:
                        min_distance = distance
        
        # Control logic
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        
        if math.isinf(min_distance):
            drive_msg.drive.speed = self.max_speed  # Move at full speed
        elif min_distance <= self.stop_distance:
            drive_msg.drive.speed = 0.0  # Stop
        elif min_distance <= self.slow_down_distance:
            # Proportional speed reduction between 2ft and 1ft
            drive_msg.drive.speed = self.max_speed * ((min_distance - self.stop_distance) 
                                                      / (self.slow_down_distance - self.stop_distance))
        else:
            drive_msg.drive.speed = self.max_speed  # Full speed
            
        drive_msg.drive.steering_angle = 0.0  # No steering for now

        self.cmd_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RacecarController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

