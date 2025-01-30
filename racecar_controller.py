import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class RacecarController(Node):
    def __init__(self):
        super().__init__('racecar_controller')
        
        # Declare parameters with defaults
        self.declare_parameter('max_speed', 0.5)  # m/s
        self.declare_parameter('slow_down_distance', 0.6096)  # 2ft in meters
        self.declare_parameter('stop_distance', 0.3048)  # 1ft in meters
        
        # Get parameters
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.slow_down_distance = self.get_parameter('slow_down_distance').get_parameter_value().double_value
        self.stop_distance = self.get_parameter('stop_distance').get_parameter_value().double_value
        
        # Subscriber and Publisher
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
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
        cmd_vel = Twist()
        if math.isinf(min_distance):
            self.get_logger().warn("No valid obstacle detected. Stopping.")
            cmd_vel.linear.x = 0.0
        elif min_distance <= self.stop_distance:
            cmd_vel.linear.x = 0.0  # Stop
        elif min_distance <= self.slow_down_distance:
            # Proportional speed reduction between 2ft and 1ft
            cmd_vel.linear.x = self.max_speed * ((min_distance - self.stop_distance) 
                                                / (self.slow_down_distance - self.stop_distance))
        else:
            cmd_vel.linear.x = self.max_speed  # Full speed
            
        cmd_vel.angular.z = 0.0  # No steering
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = RacecarController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()