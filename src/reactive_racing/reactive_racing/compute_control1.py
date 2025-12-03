'''
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


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path
import numpy as np
# import pickle # No longer needed
import math
import sys
# from nav2_msgs.srv import ClearEntireCostmap # No longer needed
import os

# Define constants at the top
LOOKAHEAD_DISTANCE = 1.2
MAX_SPEED = 0.7
MIN_SPEED = 0.5

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # --- Parameters ---
        self.declare_parameter('track_file_path', '/home/nvidia/Desktop/team3-vip-f24/build/reactive_racing/reactive_racing/optimized_raceline.csv') # <-- Uses CSV
        self.declare_parameter('wheelbase', 0.33)
        # Path Visualization Parameters
        self.declare_parameter('visualize_path', True)
        self.declare_parameter('visualize_path_topic', '/loaded_raceline_path')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('visualization_publish_period', 5.0)

        # Get Parameter Values
        track_file = self.get_parameter('track_file_path').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.visualize_path = self.get_parameter('visualize_path').get_parameter_value().bool_value
        self.visualize_path_topic = self.get_parameter('visualize_path_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.visualization_publish_period = self.get_parameter('visualization_publish_period').get_parameter_value().double_value

        # --- Initialize variables before loading ---
        self.raceline = None
        self.curvatures = None
        self.path_viz_message = None

        # --- Load Processed Raceline Data FROM CSV ---
        try:
            self.get_logger().info(f"Attempting to load raceline from CSV: {track_file}")
            if not os.path.exists(track_file):
                 raise FileNotFoundError(f"Track CSV file does not exist at {track_file}")
            # Use numpy.loadtxt to load the CSV data
            # Assumes CSV format: s,x,y,heading with a header row
            loaded_data = np.loadtxt(track_file, delimiter=',', skiprows=1)
            self.get_logger().info(f"CSV data loaded successfully. Shape: {loaded_data.shape}")

            # Extract the necessary columns
            if loaded_data.ndim != 2 or loaded_data.shape[1] < 3: # Check for at least x, y columns (index 1, 2)
                 raise ValueError(f"Loaded CSV data has unexpected shape or columns: {loaded_data.shape}. Expected at least (N, 3).")
            self.raceline = loaded_data[:, 1:3] # Extract columns 1 (x) and 2 (y)
            self.get_logger().info(f"Successfully extracted {len(self.raceline)} waypoints (x, y) from CSV.")

            # --- Compute Curvatures ---
            self.get_logger().info("Computing curvatures from loaded raceline...")
            self.curvatures = self.compute_curvatures(self.raceline)
            self.get_logger().info("Curvatures computed.")

        except FileNotFoundError as e:
            self.get_logger().error(f"{e}")
            sys.exit(1) # Exit if file not found
        except ValueError as e:
            self.get_logger().error(f"Error processing CSV file '{track_file}': {e}")
            sys.exit(1)
        except Exception as e: # Catch any other unexpected errors
             self.get_logger().error(f"An unexpected error occurred during CSV loading: {e}", exc_info=True)
             sys.exit(1)

        # --- ROS Comms ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # --- Path Visualization Setup ---
        self.path_viz_pub = None
        self.viz_timer = None
        if self.visualize_path:
            self.get_logger().info(f"Setting up path visualization on '{self.visualize_path_topic}'")
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.path_viz_pub = self.create_publisher(Path, self.visualize_path_topic, qos_profile)
            self._generate_path_viz_message() # Generate the message
            if self.path_viz_message:
                 self.get_logger().info("Publishing initial visualization path...")
                 self.path_viz_pub.publish(self.path_viz_message) # Publish once
                 if self.visualization_publish_period > 0:
                     self.viz_timer = self.create_timer( # Publish periodically
                         self.visualization_publish_period,
                         self._timer_publish_path_callback
                     )
                     self.get_logger().info(f"Path visualization timer started (publishes every {self.visualization_publish_period}s).")
            else:
                 self.get_logger().error("Failed to generate path message for visualization.")

        self.get_logger().info('Pure Pursuit Node initialized (Lap Reset Removed).')
        self.get_logger().info(f'Using wheelbase: {self.wheelbase}m')

    # --- Method to generate the Path message once ---
    def _generate_path_viz_message(self):
        """Generates the nav_msgs/Path message from self.raceline and stores it."""
        if self.raceline is None or len(self.raceline) == 0:
            self.get_logger().warn("Cannot generate visualization path: No raceline data loaded.")
            self.path_viz_message = None; return
        path_msg = Path(); path_msg.header.stamp = self.get_clock().now().to_msg(); path_msg.header.frame_id = self.map_frame
        for wp in self.raceline:
            pose = PoseStamped(); pose.header = path_msg.header
            try: pose.pose.position.x = float(wp[0]); pose.pose.position.y = float(wp[1])
            except (IndexError, TypeError): self.get_logger().error(f"Invalid waypoint: {wp}. Skip.", once=True); continue
            pose.pose.position.z = 0.0; pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_viz_message = path_msg
        self.get_logger().info(f"Generated path message with {len(self.path_viz_message.poses)} poses for visualization.")

    # --- Timer callback to simply publish the stored message ---
    def _timer_publish_path_callback(self):
        """Publishes the pre-generated Path message."""
        if self.path_viz_pub is not None and self.path_viz_message is not None:
            self.path_viz_message.header.stamp = self.get_clock().now().to_msg()
            self.path_viz_pub.publish(self.path_viz_message)

    # --- compute_curvatures (no changes needed) ---
    def compute_curvatures(self, path):
        # ... (implementation remains the same) ...
        N = len(path); curvatures = np.zeros(N)
        if N < 3: return curvatures
        for i in range(1, N - 1):
            p1, p2, p3 = path[i - 1], path[i], path[i + 1]
            v12=p2-p1; v23=p3-p2; v31=p1-p3
            a=np.linalg.norm(v23); b=np.linalg.norm(v31); c=np.linalg.norm(v12)
            area=np.abs(np.cross(v12, -v31)); den=a*b*c
            k = 0.0 if den < 1e-9 or area < 1e-9 else (2.0 * area) / den
            curvatures[i] = k
        return curvatures

    # --- pose_callback function ---
    def pose_callback(self, msg: PoseWithCovarianceStamped):
        # Extract position and orientation
        position=msg.pose.pose.position; orientation=msg.pose.pose.orientation
        x=position.x; y=position.y; theta=self.get_yaw_from_quaternion(orientation)

        # --- Lap Detection Logic REMOVED ---

        # --- Pure Pursuit Logic ---
        lookahead_point, curvature=self.find_lookahead_point(np.array([x, y]))
        if lookahead_point is None:
            # Stop car if no target found
            drive_msg=AckermannDriveStamped()
            drive_msg.header.stamp=self.get_clock().now().to_msg() # Use current time for stop command
            drive_msg.drive.steering_angle=0.0
            drive_msg.drive.speed=0.0
            self.drive_pub.publish(drive_msg)
            return # Exit callback

        # Transform lookahead point and calculate steering/speed
        dx=lookahead_point[0]-x; dy=lookahead_point[1]-y
        lx=math.cos(-theta)*dx-math.sin(-theta)*dy; ly=math.sin(-theta)*dx+math.cos(-theta)*dy
        dist_sq=lx**2+ly**2
        if dist_sq<1e-3: sa=0.0 # Check if target is too close
        else:
            # Calculate steering based on curvature to lookahead point
            steering_curvature = (2.0*ly) / dist_sq
            sa=math.atan(self.wheelbase*steering_curvature)

        # Clamp steering angle
        max_steer=0.4 # Max realistic steering angle in radians
        sa=np.clip(sa, -max_steer, max_steer)

        # Calculate speed based on path curvature AT the lookahead point
        # Adjust the curvature influence factor (e.g., 0.5) to fine-tune speed reduction
        speed=max(MIN_SPEED, MAX_SPEED*(1.0-min(abs(curvature)*0.5, 1.0)))

        # Publish drive command
        drive_msg=AckermannDriveStamped()
        drive_msg.header.stamp=msg.header.stamp # Use pose timestamp for better sync
        drive_msg.header.frame_id='base_link' # Command is usually relative to base
        drive_msg.drive.steering_angle=sa
        drive_msg.drive.speed=speed
        self.drive_pub.publish(drive_msg)

    # --- Reset Method REMOVED ---
    # def reset_localization(self): ...

    # --- Costmap Clearing Method REMOVED ---
    # def clear_costmaps(self): ...

    # --- find_lookahead_point (no changes needed) ---
    def find_lookahead_point(self, current_pos):
        # ... (implementation remains the same) ...
        if self.raceline is None or len(self.raceline)==0: return None, 0.0
        deltas=self.raceline-current_pos; dist_sq=np.einsum('ij,ij->i',deltas,deltas); closest_idx=np.argmin(dist_sq)
        li=-1; n=len(self.raceline)
        for i in range(n):
            ci=(closest_idx+i)%n; p=self.raceline[ci]
            dist=np.hypot(p[0]-current_pos[0],p[1]-current_pos[1])
            if dist>=LOOKAHEAD_DISTANCE: li=ci; break
        if li!=-1:
            fp=self.raceline[li]; fc=self.curvatures[li] if li<len(self.curvatures) else 0.0
            return fp, fc
        else:
            # Log warning only once to prevent spam if continuously failing
            self.get_logger().warn(f"No lookahead point found >= {LOOKAHEAD_DISTANCE}m.", once=True)
            return None, 0.0

    # --- get_yaw_from_quaternion (no changes needed) ---
    def get_yaw_from_quaternion(self, q):
        # ... (implementation remains the same) ...
        sycp=2.0*(q.w*q.z+q.x*q.y); cycp=1.0-2.0*(q.y*q.y+q.z*q.z)
        return math.atan2(sycp, cycp)

# --- main function (no changes needed) ---
def main(args=None):
    rclpy.init(args=args); node=None
    try: node=PurePursuitNode(); rclpy.spin(node)
    except KeyboardInterrupt: pass
    except SystemExit: pass
    finally:
        # Ensure drive command is zero on exit
        if node and hasattr(node, 'drive_pub'):
            stop_msg = AckermannDriveStamped()
            stop_msg.header.stamp = node.get_clock().now().to_msg()
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            node.drive_pub.publish(stop_msg)
            node.get_logger().info("Sent zero drive command on shutdown.")
        # Destroy node and shutdown ROS
        if node: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np
import math
from scipy import signal
import os
import sys

# Constants
LOOKAHEAD_DISTANCE = 1.2
MAX_SPEED = 0.7
MIN_SPEED = 0.3
AVOIDANCE_SPEED = 0.5

class IntegratedStanleyPurePursuit(Node):
    def __init__(self):
        super().__init__('integrated_stanley_pursuit')

        # --- Parameters ---
        self.declare_parameter('track_file_path', '/home/nvidia/Desktop/team3-vip-f24/build/reactive_racing/reactive_racing/optimized_raceline.csv')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('visualize_path', True)
        self.declare_parameter('visualize_path_topic', '/loaded_raceline_path')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('visualization_publish_period', 5.0)
        
        # Stanley Avoidance Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('grid_width_meters', 4.0)
        self.declare_parameter('cells_per_meter', 20)
        self.declare_parameter('lookahead_distance_obstacle', LOOKAHEAD_DISTANCE)
        self.declare_parameter('K_p_obstacle', 0.8)  # Proportional gain for obstacle avoidance
        self.declare_parameter('steering_limit', 0.4)  # radians
        self.declare_parameter('avoidance_target_topic', '/avoidance_target_marker')

        # Get Parameter Values
        track_file = self.get_parameter('track_file_path').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.visualize_path = self.get_parameter('visualize_path').get_parameter_value().bool_value
        self.visualize_path_topic = self.get_parameter('visualize_path_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.visualization_publish_period = self.get_parameter('visualization_publish_period').get_parameter_value().double_value

        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.grid_width_meters = self.get_parameter('grid_width_meters').get_parameter_value().double_value
        self.CELLS_PER_METER = self.get_parameter('cells_per_meter').get_parameter_value().integer_value
        self.L = self.get_parameter('lookahead_distance_obstacle').get_parameter_value().double_value
        self.K_p_obstacle = self.get_parameter('K_p_obstacle').get_parameter_value().double_value
        self.steering_limit = self.get_parameter('steering_limit').get_parameter_value().double_value
        self.avoidance_target_topic = self.get_parameter('avoidance_target_topic').get_parameter_value().string_value

        # --- Initialize variables ---
        self.raceline = None
        self.curvatures = None
        self.path_viz_message = None

        # --- Stanley Avoidance Grid Setup (MATCHING ORIGINAL) ---
        self.grid_height = int(self.L * self.CELLS_PER_METER)
        self.grid_width = int(self.grid_width_meters * self.CELLS_PER_METER)
        self.CELL_Y_OFFSET = (self.grid_width // 2) - 1
        self.occupancy_grid = np.full(shape=(self.grid_height, self.grid_width), fill_value=0, dtype=int)
        self.latest_scan_msg = None
        self.obstacle_detected = False
        
        # Constants from Stanley
        self.IS_OCCUPIED = 100
        self.IS_FREE = 0
        self.MARGIN = int(self.CELLS_PER_METER * 0.15)  # 0.15m margin for car width

        # --- Load Raceline Data ---
        try:
            self.get_logger().info(f"Loading raceline from CSV: {track_file}")
            if not os.path.exists(track_file):
                raise FileNotFoundError(f"Track CSV file does not exist at {track_file}")
            
            loaded_data = np.loadtxt(track_file, delimiter=',', skiprows=1)
            self.get_logger().info(f"CSV data loaded. Shape: {loaded_data.shape}")

            if loaded_data.ndim != 2 or loaded_data.shape[1] < 3:
                raise ValueError(f"Unexpected CSV shape: {loaded_data.shape}")
            
            self.raceline = loaded_data[:, 1:3]  # Extract x, y columns
            self.get_logger().info(f"Extracted {len(self.raceline)} waypoints")

            self.curvatures = self.compute_curvatures(self.raceline)
            self.get_logger().info("Curvatures computed.")

        except Exception as e:
            self.get_logger().error(f"Failed to load raceline: {e}")
            sys.exit(1)

        # --- ROS Comms ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10)
        self.target_viz_pub = self.create_publisher(Marker, self.avoidance_target_topic, 10)

        # --- Path Visualization Setup ---
        self.path_viz_pub = None
        self.viz_timer = None
        if self.visualize_path:
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.path_viz_pub = self.create_publisher(Path, self.visualize_path_topic, qos_profile)
            self._generate_path_viz_message()
            if self.path_viz_message:
                self.path_viz_pub.publish(self.path_viz_message)
                if self.visualization_publish_period > 0:
                    self.viz_timer = self.create_timer(
                        self.visualization_publish_period,
                        self._timer_publish_path_callback
                    )

        self.get_logger().info('Integrated Stanley + Pure Pursuit Node Initialized')

    def scan_callback(self, scan_msg: LaserScan):
        """Store latest scan message"""
        self.latest_scan_msg = scan_msg

    def populate_occupancy_grid(self, ranges, angle_increment):
        """
        Populate occupancy grid using Stanley's coordinate system
        CRITICAL: Uses backward-facing grid (x negative is forward)
        """
        self.occupancy_grid = np.full(
            shape=(self.grid_height, self.grid_width), 
            fill_value=self.IS_FREE, 
            dtype=int
        )

        ranges = np.array(ranges)
        indices = np.arange(len(ranges))
        
        # Stanley uses -45 degree offset
        ANGLE_OFFSET = np.radians(45)
        thetas = (indices * angle_increment) - ANGLE_OFFSET
        
        # Stanley coordinate transform
        xs = ranges * np.sin(thetas)
        ys = ranges * np.cos(thetas) * -1

        # Stanley's grid coordinate transform (BACKWARD FACING)
        i = np.round(xs * -self.CELLS_PER_METER + (self.grid_height - 1)).astype(int)
        j = np.round(ys * -self.CELLS_PER_METER + self.CELL_Y_OFFSET).astype(int)

        # Mark occupied cells
        valid_mask = (i >= 0) & (i < self.grid_height) & (j >= 0) & (j < self.grid_width)
        self.occupancy_grid[i[valid_mask], j[valid_mask]] = self.IS_OCCUPIED

    def convolve_occupancy_grid(self):
        """Apply convolution to expand obstacles (from Stanley)"""
        kernel = np.ones(shape=[2, 2])
        self.occupancy_grid = signal.convolve2d(
            self.occupancy_grid.astype("int"), 
            kernel.astype("int"), 
            boundary="symm", 
            mode="same"
        )
        self.occupancy_grid = np.clip(self.occupancy_grid, 0, 100)

    def local_to_grid(self, x, y):
        """Convert car frame to grid coordinates (Stanley's system)"""
        i = int(x * -self.CELLS_PER_METER + (self.grid_height - 1))
        j = int(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET)
        return (i, j)

    def grid_to_local(self, point):
        """Convert grid coordinates back to car frame"""
        i, j = point[0], point[1]
        x = (i - (self.grid_height - 1)) / -self.CELLS_PER_METER
        y = (j - self.CELL_Y_OFFSET) / -self.CELLS_PER_METER
        return (x, y)

    def check_collision(self, cell_a, cell_b, margin=0):
        """
        Check collision along path using Stanley's method with margin
        Margin accounts for car width (~0.3m)
        """
        for i in range(-margin, margin + 1):
            cell_a_margin = (cell_a[0], cell_a[1] + i)
            cell_b_margin = (cell_b[0], cell_b[1] + i)
            
            for cell in self.traverse_grid(cell_a_margin, cell_b_margin):
                if (cell[0] < 0) or (cell[1] < 0) or \
                   (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                    continue
                try:
                    if self.occupancy_grid[cell] >= self.IS_OCCUPIED:
                        return True
                except IndexError:
                    return True
        return False

    def check_collision_loose(self, cell_a, cell_b, margin=0):
        """
        Looser collision check - only checks second half of path
        Used when obstacle is very close
        """
        for i in range(-margin, margin + 1):
            cell_a_margin = (int((cell_a[0] + cell_b[0]) / 2), 
                           int((cell_a[1] + cell_b[1]) / 2) + i)
            cell_b_margin = (cell_b[0], cell_b[1] + i)
            
            for cell in self.traverse_grid(cell_a_margin, cell_b_margin):
                if (cell[0] < 0) or (cell[1] < 0) or \
                   (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                    continue
                try:
                    if self.occupancy_grid[cell] >= self.IS_OCCUPIED:
                        return True
                except IndexError:
                    return True
        return False

    def traverse_grid(self, start, end):
        """Bresenham's line algorithm for grid traversal"""
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        is_steep = abs(dy) > abs(dx)

        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = y2 - y1
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        return points

    def compute_curvatures(self, path):
        """Compute path curvature at each point"""
        N = len(path)
        curvatures = np.zeros(N)
        if N < 3:
            return curvatures
        
        for i in range(1, N - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            v12 = p2 - p1
            v23 = p3 - p2
            v31 = p1 - p3
            
            a = np.linalg.norm(v23)
            b = np.linalg.norm(v31)
            c = np.linalg.norm(v12)
            area = np.abs(np.cross(v12, -v31))
            den = a * b * c
            
            if den > 1e-9 and area > 1e-9:
                curvatures[i] = (2.0 * area) / den
                
        return curvatures

    def find_lookahead_point(self, current_pos):
        """Find lookahead point on raceline"""
        if self.raceline is None or len(self.raceline) == 0:
            return None, 0.0
        
        deltas = self.raceline - current_pos
        dist_sq = np.einsum('ij,ij->i', deltas, deltas)
        closest_idx = np.argmin(dist_sq)
        
        n = len(self.raceline)
        for i in range(n):
            ci = (closest_idx + i) % n
            p = self.raceline[ci]
            dist = np.hypot(p[0] - current_pos[0], p[1] - current_pos[1])
            
            if dist >= LOOKAHEAD_DISTANCE:
                fc = self.curvatures[ci] if ci < len(self.curvatures) else 0.0
                return p, fc
        
        return None, 0.0

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Main control callback - integrates Pure Pursuit with Stanley Avoidance"""
        if self.latest_scan_msg is None:
            self.get_logger().warn("No LiDAR data yet", throttle_duration_sec=2.0)
            return

        # Extract current pose
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        x, y = position.x, position.y
        theta = self.get_yaw_from_quaternion(orientation)

        # Update occupancy grid using Stanley's method
        self.populate_occ        lx_raceline = math.cos(-theta) * dx - math.sin(-theta) * dy
        ly_raceline = math.sin(-theta) * dx + math.cos(-theta) * dy

        # Check for obstacles using Stanley's grid system
        current_grid_pos = np.array(self.local_to_grid(0, 0))
        goal_grid_pos = np.array(self.local_to_grid(lx_raceline, ly_raceline))

        # Check collision with margin for car width
        self.obstacle_detected = self.check_collision(
            current_grid_pos, 
            goal_grid_pos, 
            margin=self.MARGIN
        )

        target_local = np.array([lx_raceline, ly_raceline])
        K_p = 0.5  # Normal pure pursuit gain
        speed = MAX_SPEED

        # STANLEY AVOIDANCE LOGIC (if obstacle detected)
        if self.obstacle_detected:
            self.get_logger().info("Obstacle detected! Finding alternative path...", 
                                 throttle_duration_sec=1.0)
            
            # Use Stanley's alternating shift pattern: [1, -1, 2, -2, 3, -3, ...]
            shifts = [i * (-1 if i % 2 else 1) for i in range(1, 21)]
            found = False

            # Strategy 1: Try lateral shifts of goal point
            for shift in shifts:
                new_goal = goal_grid_pos + np.array([0, shift])
                
                if not self.check_collision(current_grid_pos, new_goal, 
                                           margin=int(1.5 * self.MARGIN)):
                    target_local = np.array(self.grid_to_local(new_goal))
                    found = True
                    self.get_logger().info(f"Found path with shift={shift}", once=True)
                    break

            # Strategy 2: If still blocked, aim for midpoint with shift
            if not found:
                middle_grid = np.array(
                    current_grid_pos + (goal_grid_pos - current_grid_pos) / 2
                ).astype(int)
                
                for shift in shifts:
                    new_goal = middle_grid + np.array([0, shift])
                    
                    if not self.check_collision(current_grid_pos, new_goal, 
                                               margin=int(1.5 * self.MARGIN)):
                        target_local = np.array(self.grid_to_local(new_goal))
                        found = True
                        self.get_logger().info(f"Found path to midpoint, shift={shift}", 
                                             once=True)
                        break

            # Strategy 3: Use loose collision check (only check second half)
            if not found:
                for shift in shifts:
                    new_goal = middle_grid + np.array([0, shift])
                    
                    if not self.check_collision_loose(current_grid_pos, new_goal, 
                                                     margin=self.MARGIN):
                        target_local = np.array(self.grid_to_local(new_goal))
                        found = True
                        self.get_logger().info(f"Found loose path, shift={shift}", once=True)
                        break

            if not found:
                self.get_logger().error("No clear path found! Stopping.", 
                                       throttle_duration_sec=2.0)
                self.publish_stop()
                return

            # Use obstacle avoidance parameters
            K_p = self.K_p_obstacle
            speed = AVOIDANCE_SPEED

        # Calculate steering angle (Pure Pursuit)
        lx, ly = target_local[0], target_local[1]
        dist_sq = lx**2 + ly**2

        if dist_sq < 1e-3:
            steering_angle = 0.0
        else:
            steering_curvature = (2.0 * ly) / dist_sq
            steering_angle = math.atan(self.wheelbase * steering_curvature)

        # Adjust speed based on curvature
        if not self.obstacle_detected:
            speed = max(MIN_SPEED, MAX_SPEED * (1.0 - min(abs(curvature) * 0.5, 1.0)))

        # Clamp steering
        steering_angle = np.clip(steering_angle, -self.steering_limit, self.steering_limit)

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        # Log status
        self.get_logger().info(
            f"Obstacle: {self.obstacle_detected} | Speed: {speed:.2f} | "
            f"Steer: {np.degrees(steering_angle):.1f}° | K_p: {K_p:.2f}",
            throttle_duration_sec=0.5
        )

        # Visualize target
        self._publish_target_marker(target_local, x, y, theta)

    def publish_stop(self):
        """Publish zero drive command"""
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)

    def _publish_target_marker(self, target_local, car_x, car_y, car_theta):
        """Publish visualization marker for current target"""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'avoidance_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Transform to map frame
        lx, ly = target_local[0], target_local[1]
        map_x = car_x + (lx * math.cos(car_theta) - ly * math.sin(car_theta))
        map_y = car_y + (lx * math.sin(car_theta) + ly * math.cos(car_theta))

        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 0.9
        if self.obstacle_detected:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0  # Orange
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0  # Green

        self.target_viz_pub.publish(marker)

    def _generate_path_viz_message(self):
        """Generate path visualization message"""
        if self.raceline is None or len(self.raceline) == 0:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        
        for wp in self.raceline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_viz_message = path_msg

    def _timer_publish_path_callback(self):
        """Periodically publish path visualization"""
        if self.path_viz_pub and self.path_viz_message:
            self.path_viz_message.header.stamp = self.get_clock().now().to_msg()
            self.path_viz_pub.publish(self.path_viz_message)

    def get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IntegratedStanleyPurePursuit()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()upancy_grid(
            self.latest_scan_msg.ranges,
            self.latest_scan_msg.angle_increment
        )
        self.convolve_occupancy_grid()

        # Get raceline lookahead point
        raceline_point, curvature = self.find_lookahead_point(np.array([x, y]))
        
        if raceline_point is None:
            self.get_logger().warn("No lookahead point found!", throttle_duration_sec=2.0)
            self.publish_stop()
            return

        # Transform to car frame
        dx = raceline_point[0] - x
        dy = raceline_point[1] - y
        lx_raceline = math.cos(-theta) * dx - math.sin(-theta) * dy
        ly_raceline = math.sin(-theta) * dx + math.cos(-theta) * dy

        # Check for obstacles using Stanley's grid system
        current_grid_pos = np.array(self.local_to_grid(0, 0))
        goal_grid_pos = np.array(self.local_to_grid(lx_raceline, ly_raceline))

        # Check collision with margin for car width
        self.obstacle_detected = self.check_collision(
            current_grid_pos, 
            goal_grid_pos, 
            margin=self.MARGIN
        )

        target_local = np.array([lx_raceline, ly_raceline])
        K_p = 0.5  # Normal pure pursuit gain
        speed = MAX_SPEED

        # STANLEY AVOIDANCE LOGIC (if obstacle detected)
        if self.obstacle_detected:
            self.get_logger().info("Obstacle detected! Finding alternative path...", 
                                 throttle_duration_sec=1.0)
            
            # Use Stanley's alternating shift pattern: [1, -1, 2, -2, 3, -3, ...]
            shifts = [i * (-1 if i % 2 else 1) for i in range(1, 21)]
            found = False

            # Strategy 1: Try lateral shifts of goal point
            for shift in shifts:
                new_goal = goal_grid_pos + np.array([0, shift])
                
                if not self.check_collision(current_grid_pos, new_goal, 
                                           margin=int(1.5 * self.MARGIN)):
                    target_local = np.array(self.grid_to_local(new_goal))
                    found = True
                    self.get_logger().info(f"Found path with shift={shift}", once=True)
                    break

            # Strategy 2: If still blocked, aim for midpoint with shift
            if not found:
                middle_grid = np.array(
                    current_grid_pos + (goal_grid_pos - current_grid_pos) / 2
                ).astype(int)
                
                for shift in shifts:
                    new_goal = middle_grid + np.array([0, shift])
                    
                    if not self.check_collision(current_grid_pos, new_goal, 
                                               margin=int(1.5 * self.MARGIN)):
                        target_local = np.array(self.grid_to_local(new_goal))
                        found = True
                        self.get_logger().info(f"Found path to midpoint, shift={shift}", 
                                             once=True)
                        break

            # Strategy 3: Use loose collision check (only check second half)
            if not found:
                for shift in shifts:
                    new_goal = middle_grid + np.array([0, shift])
                    
                    if not self.check_collision_loose(current_grid_pos, new_goal, 
                                                     margin=self.MARGIN):
                        target_local = np.array(self.grid_to_local(new_goal))
                        found = True
                        self.get_logger().info(f"Found loose path, shift={shift}", once=True)
                        break

            if not found:
                self.get_logger().error("No clear path found! Stopping.", 
                                       throttle_duration_sec=2.0)
                self.publish_stop()
                return

            # Use obstacle avoidance parameters
            K_p = self.K_p_obstacle
            speed = AVOIDANCE_SPEED

        # Calculate steering angle (Pure Pursuit)
        lx, ly = target_local[0], target_local[1]
        dist_sq = lx**2 + ly**2

        if dist_sq < 1e-3:
            steering_angle = 0.0
        else:
            steering_curvature = (2.0 * ly) / dist_sq
            steering_angle = math.atan(self.wheelbase * steering_curvature)

        # Adjust speed based on curvature
        if not self.obstacle_detected:
            speed = max(MIN_SPEED, MAX_SPEED * (1.0 - min(abs(curvature) * 0.5, 1.0)))

        # Clamp steering
        steering_angle = np.clip(steering_angle, -self.steering_limit, self.steering_limit)

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        # Log status
        self.get_logger().info(
            f"Obstacle: {self.obstacle_detected} | Speed: {speed:.2f} | "
            f"Steer: {np.degrees(steering_angle):.1f}° | K_p: {K_p:.2f}",
            throttle_duration_sec=0.5
        )

        # Visualize target
        self._publish_target_marker(target_local, x, y, theta)

    def publish_stop(self):
        """Publish zero drive command"""
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)

    def _publish_target_marker(self, target_local, car_x, car_y, car_theta):
        """Publish visualization marker for current target"""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'avoidance_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Transform to map frame
        lx, ly = target_local[0], target_local[1]
        map_x = car_x + (lx * math.cos(car_theta) - ly * math.sin(car_theta))
        map_y = car_y + (lx * math.sin(car_theta) + ly * math.cos(car_theta))

        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 0.9
        if self.obstacle_detected:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0  # Orange
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0  # Green

        self.target_viz_pub.publish(marker)

    def _generate_path_viz_message(self):
        """Generate path visualization message"""
        if self.raceline is None or len(self.raceline) == 0:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        
        for wp in self.raceline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_viz_message = path_msg

    def _timer_publish_path_callback(self):
        """Periodically publish path visualization"""
        if self.path_viz_pub and self.path_viz_message:
            self.path_viz_message.header.stamp = self.get_clock().now().to_msg()
            self.path_viz_pub.publish(self.path_viz_message)

    def get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IntegratedStanleyPurePursuit()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
    '''
'''
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
import numpy as np
import math
import os
import sys

# Pure Pursuit Constants
BASE_LOOKAHEAD = 1.2
MIN_LOOKAHEAD = 0.8
MAX_LOOKAHEAD = 2.5
MAX_SPEED = 2.0
MIN_SPEED = 0.5
LOOKAHEAD_SPEED_GAIN = 0.3

# Avoidance Constants
GRID_RESOLUTION = 0.05  # 5cm per cell
GRID_SIZE = 200  # 10m x 10m grid
OBSTACLE_THRESHOLD = 0.5  # Consider cell occupied if > 50%
COLLISION_CHECK_POINTS = 15  # Reduced from 20
CAR_WIDTH = 0.35  # Reduced from 0.4 to be less conservative
SAFETY_MARGIN = 0.1  # Reduced from 0.2 to be less conservative
LATERAL_SEARCH_RANGE = 1.5
LATERAL_SEARCH_STEP = 0.15
AVOIDANCE_SPEED_FACTOR = 0.75

# Minimum scans before enabling avoidance
MIN_SCANS_FOR_AVOIDANCE = 10
# Minimum distance for obstacle to trigger avoidance (meters)
MIN_OBSTACLE_DISTANCE = 0.8  # Don't react to very close obstacles (likely walls)

class DynamicPurePursuitAvoidance(Node):
    def __init__(self):
        super().__init__('dynamic_pure_pursuit_avoidance')

        # --- Parameters ---
        self.declare_parameter('track_file_path', '/home/nvidia/Desktop/team3-vip-f24/build/reactive_racing/reactive_racing/optimized_raceline.csv')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('visualize_path', True)
        self.declare_parameter('visualize_path_topic', '/loaded_raceline_path')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('visualization_publish_period', 5.0)
        self.declare_parameter('target_marker_topic', '/pursuit_target_marker')
        self.declare_parameter('base_lookahead', BASE_LOOKAHEAD)
        self.declare_parameter('min_lookahead', MIN_LOOKAHEAD)
        self.declare_parameter('max_lookahead', MAX_LOOKAHEAD)
        self.declare_parameter('max_speed', MAX_SPEED)
        self.declare_parameter('min_speed', MIN_SPEED)
        self.declare_parameter('curvature_speed_factor', 2.0)
        self.declare_parameter('enable_avoidance', True)  # Can disable avoidance completely

        # Get Parameters
        track_file = self.get_parameter('track_file_path').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.visualize_path = self.get_parameter('visualize_path').get_parameter_value().bool_value
        self.visualize_path_topic = self.get_parameter('visualize_path_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.visualization_publish_period = self.get_parameter('visualization_publish_period').get_parameter_value().double_value
        self.target_marker_topic = self.get_parameter('target_marker_topic').get_parameter_value().string_value
        self.base_lookahead = self.get_parameter('base_lookahead').get_parameter_value().double_value
        self.min_lookahead = self.get_parameter('min_lookahead').get_parameter_value().double_value
        self.max_lookahead = self.get_parameter('max_lookahead').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.curvature_speed_factor = self.get_parameter('curvature_speed_factor').get_parameter_value().double_value
        self.enable_avoidance = self.get_parameter('enable_avoidance').get_parameter_value().bool_value

        # --- Initialize State ---
        self.raceline = None
        self.curvatures = None
        self.path_viz_message = None
        self.current_speed = self.min_speed
        
        # AVOIDANCE STATE
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.latest_scan = None
        self.scan_count = 0  # Count scans received
        self.current_pose = None
        self.avoiding = False
        self.avoidance_offset = 0.0
        self.target_offset = 0.0
        self.avoidance_ready = False  # Only enable after receiving enough scans

        # --- Load Raceline ---
        try:
            self.get_logger().info(f"Loading raceline from: {track_file}")
            if not os.path.exists(track_file):
                raise FileNotFoundError(f"Track file not found: {track_file}")
            
            loaded_data = np.loadtxt(track_file, delimiter=',', skiprows=1)
            
            if loaded_data.ndim != 2 or loaded_data.shape[1] < 3:
                raise ValueError(f"Invalid CSV shape: {loaded_data.shape}")
            
            self.raceline = loaded_data[:, 1:3]
            self.get_logger().info(f"Loaded {len(self.raceline)} waypoints")

            self.curvatures = self.compute_curvatures(self.raceline)

        except Exception as e:
            self.get_logger().error(f"Failed to load raceline: {e}")
            sys.exit(1)

        # --- ROS Communications ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Subscribe to odometry (preferred for simulation)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # LiDAR for obstacle detection
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Visualization
        self.target_viz_pub = self.create_publisher(Marker, self.target_marker_topic, 10)
        self.avoidance_viz_pub = self.create_publisher(Marker, '/avoidance_marker', 10)

        # Path visualization
        if self.visualize_path:
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.path_viz_pub = self.create_publisher(Path, self.visualize_path_topic, qos_profile)
            self._generate_path_viz_message()
            if self.path_viz_message:
                self.path_viz_pub.publish(self.path_viz_message)
                if self.visualization_publish_period > 0:
                    self.viz_timer = self.create_timer(
                        self.visualization_publish_period,
                        self._timer_publish_path_callback
                    )

        self.get_logger().info('=' * 60)
        self.get_logger().info('Dynamic Pure Pursuit with Avoidance Initialized')
        if self.enable_avoidance:
            self.get_logger().info(f'Avoidance: ENABLED (activates after {MIN_SCANS_FOR_AVOIDANCE} scans)')
        else:
            self.get_logger().info('Avoidance: DISABLED (pure pursuit only)')
        self.get_logger().info('=' * 60)

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan - ONLY FORWARD-FACING BEAMS"""
        self.latest_scan = msg
        self.scan_count += 1
        
        # Enable avoidance after receiving enough scans
        if not self.avoidance_ready and self.scan_count >= MIN_SCANS_FOR_AVOIDANCE:
            self.avoidance_ready = True
            self.get_logger().info('✓ Avoidance system ready (sufficient scan data)')
        
        if self.current_pose is None:
            return
        
        # Clear grid
        self.occupancy_grid.fill(0.0)
        
        x, y, theta = self.current_pose
        
        # ONLY PROCESS FORWARD-FACING BEAMS (±75 degrees from front)
        # This prevents side walls from being detected as obstacles
        forward_angle_range = math.radians(75)  # 75 degrees on each side
        
        angle = msg.angle_min
        for i, r in enumerate(msg.ranges):
            # Skip if beam is not forward-facing
            if abs(angle) > forward_angle_range:
                angle += msg.angle_increment
                continue
            
            # Only process returns within reasonable range (ignore very far walls)
            if msg.range_min < r < min(msg.range_max, 5.0):  # Max 5m range
                # Obstacle position in map frame
                obs_x = x + r * math.cos(theta + angle)
                obs_y = y + r * math.sin(theta + angle)
                
                # Convert to grid (car-centered)
                grid_x = int((obs_x - x) / GRID_RESOLUTION) + GRID_SIZE // 2
                grid_y = int((obs_y - y) / GRID_RESOLUTION) + GRID_SIZE // 2
                
                # Mark cell and neighbors (obstacle blob)
                for dx in range(-2, 3):  # Slightly larger for safety
                    for dy in range(-2, 3):
                        gx = grid_x + dx
                        gy = grid_y + dy
                        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                            self.occupancy_grid[gx, gy] = 1.0
            
            angle += msg.angle_increment

    def odom_callback(self, msg: Odometry):
        """Handle odometry and run control loop"""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        x, y = position.x, position.y
        theta = self.get_yaw_from_quaternion(orientation)
        self.current_pose = (x, y, theta)
        
        # Run control
        self.control_loop(msg.header.stamp)

    def control_loop(self, timestamp):
        """Main control loop"""
        
        if self.current_pose is None or self.raceline is None:
            return
        
        x, y, theta = self.current_pose
        
        # Dynamic lookahead
        lookahead_distance = self.compute_dynamic_lookahead(self.current_speed)
        
        # Find target on raceline
        lookahead_point, curvature = self.find_lookahead_point(
            np.array([x, y]), 
            lookahead_distance
        )
        
        if lookahead_point is None:
            self.get_logger().warn("No lookahead point!", throttle_duration_sec=2.0)
            self.publish_stop()
            return
        
        # AVOIDANCE LOGIC (only if enabled and ready)
        collision_detected = False
        if self.enable_avoidance and self.avoidance_ready:
            collision_detected = self.check_path_collision(
                np.array([x, y]), 
                lookahead_point,
                self.avoidance_offset
            )
            
            # Debug: Log occupancy grid status occasionally
            if self.scan_count % 100 == 0:
                occupied_cells = np.sum(self.occupancy_grid > OBSTACLE_THRESHOLD)
                self.get_logger().info(
                    f"Grid Status: {occupied_cells} occupied cells | Collision: {collision_detected}",
                    throttle_duration_sec=2.0
                )
        
        target_point = lookahead_point.copy()
        
        if collision_detected:
            # Search for clear path
            clear_offset = self.find_clear_lateral_path(
                np.array([x, y]), 
                lookahead_point
            )
            
            if clear_offset is not None:
                if not self.avoiding:
                    self.get_logger().info(f"OBSTACLE DETECTED! Avoiding with offset: {clear_offset:.2f}m")
                self.avoiding = True
                self.target_offset = clear_offset
            else:
                self.get_logger().warn("No clear path - STOPPING!", throttle_duration_sec=1.0)
                self.publish_stop()
                return
        else:
            # Path clear - return to raceline
            if self.avoiding and abs(self.avoidance_offset) < 0.05:
                self.avoiding = False
                self.get_logger().info("✓ Returned to raceline")
            
            # Smooth return
            self.target_offset *= 0.95
        
        # Smooth offset application
        alpha = 0.3
        self.avoidance_offset = alpha * self.target_offset + (1 - alpha) * self.avoidance_offset
        
        # Apply offset to target
        if abs(self.avoidance_offset) > 0.01:
            target_point = self.apply_lateral_offset(
                np.array([x, y]),
                lookahead_point,
                self.avoidance_offset
            )
        
        # Transform to car frame
        dx = target_point[0] - x
        dy = target_point[1] - y
        lx = math.cos(-theta) * dx - math.sin(-theta) * dy
        ly = math.sin(-theta) * dx + math.cos(-theta) * dy

        # Pure Pursuit steering
        dist_sq = lx**2 + ly**2
        if dist_sq < 1e-3:
            steering_angle = 0.0
        else:
            steering_curvature = (2.0 * ly) / dist_sq
            steering_angle = math.atan(self.wheelbase * steering_curvature)
        
        # Dynamic speed
        target_speed = self.compute_dynamic_speed(curvature)
        
        # Reduce speed during avoidance
        if self.avoiding:
            target_speed *= AVOIDANCE_SPEED_FACTOR
        
        # Smooth speed
        alpha_speed = 0.3
        self.current_speed = alpha_speed * target_speed + (1 - alpha_speed) * self.current_speed

        # Clamp steering
        steering_angle = np.clip(steering_angle, -0.4, 0.4)

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = timestamp
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.current_speed
        self.drive_pub.publish(drive_msg)

        # Log status
        if self.avoiding:
            self.get_logger().info(
                f"AVOIDING | Offset: {self.avoidance_offset:.2f}m | Speed: {self.current_speed:.2f}m/s | Steer: {np.degrees(steering_angle):.1f}°",
                throttle_duration_sec=0.5
            )
        
        # Visualization
        self._publish_target_marker(lx, ly, x, y, theta)
        if self.avoiding:
            self._publish_avoidance_marker(target_point[0], target_point[1])

    def check_path_collision(self, start_pos, end_pos, current_offset):
        """Check if path collides with obstacles - ONLY CHECK PATH AHEAD"""
        if self.latest_scan is None:
            return False
        
        # Check points along the path
        collision_found = False
        for i in range(COLLISION_CHECK_POINTS):
            t = i / (COLLISION_CHECK_POINTS - 1)
            point = start_pos + t * (end_pos - start_pos)
            
            # Distance from car to this point
            dist_from_car = np.hypot(point[0] - start_pos[0], point[1] - start_pos[1])
            
            # Skip points too close to car (avoid false positives from walls beside us)
            if dist_from_car < MIN_OBSTACLE_DISTANCE:
                continue
            
            if abs(current_offset) > 0.01:
                point = self.apply_lateral_offset(start_pos, point, current_offset)
            
            if self.is_point_in_collision(point, start_pos):
                collision_found = True
                break
        
        return collision_found

    def is_point_in_collision(self, point, car_pos):
        """Check if point collides with obstacles"""
        dx = point[0] - car_pos[0]
        dy = point[1] - car_pos[1]
        
        grid_x = int(dx / GRID_RESOLUTION) + GRID_SIZE // 2
        grid_y = int(dy / GRID_RESOLUTION) + GRID_SIZE // 2
        
        check_radius = int((CAR_WIDTH/2 + SAFETY_MARGIN) / GRID_RESOLUTION)
        
        for dx_check in range(-check_radius, check_radius + 1):
            for dy_check in range(-check_radius, check_radius + 1):
                gx = grid_x + dx_check
                gy = grid_y + dy_check
                
                if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                    if self.occupancy_grid[gx, gy] > OBSTACLE_THRESHOLD:
                        return True
        
        return False

    def find_clear_lateral_path(self, start_pos, nominal_end):
        """Search for clear lateral offset"""
        # Try center first
        if not self.check_path_collision(start_pos, nominal_end, 0.0):
            return 0.0
        
        # Search left and right
        search_offsets = []
        offset = LATERAL_SEARCH_STEP
        while offset <= LATERAL_SEARCH_RANGE:
            search_offsets.append(offset)
            search_offsets.append(-offset)
            offset += LATERAL_SEARCH_STEP
        
        for offset in search_offsets:
            if not self.check_path_collision(start_pos, nominal_end, offset):
                return offset
        
        return None

    def apply_lateral_offset(self, car_pos, target_point, offset):
        """Apply lateral offset perpendicular to path direction"""
        dx = target_point[0] - car_pos[0]
        dy = target_point[1] - car_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist < 1e-3:
            return target_point
        
        # Perpendicular direction
        perp_x = -dy / dist
        perp_y = dx / dist
        
        new_point = np.array([
            target_point[0] + offset * perp_x,
            target_point[1] + offset * perp_y
        ])
        
        return new_point

    def compute_curvatures(self, path):
        """Compute path curvature"""
        N = len(path)
        curvatures = np.zeros(N)
        if N < 3:
            return curvatures
        
        for i in range(1, N - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            v12 = p2 - p1
            v23 = p3 - p2
            v31 = p1 - p3
            
            a = np.linalg.norm(v23)
            b = np.linalg.norm(v31)
            c = np.linalg.norm(v12)
            area = np.abs(np.cross(v12, -v31))
            den = a * b * c
            
            if den > 1e-9 and area > 1e-9:
                curvatures[i] = (2.0 * area) / den
        
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures

    def compute_dynamic_lookahead(self, current_speed):
        """Compute lookahead based on speed"""
        speed_factor = (current_speed - self.min_speed) / (self.max_speed - self.min_speed)
        speed_factor = np.clip(speed_factor, 0.0, 1.0)
        
        lookahead = self.base_lookahead + (speed_factor * LOOKAHEAD_SPEED_GAIN * self.base_lookahead)
        return np.clip(lookahead, self.min_lookahead, self.max_lookahead)

    def compute_dynamic_speed(self, curvature):
        """Compute speed based on curvature"""
        speed_reduction = min(abs(curvature) * self.curvature_speed_factor, 1.0)
        speed = self.max_speed * (1.0 - speed_reduction)
        return np.clip(speed, self.min_speed, self.max_speed)

    def find_lookahead_point(self, current_pos, lookahead_distance):
        """Find lookahead point on raceline"""
        if self.raceline is None or len(self.raceline) == 0:
            return None, 0.0
        
        deltas = self.raceline - current_pos
        dist_sq = np.einsum('ij,ij->i', deltas, deltas)
        closest_idx = np.argmin(dist_sq)
        
        n = len(self.raceline)
        for i in range(n):
            ci = (closest_idx + i) % n
            p = self.raceline[ci]
            dist = np.hypot(p[0] - current_pos[0], p[1] - current_pos[1])
            
            if dist >= lookahead_distance:
                fc = self.curvatures[ci] if ci < len(self.curvatures) else 0.0
                return p, fc
        
        furthest_idx = (closest_idx + n//2) % n
        return self.raceline[furthest_idx], self.curvatures[furthest_idx]

    def publish_stop(self):
        """Stop the car"""
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)
        self.current_speed = 0.0

    def _publish_target_marker(self, lx, ly, car_x, car_y, car_theta):
        """Visualize pursuit target"""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pursuit_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        map_x = car_x + (lx * math.cos(car_theta) - ly * math.sin(car_theta))
        map_y = car_y + (lx * math.sin(car_theta) + ly * math.cos(car_theta))

        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 0.9
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.target_viz_pub.publish(marker)

    def _publish_avoidance_marker(self, x, y):
        """Visualize avoidance target"""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'avoidance_target'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.a = 0.9
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0

        self.avoidance_viz_pub.publish(marker)

    def _generate_path_viz_message(self):
        """Generate path visualization"""
        if self.raceline is None or len(self.raceline) == 0:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        
        for wp in self.raceline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_viz_message = path_msg

    def _timer_publish_path_callback(self):
        """Periodically publish path"""
        if self.path_viz_pub and self.path_viz_message:
            self.path_viz_message.header.stamp = self.get_clock().now().to_msg()
            self.path_viz_pub.publish(self.path_viz_message)

    def get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DynamicPurePursuitAvoidance()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Dynamic Pure Pursuit Controller - SMART SIDE SELECTION

NEW FEATURE: Space-aware overtaking side selection
Instead of just going "opposite of opponent", now:
1. Scans LEFT and RIGHT with LiDAR
2. Measures available space on each side
3. Chooses the side with MORE room
4. Avoids overtaking into walls!

Example:
  Wall |  Opponent  You |  Open Space
       |     O      Y   |
       
  OLD: Goes LEFT (opposite opponent) → Crashes into wall!
  NEW: Goes RIGHT (more space) → Success!
"""
'''
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np
import math
import os
import sys

# Pure Pursuit Constants
BASE_LOOKAHEAD = 1.2
MIN_LOOKAHEAD = 0.8
MAX_LOOKAHEAD = 2.5
MAX_SPEED = 0.7
MIN_SPEED = 0.3
LOOKAHEAD_SPEED_GAIN = 0.3

# Avoidance Constants
GRID_RESOLUTION = 0.05
GRID_SIZE = 200
OBSTACLE_THRESHOLD = 0.5
COLLISION_CHECK_POINTS = 15
CAR_WIDTH = 0.35
SAFETY_MARGIN = 0.1
LATERAL_SEARCH_RANGE = 1.5
LATERAL_SEARCH_STEP = 0.15
AVOIDANCE_SPEED_FACTOR = 0.75
MIN_SCANS_FOR_AVOIDANCE = 10
MIN_OBSTACLE_DISTANCE = 0.8

# OVERTAKING CONSTANTS
MIN_SPEED_ADVANTAGE = 0.0
OVERTAKE_TRIGGER_DISTANCE = 9.0
OVERTAKE_MIN_DISTANCE = 1.6
OVERTAKE_LATERAL_OFFSET = 0.48
OVERTAKE_CLEARANCE_DISTANCE = 1.2
OVERTAKE_SPEED_FACTOR = 1.7
OPPONENT_POSITION_HISTORY_SIZE = 5

# Detection - ROBUST ANTI-WALL FILTERING!
OPPONENT_DETECTION_RANGE = 8.0
OPPONENT_DETECTION_CONE = 50
MIN_OPPONENT_VELOCITY = 0.01          
OPPONENT_RACELINE_TOLERANCE = 0.1
DETECTION_PERSISTENCE_FRAMES = 3      #Was 2 - need more evidence
MAX_CURVATURE_FOR_OVERTAKING = 1.6
STRICTLY_AHEAD_THRESHOLD = 0.0
OPPONENT_POSITION_HISTORY_SIZE = 4  #Was 5 - track longer history
# TODO: still detecting that one wall as obstacle. So overtaking spped boost allows it to pass. Need to maybe scale speed 
# boost more in order to get it past in the first boost of overtaking. So it oesnt rely on that outer wall.

# Motion Consistency Filters
MIN_VELOCITY_SAMPLES = 4              # Need 5 velocity measurements
MIN_CONSISTENT_VELOCITY_RATIO = 0.4   # 60% must be above threshold
MAX_LATERAL_MOTION_RATIO = 0.9        # Motion must be mostly forward
MAX_POSITION_JITTER = 1.0            # Position shouldn't jump > 0.2m

# Smart Side Selection
SIDE_SCAN_ANGLE_MIN = 60
SIDE_SCAN_ANGLE_MAX = 120
SIDE_SCAN_DISTANCE = 1.0
MIN_SAFE_SPACE = 0.1
SPACE_ADVANTAGE_THRESHOLD = 0.3

class DynamicPurePursuitAvoidance(Node):
    def __init__(self):
        super().__init__('dynamic_pure_pursuit_avoidance')

        # Parameters
        self.declare_parameter('track_file_path', '/home/nvidia/Desktop/team3-vip-f24/build/reactive_racing/reactive_racing/optimized_raceline.csv')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('visualize_path', True)
        self.declare_parameter('visualize_path_topic', '/loaded_raceline_path')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('visualization_publish_period', 5.0)
        self.declare_parameter('target_marker_topic', '/pursuit_target_marker')
        self.declare_parameter('base_lookahead', BASE_LOOKAHEAD)
        self.declare_parameter('min_lookahead', MIN_LOOKAHEAD)
        self.declare_parameter('max_lookahead', MAX_LOOKAHEAD)
        self.declare_parameter('max_speed', MAX_SPEED)
        self.declare_parameter('min_speed', MIN_SPEED)
        self.declare_parameter('curvature_speed_factor', 2.0)
        self.declare_parameter('enable_avoidance', False)
        self.declare_parameter('enable_overtaking', True)
        self.declare_parameter('overtaking_debug_mode', True)

        # Get Parameters
        track_file = self.get_parameter('track_file_path').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.visualize_path = self.get_parameter('visualize_path').get_parameter_value().bool_value
        self.visualize_path_topic = self.get_parameter('visualize_path_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.visualization_publish_period = self.get_parameter('visualization_publish_period').get_parameter_value().double_value
        self.target_marker_topic = self.get_parameter('target_marker_topic').get_parameter_value().string_value
        self.base_lookahead = self.get_parameter('base_lookahead').get_parameter_value().double_value
        self.min_lookahead = self.get_parameter('min_lookahead').get_parameter_value().double_value
        self.max_lookahead = self.get_parameter('max_lookahead').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.curvature_speed_factor = self.get_parameter('curvature_speed_factor').get_parameter_value().double_value
        self.enable_avoidance = self.get_parameter('enable_avoidance').get_parameter_value().bool_value
        self.enable_overtaking = self.get_parameter('enable_overtaking').get_parameter_value().bool_value
        self.overtaking_debug_mode = self.get_parameter('overtaking_debug_mode').get_parameter_value().bool_value

        # State
        self.raceline = None
        self.curvatures = None
        self.path_viz_message = None
        self.current_speed = self.min_speed
        self.current_pose = None
        self.current_curvature = 0.0
        
        # Avoidance State
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.latest_scan = None
        self.scan_count = 0
        self.avoiding = False
        self.avoidance_offset = 0.0
        self.target_offset = 0.0
        self.avoidance_ready = False
        
        # Overtaking State
        self.overtaking = False
        self.overtake_state = "NORMAL"
        self.opponent_detected = False
        self.opponent_position = None
        self.opponent_position_history = []
        self.opponent_velocity = 0.0
        self.opponent_distance = float('inf')
        self.overtake_side = 0.0
        self.frames_since_clear = 0
        self.frames_in_overtaking = 0
        self.frames_no_detection = 0
        self.consecutive_detections = 0

        # Load Raceline
        try:
            self.get_logger().info(f"Loading raceline from: {track_file}")
            if not os.path.exists(track_file):
                raise FileNotFoundError(f"Track file not found: {track_file}")
            
            loaded_data = np.loadtxt(track_file, delimiter=',', skiprows=1)
            if loaded_data.ndim != 2 or loaded_data.shape[1] < 3:
                raise ValueError(f"Invalid CSV shape: {loaded_data.shape}")
            
            self.raceline = loaded_data[:, 1:3]
            self.get_logger().info(f"Loaded {len(self.raceline)} waypoints")
            self.curvatures = self.compute_curvatures(self.raceline)

        except Exception as e:
            self.get_logger().error(f"Failed to load raceline: {e}")
            sys.exit(1)

        # ROS Communications
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # 🆕 AMCL POSE SUBSCRIPTION - For Real Car Compatibility
        # Subscribes to /amcl_pose as alternative to /odom
        # Allows controller to work on real car where AMCL is primary localization
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.amcl_pose_callback, 
            10
        )
        
        self.target_viz_pub = self.create_publisher(Marker, self.target_marker_topic, 10)
        self.avoidance_viz_pub = self.create_publisher(Marker, '/avoidance_marker', 10)
        
        # 🆕 OPPONENT DETECTION VISUALIZATION
        self.opponent_viz_pub = self.create_publisher(Marker, '/opponent_marker', 10)

        # Path Visualization
        if self.visualize_path:
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.path_viz_pub = self.create_publisher(Path, self.visualize_path_topic, qos_profile)
            self._generate_path_viz_message()
            if self.path_viz_message:
                self.path_viz_pub.publish(self.path_viz_message)
                if self.visualization_publish_period > 0:
                    self.viz_timer = self.create_timer(
                        self.visualization_publish_period,
                        self._timer_publish_path_callback
                    )

        self.get_logger().info('=' * 60)
        self.get_logger().info('🧠 SMART SIDE SELECTION - Space-Aware Overtaking')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'✓ Scans both sides for available space')
        self.get_logger().info(f'✓ Side scan range: {SIDE_SCAN_ANGLE_MIN}°-{SIDE_SCAN_ANGLE_MAX}°')
        self.get_logger().info(f'✓ Min safe space: {MIN_SAFE_SPACE}m')
        self.get_logger().info(f'✓ Space advantage threshold: {SPACE_ADVANTAGE_THRESHOLD}m')
        self.get_logger().info('=' * 60)
        self.get_logger().info('🎯 OPPONENT VISUALIZATION ENABLED')
        self.get_logger().info('  ➤ Topic: /opponent_marker')
        self.get_logger().info('  ➤ Red cube shows detected opponent position')
        self.get_logger().info('  ➤ Add Marker display in RViz to see it!')
        self.get_logger().info('=' * 60)
        self.get_logger().info('🛡️  ROBUST WALL FILTERING ENABLED')
        self.get_logger().info(f'  ➤ Min velocity: {MIN_OPPONENT_VELOCITY} m/s (rejects slow walls)')
        self.get_logger().info(f'  ➤ Consistency check: {MIN_CONSISTENT_VELOCITY_RATIO:.0%} threshold')
        self.get_logger().info(f'  ➤ Motion validation: Along-track movement required')
        self.get_logger().info(f'  ➤ Position stability: Anti-flicker filtering')
        self.get_logger().info('=' * 60)
        self.get_logger().info('📍 POSE SOURCES: Subscribed to /odom AND /amcl_pose')
        self.get_logger().info('   ➤ /odom: For simulation (high rate ~100Hz)')
        self.get_logger().info('   ➤ /amcl_pose: For real car (AMCL localization ~10Hz)')
        self.get_logger().info('   ➤ Will use whichever is available')
        self.get_logger().info('=' * 60)

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan"""
        self.latest_scan = msg
        self.scan_count += 1
        
        if not self.avoidance_ready and self.scan_count >= MIN_SCANS_FOR_AVOIDANCE:
            self.avoidance_ready = True
            self.get_logger().info('✓ Avoidance system ready')
        
        if self.current_pose is None:
            return
        
        # Clear grid
        self.occupancy_grid.fill(0.0)
        
        x, y, theta = self.current_pose
        
        # ONLY PROCESS FORWARD-FACING BEAMS (±75 degrees)
        forward_angle_range = math.radians(75)
        
        angle = msg.angle_min
        for i, r in enumerate(msg.ranges):
            if abs(angle) > forward_angle_range:
                angle += msg.angle_increment
                continue
            
            if msg.range_min < r < min(msg.range_max, 5.0):
                obs_x = x + r * math.cos(theta + angle)
                obs_y = y + r * math.sin(theta + angle)
                
                grid_x = int((obs_x - x) / GRID_RESOLUTION) + GRID_SIZE // 2
                grid_y = int((obs_y - y) / GRID_RESOLUTION) + GRID_SIZE // 2
                
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        gx = grid_x + dx
                        gy = grid_y + dy
                        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                            self.occupancy_grid[gx, gy] = 1.0
            
            angle += msg.angle_increment

    def measure_lateral_space(self, side='left'):
        """
        🆕 NEW: Measure available space on left or right side using LiDAR
        
        Returns: minimum distance to obstacle on specified side (meters)
        """
        if self.latest_scan is None:
            return float('inf')
        
        # Define angle range for scanning
        # Left side: 60-120° (positive angles)
        # Right side: -60 to -120° (negative angles)
        if side == 'left':
            angle_min = math.radians(SIDE_SCAN_ANGLE_MIN)
            angle_max = math.radians(SIDE_SCAN_ANGLE_MAX)
        else:  # right
            angle_min = math.radians(-SIDE_SCAN_ANGLE_MAX)
            angle_max = math.radians(-SIDE_SCAN_ANGLE_MIN)
        
        # Scan through LiDAR data
        min_distance = float('inf')
        angle = self.latest_scan.angle_min
        
        for r in self.latest_scan.ranges:
            # Check if angle is in our scan range
            if angle_min <= angle <= angle_max or angle_max <= angle <= angle_min:
                if self.latest_scan.range_min < r < SIDE_SCAN_DISTANCE:
                    # Calculate perpendicular distance to this obstacle
                    # (approximate - good enough for side checking)
                    lateral_distance = abs(r * math.sin(angle))
                    if lateral_distance < min_distance:
                        min_distance = lateral_distance
            
            angle += self.latest_scan.angle_increment
        
        return min_distance

    def detect_opponent(self, current_pos):
        """
        🆕 ENHANCED: Robust opponent detection with multi-layer wall filtering
        
        Filters applied:
        1. Range/Cone filter
        2. Ahead filter  
        3. Raceline proximity
        4. Velocity magnitude (rejects slow-moving walls)
        5. Velocity consistency (rejects inconsistent "motion")
        6. Motion direction (must move along track)
        7. Position stability (rejects flickering detections)
        """
        if self.latest_scan is None or self.current_pose is None:
            return None
        
        x, y, theta = self.current_pose
        
        # Basic LiDAR scanning
        angle = self.latest_scan.angle_min
        forward_angle_range = math.radians(OPPONENT_DETECTION_CONE)
        
        closest_obstacle = None
        closest_distance = float('inf')
        forward_detections = 0
        
        for r in self.latest_scan.ranges:
            if abs(angle) < forward_angle_range and self.latest_scan.range_min < r < OPPONENT_DETECTION_RANGE:
                forward_detections += 1
                
                obs_x = x + r * math.cos(theta + angle)
                obs_y = y + r * math.sin(theta + angle)
                
                # Transform to car frame
                dx = obs_x - x
                dy = obs_y - y
                local_x = math.cos(-theta) * dx - math.sin(-theta) * dy
                local_y = math.sin(-theta) * dx + math.cos(-theta) * dy
                
                # Ahead check
                if local_x < STRICTLY_AHEAD_THRESHOLD:
                    angle += self.latest_scan.angle_increment
                    continue
                
                # Raceline proximity check
                if self.raceline is not None:
                    deltas = self.raceline - np.array([obs_x, obs_y])
                    dist_to_raceline = np.min(np.sqrt(np.einsum('ij,ij->i', deltas, deltas)))
                    
                    if dist_to_raceline < OPPONENT_RACELINE_TOLERANCE and r < closest_distance:
                        closest_distance = r
                        closest_obstacle = np.array([obs_x, obs_y])
            
            angle += self.latest_scan.angle_increment
        
        if closest_obstacle is None:
            if self.scan_count % 10 == 0:
                self.get_logger().info(f"[DETECT] ✗ No objects in detection zone")
            return None
        
        # ========================================
        # ENHANCED VALIDATION FILTERS
        # ========================================
        
        # Build history first
        if len(self.opponent_position_history) < MIN_VELOCITY_SAMPLES:
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f"[DETECT] 📊 Building history: {len(self.opponent_position_history)}/{MIN_VELOCITY_SAMPLES}"
                )
            return closest_obstacle
        
        # FILTER 1: Velocity Magnitude
        velocities = []
        for i in range(1, min(len(self.opponent_position_history), MIN_VELOCITY_SAMPLES)):
            pos_prev = self.opponent_position_history[-i-1][0]
            pos_curr = self.opponent_position_history[-i][0]
            time_prev = self.opponent_position_history[-i-1][1]
            time_curr = self.opponent_position_history[-i][1]
            
            dist = np.hypot(pos_curr[0] - pos_prev[0], pos_curr[1] - pos_prev[1])
            dt = time_curr - time_prev
            
            if dt > 0:
                vel = dist / dt
                velocities.append(vel)
        
        if not velocities:
            return None
        
        avg_velocity = np.mean(velocities)
        
        if avg_velocity < MIN_OPPONENT_VELOCITY:
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f"[DETECT] 🛡️ WALL REJECTED! Avg velocity: {avg_velocity:.3f} m/s < {MIN_OPPONENT_VELOCITY} m/s"
                )
            return None
        
        # FILTER 2: Velocity Consistency
        above_threshold = sum(1 for v in velocities if v >= MIN_OPPONENT_VELOCITY * 0.7)
        consistency_ratio = above_threshold / len(velocities)
        
        if consistency_ratio < MIN_CONSISTENT_VELOCITY_RATIO:
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f"[DETECT] 🛡️ INCONSISTENT MOTION! Ratio: {consistency_ratio:.2f} < {MIN_CONSISTENT_VELOCITY_RATIO}"
                )
            return None
        
        # FILTER 3: Motion Direction
        if len(self.opponent_position_history) >= 3:
            first_pos = self.opponent_position_history[-3][0]
            last_pos = self.opponent_position_history[-1][0]
            
            motion_dx = last_pos[0] - first_pos[0]
            motion_dy = last_pos[1] - first_pos[1]
            motion_dist = np.hypot(motion_dx, motion_dy)
            
            if motion_dist > 0.1:
                deltas = self.raceline - np.array([last_pos[0], last_pos[1]])
                closest_idx = np.argmin(np.sqrt(np.einsum('ij,ij->i', deltas, deltas)))
                
                next_idx = (closest_idx + 5) % len(self.raceline)
                track_dx = self.raceline[next_idx][0] - self.raceline[closest_idx][0]
                track_dy = self.raceline[next_idx][1] - self.raceline[closest_idx][1]
                track_dist = np.hypot(track_dx, track_dy)
                
                if track_dist > 0.01:
                    motion_dx /= motion_dist
                    motion_dy /= motion_dist
                    track_dx /= track_dist
                    track_dy /= track_dist
                    
                    alignment = abs(motion_dx * track_dx + motion_dy * track_dy)
                    lateral_ratio = 1.0 - alignment
                    
                    if lateral_ratio > MAX_LATERAL_MOTION_RATIO:
                        if self.scan_count % 10 == 0:
                            self.get_logger().info(
                                f"[DETECT] 🛡️ OFF-TRACK MOTION! Lateral: {lateral_ratio:.2f} > {MAX_LATERAL_MOTION_RATIO}"
                            )
                        return None
        
        # FILTER 4: Position Stability
        if len(self.opponent_position_history) >= 2:
            last_pos = self.opponent_position_history[-1][0]
            jitter = np.hypot(
                closest_obstacle[0] - last_pos[0],
                closest_obstacle[1] - last_pos[1]
            )
            
            max_allowed_jitter = MAX_POSITION_JITTER * (1.0 + avg_velocity / 2.0)
            
            if jitter > max_allowed_jitter:
                if self.scan_count % 10 == 0:
                    self.get_logger().info(
                        f"[DETECT] 🛡️ POSITION JUMP! Jitter: {jitter:.3f}m > {max_allowed_jitter:.3f}m"
                    )
                return None
        
        # ALL FILTERS PASSED!
        if self.scan_count % 10 == 0:
            dx = closest_obstacle[0] - x
            dy = closest_obstacle[1] - y
            local_y = math.sin(-theta) * dx + math.cos(-theta) * dy
            
            self.get_logger().info(
                f"[DETECT] ✅ OPPONENT VALIDATED! "
                f"Dist: {closest_distance:.2f}m | "
                f"Vel: {avg_velocity:.2f}m/s | "
                f"Consistency: {consistency_ratio:.0%} | "
                f"Side: {'RIGHT' if local_y > 0 else 'LEFT'}"
            )
        
        return closest_obstacle

    def estimate_opponent_velocity(self):
        """
        🆕 ENHANCED: More robust velocity estimation with outlier rejection
        """
        if len(self.opponent_position_history) < 3:
            return 0.0
        
        # Use more samples for better accuracy
        recent_positions = self.opponent_position_history[-MIN_VELOCITY_SAMPLES:]
        velocities = []
        
        for i in range(1, len(recent_positions)):
            pos_prev = recent_positions[i-1][0]
            pos_curr = recent_positions[i][0]
            time_prev = recent_positions[i-1][1]
            time_curr = recent_positions[i][1]
            
            dist = np.hypot(pos_curr[0] - pos_prev[0], pos_curr[1] - pos_prev[1])
            dt = time_curr - time_prev
            
            if dt > 0:
                vel = dist / dt
                velocities.append(vel)
        
        if not velocities:
            return 0.0
        
        # Remove outliers (velocities > 3 std devs from mean)
        if len(velocities) >= 3:
            mean_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            
            filtered_velocities = [v for v in velocities if abs(v - mean_vel) < 3 * std_vel]
            
            if filtered_velocities:
                return np.mean(filtered_velocities)
        
        return np.mean(velocities)

    def should_attempt_overtake(self, current_pos):
        """
        Decide if we should attempt to overtake
        
        🆕 NEW: SMART SIDE SELECTION - Chooses side with more space!
        """
        if not self.opponent_detected or self.opponent_position is None:
            return False, 0.0
        
        # Curvature blocking
        if abs(self.current_curvature) > MAX_CURVATURE_FOR_OVERTAKING:
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f"[OVERTAKE] 🛡️ BLOCKED - Sharp turn! Curvature: {self.current_curvature:.3f}"
                )
            return False, 0.0
        
        # Check distance
        if self.opponent_distance > OVERTAKE_TRIGGER_DISTANCE:
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f"[OVERTAKE] ✗ Too far: {self.opponent_distance:.2f}m"
                )
            return False, 0.0
        
        if self.opponent_distance < OVERTAKE_MIN_DISTANCE:
            if self.scan_count % 10 == 0:
                self.get_logger().info(
                    f"[OVERTAKE] ✗ Too close: {self.opponent_distance:.2f}m"
                )
            return False, 0.0
        
        # Check speed advantage
        speed_advantage = self.current_speed - self.opponent_velocity
        
        if self.scan_count % 10 == 0:
            self.get_logger().info(
                f"[OVERTAKE] My speed: {self.current_speed:.2f}m/s | Opp: {self.opponent_velocity:.2f}m/s | "
                f"Advantage: {speed_advantage:.2f}m/s"
            )
        
        if speed_advantage < MIN_SPEED_ADVANTAGE:
            return False, 0.0
        
        # 🆕 SMART SIDE SELECTION - Measure space on both sides!
        left_space = self.measure_lateral_space('left')
        right_space = self.measure_lateral_space('right')
        
        self.get_logger().info(
            f"[SPACE] Left: {left_space:.2f}m | Right: {right_space:.2f}m | "
            f"Min safe: {MIN_SAFE_SPACE}m"
        )
        
        # Check if either side has enough space
        if left_space < MIN_SAFE_SPACE and right_space < MIN_SAFE_SPACE:
            self.get_logger().warn(
                f"[OVERTAKE] ✗ NO SPACE! Left: {left_space:.2f}m, Right: {right_space:.2f}m, Need: {MIN_SAFE_SPACE}m"
            )
            return False, 0.0
        
        # Choose side with MORE space
        space_diff = left_space - right_space
        
        if abs(space_diff) > SPACE_ADVANTAGE_THRESHOLD:
            # One side has significantly more space
            if left_space > right_space:
                overtake_side = OVERTAKE_LATERAL_OFFSET  # Go LEFT
                side_name = "LEFT"
            else:
                overtake_side = -OVERTAKE_LATERAL_OFFSET  # Go RIGHT
                side_name = "RIGHT"
            
            self.get_logger().info(
                f"[OVERTAKE] 🧠 SMART CHOICE: Going {side_name} (space: {max(left_space, right_space):.2f}m vs {min(left_space, right_space):.2f}m)"
            )
        else:
            # Spaces are similar - use opponent position as tiebreaker
            x, y, theta = self.current_pose
            dx = self.opponent_position[0] - x
            dy = self.opponent_position[1] - y
            local_y = math.sin(-theta) * dx + math.cos(-theta) * dy
            
            # Go opposite of opponent
            overtake_side = OVERTAKE_LATERAL_OFFSET if local_y > 0 else -OVERTAKE_LATERAL_OFFSET
            side_name = "LEFT" if overtake_side > 0 else "RIGHT"
            
            self.get_logger().info(
                f"[OVERTAKE] 🎯 TIEBREAKER: Going {side_name} (opposite opponent, spaces similar)"
            )
        
        self.get_logger().info(
            f"[OVERTAKE] ✓ ALL CONDITIONS MET! Distance: {self.opponent_distance:.2f}m, "
            f"Advantage: {speed_advantage:.2f}m/s, Side: {side_name}"
        )
        
        return True, overtake_side

    def update_overtaking_state(self, current_pos):
        """Update overtaking state machine with persistence filtering"""
        opponent_pos = self.detect_opponent(current_pos)
        
        if opponent_pos is not None:
            self.consecutive_detections += 1
            
            if self.consecutive_detections >= DETECTION_PERSISTENCE_FRAMES:
                self.opponent_detected = True
                self.opponent_position = opponent_pos
                
                current_time = self.get_clock().now().nanoseconds / 1e9
                self.opponent_position_history.append((opponent_pos, current_time))
                
                if len(self.opponent_position_history) > OPPONENT_POSITION_HISTORY_SIZE:
                    self.opponent_position_history.pop(0)
                
                self.opponent_velocity = self.estimate_opponent_velocity()
                self.opponent_distance = np.hypot(
                    opponent_pos[0] - current_pos[0],
                    opponent_pos[1] - current_pos[1]
                )
                
                # 🆕 VISUALIZE OPPONENT IN RVIZ
                self._publish_opponent_marker(
                    opponent_pos[0], 
                    opponent_pos[1], 
                    detected=True
                )
            else:
                if self.scan_count % 10 == 0:
                    self.get_logger().info(
                        f"[DETECT] 🛡️ Waiting: {self.consecutive_detections}/{DETECTION_PERSISTENCE_FRAMES} frames"
                    )
        else:
            if self.consecutive_detections > 0:
                self.get_logger().info(f"[DETECT] 🛡️ Detection lost - resetting")
            self.consecutive_detections = 0
            self.opponent_detected = False
            
            # 🆕 REMOVE OPPONENT MARKER FROM RVIZ
            self._publish_opponent_marker(0.0, 0.0, detected=False)
        
        # DEBUG STATUS
        if self.scan_count % 10 == 0:
            self.get_logger().info(
                f"[STATUS] State: {self.overtake_state} | Detected: {self.opponent_detected} | "
                f"Distance: {self.opponent_distance:.2f}m"
            )
        
        # State Machine (unchanged from working version)
        if self.overtake_state == "NORMAL":
            if self.opponent_detected:
                should_overtake, side = self.should_attempt_overtake(current_pos)
                if should_overtake:
                    self.overtake_state = "APPROACHING"
                    self.overtake_side = side
                    self.frames_in_overtaking = 0
                    self.frames_no_detection = 0
                    self.get_logger().info(
                        f"🏁 OVERTAKE INITIATED! Side: {'LEFT' if side > 0 else 'RIGHT'}, "
                        f"Distance: {self.opponent_distance:.2f}m"
                    )
        
        elif self.overtake_state == "APPROACHING":
            if self.opponent_distance < OVERTAKE_MIN_DISTANCE:
                self.overtake_state = "OVERTAKING"
                self.overtaking = True
                self.frames_in_overtaking = 0
                self.frames_no_detection = 0
                self.get_logger().info("🏎️  OVERTAKING - executing maneuver")
        
        elif self.overtake_state == "OVERTAKING":
            self.frames_in_overtaking += 1
            
            x, y, theta = self.current_pose
            
            if not self.opponent_detected:
                self.frames_no_detection += 1
            else:
                self.frames_no_detection = 0
            
            if self.frames_no_detection > 20:
                self.overtake_state = "CLEARING"
                self.frames_since_clear = 0
                self.get_logger().info("💨 CLEARING - lost detection")
            
            elif self.frames_in_overtaking > 100:
                self.overtake_state = "MERGING_BACK"
                self.get_logger().warn("⚠️  OVERTAKING TIMEOUT - forcing merge")
            
            elif self.opponent_position is not None:
                dx = self.opponent_position[0] - x
                dy = self.opponent_position[1] - y
                local_x = math.cos(-theta) * dx - math.sin(-theta) * dy
                
                if local_x < -0.5:
                    self.overtake_state = "CLEARING"
                    self.frames_since_clear = 0
                    self.get_logger().info("💨 CLEARING - opponent behind")
        
        elif self.overtake_state == "CLEARING":
            x, y, theta = self.current_pose
            
            if self.opponent_position is not None:
                dx = self.opponent_position[0] - x
                dy = self.opponent_position[1] - y
                local_x = math.cos(-theta) * dx - math.sin(-theta) * dy
                distance_ahead = -local_x
                
                if distance_ahead > OVERTAKE_CLEARANCE_DISTANCE:
                    self.frames_since_clear += 1
                    if self.frames_since_clear > 15:
                        self.overtake_state = "MERGING_BACK"
                        self.get_logger().info("↩️  MERGING BACK")
                else:
                    self.frames_since_clear = 0
            else:
                self.frames_since_clear += 1
                if self.frames_since_clear > 30:
                    self.overtake_state = "MERGING_BACK"
                    self.get_logger().info("↩️  MERGING BACK - timeout")
        
        elif self.overtake_state == "MERGING_BACK":
            if self.opponent_detected and self.opponent_distance < 1.0:
                self.overtake_state = "NORMAL"
                self.overtaking = False
                self.avoidance_offset = 0.0
                self.target_offset = 0.0
                self.frames_in_overtaking = 0
                self.consecutive_detections = 0
                self.get_logger().warn("⚠️  EMERGENCY ABORT")
            
            elif abs(self.avoidance_offset) < 0.1:
                self.overtake_state = "NORMAL"
                self.overtaking = False
                self.frames_in_overtaking = 0
                self.consecutive_detections = 0
                self.get_logger().info("✅ OVERTAKE COMPLETE")
            
            elif self.frames_in_overtaking > 150:
                self.overtake_state = "NORMAL"
                self.overtaking = False
                self.avoidance_offset = 0.0
                self.target_offset = 0.0
                self.frames_in_overtaking = 0
                self.consecutive_detections = 0
                self.get_logger().warn("⚠️  FORCED COMPLETE")

    def odom_callback(self, msg: Odometry):
        """Handle odometry and run control loop"""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        x, y = position.x, position.y
        theta = self.get_yaw_from_quaternion(orientation)
        
        # Detect episode reset
        if self.current_pose is not None:
            old_x, old_y, _ = self.current_pose
            distance_jumped = np.hypot(x - old_x, y - old_y)
            
            if distance_jumped > 5.0:
                self.get_logger().warn("🔄 EPISODE RESET")
                self.overtaking = False
                self.overtake_state = "NORMAL"
                self.avoidance_offset = 0.0
                self.target_offset = 0.0
                self.frames_in_overtaking = 0
                self.frames_no_detection = 0
                self.consecutive_detections = 0
                self.opponent_detected = False
                self.avoiding = False
        
        self.current_pose = (x, y, theta)
        self.control_loop(msg.header.stamp)

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        🆕 Handle AMCL pose updates (for real car compatibility)
        
        This callback allows the controller to work with AMCL localization
        on the real car where /odom might not be available or reliable.
        Uses the same control logic as odom_callback.
        """
        # Extract pose from AMCL message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        x, y = position.x, position.y
        theta = self.get_yaw_from_quaternion(orientation)
        
        # Detect large position jumps (manual reset or relocalization)
        if self.current_pose is not None:
            old_x, old_y, _ = self.current_pose
            distance_jumped = np.hypot(x - old_x, y - old_y)
            
            if distance_jumped > 5.0:
                self.get_logger().warn("🔄 POSITION RESET DETECTED (AMCL)")
                # Reset all overtaking/avoidance state
                self.overtaking = False
                self.overtake_state = "NORMAL"
                self.avoidance_offset = 0.0
                self.target_offset = 0.0
                self.frames_in_overtaking = 0
                self.frames_no_detection = 0
                self.consecutive_detections = 0
                self.opponent_detected = False
                self.avoiding = False
        
        # Update current pose and run control loop
        self.current_pose = (x, y, theta)
        self.control_loop(msg.header.stamp)

    def control_loop(self, timestamp):
        """Main control loop"""
        
        if self.current_pose is None or self.raceline is None:
            return
        
        x, y, theta = self.current_pose
        
        # UPDATE OVERTAKING STATE
        if self.enable_overtaking and self.avoidance_ready:
            self.update_overtaking_state(np.array([x, y]))
        
        lookahead_distance = self.compute_dynamic_lookahead(self.current_speed)
        lookahead_point, curvature = self.find_lookahead_point(np.array([x, y]), lookahead_distance)
        
        self.current_curvature = curvature
        
        if lookahead_point is None:
            self.get_logger().warn("No lookahead point!", throttle_duration_sec=2.0)
            self.publish_stop()
            return
        
        collision_detected = False
        target_point = lookahead_point.copy()
        
        # Handle overtaking vs avoidance
        if self.overtaking and self.overtake_state != "MERGING_BACK":
            self.target_offset = self.overtake_side
        elif self.overtaking and self.overtake_state == "MERGING_BACK":
            self.target_offset = 0.0
        elif self.enable_avoidance and self.avoidance_ready:
            collision_detected = self.check_path_collision(
                np.array([x, y]), 
                lookahead_point,
                self.avoidance_offset
            )
            
            if collision_detected:
                clear_offset = self.find_clear_lateral_path(np.array([x, y]), lookahead_point)
                
                if clear_offset is not None:
                    if not self.avoiding:
                        self.get_logger().info(f"OBSTACLE! Avoiding: {clear_offset:.2f}m")
                    self.avoiding = True
                    self.target_offset = clear_offset
                else:
                    self.get_logger().warn("No clear path - STOPPING!")
                    self.publish_stop()
                    return
            else:
                if self.avoiding and abs(self.avoidance_offset) < 0.05:
                    self.avoiding = False
                    self.get_logger().info("✓ Returned to raceline")
                
                self.target_offset *= 0.95
        
        # Smooth offset
        alpha = 0.5 if self.overtaking else 0.3
        self.avoidance_offset = alpha * self.target_offset + (1 - alpha) * self.avoidance_offset
        
        if abs(self.avoidance_offset) > 0.01:
            target_point = self.apply_lateral_offset(
                np.array([x, y]),
                lookahead_point,
                self.avoidance_offset
            )
        
        # Transform to car frame
        dx = target_point[0] - x
        dy = target_point[1] - y
        lx = math.cos(-theta) * dx - math.sin(-theta) * dy
        ly = math.sin(-theta) * dx + math.cos(-theta) * dy

        # Pure Pursuit steering
        dist_sq = lx**2 + ly**2
        if dist_sq < 1e-3:
            steering_angle = 0.0
        else:
            steering_curvature = (2.0 * ly) / dist_sq
            steering_angle = math.atan(self.wheelbase * steering_curvature)
        
        # Dynamic speed
        target_speed = self.compute_dynamic_speed(curvature)
        
        if self.overtaking:
            target_speed *= OVERTAKE_SPEED_FACTOR
        elif self.avoiding:
            target_speed *= AVOIDANCE_SPEED_FACTOR
        
        # Smooth speed
        alpha_speed = 0.3
        self.current_speed = alpha_speed * target_speed + (1 - alpha_speed) * self.current_speed

        # Clamp steering
        steering_angle = np.clip(steering_angle, -0.4, 0.4)

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = timestamp
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.current_speed
        self.drive_pub.publish(drive_msg)

        # Visualization
        self._publish_target_marker(lx, ly, x, y, theta)
        if self.avoiding or self.overtaking:
            self._publish_avoidance_marker(target_point[0], target_point[1])

    # [Include all helper methods - same as before]
    def check_path_collision(self, start_pos, end_pos, current_offset):
        if self.latest_scan is None:
            return False
        collision_found = False
        for i in range(COLLISION_CHECK_POINTS):
            t = i / (COLLISION_CHECK_POINTS - 1)
            point = start_pos + t * (end_pos - start_pos)
            dist_from_car = np.hypot(point[0] - start_pos[0], point[1] - start_pos[1])
            if dist_from_car < MIN_OBSTACLE_DISTANCE:
                continue
            if abs(current_offset) > 0.01:
                point = self.apply_lateral_offset(start_pos, point, current_offset)
            if self.is_point_in_collision(point, start_pos):
                collision_found = True
                break
        return collision_found

    def is_point_in_collision(self, point, car_pos):
        dx = point[0] - car_pos[0]
        dy = point[1] - car_pos[1]
        grid_x = int(dx / GRID_RESOLUTION) + GRID_SIZE // 2
        grid_y = int(dy / GRID_RESOLUTION) + GRID_SIZE // 2
        check_radius = int((CAR_WIDTH/2 + SAFETY_MARGIN) / GRID_RESOLUTION)
        for dx_check in range(-check_radius, check_radius + 1):
            for dy_check in range(-check_radius, check_radius + 1):
                gx = grid_x + dx_check
                gy = grid_y + dy_check
                if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                    if self.occupancy_grid[gx, gy] > OBSTACLE_THRESHOLD:
                        return True
        return False

    def find_clear_lateral_path(self, start_pos, nominal_end):
        if not self.check_path_collision(start_pos, nominal_end, 0.0):
            return 0.0
        search_offsets = []
        offset = LATERAL_SEARCH_STEP
        while offset <= LATERAL_SEARCH_RANGE:
            search_offsets.append(offset)
            search_offsets.append(-offset)
            offset += LATERAL_SEARCH_STEP
        for offset in search_offsets:
            if not self.check_path_collision(start_pos, nominal_end, offset):
                return offset
        return None

    def apply_lateral_offset(self, car_pos, target_point, offset):
        dx = target_point[0] - car_pos[0]
        dy = target_point[1] - car_pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return target_point
        perp_x = -dy / dist
        perp_y = dx / dist
        new_point = np.array([
            target_point[0] + offset * perp_x,
            target_point[1] + offset * perp_y
        ])
        return new_point

    def compute_curvatures(self, path):
        N = len(path)
        curvatures = np.zeros(N)
        if N < 3:
            return curvatures
        for i in range(1, N - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            v12 = p2 - p1
            v23 = p3 - p2
            v31 = p1 - p3
            a = np.linalg.norm(v23)
            b = np.linalg.norm(v31)
            c = np.linalg.norm(v12)
            area = np.abs(np.cross(v12, -v31))
            den = a * b * c
            if den > 1e-9 and area > 1e-9:
                curvatures[i] = (2.0 * area) / den
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        return curvatures

    def compute_dynamic_lookahead(self, current_speed):
        speed_factor = (current_speed - self.min_speed) / (self.max_speed - self.min_speed)
        speed_factor = np.clip(speed_factor, 0.0, 1.0)
        lookahead = self.base_lookahead + (speed_factor * LOOKAHEAD_SPEED_GAIN * self.base_lookahead)
        return np.clip(lookahead, self.min_lookahead, self.max_lookahead)

    def compute_dynamic_speed(self, curvature):
        speed_reduction = min(abs(curvature) * self.curvature_speed_factor, 1.0)
        speed = self.max_speed * (1.0 - speed_reduction)
        return np.clip(speed, self.min_speed, self.max_speed)

    def find_lookahead_point(self, current_pos, lookahead_distance):
        if self.raceline is None or len(self.raceline) == 0:
            return None, 0.0
        deltas = self.raceline - current_pos
        dist_sq = np.einsum('ij,ij->i', deltas, deltas)
        closest_idx = np.argmin(dist_sq)
        n = len(self.raceline)
        for i in range(n):
            ci = (closest_idx + i) % n
            p = self.raceline[ci]
            dist = np.hypot(p[0] - current_pos[0], p[1] - current_pos[1])
            if dist >= lookahead_distance:
                fc = self.curvatures[ci] if ci < len(self.curvatures) else 0.0
                return p, fc
        furthest_idx = (closest_idx + n//2) % n
        return self.raceline[furthest_idx], self.curvatures[furthest_idx]

    def publish_stop(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)
        self.current_speed = 0.0

    def _publish_target_marker(self, lx, ly, car_x, car_y, car_theta):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pursuit_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        map_x = car_x + (lx * math.cos(car_theta) - ly * math.sin(car_theta))
        map_y = car_y + (lx * math.sin(car_theta) + ly * math.cos(car_theta))
        marker.pose.position.x = map_x
        marker.pose.position.y = map_y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 0.9
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.target_viz_pub.publish(marker)

    def _publish_avoidance_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'avoidance_target'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 0.9
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        self.avoidance_viz_pub.publish(marker)

    def _publish_opponent_marker(self, x, y, detected):
        """
        🆕 OPPONENT DETECTION VISUALIZATION
        
        Publishes a marker at the detected opponent position for RViz debugging.
        - RED CUBE when opponent is detected
        - Marker disappears when no opponent detected
        
        Args:
            x, y: Opponent position in map frame
            detected: Boolean - whether opponent is currently detected
        """
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'opponent_detection'
        marker.id = 2
        marker.type = Marker.CUBE  # Cube to represent opponent car
        
        if detected:
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.15  # Slightly above ground
            marker.pose.orientation.w = 1.0
            
            # Size - roughly car-shaped
            marker.scale.x = 0.5  # Length
            marker.scale.y = 0.3  # Width
            marker.scale.z = 0.3  # Height
            
            # Color - BRIGHT RED for visibility
            marker.color.a = 0.9  # Slightly transparent
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            # Lifetime - will disappear after 0.5 seconds if not updated
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 500000000  # 0.5 seconds
        else:
            # Delete marker when no opponent
            marker.action = Marker.DELETE
        
        self.opponent_viz_pub.publish(marker)

    def _generate_path_viz_message(self):
        if self.raceline is None or len(self.raceline) == 0:
            return
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        for wp in self.raceline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_viz_message = path_msg

    def _timer_publish_path_callback(self):
        if self.path_viz_pub and self.path_viz_message:
            self.path_viz_message.header.stamp = self.get_clock().now().to_msg()
            self.path_viz_pub.publish(self.path_viz_message)

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DynamicPurePursuitAvoidance()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
