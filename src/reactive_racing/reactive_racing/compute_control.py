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

'''
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
