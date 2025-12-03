#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np
import math
import os
import sys
from collections import deque

# ==============================================================================
# PART 1: CONSTANTS (Merged)
# ==============================================================================

# --- FROM SOURCE A (Working Pure Pursuit) ---
LOOKAHEAD_DISTANCE = 1.2
MAX_SPEED = 0.65
MIN_SPEED = 0.55

# --- FROM SOURCE B (Overtaking Logic) ---
# Detection
OPPONENT_DETECTION_RANGE = 8.0
OPPONENT_DETECTION_CONE = 50
MIN_OPPONENT_VELOCITY = 0.01          
OPPONENT_RACELINE_TOLERANCE = 0.05
DETECTION_PERSISTENCE_FRAMES = 3
STRICTLY_AHEAD_THRESHOLD = 0.0
OPPONENT_POSITION_HISTORY_SIZE = 4

# Overtaking
MIN_SPEED_ADVANTAGE = 0.0
OVERTAKE_TRIGGER_DISTANCE = 9.0
OVERTAKE_MIN_DISTANCE = 1.6
OVERTAKE_LATERAL_OFFSET = 0.525
OVERTAKE_CLEARANCE_DISTANCE = 1.2
OVERTAKE_SPEED_FACTOR = 1.0 # Speed boost when passing

# Side Selection
SIDE_SCAN_ANGLE_MIN = 60
SIDE_SCAN_ANGLE_MAX = 120
SIDE_SCAN_DISTANCE = 1.0
MIN_SAFE_SPACE = 0.1
SPACE_ADVANTAGE_THRESHOLD = 0.3

# Motion Filters
MIN_VELOCITY_SAMPLES = 4
MIN_CONSISTENT_VELOCITY_RATIO = 0.4
MAX_LATERAL_MOTION_RATIO = 0.9
MAX_POSITION_JITTER = 1.0


class HybridControlNode(Node):
    def __init__(self):
        super().__init__('hybrid_control_node')

        # --- PARAMETERS (From Source A) ---
        self.declare_parameter('track_file_path', '/home/nvidia/Desktop/team3-vip-f24/build/reactive_racing/reactive_racing/optimized_raceline.csv')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('visualize_path', True)
        self.declare_parameter('visualize_path_topic', '/loaded_raceline_path')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('visualization_publish_period', 5.0)
        
        # --- PARAMETERS (From Source B) ---
        self.declare_parameter('enable_overtaking', True)
        self.declare_parameter('overtaking_debug_mode', True)

        # Get Values
        track_file = self.get_parameter('track_file_path').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.visualize_path = self.get_parameter('visualize_path').get_parameter_value().bool_value
        self.visualize_path_topic = self.get_parameter('visualize_path_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.visualization_publish_period = self.get_parameter('visualization_publish_period').get_parameter_value().double_value
        self.enable_overtaking = self.get_parameter('enable_overtaking').get_parameter_value().bool_value
        self.overtaking_debug_mode = self.get_parameter('overtaking_debug_mode').get_parameter_value().bool_value

        # --- INITIALIZATION ---
        self.raceline = None
        self.curvatures = None
        self.path_viz_message = None
        
        # This is needed for Source B logic to know how fast we are going
        self.current_speed = 0.0 
        self.current_pose = None

        # Overtaking State (From Source B)
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
        self.latest_scan = None
        self.scan_count = 0

        # Load Raceline (From Source A)
        try:
            self.get_logger().info(f"Loading raceline from: {track_file}")
            if not os.path.exists(track_file):
                raise FileNotFoundError(f"Track file not found: {track_file}")
            loaded_data = np.loadtxt(track_file, delimiter=',', skiprows=1)
            self.raceline = loaded_data[:, 1:3]
            self.curvatures = self.compute_curvatures(self.raceline)
            self.get_logger().info(f"Loaded {len(self.raceline)} waypoints")
        except Exception as e:
            self.get_logger().error(f"Failed to load raceline: {e}"); sys.exit(1)

        # --- ROS COMMUNICATIONS (REAL WORLD COMPATIBLE) ---
        
        # QoS Profile for Hardware (Best Effort is required for VESC/LiDAR)
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.target_viz_pub = self.create_publisher(Marker, '/pursuit_target_marker', 10)
        self.opponent_viz_pub = self.create_publisher(Marker, '/opponent_marker', 10)

        # 1. Main Control Trigger (Reliable) -> From Source A
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # 2. LiDAR Input (Best Effort) -> From Source B
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, best_effort_qos)

        # 3. Speed Input (Best Effort) -> Just to update self.current_speed
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, best_effort_qos)

        # Visualization
        if self.visualize_path:
            qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.path_viz_pub = self.create_publisher(Path, self.visualize_path_topic, qos_profile)
            self._generate_path_viz_message()
            if self.path_viz_message:
                self.path_viz_pub.publish(self.path_viz_message)
                self.viz_timer = self.create_timer(
                    self.visualization_publish_period, self._timer_publish_path_callback)

        self.get_logger().info('✅ HYBRID CONTROLLER INITIALIZED')
        self.get_logger().info('   - Core Logic: Source A (Simple Pure Pursuit)')
        self.get_logger().info('   - Overtaking: Source B (Smart Side Selection)')
        self.get_logger().info('   - Hardware: Best Effort QoS Enabled')

    # ==========================================================================
    # PART 2: CORE CONTROL LOOP (Based on Source A, Enhanced with B)
    # ==========================================================================

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Triggered by AMCL (Reliable). This is the 'Heartbeat' of the controller.
        """
        # 1. Update State
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        x, y = position.x, position.y
        theta = self.get_yaw_from_quaternion(orientation)
        self.current_pose = (x, y, theta)

        # 2. Update Overtaking Logic (Source B Logic)
        lateral_offset = 0.0
        if self.enable_overtaking and self.latest_scan is not None:
            self.update_overtaking_state(np.array([x, y]))
            
            # If state machine says overtake, we set the offset
            if self.overtaking and self.overtake_state != "MERGING_BACK":
                lateral_offset = self.overtake_side
            # If merging, offset is 0 (return to path)
            
            # Visualize opponent
            if self.opponent_detected and self.opponent_position is not None:
                self._publish_opponent_marker(self.opponent_position[0], self.opponent_position[1], True)
            else:
                self._publish_opponent_marker(0, 0, False)

        # 3. Find Lookahead Point (Source A Logic)
        lookahead_point, curvature = self.find_lookahead_point(np.array([x, y]))
        
        if lookahead_point is None:
            # Stop car if no target found
            self.stop_car()
            return

        # 4. Apply Overtaking Offset (The Integration)
        target_point = lookahead_point
        if abs(lateral_offset) > 0.01:
            target_point = self.apply_lateral_offset(np.array([x, y]), lookahead_point, lateral_offset)

        # 5. Transform & Steer (Source A Logic)
        dx = target_point[0] - x
        dy = target_point[1] - y
        lx = math.cos(-theta) * dx - math.sin(-theta) * dy
        ly = math.sin(-theta) * dx + math.cos(-theta) * dy
        dist_sq = lx**2 + ly**2

        if dist_sq < 1e-3:
            sa = 0.0
        else:
            steering_curvature = (2.0 * ly) / dist_sq
            sa = math.atan(self.wheelbase * steering_curvature)

        # 6. Clamp steering (Source A Logic)
        max_steer = 0.4
        sa = np.clip(sa, -max_steer, max_steer)

        # 7. Speed Calculation (Source A Logic + Overtake Boost)
        # Source A: speed=max(MIN, MAX*(1-curve))
        speed = max(MIN_SPEED, MAX_SPEED * (1.0 - min(abs(curvature) * 0.5, 1.0)))
        
        if self.overtaking:
            speed *= OVERTAKE_SPEED_FACTOR # Boost speed when passing!

        # 8. Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = sa
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)
        
        # Viz Target
        self._publish_target_marker(lx, ly, x, y, theta)

    # ==========================================================================
    # PART 3: SENSOR CALLBACKS (Data Ingestion)
    # ==========================================================================

    def scan_callback(self, msg: LaserScan):
        """Just stores data for the pose_callback to use."""
        self.latest_scan = msg
        self.scan_count += 1

    def odom_callback(self, msg: Odometry):
        """Just updates current speed for decision making."""
        self.current_speed = msg.twist.twist.linear.x

    # ==========================================================================
    # PART 4: OVERTAKING LOGIC (Directly from Source B)
    # ==========================================================================

    def update_overtaking_state(self, current_pos):
        opponent_pos = self.detect_opponent(current_pos)
        
        # 1. Update Detection History
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
                self.opponent_distance = np.hypot(opponent_pos[0] - current_pos[0], opponent_pos[1] - current_pos[1])
        else:
            self.consecutive_detections = 0
            self.opponent_detected = False

        # 2. State Machine
        if self.overtake_state == "NORMAL":
            if self.opponent_detected:
                should_overtake, side = self.should_attempt_overtake(current_pos)
                if should_overtake:
                    self.overtake_state = "APPROACHING"
                    self.overtake_side = side
                    self.frames_in_overtaking = 0; self.frames_no_detection = 0
                    self.get_logger().info(f"🏁 OVERTAKE START! Side: {'LEFT' if side > 0 else 'RIGHT'}")
        
        elif self.overtake_state == "APPROACHING":
            if self.opponent_distance < OVERTAKE_MIN_DISTANCE:
                self.overtake_state = "OVERTAKING"
                self.overtaking = True
        
        elif self.overtake_state == "OVERTAKING":
            self.frames_in_overtaking += 1
            # Check if lost detection
            if not self.opponent_detected: self.frames_no_detection += 1
            else: self.frames_no_detection = 0
            
            # Check if opponent is behind us
            is_behind = False
            if self.opponent_position is not None:
                x, y, theta = self.current_pose
                dx = self.opponent_position[0] - x; dy = self.opponent_position[1] - y
                local_x = math.cos(-theta) * dx - math.sin(-theta) * dy
                if local_x < -0.5: is_behind = True

            if is_behind or self.frames_no_detection > 20:
                self.overtake_state = "CLEARING"
                self.frames_since_clear = 0
                self.get_logger().info("💨 CLEARING (Opponent passed)")
            elif self.frames_in_overtaking > 100:
                self.overtake_state = "MERGING_BACK" # Timeout
        
        elif self.overtake_state == "CLEARING":
            self.frames_since_clear += 1
            if self.frames_since_clear > 15:
                self.overtake_state = "MERGING_BACK"
                self.get_logger().info("↩️  MERGING BACK")
        
        elif self.overtake_state == "MERGING_BACK":
            self.overtaking = False # Stop applying offset
            self.overtake_state = "NORMAL" # Reset

    def detect_opponent(self, current_pos):
        if self.latest_scan is None or self.current_pose is None: return None
        x, y, theta = self.current_pose
        angle = self.latest_scan.angle_min
        forward_angle_range = math.radians(OPPONENT_DETECTION_CONE)
        
        closest_obstacle = None
        closest_distance = float('inf')
        
        # Scan processing
        for r in self.latest_scan.ranges:
            if abs(angle) < forward_angle_range and self.latest_scan.range_min < r < OPPONENT_DETECTION_RANGE:
                # Transform to map frame (Source B Logic)
                obs_x = x + r * math.cos(theta + angle)
                obs_y = y + r * math.sin(theta + angle)
                
                # Check strict ahead (Source B Logic)
                dx = obs_x - x; dy = obs_y - y
                local_x = math.cos(-theta) * dx - math.sin(-theta) * dy
                if local_x < STRICTLY_AHEAD_THRESHOLD:
                    angle += self.latest_scan.angle_increment; continue
                
                # Check raceline proximity (Source B Logic)
                if self.raceline is not None:
                    deltas = self.raceline - np.array([obs_x, obs_y])
                    dist_to_raceline = np.min(np.sqrt(np.einsum('ij,ij->i', deltas, deltas)))
                    if dist_to_raceline < OPPONENT_RACELINE_TOLERANCE and r < closest_distance:
                        closest_distance = r; closest_obstacle = np.array([obs_x, obs_y])
            angle += self.latest_scan.angle_increment
        return closest_obstacle

    def should_attempt_overtake(self, current_pos):
        # Basic Checks
        if not self.opponent_detected or self.opponent_position is None: return False, 0.0
        if self.opponent_distance > OVERTAKE_TRIGGER_DISTANCE: return False, 0.0
        
        # Speed Check
        if (self.current_speed - self.opponent_velocity) < MIN_SPEED_ADVANTAGE: return False, 0.0
        
        # Smart Side Selection (Source B)
        left_space = self.measure_lateral_space('left')
        right_space = self.measure_lateral_space('right')
        
        if left_space < MIN_SAFE_SPACE and right_space < MIN_SAFE_SPACE: return False, 0.0
        
        if abs(left_space - right_space) > SPACE_ADVANTAGE_THRESHOLD:
            return True, OVERTAKE_LATERAL_OFFSET if left_space > right_space else -OVERTAKE_LATERAL_OFFSET
        else:
            # Tiebreaker based on opponent Y
            x, y, theta = self.current_pose
            dx = self.opponent_position[0] - x; dy = self.opponent_position[1] - y
            local_y = math.sin(-theta) * dx + math.cos(-theta) * dy
            return True, OVERTAKE_LATERAL_OFFSET if local_y > 0 else -OVERTAKE_LATERAL_OFFSET

    def measure_lateral_space(self, side):
        if self.latest_scan is None: return 0.0
        ranges = np.array(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_inc = self.latest_scan.angle_increment
        
        if side == 'left':
            a_start, a_end = math.radians(SIDE_SCAN_ANGLE_MIN), math.radians(SIDE_SCAN_ANGLE_MAX)
        else:
            a_start, a_end = math.radians(-SIDE_SCAN_ANGLE_MAX), math.radians(-SIDE_SCAN_ANGLE_MIN)
            
        dists = []
        for i, r in enumerate(ranges):
            a = angle_min + i * angle_inc
            if a_start <= a <= a_end and 0.1 < r < SIDE_SCAN_DISTANCE:
                dists.append(r)
        return np.mean(dists) if dists else 10.0 # Default to open if nothing seen

    def estimate_opponent_velocity(self):
        if len(self.opponent_position_history) < 2: return 0.0
        # Simple velocity calc
        p1, t1 = self.opponent_position_history[0]
        p2, t2 = self.opponent_position_history[-1]
        dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
        dt = t2 - t1
        return dist/dt if dt > 0 else 0.0

    # ==========================================================================
    # PART 5: MATH HELPERS (From Source A)
    # ==========================================================================

    def apply_lateral_offset(self, car_pos, target_point, offset):
        dx = target_point[0] - car_pos[0]; dy = target_point[1] - car_pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3: return target_point
        # Perpendicular vector (-dy, dx)
        return np.array([target_point[0] + offset * (-dy/dist), target_point[1] + offset * (dx/dist)])

    def compute_curvatures(self, path):
        N = len(path); curvatures = np.zeros(N)
        if N < 3: return curvatures
        for i in range(1, N - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            v12=p2-p1; v23=p3-p2; v31=p1-p3
            area=np.abs(np.cross(v12, -v31))
            den=np.linalg.norm(v12)*np.linalg.norm(v23)*np.linalg.norm(v31)
            if den > 1e-9: curvatures[i] = 2.0 * area / den
        return curvatures

    def find_lookahead_point(self, current_pos):
        if self.raceline is None: return None, 0.0
        deltas = self.raceline - current_pos
        dist_sq = np.einsum('ij,ij->i', deltas, deltas)
        closest_idx = np.argmin(dist_sq)
        n = len(self.raceline)
        for i in range(n):
            ci = (closest_idx + i) % n
            p = self.raceline[ci]
            dist = np.hypot(p[0] - current_pos[0], p[1] - current_pos[1])
            if dist >= LOOKAHEAD_DISTANCE:
                return p, self.curvatures[ci] if ci < len(self.curvatures) else 0.0
        return self.raceline[(closest_idx+n//2)%n], 0.0

    def get_yaw_from_quaternion(self, q):
        return math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))

    def stop_car(self):
        msg = AckermannDriveStamped()
        self.drive_pub.publish(msg)

    # --- Visualization Helpers ---
    def _publish_target_marker(self, lx, ly, car_x, car_y, car_theta):
        m = Marker(); m.header.frame_id='map'; m.header.stamp=self.get_clock().now().to_msg()
        m.ns='target'; m.id=0; m.type=Marker.SPHERE; m.action=Marker.ADD
        m.pose.position.x = car_x + (lx*math.cos(car_theta)-ly*math.sin(car_theta))
        m.pose.position.y = car_y + (lx*math.sin(car_theta)+ly*math.cos(car_theta))
        m.scale.x=0.3; m.scale.y=0.3; m.scale.z=0.3; m.color.a=1.0; m.color.g=1.0
        self.target_viz_pub.publish(m)

    def _publish_opponent_marker(self, x, y, detected):
        m = Marker(); m.header.frame_id='map'; m.header.stamp=self.get_clock().now().to_msg()
        m.ns='opp'; m.id=1; m.type=Marker.CUBE
        if detected:
            m.action=Marker.ADD; m.pose.position.x=x; m.pose.position.y=y; m.pose.position.z=0.15
            m.scale.x=0.5; m.scale.y=0.3; m.scale.z=0.3; m.color.a=0.9; m.color.r=1.0
        else: m.action=Marker.DELETE
        self.opponent_viz_pub.publish(m)

    def _generate_path_viz_message(self):
        if self.raceline is None: return
        msg = Path(); msg.header.frame_id=self.map_frame
        for p in self.raceline:
            ps=PoseStamped(); ps.pose.position.x=float(p[0]); ps.pose.position.y=float(p[1]); msg.poses.append(ps)
        self.path_viz_message = msg

    def _timer_publish_path_callback(self):
        if self.path_viz_pub and self.path_viz_message: self.path_viz_pub.publish(self.path_viz_message)

def main(args=None):
    rclpy.init(args=args)
    node = HybridControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop_car(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
