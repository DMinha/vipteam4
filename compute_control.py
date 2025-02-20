import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import copy
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import math
from std_msgs.msg import Bool
import numpy as np
import time
from sklearn.cluster import DBSCAN

PI = math.pi
MIN_ANGLE = math.pi/2
MAX_ANGLE = -math.pi/2
DISPARITY_THRESHOLD = 0.2
C_W = 0.3
C_L = 0.5

class DisparityExtender:
    CAR_WIDTH = 0.268
    # the min difference between adjacent LiDAR points for us to call them disparate
    DIFFERENCE_THRESHOLD = 0.05
    
    # TIME TRIAL
    # MAX_SPEED = 3.0

    # STOP SIGN
    MAX_SPEED = 0.5
    
    LINEAR_DISTANCE_THRESHOLD = 5.0
    ANGLE_CHANGE_THRESHOLD = 0.0
    ANGLE_CHANGE_SPEED = 0.5
    MAX_ANGLE = 0.8
    SLOW_SPEED = 1.0
    MAX_DISTANCE_C = 0.95
    WHEELBASE_WIDTH=0.328#0.328
    coefficient_of_friction=0.62
    gravity=9.81998
    REVERSAL_THRESHHOLD = 0.85
    SLOWDOWN_SLOPE = 0.93

    prev_angle = 0.0
    prev_index = None
    is_reversing = False

    PREV_STOP = False
    STOP_CONTROLLER = False
    STOP_TIME = -1

    def __init__(self, logger):
        self.logger = logger
    

    def preprocess_lidar(self, ranges):
        """ Any preprocessing of the LiDAR data can be done in this function.
            Possible Improvements: smoothing of outliers in the data and placing
            a cap on the maximum distance a point can be.
        """
        # remove quadrant of LiDAR directly behind us
        eighth = int(len(ranges) / 6)
        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        """ Gets the absolute difference between adjacent elements in
            in the LiDAR data and returns them in an array.
            Possible Improvements: replace for loop with numpy array arithmetic
        """
        differences = [0.]  # set first element to 0
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))
        return differences

    def get_disparities(self, differences, threshold):
        """ Gets the indexes of the LiDAR points that were greatly
            different to their adjacent point.
            Possible Improvements: replace for loop with numpy array arithmetic
        """
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)
        return disparities

    def get_num_points_to_cover(self, dist, width):
        """ Returns the number of LiDAR points that correspond to a width at
            a given distance.
            We calculate the angle that would span the width at this distance,
            then convert this angle to the number of LiDAR points that
            span this angle.
            Current math for angle:
                sin(angle/2) = (w/2)/d) = w/2d
                angle/2 = sininv(w/2d)
                angle = 2sininv(w/2d)
                where w is the width to cover, and d is the distance to the close
                point.
            Possible Improvements: use a different method to calculate the angle
        """
        # angle = 2 * np.arcsin(width / (2 * dist))
        # num_points = int(np.ceil(angle / self.radians_per_point))
        # return num_points
        # angle = 2 * np.arcsin(width / (2 * dist))
        # num_points = int(np.ceil(angle / self.radians_per_point))
        # return num_points
        angle_step=(0.25)*(math.pi/180)
        arc_length=angle_step*dist
        return int(math.ceil(self.CAR_WIDTH / arc_length))

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """ 'covers' a number of LiDAR points with the distance of a closer
            LiDAR point, to avoid us crashing with the corner of the car.
            num_points: the number of points to cover
            start_idx: the LiDAR point we are using as our distance
            cover_right: True/False, decides whether we cover the points to
                         right or to the left of start_idx
            ranges: the LiDAR points

            Possible improvements: reduce this function to fewer lines
        """
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges): break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0: break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges, car_width):
        """ For each pair of points we have decided have a large difference
            between them, we choose which side to cover (the opposite to
            the closer point), call the cover function, and return the
            resultant covered array.
            Possible Improvements: reduce to fewer lines
        """
        width_to_cover = (car_width / 2)
        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx:first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist,
                                                               width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(num_points_to_cover, close_idx,
                                       cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, angle_increment, range_len):
        """ Calculate the angle that corresponds to a given LiDAR point and
            process it into a steering angle.
            Possible improvements: smoothing of aggressive steering angles
        """
        angle = -1.57 + (range_index * angle_increment)

        if angle < -1.57:
            angle = -1.57
        elif angle > 1.57:
            angle = 1.57
        return angle

        # degrees = range_index / 3
        # angle = math.radians(degrees)
        

        # if angle < -1.57:
        #     return -1.57
        # elif angle > 1.57:
        #     return 1.57
        # return angle

        # lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        # steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))
        # return steering_angle
    
    def calculate_min_turning_radius(self,angle,forward_distance):
        # angle=abs(angle)
        # if(angle<0.0872665):#if the angle is less than 5 degrees just go as fast possible
        #     return self.MAX_SPEED
        # else:
        #     turning_radius=(self.WHEELBASE_WIDTH/math.sin(angle))
        #     maximum_velocity=math.sqrt(self.coefficient_of_friction*self.gravity*turning_radius)
        #     if(maximum_velocity<self.MAX_SPEED):
        #         maximum_velocity=maximum_velocity*(maximum_velocity/self.MAX_SPEE;D)
        #     else:
        #         maximum_velocity=self.MAX_SPEED
        # return maximum_velocity
        angle = abs(angle)
        if angle < 0.0872665:  # 5 degrees in radians
            return self.MAX_SPEED
        else:
            turning_radius = (self.WHEELBASE_WIDTH / math.sin(angle))
            maximum_velocity = math.sqrt(self.coefficient_of_friction * self.gravity * turning_radius)

            # Calculate stopping distance
            # Assuming deceleration = 0.5 * gravity (modify as per actual deceleration capabilities)
            stopping_distance = (maximum_velocity ** 2) / (2 * 0.5 * self.gravity)
            if stopping_distance > forward_distance:
                # If stopping distance exceeds forward distance, reduce maximum velocity
                # Calculate new safe velocity to stop within the forward distance
                maximum_velocity = math.sqrt(2 * 0.5 * self.gravity * forward_distance)

            if maximum_velocity < self.MAX_SPEED:
                maximum_velocity = maximum_velocity * (maximum_velocity / self.MAX_SPEED)
            else:
                maximum_velocity = self.MAX_SPEED
        
        return maximum_velocity

    def _process_lidar(self, lidar_data):
        """ Run the disparity extender algorithm!
            Possible improvements: varying the speed based on the
            steering angle or the distance to the farthest point.
        """
        ranges = lidar_data.ranges
        self.radians_per_point = (2 * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(disparities, proc_ranges,
                                              self.CAR_WIDTH)
        # max_value=max(proc_ranges)

        # if self.prev_index != None and proc_ranges[self.prev_index] > 2 and abs(curr_steering_angle - self.prev_angle) < 0.1:
        #     steering_angle = self.prev_angle
        #     max_value = proc_ranges[self.prev_index]
        #     max_index = self.prev_index
        # else:
        max_value = max(proc_ranges)
        max_index = np.argmax(proc_ranges)

        # np_ranges = np.array(proc_ranges)
        # greater_indices = np.where(np_ranges >= max_value)[0]

        # if(len(greater_indices)==1):
        #     max_index = greater_indices[0]
        # else:
        #     mid=int(len(greater_indices)/2)
        #     max_index = greater_indices[mid]

        np_ranges = np.array(proc_ranges)
        # greater_indices = np_ranges >= self.MAX_DISTANCE_C_THRESHOLD
        # greater_indices = np.where(np_ranges >= min(self.MAX_DISTANCE_C_THRESHOLD, max_value*self.MAX_DISTANCE_C))[0]
        
        greater_indices = np.where(np_ranges >= max_value*self.MAX_DISTANCE_C)[0]
        differences = np.abs(greater_indices - 360)
        max_index = greater_indices[np.argmin(differences)]
        center_index = len(proc_ranges) // 2
        greater_indices = np.where(np_ranges >= max_value * self.MAX_DISTANCE_C)[0]
        #differences = np.abs(greater_indices - center_index)
        #if len(differences) > 0:
        #    max_index = greater_indices[np.argmin(differences)]
        #else:
        #    max_index = center_index

        max_value = proc_ranges[max_index]
        
        # self.logger.info(f"greater indices: {greater_indices}, max index: {max_index}, max_value: {max_value}")

        steering_angle = self.get_steering_angle(max_index, lidar_data.angle_increment, len(proc_ranges))
        #steering_angle = self.get_steering_angle(max_index, lidar_data.angle_increment, lidar_data.angle_min)

        d_theta = abs(steering_angle)

        self.prev_angle = steering_angle
        self.prev_index = max_index

        # self.logger.info(f"Checking max_value: {max_value}, Max_index: {max_index}, Angle: {steering_angle}, Disparity: {disparities}, Ranges: {len(proc_ranges)}")
        #REVERSE
        if (self.is_reversing and max_value < 1.7) or (not self.is_reversing and max_value < 1.3):
            speed = -.4
            steering_angle = -steering_angle
            self.is_reversing = True
        # elif max_value < self.LINEAR_DISTANCE_THRESHOLD:
        #     speed = max(0.5, self.MAX_SPEED - 0.9 * self.MAX_SPEED * ((self.LINEAR_DISTANCE_THRESHOLD - max_value) / self.LINEAR_DISTANCE_THRESHOLD))
        # elif d_theta > self.ANGLE_CHANGE_THRESHOLD:
        #     speed = calculate_min_turning_radius(steering_angle, max_value)
        else:
            self.is_reversing = False
            speed_d = max(
                0.5, 
                self.MAX_SPEED -  self.MAX_SPEED * (self.SLOWDOWN_SLOPE * (self.LINEAR_DISTANCE_THRESHOLD - max_value) / self.LINEAR_DISTANCE_THRESHOLD)
            )
            speed_a = self.calculate_min_turning_radius(steering_angle, max_value)
            speed = min(speed_d, speed_a)
            min_speed = 0.5

            if max_value < 0.5:
                min_speed = 0.45
            elif max_value < 1.3:
                min_speed = 0.9
            elif max_value < 1.7:
                min_speed = 1.0
            elif max_value < 2.0:
                min_speed = 1.4
            elif max_value < 2.5:
                min_speed = 1.3
            elif max_value < 3:
                min_speed = 1.7
            else:
                min_speed = 2.2
            
            # min_speed = 1.05 * min_speed
            # min_speed = min(1.5, (max_value / 3))
            speed = max(0.5, min_speed, speed)
            # speed = max(
            #     0.6,
            #     speed
            # )

        if speed > self.MAX_SPEED:
            speed = self.MAX_SPEED
        
        # speed = max(0.5, self.MAX_SPEED - 1.2 * self.MAX_SPEED * (d_theta / self.MAX_ANGLE))
        if not self.is_reversing:
            if d_theta > ((1/10) * self.MAX_ANGLE):
                speed = max(
                    0.5,
                    (1.45 * speed) - (speed * ((d_theta - (18 * np.pi / 180)) / self.MAX_ANGLE))
                )   

            # UNTESTED
            if d_theta > (20 * np.pi / 180):
                speed *= 1.1 + ((d_theta * 180 / np.pi) / 90)

            # if abs(steering_angle) < (10 * np.pi / 180):
            #     # % faster if less than X dg change in steer
            #     speed *= 1.05

            # UNTESTED SUBSTITUTE
            # if steering_angle > ((1/10) * self.MAX_ANGLE):
            #     cand_speed = (1.2 * speed) - (speed * (steering_angle / (2*self.MAX_ANGLE))) 

            #     if cand_speed > self.MAX_SPEED:
            #         cand_speed = 0.7 * self.MAX_SPEED

            #     speed = max(
            #         0.5,
            #         cand_speed
            #     )   


        
        self.logger.info(f"speed: {speed}, max_value: {max_value}, angle change (rads): {d_theta}, steering angle: {steering_angle}")

        # if abs(steering_angle) > 20.0 / 180.0 * 3.14:
        #     speed = 1.5
        # elif abs(steering_angle) > 10.0 / 180.0 * 3.14:
        #     speed = 2.0
        # else:
        #     speed = 2.3

        # if max_value < 1.5:
        #     speed = 0.0

        return speed, steering_angle, max_value, max_index, differences, disparities, proc_ranges

    def process_observation(self, lidar_data, ego_odom):
        return self._process_lidar(lidar_data)


class ackermann_publisher(Node):

    LAST_LIDAR = None
    READY_GO = False
    COMPLETED_STOP = False
    
    FAST_MODE = False

    STOP_COUNT = 0

    def __init__(self):
        super().__init__('team_1_publisher')
        self.disparity = DisparityExtender(self.get_logger())
        self.laser_subscription = self.create_subscription(
            LaserScan,  # message type
            'scan',
            self.lidar_callback,
            10
        )
        
        self.stopsub = self.create_subscription(
            Bool,
            '/stopflag',  # Replace with your actual image topic
            self.stop_callback,
            10
        )


        self.stopready = self.create_subscription(
            Bool,
            '/stop_ack',  # Replace with your actual image topic
            self.stop_callback,
            10
        )
        
        self.publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        # time.sleep(10)

    def stop_callback(self, msg: Bool):
        ready = msg.data
        if ready:
            self.READY_GO = ready
        else:
            return
            
    def detect_box(self, ranges, angle_min, angle_increment):
        """Detects a cardboard box using LiDAR data"""

	# Define thresholds for box detection (adjust as needed)
        min_distance = 0.2   # Ignore very close objects
        max_distance = 1.2   # Only consider objects within this range
        min_width = 0.1      # Minimum width of a box (in meters)
        max_width = 1.5      # Maximum width of a box (in meters)
        max_height_variation = 0.4
        max_cluster_size = 20
        

	# Convert LiDAR data to Cartesian coordinates
        points = []
        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment
            if min_distance < r < max_distance:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])

        if len(points) < 5:  # Not enough points to form a box
            return False, None, None  

	# Convert to NumPy array
        points = np.array(points)

        # Apply DBSCAN clustering to identify objects
        clustering = DBSCAN(eps=0.15, min_samples=5).fit(points)
        labels = clustering.labels_

        unique_clusters = set(labels) - {-1}  # Remove noise (-1 is noise)

        for cluster in unique_clusters:
            cluster_points = points[labels == cluster]
            
            if len(cluster_points) > max_cluster_size:
            	continue
		
            # Calculate width using max distance between any two points in cluster
            cluster_width = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
            
            height_variation = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])

            if (min_width <= cluster_width <= max_width) and (height_variation < max_height_variation):
                centroid = np.mean(cluster_points, axis = 0)
                box_distance = np.linalg.norm(centroid)
                box_angle = np.arctan2(centroid[1], centroid[0])
                return True, box_angle, box_distance  # Box detected

        return False, None, None  # No box detected

    def lidar_callback(self, msg: LaserScan):
        # speed, angle = self.reactive_controller.calc_speed_and_angle(msg)
        # self.disparity.logger.info(f"{self.READY_GO}")
        # Detect if a box is present
        
        box_detected, box_angle, box_distance = self.detect_box(msg.ranges, msg.angle_min, msg.angle_increment)
        if box_detected:
            self.get_logger().info(f"✅ Box Detected: {box_detected}, Box Distance: {box_distance}")
            # Safe distance to be from the box
            SAFE_DISTANCE = 0.7
            if box_distance > SAFE_DISTANCE: 
                steering_angle = np.sign(box_angle) * abs(box_angle)  # Ensure the sign is correct
                speed = 0.5 #slow down to follow box
            else: 
            	#stop at the box
            	self.get_logger().info("🛑 Box is too close! Stopping car.")
            	steering_angle = 0.0
            	speed = 0.0
        else:
            self.get_logger().info(f"No box, using disparity!")
            try:
            	speed, steering_angle, max_value, max_index, differences, disparities, proc_ranges = self.disparity._process_lidar(msg)
            except Exception as e:
            	self.get_logger().error(f"Error in disparity: {e}")
            	speed = 0.0
            	steering_angle = 0.0
        stamped_msg = AckermannDriveStamped()
        stamped_msg.drive = AckermannDrive()
        stamped_msg.drive.steering_angle = steering_angle
        stamped_msg.drive.speed = float(speed)  # Ensure speed is a float
        self.publisher.publish(stamped_msg)
        '''
        if True:
            self.LAST_LIDAR = msg
            if (not self.disparity.STOP_CONTROLLER):
                #speed, angle, max_value, max_index, differences, disparities, proc_ranges = self.disparity._process_lidar(msg)
                stamped_msg = AckermannDriveStamped()
                stamped_msg.drive = AckermannDrive()
                stamped_msg.drive.steering_angle = steering_angle
                stamped_msg.drive.speed = speed

                # if not self.READY_GO:
                #     stamped_msg.drive.speed = 0.

                self.publisher.publish(stamped_msg)
            else:
                return
        
        else:
            speed, angle, max_value, max_index, differences, disparities, proc_ranges = self.disparity._process_lidar(msg)
            stamped_msg = AckermannDriveStamped()
            stamped_msg.drive = AckermannDrive()
            stamped_msg.drive.steering_angle = angle
            stamped_msg.drive.speed = speed

            # if not self.READY_GO:
            #     stamped_msg.drive.speed = 0.

            self.publisher.publish(stamped_msg)
        '''

    def stop_callback(self, msg: Bool):
        self.disparity.logger.info(f"flag: {msg.data}")

        if not self.FAST_MODE:
            now = time.time()
            detected = msg.data

            # if detected:
            #     self.STOP_COUNT += 1

            # if self.STOP_COUNT >= 4:
            #     if self.disparity.STOP_TIME == -1:
            #         # haven't stopped on a callback yet
            #         self.disparity.logger.info("waiting !!")
            #         self.disparity.STOP_TIME = time.time()

            #     else:
            #         # already stopped
            #         if now - self.disparity.STOP_TIME >= 2.3:
            #             self.disparity.logger.info("uno segundo !!")
            #             self.disparity.STOP_TIME = -1
            #             self.disparity.PREV_STOP = False
            #             self.disparity.STOP_CONTROLLER = False
            #             self.STOP_COUNT = 0


            if detected and not self.disparity.PREV_STOP:
                # first detection, alternative controller
                self.disparity.PREV_STOP = True
                self.disparity.STOP_CONTROLLER = True

            elif not detected:
                # no longer in viewing range but already detected; let's stop for 1 second
                
                if self.disparity.PREV_STOP and self.disparity.STOP_TIME == -1:
                    # haven't stopped on a callback yet
                    self.disparity.logger.info("waiting !!")
                    self.disparity.STOP_TIME = time.time()
                    # self.disparity.PREV_STOP = False
                    stamped_msg = AckermannDriveStamped()
                    stamped_msg.drive = AckermannDrive()
                    stamped_msg.drive.steering_angle = 0.
                    # stamped_msg.drive.speed = speed
                    stamped_msg.drive.speed = 0.
                    self.publisher.publish(stamped_msg)

                elif self.disparity.PREV_STOP:
                    stamped_msg = AckermannDriveStamped()
                    stamped_msg.drive = AckermannDrive()
                    stamped_msg.drive.steering_angle = 0.
                    # stamped_msg.drive.speed = speed
                    stamped_msg.drive.speed = 0.
                    self.publisher.publish(stamped_msg)
                    
                    # already stopped
                    if now - self.disparity.STOP_TIME >= 1.5:
                        self.disparity.logger.info(f"{now - self.disparity.STOP_TIME} uno segundo !!")
                        self.disparity.STOP_TIME = -1
                        self.disparity.PREV_STOP = False
                        self.disparity.STOP_CONTROLLER = False

            if self.disparity.STOP_CONTROLLER and self.disparity.STOP_TIME == -1:
                self.disparity.logger.info("stop controller")
                if self.LAST_LIDAR:
                    speed, angle, max_value, max_index, differences, disparities, proc_ranges = self.disparity._process_lidar(self.LAST_LIDAR)
                    stamped_msg = AckermannDriveStamped()
                    stamped_msg.drive = AckermannDrive()
                    stamped_msg.drive.steering_angle = angle
                    # stamped_msg.drive.speed = speed
                    stamped_msg.drive.speed = 0.5
                else:
                    stamped_msg = AckermannDriveStamped()
                    stamped_msg.drive = AckermannDrive()
                    stamped_msg.drive.steering_angle = 0.
                    # stamped_msg.drive.speed = speed
                    stamped_msg.drive.speed = 0.3

                # if not self.READY_GO:
                #     stamped_msg.drive.speed = 0.

                self.publisher.publish(stamped_msg)



def main(args=None):
    rclpy.init(args=args)
    ackermann_publisher_i = ackermann_publisher()
    rclpy.spin(ackermann_publisher_i)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ackermann_publisher_i.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
