from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    vesc_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'vesc.yaml'
    )
    sensors_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'sensors.yaml'
    )
    mux_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'mux.yaml'
    )

    vesc_la = DeclareLaunchArgument(
        'vesc_config',
        default_value=vesc_config,
        description='Descriptions for vesc configs')
    sensors_la = DeclareLaunchArgument(
        'sensors_config',
        default_value=sensors_config,
        description='Descriptions for sensor configs')
    mux_la = DeclareLaunchArgument(
        'mux_config',
        default_value=mux_config,
        description='Descriptions for ackermann mux configs')

    ld = LaunchDescription([vesc_la, sensors_la, mux_la])

    ackermann_to_vesc_node = Node(
        package='vesc_ackermann',
        executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]
    )
    ackermann_mux_node = Node(
        package='ackermann_mux',
        executable='ackermann_mux',
        name='ackermann_mux',
        output="screen",
        parameters=[LaunchConfiguration('mux_config')],
        remappings=[('ackermann_drive_out', 'ackermann_cmd')]
    )
    vesc_to_odom_node = Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]  # This will use your vesc.yaml file
    )
    

    lidar_node = Node(
         package='urg_node',
         executable='urg_node_driver',
        name='urg_node',
        parameters=[LaunchConfiguration('sensors_config')]
     )

    
    perception_node = Node(
        package='reactive_racing',
        executable='perception',
        name='perception',
        output="screen",
    )
    ctrl_node = Node(
        package='reactive_racing',
        executable='control',
        name='control',
        output="screen",
    )

    zed2node = Node(
        package='reactive_racing',
        executable='zed2_custom_node',
        name='zed2_custom_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'camera_resolution': 'HD720'},
            {'camera_fps': 30},
            {'depth_mode': 'ULTRA'}
        ]
    )

    joystick_driver_node = Node(
        package='joy',  # ✅ Standard joystick driver
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    # ✅ Joy Teleop Node (Maps joystick inputs to commands)
    config_file = os.path.join(
        get_package_share_directory('joy_teleop'),  # Replace with your package name
        'config',
        'joy_teleop_example.yaml'
    )   
    joystick_teleop_node = Node(
        package='joy_teleop',  
        executable='joy_teleop',
        name='joy_teleop',
        output='screen',
        parameters=[config_file]  # Update this path!
    )
    joystick_ackermann_node = Node(
        package='reactive_racing',  # Replace with the actual package name
        executable='joystick_ackermann_teleop',  # Make sure this matches your node's executable name
        name='joystick_ackermann_teleop',
        output='screen',
        parameters=[{
            'axis_speed': 1,  # Adjust based on your joystick
            'axis_steering': 0,
            'scale_speed': 1.0,
            'scale_steering': 0.5,
            'deadzone': 0.1,
            'ackermann_topic': 'ackermann_cmd'
        }]
    )
    
    
    # Add nodes to LaunchDescription
    ld.add_action(ackermann_to_vesc_node)
    ld.add_action(vesc_driver_node)
    ld.add_action(ackermann_mux_node)
    ld.add_action(lidar_node)  # Commented out original lidar node
    #ld.add_action(racecar_controller_node)  # Added new lidar processing node
    #ld.add_action(vesc_to_odom_node)
    ld.add_action(perception_node)
    ld.add_action(zed2node)
    #ld.add_action(joystick_driver_node)
    #ld.add_action(joystick_teleop_node)
    #ld.add_action(joystick_ackermann_node)
    ld.add_action(TimerAction(
       period=2.0,
       actions=[ctrl_node]
    ))

    return ld
    
    
    
'''
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    default_track_filename = 'map.p'
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    default_track_path = os.path.join(launch_file_dir, default_track_filename)
    track_path_la = DeclareLaunchArgument(
        'track_path',
        default_value=default_track_path, 
        description=f'Path to the track file (default: {default_track_path})'
    )
    
    vesc_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'vesc.yaml'
    )
    sensors_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'sensors.yaml'
    )
    mux_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'mux.yaml'
    )

    vesc_la = DeclareLaunchArgument(
        'vesc_config',
        default_value=vesc_config,
        description='Descriptions for vesc configs')
    sensors_la = DeclareLaunchArgument(
        'sensors_config',
        default_value=sensors_config,
        description='Descriptions for sensor configs')
    mux_la = DeclareLaunchArgument(
        'mux_config',
        default_value=mux_config,
        description='Descriptions for ackermann mux configs')

    ld = LaunchDescription([vesc_la, sensors_la, mux_la])

    ackermann_to_vesc_node = Node(
        package='vesc_ackermann',
        executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        output="screen",
        parameters=[LaunchConfiguration('vesc_config')]
    )
    ackermann_mux_node = Node(
        package='ackermann_mux',
        executable='ackermann_mux',
        name='ackermann_mux',
        output="screen",
        parameters=[LaunchConfiguration('mux_config')],
        remappings=[('ackermann_drive_out', 'ackermann_cmd')]
    )
    

    lidar_node = Node(
         package='urg_node',
         executable='urg_node_driver',
        name='urg_node',
        parameters=[LaunchConfiguration('sensors_config')]
     )

    
    perception_node = Node(
        package='reactive_racing',
        executable='perception',
        name='perception',
        output="screen",
    )
    ctrl_node = Node(
        package='reactive_racing',
        executable='control', # This matches the entry point in setup.py
        name='pure_pursuit_controller',
        output="screen",
        parameters=[{
            'wheelbase': 0.33,
            'lookahead_base': 0.4,
            'lookahead_gain': 0.2,
            'track_file_path': LaunchConfiguration('track_path'),
            'control_frequency': 50.0,
            'target_topic': '/drive', # <<< CHANGED TO MATCH mux.yaml 'navigation' topic
            'odom_topic': '/odom' # <<< TOPIC WHERE SLAM/LOCALIZATION PUBLISHES Odometry
        }]
    )

    zed2node = Node(
        package='reactive_racing',
        executable='zed2_custom_node',
        name='zed2_custom_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'camera_resolution': 'HD720'},
            {'camera_fps': 30},
            {'depth_mode': 'ULTRA'}
        ]
    )

    joystick_driver_node = Node(
        package='joy',  # ✅ Standard joystick driver
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    # ✅ Joy Teleop Node (Maps joystick inputs to commands)
    config_file = os.path.join(
        get_package_share_directory('joy_teleop'),  # Replace with your package name
        'config',
        'joy_teleop_example.yaml'
    )   
    joystick_teleop_node = Node(
        package='joy_teleop',  
        executable='joy_teleop',
        name='joy_teleop',
        output='screen',
        parameters=[config_file]  # Update this path!
    )
    joystick_ackermann_node = Node(
        package='reactive_racing',  # Replace with the actual package name
        executable='joystick_ackermann_teleop',  # Make sure this matches your node's executable name
        name='joystick_ackermann_teleop',
        output='screen',
        parameters=[{
            'axis_speed': 1,  # Adjust based on your joystick
            'axis_steering': 0,
            'scale_speed': 1.0,
            'scale_steering': 0.5,
            'deadzone': 0.1,
            'ackermann_topic': 'ackermann_cmd'
        }]
    )
    
    # Add nodes to LaunchDescription
    ld.add_action(ackermann_to_vesc_node)
    ld.add_action(vesc_driver_node)
    ld.add_action(ackermann_mux_node)
    ld.add_action(lidar_node)  # Commented out original lidar node
    #ld.add_action(racecar_controller_node)  # Added new lidar processing node
    ld.add_action(perception_node)
    ld.add_action(zed2node)
    ld.add_action(joystick_driver_node)
    ld.add_action(joystick_teleop_node)
    ld.add_action(joystick_ackermann_node)
    ld.add_action(TimerAction(
       period=1.0,
       actions=[ctrl_node]
    ))

    return ld
    '''
