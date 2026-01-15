from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    strategy = LaunchConfiguration('strategy')
    fixed_window = LaunchConfiguration('fixed_window')
    policy_path = LaunchConfiguration('policy_path')

    return LaunchDescription([
        DeclareLaunchArgument('strategy', default_value='baseline_fw'),
        DeclareLaunchArgument('fixed_window', default_value='16'),
        DeclareLaunchArgument('policy_path', default_value=''),

        Node(package='aswrl_sync', executable='sync_client',
             name='sync_client',
             parameters=[{
                 'strategy': strategy,
                 'fixed_window': fixed_window,
                 'policy_path': policy_path,
                 'sync_dt': 0.1
             }]),

        Node(package='aswrl_sync', executable='compensation_node',
             name='compensation_node',
             parameters=[{'input_odom': 'odom_local', 'output_odom': 'odom_synced'}]),
    ])
