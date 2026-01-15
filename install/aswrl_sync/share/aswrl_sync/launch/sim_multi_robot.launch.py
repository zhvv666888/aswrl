from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare('aswrl_sync')
    world = PathJoinSubstitution([pkg_share, 'gazebo', 'worlds', 'multi_robot.world'])
    model_path = PathJoinSubstitution([pkg_share, 'gazebo', 'models'])

    # Make Gazebo find our models
    set_model_path = SetEnvironmentVariable(name='GAZEBO_MODEL_PATH', value=model_path)

    gzserver = ExecuteProcess(
        cmd=['gazebo', '--verbose', world],
        output='screen'
    )

    return LaunchDescription([
        set_model_path,
        gzserver,
    ])
