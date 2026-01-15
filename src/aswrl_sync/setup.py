from setuptools import setup
from glob import glob
import os

package_name = 'aswrl_sync'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f"{package_name}.rl"],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/gazebo/worlds', glob('gazebo/worlds/*.world')),
        ('share/' + package_name + '/gazebo/models/dog1', glob('gazebo/models/dog1/*')),
        ('share/' + package_name + '/gazebo/models/dog2', glob('gazebo/models/dog2/*')),
        ('share/' + package_name + '/gazebo/models/uav1', glob('gazebo/models/uav1/*')),
        ('share/' + package_name + '/config', glob('config/*.xml')),
        ('share/' + package_name + '/config/net_scenarios', glob('config/net_scenarios/*.yaml')),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh')),
        ('share/' + package_name + '/README', ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Butterfly Dark',
    maintainer_email='woodward_christopher59484@hotmail.com',
    description='ASW-RL adaptive sync window (DQN) + Kalman clock sync + netns/netem network emulation.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sync_master = aswrl_sync.sync_master:main',
            'sync_client = aswrl_sync.sync_client:main',
            'compensation_node = aswrl_sync.compensation_node:main',
            'netem_controller = aswrl_sync.netem_controller:main',
            'formation_controller = aswrl_sync.formation_controller:main',
            'dummy_odom = aswrl_sync.dummy_odom:main',
            # RL tools (non-ROS)
            'train_dqn = aswrl_sync.rl.train_dqn:main',
            'export_policy = aswrl_sync.rl.export_policy:main',
        ],
    },
)
