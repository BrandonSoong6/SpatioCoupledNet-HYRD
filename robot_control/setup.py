from setuptools import setup
import os
from glob import glob

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    # 包含所有子模块
    packages=[
        package_name,
        package_name + '.algorithms',
        package_name + '.drivers',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 安装 Launch 文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        
        # 【关键】安装 data 文件夹下的所有数据文件 (.npy 和 .npz)
        (os.path.join('share', package_name, 'data'), glob('data/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Brandon',
    maintainer_email='brandon@nus.edu.sg',
    description='StatSculpt-10D: Motor Babbling for HYRD Robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = robot_control.vision_node:main',
            'teleop_keyboard = robot_control.teleop_keyboard:main',
            'babbling_player = robot_control.babbling_player:main',
            'latency_calibrator = robot_control.latency_calibrator:main',
            'xbox_teleop = robot_control.xbox_teleop:main',  # <-- 新增的 Xbox 控制节点
        ],
    },
)