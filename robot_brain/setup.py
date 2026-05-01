from setuptools import setup
import os
from glob import glob

package_name = 'robot_brain'

setup(
    name=package_name,
    version='0.0.1',
    # 包含主包和 core 子包
    packages=[package_name, 'robot_brain.core'], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Brandon',
    maintainer_email='brandon@nus.edu.sg',
    description='Lifelong Learning Framework',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # === 1. 核心主控 (旧版) ===
            'orchestrator = robot_brain.orchestrator:main',
            
            # === 2. 神经网络控制器 (新版 - 实机闭环) ===
            'neural_controller_trajectory = robot_brain.neural_controller_trajectory:main',
            'topo_controller = robot_brain.topo_controller:main',
            'hybrid_sim = robot_brain.hybrid_phantom_sim:main',
            
            # === 3. 交互与规划工具 ===
            'target_generator = robot_brain.target_generator:main',
            # 🔥 新增动态避障目标生成器
            'dynamic_target = robot_brain.dynamic_target:main',

            # === 4. 离线训练与推理 ===
            'train_offline = robot_brain.train_offline:train_offline',
            'test_brain = robot_brain.inference:main', 
            
            # === 5. 工具链 (数据处理) ===
            'calibrate = robot_brain.core.latency_calib:main',
            'babble = robot_brain.core.babbling_node:main',
            'clean = robot_brain.core.data_cleaner:main',
            'feature = robot_brain.core.feature_eng:main',
        ],
    },
)