"""haiying_mission 包安装配置（ament_python）"""
import os
from glob import glob

from setuptools import setup

package_name = 'haiying_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        (os.path.join('share', package_name, 'docs'),
         glob('docs/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='决策组',
    maintainer_email='decision@haiying.local',
    description='唯一任务状态机（六态 + 3 秒目标看门狗）',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mission_fsm_node = haiying_mission.mission_fsm_node:main',
        ],
    },
)
