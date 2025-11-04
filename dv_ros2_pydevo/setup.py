import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dv_ros2_pydevo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Richeek Das',
    maintainer_email='richeek@seas.upenn.edu',
    description='A Python ROS2 node to run DEVO',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'devo_runner = dv_ros2_pydevo.devo_runner:main',
        ],
    },
)
