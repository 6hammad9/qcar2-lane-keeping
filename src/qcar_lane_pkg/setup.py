from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'qcar_lane_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = qcar_lane_pkg.camera_node:main',
            'lane_keeping_node = qcar_lane_pkg.lane_keeping_node:main',
            'driver_node=qcar_lane_pkg.driver_node:main',
            'pure_pursuit_node= qcar_lane_pkg.pure_pursuit_node:main',
            'qcar2_odom_node = qcar_lane_pkg.qcar2_odom_node:main',
            'pure_pursuit_path_publisher = qcar_lane_pkg.pure_pursuit_path_publisher:main',
            'object_detection_node = qcar_lane_pkg.object_detetion:main'
        ],
    },
)
