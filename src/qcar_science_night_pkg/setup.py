from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'qcar_science_night_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='mohammadafreed.dudekula@tu-ilmenau.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_mpc = qcar_science_night_pkg.path_mpc_node:main',
            'lidar_overtake = qcar_science_night_pkg.lidar_overtake_node:main',
            'v2v_receiver = qcar_science_night_pkg.v2v_receiver_node:main',
            'lane_centering_node = qcar_science_night_pkg.lane_centering_node:main',
            'depth_emergency_node = qcar_science_night_pkg.depth_emergency_node:main',
            'reverse_trailer_mpc = qcar_science_night_pkg.qcar_reverse_trailer_mpc:main',
            'sound_node = qcar_science_night_pkg.sound_node:main',
            'amcl_nudge = qcar_science_night_pkg.amcl_nudge:main',
            'trajectory_initial_pose = qcar_science_night_pkg.trajectory_initial_pose:main',
            'validate_path_map = qcar_science_night_pkg.path_map_validator:main',
            'trajectory_recorder = qcar_science_night_pkg.trajectory_recorder:main',
        ],
    },
)
