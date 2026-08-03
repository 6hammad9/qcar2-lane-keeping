from setuptools import find_packages, setup

package_name = 'rosbot_lane'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='husarion',
    maintainer_email='husarion@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lane_keeping_node = rosbot_lane.lane_keeping_node:main',
            'straight_lane_node = rosbot_lane.straight_lane_node:main',
            'rosbot_v2v_broadcaster = rosbot_lane.rosbot_v2v_broadcaster:main',
            'rosbot_v2v_gate = rosbot_lane.rosbot_v2v_gate:main',
        ],
    },
)
