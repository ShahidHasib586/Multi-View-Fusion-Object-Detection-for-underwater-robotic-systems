from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'multicam_fusion_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shahid',
    maintainer_email='shahid@todo.todo',
    description='Multi-camera fusion detector (enhance + yolo)',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multicam_fusion_detector_node = multicam_fusion_detector.node:main',
            'video_publisher = multicam_fusion_detector.video_publisher:main'
        ],
    },
)
