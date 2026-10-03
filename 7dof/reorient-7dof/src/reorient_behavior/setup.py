import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'reorient_behavior'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nishrraj',
    maintainer_email='nishrraj@todo.todo',
    description='Section B: collaborative two-arm in-place reorientation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'task_node = reorient_behavior.task_node:main',
        'verify_node = reorient_behavior.verify_node:main',
    ]},
)
