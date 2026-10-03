import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'perceive_behavior'

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
    description='Section C: perceive each object, decide, stand it up with both arms.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'perception_node = perceive_behavior.perception_node:main',
        'task_node = perceive_behavior.task_node:main',
        'verify_node = perceive_behavior.verify_node:main',
    ]},
)
