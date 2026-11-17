from setuptools import setup
from glob import glob

package_name = 'roman_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.*')),
        ('share/' + package_name + '/cfg/d455', glob('cfg/d455/*.yaml')),
        ('share/' + package_name + '/cfg/zed2i', glob('cfg/zed2i/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'transforms3d'
        # TODO: do these actually get installed by being here?
    ],
    zip_safe=True,
    maintainer='masonbp',
    maintainer_email='mbpeterson70@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fastsam_node.py = roman_ros2.fastsam_node:main',
            'roman_map_node.py = roman_ros2.roman_map_node:main',
            'roman_align_node.py = roman_ros2.roman_align_node:main',
            'roman_loop_closure_node.py = roman_ros2.roman_loop_closure_node:main',
            'lc_viz_node.py = roman_ros2.lc_viz_node:main',
            'meridian_rpgo_relocalization_node.py = roman_ros2.meridian_rpgo_relocalization_node:main'
        ],
    },
)
