from setuptools import setup

package_name = 'roman_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/cfg', ['cfg/default_fastsam.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='masonbp',
    maintainer_email='mbpeterson70@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fastsam_node.py = roman_ros2.fastsam_node:main',
            'roman_map_node.py = roman_ros2.roman_map_node:main'
        ],
    },
)
