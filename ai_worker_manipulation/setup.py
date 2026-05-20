from setuptools import find_packages, setup

package_name = 'ai_worker_manipulation'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hamin',
    maintainer_email='chlgkals0730@gmail.com',
    description='Manipulation stack for the 2026 Humanoid Challenge',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'test_move_to_pose = ai_worker_manipulation.tests.test_move_to_pose:main',
            'move_home = ai_worker_manipulation.tests.move_home:main',
            'move_to_pose = ai_worker_manipulation.tests.move_to_pose:main',
            'demo_0513 = ai_worker_manipulation.tests.demo_0513:main',
            'gpd_dual_view = ai_worker_manipulation.tests.gpd_dual_view_node:main',
        ],
    },
)
