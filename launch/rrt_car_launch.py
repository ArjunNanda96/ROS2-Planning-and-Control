import os
from math import radians
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="lab7_pkg",
            executable="rrt_node.py",
            name="rrt_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"rrt_lookahead_distance": 3.0},
                {"pursuit_lookahead_distance": 1.2},
                {"max_iter": 200},
                {"max_steer_dist": 0.5},
                {"goal_dist_thresh": 0.5},
                {"grid_resolution": 0.05},
                {"grid_width": 2.0},
                {"grid_length": 3.0},
                {"waypoint_distance": 0.1},
                {"steering_angle_factor": 0.7},
                {"speed_factor": 1.0},
                {"desired_speed": 1.0},
                {"min_speed": 0.3},
                {"sparse_filename": 'real_sparse_new_1'},
            ]
        )
    ])