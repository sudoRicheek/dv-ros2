#!/usr/bin/env python3
"""
Launch file for DEVO runner node
"""

import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name='devo_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='dv_ros2_incremental_voxelgrid',
                    plugin='voxelgrid_converter::VoxelgridConverter',
                    name='live_voxelgrid_converter_component',
                    parameters=[{
                        'input_topic': '/events',
                        'output_topic': '/voxelgrid',
                        'frame_rate': 20.0,
                        'ev_width': 640,
                        'ev_height': 480,
                        'events_kept': 150_000,
                        'time_bins': 5}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                ComposableNode(
                    package='dv_ros2_unified',
                    plugin='dv_ros2_unified::Capture',
                    name='live_capture_component',
                    parameters=[{
                        'time_increment': 1000,
                        'frames': True,
                        'events': True,
                        'imu': True,
                        'triggers': True,
                        # 'camera_name': "default_evcam_name",
                        # 'aedat4_file_path': "",
                        # 'camera_calibration_file_path': "",
                        # 'camera_frame_name': "camera",
                        # 'imu_frame_name': "imu",
                        # 'transform_imu_to_camera_frame': True,
                        # 'unbiased_imu_data': True,
                        # 'noise_filtering': False,
                        # 'noise_ba_time': 2000,
                        # 'wait_for_sync': False,
                        # 'global_hold': False,
                        'bias_sensitivity': 1}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
                # ComposableNode(
                #     package='dv_ros2_unified',
                #     plugin='dv_ros2_unified::Visualizer',
                #     name='live_visualizer_component',
                #     parameters=[{
                #         'image_topic': '/image_event_visualizer',
                #         'frame_rate': 50.0,
                #         'background_color_r': 255,
                #         'background_color_g': 255,
                #         'background_color_b': 255,
                #         'positive_event_color_r': 255,
                #         'positive_event_color_g': 0,
                #         'positive_event_color_b': 0,
                #         'negative_event_color_r': 0,
                #         'negative_event_color_g': 0,
                #         'negative_event_color_b': 255,}],
                #     extra_arguments=[{'use_intra_process_comms': True}]),
            ],
            output='both',
        ),
        # Node(
        #     package='rqt_image_view',
        #     executable='rqt_image_view',
        #     name='rqt_image_view',
        #     output='screen',
        #     emulate_tty=True,
        # ),
        Node(
            package='dv_ros2_pydevo',
            executable='devo_runner',
            name='live_devo_runner_component',
            parameters=[{
                'voxelgrid_topic': '/voxelgrid',
                'time_bins': 5,
                'height': 480,
                'width': 640,
                'frequency': 5.0,
                'model_path': '/home/neurofly/dvs_ws/src/dv-ros2/dv_ros2_pydevo/dv_ros2_pydevo/ckpt/DEVO.pth',
                'model_cfg': '/home/neurofly/dvs_ws/src/dv-ros2/dv_ros2_pydevo/dv_ros2_pydevo/config/cfg.yaml',
                'intrinsics': [683.0, 685.0, 302.0, 217.0]
            }],
            output='screen',
            emulate_tty=True,
        ),
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     output='screen'
        # ),
    ])
