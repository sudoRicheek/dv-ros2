import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    return launch.LaunchDescription([
        ComposableNodeContainer(
            name='dv_ros2_monodepth',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='dv_ros2_monodepth',
                    plugin='torch_converter::TorchConverter',
                    name='live_torch_converter_component',
                    parameters=[{
                        'input_topic': '/events',
                        'output_topic': '/depth',
                        'frame_rate': 50.0,
                        'f3_pt2_path': '/home/neurofly/fast-feature-fields/_aoti_pt2/f3_m3edcarfalcon_vga_480x640_aoti.pt2',
                        'dav2_pt2_path': '/home/neurofly/fast-feature-fields/_aoti_pt2/dav2b_m3edcarfalcon_vga_308x406_aoti.pt2',
                        # 'dav2_pt2_path': '/home/neurofly/fast-feature-fields/_aoti_pt2/dav2s_m3edcarfalcon_vga_308x406_aoti.pt2',
                        'ev_width': 640,
                        'ev_height': 480,
                        'ev_time': 20, # in milliseconds
                        'events_kept': 200_000,
                        'dav2_height': 308,
                        'dav2_width': 406}],
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
                ComposableNode(
                    package='dv_ros2_unified',
                    plugin='dv_ros2_unified::Visualizer',
                    name='live_visualizer_component',
                    parameters=[{
                        'image_topic': '/image_event_visualizer',
                        'frame_rate': 50.0,
                        'background_color_r': 255,
                        'background_color_g': 255,
                        'background_color_b': 255,
                        'positive_event_color_r': 255,
                        'positive_event_color_g': 0,
                        'positive_event_color_b': 0,
                        'negative_event_color_r': 0,
                        'negative_event_color_g': 0,
                        'negative_event_color_b': 255,}],
                    extra_arguments=[{'use_intra_process_comms': True}]),
            ],
            output='both',
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            output='screen',
            emulate_tty=True,
        )
    ])

