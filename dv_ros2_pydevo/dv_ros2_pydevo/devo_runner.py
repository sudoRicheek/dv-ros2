"""
DEVO Runner Node - ROS2 node for running DEVO visual odometry on voxelgrid data
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

# import cv2
import torch

from .devo.devo import DEVO
from .devo.config import cfg


class DEVO_rclpy(Node):
    """
    ROS2 node that subscribes to voxelgrid topics and runs DEVO model
    at a fixed frequency for visual odometry estimation.
    """

    def __init__(self):
        super().__init__("devo_runner")

        # Declare parameters
        self.declare_parameter("voxelgrid_topic", "/voxelgrid")
        self.declare_parameter("frequency", 5.0)  # Default 5 Hz
        self.declare_parameter("model_path", "")
        self.declare_parameter("model_cfg", "")

        # Parameters for voxelgrid dimensions
        self.declare_parameter("time_bins", 5)
        self.declare_parameter("height", 480)
        self.declare_parameter("width", 640)

        # Intrinsics
        self.declare_parameter("intrinsics", [683.0, 685.0, 302.0, 217.0])

        # Get parameters
        self.voxelgrid_topic = self.get_parameter("voxelgrid_topic").get_parameter_value().string_value
        self.frequency = self.get_parameter("frequency").get_parameter_value().double_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.model_cfg = self.get_parameter("model_cfg").get_parameter_value().string_value

        self.time_bins = self.get_parameter("time_bins").get_parameter_value().integer_value
        self.height = self.get_parameter("height").get_parameter_value().integer_value
        self.width = self.get_parameter("width").get_parameter_value().integer_value

        self.intrinsics = self.get_parameter("intrinsics").get_parameter_value().double_array_value
        self.intrinsics = torch.tensor(self.intrinsics, dtype=torch.float32).to(device='cuda', non_blocking=True)
        print(f"Using intrinsics: {self.intrinsics}")

        self.latest_voxelgrid = None

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=100
        )

        # Subscribe to voxelgrid topic
        self.voxelgrid_subscriber = self.create_subscription(
            Float32MultiArray, self.voxelgrid_topic, self.voxelgrid_callback, qos_profile
        )

        # Publisher for pose
        self.pose_publisher = self.create_publisher(PoseStamped, '/devo_pose', qos_profile)

        # DEVO model (placeholder - to be initialized later)
        self.slam = None
        self.timestamp_us = 0 # timestamp passed to DEVO in microseconds
        self.microseconds_per_frame = int(1e6 / self.frequency)

        # Statistics
        self.frame_count = 0
        self.last_process_time = self.get_clock().now()

        # Initialize DEVO model if model_path is provided
        self.initialize_devo_model()

        self.get_logger().info(f"DEVO runner initialized")
        self.get_logger().info(f"  Voxelgrid topic: {self.voxelgrid_topic}")
        self.get_logger().info(f"  Processing frequency: {self.frequency} Hz")
        self.get_logger().info(
            f'  Model path: {self.model_path if self.model_path else "Not specified"}'
        )

    @torch.no_grad()
    def voxelgrid_callback(self, msg):
        """
        Callback for receiving voxelgrid data.
        Stores the latest voxelgrid in a thread-safe manner.

        Args:
            msg: Voxelgrid message a Float32MultiArray
        """
        self.latest_voxelgrid = self.extract_voxelgrid(msg)

        try:
            self.frame_count += 1
            self.timestamp_us += self.microseconds_per_frame

            pose = self.slam(self.timestamp_us, self.latest_voxelgrid, self.intrinsics)
            #! They do 12 updates at the end of the trajectory to refine the pose estimate

            # Publish pose
            self.publish_pose(pose)

            self.get_logger().info(
                f"Frame {self.frame_count} | Timestamp: {self.timestamp_us} us | Pose: {pose.numpy()}"
            )

            # if self.frame_count % 10 == 0:
            #     current_time = self.get_clock().now()
            #     elapsed = (current_time - self.last_process_time).nanoseconds / 1e9
            #     actual_freq = 10.0 / elapsed if elapsed > 0 else 0.0
            #     self.get_logger().info(
            #         f"Received {self.frame_count} frames | "
            #         f"Actual freq: {actual_freq:.2f} Hz | "
            #         f"Target freq: {self.frequency} Hz"
            #     )
            #     self.last_process_time = current_time

                # vg_img = self.latest_voxelgrid.norm(p=2, dim=0).cpu().numpy()
                # vg_img = (vg_img - vg_img.min()) / (vg_img.max() - vg_img.min() + 1e-6) * 255.0
                # vg_img = vg_img.astype('uint8')
                # cv2.imwrite(f"voxelgrid_{self.frame_count}.png", vg_img)

        except Exception as e:
            self.get_logger().error(f"Error processing voxelgrid: {str(e)}")

    def extract_voxelgrid(self, msg):
        """
        Extract voxelgrid float32multiarray from message and move to GPU.

        Args:
            msg: Voxelgrid Float32MultiArray message
        """
        # Assuming msg.data is a flat list representing the voxelgrid
        voxelgrid_gpu = torch.tensor(msg.data, dtype=torch.float32).view(
            self.time_bins, self.height, self.width
        ).to(device='cuda', non_blocking=True)

        self.get_logger().info("Voxelgrid extracted and moved to GPU")

        return voxelgrid_gpu

    def publish_pose(self, pose_tensor):
        """
        Publish pose as a PoseStamped message.
        
        Args:
            pose_tensor: Tensor of size 7 containing [x, y, z, qx, qy, qz, qw]
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        # Position (first 3 elements)
        pose_msg.pose.position.x = float(pose_tensor[0])
        pose_msg.pose.position.y = float(pose_tensor[1])
        pose_msg.pose.position.z = float(pose_tensor[2])

        # Orientation as quaternion (last 4 elements)
        pose_msg.pose.orientation.x = float(pose_tensor[3])
        pose_msg.pose.orientation.y = float(pose_tensor[4])
        pose_msg.pose.orientation.z = float(pose_tensor[5])
        pose_msg.pose.orientation.w = float(pose_tensor[6])

        self.pose_publisher.publish(pose_msg)

    def initialize_devo_model(self):
        """
        Initialize DEVO model from model_path.
        This should be called when model_path is set.
        """
        if not self.model_path:
            self.get_logger().error("No model path specified, DEVO model not loaded")
            return False

        try:
            config = cfg
            config.merge_from_file(self.model_cfg)
            self.get_logger().info(f"Loading DEVO model from {self.model_path} with config {self.model_cfg}")

            self.slam = DEVO(
                config,
                self.model_path,
                evs=True,
                ht=self.height,
                wd=self.width,
                viz=False,
                viz_flow=False
            )

            self.get_logger().info(f"DEVO model loaded from {self.model_path} with config {self.model_cfg}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load DEVO model: {str(e)}")
            return False


def main(args=None):
    """Main entry point for the DEVO runner node."""
    rclpy.init(args=args)

    try:
        devo_node = DEVO_rclpy()

        rclpy.spin(devo_node)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in DEVO runner: {str(e)}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
