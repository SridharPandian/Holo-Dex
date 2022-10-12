import numpy as np
import rospy
import pyrealsense2 as rs

from holodex.utils.network import ImagePublisher, FloatArrayPublisher
from holodex.utils.images import *
from holodex.constants import *


class RealSenseRobotStream(object):
    def __init__(self, cam_serial_num, robot_cam_num, rotation_angle = 0):
        # Initializing ROS Node
        rospy.init_node('robot_cam_{}_stream'.format(robot_cam_num))

        # Disabling scientific notations
        np.set_printoptions(suppress=True)

        # Creating ROS Publishers
        self.color_image_publisher = ImagePublisher(publisher_name = '/robot_camera_{}/color_image'.format(robot_cam_num), color_image = True)
        self.depth_image_publisher = ImagePublisher(publisher_name = '/robot_camera_{}/depth_image'.format(robot_cam_num), color_image = False)
        self.intrinsics_publisher = FloatArrayPublisher(publisher_name = '/robot_camera_{}/intrinsics'.format(robot_cam_num))

        # Setting rotation settings
        self.rotation_angle = rotation_angle

        # Setting ROS frequency
        self.rate = rospy.Rate(CAM_FPS)

        # Starting the realsense camera stream
        self._start_realsense(cam_serial_num, WIDTH, HEIGHT, CAM_FPS, PROCESSING_PRESET)
        print(f"Started the Realsense pipeline for camera: {cam_serial_num}!")

    def _start_realsense(self, cam_serial_num, width, height, fps, processing_preset):
        config = rs.config()
        pipeline = rs.pipeline()
        config.enable_device(cam_serial_num)

        # Enabling camera streams
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Starting the pipeline
        cfg = pipeline.start(config)
        device = cfg.get_device()

        # Setting the depth mode to high accuracy mode
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, processing_preset) # High accuracy post-processing mode
        self.realsense = pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def stream(self):
        print("Starting stream!\n")
        while True:
            color_image, depth_image = self.get_rgb_depth_images()
            color_image, depth_image = rotate_image(color_image, self.rotation_angle), rotate_image(depth_image, self.rotation_angle)

            # Publishing the intrinsics of the camera
            self.intrinsics_publisher.publish(self.intrinsics_matrix.reshape(9).tolist())

            # Publishing the original color and depth images
            self.color_image_publisher.publish(color_image)
            self.depth_image_publisher.publish(depth_image)

            self.rate.sleep()