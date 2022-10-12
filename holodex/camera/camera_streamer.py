import cv2
import numpy as np
import rospy

from holodex.utils.network import create_push_socket, frequency_timer, ImageSubscriber
from holodex.utils.images import rescale_image, rotate_image
from holodex.constants import *

class RobotCameraStreamer(object):
    def __init__(self, robot_camera_num, host, port, rotation_angle = 0):
        rospy.init_node('robot_cam_stream_server_{}'.format(robot_camera_num))

        self.socket = create_push_socket(host, port)

        self.robot_image_subscriber = ImageSubscriber(
            '/robot_camera_{}/color_image'.format(robot_camera_num),
            'robot_camera_{}_streamer'.format(robot_camera_num)
        )
        
        self.stream_rotation_angle = rotation_angle
        self.frequency_timer = frequency_timer(VR_FREQ)
        print('Server started for camera {}!'.format(robot_camera_num))

    def _serialize_image_buffer(self, image):
        _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_WEBP_QUALITY), 10])
        data = np.array(buffer).tobytes()
        return data

    def stream(self):
        while True:
            if self.robot_image_subscriber.get_image() is None:
                continue

            image = rotate_image(
                rescale_image(self.robot_image_subscriber.get_image(), VISUAL_RESCALE_FACTOR), 
                self.stream_rotation_angle
            )
            data = self._serialize_image_buffer(image)
            self.socket.send(data)

            self.frequency_timer.sleep()