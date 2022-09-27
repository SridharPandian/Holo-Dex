import cv2
import rospy
from holodex.constants import *
from holodex.utils.images import *
from holodex.utils.network import frequency_timer, ImageSubscriber


class MPImageVisualizer(object):
    def __init__(self, rotation_angle):
        try:
            rospy.init_node("mediapipe_hand_image_visualizer")
        except:
            pass

        self.color_image_subscriber = ImageSubscriber(
            subscriber_name = MP_PRED_RGB_IMAGE_TOPIC,
            node_name = 'mediapipe_color_image_streamer'
        )

        self.depth_image_subscriber = ImageSubscriber(
            subscriber_name = MP_PRED_DEPTH_IMAGE_TOPIC,
            node_name = 'mediapipe_depth_image_streamer',
            color = False
        )
        
        # Assigning visualizer rotation angle
        self.rotation_angle = rotation_angle

        # Setting frequency
        self.frequency_timer = frequency_timer(MP_FREQ) # 30 Hz (realsense frequency)        

    def stream(self):
        while True:
            if self.color_image_subscriber.get_image() is not None and self.depth_image_subscriber.get_image() is not None:
                # Stacking the images together and displaying them together
                color_image = rotate_image(self.color_image_subscriber.get_image(), self.rotation_angle)
                depth_image = rotate_image(self.depth_image_subscriber.get_image(), self.rotation_angle)

                images = stack_images([
                    rescale_image(color_image, VISUAL_RESCALE_FACTOR), 
                    rescale_image(depth_image, VISUAL_RESCALE_FACTOR)
                ])
                cv2.imshow('Hand Detector Images', images)
                cv2.waitKey(1)

                # Sleeping to maintain the frequency
                self.frequency_timer.sleep()


class RobotImageVisualizer(object):
    def __init__(self, camera_number):
        try:
            rospy.init_node('robot_image_{}_visualizer').format(camera_number)
        except:
            pass
        
        self.camera_number = camera_number

        self.color_image_subscriber = ImageSubscriber(
            subscriber_name = '/robot_camera_{}/color_image'.format(camera_number),
            node_name = 'robot_camera_{}_display'.format(camera_number)
        )
        
        # Setting frequency
        self.frequency_timer = frequency_timer(CAM_FPS) 

    def stream(self):
        while True:
            if self.color_image_subscriber.get_image() is None:
                continue

            image = rescale_image(self.color_image_subscriber.get_image(), VISUAL_RESCALE_FACTOR)
            cv2.imshow('Robot image {} - stream'.format(self.camera_number), image)
            cv2.waitKey(1)

            self.frequency_timer.sleep()