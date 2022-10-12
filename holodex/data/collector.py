import os
import sys
import rospy 
import time
from sensor_msgs.msg import JointState
from holodex.utils.files import *
from holodex.constants import *
from holodex.robot import AllegroHand
from holodex.utils.network import ImageSubscriber, frequency_timer


class DataCollector(object):
    def __init__(
        self,
        num_cams,
        storage_path
    ):
        rospy.init_node('data_extractor', disable_signals = True)

        self.storage_path = storage_path

        # ROS Subscribers based on the number of cameras used
        self.num_cams = num_cams

        self.color_image_subscribers, self.depth_image_subscribers = [], []
        for cam_num in range(self.num_cams):
            self.color_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/color_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_color_data_collector'.format(cam_num + 1)
                )
            )
            self.depth_image_subscribers.append(
                ImageSubscriber(
                    subscriber_name = '/robot_camera_{}/depth_image'.format(cam_num + 1),
                    node_name = 'robot_camera_{}_depth_data_collector'.format(cam_num + 1),
                    color = False
                )
            )

        # Allegro Hand controller initialization
        self.allegro_hand = AllegroHand()

        # ROS Subscriber to get the arm information
        self.kinova_joint_state = None
        rospy.Subscriber(KINOVA_JOINT_STATE_TOPIC, JointState, self._callback_kinova_joint_state, queue_size = 1)

        self.frequency_timer = frequency_timer(RECORD_FPS)

    def _callback_kinova_joint_state(self, data):
        self.kinova_joint_state = data

    def extract(self, offset = 0):
        counter = offset + 1
        try:
            while True:
                skip_loop = False
                # Checking for broken data streams
                if self.allegro_hand.get_hand_position() is None or self.kinova_joint_state is None:
                    skip_loop = True

                for color_image_subscriber in self.color_image_subscribers:
                    if color_image_subscriber.get_image() is None:
                        skip_loop = True

                for depth_image_subscriber in self.depth_image_subscribers:
                    if depth_image_subscriber.get_image() is None:
                        skip_loop = True

                if skip_loop:
                    continue
            
                state = dict()

                for cam_num in range(self.num_cams):
                    state['camera_{}_color_image'.format(cam_num + 1)] = self.color_image_subscribers[cam_num].get_image()
                    state['camera_{}_depth_image'.format(cam_num + 1)] = self.depth_image_subscribers[cam_num].get_image()
                
                # Allegro data
                state['allegro_joint_positions'] = self.allegro_hand.get_hand_position()
                state['allegro_joint_velocity'] = self.allegro_hand.get_hand_velocity()
                state['allegro_joint_effort'] = self.allegro_hand.get_hand_torque()
                state['allegro_commanded_joint_position'] = self.allegro_hand.get_commanded_joint_position()

                # Kinova data
                state['kinova_joint_positions'] = self.kinova_joint_state.position

                # Temporal information
                state['time'] = time.time()

                # Saving the pickle file
                state_pickle_path = os.path.join(self.storage_path, f'{counter}')
                store_pickle_data(state_pickle_path, state)

                counter += 1

                self.frequency_timer.sleep()
        
        except KeyboardInterrupt:
            print('Finished recording! Data can be found in {}'.format(self.storage_path))
            sys.exit(0)