import rospy
from holodex.utils.network import *
from holodex.constants import *

class OculusVRHandDetector(object):
    def __init__(self, HOST, PORT):
        # Initializing the ROS Node
        rospy.init_node('oculus_keypoint_extractor')

        # Initializing the network socket
        self.socket = create_pull_socket(HOST, PORT)

        # Initializing the hand keypoint publishers
        self.hand_keypoint_publishers = dict(
            left = FloatArrayPublisher(publisher_name = VR_LEFT_HAND_KEYPOINTS_TOPIC),
            right = FloatArrayPublisher(publisher_name = VR_RIGHT_HAND_KEYPOINTS_TOPIC)
        )

        self.frequency_timer = frequency_timer(VR_FREQ)

    def _process_data_token(self, data_token):
        return data_token.decode().strip()

    def _extract_data_from_token(self, token):        
        data = self._process_data_token(token)

        information = dict(
            hand = 'right' if data.startswith('right') else 'left'
        )

        vector_strings = data.split(':')[1].strip().split('|')
        keypoint_vals = []
        for vector_str in vector_strings:
            vector_vals = vector_str.split(',')
            for float_str in vector_vals[:3]:
                keypoint_vals.append(float(float_str))
            
        information['keypoints'] = keypoint_vals
        return information

    def _publish_data(self, keypoint_dict):
        self.hand_keypoint_publishers[keypoint_dict['hand']].publish(keypoint_dict['keypoints'])

    def stream(self):
        while True:
            # Getting the raw keypoints
            raw_keypoints = self.socket.recv()

            # Processing the keypoints and publishing them
            keypoint_dict = self._extract_data_from_token(raw_keypoints)
            self._publish_data(keypoint_dict)
            
            self.frequency_timer.sleep()