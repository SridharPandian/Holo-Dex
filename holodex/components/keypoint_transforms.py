import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from holodex.utils.network import FloatArrayPublisher, frequency_timer
from holodex.utils.vec_ops import *
from holodex.constants import *


class TransformHandCoords(object):
    def __init__(self, detector_type, moving_average_limit = 1):
        # Initializing the ROS node
        rospy.init_node('hand_transformation_coords_{}'.format(detector_type))

        self.detector_type = detector_type

        # Initializing subscriber to get the raw keypoints
        self.hand_coords = None

        if detector_type == 'MP':
            self.num_keypoints = MP_NUM_KEYPOINTS
            self.knuckle_points = (MP_JOINTS['knuckles'][0], MP_JOINTS['knuckles'][-1])
            rospy.Subscriber(MP_KEYPOINT_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(MP_HAND_TRANSFORM_COORDS_TOPIC)

        elif detector_type == 'VR_RIGHT':
            self.num_keypoints = OCULUS_NUM_KEYPOINTS
            self.knuckle_points = (OCULUS_JOINTS['knuckles'][0], OCULUS_JOINTS['knuckles'][-1])
            rospy.Subscriber(VR_RIGHT_HAND_KEYPOINTS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(VR_RIGHT_TRANSFORM_COORDS_TOPIC)

        elif detector_type == 'VR_LEFT':
            self.num_keypoints = OCULUS_NUM_KEYPOINTS
            self.knuckle_points = (OCULUS_JOINTS['knuckles'][0], OCULUS_JOINTS['knuckles'][-1])
            rospy.Subscriber(VR_LEFT_HAND_KEYPOINTS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
            self.keypoint_publisher = FloatArrayPublisher(VR_LEFT_TRANSFORM_DIR_TOPIC)
        
        else:
            raise NotImplementedError("There are no other detectors available. \
            The only options are Mediapipe or Oculus!")

        # Setting the frequency to 30 Hz
        if detector_type == 'MP':
            self.frequency_timer = frequency_timer(MP_FREQ)  
        elif detector_type == 'VR_RIGHT' or 'VR_LEFT': 
            self.frequency_timer = frequency_timer(VR_FREQ)

        # Moving average queue
        self.moving_average_limit = moving_average_limit
        self.moving_average_queue = []

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(self.num_keypoints, 3)

    def _translate_coords(self, hand_coords):
        return hand_coords - hand_coords[0]

    def _get_coord_frame(self, index_knuckle_coord, pinky_knuckle_coord):
        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Current Z
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)         # Current Y
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))                # Current X
        return [cross_product, palm_direction, palm_normal]

    def transform_right_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(translated_coords[self.knuckle_points[0]], translated_coords[self.knuckle_points[1]])

        # Finding the rotation matrix and rotating the coordinates
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        transformed_coords = (rotation_matrix @ translated_coords.T).T
        return transformed_coords
    
    def transform_left_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(translated_coords[self.knuckle_points[0]], translated_coords[self.knuckle_points[1]])

        translated_coord_frame = np.hstack([
            self.hand_coords[0], 
            original_coord_frame[0] + self.hand_coords[0], 
            original_coord_frame[1] + self.hand_coords[0],
            original_coord_frame[2] + self.hand_coords[0]
        ])
        return translated_coord_frame

    def stream(self):
        while True:
            if self.hand_coords is None:
                continue

            # Shift the points to required axes
            if self.detector_type == "VR_LEFT":
                transformed_coords = self.transform_left_keypoints(self.hand_coords)
            elif self.detector_type == "VR_RIGHT" or self.detector_type == "MP":
                transformed_coords = self.transform_right_keypoints(self.hand_coords)

            # Passing the transformed coords into a moving average
            averaged_coords = moving_average(transformed_coords, self.moving_average_queue, self.moving_average_limit)
            self.keypoint_publisher.publish(averaged_coords.flatten().tolist())

            self.frequency_timer.sleep()