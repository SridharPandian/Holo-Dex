import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray

from holodex.viz.plotters.plotter_3d import *
from holodex.constants import *

class Hand3DVisualizer(object):
    def __init__(self, detector_type):
        # Initializing a ROS node
        try:
            rospy.init_node("{}_hand_3d_visualizer".format(detector_type))
        except:
            pass

        self.keypoints = None

        if detector_type == 'VR_RIGHT':
            self.num_keypoints = OCULUS_NUM_KEYPOINTS
            rospy.Subscriber(
                VR_RIGHT_TRANSFORM_COORDS_TOPIC, 
                Float64MultiArray, 
                self._callback_keypoints, 
                queue_size = 1
            )
        elif detector_type == 'MP':
            self.num_keypoints = MP_NUM_KEYPOINTS
            rospy.Subscriber(
                MP_HAND_TRANSFORM_COORDS_TOPIC, 
                Float64MultiArray, 
                self._callback_keypoints, 
                queue_size = 1
            )
        else:
            raise NotImplementedError("There are no other detectors available. \
            The only options are Mediapipe or Oculus!")

        # Initializing the plotting object
        self.plotter3D = Plot3DRightHand(detector_type = detector_type)

    def _callback_keypoints(self, keypoints):
        self.keypoints = np.array(keypoints.data).reshape(self.num_keypoints, 3)

    def stream(self):
        while True:
            if self.keypoints is None:
                keypoints = np.zeros((self.num_keypoints, 3))
                self.plotter3D.draw(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])
            else:
                self.plotter3D.draw(self.keypoints[:, 0], self.keypoints[:, 1], self.keypoints[:, 2])


class OculusLeftHandDirVisualizer(object):
    def __init__(self, scaling_factor = 0.1):
        # Other parameters
        self.scaling_factor = scaling_factor

        # Initializing the Keypoint variable and subscriber
        self.directions = None
        rospy.Subscriber(VR_LEFT_TRANSFORM_DIR_TOPIC, Float64MultiArray, self._callback_directions, queue_size = 1)

        # Initializing the plotting object
        self.dir_plotter = Plot3DLeftHandDirection()

    def _callback_directions(self, directions):
        self.directions = np.array(directions.data).reshape(4, 3) * self.scaling_factor

    def stream(self):
        while True:
            if self.directions is None:
                directions = np.zeros((4, 3))
                self.dir_plotter.draw(directions[:, 0], directions[:, 1], directions[:, 2])
            else:
                self.dir_plotter.draw(self.directions[:, 0], self.directions[:, 1], self.directions[:, 2])