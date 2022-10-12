import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray

from holodex.viz.plotters.plotter_2d import *
from holodex.constants import *

class Hand2DVisualizer(object):
    def __init__(self, detector_type, *args):
        # Initializing a ROS node
        try:
            rospy.init_node("{}_hand_2d_visualizer".format(detector_type))
        except:
            pass

        self.keypoints = None

        # Selecting the VR detector
        if detector_type == 'VR_RIGHT':
            self.num_keypoints = OCULUS_NUM_KEYPOINTS
            rospy.Subscriber(
                VR_RIGHT_TRANSFORM_COORDS_TOPIC, 
                Float64MultiArray, 
                self._callback_keypoints, 
                queue_size = 1
            )
            self.plotter2D = Plot2DOculusHand(*args)

        # Selecting the Mediapipe detector
        elif detector_type == 'MP':
            self.num_keypoints = MP_NUM_KEYPOINTS
            rospy.Subscriber(
                MP_HAND_TRANSFORM_COORDS_TOPIC, 
                Float64MultiArray, 
                self._callback_keypoints, 
                queue_size = 1
            )
            self.plotter2D = Plot2DMPHand()
        
        else:
            raise NotImplementedError("There are no other detectors available. \
            The only options are Mediapipe or Oculus!")

    def _callback_keypoints(self, keypoints):
        self.keypoints = np.array(keypoints.data).reshape(self.num_keypoints, 3)

    def stream(self):
        while True:
            if self.keypoints is None:
                keypoints = np.zeros((self.num_keypoints, 2))
                self.plotter2D.draw(keypoints[:, 0], keypoints[:, 1])
            else:
                self.plotter2D.draw(self.keypoints[:, 0], self.keypoints[:, 1])
