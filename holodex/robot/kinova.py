import numpy as np
import rospy
from sensor_msgs.msg import JointState
from kinova_arm.controller import KinovaController # From https://github.com/NYU-robot-learning/DIME-Controllers
from holodex.constants import KINOVA_JOINT_STATE_TOPIC, KINOVA_POSITIONS

class KinovaArm(object):
    def __init__(self):
        rospy.init_node('holodex_kinova_arm_controller')

        self.robot = KinovaController()

        self.kinova_joint_state = None
        rospy.Subscriber(KINOVA_JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)

    def _callback_joint_state(self, joint_state):
        self.kinova_joint_state = joint_state

    def get_arm_position(self):
        if self.kinova_joint_state is None:
            return None
        
        return np.array(self.kinova_joint_state, dtype = np.float32)

    def home_robot(self):
        self.robot.move(KINOVA_POSITIONS['flat'])

    def reset(self):
        self.robot.move(KINOVA_POSITIONS['flat'])

    def move(self, input_angles):
        self.robot.joint_movement(input_angles, False)