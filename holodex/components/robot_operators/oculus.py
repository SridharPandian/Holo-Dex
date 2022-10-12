import rospy
from std_msgs.msg import Float64MultiArray
from .calibrators import OculusThumbBoundCalibrator
from holodex.robot import AllegroKDLControl, AllegroJointControl, AllegroHand
from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound
from holodex.constants import *
from copy import deepcopy as copy

class VRDexArmTeleOp(object):
    def __init__(self):
        # Initializing the ROS Node
        rospy.init_node("vr_dexarm_teleop")

        # Storing the transformed hand coordinates
        self.hand_coords = None
        rospy.Subscriber(VR_RIGHT_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)

        # Initializing the solvers
        self.fingertip_solver = AllegroKDLControl()
        self.finger_joint_solver = AllegroJointControl()

        # Initializing the robot controller
        self.robot = AllegroHand()

        # Initialzing the moving average queues
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        # Calibrating to get the thumb bounds
        self._calibrate_bounds()

        # Getting the bounds for the allegro hand
        allegro_bounds_path = get_path_in_package('components/robot_operators/configs/allegro_vr.yaml')
        with open(allegro_bounds_path, 'r') as file:
            self.allegro_bounds = yaml.safe_load(file)

    def _calibrate_bounds(self):
        print("***************************************************************")
        print("     Starting calibration process ")
        print("***************************************************************")
        calibrator = OculusThumbBoundCalibrator()
        self.thumb_index_bounds, self.thumb_middle_bounds, self.thumb_ring_bounds = calibrator.get_bounds()

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(24, 3)

    def _get_finger_coords(self, finger_type):
        return np.vstack([self.hand_coords[0], self.hand_coords[OCULUS_JOINTS[finger_type]]])

    def _get_2d_thumb_angles(self, curr_angles):
        if coord_in_bound(self.thumb_index_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_index_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['top_right'], 
                    self.allegro_bounds['thumb']['bottom_right'],
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['index_top']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_middle_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_middle_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['index_top'], 
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['middle_top']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_ring_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_ring_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['middle_top'], 
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['ring_bottom'],
                    self.allegro_bounds['thumb']['ring_top']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        else:
            return curr_angles

    def _get_3d_thumb_angles(self, curr_angles):
        if coord_in_bound(self.thumb_index_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_index_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['top_right'], 
                    self.allegro_bounds['thumb']['bottom_right'],
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['index_top']
                ], 
                z_hand_bound = self.thumb_index_bounds[4], 
                x_robot_bound = [self.allegro_bounds['thumb']['index_x_bottom'], self.allegro_bounds['thumb']['index_x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_middle_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_middle_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['index_top'], 
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['middle_top']
                ], 
                z_hand_bound = self.thumb_middle_bounds[4], 
                x_robot_bound = [self.allegro_bounds['thumb']['middle_x_bottom'], self.allegro_bounds['thumb']['middle_x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_ring_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_ring_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['middle_top'], 
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['ring_bottom'],
                    self.allegro_bounds['thumb']['ring_top']
                ], 
                z_hand_bound = self.thumb_ring_bounds[4], 
                x_robot_bound = [self.allegro_bounds['thumb']['ring_x_bottom'], self.allegro_bounds['thumb']['ring_x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        else:
            return curr_angles

    def motion(self, finger_configs):
        desired_joint_angles = copy(self.robot.get_hand_position())

        # Movement for the index finger
        if not finger_configs['freeze_index']:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'index',
                finger_joint_coords = self._get_finger_coords('index'),
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['index']
            )
        else:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                if idx > 0:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0.05
                else:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0

        # Movement for the middle finger
        if not finger_configs['freeze_middle']:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'middle',
                finger_joint_coords = self._get_finger_coords('middle'),
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['middle']
            )
        else:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                if idx > 0:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0.05
                else:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0

        # Movement for the ring finger
        # Calculating the translatory joint angles
        desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
            finger_type = 'ring',
            finger_joint_coords = self._get_finger_coords('ring'),
            curr_angles = desired_joint_angles,
            moving_avg_arr = self.moving_average_queues['ring']
        )

        # Movement for the thumb finger - we disable 3D motion just for the thumb
        if finger_configs['three_dim']:
            desired_joint_angles = self._get_3d_thumb_angles(desired_joint_angles)
        else:
            desired_joint_angles = self._get_2d_thumb_angles(desired_joint_angles)
        
        return desired_joint_angles


    def move(self, finger_configs):
        print("\n******************************************************************************")
        print("     Controller initiated. ")
        print("******************************************************************************\n")
        print("Start controlling the robot hand using the Oculus Headset.\n")

        while True:
            if self.hand_coords is not None and self.robot.get_hand_position() is not None:
                # Obtaining the desired angles
                desired_joint_angles = self.motion(finger_configs)

                # Move the hand based on the desired angles
                self.robot.move(desired_joint_angles)