import rospy
from std_msgs.msg import Float64MultiArray
from .calibrators import MPHandBoundCalibrator
from holodex.robot import AllegroKDLControl, AllegroHand
from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound
from holodex.constants import *
from copy import deepcopy as copy


class MPDexArmTeleOp(object):
    def __init__(self):
        # Initializing the ROS Node
        rospy.init_node("mp_dexarm_teleop")

        # Storing the transformed hand coordinates
        self.hand_coords = None
        rospy.Subscriber(MP_HAND_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)

        # Calibrating the hand pose and getting hand bounds
        self._calibrate_bounds()

        # Initializing the solver
        self.finger_tip_solver = AllegroKDLControl()

        # Initializing the robot controller
        self.robot = AllegroHand()

        # Initialzing the moving average queues
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        # Getting the bounds for the human hand and the allegro hand
        allegro_bounds_path = get_path_in_package('components/robot_operators/configs/allegro_mp.yaml')
        self.allegro_bounds = get_yaml_data(allegro_bounds_path)

    def _callback_hand_coords(self, coords):
        self.hand_coords = np.array(list(coords.data)).reshape(MP_NUM_KEYPOINTS, 3)

    def _calibrate_bounds(self):
        print("***************************************************************")
        print("     Starting calibration process ")
        print("***************************************************************")
        calibrator = MPHandBoundCalibrator()
        self.hand_bounds = calibrator.get_bounds()

    def get_finger_tip_coords(self):
        finger_tip_coords = {}

        for key in ['index', 'middle', 'ring', 'pinky', 'thumb']:
            finger_tip_coords[key] = self.hand_coords[MP_JOINTS[key][-1]]

        return finger_tip_coords

    def motion_2D(self, finger_configs):
        finger_tip_coords = self.get_finger_tip_coords()
        desired_joint_angles = copy(self.robot.get_hand_position())

        # Movement for the index finger
        if not finger_configs['freeze_index']:
            desired_joint_angles = self.finger_tip_solver.finger_1D_motion(
                finger_type = "index",
                hand_y_val = finger_tip_coords['index'][1], 
                robot_x_val = self.allegro_bounds['index']['x_coord'],
                robot_y_val = self.allegro_bounds['index']['y_coord'], 
                y_hand_bound = self.hand_bounds[0], 
                z_robot_bound = [self.allegro_bounds['index']['z_bottom'], self.allegro_bounds['index']['z_top']], 
                moving_avg_arr = self.moving_average_queues['index'], 
                curr_angles = desired_joint_angles
            )
        else:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0

        # Movement for the middle finger
        if not finger_configs['freeze_middle']:
            desired_joint_angles = self.finger_tip_solver.finger_1D_motion(
                finger_type = "middle",
                hand_y_val = finger_tip_coords['middle'][1], 
                robot_x_val = self.allegro_bounds['middle']['x_coord'],
                robot_y_val = self.allegro_bounds['middle']['y_coord'], 
                y_hand_bound = self.hand_bounds[0], 
                z_robot_bound = [self.allegro_bounds['middle']['z_bottom'], self.allegro_bounds['middle']['z_top']], 
                moving_avg_arr = self.moving_average_queues['middle'], 
                curr_angles = desired_joint_angles
            )
        else:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0

        # Movement for the ring finger
        desired_joint_angles = self.finger_tip_solver.finger_2D_motion(
            finger_type = "ring",
            hand_x_val = finger_tip_coords['pinky'][1], 
            hand_y_val = finger_tip_coords['ring'][1], 
            robot_x_val = self.allegro_bounds['ring']['x_coord'],
            x_hand_bound = self.hand_bounds[6], 
            y_hand_bound = self.hand_bounds[4], 
            y_robot_bound = [self.allegro_bounds['ring']['y_left'], self.allegro_bounds['ring']['y_right']], 
            z_robot_bound = [self.allegro_bounds['ring']['z_bottom'], self.allegro_bounds['ring']['z_top']], 
            moving_avg_arr = self.moving_average_queues['ring'], 
            curr_angles = desired_joint_angles
        )

        # Movement for the thumb finger
        if coord_in_bound(self.hand_bounds[7:11], finger_tip_coords['thumb'][:2]) > -1:
            desired_joint_angles = self.finger_tip_solver.thumb_motion_2D(
                hand_coordinates = finger_tip_coords['thumb'], 
                xy_hand_bounds = self.hand_bounds[7:11],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['yz_top_right'], 
                    self.allegro_bounds['thumb']['yz_bottom_right'],
                    self.allegro_bounds['thumb']['yz_bottom_left'],
                    self.allegro_bounds['thumb']['yz_top_left']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = desired_joint_angles
            )
        
        return desired_joint_angles

    def motion_3D(self, finger_configs):
        finger_tip_coords = self.get_finger_tip_coords()
        desired_joint_angles = copy(self.robot.get_hand_position())

        # Movement for the index finger
        if not finger_configs['freeze_index']:
            desired_joint_angles = self.finger_tip_solver.finger_2D_depth_motion(
                finger_type = "index",
                hand_y_val = finger_tip_coords['index'][1], 
                robot_y_val = self.allegro_bounds['index']['y_coord'], 
                hand_z_val = finger_tip_coords['index'][2], 

                y_hand_bound = self.hand_bounds[0], 
                z_hand_bound = self.hand_bounds[1], 

                x_robot_bound = [self.allegro_bounds['index']['x_bottom'], self.allegro_bounds['index']['x_top']], 
                z_robot_bound = [self.allegro_bounds['index']['z_bottom'], self.allegro_bounds['index']['z_top']], 
                moving_avg_arr = self.moving_average_queues['index'], 
                curr_angles = desired_joint_angles
            )
        else:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                if idx > 0:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0.05
                else:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0

        # Movement for the middle finger
        if not finger_configs['freeze_middle']:
            desired_joint_angles = self.finger_tip_solver.finger_2D_depth_motion(
                finger_type = "middle",
                hand_y_val = finger_tip_coords['middle'][1], 
                robot_y_val = self.allegro_bounds['middle']['y_coord'], 
                hand_z_val = finger_tip_coords['middle'][2], 

                y_hand_bound = self.hand_bounds[2], 
                z_hand_bound = self.hand_bounds[3], 

                x_robot_bound = [self.allegro_bounds['middle']['x_bottom'], self.allegro_bounds['middle']['x_top']], 
                z_robot_bound = [self.allegro_bounds['middle']['z_bottom'], self.allegro_bounds['middle']['z_top']], 
                moving_avg_arr = self.moving_average_queues['middle'], 
                curr_angles = desired_joint_angles
            )
        else:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                if idx > 0:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0.05
                else:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0

        # Movement for the ring finger
        desired_joint_angles = self.finger_tip_solver.finger_3D_motion(
            finger_type = "ring",
            hand_x_val = finger_tip_coords['pinky'][1], 
            hand_y_val = finger_tip_coords['ring'][1], 
            hand_z_val = finger_tip_coords['ring'][2], 

            x_hand_bound = self.hand_bounds[6], 
            y_hand_bound = self.hand_bounds[4], 
            z_hand_bound = self.hand_bounds[5], 

            x_robot_bound = [self.allegro_bounds['ring']['x_bottom'], self.allegro_bounds['ring']['x_top']], 
            y_robot_bound = [self.allegro_bounds['ring']['y_left'], self.allegro_bounds['ring']['y_right']], 
            z_robot_bound = [self.allegro_bounds['ring']['z_bottom'], self.allegro_bounds['ring']['z_top']], 
            moving_avg_arr = self.moving_average_queues['ring'], 
            curr_angles = desired_joint_angles
        )

        # Movement for the thumb finger
        if coord_in_bound(self.hand_bounds[7:11], finger_tip_coords['thumb'][:2]) > -1:
            desired_joint_angles = self.finger_tip_solver.thumb_motion_3D(
                hand_coordinates = finger_tip_coords['thumb'], 
                xy_hand_bounds = self.hand_bounds[7:11],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['yz_top_right'], 
                    self.allegro_bounds['thumb']['yz_bottom_right'],
                    self.allegro_bounds['thumb']['yz_bottom_left'],
                    self.allegro_bounds['thumb']['yz_top_left']
                ], 
                z_hand_bound = self.hand_bounds[11], 
                x_robot_bound = [self.allegro_bounds['thumb']['x_bottom'], self.allegro_bounds['thumb']['x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = desired_joint_angles
            )
        
        return desired_joint_angles

    def move(self, finger_configs):
        print("\n******************************************************************************")
        print("     Controller initiated. ")
        print("******************************************************************************\n")
        print("Start controlling the robot hand using the tele-op.\n")

        while True:
            if self.hand_coords is not None and self.robot.get_hand_position() is not None:
                if finger_configs['three_dim']:
                    desired_joint_angles = self.motion_3D(finger_configs)  
                else: 
                    desired_joint_angles = self.motion_2D(finger_configs)
                    
                self.robot.move(desired_joint_angles)