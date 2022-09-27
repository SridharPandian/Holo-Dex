import sys
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from holodex.constants import *
from holodex.utils.files import *

class MPHandBoundCalibrator(object):
    def __init__(self):
        # Initializing the keypoint subscriber
        self.hand_coords = None
        rospy.Subscriber(MP_HAND_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_keypoints, queue_size = 1)

        # Storage paths
        make_dir(CALIBRATION_FILES_PATH)
        
    def _callback_keypoints(self, coord_data):
        self.hand_coords = np.array(list(coord_data.data)).reshape(MP_NUM_KEYPOINTS, 3)

    def _get_bounds(self, finger_type, coord_axis):
        tip_idx = MP_JOINTS[finger_type][-1]

        if coord_axis == 'Y':
            axis_idx = 1
        if coord_axis == 'Z':
            axis_idx = 2

        # Getting Y Index finger bounds
        print("Getting the {} bounds for the {} finger.".format(coord_axis, finger_type))
        bounds = np.zeros((1, 2))

        register = input("Press Enter to register the upper {} bound for the {} finger.".format(coord_axis, finger_type))
        bounds[0][1] = self.hand_coords[tip_idx][axis_idx]  
        print("Registered upper bound is {}.\n".format(bounds[0][1]))

        register = input("Press Enter to register the lower {} bound for the {} finger.".format(coord_axis, finger_type))
        bounds[0][0] = self.hand_coords[tip_idx][axis_idx]    
        print("Registered lower bound is {}.\n".format(bounds[0][0]))

        return bounds

    def _get_thumb_2d_bounds(self):
        tip_idx = MP_JOINTS['thumb'][-1]

        print("Getting the quadrilateral bounds for the Thumb finger.")
        thumb_xy_bounds = np.zeros((4, 2))

        register = input("Press Enter to register the upper right bound for the Thumb finger.")
        thumb_xy_bounds[0][0] = self.hand_coords[tip_idx][0]
        thumb_xy_bounds[0][1] = self.hand_coords[tip_idx][1]
        print("Registered upper right bound is {}.\n".format(thumb_xy_bounds[0]))
        
        register = input("Press Enter to register the lower right bound for the Thumb finger.")
        thumb_xy_bounds[1][0] = self.hand_coords[tip_idx][0]
        thumb_xy_bounds[1][1] = self.hand_coords[tip_idx][1]
        print("Registered lower right bound is {}.\n".format(thumb_xy_bounds[1]))

        register = input("Press Enter to register the lower left bound for the Thumb finger.")
        thumb_xy_bounds[2][0] = self.hand_coords[tip_idx][0]
        thumb_xy_bounds[2][1] = self.hand_coords[tip_idx][1]
        print("Registered lower left bound is {}.\n".format(thumb_xy_bounds[2]))

        register = input("Press Enter to register the upper left bound for the Thumb finger.")
        thumb_xy_bounds[3][0] = self.hand_coords[tip_idx][0]
        thumb_xy_bounds[3][1] = self.hand_coords[tip_idx][1]
        print("Registered upper left bound is {}.\n".format(thumb_xy_bounds[3]))

        return thumb_xy_bounds

    def _calibrate(self):
        print('Starting calibration.\n')
        
        index_y_bound = self._get_bounds('index', 'Y')
        index_z_bound = self._get_bounds('index', 'Z')

        middle_y_bound = self._get_bounds('middle', 'Y')
        middle_z_bound = self._get_bounds('middle', 'Z')

        ring_y_bound = self._get_bounds('ring', 'Y')
        ring_z_bound = self._get_bounds('ring', 'Z')

        pinky_y_bound = self._get_bounds('pinky', 'Y')

        thumb_xy_bound = self._get_thumb_2d_bounds()
        thumb_z_bound = self._get_bounds('thumb', 'Z')

        calibrated_bounds = np.vstack([
            index_y_bound, 
            index_z_bound, 
            middle_y_bound, 
            middle_z_bound, 
            ring_y_bound, 
            ring_z_bound, 
            pinky_y_bound, 
            thumb_xy_bound, 
            thumb_z_bound
        ])

        np.save(MP_THUMB_BOUNDS_PATH, calibrated_bounds)

        return calibrated_bounds

    def get_bounds(self):
        sys.stdin = open(0) # To get inputs while spawning multiple processes

        if check_file(MP_THUMB_BOUNDS_PATH):
            print("\nCalibration file already exists. Do you want to create a new one? Press y for Yes else press Enter")
            use_calibration_file = input()

            if use_calibration_file == "y":
                calibrated_bounds = self._calibrate()
            else:
                calibrated_bounds = np.load(MP_THUMB_BOUNDS_PATH)

        else:
            print("\nNo calibration file found. Need to calibrate hand poses.\n")
            calibrated_bounds = self._calibrate()

        return calibrated_bounds


class OculusThumbBoundCalibrator(object):
    def __init__(self):
        # Initializing the keypoint subscriber
        self.hand_coords = None
        rospy.Subscriber(VR_RIGHT_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_keypoints, queue_size = 1)

        # Storage paths
        make_dir(CALIBRATION_FILES_PATH)

    def _callback_keypoints(self, coord_data):
        self.hand_coords = np.array(list(coord_data.data)).reshape(OCULUS_NUM_KEYPOINTS, 3)

    def _get_thumb_tip_coord(self):
        return self.hand_coords[OCULUS_JOINTS['thumb'][-1]]

    def _get_xy_coords(self):
        return [self._get_thumb_tip_coord()[0], self._get_thumb_tip_coord()[1]]

    def _get_z_coord(self):
        return self._get_thumb_tip_coord()[-1]

    def _calibrate(self):
        register = input("Place the thumb in the top right corner.")
        top_right_coord = self._get_xy_coords()

        register = input("Place the thumb in the bottom right corner.")
        bottom_right_coord = self._get_xy_coords()

        register = input("Place the thumb in the index bottom corner.")
        index_bottom_coord = self._get_xy_coords()

        register = input("Place the thumb in the index top corner.")
        index_top_coord = self._get_xy_coords()

        register = input("Stretch the thumb to get highest index bound z value.")
        index_high_z = self._get_z_coord()

        register = input("Relax the thumb to get the lowest index bound z value.")
        index_low_z = self._get_z_coord()

        register = input("Place the thumb in the middle bottom corner.")
        middle_bottom_coord = self._get_xy_coords()

        register = input("Place the thumb in the middle top corner.")
        middle_top_coord = self._get_xy_coords()

        register = input("Stretch the thumb to get highest middle bound z value.")
        middle_high_z = self._get_z_coord()

        register = input("Relax the thumb to get the lowest middle bound z value.")
        middle_low_z = self._get_z_coord()

        register = input("Place the thumb in the ring bottom corner.")
        ring_bottom_coord = self._get_xy_coords()

        register = input("Place the thumb in the ring top corner.")
        ring_top_coord = self._get_xy_coords()
        
        register = input("Stretch the thumb to get highest ring bound z value.")
        ring_high_z = self._get_z_coord()

        register = input("Relax the thumb to get the lowest ring bound z value.")
        ring_low_z = self._get_z_coord()

        thumb_index_bounds = np.array([
            top_right_coord,
            bottom_right_coord,
            index_bottom_coord,
            index_top_coord,
            [index_low_z, index_high_z]
        ])

        thumb_middle_bounds = np.array([
            index_top_coord,
            index_bottom_coord,
            middle_bottom_coord,
            middle_top_coord,
            [middle_low_z, middle_high_z]
        ])

        thumb_ring_bounds = np.array([
            middle_top_coord,
            middle_bottom_coord,
            ring_bottom_coord,
            ring_top_coord,
            [ring_low_z, ring_high_z]
        ])

        thumb_bounds = np.vstack([thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds])

        handpose_coords = np.array([
            top_right_coord,
            bottom_right_coord,
            index_bottom_coord,
            middle_bottom_coord,
            ring_bottom_coord,
            ring_top_coord,
            middle_top_coord,
            index_top_coord
        ])

        np.save(VR_DISPLAY_THUMB_BOUNDS_PATH, handpose_coords)
        np.save(VR_THUMB_BOUNDS_PATH, thumb_bounds)

        return thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds

    def get_bounds(self):
        sys.stdin = open(0) # To take inputs while spawning multiple processes

        if check_file(VR_THUMB_BOUNDS_PATH):
            use_calibration_file = input("\nCalibration file already exists. Do you want to create a new one? Press y for Yes else press Enter")

            if use_calibration_file == "y":
                thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds = self._calibrate()
            else:
                calibrated_bounds = np.load(VR_THUMB_BOUNDS_PATH)
                thumb_index_bounds = calibrated_bounds[:5]
                thumb_middle_bounds = calibrated_bounds[5:10]
                thumb_ring_bounds = calibrated_bounds[10:]

        else:
            print("\nNo calibration file found. Need to calibrate hand poses.\n")
            thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds = self._calibrate()

        return thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds