import numpy as np
from copy import deepcopy as copy

from holodex.utils.files import *
from holodex.utils.vec_ops import *


class AllegroJointControl(object):
    def __init__(self, bounded_angles = True):
        np.set_printoptions(suppress = True)

        self.hand_configs = get_yaml_data(get_path_in_package("robot/configs/allegro_info.yaml"))
        self.finger_configs = get_yaml_data(get_path_in_package("robot/configs/allegro_link_info.yaml"))
        self.bound_info = get_yaml_data(get_path_in_package("robot/configs/allegro_bounds.yaml"))

        self.time_steps = self.bound_info['time_steps']
        self.bounded_angles = bounded_angles

        self.bounds = {}
        for finger in self.hand_configs['fingers'].keys():
            self.bounds[finger] = np.array(self.bound_info['jointwise_angle_bounds'][
                self.finger_configs['links_info'][finger]['offset'] : self.finger_configs['links_info'][finger]['offset'] + 4
            ])

        self.linear_scaling_factors = self.bound_info['linear_scaling_factors']
        self.rotatory_scaling_factors = self.bound_info['rotatory_scaling_factors']

    def _get_curr_finger_angles(self, curr_angles, finger_type):
        return np.array(curr_angles[
            self.finger_configs['links_info'][finger_type]['offset'] : self.finger_configs['links_info'][finger_type]['offset'] + 4
        ])

    def _get_filtered_angles(self, finger_type, calc_finger_angles, curr_angles, moving_avg_arr):
        curr_finger_angles = self._get_curr_finger_angles(curr_angles, finger_type)
        avg_finger_angles = moving_average(calc_finger_angles, moving_avg_arr, self.time_steps)       
        desired_angles = np.array(copy(curr_angles))

        # Applying angular bounds
        if self.bounded_angles is True:
            del_finger_angles = avg_finger_angles - curr_finger_angles
            clipped_del_finger_angles = np.clip(del_finger_angles, - self.bounds[finger_type], self.bounds[finger_type])

            for idx in range(self.hand_configs['joints_per_finger']):
                desired_angles[self.finger_configs['links_info'][finger_type]['offset'] + idx] += clipped_del_finger_angles[idx]
        else:
            for idx in range(self.hand_configs['joints_per_finger']):
                desired_angles[self.finger_configs['links_info'][finger_type]['offset'] + idx] = avg_finger_angles[idx]

        return desired_angles 

    def calculate_finger_angles(self, finger_type, finger_joint_coords, curr_angles, moving_avg_arr):
        translatory_angles = []
        for idx in range(self.hand_configs['joints_per_finger'] - 1): # Ignoring the rotatory joint
            angle = calculate_angle(
                finger_joint_coords[idx],
                finger_joint_coords[idx + 1],
                finger_joint_coords[idx + 2]
            )
            translatory_angles.append(angle * self.linear_scaling_factors[idx])

        rotatory_angle = [self.calculate_finger_rotation(finger_joint_coords) * self.rotatory_scaling_factors[finger_type]] 
        calc_finger_angles = rotatory_angle + translatory_angles
        filtered_angles = self._get_filtered_angles(finger_type, calc_finger_angles, curr_angles, moving_avg_arr)
        return filtered_angles


    def calculate_finger_rotation(self, finger_joint_coords):
        angle = calculate_angle(finger_joint_coords[0][:1], finger_joint_coords[1][:1], finger_joint_coords[-1][:1])
        
        # Checking if the finger tip is on the left side or the right side of the knuckle
        knuckle_vector = finger_joint_coords[1] - finger_joint_coords[0]
        tip_vector = finger_joint_coords[-1] - finger_joint_coords[0]
        knuckle_vector_slope = knuckle_vector[1] / knuckle_vector[0]
        tip_vector_slope = tip_vector[1] / tip_vector[0]

        if knuckle_vector_slope > tip_vector_slope:
            return angle
        else:
            return -1 * angle