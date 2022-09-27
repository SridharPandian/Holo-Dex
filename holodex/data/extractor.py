import os
import numpy as np
import cv2
from PIL import Image
import torch

from holodex.robot.allegro_kdl import AllegroKDL
from holodex.utils.files import make_dir, get_pickle_data


class ColorImageExtractor(object):
    def __init__(self, data_path, num_cams, image_size, crop_sizes = None):
        assert num_cams == len(crop_sizes)
        
        self.data_path = data_path
        self.num_cams = num_cams
        self.image_size = image_size
        self.crop_sizes = crop_sizes

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        color_cam_image_paths = []
        for cam_num in range(self.num_cams):
            color_cam_image_path = os.path.join(target_path, 'camera_{}_color_image').format(cam_num + 1)
            color_cam_image_paths.append(color_cam_image_path)
            make_dir(color_cam_image_path)

        for state in states:
            state_data = get_pickle_data(os.path.join(demo_path, state))
            
            color_images = [state_data['camera_{}_color_image'.format(cam_num + 1)] for cam_num in range(self.num_cams)]
            
            if self.crop_sizes is not None:
                for cam_num in range(self.num_cams):
                    color_image = Image.fromarray(color_images[cam_num])
                    color_image = color_image.crop(self.crop_sizes[cam_num])
                    color_image = color_image.resize((self.image_size, self.image_size))
                    color_image = np.array(color_image)
                    cv2.imwrite(os.path.join(color_cam_image_paths[cam_num], f'{state}.PNG'), color_image)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)
            make_dir(demo_target_path)

            print(f"\nExtracting images from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class DepthImageExtractor(object):
    def __init__(self, data_path, num_cams, image_size, crop_sizes = None):
        assert num_cams == len(crop_sizes)
        
        self.data_path = data_path
        self.num_cams = num_cams
        self.image_size = image_size
        self.crop_sizes = crop_sizes

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        depth_cam_image_paths = []
        for cam_num in range(self.num_cams):
            depth_cam_image_path = os.path.join(target_path, 'camera_{}_depth_image').format(cam_num + 1)
            depth_cam_image_paths.append(depth_cam_image_path)
            make_dir(depth_cam_image_path)

        for state in states:
            state_data = get_pickle_data(os.path.join(demo_path, state))
            
            depth_images = [state_data['camera_{}_depth_image'.format(cam_num + 1)] for cam_num in range(self.num_cams)]
            
            if self.crop_sizes is not None:
                for cam_num in range(self.num_cams):
                    depth_image = Image.fromarray(depth_images[cam_num])
                    depth_image = depth_image.crop(self.crop_sizes[cam_num])
                    depth_image = depth_image.resize((self.image_size, self.image_size))
                    depth_image = np.array(depth_image)
                    np.save(os.path.join(depth_cam_image_paths[cam_num], f'{state}.npy'), depth_image)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)
            make_dir(demo_target_path)

            print(f"\nExtracting images from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class StateExtractor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.kdl_solver = AllegroKDL()

    def _get_coords(self, joint_angles):
        index_coords, _ = self.kdl_solver.finger_forward_kinematics('index', list(joint_angles)[0:4])
        middle_coords, _ = self.kdl_solver.finger_forward_kinematics('middle', list(joint_angles)[4:8])
        ring_coords, _ = self.kdl_solver.finger_forward_kinematics('ring', list(joint_angles)[8:12])
        thumb_coords, _ = self.kdl_solver.finger_forward_kinematics('thumb', list(joint_angles)[12:16])

        finger_coords = list(index_coords) + list(middle_coords) + list(ring_coords) + list(thumb_coords)
        return np.array(finger_coords)

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_state_data = []

        for idx in range(len(states) - 1):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))
            state_joint_angles = state_data['allegro_joint_positions']
            state_joint_coords = self._get_coords(state_joint_angles)
            demo_state_data.append(state_joint_coords)

        demo_state_data = torch.tensor(np.array(demo_state_data)).squeeze()
        torch.save(demo_state_data, target_path)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, f'{demo}.pth')

            print(f"Extracting states from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)


class ActionExtractor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.kdl_solver = AllegroKDL()

    def _get_coords(self, joint_angles):
        index_coords, _ = self.kdl_solver.finger_forward_kinematics('index', list(joint_angles)[0:4])
        middle_coords, _ = self.kdl_solver.finger_forward_kinematics('middle', list(joint_angles)[4:8])
        ring_coords, _ = self.kdl_solver.finger_forward_kinematics('ring', list(joint_angles)[8:12])
        thumb_coords, _ = self.kdl_solver.finger_forward_kinematics('thumb', list(joint_angles)[12:16])

        finger_coords = list(index_coords) + list(middle_coords) + list(ring_coords) + list(thumb_coords)
        return np.array(finger_coords)

    def extract_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_action_data = []

        first_state_data = get_pickle_data(os.path.join(demo_path, states[0]))
        first_state_joint_angles = first_state_data['allegro_joint_positions']
        prev_joint_coords = self._get_coords(first_state_joint_angles)

        for idx in range(1, len(states)):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))
            state_joint_angles = state_data['allegro_joint_positions']
            state_joint_coords = self._get_coords(state_joint_angles)

            action = state_joint_coords - prev_joint_coords # action = s2 - s1
            demo_action_data.append(action)

            prev_joint_coords = state_joint_coords

        demo_action_data = torch.tensor(np.array(demo_action_data)).squeeze()

        torch.save(demo_action_data, target_path)

    def extract(self, target_path):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, f'{demo}.pth')

            print(f"Extracting actions from {demo_path}")
            self.extract_demo(demo_path, demo_target_path)