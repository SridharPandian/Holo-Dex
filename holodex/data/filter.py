import os

from holodex.robot.allegro_kdl import AllegroKDL
from holodex.utils.files import make_dir, get_pickle_data, store_pickle_data
from holodex.utils.vec_ops import get_distance

class FilterData(object):
    def __init__(self, data_path, delta):
        self.delta = delta
        self.data_path = data_path

        self.kdl_solver = AllegroKDL()

    def _get_coords(self, joint_angles):
        index_coords, _ = self.kdl_solver.finger_forward_kinematics('index', list(joint_angles)[0:4])
        middle_coords, _ = self.kdl_solver.finger_forward_kinematics('middle', list(joint_angles)[4:8])
        ring_coords, _ = self.kdl_solver.finger_forward_kinematics('ring', list(joint_angles)[8:12])
        thumb_coords, _ = self.kdl_solver.finger_forward_kinematics('thumb', list(joint_angles)[12:16])

        return index_coords, middle_coords, ring_coords, thumb_coords

    def _get_coords_from_state(self, state_path):
        state_data = get_pickle_data(state_path)
        joint_angles = state_data['allegro_joint_positions']
        return self._get_coords(joint_angles)

    def filter_demo(self, demo_path, target_path):
        states = os.listdir(demo_path)
        states.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        filtered_state_idxs = []

        first_state_path = os.path.join(demo_path, states[0])
        prev_index_coords, prev_middle_coords, prev_ring_coords, prev_thumb_coords = self._get_coords_from_state(first_state_path)

        for idx in range(1, len(states)):
            state_path = os.path.join(demo_path, states[idx])
            index_coords, middle_coords, ring_coords, thumb_coords = self._get_coords_from_state(state_path)

            delta_index = get_distance(prev_index_coords, index_coords)
            delta_middle = get_distance(prev_middle_coords, middle_coords)
            delta_ring = get_distance(prev_ring_coords, ring_coords)
            delta_thumb = get_distance(prev_thumb_coords, thumb_coords)

            delta_total = delta_index + delta_middle + delta_ring + delta_thumb

            if delta_total >= self.delta:
                filtered_state_idxs.append(idx)
                prev_index_coords, prev_middle_coords, prev_ring_coords, prev_thumb_coords = index_coords, middle_coords, ring_coords, thumb_coords

        for counter, idx in enumerate(filtered_state_idxs):
            state_data = get_pickle_data(os.path.join(demo_path, states[idx]))
            state_pickle_path = os.path.join(target_path, f'{counter + 1}')
            store_pickle_data(state_pickle_path, state_data)

    def filter(self, target_path, fresh_data = False):
        demo_list = os.listdir(self.data_path)
        demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for demo in demo_list:
            demo_path = os.path.join(self.data_path, demo)
            demo_target_path = os.path.join(target_path, demo)

            if os.path.exists(demo_target_path) and fresh_data is False:
                print('{} already filtered!'.format(demo))
                continue

            make_dir(demo_target_path)

            print(f"Filtering demonstration from {demo_path}")
            self.filter_demo(demo_path, demo_target_path)