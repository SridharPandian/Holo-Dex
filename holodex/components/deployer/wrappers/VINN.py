import os
import numpy as np
import cv2
from torchvision import transforms as T
from holodex.models import KNearestNeighbors
from holodex.utils.files import make_dir
from holodex.utils.models import load_encoder
from .NearestNeighborBuffer import NearestNeighborBuffer
from copy import deepcopy as copy
from .helpers import *

class DeployVINN(object):
    def __init__(
        self,
        encoder_configs,
        encoder_weights_path,
        data_path,
        demo_list,
        min_action_distance,
        run_store_path,
        absolute_actions,
        selected_view,
        nn_buffer_limit,
        transform = None
    ):
        # Encoder configs
        self.selected_view = selected_view
        self._load_encoder(encoder_configs, encoder_weights_path)

        # Action data
        states_path = os.path.join(data_path, 'states')
        self._states = load_tensors(states_path, demo_list)

        actions_path = os.path.join(data_path, 'actions')
        self._actions = load_tensors(actions_path, demo_list)   

        self.min_action_distance = min_action_distance
        self.absolute_actions = absolute_actions
        if absolute_actions:
            self._actions = self._states + self._actions

        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor()
            ])

        # Representations
        self._input_representations = load_encoder_representations(
                data_path = data_path, 
                encoder = self.encoder, 
                selected_view = self.selected_view, 
                transform = self.transform, 
                demo_list = demo_list
            )

        # NN extractors and buffers
        self.nn_extractor = KNearestNeighbors(
            self._input_representations, 
            self._actions
        )
        self.buffer = NearestNeighborBuffer(nn_buffer_limit)

        # Visualizations
        self.input_image_paths, self.output_image_paths, self.cumm_len, self.traj_idx = load_image_paths(
            data_path, 
            self.selected_view, 
            demo_list
        )

        # Logging
        self.run_store_path = run_store_path
        make_dir(self.run_store_path)
        self.state_counter = 1

    def _load_encoder(self, encoder_configs, encoder_weights_path):
        self.encoder = load_encoder(
            encoder_type = encoder_configs.settings.encoder_type,
            weights_path = encoder_weights_path
        ).eval()

    def _process_image_representation(self, input_image):
        processed_image = self.transform(input_image).float().unsqueeze(0)
        representation = self.encoder(processed_image).squeeze().detach()
        return representation

    def get_nn_action(self, input_dict):
        representation = self._process_image_representation(input_dict['image'])
        nn_actions, nn_idxs = self.nn_extractor.get_k_nearest_neighbors(representation, 10)
        
        # Appending the item in the buffer and choosing the idx
        choosen_idx = self.buffer.choose(nn_idxs)
        choosen_nn_idx, applied_action = nn_idxs[choosen_idx].item(), nn_actions[choosen_idx]

        # Getting the NN idx for visualization
        nn_traj_idx, nn_traj_image_idx = get_traj_state_idxs(choosen_nn_idx, self.traj_idx, self.cumm_len)
        input_nn_image_path = self.input_image_paths[nn_traj_idx][nn_traj_image_idx] 
        output_nn_image_path = self.output_image_paths[nn_traj_idx][nn_traj_image_idx]

        return applied_action, input_nn_image_path, output_nn_image_path

    def get_action(self, input_dict):
        action, input_nn_image_path, output_nn_image_path = self.get_nn_action(input_dict)

        nn_image, output_image = cv2.imread(input_nn_image_path), cv2.imread(output_nn_image_path)
        display_images = np.hstack([input_dict['image'], nn_image, output_image])
        cv2.imshow('VINN', display_images)
        cv2.waitKey(1)

        cv2.imwrite(os.path.join(self.run_store_path, '{}.PNG'.format(self.state_counter)), display_images)
        self.state_counter += 1

        return action