import os
import numpy as np
import cv2
import torch
from torchvision import transforms as T
from holodex.models import BehaviorCloning
from holodex.utils.files import make_dir
from holodex.utils.models import create_fc, load_encoder

class DeployBC(object):
    def __init__(
        self,
        encoder_configs,
        predictor_configs,
        model_weights_path,
        run_store_path,
        selected_view,
        transform = None
    ):
        self.selected_view = selected_view
        self._load_bc_model(encoder_configs, predictor_configs, model_weights_path)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor()
            ])

        self.run_store_path = run_store_path
        make_dir(self.run_store_path)
        self.state_counter = 1

    def _load_bc_model(self, encoder_configs, predictor_configs, model_weights_path):
        encoder = load_encoder(
            encoder_type = encoder_configs.settings.encoder_type
        )

        predictor = create_fc(
            input_dim = predictor_configs['input_dim'],
            output_dim = predictor_configs['output_dim'],
            hidden_dims = predictor_configs['hidden_dims'],
            use_batchnorm = predictor_configs['use_batchnorm']
        )        

        self.model = BehaviorCloning(
            encoder = encoder, 
            predictor = predictor
        )   

        model_weights = torch.load(model_weights_path)
        self.model.load_state_dict(model_weights, strict = False)
        self.model.eval()

    def get_action(self, input_dict):
        processed_image = self.transform(input_dict['image']).float().unsqueeze(0)
        action = self.model(processed_image).squeeze().detach()

        display_images = np.hstack(input_dict['image'])
        cv2.imshow('BC', display_images)

        cv2.imwrite(os.path.join(self.run_store_path, '{}.PNG'.format(self.state_counter)), display_images)
        self.state_counter += 1

        return action