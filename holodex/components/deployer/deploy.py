import sys
import numpy as np
from PIL import Image as PILImage
import torch
from torchvision import transforms as T
from holodex.robot import AllegroKDL, AllegroHand, KinovaArm
from holodex.components.deployer.wrappers import *
from holodex.utils.network import ImageSubscriber, frequency_timer
from holodex.constants import *

class DexArmDeploy(object):
    def __init__(
        self,
        deploy_configs
    ):
        torch.set_printoptions(sci_mode = False)
        self.configs = deploy_configs
        model = self.configs.model
        self.cam_num = self.configs.task.selected_view

        # Image transform
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = torch.tensor(self.configs['task']['image_parameters']['mean_tensors'][self.cam_num - 1]),
                std = torch.tensor(self.configs['task']['image_parameters']['std_tensors'][self.cam_num - 1])
            )
        ])

        # Model initialization
        if model == 'VINN':
            print('Initializing the VINN deployment module...')
            self.model = DeployVINN(
                encoder_configs = self.configs['task']['encoder'],
                encoder_weights_path = self.configs['task']['vinn']['encoder_weights_path'],
                data_path = self.configs['task']['vinn']['data_path'],
                demo_list = self.configs['task']['vinn']['demos_list'],
                min_action_distance = self.configs['task']['vinn']["min_action_distance"],
                run_store_path = self.configs['task']['run_store_path'],
                absolute_actions = self.configs['absolute_actions'],
                selected_view = self.cam_num,
                nn_buffer_limit = self.configs['task']['vinn']['nn_buffer_limit'],
                transform = transform
            )

        elif model == 'BC':
            print('Initializing the BC deployment module...')
            self.model = DeployBC(
                encoder_configs = self.configs['task']['encoder'],
                predictor_configs = self.configs['task']['bc']['predictor'],
                model_weights_path = self.configs['task']['bc']['model_weights'],
                run_store_path = self.configs['task']['run_store_path'],
                selected_view = self.cam_num,
                transforms = transform
            )
    
        # Image subscriber initialization
        self.robot_image_subscriber = ImageSubscriber(
            '/robot_camera_{}/color_image'.format(self.cam_num), 
            'robot_camera_{}_color'.format(self.cam_num)
        )

        # Robot controller initialization
        self.kdl_solver = AllegroKDL()
        self.hand_robot = AllegroHand()
        self.arm_robot = KinovaArm()

        # Moving Arm to position
        self.arm_robot.move(KINOVA_POSITIONS[self.configs.task.arm_position])
        self.hand_robot.reset()

        if self.configs['run_loop']:
            self.frequency_timer = frequency_timer(self.configs.loop_rate)

    def _transform_image(self, image):
        image = PILImage.fromarray(image)
        image = image.crop(self.configs['task']['image_parameters']['crop_sizes'][self.cam_num - 1])
        image = image.resize((
            self.configs['task']['image_parameters']['image_size'], 
            self.configs['task']['image_parameters']['image_size']
        ))
        return np.asarray(image)

    def _get_transformed_image(self):
        return self._transform_image(self.robot_image_subscriber.get_image())

    def solve(self):
        sys.stdin = open(0) # To get inputs while spawning multiple processes

        while True:
            if self.robot_image_subscriber.get_image() is None:
                continue

            if self.hand_robot.get_hand_position() is None:
                continue

            print('\n***************************************************************')

            if not self.configs['run_loop']:
                register = input('\nPress a key to perform an action...')

                if register == 'h':
                    print('Reseting the Robot!')
                    self.hand_robot.reset()
                    continue

            finger_tip_coords = self.hand_robot.get_fingertip_coords(self.hand_robot.get_hand_position())
            print('\nCurrent joint state: {}'.format(finger_tip_coords))

            transformed_image = self._get_transformed_image()

            input_dict = dict(
                key_press = register if not self.configs['run_loop'] else None,
                image = transformed_image,
                joint_state = finger_tip_coords
            )

            action = self.model.get_action(input_dict)
            print('\nObtained action: {}'.format(action))

            if not self.configs['absolute_actions']:
                desired_finger_tip_coords = np.array(finger_tip_coords) + np.array(action)
            else:
                desired_finger_tip_coords = action

            print('\nApplied joint state coord: {}'.format(desired_finger_tip_coords))
            self.hand_robot.move_to_coords(desired_finger_tip_coords)

            if self.configs['run_loop']:
                self.frequency_timer.sleep()