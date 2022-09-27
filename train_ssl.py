import os
import hydra
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from holodex.utils.optimizers import *
from byol_pytorch import BYOL
from holodex.models import VICReg, SimCLR, MoCo
from holodex.models.self_supervised_pretraining.mocov3 import adjust_moco_momentum
from holodex.datasets import get_image_dataset
from holodex.utils.models import set_seed_everywhere, create_fc
from holodex.utils.augmentations import get_augment_function, get_simclr_augmentation, get_moco_augmentations
from holodex.utils.logger import Logger
from holodex.utils.files import make_dir
from hydra.utils import instantiate
from tqdm import tqdm
from copy import deepcopy as copy

class Workspace(object):
    def __init__(self, configs):
        self.configs = configs
        self.ssl_method = self.configs.ssl_method.name

        make_dir(self.configs.checkpoint_path)
        set_seed_everywhere(self.configs.seed)

        self._init_datasets() 
        self._init_model() 
        self._init_optimizer()
        self.logger = Logger(configs)

    def _init_datasets(self):
        self.train_dataset = get_image_dataset(
            data_path = self.configs.dataset.dataset_path,
            selected_views = [self.configs.selected_view], 
            image_type = self.configs.image_type,
            demos_list = self.configs.dataset.complete_demos,
            mean_tensors = self.configs.image_parameters.mean_tensors,
            std_tensors = self.configs.image_parameters.std_tensors,
            dataset_type = 'pretrain',
            absolute = None
        )

        print('Loaded train dataset. Number of datapoints in the dataset: {}.'.format(len(self.train_dataset)))

        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size = self.configs.batch_size,
            shuffle = True,
            num_workers = self.configs.num_workers,
            pin_memory = True,
        )

    def _init_model(self):
        encoder = instantiate(self.configs.encoder.settings).to(self.configs.device)

        if self.ssl_method == 'BYOL':
            augmentation_function = get_augment_function(
                mean_tensor = self.configs.image_parameters.mean_tensors[self.configs.selected_view - 1], 
                std_tensor = self.configs.image_parameters.std_tensors[self.configs.selected_view - 1]
            )

            self.learner = BYOL(
                encoder,
                image_size = self.configs.image_parameters.image_size,
                hidden_layer = 'avgpool',
                augment_fn = augmentation_function
            )

        elif self.ssl_method == 'VICReg':
            projector = create_fc(
                input_dim = self.configs.encoder.output_size, 
                output_dim = self.configs.ssl_method.projector.output_size,
                hidden_dims = self.configs.ssl_method.projector.hidden_sizes,
                use_batchnorm = self.configs.ssl_method.projector.enable_batchnorm
            )

            augmentation_function = get_augment_function(
                mean_tensor = self.configs.image_parameters.mean_tensors[self.configs.selected_view - 1], 
                std_tensor = self.configs.image_parameters.std_tensors[self.configs.selected_view - 1]
            )

            self.learner = VICReg(
                backbone = encoder,
                projector = projector,
                augment_fn = augmentation_function,
                sim_coef = self.configs.ssl_method.sim_coef,
                std_coef = self.configs.ssl_method.std_coef,
                cov_coef = self.configs.ssl_method.cov_coef
            )

        elif self.ssl_method == 'SimCLR':
            augmentation_function = get_simclr_augmentation(
                color_jitter_const = self.configs.ssl_method.color_jitter_const,
                mean_tensor = self.configs.image_parameters.mean_tensors[self.configs.selected_view - 1], 
                std_tensor = self.configs.image_parameters.std_tensors[self.configs.selected_view - 1]
            )

            projector = create_fc(
                input_dim = self.configs.encoder.output_size, 
                output_dim = self.configs.ssl_method.projector.output_size,
                hidden_dims = self.configs.ssl_method.projector.hidden_sizes,
                use_batchnorm = self.configs.ssl_method.projector.enable_batchnorm
            )

            self.learner = SimCLR(
                encoder = encoder,
                projector = projector,
                augment_fn = augmentation_function,
                sec_augment_fn = augmentation_function, # Since we do not use Random crop, both the augment functions are the same
                temperature = self.configs.ssl_method.temperature,
            )

        elif self.ssl_method == 'MoCo':
            encoder.fc = create_fc(
                input_dim = self.configs.encoder.output_size,
                output_dim = self.configs.ssl_method.expander.output_size,
                hidden_dims = self.configs.ssl_method.expander.hidden_sizes,
                use_batchnorm = self.configs.ssl_method.expander.enable_batchnorm,
                end_batchnorm = True
            )

            momentum_encoder = copy(encoder)

            projector = create_fc(
                input_dim = self.configs.ssl_method.expander.output_size,
                output_dim = self.configs.ssl_method.projector.output_size,
                hidden_dims = self.configs.ssl_method.projector.hidden_sizes,
                use_batchnorm = self.configs.ssl_method.projector.enable_batchnorm
            ) 

            first_augmentation_function, second_augmentation_function = get_moco_augmentations(
                mean_tensor = self.configs.image_parameters.mean_tensors[self.configs.selected_view], 
                std_tensor = self.configs.image_parameters.std_tensors[self.configs.selected_view]
            )

            self.learner = MoCo(
                base_encoder = encoder,
                momentum_encoder = momentum_encoder,
                predictor = projector,
                first_augment_fn = first_augmentation_function,
                sec_augment_fn = second_augmentation_function,
                temperature = self.configs.ssl_method.temperature
            ).to(self.configs.device)

        self.learner.to(self.configs.device)

        print('Created the {} learner.'.format(self.ssl_method))

    def _init_optimizer(self):
        if self.configs.optimizer == 'SGD':
            self.optimizer = SGD(
                self.learner.parameters(), 
                lr = self.configs.lr, 
                momentum = self.configs.momentum, 
                weight_decay = self.configs.weight_decay
            )
        elif self.configs.optimizer == 'Adam':
            self.optimizer = Adam(
                self.learner.parameters(),
                lr = self.configs.lr,
                weight_decay = self.configs.weight_decay
            )
        elif self.configs.optimizer == 'LARS':
            self.optimizer = LARS(
                self.learner.parameters(),
                lr = 0,
                weight_decay = self.configs.weight_decay,
                weight_decay_filter = exclude_bias_and_norm,
                lars_adaptation_filter = exclude_bias_and_norm,
            )

    def train_one_epoch(self, epoch_num):
        self.learner.train()
        total_train_loss = 0

        for idx, input_images in enumerate(tqdm(self.train_dataloader)):
            input_images = input_images[0].cuda(self.configs.device) 

            # Scheduling the Learning Rate for the LARS optimizer
            if self.configs.optimizer == 'LARS':
                lr = adjust_learning_rate(self.configs, self.optimizer, self.train_dataloader, idx)

            if self.ssl_method == 'BYOL':
                loss = self.learner(input_images)

            elif self.ssl_method == 'VICReg':
                loss, _ = self.learner(input_images)

            elif self.ssl_method == 'SimCLR':
                loss = self.learner(input_images)

            elif self.ssl_method == 'MoCo':
                moco_momentum = adjust_moco_momentum(
                    epoch = epoch_num + (idx / len(self.train_dataloader)), 
                    momentum = self.configs.ssl_method.momentum,
                    total_epochs = self.configs.epochs
                )
                loss = self.learner(input_images, moco_momentum)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_train_loss += loss.item()

            if self.ssl_method == 'BYOL':
                self.learner.update_moving_average()

        average_train_loss = total_train_loss / len(self.train_dataloader)
        print('Epoch Loss:', average_train_loss)

        return average_train_loss

    def _save_snapshot(self, obs_loss, epoch):
        if obs_loss < self.best_loss:
            torch.save(
                self.learner.state_dict(),
                os.path.join(
                    self.configs.checkpoint_path, 
                    '{}_best.pt'.format(self.configs.run_name)
                )
            )

            print('Best loss observed: {}'.format(obs_loss))
            self.best_loss = obs_loss

            if (epoch + 1) % self.configs.checkpoint_interval == 0:
                torch.save(
                    self.learner.state_dict(),
                    os.path.join(
                        self.configs.checkpoint_path, 
                        '{}_epoch_{}.pt'.format(self.configs.run_name, epoch + 1)
                    )
                )

    def train(self):
        self.best_loss = torch.inf

        for epoch in range(self.configs.epochs):
            print('\nTraining epoch: {}'.format(epoch + 1))

            # Training the encoder for one epoch
            average_train_loss = self.train_one_epoch(epoch)

            log_data = {
                'avg_train_loss': average_train_loss
            }

            # Log the values
            self.logger.update(data = log_data, step = epoch)

            self._save_snapshot(
                obs_loss = average_train_loss,
                epoch = epoch
            )

        print('\n{}: {} Pretraining finished!'.format(self.ssl_method, self.configs.run_name))
        print('Checkpoints can be found at: {}'.format(self.configs.checkpoint_path))


@hydra.main(version_base = '1.2', config_path='configs', config_name='train_ssl')
def main(configs):
    workspace = Workspace(configs)
    workspace.train()

if __name__ == '__main__':
    main()