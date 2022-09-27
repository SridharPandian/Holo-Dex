import os
import torch
import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from holodex.utils.optimizers import *
from torch.nn.functional import mse_loss
from holodex.datasets import get_image_dataset
from holodex.models import BehaviorCloning, BehaviorCloningRep
from holodex.utils.models import create_fc
from holodex.utils.logger import Logger
from holodex.utils.files import make_dir, set_seed_everywhere
from tqdm import tqdm

class Workspace(object):
    def __init__(self, configs):
        self.configs = configs

        make_dir(self.configs.checkpoint_path)
        set_seed_everywhere(self.configs.seed)

        self._init_datasets() 
        self._init_model() 
        self._init_optimizer()
        self.logger = Logger(configs)

    def _init_datasets(self):
        print('Loading the dataset(s).')
        self.train_dataset = get_image_dataset(
            data_path = self.configs.dataset.dataset_path,
            selected_views = [self.configs.selected_view],
            image_type = self.configs.image_type,
            demos_list = self.configs.dataset.train_demos if self.configs.train_test_split else self.configs.dataset.complete_demos,
            mean_tensors = self.configs.image_parameters.mean_tensors,
            std_tensors = self.configs.image_parameters.std_tensors,
            dataset_type = 'action',
            absolute = self.configs.absolute_actions
        )

        print('Loaded train dataset. Number of datapoints in the dataset: {}.'.format(len(self.train_dataset)))

        self.train_dataloader = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.configs.batch_size,
            shuffle = True,
            num_workers = self.configs.num_workers,
            pin_memory = True
        )

        if self.configs.train_test_split:
            self.test_dataset = get_image_dataset(
                data_path = self.configs.dataset.dataset_path,
                selected_views = [self.configs.selected_view],
                image_type = self.configs.image_type,
                demos_list = self.configs.dataset.test_demos,
                mean_tensors = self.configs.image_parameters.mean_tensors,
                std_tensors = self.configs.image_parameters.std_tensors,
                dataset_type = 'action',
                absolute = self.configs.absolute_actions
            )

            print('Loaded test dataset. Number of datapoints in the dataset: {}.'.format(len(self.train_dataset)))

            self.test_dataloader = DataLoader(
                dataset = self.test_dataset,
                batch_size = self.configs.batch_size,
                shuffle = True,
                num_workers = self.configs.num_workers,
                pin_memory = True
            )

    def _init_model(self):
        print('Initializing the model.')

        encoder = instantiate(self.configs.encoder.settings).to(self.configs.device)
        predictor = create_fc(
            input_dim = self.configs.predictor.input_dim,
            output_dim = self.configs.predictor.output_dim,
            hidden_dims = self.configs.predictor.hidden_dims,
            use_batchnorm = self.configs.predictor.use_batchnorm,
            dropout = self.configs.predictor.dropout
        )

        if self.configs.encoder_gradient:
            self.model = BehaviorCloning(
                encoder = encoder,
                predictor = predictor
            )
        else:
            self.model = BehaviorCloningRep(
                encoder = encoder,
                predictor = predictor
            )

        self.model.to(self.configs.device)

    def _init_optimizer(self):
        if self.configs.optimizer == 'SGD':
            self.optimizer = SGD(
                self.model.parameters(), 
                lr = self.configs.lr, 
                momentum = self.configs.momentum, 
                weight_decay = self.configs.weight_decay
            )
        elif self.configs.optimizer == 'Adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr = self.configs.lr,
                weight_decay = self.configs.weight_decay
            )
        elif self.configs.optimizer == 'LARS':
            self.optimizer = LARS(
                self.model.parameters(),
                lr = 0,
                weight_decay = self.configs.weight_decay,
                weight_decay_filter = exclude_bias_and_norm,
                lars_adaptation_filter = exclude_bias_and_norm,
            )

    def train_one_epoch(self):
        self.model.train()
        total_train_loss = 0

        for idx, (input_images, actions) in enumerate(tqdm(self.train_dataloader)):
            input_images = input_images[0].to(self.configs.device)
            actions = actions.float().to(self.configs.device)

            if self.configs.optimizer == 'LARS':
                adjust_learning_rate(self.configs, self.optimizer, self.train_dataloader, idx)

            predicted_action = self.model(input_images)
            loss = mse_loss(predicted_action, actions) * self.configs['action_scaling_factor']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(self.train_dataloader)
        return average_train_loss 

    def test_one_epoch(self):
        self.model.eval()
        total_test_loss = 0

        for input_images, actions in tqdm(self.test_dataloader):
            input_images = input_images[0].to(self.configs.device)
            actions = actions.float().to(self.configs.device)

            predicted_action = self.model(input_images).detach()
            loss = mse_loss(predicted_action, actions) * self.configs['action_scaling_factor']
            total_test_loss += loss.item()

        average_test_loss = total_test_loss / len(self.test_dataloader)
        return average_test_loss 

    def _save_snapshot(self, obs_loss, epoch):
        if obs_loss < self.best_loss:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.configs.checkpoint_path, 
                    '{}_best.pt'.format(self.configs.run_name)
                )
            )
            self.best_loss = obs_loss

        if (epoch + 1) % self.configs.checkpoint_interval == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.configs.checkpoint_path, 
                    '{}_epoch_{}.pt'.format(self.configs.run_name, epoch + 1)
                )
            )

    def train(self):
        self.best_loss = torch.inf

        for epoch in range(self.configs.epochs):
            print('\nTraining epoch: {}'.format(epoch + 1))

            average_train_loss = self.train_one_epoch()

            if self.configs.train_test_split:
                print('Testing.')
                average_test_loss = self.test_one_epoch()

            log_data = {
                'avg_train_loss': average_train_loss
            }

            if self.configs.train_test_split:
                log_data['avg_test_loss'] = average_test_loss

            print('Losses:', log_data)
            self.logger.update(data = log_data, step = epoch)

            if self.configs.train_test_split:
                comparison_loss = average_test_loss
            else:
                comparison_loss = average_train_loss

            self._save_snapshot(comparison_loss, epoch)

        self.logger.dump()
        self.logger.close()

        print('\nBehavior cloning: {} training finished!'.format(self.configs.run_name))
        print('Checkpoints can be found at: {}'.format(self.configs.checkpoint_path))

@hydra.main(version_base = '1.2', config_path='configs', config_name='train_bc')
def main(configs):
    workspace = Workspace(configs = configs)
    workspace.train()

if __name__ == '__main__':
    main()