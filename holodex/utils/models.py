import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_fc(input_dim, output_dim, hidden_dims, use_batchnorm, dropout = None, end_batchnorm = False):
    if hidden_dims is None:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim)])

    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    if dropout is not None:
        layers.append(nn.Dropout(p = dropout))

    for idx in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
        layers.append(nn.ReLU())

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[idx + 1]))

        if dropout is not None:
            layers.append(nn.Dropout(p = dropout))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))

    if end_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
        
    return nn.Sequential(*layers)

def load_encoder(encoder_type, input_channels = None, weights_path = None, pretrained = True):
    encoder = models.__dict__[encoder_type](pretrained = pretrained)
    encoder.fc = nn.Identity()

    if input_channels is not None:
        encoder.conv1 = nn.Conv2d(
            in_channels = input_channels, 
            out_channels = 64, 
            kernel_size = (7, 7), 
            stride = (2, 2), 
            padding = (3, 3), 
            bias = False
        )

    if weights_path is not None:
        print('Loading weights from: {}'.format(weights_path))
        weights = torch.load(weights_path, map_location = 'cpu')
        encoder.load_state_dict(weights, strict = False)

    return encoder