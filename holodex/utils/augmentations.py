import torch
from PIL import ImageOps
from torch import nn
from torchvision import transforms as T

class Solarize(nn.Module):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def forward(self, x):
        return ImageOps.solarize(x)

# Used for BYOL and VICReg
def get_augment_function(mean_tensor, std_tensor):
    return T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8, .8, .8, .2)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p = 0.2), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = mean_tensor, 
            std = std_tensor
        )
    ])

# Official SimCLR augmentations: 
def get_simclr_augmentation(color_jitter_const, mean_tensor, std_tensor):
    return T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8 * color_jitter_const, .8 * color_jitter_const, .8 * color_jitter_const, .2 * color_jitter_const)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p = 0.2), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.5, 1.5))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = mean_tensor, 
            std = std_tensor
        )
    ])


# Modified version of the official implementation of MoCoV3: https://github.com/facebookresearch/moco-v3/
def get_moco_augmentations(mean_tensor, std_tensor):
    first_augmentation_function = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(0.4, 0.4, 0.2, 0.1)])  # not strengthened
            , p=0.8
        ),
        T.RandomGrayscale(p = 0.2),
        T.RandomApply(
            torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.Normalize(
            mean = mean_tensor,
            std = std_tensor
        )
    ])

    second_augmentation_function = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(0.4, 0.4, 0.2, 0.1)])  # not strengthened
            , p=0.8
        ),
        T.RandomGrayscale(p = 0.2),
        T.RandomApply(
            torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.RandomSolarize(threshold = 0.5, p = 0.2),
        T.Normalize(
            mean = mean_tensor,
            std = std_tensor
        )
    ])

    return first_augmentation_function, second_augmentation_function

# Standard augmentations used in SimCLR 
def get_imagenet_augment_function(mean_tensor, std_tensor):
    return T.Compose([
        T.RandomApply(
            nn.ModuleList([
                T.CenterCrop(140),
                T.Resize(224)
            ]),
            p = 0.5
        ),
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomApply(
            torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = mean_tensor,
            std = std_tensor
        )
    ])