import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from holodex.constants import NUM_CAMS
from copy import deepcopy as copy

class ImageDataset(Dataset):
    def __init__(
        self,
        data_path,
        selected_views,
        image_type,
        demos_list = None,
        transforms = None
    ):
        self.selected_views = selected_views
        self.demos = demos_list
        
        self.color_image, self.depth_image = False, False
        if image_type == 'color':
            self.color_image = True
        elif image_type == 'depth':
            self.depth_image = True
        elif image_type == 'all':
            self.color_image = True
            self.depth_image = True
        else:
            raise NotImplementedError('Image type does not exist!')

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = {
                'color_image': [],
                'depth_image': []
            }
            for _ in range(len(self.selected_views)):
                if self.color_image:
                    self.transforms['color_image'].append(T.Compose([
                        T.ToTensor()
                    ]))
                if self.depth_image:
                    self.transforms['depth_image'].append(T.Compose([
                        T.ToTensor()
                    ]))
 
        self.state_offset = 0

    def _load_image_paths(self, images_path):
        if self.demos is None:
            self.demos = os.listdir(images_path)
            self.demos.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_paths = [os.path.join(images_path, demo_name) for demo_name in self.demos]

        if self.color_image:
            self.color_image_paths = [[] for _ in range(len(self.selected_views))]
        if self.depth_image:
            self.depth_image_paths = [[] for _ in range(len(self.selected_views))]

        for idx, demo_path in enumerate(demo_paths):
            if self.color_image:
                color_image_names = os.listdir(os.path.join(demo_path, 'camera_{}_color_image'.format(self.selected_views[0])))
                color_image_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
                
            if self.depth_image:
                depth_image_names = os.listdir(os.path.join(demo_path, 'camera_{}_depth_image'.format(self.selected_views[0])))
                depth_image_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

            for idx, cam_num in enumerate(self.selected_views):
                if self.color_image:
                    demo_color_image_paths = [
                        os.path.join(demo_path, 'camera_{}_color_image'.format(cam_num), image_name) for image_name in color_image_names
                    ]
                    self.color_image_paths[idx].append(copy(demo_color_image_paths))
                
                if self.depth_image:
                    demo_depth_image_paths = [
                        os.path.join(demo_path, 'camera_{}_depth_image'.format(cam_num), image_name) for image_name in depth_image_names
                    ]
                    self.depth_image_paths[idx].append(copy(demo_depth_image_paths))

    def _get_trajectory_data(self):
        self.cumm_len = [0]
        self.traj_markers = []

        images = self.color_image_paths[0] if self.color_image else self.depth_image_paths[0]

        for idx, demo_images in enumerate(images):
            cumm_value = len(demo_images) + self.cumm_len[-1] - self.state_offset
            self.cumm_len.append(cumm_value)

            for _ in range(len(demo_images) - self.state_offset):
                self.traj_markers.append(idx)

    def _get_traj_state_idx(self, idx):
        traj_num = self.traj_markers[idx]
        state_num = idx - self.cumm_len[traj_num]
        return traj_num, state_num

    def _get_color_image(self, traj_num, state_num, cam_num):
        image_path = self.color_image_paths[cam_num][traj_num][state_num]
        image = Image.open(image_path)
        image_tensor = self.transforms['color_image'][cam_num](image) # Odd indexed transforms
        image.close()

        return image_tensor

    def _get_depth_image(self, traj_num, state_num, cam_num):
        image_path = self.depth_image_paths[cam_num][traj_num][state_num]
        image_array = np.load(image_path) / 1000 # converting from millimeters to meters

        image = Image.fromarray(image_array)
        image_tensor = self.transforms['depth_image'][cam_num](image)
        image.close()

        return image_tensor

    def __len__(self):
        return len(self.traj_markers)

    def __getitem__(self, idx):
        raise NotImplementedError()

class ImageSSLDataset(ImageDataset):
    def __init__(
        self, 
        data_path, 
        selected_views, 
        image_type, 
        demos_list = None,
        transforms = None
    ):
        super().__init__(data_path, selected_views, image_type, demos_list, transforms)
    
        images_path = os.path.join(data_path, 'images')
        self._load_image_paths(images_path)
        self._get_trajectory_data()

    def __getitem__(self, idx):
        traj_num, state_num = self._get_traj_state_idx(idx)

        if self.color_image:
            color_images = []
            for cam_num in range(len(self.selected_views)):
                color_images.append(self._get_color_image(traj_num, state_num, cam_num))

            if not self.depth_image:
                return color_images

        if self.depth_image:
            depth_images = []
            for cam_num in range(len(self.selected_views)):
                depth_images.append(self._get_depth_image(traj_num, state_num, cam_num))

            if not self.color_image:
                return depth_images

        rgbd_images = [torch.cat([color_images[cam_num], depth_images[cam_num]], dim = 0) for cam_num in range(len(self.selected_views))]
        return rgbd_images

class ImageActionDataset(ImageDataset):
    def __init__(
        self, 
        data_path, 
        selected_views, 
        image_type, 
        absolute,
        demos_list = None,
        transforms = None
    ):
        super().__init__(data_path, selected_views, image_type, demos_list, transforms)
    
        self.state_offset = 1
        images_path = os.path.join(data_path, 'images')
        self._load_image_paths(images_path)
        self._get_trajectory_data()

        actions_path = os.path.join(data_path, 'actions')
        self.actions = load_tensors(actions_path, self.demos)

        if absolute:
            states_path = os.path.join(data_path, 'states')
            states = load_tensors(states_path, self.demos)
            self.actions = self.actions + states

    def __getitem__(self, idx):
        traj_num, state_num = self._get_traj_state_idx(idx)

        if self.color_image:
            color_images = []
            for cam_num in range(len(self.selected_views)):
                color_images.append(self._get_color_image(traj_num, state_num, cam_num))

            if not self.depth_image:
                return color_images, self.actions[idx]

        if self.depth_image:
            depth_images = []
            for cam_num in range(len(self.selected_views)):
                depth_images.append(self._get_depth_image(traj_num, state_num, cam_num))

            if not self.color_image:
                return depth_images, self.actions[idx]

        rgbd_images = [torch.cat([color_images[cam_num], depth_images[cam_num]], dim = 0) for cam_num in range(len(self.selected_views))]
        return rgbd_images, self.actions[idx]

def load_tensors(path, demos_list):
    if demos_list is None:
        tensor_names = os.listdir(path)
        tensor_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
        tensor_paths = [os.path.join(path, tensor_name) for tensor_name in tensor_names]
    else:
        demos_list.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
        tensor_paths = [os.path.join(path, '{}.pth'.format(tensor_name)) for tensor_name in demos_list]            

    tensor_array = []
    for tensor_path in tensor_paths:
        tensor_array.append(torch.load(tensor_path))

    return torch.cat(tensor_array, dim = 0)


def get_image_dataset(
    data_path,
    selected_views,
    image_type,
    demos_list,
    mean_tensors,
    std_tensors,
    dataset_type,
    absolute = None
):
    color_image, depth_image = False, False
    if image_type == 'color':
        color_image = True
    elif image_type == 'depth':
        depth_image = True
    elif image_type == 'all':
        color_image = True
        depth_image = True

    transforms = {
        'color_image': [],
        'depth_image': []
    }

    for cam_num in selected_views:
        if color_image:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(
                    mean = mean_tensors[cam_num - 1],
                    std = std_tensors[cam_num - 1]
                )
            ])
            transforms['color_image'].append(transform)

        if depth_image:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(
                    mean = mean_tensors[cam_num - 1 + NUM_CAMS],
                    std = std_tensors[cam_num - 1 + NUM_CAMS]
                )
            ])
            transforms['depth_image'].append(transform)

    if dataset_type == 'pretrain':
        return ImageSSLDataset(
            data_path = data_path,
            selected_views = selected_views, 
            image_type = image_type,
            demos_list = demos_list, 
            transforms = transforms
        )
    elif dataset_type == 'action':
        return ImageActionDataset(
            data_path = data_path, 
            selected_views = selected_views, 
            image_type = image_type,
            demos_list = demos_list, 
            absolute = absolute, 
            transforms = transforms
        )
    else:
        raise NotImplementedError()