import os
import torch
from torch.utils.data import DataLoader
from holodex.datasets.image import ImageActionDataset
from tqdm import tqdm
from copy import deepcopy as copy

def load_encoder_representations(
    data_path, 
    encoder, 
    selected_view, 
    transform, 
    demo_list = None
):
    transform_dict = {
        'color_image': [transform],
        'depth_image': []
    }

    dataset = ImageActionDataset(
        data_path = data_path, 
        selected_views = [selected_view], 
        image_type = 'color',
        demos_list = demo_list,
        absolute = None, 
        transforms = transform_dict
    )
    encoder = encoder.cuda()
    dataloader = DataLoader(dataset = dataset, batch_size = 64, shuffle = False)

    input_representation_array = []
    print('Obtaining all the representations:')
    for input_images, _ in tqdm(dataloader):
        input_images = input_images[0].cuda().float()
        input_representation = encoder(input_images).detach()
        input_representation_array.append(input_representation)

    input_representations = torch.cat(input_representation_array, dim = 0)
    torch.save(input_representations, os.path.join(data_path, 'input_representations.pt'))

    encoder = encoder.cpu()
    return input_representations

def load_tensors(path, demo_list):
    if demo_list is None:
        demo_list = os.listdir(path)
    else:
        demo_list = ['{}.pth'.format(demo_name) for demo_name in demo_list]

    demo_list.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
    tensor_paths = [os.path.join(path, tensor_name) for tensor_name in demo_list]

    tensor_array = []
    for tensor_path in tensor_paths:
        tensor_array.append(torch.load(tensor_path))

    return torch.cat(tensor_array, dim = 0)

def load_image_paths(data_path, selected_view, demo_list = None):
    images_path = os.path.join(data_path, 'images')

    if demo_list is None:
        demo_list = os.listdir(images_path)

    demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    input_image_paths = []
    output_image_paths = []
    traj_idx = []
    cumm_len = [0]
    
    for idx, demo in enumerate(demo_list):
        image_names = os.listdir(os.path.join(images_path, demo, 'camera_{}_color_image'.format(selected_view)))
        image_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
        
        cumm_value = cumm_len[-1] + len(image_names) - 1
        cumm_len.append(cumm_value)

        for _ in range(len(image_names) - 1):
            traj_idx.append(idx)

        demo_input_image_paths = [os.path.join(images_path, demo, 'camera_{}_color_image'.format(selected_view), image_names[image_num]) for image_num in range(len(image_names) - 1)]
        demo_output_image_paths = [os.path.join(images_path, demo, 'camera_{}_color_image'.format(selected_view), image_names[image_num + 1]) for image_num in range(len(image_names) - 1)]
        input_image_paths.append(copy(demo_input_image_paths))
        output_image_paths.append(copy(demo_output_image_paths))

    return input_image_paths, output_image_paths, cumm_len, traj_idx

def get_traj_state_idxs(idx, traj_idx, cumm_len):
    traj_idx = traj_idx[idx]
    path_idx = idx - cumm_len[traj_idx]
    return traj_idx, path_idx