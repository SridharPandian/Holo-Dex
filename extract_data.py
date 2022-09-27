import os
import hydra
from dexVR.data import FilterData, ColorImageExtractor, DepthImageExtractor, StateExtractor, ActionExtractor
from dexVR.utils.files import *

@hydra.main(version_base = '1.2', config_path='configs', config_name='demo_extract')
def main(configs):
    if not configs.ssl_data:
        configs.target_path = os.path.join(configs.target_path, 'filtered') # When we sample and extract min_action data 
    else:
        configs.filter_path = configs.storage_path # Use the raw data as the filtered data
        configs.target_path = os.path.join(configs.target_path, 'ssl') # When we extract every datapoint for training the encoder

        # We extract only the image data
        configs.sample = False
        configs.states = False
        configs.actions = False

    make_dir(configs.target_path)
    make_dir(configs.filter_path)
    
    if not os.path.exists(configs.storage_path):
        print("No data available.")

    print(f"\nUsing min-action distance: {configs['min_action_distance']}")

    if configs.sample:
        print("\nFiltering demonstrations!")
        data_filter = FilterData(data_path = configs.storage_path, delta = 0.01 * configs['min_action_distance'])        
        data_filter.filter(configs.filter_path)

    if configs.color_images:
        print("\nExtracting color images!")
        extractor = ColorImageExtractor(configs.filter_path, num_cams = configs.num_cams, image_size = configs.image_parameters.image_size, crop_sizes = configs.image_parameters.crop_sizes)
        images_path = os.path.join(configs.target_path, 'images')
        make_dir(images_path)
        extractor.extract(images_path)

    if configs.depth_images:
        print("\nExtracting depth images!")
        extractor = DepthImageExtractor(configs.filter_path, num_cams = configs.num_cams, image_size = configs.image_parameters.image_size, crop_sizes = configs.image_parameters.crop_sizes)
        images_path = os.path.join(configs.target_path, 'images')
        make_dir(images_path)
        extractor.extract(images_path)

    if configs.states:
        print("\nExtracting states!")
        extractor = StateExtractor(configs.filter_path)
        states_path = os.path.join(configs.target_path, 'states')
        make_dir(states_path)
        extractor.extract(states_path)

    if configs.actions:
        print("\nExtracting actions!")
        extractor = ActionExtractor(configs.filter_path)
        actions_path = os.path.join(configs.target_path, 'actions')
        make_dir(actions_path)
        extractor.extract(actions_path)

if __name__ == '__main__':
    main()