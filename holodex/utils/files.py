import os
import pickle 
import yaml
import numpy as np
import holodex

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    else:
        return False

def get_path_in_package(relative_path):
    return os.path.join(holodex.__path__[0], relative_path)

def store_pickle_data(path, dictionary):
    with open(path, 'ab') as file:
        pickle.dump(dictionary, file)

def get_pickle_data(path):    
    with open(path, 'rb') as file:
        return pickle.load(file)

def check_file(path):
    return os.path.exists(path)

def get_npz_data(path):
    return np.load(path)

def get_yaml_data(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)