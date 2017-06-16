
import h5py
import numpy as np
import os

from . import build_envs

class Dataset(object):
    def __init__(self, x, y, w, feature_names=None):
        self.x = x
        self.y = y
        self.w = w
        self.feature_names = feature_names

    def next_batch(self):
        # generator for the data
        return ((x, y, w) for (x, y, w) in zip(self.x, self.y, self.w))

def transfer_dataset_settings_to_config(filepath, config):
    infile = h5py.File(config.validation_dataset_filepath, 'r')

    # transfer keys
    for k in infile['risk'].attrs.keys():
        
        try:
            v = infile['risk'].attrs[k]
        except Exception as e:
            print('exception occurred during transfer of key: {}; ignoring'.format(k))
        else:
            if k.startswith('utf8_'):
                k = k.replace('utf8_', '')
                v = ''.join(chr(i) for i in v)
                if v.lower() == 'true':
                    v = True
                elif v.lower() == 'false':
                    v = False
            elif isinstance(v, np.generic):
                v = np.asscalar(v)
            config.__dict__[k] = v

    return config

def build_dataset(config, env, normalize=True):
    if not os.path.exists(config.validation_dataset_filepath):
        return None

    # load the dataset
    infile = h5py.File(config.validation_dataset_filepath, 'r')
    x = infile['risk/features'][:config.max_validation_samples]
    y = infile['risk/targets'][:config.max_validation_samples]
    if 'risk/weights' in infile.keys():
        w = infile['risk/weights'][:config.max_validation_samples]
    else:
        w = np.ones(len(x))

    # assume that the environment is a range normalizing one 
    # and use its normalize function
    # this way, can be sure that the normalization of the environment and 
    # the dataset are the same, and don't have to store normalization with 
    # the dataset
    if normalize:
        normalizer = build_envs.get_normalizing_subenv(env)
        x = normalizer._normalize(x)

    # build and return dataset
    feature_names = infile['risk'].attrs['feature_names']
    return Dataset(x, y, w, feature_names)