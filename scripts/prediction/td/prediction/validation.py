
import h5py
import os

from . import build_envs

class Dataset(object):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def next_batch(self):
        # generator for the data
        return ((x, y, w) for (x, y, w) in zip(self.x, self.y, self.w))

def build_dataset(config, env):
    if not os.path.exists(config.validation_dataset_filepath):
        return None

    # load the dataset
    infile = h5py.File(config.validation_dataset_filepath, 'r')
    x = infile['risk/features'][:config.max_validation_samples]
    y = infile['risk/targets'][:config.max_validation_samples]
    w = infile['risk/weights'][:config.max_validation_samples]

    # assume that the environment is a range normalizing one 
    # and use its normalize function
    # this way, can be sure that the normalization of the environment and 
    # the dataset are the same, and don't have to store normalization with 
    # the dataset
    normalizer = build_envs.get_normalizing_subenv(env)
    x = normalizer._normalize(x)

    # build and return dataset
    return Dataset(x,y,w)