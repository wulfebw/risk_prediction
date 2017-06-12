
import numpy as np
import gym
from gym.envs.registration import register
import logging
import sys 
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('..')

import envs.julia_env
from . import normalizing_wrapper

def create_env(config):
    # register conditional on config, so have to wait until this point
    register(
        id=config.env_id,
        entry_point='envs.julia_env:JuliaEnv',
        max_episode_steps=config.max_timesteps,
        kwargs={
            'env_id': config.env_id.replace('-v0',''),
            'env_params': config.__dict__,
            'julia_envs_path': '../../julia/RiskEnvs.jl/RiskEnvs.jl'
        }
    )
    env = gym.make(config.env_id)
    if config.normalization_type == 'std':
        env = normalizing_wrapper.NormalizingWrapper(env)
    elif config.normalization_type == 'range':
        env = normalizing_wrapper.RangeNormalizingWrapper(env)
    return env

def get_julia_env(env):
    julia_env = env
    while type(julia_env) != envs.julia_env.JuliaEnv:
        julia_env = julia_env.env
    return julia_env