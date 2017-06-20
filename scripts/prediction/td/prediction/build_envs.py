
import numpy as np
import gym
from gym.envs.registration import register, registry
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
    if config.env_id not in registry.env_specs.keys():
        register(
            id=config.env_id,
            entry_point='envs.julia_env:JuliaEnv',
            max_episode_steps=config.max_timesteps,
            kwargs={
                'env_id': config.julia_env_id.replace('-v0',''),
                'env_params': config.__dict__,
                'julia_envs_path': '../../julia/RiskEnvs.jl/RiskEnvs.jl'
            }
        )
    env = gym.make(config.env_id)
    env = normalizing_wrapper.RangeNormalizingWrapper(env)
    return env

def get_subenv_by_type(env, envtype):
    while hasattr(env, 'env'):
        if type(env) == envtype:
            return env
        else:
            env = env.env
    return None

def get_julia_subenv(env):
    return get_subenv_by_type(env, envs.julia_env.JuliaEnv)

def get_normalizing_subenv(env):
    return get_subenv_by_type(env, normalizing_wrapper.RangeNormalizingWrapper)

def get_target_names(env, target_dim):
    julia_env = get_julia_subenv(env)
    if julia_env is None:
        return range(target_dim)
    else:
        return julia_env.reward_names()

def get_obs_var_names(env):
    julia_env = get_julia_subenv(env)
    if julia_env is None:
        return range(env.observation_space.shape[0])
    else:
        return julia_env.obs_var_names()
