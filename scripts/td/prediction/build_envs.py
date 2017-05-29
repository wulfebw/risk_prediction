
import numpy as np
import gym
from gym.envs.registration import register
import logging
import sys 
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('..')

def create_env(config):
    # register conditional on config, so have to wait until this point
    register(
        id=config.env_id,
        entry_point='envs.julia_env:JuliaEnv',
        max_episode_steps=config.max_timesteps,
        kwargs={
            'env_id': 'RiskEnv',
            'env_params': config.__dict__,
            'julia_envs_path': '../julia/JuliaEnvs.jl'
        }
    )
    env = gym.make(config.env_id)
    return env