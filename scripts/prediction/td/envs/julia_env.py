
import gym
from gym import spaces
import julia
import numpy as np
import os

def build_observation_space(shape, space_type):
    if space_type == 'Box':
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape)
    elif space_type == 'Discrete':
        assert len(shape) == 1, 'invalid shape for Discrete space'
        return spaces.Discrete(shape)
    else:
        raise(ValueError('space type not implemented: {}'.format(space_type)))

def build_action_space(shape, space_type):
    return gym.spaces.Discrete(0)

class JuliaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, env_params, julia_envs_path):

        # initialize julia
        self.j = julia.Julia()
        self.j.eval('include(\"{}\")'.format(
            os.path.join(os.path.expanduser('~/.juliarc.jl'))))
        self.j.eval('include(\"{}\")'.format(julia_envs_path))
        self.j.using("RiskEnvs")

        # initialize environment
        self.env = self.j.make(env_id, env_params)
        self.observation_space = build_observation_space(
            *self.j.observation_space_spec(self.env))
        self.action_space = build_action_space(
            *self.j.action_space_spec(self.env))

    def _reset(self):
        return self.j.reset(self.env)

    def _step(self, action):
        if action is None:
            return self.j.step(self.env)
        else:
            return self.j.step(self.env, action)

    def _render(self, mode='human', close=False):
        return self.j.render(self.env)

    def reward_names(self):
        return self.j.reward_names(self.env)
        
    def obs_var_names(self):
        return self.j.obs_var_names(self.env)



