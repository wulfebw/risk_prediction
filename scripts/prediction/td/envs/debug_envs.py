
import copy
import numpy as np
import gym
from gym import spaces

class SeqSumDebugEnv(gym.Env):

    def __init__(self, obs_shape=(1,), horizon=2):
        self.obs_shape = obs_shape
        self.horizon = horizon
        self.action_space = spaces.Discrete(0)
        self.observation_space = spaces.Box(low=-1., high=1., shape=obs_shape)
        
    def _step(self, action=None):
        self.t -= 1
        t = self.t <= 0
        return [self.state], [self.state], t, {'weight':1}

    def _reset(self):
        self.t = self.horizon
        self.state = -1 if np.random.rand() < .5 else 1 
        return [self.state]

class RandObsConstRewardEnv(gym.Env):

    def __init__(self, obs_shape=(1,), horizon=2, reward=0, value_dim=2, rand_obs=True):
        self.obs_shape = obs_shape
        self.n_obs_el = np.prod(obs_shape)
        self.horizon = horizon
        self.reward = [reward] * value_dim
        self.action_space = spaces.Discrete(0)
        self.observation_space = spaces.Box(low=0., high=1., shape=obs_shape)
        self.obs_gen = np.random.randn if rand_obs else np.ones
        
    def _get_obs(self):
        return self.obs_gen(self.n_obs_el).reshape(self.obs_shape)

    def _step(self, action=None):
        self.t -= 1
        return (self._get_obs(), copy.deepcopy(self.reward), self.t <= 0, {'weight':1})

    def _reset(self):
        self.t = self.horizon
        return self._get_obs()

if __name__ == '__main__':
    env = SeqSumDebugEnv()
    x = env.reset()
    print(x)
    nx, r, t, _ = env.step(None)
    print(nx, r, t)
    nx, r, t, _ = env.step(None)
    print(nx, r, t)