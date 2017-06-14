
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

if __name__ == '__main__':
    env = SeqSumDebugEnv()
    x = env.reset()
    print(x)
    nx, r, t, _ = env.step(None)
    print(nx, r, t)
    nx, r, t, _ = env.step(None)
    print(nx, r, t)