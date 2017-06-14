
import gym
from gym import spaces
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

from prediction import normalizing_wrapper

class TestEnv(gym.Env):
    def __init__(self, low, high):
        self.action_space = spaces.Discrete(0)
        self.observation_space = spaces.Box(low=low, high=high)
    def _reset(self):
        return 0
    def _step(self, action):
        return 0, 0, False, {}

class TestRangeNormalizingWrapper(unittest.TestCase):

    def test_normalize(self):
        low = np.array([0,-3])
        high = np.array([1,-1])
        env = TestEnv(low=low, high=high)
        wrapper = normalizing_wrapper.RangeNormalizingWrapper(env)

        x = [0,0]
        actual = wrapper._normalize(x)
        expected = [-1, 1]
        np.testing.assert_array_almost_equal(actual, expected)

        x = [2,-4]
        actual = wrapper._normalize(x)
        expected = [1, -1]
        np.testing.assert_array_almost_equal(actual, expected)

        x = [.5,-2]
        actual = wrapper._normalize(x)
        expected = [0, 0]
        np.testing.assert_array_almost_equal(actual, expected)

        # multidim
        x = np.array([[0,-1],[1,-3]])
        actual = wrapper._normalize(x)
        expected = [[-1, 1],[1,-1]]
        np.testing.assert_array_almost_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()