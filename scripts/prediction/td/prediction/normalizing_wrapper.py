
import gym
from gym import spaces
import numpy as np

class RangeNormalizingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RangeNormalizingWrapper, self).__init__(env)

        assert type(env.observation_space) == spaces.Box
        high = np.array(env.observation_space.high)
        low = np.array(env.observation_space.low)
        assert not any(np.isnan(high))
        assert not any(np.isinf(high))
        assert not any(np.isnan(low))
        assert not any(np.isinf(low))

        self.high = high
        self.low = low
        self.mean = (self.high + self.low) / 2.
        self.half_range = (self.high - self.low) / 2

    def _normalize(self, obs):
        obs = np.clip(obs, self.low, self.high)
        obs = (obs - self.mean) / self.half_range
        return obs

    def _reset(self):
        return self._normalize(self.env.reset())

    def _step(self, action):
        obs, r, done, info = self.env.step(action)
        return self._normalize(obs), r, done, info

# class AdaptiveNormalizingWrapper(gym.Wrapper):
#     def __init__(self, env, std_bound=5, stationary_after=1e7, clip_after=1e5,
#             global_max_clip=200):
#         super(NormalizingWrapper, self).__init__(env)
#         assert len(np.shape(env.observation_space.high)) == 1, 'only single dim spaces implemented'
#         self.obs_dim = len(env.observation_space.high)
#         self.std_bound = std_bound
#         self.stationary_after = stationary_after
#         self.clip_after = clip_after
#         self.global_max_clip = global_max_clip

#         self.means = np.zeros(self.obs_dim)
#         self.diff_sums = np.ones(self.obs_dim)
#         self.stds = np.ones(self.obs_dim)
#         self.count = 0.

#     def _clip(self, obs):
#         if self.count > self.clip_after:
#             bounds = self.means + self.std_bound * self.stds
#             obs = np.clip(obs, -self.global_max_clip, self.global_max_clip)
#             obs = np.clip(obs, -bounds, bounds)
#         return obs

#     def _normalize(self, obs):
#         self.count += 1

#         # clip the values 
#         obs = self._clip(obs)

#         # check if observation is actually multiple observations
#         if len(np.shape(obs)) > 1:
#             multidim = True
#         else:
#             multidim = False

#         # update statistics
#         if self.count < self.stationary_after:
#             tmp_means = self.means[:]
#             diff = obs - tmp_means
#             if multidim:
#                 diff = diff.mean(0)
#             self.means += diff / self.count
#             new_diff = obs - self.means
#             if multidim:
#                 new_diff = new_diff.mean(0)
#             self.diff_sums += diff * new_diff
#             self.stds = np.sqrt(self.diff_sums / max(1, self.count - 2))
#         obs = (obs - self.means) / self.stds
#         return obs

#     def _reset(self):
#         return self._normalize(self.env.reset())

#     def _step(self, action):
#         obs, r, done, info = self.env.step(action)
#         return self._normalize(obs), r, done, info

# class AdaptiveRangeNormalizingWrapper(gym.Wrapper):
#     def __init__(self, env, stationary_after=1e6, clip_after=1e5, global_max_clip=200, eps=1e-8):
#         super(RangeNormalizingWrapper, self).__init__(env)
#         assert len(np.shape(env.observation_space.high)) == 1, 'only single dim spaces implemented'
#         self.obs_dim = len(env.observation_space.high)
#         self.stationary_after = stationary_after
#         self.clip_after = clip_after
#         self.global_max_clip = global_max_clip
#         self.eps = eps

#         self.high = np.zeros(self.obs_dim)
#         self.low = np.zeros(self.obs_dim)
#         self.count = 0.

#     def _clip(self, obs):
#         if self.count > self.clip_after:
#             obs = np.clip(obs, -self.global_max_clip, self.global_max_clip)
#             obs = np.clip(obs, self.low, self.high)
#         return obs

#     def _normalize(self, obs):
#         self.count += 1

#         # clip the values 
#         obs = self._clip(obs)

#         # check if observation is actually multiple observations
#         if len(np.shape(obs)) > 1:
#             multidim = True
#         else:
#             multidim = False

#         # update statistics
#         if self.count < self.stationary_after:
#             self.high = np.max(np.vstack((self.high, obs)), axis=0)
#             self.low = np.min(np.vstack((self.low, obs)), axis=0)

#         means = (self.high + self.low) / 2.
#         scales = (self.high - self.low) / 2. + self.eps
#         obs = (obs - means) / scales
#         obs = np.clip(obs, -1, 1)
#         return obs

#     def _reset(self):
#         return self._normalize(self.env.reset())

#     def _step(self, action):
#         obs, r, done, info = self.env.step(action)
#         return self._normalize(obs), r, done, info

# if __name__ == '__main__':

#     import copy
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt

#     env = gym.make('CartPole-v0')
#     wrapper_type = 'range_normalizing'

#     if wrapper_type == 'range_normalizing':
#         env = RangeNormalizingWrapper(env)
#     elif wrapper_type == 'normalizing':
#         env = NormalizingWrapper(env)

#     n_steps = 50000
#     means = []
#     stds = []
#     x = env.reset()
#     for step in range(n_steps):
#         _, _, done, _ = env.step(env.action_space.sample())
#         if done: 
#             env.reset()
#         if wrapper_type == 'normalizing':
#             means.append(copy.deepcopy(env.means))
#             stds.append(copy.deepcopy(env.stds))
#         elif wrapper_type == 'range_normalizing':
#             mean = (env.high + env.low) / 2.
#             scale = (env.high - env.low) / 2.
#             means.append(copy.deepcopy(mean))
#             stds.append(copy.deepcopy(scale))


#     for vals in [means, stds]:
#         vals = np.array(vals)
#         n_steps, obs_dim = vals.shape
#         for var in range(obs_dim):
#             plt.plot(range(n_steps), vals[:, var], c='green', alpha=.5)
#             plt.show()
