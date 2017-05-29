
import gym
import numpy as np

class NormalizingWrapper(gym.Wrapper):
    def __init__(self, env, std_bound=5, stationary_after=1e7, clip_after=1e4,
            global_max_clip=300):
        super(NormalizingWrapper, self).__init__(env)
        assert len(np.shape(env.observation_space.high)) == 1, 'only single dim spaces implemented'
        self.obs_dim = len(env.observation_space.high)
        self.std_bound = std_bound
        self.stationary_after = stationary_after
        self.clip_after = clip_after
        self.global_max_clip = global_max_clip

        self.means = np.zeros(self.obs_dim)
        self.diff_sums = np.ones(self.obs_dim)
        self.stds = np.ones(self.obs_dim)
        self.count = 0.

    def _clip(self, obs):
        if self.count > self.clip_after:
            bounds = self.means + self.std_bound * self.stds
            obs = np.clip(obs, -self.global_max_clip, self.global_max_clip)
            obs = np.clip(obs, -bounds, bounds)
        return obs

    def _normalize(self, obs):
        self.count += 1

        # clip the values 
        obs = self._clip(obs)

        # update statistics
        if self.count < self.stationary_after:
            tmp_means = self.means[:]
            diff = obs - tmp_means
            self.means += diff / self.count
            new_diff = obs - self.means
            self.diff_sums += diff * new_diff
            self.stds = np.sqrt(self.diff_sums / max(1, self.count - 2))
        obs = (obs - self.means) / self.stds
        return obs

    def _reset(self):
        return self._normalize(self.env.reset())

    def _step(self, action):
        obs, r, done, info = self.env.step(action)
        return self._normalize(obs), r, done, info


if __name__ == '__main__':

    import copy
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    env = gym.make('CartPole-v0')
    env = NormalizingWrapper(env)

    n_steps = 50000
    means = []
    stds = []
    x = env.reset()
    for step in range(n_steps):
        _, _, done, _ = env.step(env.action_space.sample())
        if done: 
            env.reset()
        means.append(copy.deepcopy(env.means))
        stds.append(copy.deepcopy(env.stds))

    for vals in [means, stds]:
        vals = np.array(vals)
        n_steps, obs_dim = vals.shape
        for var in range(obs_dim):
            plt.plot(range(n_steps), vals[:, var], c='green', alpha=.5)
            plt.show()
