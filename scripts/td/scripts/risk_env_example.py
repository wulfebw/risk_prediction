import gym 
from gym.envs.registration import register
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=5)
import sys 
import time 

sys.path.append('..')

import envs.julia_env
import configs.risk_env_config
config = configs.risk_env_config.Config()

register(
    id='RiskEnv-v0',
    entry_point='envs.julia_env:JuliaEnv',
    max_episode_steps=config.max_timesteps,
    kwargs={
        'env_id': 'RiskEnv',
        'env_params': config.__dict__,
        'julia_envs_path': '../julia/JuliaEnvs.jl'
    }
)

env = gym.make('RiskEnv-v0')
ep_rs = []
means = []
n_episodes = 10
st = time.time()
for ep in range(n_episodes):
    x = env.reset() 
    done = False
    t = 0
    ep_r = np.zeros(5)
    while not done and t < config.max_timesteps:
        t += 1
        x, r, done, _ = env.step(None)
        ep_r += r
        if (ep + 1) % 1 == 0:
            env.render()

    ep_rs.append(ep_r)
    means.append(np.mean(ep_rs, axis=0))
    print('episode: {} / {}\tavg_r: {}\ttime: {:.5f}'.format(
        ep, n_episodes, means[ep], time.time() - st))

means = np.array(means)
base_env = env
while type(base_env) != envs.julia_env.JuliaEnv:
    base_env = base_env.env
reward_names = base_env.reward_names()
print(reward_names)

for t in range(5):
    plt.plot(range(len(means[:,t])), means[:,t], label=reward_names[t])
plt.legend()
plt.show()