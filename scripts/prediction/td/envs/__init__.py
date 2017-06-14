

from gym.envs.registration import register
import os
import sys

path = os.path.join(os.path.dirname(__file__))
sys.path.append(os.path.abspath(path))

register(
    id='SeqSumDebugEnv-v0',
    entry_point='debug_envs:SeqSumDebugEnv',
    max_episode_steps=10,
    kwargs = {
        'horizon': 4
    },
)