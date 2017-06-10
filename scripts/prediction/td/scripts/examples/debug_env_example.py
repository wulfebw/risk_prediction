
import sys 
sys.path.append('../../')

import envs.julia_env

env = envs.julia_env.JuliaEnv(
    env_id = 'DeterministicSingleStepDebugEnv', 
    env_params = {}, 
    julia_envs_path = '../../julia/RiskEnvs.jl/RiskEnvs.jl'
)

x = env.reset()
print(x)
nx, r, done, _ = env.step(None)
print(nx)
print(r)
print(done)
