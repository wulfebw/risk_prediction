export 
    DebugEnvDeterministicSingleStep,
    reset,
    step,
    observation_space_spec
    action_space_spec

# takes a single step from state 0, yielding reward 1, and terminates
type DeterministicSingleStepDebugEnv <: Env
    function DeterministicSingleStepDebugEnv(params::Dict)
        return new()
    end
end
Base.reset(env::DeterministicSingleStepDebugEnv) = [0.]
Base.step(env::DeterministicSingleStepDebugEnv) = ([0.], [1.], true, Dict())
observation_space_spec(env::DeterministicSingleStepDebugEnv) = (1,), "Discrete"
action_space_spec(env::DeterministicSingleStepDebugEnv) = (0,), "None"

