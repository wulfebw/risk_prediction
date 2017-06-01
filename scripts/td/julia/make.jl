export
    make

function make(env_id::String, env_params::Dict)
    if env_id == "DeterministicSingleStepDebugEnv"
        return DeterministicSingleStepDebugEnv(env_params)
    elseif env_id == "BayesNetRiskEnv"
        return BayesNetRiskEnv(env_params)
    elseif env_id == "HeuristicRiskEnv"
        return HeuristicRiskEnv(env_params)
    else
        throw(ArgumentError("Invalid env_id: $(env_id)"))
    end
end