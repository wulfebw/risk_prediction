export
    make

function make(env_id::String, env_params::Dict)
    try
        if env_id == "DeterministicSingleStepDebugEnv"
            return DeterministicSingleStepDebugEnv(env_params)
        elseif env_id == "BayesNetRiskEnv"
            return BayesNetRiskEnv(env_params)
        elseif env_id == "HeuristicRiskEnv"
            return HeuristicRiskEnv(env_params)
        elseif env_id == "MonteCarloRiskEnv"
            return MonteCarloRiskEnv(env_params)
        else
            throw(ArgumentError("Invalid env_id: $(env_id)"))
        end
    catch e
        println("exception raised while making environment")
        println(backtrace(e))
        println(e)
        rethrow(e)
    end
end