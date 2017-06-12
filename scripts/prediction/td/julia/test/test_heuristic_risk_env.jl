push!(LOAD_PATH, "../RiskEnvs.jl")
using RiskEnvs
using Base.Test

include("testing_params.jl")

@testset "HeuristicRiskEnv Tests" begin

    @testset "HeuristicRiskEnv.step Tests" begin
        srand(0)
        params = build_testing_params() 
        params["discount"] = 0.
        params["max_timesteps"] = 500
        params["hard_brake_threshold"] = 0.
        params["hard_brake_n_past_frames"] = 1
        env = HeuristicRiskEnv(params)
        x = reset(env)
        t = false
        r = [0,0,0,0,0]
        while !t
            nx, r, t, _ = step(env)
        end
        @test sum(r) > 0
    end

end

"""
test list:
- reset
- step
- render
- get_targets
- get_features
"""