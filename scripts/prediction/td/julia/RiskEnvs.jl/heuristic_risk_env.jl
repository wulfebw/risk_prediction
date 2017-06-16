export
    HeuristicRiskEnv

type HeuristicRiskEnv <: RiskEnv
    gen::FactoredGenerator
    feature_ext::AbstractFeatureExtractor
    target_ext::AbstractFeatureExtractor

    rec::SceneRecord
    scene::Scene
    roadway::Roadway
    models::Dict{Int, DriverModel}

    sim_time::Float64
    prime_time::Float64
    ego_index::Int

    n_local_steps::Int
    n_episodes::Int
    seed::Int

    params::Dict

    function HeuristicRiskEnv(params::Dict)
        # set some values specific to this env
        params["generator_type"] = "factored"

        # extractors and generator
        feature_ext = build_extractor(params)
        target_ext = build_target_extractor(params)
        gen = build_factored_generator(params)

        # rec, scene, roadway, models
        Δt = .1
        rec = build_scene_record(params, Δt)
        scene = Scene(params["max_num_vehicles"])
        roadway = build_roadway(params)
        models = Dict{Int, DriverModel}()
        return new(
            gen, 
            feature_ext, 
            target_ext, 
            rec, 
            scene, 
            roadway, 
            models,
            params["sampling_time"], 
            params["prime_time"], 
            0, 
            0, 
            0,
            0,
            params
        )
    end
end
