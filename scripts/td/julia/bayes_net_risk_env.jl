export
    BayesNetRiskEnv

type BayesNetRiskEnv <: RiskEnv
    gen::BayesNetLaneGenerator
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

    params::Dict

    function BayesNetRiskEnv(params::Dict)

        # generator
        beh_gen = build_behavior_generator(params)
        gen = BayesNetLaneGenerator(
            params["base_bn_filepath"], 
            params["prop_bn_filepath"], 
            params["num_veh_per_lane"], 
            beh_gen
        )

        # extraction
        feature_ext = build_feature_extractor(params)
        target_ext = build_target_extractor(params)

        # rec, scene, roadway, models
        Δt = .1
        rec = build_scene_record(params, Δt)
        scene = Scene(params["max_num_vehicles"])
        # the roadway is long enough s.t. the vehicles will not reach the end
        roadway = gen_straight_roadway(params["num_lanes"], 100000.)
        models = Dict{Int, DriverModel}()

        return new(
            gen, 
            feature_ext, 
            target_ext, 
            rec, 
            scene, 
            roadway, 
            models,
            params["sim_timesteps"] * Δt, 
            params["prime_timesteps"] * Δt, 
            0, 
            0, 
            0,
            params
        )
    end
end