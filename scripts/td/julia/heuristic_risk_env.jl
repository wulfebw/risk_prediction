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

        # generator
        roadway = gen_stadium_roadway(
            params["num_lanes"], 
            length = params["roadway_length"], 
            radius = params["roadway_radius"]
        )
        roadway_gen = StaticRoadwayGenerator(roadway)
        scene_gen = HeuristicSceneGenerator(
            params["min_num_veh"], 
            params["max_num_veh"], 
            params["min_base_speed"],
            params["max_base_speed"],
            params["min_vehicle_length"],
            params["max_vehicle_length"],
            params["min_vehicle_width"], 
            params["max_vehicle_width"],
            params["min_init_dist"]
        )
        behavior_gen = build_behavior_generator(params)
        gen = FactoredGenerator(roadway_gen, scene_gen, behavior_gen)

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
            0,
            params
        )
    end
end
