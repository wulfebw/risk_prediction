export
    MonteCarloRiskEnv,
    reset,
    step,
    get_features

type MonteCarloRiskEnv <: RiskEnv
    gen::Generator
    eval::Evaluator
    scene::Scene
    roadway::Roadway
    models::Dict{Int, DriverModel}

    prime_time::Float64

    ego_index::Int
    n_local_steps::Int
    n_episodes::Int
    seed::Int

    params::Dict

    function MonteCarloRiskEnv(params::Dict)

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

        # evaluation
        Δt = .1
        feature_ext = build_feature_extractor(params)
        target_ext = build_target_extractor(params)
        eval = build_evaluator(params, feature_ext, target_ext, Δt)
        
        scene = Scene(params["max_num_vehicles"])
        # the roadway is long enough s.t. the vehicles will not reach the end
        roadway = gen_straight_roadway(params["num_lanes"], 100000.)
        models = Dict{Int, DriverModel}()

        return new(
            gen, 
            eval,  
            scene, 
            roadway, 
            models,
            params["prime_timesteps"] * Δt, 
            0, 
            0, 
            0,
            0,
            params
        )
    end
end

function Base.reset(env::MonteCarloRiskEnv)
    env.n_episodes += 1
    empty!(env.eval.rec)
    env.seed = rand(1:typemax(Int))
    rand!(env.gen, env.roadway, env.scene, env.models, env.seed)
    srand(env.eval, env.seed)
    # simulate for prime time to populate features
    simulate!(Any, env.eval.rec, env.scene, env.roadway, env.models, env.prime_time)
    env.ego_index = get_ego_index(env)
    return get_features(env)
end

function Base.step(env::MonteCarloRiskEnv)
    env.n_local_steps += 1
    features = get_features(env)
    done = evaluate!(env.eval, env.scene, env.models, env.roadway, env.ego_index)
    return (
        features, 
        get_targets(env.eval), 
        done, 
        Dict("weight"=>get_weight(env), "seed"=>env.seed)
    )
end

function AutoRisk.get_features(env::MonteCarloRiskEnv)
    pull_features!(env.eval.ext, env.eval.rec, env.roadway, env.ego_index, env.models)
    return env.eval.ext.features[:]
end
