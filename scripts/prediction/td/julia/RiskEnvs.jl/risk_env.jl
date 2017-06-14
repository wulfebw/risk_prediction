export
    RiskEnv,
    reset,
    step, 
    observation_space_spec,
    action_space_spec,
    render,
    obs_var_names,
    reward_names,
    get_features,
    get_targets,
    get_weight,
    get_ego_index,
    build_feature_extractor,
    build_target_extractor,
    build_behavior_generator,
    build_scene_record,
    build_evaluator

abstract RiskEnv <: Env

function Base.reset(env::RiskEnv)
    env.n_episodes += 1
    empty!(env.rec)
    env.seed = rand(1:typemax(Int))
    rand!(env.gen, env.roadway, env.scene, env.models, env.seed)
    # simulate for prime time to populate features
    simulate!(Any, env.rec, env.scene, env.roadway, env.models, env.prime_time)
    env.ego_index = get_ego_index(env)
    return get_features(env)
end

function Base.step(env::RiskEnv)
    env.n_local_steps += 1
    simulate!(Any, env.rec, env.scene, env.roadway, env.models, env.sim_time,
        update_first_scene = false)    
    
    features = get_features(env)
    targets = get_targets(env)
    done = any(env.target_ext.features .> 0)
    return (features, targets, done,
        Dict("weight"=>get_weight(env), "seed"=>env.seed)
    )
end

function render(env::RiskEnv; zoom::Float64 = 7.5)
    veh_id = env.ego_index
    carcolors = Dict{Int,Colorant}()
    for veh in env.scene
        carcolors[veh.id] = veh.id == veh_id ? colorant"red" : colorant"green"
    end
    cam = AutoViz.CarFollowCamera{Int}(veh_id, zoom)
    stats = [
        CarFollowingStatsOverlay(veh_id, 2), 
        NeighborsOverlay(veh_id, textparams = TextParams(x = 600, y_start=300))
    ]

    frame = render(env.scene, env.roadway, stats, cam = cam, car_colors = carcolors)

    if !isdir(env.params["viz_dir"])
        mkdir(env.params["viz_dir"])
    end
    ep_dir = joinpath(env.params["viz_dir"], "episode_$(env.n_episodes)")
    if !isdir(ep_dir)
        mkdir(ep_dir)
    end
    write_to_png(frame,  joinpath(ep_dir, "step_$(env.n_local_steps).png"))
    return frame
end

function observation_space_spec(env::RiskEnv)
    info = feature_info(env.feature_ext)
    feature_dim = length(env.feature_ext)
    high = zeros(feature_dim)
    low = zeros(feature_dim)
    for (i, name) in enumerate(feature_names(env.feature_ext))
        high[i] = info[name]["high"]
        low[i] = info[name]["low"]
    end
    return (feature_dim,), "Box", Dict("high"=>high, "low"=>low)
end
action_space_spec(env::RiskEnv) = (0,), "None"

obs_var_names(env::RiskEnv) = feature_names(env.feature_ext)
reward_names(env::RiskEnv) = feature_names(env.target_ext)

function AutoRisk.get_features(env::RiskEnv)
    pull_features!(env.feature_ext, env.rec, env.roadway, env.ego_index, env.models)
    return env.feature_ext.features
end

function AutoRisk.get_targets(env::RiskEnv)
    # TODO: this should pull targets from the past sim_timesteps
    pull_features!(env.target_ext, env.rec, env.roadway, env.ego_index)
    return env.target_ext.features
end

function get_weight(env::RiskEnv)
    weights = get_weights(env.gen)
    if weights != nothing
        weight = weights[env.ego_index]
    else
        weight = 1.
    end
    return weight
end

function get_ego_index(env::RiskEnv)
    target_veh_id = get_target_vehicle_id(env.gen)
    if target_veh_id == nothing
        # random vehicle
        ego_index = rand(1:length(env.scene))
    else
        ego_index = findfirst(env.scene, target_veh_id)
    end
    return ego_index
end

function build_feature_extractor(params::Dict)
    feature_ext = MultiFeatureExtractor(
        extract_core = params["extract_core"],
        extract_temporal = params["extract_temporal"],
        extract_well_behaved = params["extract_well_behaved"],
        extract_neighbor = params["extract_neighbor"],
        extract_behavioral = params["extract_behavioral"],
        extract_neighbor_behavioral = params["extract_neighbor_behavioral"],
        extract_car_lidar = params["extract_car_lidar"],
        extract_car_lidar_range_rate = params["extract_car_lidar_range_rate"],
        extract_road_lidar = params["extract_road_lidar"]
    )
    return feature_ext
end

function build_target_extractor(params::Dict)
    target_ext = TargetExtractor(
        hard_brake_threshold = params["hard_brake_threshold"],
        hard_brake_n_past_frames = params["hard_brake_n_past_frames"],
        ttc_threshold = params["ttc_threshold"]
    )
    return target_ext
end

function build_behavior_generator(params::Dict)
    min_p = get_passive_behavior_params(
        lon_σ = params["lon_accel_std_dev"], 
        lat_σ = params["lat_accel_std_dev"], 
        err_p_a_to_i = params["err_p_a_to_i"],
        err_p_i_to_a = params["err_p_i_to_a"],
        overall_response_time = params["overall_response_time"]
    )
    normal = get_normal_behavior_params(
        lon_σ = params["lon_accel_std_dev"], 
        lat_σ = params["lat_accel_std_dev"], 
        err_p_a_to_i = params["err_p_a_to_i"],
        err_p_i_to_a = params["err_p_i_to_a"],
        overall_response_time = params["overall_response_time"]
    )
    max_p = get_aggressive_behavior_params(
        lon_σ = params["lon_accel_std_dev"], 
        lat_σ = params["lat_accel_std_dev"], 
        err_p_a_to_i = params["err_p_a_to_i"],
        err_p_i_to_a = params["err_p_i_to_a"],
        overall_response_time = params["overall_response_time"]
    )
    if params["heuristic_behavior_type"] == "normal"
        beh_params = [normal]
        weights = StatsBase.Weights([1.])
        beh_gen = PredefinedBehaviorGenerator(beh_params, weights)
    else
        beh_gen = CorrelatedBehaviorGenerator(min_p, max_p)
    end
    
    return beh_gen
end

function build_scene_record(params::Dict, Δt::Float64)
    rec = SceneRecord(
        Int(ceil(params["max_timesteps"] * params["sim_timesteps"])) + params["prime_timesteps"], 
        Δt, 
        params["max_num_vehicles"]
    )
    return rec
end

function build_evaluator(
        params::Dict, 
        feature_ext::AbstractFeatureExtractor,
        target_ext::AbstractFeatureExtractor,
        Δt::Float64
    )
    rec = build_scene_record(params, Δt)
    features = Array{Float64}(params["n_monte_carlo_runs"], length(feature_ext))
    targets = Array{Float64}(params["n_monte_carlo_runs"], length(target_ext))
    agg_targets = Array{Float64}(length(target_ext))

    # priming is done in the reset function
    prime_time = 0.
    sampling_time = params["sim_timesteps"] * Δt
    eval = MonteCarloEvaluator(
        feature_ext, 
        target_ext, 
        params["n_monte_carlo_runs"], 
        prime_time, 
        sampling_time,
        false, 
        rec, 
        features, 
        targets, 
        agg_targets
    )
    return eval
end
