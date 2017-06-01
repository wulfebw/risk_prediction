export
    RiskEnv,
    reset,
    step, 
    observation_space_spec,
    action_space_spec,
    render,
    obs_var_names,
    reward_names,
    build_feature_extractor,
    build_target_extractor,
    build_behavior_generator,
    build_scene_record

abstract RiskEnv <: Env

function Base.reset(env::RiskEnv)
    env.n_episodes += 1
    empty!(env.rec)
    seed = rand(1:typemax(Int))
    rand!(env.gen, env.roadway, env.scene, env.models, seed)
    # simulate for prime time to populate features
    simulate!(Any, env.rec, env.scene, env.roadway, env.models, env.prime_time)
    target_veh_id = get_target_vehicle_id(env.gen)
    if target_veh_id == nothing
        # random vehicle
        env.ego_index = rand(1:length(env.scene))
    else
        env.ego_index = findfirst(env.scene, target_veh_id)
    end
    pull_features!(
        env.feature_ext, env.rec, env.roadway, env.ego_index, env.models)
    return env.feature_ext.features
end

function Base.step(env::RiskEnv)
    env.n_local_steps += 1
    simulate!(Any, env.rec, env.scene, env.roadway, env.models, env.sim_time,
        update_first_scene = false)
    pull_features!(
        env.feature_ext, env.rec, env.roadway, env.ego_index, env.models)
    # TODO: this should pull targets from the past sim_timesteps
    pull_features!(
        env.target_ext, env.rec, env.roadway, env.ego_index)
    weights = get_weights(env.gen)
    if weights != nothing
        weight = get_weights(env.gen)[env.ego_index]
    else
        weight = 1.
    end
    done = any(env.target_ext.features .> 0)
    return (
        env.feature_ext.features, 
        env.target_ext.features, 
        done, 
        Dict("weight"=>weight)
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

observation_space_spec(env::RiskEnv) = length(env.feature_ext), "Box"
action_space_spec(env::RiskEnv) = (0,), "None"

obs_var_names(env::RiskEnv) = feature_names(env.feature_ext)
reward_names(env::RiskEnv) = feature_names(env.target_ext)


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
