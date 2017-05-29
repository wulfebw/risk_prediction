export
    RiskEnv,
    reset,
    step, 
    observation_space_spec,
    action_space_spec,
    render,
    obs_var_names,
    reward_names

type RiskEnv <: Env
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

    function RiskEnv(params::Dict)

        # generator
        min_p = get_passive_behavior_params(
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
        beh_gen = CorrelatedBehaviorGenerator(min_p, max_p)
        gen = BayesNetLaneGenerator(
            params["base_bn_filepath"], 
            params["prop_bn_filepath"], 
            params["num_veh_per_lane"], 
            beh_gen
        )

        # extraction
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
        target_ext = TargetExtractor()

        # rec, scene, roadway, models
        Δt = .1
        rec = SceneRecord(
            params["max_timesteps"] + params["prime_timesteps"], 
            Δt, 
            params["max_num_vehicles"]
        )
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

function Base.reset(env::RiskEnv)
    env.n_episodes += 1
    empty!(env.rec)
    seed = rand(1:typemax(Int))
    rand!(env.gen, env.roadway, env.scene, env.models, seed)
    # simulate for prime time to populate features
    simulate!(Any, env.rec, env.scene, env.roadway, env.models, env.prime_time)
    env.ego_index = findfirst(env.scene, env.gen.target_veh_id)
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
    pull_features!(
        env.target_ext, env.rec, env.roadway, env.ego_index)
    weight = get_weights(env.gen)[env.ego_index]
    done = any(env.target_ext.features[1:3] .> 0)
    return (
        env.feature_ext.features, 
        env.target_ext.features, 
        done, 
        Dict("weight"=>weight)
    )
end

function render(env::RiskEnv; zoom::Float64 = 7.5)
    veh_id = env.gen.target_veh_id
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

obs_var_names(env::Env) = feature_names(env.feature_ext)
reward_names(env::Env) = feature_names(env.target_ext)
