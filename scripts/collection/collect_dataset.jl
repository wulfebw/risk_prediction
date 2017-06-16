using AutoRisk

function build_extractor(flags)
    extractor_type = flags["extractor_type"]
    if extractor_type == "heuristic"
        ext = MultiFeatureExtractor()
    elseif extractor_type == "multi"
        subexts::Vector{AbstractFeatureExtractor} = []
        if flags["extract_core"]
            push!(subexts, CoreFeatureExtractor())
        end
        if flags["extract_temporal"]
            push!(subexts, TemporalFeatureExtractor())
        end
        if flags["extract_well_behaved"]
            push!(subexts, WellBehavedFeatureExtractor())
        end
        if flags["extract_neighbor"]
            push!(subexts, NeighborFeatureExtractor())
        end
        if flags["extract_behavioral"]
            push!(subexts, BehavioralFeatureExtractor())
        end
        if flags["extract_neighbor_behavioral"]
            push!(subexts, NeighborBehavioralFeatureExtractor())
        end
        if flags["extract_car_lidar"]
            push!(subexts, CarLidarFeatureExtractor(
                extract_carlidar_rangerate = 
                flags["extract_car_lidar_range_rate"]))
        end
        if flags["extract_road_lidar"]
            push!(subexts, RoadLidarFeatureExtractor())
        end
        ext = MultiFeatureExtractor(subexts)
    else
        throw(ArgumentError(
            "invalid extractor_type $(extractor_type)"))
    end
    return ext
end

function build_factored_generator(flags)
    roadway_type = flags["roadway_type"]
    roadway_length = flags["roadway_length"]
    roadway_radius = flags["roadway_radius"]
    min_num_veh = flags["min_num_vehicles"]
    max_num_veh = flags["max_num_vehicles"]
    min_base_speed = flags["min_base_speed"]
    max_base_speed = flags["max_base_speed"]
    min_vehicle_length = flags["min_vehicle_length"]
    max_vehicle_length = flags["max_vehicle_length"]
    min_vehicle_width = flags["min_vehicle_width"]
    max_vehicle_width = flags["max_vehicle_width"]
    min_init_dist = flags["min_init_dist"]
    num_lanes = flags["num_lanes"]
    behavior_type = flags["behavior_type"]
    heuristic_behavior_type = flags["heuristic_behavior_type"]
    lon_accel_std_dev = flags["lon_accel_std_dev"]
    lat_accel_std_dev = flags["lat_accel_std_dev"]
    overall_response_time = flags["overall_response_time"]
    lon_response_time = flags["lon_response_time"]
    err_p_a_to_i = flags["err_p_a_to_i"]
    err_p_i_to_a = flags["err_p_i_to_a"]

    # roadway gen
    if roadway_type == "straight"
        roadway = gen_straight_roadway(num_lanes, roadway_length)
    else
        roadway = gen_stadium_roadway(num_lanes, length = roadway_length, 
            radius = roadway_radius)
    end
    roadway_gen = StaticRoadwayGenerator(roadway)

    # scene gen
    scene = Scene(max_num_veh)
    scene_gen = HeuristicSceneGenerator(
        min_num_veh, 
        max_num_veh, 
        min_base_speed,
        max_base_speed,
        min_vehicle_length,
        max_vehicle_length,
        min_vehicle_width, 
        max_vehicle_width,
        min_init_dist)

    passive = get_passive_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                lon_response_time = lon_response_time,
                overall_response_time = overall_response_time,
                err_p_a_to_i = err_p_a_to_i,
                err_p_i_to_a = err_p_i_to_a)
    normal = get_normal_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                lon_response_time = lon_response_time,
                overall_response_time = overall_response_time,
                err_p_a_to_i = err_p_a_to_i,
                err_p_i_to_a = err_p_i_to_a)
    aggressive = get_aggressive_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                lon_response_time = lon_response_time,
                overall_response_time = overall_response_time,
                err_p_a_to_i = err_p_a_to_i,
                err_p_i_to_a = err_p_i_to_a)

    if behavior_type == "heuristic"
        if heuristic_behavior_type == "aggressive"
            params = [aggressive]
            weights = StatsBase.Weights([1.])
            behavior_gen = PredefinedBehaviorGenerator(params, weights)
        elseif heuristic_behavior_type == "passive"
            params = [passive]
            weights = StatsBase.Weights([1.])
            behavior_gen = PredefinedBehaviorGenerator(params, weights)
        elseif heuristic_behavior_type == "normal"
            params = [normal]
            weights = StatsBase.Weights([1.])
            behavior_gen = PredefinedBehaviorGenerator(params, weights)
        elseif heuristic_behavior_type == "fixed_ratio"
            params = [aggressive, passive, normal]
            weights = StatsBase.Weights([.2,.3,.5])
            behavior_gen = PredefinedBehaviorGenerator(params, weights)
        elseif heuristic_behavior_type == "correlated"
            behavior_gen = CorrelatedBehaviorGenerator(passive, aggressive)
        else
            throw(ArgumentError(
                "invalid heuristic behavior type $(heuristic_behavior_type)"))
        end
    elseif behavior_type == "learned"
        behavior_gen = LearnedBehaviorGenerator(flags["driver_network_filepath"])
    else
        throw(ArgumentError(
                "invalid behavior type $(behavior_type)"))
    end
    gen = FactoredGenerator(roadway_gen, scene_gen, behavior_gen)
    return gen
end

function build_joint_generator(flags)
    prime_time = flags["prime_time"]
    sampling_time = flags["sampling_time"]
    sampling_period = flags["sampling_period"]
    lon_accel_std_dev = flags["lon_accel_std_dev"]
    lat_accel_std_dev = flags["lat_accel_std_dev"]
    overall_response_time = flags["overall_response_time"]
    lon_response_time = flags["lon_response_time"]
    err_p_a_to_i = flags["err_p_a_to_i"]
    err_p_i_to_a = flags["err_p_i_to_a"]

    # for the BN case, only use straight roadway
    # and since the BN assumes infinite roadway, increase the length
    flags["roadway_type"] = "straight"
    flags["roadway_length"] = 10000.

    # load the bayes nets
    d = JLD.load(flags["base_bn_filepath"]) 
    base_bn = d["bn"]
    base_sampler = AssignmentSampler(d["discs"])

    d = JLD.load(flags["prop_bn_filepath"]) 
    prop_bn = d["bn"]
    prop_sampler = AssignmentSampler(d["discs"])
    

    # we want the simulation to be valid for prime_time + sampling_time 
    # the first vehicle in the scene is invalid after the first timestep 
    # this is because it acts with no vehicles in front
    # the second vehicle is therefore invalid after the second timestep 
    # because the action it takes during the third timestep depends on the 
    # first vehicle at the second timestep, at which point it is invalid
    # in general vehicles are valid for (# vehicles in front - 1) timesteps
    # assume the target vehicle is 2nd to last in its row
    # then we need (prime_time + sampling_time + timestep) / sampling period
    # vehicles in front of the target vehicle and one behind it 
    num_veh_per_lane = Int(ceil((prime_time + sampling_time) / sampling_period))
    num_veh_per_lane += 2 # proposal vehicle + following vehicle
    num_veh_per_lane = min(num_veh_per_lane, flags["max_num_vehicles"])
    flags["max_num_vehicles"] = min(
        num_veh_per_lane * flags["num_lanes"], flags["max_num_vehicles"])
    min_p = get_passive_behavior_params(
        lon_σ = lon_accel_std_dev, 
        lat_σ = lat_accel_std_dev, 
        err_p_a_to_i = err_p_a_to_i,
        err_p_i_to_a = err_p_i_to_a,
        overall_response_time = overall_response_time
    )
    max_p = get_aggressive_behavior_params(
        lon_σ = lon_accel_std_dev, 
        lat_σ = lat_accel_std_dev, 
        err_p_a_to_i = err_p_a_to_i,
        err_p_i_to_a = err_p_i_to_a,
        overall_response_time = overall_response_time
    )
    behgen = CorrelatedBehaviorGenerator(min_p, max_p)
    gen = BayesNetLaneGenerator(base_bn, base_sampler, prop_bn, prop_sampler,
        num_veh_per_lane, behgen)
    return gen
end

function build_generator(flags)
    if flags["generator_type"] == "factored"
        gen = build_factored_generator(flags)
    elseif flags["generator_type"] == "joint"
        gen = build_joint_generator(flags)
    else
        throw(ArgumentError("invalid generator type $(flags["generator_type"])"))
    end
    return gen
end

function build_evaluator(flags, ext::AbstractFeatureExtractor)
    evaluator_type = flags["evaluator_type"]
    prediction_model_type = flags["prediction_model_type"]
    num_runs = flags["num_monte_carlo_runs"]
    prime_time = flags["prime_time"]
    sampling_time = flags["sampling_time"]
    veh_idx_can_change = flags["veh_idx_can_change"]
    sampling_period = flags["sampling_period"]
    max_num_veh = flags["max_num_vehicles"]
    target_dim = flags["target_dim"]
    feature_timesteps = flags["feature_timesteps"]
    feature_step_size = flags["feature_step_size"]
    bootstrap_discount = flags["bootstrap_discount"]
    hard_brake_threshold = flags["hard_brake_threshold"]
    hard_brake_n_past_frames = flags["hard_brake_n_past_frames"]
    ttc_threshold = flags["ttc_threshold"]

    feature_dim = length(ext)
    target_ext = TargetExtractor(
        hard_brake_threshold = hard_brake_threshold,
        hard_brake_n_past_frames = hard_brake_n_past_frames,
        ttc_threshold = ttc_threshold
        )
    # must prime for at least feature_timesteps
    # @assert feature_timesteps < prime_time / sampling_period
    if feature_timesteps > (prime_time / sampling_period)
        println("feature_timesteps < (prime_time / sampling_period)")
        println("you should be sure that you intended to do this")
        println("because some features may be invalid in this case")
    end
    max_num_scenes = Int(ceil((prime_time + sampling_time) / sampling_period))
    rec = SceneRecord(max_num_scenes, sampling_period, max_num_veh)
    features = Array{Float64}(feature_dim, feature_timesteps, max_num_veh)
    targets = Array{Float64}(target_dim, max_num_veh)
    agg_targets = Array{Float64}(target_dim, max_num_veh)

    if evaluator_type == "bootstrap"
        if prediction_model_type == "neural_network"
            prediction_model = Network(network_filepath)
        else
            throw(ArgumentError(
                "invalid prediction model type $(prediction_model_type)"))
        end
        eval = BootstrappingMonteCarloEvaluator(ext, target_ext, num_runs, prime_time,
            sampling_time, veh_idx_can_change, rec, features, targets, 
            agg_targets, prediction_model, discount = bootstrap_discount)
    else
        eval = MonteCarloEvaluator(ext, target_ext, num_runs, prime_time, sampling_time,
            veh_idx_can_change, rec, features, targets, agg_targets, 
            feature_step_size = feature_step_size)
    end
end

function build_dataset(output_filepath::String, flags, 
        ext::AbstractFeatureExtractor, target_ext::AbstractFeatureExtractor,
        weights::Union{Array{Float64},Void})
    # formulate attributes of the dataset
    feature_timesteps = flags["feature_timesteps"]
    chunk_dim = flags["chunk_dim"]
    target_dim = flags["target_dim"]
    max_num_samples = flags["num_scenarios"] * flags["max_num_vehicles"]
    feature_dim = length(ext)
    use_weights = typeof(weights) == Void ? false : true
    attrs = convert(Dict, flags)
    attrs["feature_names"] = feature_names(ext)
    attrs["target_names"] = feature_names(target_ext)
    dataset = Dataset(output_filepath, feature_dim, feature_timesteps, target_dim,
        max_num_samples, chunk_dim = chunk_dim, init_file = false, attrs = attrs,
        use_weights = use_weights)
    return dataset
end

function build_monitor(flags)
    # optionally include monitor
    if flags["monitor_scenario_record_freq"] > 0
        submonitors = Submonitor[ScenarioRecorderMonitor(
            freq = flags["monitor_scenario_record_freq"])]
        monitor = Monitor(flags["monitoring_directory"], submonitors)
    else
        monitor = nothing
    end
    return monitor
end

function build_roadway(flags)
    roadway_type = flags["roadway_type"]
    num_lanes = flags["num_lanes"]
    roadway_length = flags["roadway_length"]
    roadway_radius = flags["roadway_radius"]
    if roadway_type == "straight"
        roadway = gen_straight_roadway(num_lanes, roadway_length)
    else
        roadway = gen_stadium_roadway(num_lanes, length = roadway_length, 
            radius = roadway_radius)
    end
    return roadway
end

function build_dataset_collector(output_filepath::String, flags, 
        col_id::Int = 0)
    gen = build_generator(flags)
    ext = build_extractor(flags)
    eval = build_evaluator(flags, ext)
    dataset = build_dataset(output_filepath, flags, ext, eval.target_ext, 
        get_weights(gen))
    monitor = build_monitor(flags)
    seeds = Vector{Int}() # seeds are replaced by parallel collector
    scene = Scene(flags["max_num_vehicles"])
    models = Dict{Int, DriverModel}()
    roadway = build_roadway(flags)
    col = DatasetCollector(seeds, gen, eval, dataset, scene, models, roadway, 
        id = col_id, monitor = monitor)
    return col
end

function get_filepaths(filepath, n)
    dir = dirname(filepath)
    filename = basename(filepath)
    return [string(dir, "/proc_$(i)_$(filename)") for i in 1:n]
end

function build_parallel_dataset_collector(flags)
    num_col = flags["num_proc"]
    output_filepath = flags["output_filepath"]

    filepaths = get_filepaths(output_filepath, num_col)
    cols = [build_dataset_collector(filepaths[i], flags, i) for i in 1:num_col]
    seeds = collect(flags["initial_seed"]:(
        flags["num_scenarios"] + flags["initial_seed"] - 1))
    pcol = ParallelDatasetCollector(cols, seeds, output_filepath)
    return pcol
end
