using AutoRisk

push!(LOAD_PATH, ".")
include("heuristic_dataset_config.jl")

function build_debug_collector(flags::Flags)
    # build evaluator
    ext = MultiFeatureExtractor(AbstractFeatureExtractor[CarLidarFeatureExtractor()])
    context = IntegratedContinuous(.1, 1)
    num_veh = 2
    prime_time = flags["prime_time"]
    sampling_time = flags["sampling_time"]
    sampling_period = .1
    veh_idx_can_change = false
    num_scenes = Int(ceil((prime_time + sampling_time) / sampling_period)) + 1
    num_features = length(ext)
    num_targets = 5
    rec = SceneRecord(num_scenes, .1, num_veh)
    features = Array{Float64}(num_features, 1, num_veh)
    targets = Array{Float64}(num_targets, num_veh)
    agg_targets = Array{Float64}(num_targets, num_veh)
    if flags["evaluator_type"] == "base"
        eval = MonteCarloEvaluator(ext, flags["num_monte_carlo_runs"], context, 
            prime_time, sampling_time, veh_idx_can_change, rec, features, 
            targets, agg_targets)
    else
        prediction_model = Network(flags["network_filepath"])
        eval = BootstrappingMonteCarloEvaluator(ext, flags["num_monte_carlo_runs"], 
            context, prime_time, sampling_time, veh_idx_can_change, rec, 
            features, targets, agg_targets, prediction_model)
    end

    # build collector
    seeds = collect(flags["initial_seed"]:(flags["initial_seed"] + flags["num_scenarios"] - 1))
    roadway_gen = StaticRoadwayGenerator(gen_straight_roadway(1))
    scene_gen = DebugSceneGenerator(
        lo_Δs = flags["debug_lo_delta_s"],
        hi_Δs = flags["debug_hi_delta_s"],
        lo_v_rear = flags["debug_lo_v_rear"],
        hi_v_rear = flags["debug_hi_v_rear"],
        lo_v_fore = flags["debug_lo_v_fore"],
        hi_v_fore = flags["debug_hi_v_fore"],
        v_eps = flags["debug_v_eps"],
        s_eps = flags["debug_s_eps"]
    )
    behavior_gen = DebugBehaviorGenerator(
        rear_lon_σ = flags["debug_rear_sigma"],
        fore_lon_σ = flags["debug_fore_sigma"]
    )
    gen = FactoredGenerator(roadway_gen, scene_gen, behavior_gen)

    # dataset
    max_num_samples = num_veh * flags["num_scenarios"]
    dataset = Dataset(flags["output_filepath"], num_features, 1, num_targets, max_num_samples, init_file = false, chunk_dim = 10)

    # monitoring
    submonitors = Submonitor[ScenarioRecorderMonitor(
        freq = flags["monitor_scenario_record_freq"])]
    monitor = Monitor(flags["monitoring_directory"], submonitors)

    scene = Scene(num_veh)
    models = Dict{Int, DriverModel}()
    roadway = rand!(roadway_gen, 1)
    col = DatasetCollector(seeds, gen, eval, dataset, scene, models, roadway, 
        monitor = monitor)
    return col
end

function set_debug_values!(flags)
    flags["extract_core"] = false
    flags["extract_temporal"] = false
    flags["extract_neighbor"] = false
    flags["extract_behavioral"] = false
    flags["extract_neighbor_behavioral"] = false
    flags["extract_road_lidar"] = false
    flags["extract_car_lidar"] = true
    flags["extract_car_lidar_range_rate"] = true
end

function analyze_risk_dataset(output_filepath)
    dataset = h5open(output_filepath)
    features = read(dataset["risk/features"])
    targets = read(dataset["risk/targets"])
    println("avg features: $(mean(features, (3)))")
    println("avg targets: $(mean(targets, (2)))")
    println("size of dataset features: $(size(features))")
    println("size of dataset targets: $(size(targets))")
end

function main()
    parse_flags!(FLAGS, ARGS)
    set_debug_values!(FLAGS)
    col = build_debug_collector(FLAGS)
    generate_dataset(col)
    analyze_risk_dataset(col.dataset.filepath)
end

@time main()