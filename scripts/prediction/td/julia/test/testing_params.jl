

function build_testing_params()

    params = Dict()

    ## generation
    params["num_lanes"] = 1
    params["max_num_vehicles"] = 1
    params["base_bn_filepath"] = "../../data/bayesnets/base_test.jld"
    params["prop_bn_filepath"] = "../../data/bayesnets/prop_test.jld"
    params["lon_accel_std_dev"] = 1.
    params["lat_accel_std_dev"] = .0
    params["overall_response_time"] = .0
    params["lon_response_time"] = .0
    params["err_p_a_to_i"] = .0
    params["err_p_i_to_a"] = .0
    params["prime_timesteps"] = 0
    params["sim_timesteps"] = 1
    params["num_veh_per_lane"] = 1
    params["max_timesteps"] = 50
    params["hard_brake_threshold"] = 0.
    params["hard_brake_n_past_frames"] = 1
    params["ttc_threshold"] = 3.

    ### heuristic
    params["roadway_radius"] = 400.
    params["roadway_length"] = 100.
    params["min_num_veh"] = 1
    params["max_num_veh"] = 1
    params["min_base_speed"] = 30.
    params["max_base_speed"] = 30.
    params["min_vehicle_length"] = 5.
    params["max_vehicle_length"] = 5.
    params["min_vehicle_width"] = 2.5
    params["max_vehicle_width"] = 2.5
    params["min_init_dist"] = 10.
    params["heuristic_behavior_type"] = "" # correlated behavior

    ## feature extraction
    params["extract_core"] = true
    params["extract_temporal"] = true
    params["extract_well_behaved"] = true
    params["extract_neighbor"] = true
    params["extract_behavioral"] = false
    params["extract_neighbor_behavioral"] = false
    params["extract_car_lidar"] = true
    params["extract_car_lidar_range_rate"] = true
    params["extract_road_lidar"] = false

    # prediction
    params["hidden_layer_sizes"] = [256, 128, 64]
    params["value_dim"] = 5
    params["local_steps_per_update"] = 500
    params["grad_clip_norm"] = 40
    params["learning_rate"] = 1e-4
    params["dropout_keep_prob"] = 1.
    params["discount"] = .99
    params["n_global_steps"] = 100000000
    params["summary_every"] = 11
    params["target_loss_index"] = 3

    # monitoring
    params["viz_dir"] = "../../data/viz/test/"
    params["summarize_features"] = true

    params
end