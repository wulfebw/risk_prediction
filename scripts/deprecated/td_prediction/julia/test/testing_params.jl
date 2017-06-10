

function build_testing_params()

    params = Dict()

    ## generation
    params["num_lanes"] = 1
    params["max_num_vehicles"] = 100
    params["base_bn_filepath"] = "../../data/bayesnets/base_test.jld"
    params["prop_bn_filepath"] = "../../data/bayesnets/prop_test.jld"
    params["lon_accel_std_dev"] = 1.
    params["lat_accel_std_dev"] = .1
    params["overall_response_time"] = .2
    params["lon_response_time"] = .2
    params["err_p_a_to_i"] = .01
    params["err_p_i_to_a"] = .3
    params["prime_timesteps"] = 0
    params["sim_timesteps"] = 5
    params["num_veh_per_lane"] = 20
    params["max_timesteps"] = 50
    params["hard_brake_threshold"] = -3.
    params["hard_brake_n_past_frames"] = 2
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
    params["extract_core"] = True
    params["extract_temporal"] = True
    params["extract_well_behaved"] = True
    params["extract_neighbor"] = True
    params["extract_behavioral"] = False
    params["extract_neighbor_behavioral"] = False
    params["extract_car_lidar"] = True
    params["extract_car_lidar_range_rate"] = True
    params["extract_road_lidar"] = False

    # prediction
    params["hidden_layer_sizes"] = [256, 128, 64]
    params["value_dim"] = 5
    params["local_steps_per_update"] = 500
    params["grad_clip_norm"] = 40
    params["learning_rate"] = 1e-4
    params["dropout_keep_prob"] = .5
    params["discount"] = .99
    params["n_global_steps"] = 100000000
    params["summary_every"] = 11
    params["target_loss_index"] = 3

    # monitoring
    params["viz_dir"] = "../../data/viz/test/"
    params["summarize_features"] = True

    params
end