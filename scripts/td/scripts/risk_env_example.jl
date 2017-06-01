include("../julia/JuliaEnvs.jl")
using JuliaEnvs

params = Dict()
params["num_lanes"] = 1
params["max_num_vehicles"] = 1
params["base_bn_filepath"] = "../data/bayesnets/base_test.jld"
params["prop_bn_filepath"] = "../data/bayesnets/prop_test.jld"
params["lon_accel_std_dev"] = 1.
params["lat_accel_std_dev"] = .1
params["overall_response_time"] = .0
params["lon_response_time"] = .0
params["err_p_a_to_i"] = .0
params["err_p_i_to_a"] = .0
params["prime_timesteps"] = 2 # 300
params["sim_timesteps"] = 1
params["num_veh_per_lane"] = 10
params["max_timesteps"] = 1000 # 600

# heuristic
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
params["heuristic_behavior_type"] = "normal"

# evaluator
params["n_monte_carlo_runs"] = 10

# feature extraction
params["extract_core"] = true
params["extract_temporal"] = true
params["extract_well_behaved"] = true
params["extract_neighbor"] = true
params["extract_behavioral"] = true
params["extract_neighbor_behavioral"] = true
params["extract_car_lidar"] = true
params["extract_car_lidar_range_rate"] = true
params["extract_road_lidar"] = false

params["hard_brake_threshold"] = -3.090232306168
params["hard_brake_n_past_frames"] = 1
params["ttc_threshold"] = 3.

params["viz_dir"] = "../data/viz/test/"

srand(1)
env = MonteCarloRiskEnv(params)
x = reset(env)
t = 0
done = false
max_t = 10000
while !done && t < max_t
    t += 1
    println(t)
    nx, r, done, info = step(env)
    # frame = render(env)
end