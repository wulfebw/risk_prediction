include("../julia/JuliaEnvs.jl")
using JuliaEnvs

params = Dict()
params["num_lanes"] = 5
params["max_num_vehicles"] = 200
params["base_bn_filepath"] = "../data/bayesnets/base_test.jld"
params["prop_bn_filepath"] = "../data/bayesnets/prop_test.jld"
params["lon_accel_std_dev"] = 1.
params["lat_accel_std_dev"] = .1
params["overall_response_time"] = .2
params["lon_response_time"] = .2
params["err_p_a_to_i"] = .01
params["err_p_i_to_a"] = .3
params["prime_timesteps"] = 0
params["sim_timesteps"] = 5
params["num_veh_per_lane"] = 10
params["max_timesteps"] = 1000

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

params["viz_dir"] = "../data/viz/test/"

env = RiskEnv(params)
x = reset(env)
nx, r, done, info = step(env)
frame = render(env)